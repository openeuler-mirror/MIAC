/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/matrix.h"
#include "xla/hlo/builder/lib/pooling.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Create a diagonal / batch diagonal matrix with 'input' on the diagonal.
xla::XlaOp CreateDiagonal(xla::XlaOp input, int64_t last_dim_size,
                          const xla::DExpr& last_dim_expr,
                          absl::Span<const int64_t> other_dims,
                          absl::Span<const xla::DExpr> other_dim_exprs) {
  xla::XlaBuilder* builder = input.builder();
  // Create two matrices that have the following forms, and compare them:
  //
  // [[0, 0, 0, 0]            [[0, 1, 2, 3]
  //  [1, 1, 1, 1]             [0, 1, 2, 3]
  //  [2, 2, 2, 2]             [0, 1, 2, 3]
  //  [3, 3, 3, 3]]            [0, 1, 2, 3]]
  //
  // This produces a predicate matrix of the right size, with "true" on the
  // diagonal.
  xla::XlaOp iota = xla::Iota(
      builder,
      xla::ShapeUtil::MakeShape(xla::S32, std::vector<int64_t>{last_dim_size},
                                std::vector<xla::DExpr>{last_dim_expr}),
      /*iota_dimension=*/0);
  xla::XlaOp iota_broadcast = xla::Broadcast(
      iota, {last_dim_size}, {last_dim_expr, last_dim_expr});
  xla::XlaOp mask = xla::Eq(iota_broadcast, iota, {0});

  // If this is a batched diagonal, broadcast the mask across the other
  // dimensions.
  if (!other_dims.empty()) {
    std::vector<xla::DExpr> mask_exprs(other_dim_exprs.begin(),
                                       other_dim_exprs.end());
    mask_exprs.push_back(last_dim_expr);
    mask_exprs.push_back(last_dim_expr);
    mask = xla::Broadcast(mask, other_dims, mask_exprs);
  }

  // Broadcast the input, and then use the mask computed above to select the
  // diagonal:
  // e.g, in 2D:
  //         [[t, f, f]    [[1, 1, 1]    [[0, 0, 0]      [[1, 0, 0]
  // select(  [f, t, f]  ,  [4, 4, 4]  ,  [0, 0, 0]  ) =  [0, 4, 0]
  //          [f, f, t]]    [9, 9, 9]]    [0, 0, 0]]      [0, 0, 9]]
  //
  std::vector<int64_t> out_dim_sizes(other_dims.begin(), other_dims.end());
  out_dim_sizes.push_back(last_dim_size);
  out_dim_sizes.push_back(last_dim_size);
  std::vector<xla::DExpr> out_dim_exprs(other_dim_exprs.begin(),
                                        other_dim_exprs.end());
  out_dim_exprs.push_back(last_dim_expr);
  out_dim_exprs.push_back(last_dim_expr);

  // Broadcast into the second to last dimension.
  std::vector<int64_t> broadcast_dimensions(other_dims.size() + 1);
  absl::c_iota(broadcast_dimensions, 0);
  ++broadcast_dimensions.back();
  xla::XlaOp input_broadcast = xla::BroadcastInDim(
      input, out_dim_sizes, broadcast_dimensions, out_dim_exprs);
  return xla::Select(mask, input_broadcast, xla::ZerosLike(input_broadcast));
}

class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("Diag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp input = ctx->Input(0);

    // Picture:
    // tf.diag([1, 2, 3, 4]) ==> [[1, 0, 0, 0]
    //                            [0, 2, 0, 0]
    //                            [0, 0, 3, 0]
    //                            [0, 0, 0, 4]]

    // Flattens the input to 1D.
    xla::DExpr flattened_expr = xla::DExpr::Const(1);
    std::vector<xla::DExpr> input_exprs = input_shape.get_filled_expressions();
    for (const xla::DExpr& expr : input_exprs) {
      flattened_expr = (flattened_expr * expr).simplify();
    }
    int64_t size = input_shape.num_elements();
    input = xla::Reshape(input, {size}, {flattened_expr});

    // Create an R2 with the R1 diagonal.
    xla::XlaOp diag =
        CreateDiagonal(input, size, flattened_expr, /*other_dims=*/{},
                       /*other_dim_exprs=*/{});

    // Reshapes to the final shape.
    std::vector<int64_t> new_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), new_dims.begin());
    std::copy(dims.begin(), dims.end(), new_dims.begin() + dims.size());
    std::vector<xla::DExpr> new_exprs(input_exprs.begin(), input_exprs.end());
    new_exprs.insert(new_exprs.end(), input_exprs.begin(), input_exprs.end());
    diag = xla::Reshape(diag, new_dims, new_exprs);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("Diag"), DiagOp);

REGISTER_XLA_OP(Name("DiagPart"), MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow

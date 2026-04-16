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

// XLA Pack operator.

#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/symbolic_content_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace {

bool TryBuildPackedContents(XlaOpKernelContext* ctx, int num, int axis,
                            std::vector<xla::DExpr>* contents) {
  contents->clear();
  if (axis != 0) {
    return false;
  }
  for (int i = 0; i < num; ++i) {
    if (ctx->InputShape(i).dims() != 0) {
      contents->clear();
      return false;
    }
    const auto& input_contents = ctx->InputExpression(i).contents();
    if (!input_contents.empty()) {
      if (input_contents.size() != 1) {
        contents->clear();
        return false;
      }
      contents->push_back(input_contents[0]);
      continue;
    }
    contents->push_back(xla::DExpr::Unknown(xla::kUnknownContentSentinel));
  }
  return absl::c_any_of(*contents, [](const xla::DExpr& expr) {
    return expr && expr->is_dynamic();
  });
}

class PackOp : public XlaOpKernel {
 public:
  explicit PackOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("axis", &axis_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    std::vector<xla::XlaOp> values;
    std::vector<TensorShape> shapes;
    OP_REQUIRES_OK(ctx, ctx->InputList("values", &values, &shapes));
    const int num = values.size();

    OP_REQUIRES(ctx, num >= 0,
                errors::InvalidArgument("Pack requires >= 1 arguments"));

    // Verify that all input shapes match
    for (int i = 1; i < num; i++) {
      OP_REQUIRES(ctx, shapes[0].IsSameSize(shapes[i]),
                  errors::InvalidArgument(
                      "Shapes of all inputs must match: values[0].shape = ",
                      shapes[0].DebugString(), " != values[", i, "].shape = ",
                      shapes[i].DebugString()));
    }

    int expanded_num_dims = shapes[0].dims() + 1;
    int axis = axis_;
    if (axis < 0) axis += expanded_num_dims;

    OP_REQUIRES(ctx, 0 <= axis && axis < expanded_num_dims,
                errors::InvalidArgument("axis = ", axis_, " not in [",
                                        -expanded_num_dims, ", ",
                                        expanded_num_dims, ")"));

    std::vector<xla::XlaOp> reshaped_inputs(num);

    TensorShape child_shape(shapes[0]);
    std::vector<xla::DExpr> exprs = child_shape.get_filled_expressions();
    child_shape.InsertDim(axis, 1);
    exprs.insert(exprs.begin() + axis, xla::DExpr::Const(1));
    std::vector<xla::DExpr> output_contents;
    const bool has_output_contents =
        SymbolicContentEnabled() &&
        TryBuildPackedContents(ctx, num, axis, &output_contents);
    for (int i = 0; i < num; ++i) {
      // Reshape the inputs to have an extra dimension of size 1.
      reshaped_inputs[i] = xla::Reshape(values[i], child_shape.dim_sizes(),
                                        exprs);
      if (has_output_contents) {
        OP_REQUIRES_OK(
            ctx, ctx->builder()->SetInstructionContents(
                     reshaped_inputs[i],
                     {output_contents[static_cast<size_t>(i)]}));
      }
    }

    auto output = xla::ConcatInDim(ctx->builder(), reshaped_inputs, axis);
    if (has_output_contents) {
      OP_REQUIRES_OK(
          ctx, ctx->builder()->SetInstructionContents(output, output_contents));
      auto output_expr =
          XlaExpression::XlaOp(output, ctx->expected_output_dtype(0));
      output_expr.set_contents(std::move(output_contents));
      ctx->SetOutputExpression(0, output_expr);
      return;
    }
    ctx->SetOutput(0, output);
  }

 private:
  int axis_;
};

REGISTER_XLA_OP(Name("Pack"), PackOp);

}  // namespace
}  // namespace tensorflow

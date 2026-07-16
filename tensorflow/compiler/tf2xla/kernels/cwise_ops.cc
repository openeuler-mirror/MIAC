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

// XLA-specific base classes for Unary and Binary Ops.

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

namespace {

bool IsSymbolicContentType(DataType type) {
  type = BaseType(type);
  return type == DT_INT32 || type == DT_INT64;
}

bool TryGetIntContentsFromConstant(const Tensor& tensor,
                                   std::vector<xla::DExpr>* contents) {
  contents->clear();
  if (tensor.dims() > 1) {
    return false;
  }
  if (tensor.dtype() == DT_INT32) {
    if (tensor.dims() == 0) {
      contents->push_back(xla::DExpr::Const(tensor.scalar<int32>()()));
      return true;
    }
    auto flat = tensor.flat<int32>();
    contents->reserve(flat.size());
    for (int i = 0; i < flat.size(); ++i) {
      contents->push_back(xla::DExpr::Const(flat(i)));
    }
    return true;
  }
  if (tensor.dtype() == DT_INT64) {
    if (tensor.dims() == 0) {
      contents->push_back(xla::DExpr::Const(tensor.scalar<int64_t>()()));
      return true;
    }
    auto flat = tensor.flat<int64_t>();
    contents->reserve(flat.size());
    for (int i = 0; i < flat.size(); ++i) {
      contents->push_back(xla::DExpr::Const(flat(i)));
    }
    return true;
  }
  return false;
}

bool TryGetIntContentsFromLiteral(const xla::LiteralSlice& literal,
                                  std::vector<xla::DExpr>* contents) {
  contents->clear();
  if (literal.shape().dimensions_size() > 1) {
    return false;
  }
  if (literal.shape().element_type() == xla::S32) {
    if (literal.shape().dimensions_size() == 0) {
      contents->push_back(xla::DExpr::Const(literal.Get<int32>({})));
      return true;
    }
    const int64_t size = literal.shape().dimensions(0);
    contents->reserve(size);
    for (int64_t i = 0; i < size; ++i) {
      contents->push_back(xla::DExpr::Const(literal.Get<int32>({i})));
    }
    return true;
  }
  if (literal.shape().element_type() == xla::S64) {
    if (literal.shape().dimensions_size() == 0) {
      contents->push_back(xla::DExpr::Const(literal.Get<int64_t>({})));
      return true;
    }
    const int64_t size = literal.shape().dimensions(0);
    contents->reserve(size);
    for (int64_t i = 0; i < size; ++i) {
      contents->push_back(xla::DExpr::Const(literal.Get<int64_t>({i})));
    }
    return true;
  }
  return false;
}

bool TryGetInputContents(XlaOpKernelContext* ctx, const XlaExpression& expr,
                         const TensorShape& shape,
                         std::vector<xla::DExpr>* contents) {
  if (shape.dims() > 1) {
    return false;
  }
  if (!expr.contents().empty()) {
    if (absl::c_any_of(expr.contents(),
                       [](const xla::DExpr& e) { return !e; })) {
      return false;
    }
    contents->assign(expr.contents().begin(), expr.contents().end());
    return true;
  }
  auto constant = expr.constant_value();
  if (!constant.has_value()) {
    if (!expr.handle().valid() || expr.handle().IsUninitialized()) {
      return false;
    }
    auto literal_or =
        ctx->value_inference().AnalyzeConstant(expr.handle(),
                                               xla::ValueInferenceMode::kValue);
    if (!literal_or.ok() || !literal_or->AllValid()) {
      return false;
    }
    auto literal = literal_or->GetValue();
    if (!literal.has_value()) {
      return false;
    }
    return TryGetIntContentsFromLiteral(*literal, contents);
  }
  return TryGetIntContentsFromConstant(*constant, contents);
}

xla::DExpr BroadcastedContentAt(absl::Span<const xla::DExpr> contents,
                                const TensorShape& shape,
                                int64_t output_index) {
  if (shape.dims() == 0) {
    return contents.empty() ? xla::DExpr() : contents[0];
  }
  if (contents.empty()) {
    return xla::DExpr();
  }
  if (shape.dim_size(0) == 1) {
    return contents[0];
  }
  if (output_index >= contents.size()) {
    return xla::DExpr();
  }
  return contents[output_index];
}

bool TryBuildSymbolicBinaryContents(XlaOpKernelContext* ctx,
                                    XlaBinaryOp* op,
                                    const TensorShape& lhs_shape,
                                    const TensorShape& rhs_shape,
                                    const BCast& bcast,
                                    std::vector<xla::DExpr>* contents) {
  contents->clear();
  if (!IsSymbolicContentType(ctx->input_type(0)) ||
      !IsSymbolicContentType(ctx->expected_output_dtype(0))) {
    return false;
  }
  const auto& output_shape = bcast.output_shape();
  if (output_shape.size() > 1) {
    return false;
  }

  std::vector<xla::DExpr> lhs_contents;
  std::vector<xla::DExpr> rhs_contents;
  if (!TryGetInputContents(ctx, ctx->InputExpression(0), lhs_shape,
                           &lhs_contents) ||
      !TryGetInputContents(ctx, ctx->InputExpression(1), rhs_shape,
                           &rhs_contents)) {
    return false;
  }

  int64_t output_elements = output_shape.empty() ? 1 : output_shape[0];
  contents->reserve(output_elements);
  for (int64_t i = 0; i < output_elements; ++i) {
    xla::DExpr lhs_expr = BroadcastedContentAt(lhs_contents, lhs_shape, i);
    xla::DExpr rhs_expr = BroadcastedContentAt(rhs_contents, rhs_shape, i);
    if (!lhs_expr || !rhs_expr) {
      contents->clear();
      return false;
    }
    xla::DExpr out_expr = op->SymbolicComputation(lhs_expr, rhs_expr);
    if (!out_expr) {
      contents->clear();
      return false;
    }
    contents->push_back(out_expr.simplify());
  }
  return true;
}

}  // namespace

void XlaBinaryOp::Compile(XlaOpKernelContext* ctx) {
  TensorShape lhs_shape = ctx->InputShape(0);
  TensorShape rhs_shape = ctx->InputShape(1);
  xla::Shape lhs_xla_shape = ctx->InputXlaShape(0).value();
  xla::Shape rhs_xla_shape = ctx->InputXlaShape(1).value();
  // Fetch the expressions containing the input tensors.
  auto lhs_handle = ctx->Input(0);
  auto rhs_handle = ctx->Input(1);
  if (lhs_shape.dims() == rhs_shape.dims()) {
    auto reconcile_tensor_mismatched_dims = [ctx](
                                                xla::XlaOp lhs, xla::XlaOp rhs,
                                                const xla::Shape& lhs_xla_shape,
                                                const xla::Shape& rhs_xla_shape,
                                                TensorShape* lhs_tensor_shape) {
      // Find out mismatched dimensions that are non-broadcastable.
      // Reconcile the
      // difference by slicing the bigger dimension.
      for (int64_t i = 0; i < lhs_xla_shape.dimensions().size(); ++i) {
        if (lhs_xla_shape.is_dynamic_dimension(i)) {
          if (!rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) > rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1) {
            // e.g., :
            // lhs = [..., <=N, ...]
            // rhs = [..., 2  , ...]
            // Slice N into 2.
            // Size 1 dim doesn't need slice as the other side is
            // broadcastable.
            auto size = xla::GetDimensionSize(lhs, i);
            lhs = xla::SliceInDim(lhs, 0, rhs_xla_shape.dimensions(i), 1,
                                  /*dimno=*/i);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            lhs_tensor_shape->set_expression(i, rhs_xla_shape.expressions(i));
            // Propagate dynamic dimension.
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (rhs_xla_shape.is_dynamic_dimension(i) &&
              lhs_xla_shape.dimensions(i) < rhs_xla_shape.dimensions(i) &&
              rhs_xla_shape.dimensions(i) != 1 &&
              lhs_xla_shape.dimensions(i) != 1) {
            // e.g., :
            // lhs = [..., <=M, ...]
            // rhs = [..., <=N  , ...]
            // where M < N
            //
            // In this case we pad M into N to make the bounds the same.
            // Note that we can't slice N into M because M could be a
            // dynamic size 1 dim that's meant to be broadcasted to N.
            auto size = xla::GetDimensionSize(lhs, i);
            int64_t diff =
                rhs_xla_shape.dimensions(i) - lhs_xla_shape.dimensions(i);
            lhs = xla::PadInDim(
                lhs, xla::Zero(ctx->builder(), lhs_xla_shape.element_type()), i,
                0, diff);
            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            lhs_tensor_shape->set_expression(i, rhs_xla_shape.expressions(i));
            // Propagate dynamic dimension.
            lhs = xla::SetDimensionSize(lhs, size, i);
          }
          if (lhs_xla_shape.dimensions(i) == 1 &&
              rhs_xla_shape.dimensions(i) != 1) {
            // lhs = [..., <=1, ...]
            // rhs = [...,   N, ...] or [..., <=N, ...]
            // where N != 1.
            //
            // In this case we will need to broadcast this dimension to N.
            // If the dynamic size is 0, the result size is zero.
            // If the dynamic size is 1, the result size is N.
            //
            // However, XLA only does degenerate broadcasts for non-dynamic
            // dimensions of size 1.

            // Get the original size.
            auto size = xla::GetDimensionSize(lhs, i);

            // Remove the dynamic dimension.
            lhs = xla::RemoveDynamicDimension(lhs, i);

            // Broadcast the dimension to N.
            std::vector<int64_t> dimensions(lhs_xla_shape.dimensions().begin(),
                                            lhs_xla_shape.dimensions().end());
            dimensions[i] = rhs_xla_shape.dimensions(i);
            std::vector<int64_t> broadcast_dimensions(
                lhs_xla_shape.dimensions().size());
            absl::c_iota(broadcast_dimensions, 0);
            lhs = xla::BroadcastInDim(lhs, dimensions, broadcast_dimensions);

            xla::XlaOp rhs_size;
            if (rhs_xla_shape.is_dynamic_dimension(i)) {
              rhs_size = xla::GetDimensionSize(rhs, i);
            } else {
              rhs_size = xla::ConstantR0<int32_t>(lhs.builder(),
                                                  rhs_xla_shape.dimensions(i));
            }

            // The original size is 0 or 1, so we can multiply it by the RHS
            // size to get the size of the resulting broadcast.
            size = xla::Mul(size, rhs_size);

            // Set the resulting dimension size.
            lhs = xla::SetDimensionSize(lhs, size, i);

            lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
            lhs_tensor_shape->set_expression(
                i, (lhs_tensor_shape->get_filled_expression(i) *
                    rhs_xla_shape.expressions(i))
                       .simplify());
          }
        }
      }
      return lhs;
    };
    lhs_handle = reconcile_tensor_mismatched_dims(
        lhs_handle, rhs_handle, lhs_xla_shape, rhs_xla_shape, &lhs_shape);
    rhs_handle = reconcile_tensor_mismatched_dims(
        rhs_handle, lhs_handle, rhs_xla_shape, lhs_xla_shape, &rhs_shape);
  }
  // By TensorFlow conventions the inputs may not have the same
  // shapes, in which case they will be automatically broadcast if
  // possible before mapping. Use the standard TensorFlow helper to
  // compute valid broadcast shapes, but rely below on XLA to
  // automatically perform the broadcast assuming its valid shapes are
  // a superset of TensorFlow's valid shapes.
  BCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape),
              /*fewer_dims_optimization=*/false);
  if (!bcast.IsValid()) {
    ctx->SetStatus(absl::InvalidArgumentError(
        absl::StrCat("Incompatible shapes: ", lhs_shape.DebugString(), " vs. ",
                     rhs_shape.DebugString())));
    return;
  }

  auto build_broadcast_output_expressions =
      [&lhs_shape, &rhs_shape, &bcast]() -> std::vector<xla::DExpr> {
    auto merge_broadcast_dim = [](bool has_lhs, int64_t lhs_dim,
                                  const xla::DExpr& lhs_expr, bool has_rhs,
                                  int64_t rhs_dim, const xla::DExpr& rhs_expr,
                                  int64_t output_dim) {
      if (!has_lhs) {
        return rhs_expr;
      }
      if (!has_rhs) {
        return lhs_expr;
      }
      if (lhs_dim == 1 && rhs_dim != 1) {
        // A broadcasted singleton usually inherits the other side's
        // expression, but keep a dynamic singleton visible by folding it into
        // the result.
        return lhs_expr && lhs_expr->is_dynamic()
                   ? (lhs_expr * rhs_expr).simplify()
                   : rhs_expr;
      }
      if (rhs_dim == 1 && lhs_dim != 1) {
        // Symmetric case for a singleton rhs broadcast.
        return rhs_expr && rhs_expr->is_dynamic()
                   ? (rhs_expr * lhs_expr).simplify()
                   : lhs_expr;
      }
      // When both sides describe the same logical dimension, prefer whichever
      // side still carries a dynamic symbolic expression.
      if (lhs_expr && lhs_expr->is_dynamic()) {
        return lhs_expr;
      }
      if (rhs_expr && rhs_expr->is_dynamic()) {
        return rhs_expr;
      }
      return xla::DExpr::Const(output_dim);
    };

    const auto& output_shape = bcast.output_shape();
    std::vector<xla::DExpr> output_exprs(output_shape.size());

    for (int out_i = output_shape.size() - 1, lhs_i = lhs_shape.dims() - 1,
             rhs_i = rhs_shape.dims() - 1;
         out_i >= 0; --out_i, --lhs_i, --rhs_i) {
      const bool has_lhs = lhs_i >= 0;
      const bool has_rhs = rhs_i >= 0;
      xla::DExpr lhs_expr = has_lhs ? lhs_shape.get_filled_expression(lhs_i)
                                    : xla::DExpr::Const(1);
      xla::DExpr rhs_expr = has_rhs ? rhs_shape.get_filled_expression(rhs_i)
                                    : xla::DExpr::Const(1);
      const int64_t lhs_dim = has_lhs ? lhs_shape.dim_size(lhs_i) : 1;
      const int64_t rhs_dim = has_rhs ? rhs_shape.dim_size(rhs_i) : 1;
      output_exprs[out_i] = merge_broadcast_dim(
          has_lhs, lhs_dim, lhs_expr, has_rhs, rhs_dim, rhs_expr,
          output_shape[out_i]);
    }

    return output_exprs;
  };
  std::vector<xla::DExpr> broadcast_output_exprs =
      build_broadcast_output_expressions();

  // If the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a
  // prefix. In other words to pad a 5-vector to a 3-dimensional
  // tensor it is reshaped to have shape [1,1,5]. XLA's automatic
  // broadcast code is able to broadcast from lower to higher rank,
  // but doesn't assume you want to pad as a prefix of the dimensions,
  // and instead needs to be told which dimensions of the higher rank
  // tensor to match to the lower rank tensor. In this example it
  // would be dimensions [2]. If we were matching a matrix against a
  // 4-D tensor the dimensions to match would be [2,3],
  // etc. extend_dimension encodes the general case.
  std::vector<int64_t> extend_dimension;
  int max_rank = std::max(lhs_shape.dims(), rhs_shape.dims());
  int min_rank = std::min(lhs_shape.dims(), rhs_shape.dims());
  if (min_rank != max_rank) {
    for (int i = 0; i < min_rank; ++i) {
      // Match the lower rank tensor along the larger-numbered
      // dimensions of the higher rank tensor.
      extend_dimension.push_back(max_rank - min_rank + i);
    }
  }

  // Call virtual method to emit the computation.
  xla::XlaOp output =
      Computation(ctx, lhs_handle, lhs_shape.dim_sizes(), rhs_handle,
                  rhs_shape.dim_sizes(), bcast,
                  broadcast_output_exprs, extend_dimension);

  // The TensorFlow helper computed the post-broadcast shape in
  // output_shape: we rely on subclassed Computations to implement the
  // same broadcast semantics.
  std::vector<xla::DExpr> output_contents;
  if (TryBuildSymbolicBinaryContents(ctx, this, lhs_shape, rhs_shape, bcast,
                                     &output_contents)) {
    auto output_expr =
        XlaExpression::XlaOp(output, ctx->expected_output_dtype(0));
    output_expr.set_contents(std::move(output_contents));
    ctx->SetOutputExpression(0, output_expr);
    return;
  }
  ctx->SetOutput(0, output);
}

/* static */ std::pair<xla::XlaOp, xla::XlaOp> XlaBinaryOp::Broadcast(
    xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper,
    absl::Span<const xla::DExpr> output_exprs) {
  CHECK_EQ(output_exprs.size(), broadcast_helper.output_shape().size());
  for (const xla::DExpr& expr : output_exprs) {
    CHECK(expr);
  }
  auto lhs_output =
      BroadcastTo(lhs, broadcast_helper.output_shape(), output_exprs);
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }
  auto rhs_output =
      BroadcastTo(rhs, broadcast_helper.output_shape(), output_exprs);
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.value(), rhs_output.value()};
}

}  // namespace tensorflow

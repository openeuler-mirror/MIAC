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

// XLA-specific sequence and range Ops.

#include <cstdint>
#include <type_traits>
#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/symbolic_content_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

template <typename T>
xla::DExpr GetScalarExpr(const XlaExpression& expression,
                         const xla::LiteralSlice& literal) {
  const auto& contents = expression.contents();
  if (!contents.empty() && contents[0]) {
    return contents[0];
  }
  return xla::DExpr::Const(literal.Get<T>({}));
}

bool HasStaticScalarContent(const XlaExpression& expression) {
  const auto& contents = expression.contents();
  return contents.empty() ||
         (contents[0] && contents[0]->is_constant());
}

bool HasDynamicContent(const XlaExpression& expression) {
  return absl::c_any_of(expression.contents(), [](const xla::DExpr& expr) {
    return expr && expr->is_dynamic();
  });
}

template <typename T>
std::vector<xla::DExpr> BuildRangeContents(const XlaExpression& start_expr,
                                           const XlaExpression& delta_expr,
                                           const xla::LiteralSlice& start,
                                           const xla::LiteralSlice& delta,
                                           int64_t size) {
  std::vector<xla::DExpr> contents;
  contents.reserve(size);
  xla::DExpr start_symbol = GetScalarExpr<T>(start_expr, start);
  xla::DExpr delta_symbol = GetScalarExpr<T>(delta_expr, delta);
  for (int64_t i = 0; i < size; ++i) {
    xla::DExpr offset = xla::DExpr::Const(static_cast<T>(i));
    contents.push_back((start_symbol + (delta_symbol * offset).simplify()).simplify());
  }
  return contents;
}

template <typename T>
xla::DExpr BuildRangeSizeExpr(const XlaExpression& start_expr,
                              const XlaExpression& limit_expr,
                              const XlaExpression& delta_expr,
                              const xla::LiteralSlice& start,
                              const xla::LiteralSlice& limit,
                              const xla::LiteralSlice& delta,
                              int64_t fallback_size) {
  xla::DExpr start_symbol = GetScalarExpr<T>(start_expr, start);
  xla::DExpr limit_symbol = GetScalarExpr<T>(limit_expr, limit);
  xla::DExpr delta_symbol = GetScalarExpr<T>(delta_expr, delta);

  if (delta.Get<T>({}) > 0) {
    xla::DExpr diff = (limit_symbol - start_symbol).simplify();
    xla::DExpr adjusted = (diff - 1).simplify();
    xla::DExpr quotient = (adjusted / delta_symbol).simplify();
    return (quotient + 1).simplify();
  }
  xla::DExpr step_symbol = (xla::DExpr::Const(0) - delta_symbol).simplify();
  xla::DExpr diff = (start_symbol - limit_symbol).simplify();
  xla::DExpr adjusted = (diff - 1).simplify();
  xla::DExpr quotient = (adjusted / step_symbol).simplify();
  return (quotient + 1).simplify();
}

// The type-specific part of the implementation of Range.
template <typename T>
absl::StatusOr<xla::XlaOp> CreateRangeTensor(
    const xla::LiteralSlice& start_literal,
    const xla::LiteralSlice& limit_literal,
    const xla::LiteralSlice& delta_literal, xla::XlaBuilder* builder,
    xla::DExpr size_expr = xla::DExpr()) {
  T start = start_literal.Get<T>({});
  T limit = limit_literal.Get<T>({});
  T delta = delta_literal.Get<T>({});

  if (delta == 0) {
    return errors::InvalidArgument("Requires delta != 0: ", delta);
  }
  if (delta > 0) {
    if (start > limit) {
      return errors::InvalidArgument(
          "Requires start <= limit when delta > 0: ", start, "/", limit);
    }
  } else {
    if (start < limit) {
      return errors::InvalidArgument(
          "Requires start >= limit when delta < 0: ", start, "/", limit);
    }
  }
  int64_t size =
      (std::is_integral<T>::value
           ? static_cast<T>(
                 limit == start
                     ? 0
                     : (std::abs(limit - start) - 1) / std::abs(delta) + 1)
           : std::ceil(std::abs((limit - start) / delta)));

  xla::XlaOp iota =
      (std::is_integral<T>::value && size_expr)
          ? xla::Iota(builder,
                      xla::ShapeUtil::MakeShape(
                          xla::primitive_util::NativeToPrimitiveType<T>(),
                          {size}, {size_expr}),
                      /*iota_dimension=*/0)
          : xla::Iota(builder, xla::primitive_util::NativeToPrimitiveType<T>(),
                      size);

  return xla::ConstantR0(builder, start) + xla::ConstantR0(builder, delta) * iota;
}

class RangeOp : public XlaOpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape(0);
    const TensorShape limit_in_shape = ctx->InputShape(1);
    const TensorShape delta_in_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(limit_in_shape),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(delta_in_shape),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in_shape.DebugString()));
    xla::Literal start, limit, delta;
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            0, &start, xla::ValueInferenceMode::kLowerBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(
                            1, &limit, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES_OK(ctx, ctx->ConstantInput(2, &delta));

    DataType type = input_type(0);
    absl::StatusOr<xla::XlaOp> output;
    switch (type) {
      case DT_INT32: {
        int32 start_value = start.Get<int32>({});
        int32 limit_value = limit.Get<int32>({});
        int32 delta_value = delta.Get<int32>({});
        int64_t size = static_cast<int32>(
            limit_value == start_value
                ? 0
                : (std::abs(limit_value - start_value) - 1) /
                          std::abs(delta_value) +
                      1);
        xla::DExpr size_expr =
            HasStaticScalarContent(ctx->InputExpression(2))
                ? BuildRangeSizeExpr<int32>(ctx->InputExpression(0),
                                            ctx->InputExpression(1),
                                            ctx->InputExpression(2), start,
                                            limit, delta, size)
                : xla::DExpr::Const(size);
        output = CreateRangeTensor<int32>(start, limit, delta, ctx->builder(),
                                          size_expr);
        break;
      }
      case DT_INT64: {
        int64_t start_value = start.Get<int64_t>({});
        int64_t limit_value = limit.Get<int64_t>({});
        int64_t delta_value = delta.Get<int64_t>({});
        int64_t size =
            limit_value == start_value
                ? 0
                : (std::abs(limit_value - start_value) - 1) /
                          std::abs(delta_value) +
                      1;
        xla::DExpr size_expr =
            HasStaticScalarContent(ctx->InputExpression(2))
                ? BuildRangeSizeExpr<int64_t>(ctx->InputExpression(0),
                                              ctx->InputExpression(1),
                                              ctx->InputExpression(2), start,
                                              limit, delta, size)
                : xla::DExpr::Const(size);
        output = CreateRangeTensor<int64_t>(start, limit, delta, ctx->builder(),
                                            size_expr);
        break;
      }
      case DT_FLOAT:
        output = CreateRangeTensor<float>(start, limit, delta, ctx->builder());
        break;
      case DT_DOUBLE:
        output = CreateRangeTensor<double>(start, limit, delta, ctx->builder());
        break;
      default:
        output = errors::InvalidArgument("Invalid type for Range ",
                                         DataTypeString(type));
    }
    OP_REQUIRES_OK(ctx, output.status());
    bool start_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(0, &start_is_dynamic));
    bool limit_is_dynamic = false;
    OP_REQUIRES_OK(ctx,
                   ctx->ResolveInputDynamismIntoPred(1, &limit_is_dynamic));

    if (start_is_dynamic || limit_is_dynamic) {
      xla::XlaOp delta = ctx->Input(2);
      xla::XlaOp limit = ctx->Input(1);
      xla::XlaOp start = ctx->Input(0);
      if (type == DT_INT32 || type == DT_INT64) {
        auto dynamic_size = (xla::Abs(limit - start) + xla::Abs(delta) -
                             xla::One(ctx->builder(), ctx->input_xla_type(0))) /
                            xla::Abs(delta);
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.value(), dynamic_size, 0);
      } else {
        auto dynamic_size = (xla::Ceil(xla::Abs((limit - start) / delta)));
        dynamic_size = xla::ConvertElementType(dynamic_size, xla::S32);
        output = xla::SetDimensionSize(output.value(), dynamic_size, 0);
      }
    }

    const XlaExpression& start_expr = ctx->InputExpression(0);
    const XlaExpression& delta_expr = ctx->InputExpression(2);
    const bool symbolic_enabled = SymbolicContentEnabled();
    const bool has_dynamic_content =
        HasDynamicContent(start_expr) || HasDynamicContent(delta_expr);

    if (type == DT_INT32) {
      int32 start_value = start.Get<int32>({});
      int32 limit_value = limit.Get<int32>({});
      int32 delta_value = delta.Get<int32>({});
      if (symbolic_enabled && has_dynamic_content) {
        int64_t size = static_cast<int32>(
            limit_value == start_value
                ? 0
                : (std::abs(limit_value - start_value) - 1) /
                          std::abs(delta_value) +
                      1);
        auto output_expr =
            XlaExpression::XlaOp(output.value(), ctx->expected_output_dtype(0));
        output_expr.set_contents(BuildRangeContents<int32>(
            start_expr, delta_expr, start, delta, size));
        ctx->SetOutputExpression(0, output_expr);
      } else {
        ctx->SetOutput(0, output.value());
      }
    } else if (type == DT_INT64) {
      int64_t start_value = start.Get<int64_t>({});
      int64_t limit_value = limit.Get<int64_t>({});
      int64_t delta_value = delta.Get<int64_t>({});
      if (symbolic_enabled && has_dynamic_content) {
        int64_t size = static_cast<int64_t>(
            limit_value == start_value
                ? 0
                : (std::abs(limit_value - start_value) - 1) /
                          std::abs(delta_value) +
                      1);
        auto output_expr =
            XlaExpression::XlaOp(output.value(), ctx->expected_output_dtype(0));
        output_expr.set_contents(BuildRangeContents<int64_t>(
            start_expr, delta_expr, start, delta, size));
        ctx->SetOutputExpression(0, output_expr);
      } else {
        ctx->SetOutput(0, output.value());
      }
    } else {
      ctx->SetOutput(0, output.value());
    }
  }
};

REGISTER_XLA_OP(Name("Range")
                    .CompileTimeConstantInput("start")
                    .CompileTimeConstantInput("limit")
                    .CompileTimeConstantInput("delta"),
                RangeOp);

class LinSpaceOp : public XlaOpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}

  void Compile(XlaOpKernelContext* ctx) override {
    const TensorShape start_in_shape = ctx->InputShape("start");
    const TensorShape stop_in_shape = ctx->InputShape("stop");
    const TensorShape num_in_shape = ctx->InputShape("num");
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(start_in_shape),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(stop_in_shape),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in_shape.DebugString()));
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(num_in_shape),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in_shape.DebugString()));

    int64_t num;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntScalar("num", &num));
    OP_REQUIRES(ctx, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    xla::XlaOp start = ctx->Input("start");
    xla::XlaOp stop = ctx->Input("stop");
    xla::XlaOp iota = xla::Iota(ctx->builder(), ctx->output_xla_type(0), num);
    xla::XlaOp step =
        (stop - start) / xla::ScalarLike(start, (num > 1 ? num - 1 : num));
    xla::XlaOp result = iota * step + start;
    if (num > 1) {
      // According to linspace spec, start has to be the first element and end
      // has to be last element.
      xla::XlaOp mask = xla::Iota(ctx->builder(), xla::S64, num);
      xla::XlaOp eq = xla::Eq(mask, xla::ScalarLike(mask, num - 1));
      result = xla::Select(eq, stop, result);
    }
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("LinSpace").CompileTimeConstantInput("num"), LinSpaceOp);

}  // namespace
}  // namespace tensorflow

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

#include <cstdint>
#include <type_traits>
#include <vector>

#include "xla/printer.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape_expr.h"
#include "tsl/platform/protobuf.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape_expr.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

constexpr char kUserInferredValueContentsAttrName[] =
    "_user_inferred_value_contents";

template <typename DstT, typename SrcT>
DstT CastTo(SrcT src) {
  return static_cast<DstT>(src);
}

template <typename DstT,
          typename std::enable_if<std::is_same<DstT, Eigen::half>::value ||
                                  std::is_same<DstT, bfloat16>::value>::type* =
              nullptr>
DstT CastTo(int32_t src) {
  return absl::bit_cast<DstT>(static_cast<uint16>(src));
}

// Returns scalar constant with the value in the tensor, if the given proto has
// exactly one value but more than one elements. This encoding is used to
// efficiently serialize tensors that have one value repeated for all the
// indices.
xla::XlaOp GetScalarConst(const TensorProto& proto, xla::XlaBuilder* b) {
  if (!proto.tensor_content().empty()) return xla::XlaOp();
  TensorShape shape(proto.tensor_shape());
  if (shape.num_elements() > 1) {
    switch (proto.dtype()) {
#define HANDLE_SPLAT(DTYPE, field_name, xla_type)                             \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 0) {                                 \
      return xla::ConstantR0(b, CastTo<xla_type>(0));                         \
    } else if (proto.field_name##_val_size() == 1) {                          \
      return xla::ConstantR0(b, CastTo<xla_type>(proto.field_name##_val(0))); \
    }                                                                         \
    break;

      HANDLE_SPLAT(DT_BOOL, bool, bool);

      HANDLE_SPLAT(DT_INT8, int, int8_t);
      HANDLE_SPLAT(DT_INT16, int, int16_t);
      HANDLE_SPLAT(DT_INT32, int, int32_t);
      HANDLE_SPLAT(DT_INT64, int64, int64_t);

      HANDLE_SPLAT(DT_UINT8, int, uint8_t);
      HANDLE_SPLAT(DT_UINT16, int, uint16_t);
      HANDLE_SPLAT(DT_UINT32, uint32, uint32_t);
      HANDLE_SPLAT(DT_UINT64, uint64, uint64_t);

      HANDLE_SPLAT(DT_FLOAT, float, float);
      HANDLE_SPLAT(DT_DOUBLE, double, double);

      HANDLE_SPLAT(DT_BFLOAT16, half, bfloat16);
      HANDLE_SPLAT(DT_HALF, half, Eigen::half);

#undef HANDLE_SPLAT

#define HANDLE_COMPLEX_SPLAT(DTYPE, field_name, xla_type)                     \
  case DTYPE:                                                                 \
    if (proto.field_name##_val_size() == 2) {                                 \
      return xla::ConstantR0<xla_type>(                                       \
          b, xla_type(proto.field_name##_val(0), proto.field_name##_val(1))); \
    }                                                                         \
    break;

      HANDLE_COMPLEX_SPLAT(DT_COMPLEX64, scomplex, xla::complex64);
      HANDLE_COMPLEX_SPLAT(DT_COMPLEX128, dcomplex, xla::complex128);

#undef HANDLE_COMPLEXSPLAT

      default:
        break;
    }
  }

  return xla::XlaOp();
}

bool IsDynamicExpressionProto(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kVariableId:
      return true;
    case ExpressionProto::kAddNode:
      return IsDynamicExpressionProto(proto.add_node().lhs()) ||
             IsDynamicExpressionProto(proto.add_node().rhs());
    case ExpressionProto::kSubNode:
      return IsDynamicExpressionProto(proto.sub_node().lhs()) ||
             IsDynamicExpressionProto(proto.sub_node().rhs());
    case ExpressionProto::kMulNode:
      return IsDynamicExpressionProto(proto.mul_node().lhs()) ||
             IsDynamicExpressionProto(proto.mul_node().rhs());
    case ExpressionProto::kDivNode:
      return IsDynamicExpressionProto(proto.div_node().lhs()) ||
             IsDynamicExpressionProto(proto.div_node().rhs());
    case ExpressionProto::kMaxNode:
      return IsDynamicExpressionProto(proto.max_node().lhs()) ||
             IsDynamicExpressionProto(proto.max_node().rhs());
    case ExpressionProto::kGtNode:
      return IsDynamicExpressionProto(proto.gt_node().lhs()) ||
             IsDynamicExpressionProto(proto.gt_node().rhs());
    case ExpressionProto::kSelectNode:
      return IsDynamicExpressionProto(proto.select_node().pred()) ||
             IsDynamicExpressionProto(proto.select_node().on_true()) ||
             IsDynamicExpressionProto(proto.select_node().on_false());
    case ExpressionProto::kConstantValue:
    case ExpressionProto::NODE_TYPE_NOT_SET:
      return false;
  }
}

static xla::DExpr DimExprToDExpr(const DimExpr* e) {
  if (e == nullptr) {
    return xla::DExpr();
  }
  switch (e->kind()) {
    case DimExpr::Kind::kConstant: {
      const auto* ac = static_cast<const Constant*>(e);
      return xla::DExpr::Const(ac->value());
    }
    case DimExpr::Kind::kVariable: {
      const auto* av = static_cast<const Variable*>(e);
      return xla::DExpr::Var(av->id());
    }
    case DimExpr::Kind::kAdd: {
      const auto* ee = static_cast<const ExprAdd*>(e);
      return DimExprToDExpr(ee->lhs()) + DimExprToDExpr(ee->rhs());
    }
    case DimExpr::Kind::kSub: {
      const auto* ee = static_cast<const ExprSub*>(e);
      return DimExprToDExpr(ee->lhs()) - DimExprToDExpr(ee->rhs());
    }
    case DimExpr::Kind::kMul: {
      const auto* ee = static_cast<const ExprMul*>(e);
      return DimExprToDExpr(ee->lhs()) * DimExprToDExpr(ee->rhs());
    }
    case DimExpr::Kind::kDiv: {
      const auto* ee = static_cast<const ExprDiv*>(e);
      return DimExprToDExpr(ee->lhs()) / DimExprToDExpr(ee->rhs());
    }
    case DimExpr::Kind::kMax: {
      const auto* ee = static_cast<const ExprMax*>(e);
      return xla::DExpr::Max(DimExprToDExpr(ee->lhs()),
                             DimExprToDExpr(ee->rhs()));
    }
    case DimExpr::Kind::kGt: {
      auto* ee = static_cast<const ExprGt*>(e);
      return xla::DExpr::Gt(DimExprToDExpr(ee->lhs()),
                            DimExprToDExpr(ee->rhs()));
    }
    case DimExpr::Kind::kSelect: {
      auto* ee = static_cast<const ExprSelect*>(e);
      return xla::DExpr::Select(DimExprToDExpr(ee->pred()),
                                DimExprToDExpr(ee->on_true()),
                                DimExprToDExpr(ee->on_false()));
    }
  }
  return xla::DExpr();
}

std::vector<xla::DExpr> BuildShapeContentsFromTensorShapeProto(
    const TensorShapeProto& shape) {
  std::vector<xla::DExpr> contents;
  contents.reserve(shape.dim_size());
  for (int i = 0; i < shape.dim_size(); ++i) {
    xla::DExpr expr;
    if (i < shape.expressions_size()) {
      auto tf_expr = DimExpr::FromProto(shape.expressions(i));
      expr = DimExprToDExpr(tf_expr.get());
    }
    if (expr) {
      xla::StringPrinter printer;
      expr->print(&printer);
      VLOG(1) << "BuildShapeContentsFromTensorShape dim=" << i
              << " expr=" << std::move(printer).ToString()
              << " dynamic=" << (expr->is_dynamic() ? "true" : "false");
    } else {
      VLOG(1) << "BuildShapeContentsFromTensorShape dim=" << i
              << " expr=_ dynamic=false";
    }
    contents.push_back(expr && expr->is_dynamic()
                           ? std::move(expr)
                           : xla::DExpr::Unknown(
                                 xla::kUnknownContentSentinel));
  }
  return contents;
}

bool CanAttachContentsFromTensorShapeProto(const TensorShape& tensor_shape,
                                           const TensorShapeProto& contents) {
  return (tensor_shape.dims() == 0 && contents.dim_size() == 1) ||
         (tensor_shape.dims() == 1 &&
          tensor_shape.dim_size(0) == contents.dim_size());
}

class ConstOp : public XlaOpKernel {
 public:
  explicit ConstOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
    const TensorProto* proto = nullptr;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("value", &proto));
    proto_ = *proto;
    OP_REQUIRES(
        ctx, ctx->output_type(0) == proto_.dtype(),
        errors::InvalidArgument("Type mismatch between value (",
                                DataTypeString(proto_.dtype()), ") and dtype (",
                                DataTypeString(ctx->output_type(0)), ")"));
    OP_REQUIRES_OK(ctx, TensorShape::IsValidShape(proto_.tensor_shape()));
  }

  void Compile(XlaOpKernelContext* ctx) override {
    xla::XlaBuilder* b = ctx->builder();

    bool has_dynamic = false;
    TensorShapeProto inferred_shape_proto;
    TensorShapeProto inferred_value_contents_proto;
    string inferred_value_contents_serialized;
    if (GetNodeAttr(ctx->op_kernel().def(), "has_dynamic", &has_dynamic).ok() &&
        has_dynamic) {
      GetNodeAttr(ctx->op_kernel().def(), "user_inferred_shape",
                  &inferred_shape_proto)
          .IgnoreError();
    }
    if (GetNodeAttr(ctx->op_kernel().def(), kUserInferredValueContentsAttrName,
                    &inferred_value_contents_serialized)
            .ok()) {
      if (!inferred_value_contents_proto.ParseFromString(
              inferred_value_contents_serialized)) {
        inferred_value_contents_proto.Clear();
      }
    }
    const bool has_contents_proto = inferred_value_contents_proto.dim_size() > 0;
    const TensorShapeProto& contents_proto =
        has_contents_proto ? inferred_value_contents_proto : inferred_shape_proto;

    // To avoid blowups for large constants filled with the same value,
    // recognize that case and emit a scalar broadcast instead.
    TensorShape shape(proto_.tensor_shape());
    if (shape.num_elements() > 1) {
      xla::XlaOp value = GetScalarConst(proto_, b);
      if (value.valid()) {
        if (has_dynamic) {
          VLOG(1) << "ConstOp broadcast fast path shape="
                    << shape.DebugString() << " inferred_rank="
                    << inferred_shape_proto.dim_size();
        }
        xla::XlaOp broadcast =
            xla::Broadcast(value, shape.dim_sizes(), shape.get_expressions());
        XlaExpression output =
            XlaExpression::XlaOp(broadcast, ctx->expected_output_dtype(0));
        if ((has_contents_proto || has_dynamic) &&
            CanAttachContentsFromTensorShapeProto(shape, contents_proto)) {
          output.set_contents(
              BuildShapeContentsFromTensorShapeProto(contents_proto));
        }
        ctx->SetOutputExpression(0, output);
        return;
      }
    }

    Tensor tensor(proto_.dtype());
    OP_REQUIRES(ctx, tensor.FromProto(cpu_allocator(), proto_),
                errors::InvalidArgument("Cannot parse tensor from proto: ",
                                        proto_.DebugString()));
    XlaExpression output = XlaExpression::Constant(tensor);
    if ((has_contents_proto || has_dynamic) &&
        CanAttachContentsFromTensorShapeProto(tensor.shape(), contents_proto)) {
      output.set_contents(
          BuildShapeContentsFromTensorShapeProto(contents_proto));
    }
    ctx->SetOutputExpression(0, output);
  }

 private:
  TensorProto proto_;
  ConstOp(const ConstOp&) = delete;
  void operator=(const ConstOp&) = delete;
};

// XLA_* devices also register a "real" Const operator so we suppress the
// dummy operator using CompilationOnly().
REGISTER_XLA_OP(Name("Const").CompilationOnly(), ConstOp);

}  // namespace
}  // namespace tensorflow

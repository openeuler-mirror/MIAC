#include "tensorflow/core/framework/tensor_shape_expr.h"

#include <memory>
#include <utility>
#include <vector>

#include "xla/parse_flags_from_env.h"
#include "xla/printer.h"
#include "xla/tsl/util/command_line_flags.h"

namespace tensorflow {
namespace {

bool ParseTensorShapeExpressionsEnabled() {
  bool tf_xla_enable_dynamic_sizes = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("tf_xla_enable_dynamic_sizes", &tf_xla_enable_dynamic_sizes,
                "XLA flag for enabling XLA dynamic sizes."),
  };
  xla::ParseFlagsFromEnvAndIgnoreUnknown(
      "TF_XLA_FLAGS", flag_list, /*reset_envvar=*/true);
  return tf_xla_enable_dynamic_sizes;
}

std::optional<bool>& TensorShapeExpressionsEnabledOverride() {
  static auto* enabled_override = new std::optional<bool>();
  return *enabled_override;
}

void DynExprToTensorFlowProto(const xla::DynExpr& expr,
                              ExpressionProto* proto) {
  proto->Clear();
  switch (expr.kind()) {
    case xla::DExpr::Kind::kUnknown:
      return;
    case xla::DExpr::Kind::kConstant:
      proto->set_constant_value(
          static_cast<const xla::Constant&>(expr).get_val());
      return;
    case xla::DExpr::Kind::kVariable:
      proto->set_variable_id(
          static_cast<const xla::Variable&>(expr).get_id());
      return;
    case xla::DExpr::Kind::kAdd: {
      const auto& node = static_cast<const xla::Add&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_add_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_add_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kSub: {
      const auto& node = static_cast<const xla::Sub&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_sub_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_sub_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kMul: {
      const auto& node = static_cast<const xla::Mul&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_mul_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_mul_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kDiv: {
      const auto& node = static_cast<const xla::Div&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_div_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_div_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kMax: {
      const auto& node = static_cast<const xla::MaxExpr&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_max_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_max_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kGt: {
      const auto& node = static_cast<const xla::GtExpr&>(expr);
      DynExprToTensorFlowProto(*node.get_lhs(),
                               proto->mutable_gt_node()->mutable_lhs());
      DynExprToTensorFlowProto(*node.get_rhs(),
                               proto->mutable_gt_node()->mutable_rhs());
      return;
    }
    case xla::DExpr::Kind::kSelect: {
      const auto& node = static_cast<const xla::SelectExpr&>(expr);
      DynExprToTensorFlowProto(*node.get_pred(),
                               proto->mutable_select_node()->mutable_pred());
      DynExprToTensorFlowProto(
          *node.get_on_true(), proto->mutable_select_node()->mutable_on_true());
      DynExprToTensorFlowProto(
          *node.get_on_false(), proto->mutable_select_node()->mutable_on_false());
      return;
    }
  }
}

}  // namespace

bool TensorShapeExpressionsEnabled() {
  if (TensorShapeExpressionsEnabledOverride().has_value()) {
    return *TensorShapeExpressionsEnabledOverride();
  }
  static const bool enabled = ParseTensorShapeExpressionsEnabled();
  return enabled;
}

void SetTensorShapeExpressionsEnabledForTesting(std::optional<bool> enabled) {
  TensorShapeExpressionsEnabledOverride() = enabled;
}

DimExpr DimExprFromProto(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kConstantValue:
      return DimExpr::Const(proto.constant_value());
    case ExpressionProto::kVariableId:
      return DimExpr::Var(proto.variable_id());
    case ExpressionProto::kAddNode:
      return DimExprFromProto(proto.add_node().lhs()) +
             DimExprFromProto(proto.add_node().rhs());
    case ExpressionProto::kSubNode:
      return DimExprFromProto(proto.sub_node().lhs()) -
             DimExprFromProto(proto.sub_node().rhs());
    case ExpressionProto::kMulNode:
      return DimExprFromProto(proto.mul_node().lhs()) *
             DimExprFromProto(proto.mul_node().rhs());
    case ExpressionProto::kDivNode:
      return DimExprFromProto(proto.div_node().lhs()) /
             DimExprFromProto(proto.div_node().rhs());
    case ExpressionProto::kMaxNode:
      return DimExpr::Max(DimExprFromProto(proto.max_node().lhs()),
                          DimExprFromProto(proto.max_node().rhs()));
    case ExpressionProto::kGtNode:
      return DimExpr::Gt(DimExprFromProto(proto.gt_node().lhs()),
                         DimExprFromProto(proto.gt_node().rhs()));
    case ExpressionProto::kSelectNode:
      return DimExpr::Select(DimExprFromProto(proto.select_node().pred()),
                             DimExprFromProto(proto.select_node().on_true()),
                             DimExprFromProto(proto.select_node().on_false()));
    case ExpressionProto::NODE_TYPE_NOT_SET:
    default:
      return DimExpr::Unknown(xla::kMissingExpressionSentinel);
  }
}

void DimExprToProto(const DimExpr& expr, ExpressionProto* proto) {
  if (!expr) {
    proto->Clear();
    return;
  }
  DynExprToTensorFlowProto(*expr, proto);
}

std::string DimExprDebugString(const DimExpr& expr) {
  if (!expr) return "_";
  xla::StringPrinter printer;
  expr->print(&printer);
  return std::move(printer).ToString();
}

DimExpr* SimplifyExpr(DimExpr* expr,
                      std::vector<std::unique_ptr<DimExpr>>* arena) {
  if (expr == nullptr) return nullptr;
  auto owned = std::make_unique<DimExpr>(expr->simplify());
  DimExpr* result = owned.get();
  arena->push_back(std::move(owned));
  return result;
}

bool IsDynamicDimExpr(const ExpressionProto& proto) {
  DimExpr expr = DimExprFromProto(proto);
  return expr && expr->is_dynamic();
}

bool HasDynamicDimExprs(const TensorShapeProto& proto) {
  for (const auto& expr : proto.expressions()) {
    if (IsDynamicDimExpr(expr)) return true;
  }
  return false;
}

}  // namespace tensorflow

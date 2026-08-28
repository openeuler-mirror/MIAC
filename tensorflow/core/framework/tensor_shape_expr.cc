/* Copyright 2026 The TensorFlow Authors.

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

#include "tensorflow/core/framework/tensor_shape_expr.h"

#include <vector>

#include "xla/parse_flags_from_env.h"
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

bool IsDynamicDimExpr(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kVariableId:
      return true;
    case ExpressionProto::kAddNode:
      return IsDynamicDimExpr(proto.add_node().lhs()) ||
             IsDynamicDimExpr(proto.add_node().rhs());
    case ExpressionProto::kSubNode:
      return IsDynamicDimExpr(proto.sub_node().lhs()) ||
             IsDynamicDimExpr(proto.sub_node().rhs());
    case ExpressionProto::kMulNode:
      return IsDynamicDimExpr(proto.mul_node().lhs()) ||
             IsDynamicDimExpr(proto.mul_node().rhs());
    case ExpressionProto::kDivNode:
      return IsDynamicDimExpr(proto.div_node().lhs()) ||
             IsDynamicDimExpr(proto.div_node().rhs());
    case ExpressionProto::kMaxNode:
      return IsDynamicDimExpr(proto.max_node().lhs()) ||
             IsDynamicDimExpr(proto.max_node().rhs());
    case ExpressionProto::kGtNode:
      return IsDynamicDimExpr(proto.gt_node().lhs()) ||
             IsDynamicDimExpr(proto.gt_node().rhs());
    case ExpressionProto::kSelectNode:
      return IsDynamicDimExpr(proto.select_node().pred()) ||
             IsDynamicDimExpr(proto.select_node().on_true()) ||
             IsDynamicDimExpr(proto.select_node().on_false());
    case ExpressionProto::kConstantValue:
    case ExpressionProto::NODE_TYPE_NOT_SET:
      return false;
  }
}

bool HasDynamicDimExprs(const TensorShapeProto& proto) {
  for (const auto& expr : proto.expressions()) {
    if (IsDynamicDimExpr(expr)) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<DimExpr> DimExpr::Cons(int64_t val) {
  return std::make_unique<Constant>(val);
}

std::unique_ptr<DimExpr> DimExpr::Var(int32_t id) {
  return std::make_unique<Variable>(id);
}

std::string DimExpr::DebugString() const {
  ExpressionProto proto;
  ToProto(&proto);
  return proto.DebugString();
}

static bool EqualsImpl(const DimExpr* a, const DimExpr* b) {
  if (a == b) return true;
  if (a == nullptr || b == nullptr) return false;
  if (a->kind() != b->kind()) return false;

  switch (a->kind()) {
    case DimExpr::Kind::kConstant: {
      auto* ac = static_cast<const Constant*>(a);
      auto* bc = static_cast<const Constant*>(b);
      return ac->value() == bc->value();
    }
    case DimExpr::Kind::kVariable: {
      auto* av = static_cast<const Variable*>(a);
      auto* bv = static_cast<const Variable*>(b);
      return av->id() == bv->id();
    }
    case DimExpr::Kind::kAdd: {
      auto* aa = static_cast<const ExprAdd*>(a);
      auto* ba = static_cast<const ExprAdd*>(b);
      return EqualsImpl(aa->lhs(), ba->lhs()) &&
             EqualsImpl(aa->rhs(), ba->rhs());
    }
    case DimExpr::Kind::kSub: {
      auto* as = static_cast<const ExprSub*>(a);
      auto* bs = static_cast<const ExprSub*>(b);
      return EqualsImpl(as->lhs(), bs->lhs()) &&
             EqualsImpl(as->rhs(), bs->rhs());
    }
    case DimExpr::Kind::kMul: {
      auto* am = static_cast<const ExprMul*>(a);
      auto* bm = static_cast<const ExprMul*>(b);
      return EqualsImpl(am->lhs(), bm->lhs()) &&
             EqualsImpl(am->rhs(), bm->rhs());
    }
    case DimExpr::Kind::kDiv: {
      auto* ad = static_cast<const ExprDiv*>(a);
      auto* bd = static_cast<const ExprDiv*>(b);
      return EqualsImpl(ad->lhs(), bd->lhs()) &&
             EqualsImpl(ad->rhs(), bd->rhs());
    }
    case DimExpr::Kind::kMax: {
      auto* am = static_cast<const ExprMax*>(a);
      auto* bm = static_cast<const ExprMax*>(b);
      return (EqualsImpl(am->lhs(), bm->lhs()) &&
              EqualsImpl(am->rhs(), bm->rhs())) ||
             (EqualsImpl(am->lhs(), bm->rhs()) &&
              EqualsImpl(am->rhs(), bm->lhs()));
    }
    case DimExpr::Kind::kGt: {
      auto* ag = static_cast<const ExprGt*>(a);
      auto* bg = static_cast<const ExprGt*>(b);
      return EqualsImpl(ag->lhs(), bg->lhs()) &&
             EqualsImpl(ag->rhs(), bg->rhs());
    }
    case DimExpr::Kind::kSelect: {
      auto* as = static_cast<const ExprSelect*>(a);
      auto* bs = static_cast<const ExprSelect*>(b);
      return EqualsImpl(as->pred(), bs->pred()) &&
             EqualsImpl(as->on_true(), bs->on_true()) &&
             EqualsImpl(as->on_false(), bs->on_false());
    }
  }

  return false;
}

bool DimExpr::Equals(const DimExpr* a, const DimExpr* b) {
  return EqualsImpl(a, b);
}

std::unique_ptr<DimExpr> DimExpr::FromProto(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kConstantValue:
      return DimExpr::Cons(proto.constant_value());
    case ExpressionProto::kVariableId:
      return DimExpr::Var(proto.variable_id());
    case ExpressionProto::kAddNode: {
      auto lhs = FromProto(proto.add_node().lhs());
      auto rhs = FromProto(proto.add_node().rhs());
      // Note: These are owning pointers, but ExprAdd takes raw pointers.
      // The caller must manage lifetime appropriately.
      return std::make_unique<ExprAdd>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kSubNode: {
      auto lhs = FromProto(proto.sub_node().lhs());
      auto rhs = FromProto(proto.sub_node().rhs());
      return std::make_unique<ExprSub>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kMulNode: {
      auto lhs = FromProto(proto.mul_node().lhs());
      auto rhs = FromProto(proto.mul_node().rhs());
      return std::make_unique<ExprMul>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kDivNode: {
      auto lhs = FromProto(proto.div_node().lhs());
      auto rhs = FromProto(proto.div_node().rhs());
      return std::make_unique<ExprDiv>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kMaxNode: {
      auto lhs = FromProto(proto.max_node().lhs());
      auto rhs = FromProto(proto.max_node().rhs());
      return std::make_unique<ExprMax>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kGtNode: {
      auto lhs = FromProto(proto.gt_node().lhs());
      auto rhs = FromProto(proto.gt_node().rhs());
      return std::make_unique<ExprGt>(lhs.release(), rhs.release());
    }
    case ExpressionProto::kSelectNode: {
      auto pred = FromProto(proto.select_node().pred());
      auto on_true = FromProto(proto.select_node().on_true());
      auto on_false = FromProto(proto.select_node().on_false());
      return std::make_unique<ExprSelect>(pred.release(), on_true.release(),
                                          on_false.release());
    }
    case ExpressionProto::NODE_TYPE_NOT_SET:
    default:
      return nullptr;
  }
}

DimExpr* SimplifyExpr(DimExpr* expr,
                      std::vector<std::unique_ptr<DimExpr>>* arena) {
  if (!expr) return nullptr;

  auto own = [arena](std::unique_ptr<DimExpr> e) -> DimExpr* {
    DimExpr* ptr = e.get();
    arena->push_back(std::move(e));
    return ptr;
  };

  switch (expr->kind()) {
    case DimExpr::Kind::kConstant:
    case DimExpr::Kind::kVariable:
      return expr;

    case DimExpr::Kind::kAdd: {
      auto* add = static_cast<ExprAdd*>(expr);
      DimExpr* lhs = SimplifyExpr(add->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(add->rhs(), arena);

      // Constant folding
      if (lhs->IsConstant() && rhs->IsConstant()) {
        return own(DimExpr::Cons(lhs->ConstantValue() + rhs->ConstantValue()));
      }

      // x + 0 → x
      if (rhs->IsConstant() && rhs->ConstantValue() == 0) return lhs;
      if (lhs->IsConstant() && lhs->ConstantValue() == 0) return rhs;

      return own(std::make_unique<ExprAdd>(lhs, rhs));
    }

    case DimExpr::Kind::kSub: {
      auto* sub = static_cast<ExprSub*>(expr);
      DimExpr* lhs = SimplifyExpr(sub->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(sub->rhs(), arena);

      // Constant folding
      if (lhs->IsConstant() && rhs->IsConstant()) {
        return own(DimExpr::Cons(lhs->ConstantValue() - rhs->ConstantValue()));
      }

      // x - 0 → x
      if (rhs->IsConstant() && rhs->ConstantValue() == 0) return lhs;

      return own(std::make_unique<ExprSub>(lhs, rhs));
    }

    case DimExpr::Kind::kMul: {
      auto* mul = static_cast<ExprMul*>(expr);
      DimExpr* lhs = SimplifyExpr(mul->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(mul->rhs(), arena);

      // Constant folding
      if (lhs->IsConstant() && rhs->IsConstant()) {
        return own(DimExpr::Cons(lhs->ConstantValue() * rhs->ConstantValue()));
      }

      // x * 1 → x
      if (rhs->IsConstant() && rhs->ConstantValue() == 1) return lhs;
      if (lhs->IsConstant() && lhs->ConstantValue() == 1) return rhs;

      // x * 0 → 0
      if (rhs->IsConstant() && rhs->ConstantValue() == 0)
        return own(DimExpr::Cons(0));
      if (lhs->IsConstant() && lhs->ConstantValue() == 0)
        return own(DimExpr::Cons(0));

      return own(std::make_unique<ExprMul>(lhs, rhs));
    }

    case DimExpr::Kind::kDiv: {
      auto* div = static_cast<ExprDiv*>(expr);
      DimExpr* lhs = SimplifyExpr(div->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(div->rhs(), arena);

      // Constant folding (avoid div by zero)
      if (lhs->IsConstant() && rhs->IsConstant()) {
        int64_t r = rhs->ConstantValue();
        if (r != 0) {
          return own(DimExpr::Cons(lhs->ConstantValue() / r));
        }
      }

      // x / 1 → x
      if (rhs->IsConstant() && rhs->ConstantValue() == 1) return lhs;

      return own(std::make_unique<ExprDiv>(lhs, rhs));
    }
    case DimExpr::Kind::kMax: {
      auto* max = static_cast<ExprMax*>(expr);
      DimExpr* lhs = SimplifyExpr(max->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(max->rhs(), arena);
      if (lhs->IsConstant() && rhs->IsConstant()) {
        return own(DimExpr::Cons(
            std::max(lhs->ConstantValue(), rhs->ConstantValue())));
      }
      if (lhs->IsConstant() && lhs->ConstantValue() == 0 &&
          rhs->kind() == DimExpr::Kind::kVariable) {
        return rhs;
      }
      if (rhs->IsConstant() && rhs->ConstantValue() == 0 &&
          lhs->kind() == DimExpr::Kind::kVariable) {
        return lhs;
      }
      if (DimExpr::Equals(lhs, rhs)) return lhs;
      return own(std::make_unique<ExprMax>(lhs, rhs));
    }
    case DimExpr::Kind::kGt: {
      auto* gt = static_cast<ExprGt*>(expr);
      DimExpr* lhs = SimplifyExpr(gt->lhs(), arena);
      DimExpr* rhs = SimplifyExpr(gt->rhs(), arena);
      if (lhs->IsConstant() && rhs->IsConstant()) {
        return own(DimExpr::Cons(lhs->ConstantValue() > rhs->ConstantValue()));
      }
      if (DimExpr::Equals(lhs, rhs)) return own(DimExpr::Cons(0));
      return own(std::make_unique<ExprGt>(lhs, rhs));
    }
    case DimExpr::Kind::kSelect: {
      auto* select = static_cast<ExprSelect*>(expr);
      DimExpr* pred = SimplifyExpr(select->pred(), arena);
      DimExpr* on_true = SimplifyExpr(select->on_true(), arena);
      DimExpr* on_false = SimplifyExpr(select->on_false(), arena);
      if (DimExpr::Equals(on_true, on_false)) return on_true;
      if (pred->IsConstant()) {
        return pred->ConstantValue() != 0 ? on_true : on_false;
      }
      return own(std::make_unique<ExprSelect>(pred, on_true, on_false));
    }
  }

  return expr;
}

}  // namespace tensorflow

/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/shape.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <ostream>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "xla/printer.h"

namespace xla {

namespace {

Constant* AsConstant(DynExpr* expr) {
  return expr != nullptr && expr->kind() == DExpr::Kind::kConstant
             ? static_cast<Constant*>(expr)
             : nullptr;
}

std::vector<DynExpr*> ExpressionChildren(DynExpr* expr) {
  CHECK(expr != nullptr);
  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
    case DExpr::Kind::kConstant:
    case DExpr::Kind::kVariable:
      return {};
    case DExpr::Kind::kAdd: {
      auto* add = static_cast<Add*>(expr);
      return {add->get_lhs(), add->get_rhs()};
    }
    case DExpr::Kind::kSub: {
      auto* sub = static_cast<Sub*>(expr);
      return {sub->get_lhs(), sub->get_rhs()};
    }
    case DExpr::Kind::kMul: {
      auto* mul = static_cast<Mul*>(expr);
      return {mul->get_lhs(), mul->get_rhs()};
    }
    case DExpr::Kind::kDiv: {
      auto* div = static_cast<Div*>(expr);
      return {div->get_lhs(), div->get_rhs()};
    }
  }
  return {};
}

DynExpr* FindSmallestCoveringSubexpression(DynExpr* expr) {
  CHECK(expr != nullptr);
  if (expr->kind() == DExpr::Kind::kVariable) {
    return expr;
  }

  DynExpr* common_core = nullptr;
  for (DynExpr* child : ExpressionChildren(expr)) {
    DynExpr* child_core = FindSmallestCoveringSubexpression(child);
    if (child_core == nullptr) {
      continue;
    }
    if (common_core == nullptr) {
      common_core = child_core;
    } else if (!DynExpr::equal(common_core, child_core)) {
      return expr;
    }
  }
  return common_core;
}

std::unique_ptr<DynExpr> ReplaceSubexpression(DynExpr* expr, DynExpr* target,
                                              DynExpr* replacement) {
  CHECK(expr != nullptr);
  CHECK(target != nullptr);
  CHECK(replacement != nullptr);
  if (DynExpr::equal(expr, target)) {
    return replacement->clone();
  }

  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
    case DExpr::Kind::kConstant:
    case DExpr::Kind::kVariable:
      return expr->clone();
    case DExpr::Kind::kAdd: {
      auto* add = static_cast<Add*>(expr);
      return std::make_unique<Add>(
          ReplaceSubexpression(add->get_lhs(), target, replacement).release(),
          ReplaceSubexpression(add->get_rhs(), target, replacement).release());
    }
    case DExpr::Kind::kSub: {
      auto* sub = static_cast<Sub*>(expr);
      return std::make_unique<Sub>(
          ReplaceSubexpression(sub->get_lhs(), target, replacement).release(),
          ReplaceSubexpression(sub->get_rhs(), target, replacement).release());
    }
    case DExpr::Kind::kMul: {
      auto* mul = static_cast<Mul*>(expr);
      return std::make_unique<Mul>(
          ReplaceSubexpression(mul->get_lhs(), target, replacement).release(),
          ReplaceSubexpression(mul->get_rhs(), target, replacement).release());
    }
    case DExpr::Kind::kDiv: {
      auto* div = static_cast<Div*>(expr);
      return std::make_unique<Div>(
          ReplaceSubexpression(div->get_lhs(), target, replacement).release(),
          ReplaceSubexpression(div->get_rhs(), target, replacement).release());
    }
  }
  return expr->clone();
}

void NormalizeFraction(int64_t* numerator, int64_t* denominator) {
  CHECK(denominator != nullptr);
  CHECK(*denominator != 0);
  int64_t divisor = std::gcd(std::llabs(*numerator), std::llabs(*denominator));
  if (divisor > 1) {
    *numerator /= divisor;
    *denominator /= divisor;
  }
  if (*denominator < 0) {
    *numerator = -*numerator;
    *denominator = -*denominator;
  }
}

// We normalize affine expressions into:
//   (constant + sum_i coeff_i * var_i) / denominator
//
// Concretely, the canonical forms we want to preserve are:
//   p
//   mX
//   mX + p
//   mX + nY
//   (mX) / d
//   (mX + p) / d
//   (mX + nY + p) / d
//
// This lets us combine equivalent trees into one stable representation, e.g.:
//   A/2 + A/2     -> A
//   A/2 + B/2     -> (A + B) / 2
//   4/8 * A       -> A / 2
//   A + (B + 1)   -> A + B + 1   (represented via coefficients + constant)
//
// The important constraint is that we only canonicalize affine expressions over
// an integer denominator. Non-affine expressions stay in tree form and only get
// a minimal local simplification in the fallback path below.
struct CanonicalAffineExpr {
  int64_t denominator = 1;
  int64_t constant = 0;
  std::map<int, int64_t> coefficients;

  bool IsPureConstant() const { return coefficients.empty(); }
};

void NormalizeAffine(CanonicalAffineExpr* expr) {
  CHECK(expr != nullptr);
  CHECK(expr->denominator != 0);
  for (auto it = expr->coefficients.begin(); it != expr->coefficients.end();) {
    if (it->second == 0) {
      it = expr->coefficients.erase(it);
    } else {
      ++it;
    }
  }
  int64_t divisor = std::llabs(expr->constant);
  for (const auto& [_, coeff] : expr->coefficients) {
    divisor = std::gcd(divisor, std::llabs(coeff));
  }
  divisor = std::gcd(divisor, std::llabs(expr->denominator));
  if (divisor > 1) {
    expr->constant /= divisor;
    expr->denominator /= divisor;
    for (auto& [_, coeff] : expr->coefficients) {
      coeff /= divisor;
    }
  }
  if (expr->denominator < 0) {
    expr->denominator = -expr->denominator;
    expr->constant = -expr->constant;
    for (auto& [_, coeff] : expr->coefficients) {
      coeff = -coeff;
    }
  }
  if (expr->constant == 0 && expr->coefficients.empty()) {
    expr->denominator = 1;
  }
}

CanonicalAffineExpr MakeConstantAffine(int64_t value) {
  CanonicalAffineExpr expr;
  expr.constant = value;
  return expr;
}

CanonicalAffineExpr MakeVariableAffine(int id) {
  CanonicalAffineExpr expr;
  expr.coefficients[id] = 1;
  return expr;
}

CanonicalAffineExpr AddAffine(const CanonicalAffineExpr& lhs,
                              const CanonicalAffineExpr& rhs, int rhs_sign) {
  CHECK(rhs_sign == 1 || rhs_sign == -1);
  CanonicalAffineExpr result;
  int64_t gcd = std::gcd(lhs.denominator, rhs.denominator);
  int64_t lhs_scale = rhs.denominator / gcd;
  int64_t rhs_scale = lhs.denominator / gcd;
  result.denominator = lhs.denominator * lhs_scale;
  result.constant =
      lhs.constant * lhs_scale + rhs_sign * rhs.constant * rhs_scale;
  for (const auto& [id, coeff] : lhs.coefficients) {
    result.coefficients[id] = coeff * lhs_scale;
  }
  for (const auto& [id, coeff] : rhs.coefficients) {
    result.coefficients[id] += rhs_sign * coeff * rhs_scale;
  }
  NormalizeAffine(&result);
  return result;
}

CanonicalAffineExpr MultiplyAffineByRational(const CanonicalAffineExpr& expr,
                                             int64_t numerator,
                                             int64_t denominator) {
  CHECK(denominator != 0);
  CanonicalAffineExpr result = expr;
  result.constant *= numerator;
  result.denominator *= denominator;
  for (auto& [_, coeff] : result.coefficients) {
    coeff *= numerator;
  }
  NormalizeAffine(&result);
  return result;
}

std::optional<CanonicalAffineExpr> ToCanonicalAffine(const DynExpr* expr) {
  CHECK(expr != nullptr);
  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
      return std::nullopt;
    case DExpr::Kind::kConstant:
      return MakeConstantAffine(static_cast<const Constant*>(expr)->get_val());
    case DExpr::Kind::kVariable:
      return MakeVariableAffine(static_cast<const Variable*>(expr)->get_id());
    case DExpr::Kind::kAdd: {
      const auto* add = static_cast<const Add*>(expr);
      auto lhs = ToCanonicalAffine(add->get_lhs());
      auto rhs = ToCanonicalAffine(add->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AddAffine(*lhs, *rhs, /*rhs_sign=*/1);
    }
    case DExpr::Kind::kSub: {
      const auto* sub = static_cast<const Sub*>(expr);
      auto lhs = ToCanonicalAffine(sub->get_lhs());
      auto rhs = ToCanonicalAffine(sub->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      return AddAffine(*lhs, *rhs, /*rhs_sign=*/-1);
    }
    case DExpr::Kind::kMul: {
      const auto* mul = static_cast<const Mul*>(expr);
      auto lhs = ToCanonicalAffine(mul->get_lhs());
      auto rhs = ToCanonicalAffine(mul->get_rhs());
      if (!lhs.has_value() || !rhs.has_value()) return std::nullopt;
      if (lhs->IsPureConstant()) {
        return MultiplyAffineByRational(*rhs, lhs->constant, lhs->denominator);
      }
      if (rhs->IsPureConstant()) {
        return MultiplyAffineByRational(*lhs, rhs->constant, rhs->denominator);
      }
      return std::nullopt;
    }
    case DExpr::Kind::kDiv: {
      const auto* div = static_cast<const Div*>(expr);
      auto lhs = ToCanonicalAffine(div->get_lhs());
      auto rhs = ToCanonicalAffine(div->get_rhs());
      if (!lhs.has_value() || !rhs.has_value() || !rhs->IsPureConstant() ||
          rhs->constant == 0) {
        return std::nullopt;
      }
      return MultiplyAffineByRational(*lhs, rhs->denominator, rhs->constant);
    }
    case DExpr::Kind::kMax:
    case DExpr::Kind::kGt:
    case DExpr::Kind::kSelect:
      return std::nullopt;
    default:
      return std::nullopt;
  }
  return std::nullopt;
}

std::unique_ptr<DynExpr> BuildScaledVariableTerm(int id, int64_t coefficient) {
  CHECK(coefficient != 0);
  if (coefficient == 1) {
    return std::make_unique<Variable>(id);
  }
  return std::make_unique<Mul>(DynExpr::_(coefficient), DynExpr::V(id));
}

std::unique_ptr<DynExpr> BuildAffineNumerator(const CanonicalAffineExpr& expr) {
  std::unique_ptr<DynExpr> result;
  for (const auto& [id, coeff] : expr.coefficients) {
    auto term = BuildScaledVariableTerm(id, coeff);
    if (result == nullptr) {
      result = std::move(term);
    } else {
      result = std::make_unique<Add>(result.release(), term.release());
    }
  }
  if (expr.constant != 0 || result == nullptr) {
    if (result == nullptr) {
      result = std::make_unique<Constant>(expr.constant);
    } else if (expr.constant > 0) {
      result = std::make_unique<Add>(result.release(),
                                     DynExpr::_(expr.constant));
    } else {
      result = std::make_unique<Sub>(result.release(),
                                     DynExpr::_(-expr.constant));
    }
  }
  return result;
}

std::unique_ptr<DynExpr> BuildCanonicalExpr(const CanonicalAffineExpr& expr) {
  CanonicalAffineExpr normalized = expr;
  NormalizeAffine(&normalized);
  auto numerator = BuildAffineNumerator(normalized);
  if (normalized.denominator == 1) {
    return numerator;
  }
  return std::make_unique<Div>(numerator.release(),
                               DynExpr::_(normalized.denominator));
}

std::unique_ptr<DynExpr> SimplifyFallback(const DynExpr* expr) {
  switch (expr->kind()) {
    case DExpr::Kind::kUnknown:
    case DExpr::Kind::kConstant:
    case DExpr::Kind::kVariable:
      return expr->clone();
    case DExpr::Kind::kAdd: {
      const auto* add = static_cast<const Add*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(add->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(add->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      if (l && r) return std::make_unique<Constant>(l->get_val() + r->get_val());
      if (l && l->get_val() == 0) return rhs;
      if (r && r->get_val() == 0) return lhs;
      return std::make_unique<Add>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kSub: {
      const auto* sub = static_cast<const Sub*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(sub->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(sub->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      if (l && r) return std::make_unique<Constant>(l->get_val() - r->get_val());
      if (r && r->get_val() == 0) return lhs;
      if (*lhs == *rhs) return std::make_unique<Constant>(0);
      return std::make_unique<Sub>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kMul: {
      const auto* mul = static_cast<const Mul*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(mul->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(mul->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      if (l && r) return std::make_unique<Constant>(l->get_val() * r->get_val());
      if ((l && l->get_val() == 0) || (r && r->get_val() == 0)) {
        return std::make_unique<Constant>(0);
      }
      if (l && l->get_val() == 1) return rhs;
      if (r && r->get_val() == 1) return lhs;
      if (r != nullptr) std::swap(lhs, rhs);
      return std::make_unique<Mul>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kDiv: {
      const auto* div = static_cast<const Div*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(div->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(div->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* lhs_constant = AsConstant(lhs.get());
      Constant* rhs_constant = AsConstant(rhs.get());
      if (rhs_constant != nullptr) {
        int64_t denominator = rhs_constant->get_val();
        CHECK_NE(denominator, 0) << "Cannot simplify division by zero";
        if (lhs_constant != nullptr) {
          int64_t numerator = lhs_constant->get_val();
          NormalizeFraction(&numerator, &denominator);
          if (denominator == 1) {
            return std::make_unique<Constant>(numerator);
          }
          return std::make_unique<Div>(DynExpr::_(numerator),
                                       DynExpr::_(denominator));
        }
        if (denominator == 1) {
          return lhs;
        }
      }
      if (*lhs == *rhs) return std::make_unique<Constant>(1);
      if (lhs->kind() == DExpr::Kind::kMul) {
        auto* mul = static_cast<Mul*>(lhs.get());
        auto lhs_l = std::unique_ptr<DynExpr>(mul->get_lhs()->s());
        auto lhs_r = std::unique_ptr<DynExpr>(mul->get_rhs()->s());
        if (*lhs_l == *rhs) {
          return lhs_r;
        }
        if (*lhs_r == *rhs) {
          return lhs_l;
        }
      }
      return std::make_unique<Div>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kMax: {
      const auto* max = static_cast<const MaxExpr*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(max->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(max->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      Constant* l = AsConstant(lhs.get());
      Constant* r = AsConstant(rhs.get());
      if (l && r) {
        return std::make_unique<Constant>(
            std::max(l->get_val(), r->get_val()));
      }
      if (l && l->get_val() == 0 &&
          rhs->kind() == DExpr::Kind::kVariable) {
        return rhs;
      }
      if (r && r->get_val() == 0 &&
          lhs->kind() == DExpr::Kind::kVariable) {
        return lhs;
      }
      if (*lhs == *rhs) return lhs;
      return std::make_unique<MaxExpr>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kGt: {
      const auto* gt = static_cast<const GtExpr*>(expr);
      auto lhs = std::unique_ptr<DynExpr>(gt->get_lhs()->s());
      auto rhs = std::unique_ptr<DynExpr>(gt->get_rhs()->s());
      if (lhs->kind() == DExpr::Kind::kUnknown ||
          rhs->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      if (lhs->is_constant() && rhs->is_constant()) {
        return std::make_unique<Constant>(lhs->get_val() > rhs->get_val());
      }
      if (*lhs == *rhs) return std::make_unique<Constant>(0);
      return std::make_unique<GtExpr>(lhs.release(), rhs.release());
    }
    case DExpr::Kind::kSelect: {
      const auto* select = static_cast<const SelectExpr*>(expr);
      auto pred = std::unique_ptr<DynExpr>(select->get_pred()->s());
      auto on_true = std::unique_ptr<DynExpr>(select->get_on_true()->s());
      auto on_false = std::unique_ptr<DynExpr>(select->get_on_false()->s());
      if (pred->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      if (pred->is_constant()) {
        return pred->get_val() != 0 ? std::move(on_true) : std::move(on_false);
      }
      if (on_true->kind() == DExpr::Kind::kUnknown ||
          on_false->kind() == DExpr::Kind::kUnknown) {
        return std::make_unique<UnknownExpr>();
      }
      if (*on_true == *on_false) return on_true;
      return std::make_unique<SelectExpr>(pred.release(), on_true.release(),
                                           on_false.release());
    }
    default:
      return expr->clone();
  }
}

std::unique_ptr<DynExpr> SimplifyCanonical(const DynExpr* expr) {
  if (expr->kind() == DExpr::Kind::kUnknown) {
    return std::make_unique<UnknownExpr>();
  }
  if (auto canonical = ToCanonicalAffine(expr); canonical.has_value()) {
    if (canonical->IsPureConstant()) {
      CHECK_NE(canonical->denominator, 0);
      return std::make_unique<Constant>(canonical->constant /
                                        canonical->denominator);
    }
    return BuildCanonicalExpr(*canonical);
  }
  return SimplifyFallback(expr);
}

}  // namespace

DExpr DExpr::find_smallest_subexpression_covering_all_variables() const {
  CHECK(expr_ != nullptr);
  const std::set<int> ids = expr_->get_all_ids();
  CHECK(!ids.empty());
  DynExpr* result = FindSmallestCoveringSubexpression(expr_.get());
  CHECK(result != nullptr);
  return DExpr::Adopt(result->clone().release());
}

DExpr DExpr::replace_subexpression(const DExpr& target,
                                   const DExpr& replacement) const {
  if (expr_ == nullptr) {
    return DExpr();
  }
  return DExpr(ReplaceSubexpression(expr_.get(), target.get(),
                                    replacement.get()));
}

const DExpr& Shape::MissingExpression() {
  static const DExpr missing = DExpr::Unknown(kMissingExpressionSentinel);
  return missing;
}

DynExpr* operator*(DynExpr& lhs, DynExpr& rhs) {
  return new Mul(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator*(int64_t k, DynExpr& rhs) {
  return new Mul(DynExpr::_(k), rhs.clone().release());
}
DynExpr* operator/(DynExpr& lhs, DynExpr& rhs) {
  return new Div(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator/(DynExpr& lhs, int64_t d) {
  return new Div(lhs.clone().release(), DynExpr::_(d));
}
DynExpr* operator+(DynExpr& lhs, DynExpr& rhs) {
  return new Add(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator+(DynExpr& lhs, int64_t d) {
  return new Add(lhs.clone().release(), DynExpr::_(d));
}
DynExpr* operator-(DynExpr& lhs, DynExpr& rhs) {
  return new Sub(lhs.clone().release(), rhs.clone().release());
}
DynExpr* operator-(DynExpr& lhs, int64_t d) {
  return new Sub(lhs.clone().release(), DynExpr::_(d));
}
bool operator==(DynExpr& lhs, DynExpr& rhs) {
  return DynExpr::equal(&lhs, &rhs);
}
bool operator==(DynExpr& lhs, int64_t d) {
  auto rhs = std::unique_ptr<DynExpr>(DynExpr::_(d));
  return DynExpr::equal(&lhs, rhs.get());
}
bool operator<(DynExpr& lhs, int64_t d) {
  return lhs.is_constant() && lhs.get_val() < d;
}

DExpr DExpr::Max(const DExpr& lhs, const DExpr& rhs) {
  return Adopt(new xla::MaxExpr(lhs.clone().release(), rhs.clone().release()));
}

DExpr DExpr::Gt(const DExpr& lhs, const DExpr& rhs) {
  return Adopt(new xla::GtExpr(lhs.clone().release(), rhs.clone().release()));
}

DExpr DExpr::Select(const DExpr& pred, const DExpr& on_true,
                    const DExpr& on_false) {
  return Adopt(new xla::SelectExpr(pred.clone().release(),
                                    on_true.clone().release(),
                                    on_false.clone().release()));
}

bool DynExpr::equal(DynExpr* expr1, DynExpr* expr2) {
  auto e1 = std::unique_ptr<DynExpr>(expr1->s());
  auto e2 = std::unique_ptr<DynExpr>(expr2->s());
  if (e1 == nullptr || e2 == nullptr) return false;
  auto a1 = ToCanonicalAffine(e1.get());
  auto a2 = ToCanonicalAffine(e2.get());
  if (a1.has_value() && a2.has_value()) {
    return a1->denominator == a2->denominator &&
           a1->constant == a2->constant &&
           a1->coefficients == a2->coefficients;
  }
  if (e1->kind() == DExpr::Kind::kConstant &&
      e2->kind() == DExpr::Kind::kConstant) {
    return static_cast<Constant*>(e1.get())->get_val() ==
           static_cast<Constant*>(e2.get())->get_val();
  }
  if (e1->kind() == DExpr::Kind::kVariable &&
      e2->kind() == DExpr::Kind::kVariable) {
    return static_cast<Variable*>(e1.get())->get_id() ==
           static_cast<Variable*>(e2.get())->get_id();
  }
  if (e1->kind() == DExpr::Kind::kUnknown &&
      e2->kind() == DExpr::Kind::kUnknown) {
    int lhs_id = static_cast<UnknownExpr*>(e1.get())->get_id();
    int rhs_id = static_cast<UnknownExpr*>(e2.get())->get_id();
    return lhs_id != 0 && lhs_id == rhs_id;
  }
  if (e1->kind() == DExpr::Kind::kMul && e2->kind() == DExpr::Kind::kMul) {
    auto* ab = static_cast<Mul*>(e1.get());
    auto* cd = static_cast<Mul*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return (*a == *c && *b == *d) || (*a == *d && *b == *c);
  }
  if (e1->kind() == DExpr::Kind::kDiv && e2->kind() == DExpr::Kind::kDiv) {
    auto* ab = static_cast<Div*>(e1.get());
    auto* cd = static_cast<Div*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return *a == *c && *b == *d;
  }
  if (e1->kind() == DExpr::Kind::kAdd && e2->kind() == DExpr::Kind::kAdd) {
    auto* ab = static_cast<Add*>(e1.get());
    auto* cd = static_cast<Add*>(e2.get());
    auto a = ab->get_lhs();
    auto b = ab->get_rhs();
    auto c = cd->get_lhs();
    auto d = cd->get_rhs();
    return (*a == *c && *b == *d) || (*a == *d && *b == *c);
  }
  if (e1->kind() == DExpr::Kind::kSub && e2->kind() == DExpr::Kind::kSub) {
    auto* ab = static_cast<Sub*>(e1.get());
    auto* cd = static_cast<Sub*>(e2.get());
    auto* a = ab->get_lhs();
    auto* b = ab->get_rhs();
    auto* c = cd->get_lhs();
    auto* d = cd->get_rhs();
    return *a == *c && *b == *d;
  }
  if (e1->kind() == DExpr::Kind::kMax && e2->kind() == DExpr::Kind::kMax) {
    auto* ab = static_cast<MaxExpr*>(e1.get());
    auto* cd = static_cast<MaxExpr*>(e2.get());
    auto* a = ab->get_lhs();
    auto* b = ab->get_rhs();
    auto* c = cd->get_lhs();
    auto* d = cd->get_rhs();
    return (*a == *c && *b == *d) || (*a == *d && *b == *c);
  }
  if (e1->kind() == DExpr::Kind::kGt && e2->kind() == DExpr::Kind::kGt) {
    auto* lhs = static_cast<GtExpr*>(e1.get());
    auto* rhs = static_cast<GtExpr*>(e2.get());
    return *lhs->get_lhs() == *rhs->get_lhs() &&
           *lhs->get_rhs() == *rhs->get_rhs();
  }
  if (e1->kind() == DExpr::Kind::kSelect &&
      e2->kind() == DExpr::Kind::kSelect) {
    auto* lhs = static_cast<SelectExpr*>(e1.get());
    auto* rhs = static_cast<SelectExpr*>(e2.get());
    return *lhs->get_pred() == *rhs->get_pred() &&
           *lhs->get_on_true() == *rhs->get_on_true() &&
           *lhs->get_on_false() == *rhs->get_on_false();
  }
  return false;
}

DynExpr* Constant::s() { return SimplifyCanonical(this).release(); }

DynExpr* Variable::s() { return SimplifyCanonical(this).release(); }

DynExpr* Mul::s() { return SimplifyCanonical(this).release(); }

DynExpr* Add::s() { return SimplifyCanonical(this).release(); }

DynExpr* Sub::s() { return SimplifyCanonical(this).release(); }

DynExpr* Div::s() { return SimplifyCanonical(this).release(); }

DynExpr* MaxExpr::s() { return SimplifyCanonical(this).release(); }

DynExpr* GtExpr::s() { return SimplifyCanonical(this).release(); }

DynExpr* SelectExpr::s() { return SimplifyCanonical(this).release(); }

std::ostream& operator<<(std::ostream& os, DynExpr* expr) {
  auto simplified = std::unique_ptr<DynExpr>(expr->s());
  StringPrinter printer;
  simplified->print(&printer);
  os << std::move(printer).ToString();
  return os;
}

DynExpr* DynExpr::zero = new Constant(0);
DynExpr* DynExpr::one = new Constant(1);

}  // namespace xla

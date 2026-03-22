#include "xla/service/dynamic_constant_rewriter.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/protobuf.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

DynExpr* DynExprFromProto(const ExpressionProto& proto) {
  switch (proto.node_type_case()) {
    case ExpressionProto::kConstantValue:
      return DynExpr::_(proto.constant_value());
    case ExpressionProto::kVariableId:
      return DynExpr::V(proto.variable_id());
    case ExpressionProto::kAddNode: {
      const auto& add = proto.add_node();
      return new Add(DynExprFromProto(add.lhs()), DynExprFromProto(add.rhs()));
    }
    case ExpressionProto::kSubNode: {
      const auto& sub = proto.sub_node();
      return new Sub(DynExprFromProto(sub.lhs()), DynExprFromProto(sub.rhs()));
    }
    case ExpressionProto::kMulNode: {
      const auto& mul = proto.mul_node();
      return new Mul(DynExprFromProto(mul.lhs()), DynExprFromProto(mul.rhs()));
    }
    case ExpressionProto::kDivNode: {
      const auto& div = proto.div_node();
      return new Div(DynExprFromProto(div.lhs()), DynExprFromProto(div.rhs()));
    }
    case ExpressionProto::NODE_TYPE_NOT_SET:
    default:
      return nullptr;
  }
}

absl::StatusOr<HloInstruction*> BuildDynamicConstantReplacement(
    HloInstruction* constant_instr) {
  TF_RET_CHECK(constant_instr->opcode() == HloOpcode::kConstant);
  TF_RET_CHECK(constant_instr->has_frontend_attributes());

  const auto& attrs = constant_instr->frontend_attributes().map();
  auto index_it = attrs.find("dynamic_constant_index");
  auto expr_it = attrs.find("dynamic_constant_expr");
  TF_RET_CHECK(index_it != attrs.end());
  TF_RET_CHECK(expr_it != attrs.end());

  int64_t dynamic_index;
  TF_RET_CHECK(absl::SimpleAtoi(index_it->second, &dynamic_index))
      << "Failed to parse dynamic_constant_index=" << index_it->second;

  ExpressionProto expr_proto;
  TF_RET_CHECK(tsl::protobuf::TextFormat::ParseFromString(expr_it->second,
                                                          &expr_proto))
      << "Failed to parse dynamic_constant_expr=" << expr_it->second;
  DynExpr* expr = DynExprFromProto(expr_proto);
  TF_RET_CHECK(expr != nullptr);

  const Shape& shape = constant_instr->shape();
  TF_RET_CHECK(shape.IsArray());
  TF_RET_CHECK(shape.element_type() == S32 || shape.element_type() == S64)
      << "Only s32/s64 marked constants are supported";
  TF_RET_CHECK(shape.dimensions_size() <= 1)
      << "Only scalar and rank-1 marked constants are supported";

  int64_t carrier_bound;
  if (shape.dimensions_size() == 0) {
    dynamic_index = 0;
    carrier_bound = shape.element_type() == S32
                        ? constant_instr->literal().GetFirstElement<int32_t>()
                        : constant_instr->literal().GetFirstElement<int64_t>();
  } else {
    TF_RET_CHECK(dynamic_index >= 0 && dynamic_index < shape.dimensions(0))
        << "dynamic_constant_index=" << dynamic_index
        << " out of bounds for shape " << shape.ToString();
    carrier_bound = shape.element_type() == S32
                        ? constant_instr->literal().data<int32_t>()[dynamic_index]
                        : constant_instr->literal().data<int64_t>()[dynamic_index];
  }

  TF_RET_CHECK(carrier_bound <= std::numeric_limits<int32_t>::max() &&
               carrier_bound >= std::numeric_limits<int32_t>::min())
      << "GetExpressionValue carriers must fit in s32, got " << carrier_bound;

  HloComputation* computation = constant_instr->parent();
  Shape carrier_shape = ShapeUtil::MakeShape(S32, {1});
  carrier_shape.set_expression(0, expr);
  HloInstruction* carrier = computation->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int32_t>(
              {static_cast<int32_t>(carrier_bound)})));
  *carrier->mutable_shape() = carrier_shape;

  HloInstruction* runtime_value = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          ShapeUtil::MakeShape(S32, {}), {carrier}, "GetExpressionValue"));
  if (shape.element_type() == S64) {
    runtime_value = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::MakeShape(S64, {}), runtime_value));
  }

  if (shape.dimensions_size() == 0) {
    return runtime_value;
  }

  HloInstruction* base_constant =
      computation->AddInstruction(constant_instr->Clone());
  base_constant->erase_frontend_attribute("dynamic_constant_index");
  base_constant->erase_frontend_attribute("dynamic_constant_expr");
  Shape update_shape = ShapeUtil::MakeShape(shape.element_type(), {1});
  HloInstruction* update = computation->AddInstruction(
      HloInstruction::CreateReshape(update_shape, runtime_value));
  HloInstruction* start_index = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
          static_cast<int32_t>(dynamic_index))));
  return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      shape, base_constant, update, {start_index}));
}

bool IsMarkedDynamicConstant(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kConstant &&
         instr->has_frontend_attributes() &&
         instr->get_frontend_attribute("dynamic_constant_index").has_value() &&
         instr->get_frontend_attribute("dynamic_constant_expr").has_value();
}

}  // namespace

absl::StatusOr<bool> DynamicConstantRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::vector<HloInstruction*> marked_constants;
    for (HloInstruction* instruction : computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kConstant) {
        const bool is_marked = IsMarkedDynamicConstant(instruction);
        VLOG(1) << "DynamicConstantRewriter sees constant "
                << instruction->name() << " shape="
                << instruction->shape().ToString()
                << " literal=" << instruction->literal().ToString()
                << " marked=" << is_marked;
        if (is_marked) {
          VLOG(1) << "  dynamic_constant_index="
                  << *instruction->get_frontend_attribute(
                         "dynamic_constant_index")
                  << " dynamic_constant_expr="
                  << *instruction->get_frontend_attribute(
                         "dynamic_constant_expr");
          marked_constants.push_back(instruction);
        }
      }
    }

    for (HloInstruction* constant_instr : marked_constants) {
      if (constant_instr->IsDead()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(HloInstruction * replacement,
                          BuildDynamicConstantReplacement(constant_instr));
      VLOG(1) << "Rewriting marked constant " << constant_instr->name()
              << " into " << replacement->name() << " ("
              << HloOpcodeString(replacement->opcode()) << ")";
      TF_RETURN_IF_ERROR(
          computation->ReplaceInstruction(constant_instr, replacement));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla

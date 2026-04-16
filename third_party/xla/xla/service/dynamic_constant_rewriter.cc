#include "xla/service/dynamic_constant_rewriter.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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

absl::StatusOr<HloInstruction*> BuildDynamicConstantReplacement(
    HloInstruction* constant_instr) {
  TF_RET_CHECK(constant_instr->opcode() == HloOpcode::kConstant);
  TF_RET_CHECK(constant_instr->has_contents());

  const Shape& shape = constant_instr->shape();
  TF_RET_CHECK(shape.IsArray());
  TF_RET_CHECK(shape.element_type() == S32 || shape.element_type() == S64)
      << "Only s32/s64 marked constants are supported";
  TF_RET_CHECK(shape.dimensions_size() <= 1)
      << "Only scalar and rank-1 marked constants are supported";

  int64_t dynamic_index = 0;
  DExpr expr;
  for (int64_t i = 0; i < constant_instr->contents().size(); ++i) {
    DExpr candidate = DExprFromProto(constant_instr->contents()[i]);
    if (candidate && candidate->is_dynamic()) {
      dynamic_index = i;
      expr = std::move(candidate);
      break;
    }
  }
  TF_RET_CHECK(expr) << "Marked dynamic constant is missing dynamic contents";

  int64_t carrier_bound;
  if (shape.dimensions_size() == 0) {
    dynamic_index = 0;
    carrier_bound = shape.element_type() == S32
                        ? constant_instr->literal().GetFirstElement<int32_t>()
                        : constant_instr->literal().GetFirstElement<int64_t>();
  } else {
    TF_RET_CHECK(dynamic_index >= 0 && dynamic_index < shape.dimensions(0))
        << "dynamic content index=" << dynamic_index
        << " out of bounds for shape " << shape.ToString();
    carrier_bound = shape.element_type() == S32
                        ? constant_instr->literal().data<int32_t>()[dynamic_index]
                        : constant_instr->literal().data<int64_t>()[dynamic_index];
  }

  TF_RET_CHECK(carrier_bound <= std::numeric_limits<int32_t>::max() &&
               carrier_bound >= std::numeric_limits<int32_t>::min())
      << "GetExpressionValue carriers must fit in s32, got " << carrier_bound;

  HloComputation* computation = constant_instr->parent();
  HloInstruction* carrier = computation->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::CreateR1<int32_t>(
              {static_cast<int32_t>(carrier_bound)})));
  ExpressionProto expr_proto;
  expr.to_proto(&expr_proto);

  HloInstruction* runtime_value = computation->AddInstruction(
      HloInstruction::CreateCustomCall(
          ShapeUtil::MakeShape(S32, {}), {carrier}, "GetExpressionValue"));
  runtime_value->set_contents({std::move(expr_proto)});
  if (shape.element_type() == S64) {
    runtime_value = computation->AddInstruction(HloInstruction::CreateConvert(
        ShapeUtil::MakeShape(S64, {}), runtime_value));
  }

  if (shape.dimensions_size() == 0) {
    return runtime_value;
  }

  HloInstruction* base_constant =
      computation->AddInstruction(constant_instr->Clone());
  base_constant->set_contents({});
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
  return instr->opcode() == HloOpcode::kConstant && instr->has_contents();
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
          VLOG(1) << "  contents_size=" << instruction->contents().size();
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
      replacement->erase_frontend_attribute("dynamic_constant_index");
      replacement->erase_frontend_attribute("dynamic_constant_expr");
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla

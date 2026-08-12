/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/dynamic_constant_rewriter.h"

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/shape_expr.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using DynamicConstantRewriterTest = HloTestBase;

TEST_F(DynamicConstantRewriterTest, ReusesIdenticalRuntimeExpression) {
  HloComputation::Builder builder(TestName());
  auto make_dynamic_constant = [&builder]() {
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(32)));
    ExpressionProto expression;
    DExpr::Var(1).to_proto(&expression);
    constant->set_contents({std::move(expression)});
    return constant;
  };

  HloInstruction* first = make_dynamic_constant();
  HloInstruction* second = make_dynamic_constant();
  builder.AddInstruction(HloInstruction::CreateTuple({first, second}));

  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build());

  ASSERT_TRUE(DynamicConstantRewriter().Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  ASSERT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->operand(0), root->operand(1));
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kCustomCall);
  EXPECT_EQ(root->operand(0)->custom_call_target(), "GetExpressionValue");
}

TEST_F(DynamicConstantRewriterTest,
       ReusesRuntimeExpressionWithLargestCarrierBound) {
  for (const auto& [first_bound, second_bound] :
       {std::pair<int32_t, int32_t>{32, 64}, {64, 32}}) {
    SCOPED_TRACE(testing::Message()
                 << "first_bound=" << first_bound
                 << ", second_bound=" << second_bound);
    HloComputation::Builder builder(TestName());
    auto make_dynamic_constant = [&builder](int32_t carrier_bound) {
      HloInstruction* constant = builder.AddInstruction(
          HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(carrier_bound)));
      ExpressionProto expression;
      DExpr::Var(1).to_proto(&expression);
      constant->set_contents({std::move(expression)});
      return constant;
    };

    HloInstruction* first = make_dynamic_constant(first_bound);
    HloInstruction* second = make_dynamic_constant(second_bound);
    builder.AddInstruction(HloInstruction::CreateTuple({first, second}));

    std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
    HloComputation* computation =
        module->AddEntryComputation(builder.Build());

    ASSERT_TRUE(DynamicConstantRewriter().Run(module.get()).value());

    HloInstruction* root = computation->root_instruction();
    ASSERT_EQ(root->operand_count(), 2);
    ASSERT_EQ(root->operand(0), root->operand(1));
    HloInstruction* runtime_value = root->mutable_operand(0);
    ASSERT_EQ(runtime_value->opcode(), HloOpcode::kCustomCall);
    ASSERT_EQ(runtime_value->custom_call_target(), "GetExpressionValue");
    ASSERT_EQ(runtime_value->operand_count(), 1);
    EXPECT_EQ(runtime_value->operand(0)->literal().GetFirstElement<int32_t>(),
              64);
  }
}

TEST_F(DynamicConstantRewriterTest, ReusesEquivalentRuntimeExpressions) {
  HloComputation::Builder builder(TestName());
  auto make_dynamic_constant = [&builder](const DExpr& expr) {
    HloInstruction* constant = builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(32)));
    ExpressionProto expression;
    expr.to_proto(&expression);
    constant->set_contents({std::move(expression)});
    return constant;
  };

  HloInstruction* first =
      make_dynamic_constant(DExpr::Var(1) + DExpr::Const(1));
  HloInstruction* second =
      make_dynamic_constant(DExpr::Const(1) + DExpr::Var(1));
  builder.AddInstruction(HloInstruction::CreateTuple({first, second}));

  std::unique_ptr<HloModule> module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(builder.Build());

  ASSERT_TRUE(DynamicConstantRewriter().Run(module.get()).value());

  HloInstruction* root = computation->root_instruction();
  ASSERT_EQ(root->operand_count(), 2);
  EXPECT_EQ(root->operand(0), root->operand(1));
}

}  // namespace
}  // namespace xla

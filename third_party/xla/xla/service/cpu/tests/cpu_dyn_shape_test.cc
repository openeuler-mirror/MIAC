/* Copyright 2020 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/test_target_triple_helper.h"
#include "xla/service/cpu/tests/cpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace {

using CpuDynamicShapeTest = CpuCodegenTest;

TEST_F(CpuDynamicShapeTest, DynamicShapeR2) {
  HloComputation::Builder builder(TestName());

  xla::Shape dyn_input_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 4});
  dyn_input_shape.set_dynamic_dimension(0, true);
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dyn_input_shape, "x"));

  builder.AddInstruction(HloInstruction::CreateUnary(
      dyn_input_shape, HloOpcode::kNegate, param_x));
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  std::string filecheck_pattern = R"(
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size:.*]] to i64
; CHECK: icmp uge i64 %[[custom:.*]], %[[i64_dyn_dim_size:.*]]
; CHECK: %[[multiplier:.*]] = mul i64 1, %[[i64_dyn_dim_size:.*]]
; CHECK: mul nuw nsw i64 %[[custom:.*]], %[[multiplier:.*]]
)";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options,
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuDynamicShapeTest, DynamicExpressionConcatDoesNotUseFastPath) {
  HloComputation::Builder builder(TestName());

  std::vector<bool> dynamic_dimensions = {true, false, false};
  std::vector<DExpr> dyn_operand_exprs = {DExpr::Var(0), DExpr::Const(26),
                                          DExpr::Const(4)};
  std::vector<DExpr> concat_exprs = {DExpr::Var(0), DExpr::Const(26),
                                     DExpr::Const(8)};
  Shape dyn_operand_shape = ShapeUtil::MakeShape(
      F32, {128, 26, 4}, dynamic_dimensions, dyn_operand_exprs);
  Shape concat_shape =
      ShapeUtil::MakeShape(F32, {128, 26, 8}, dynamic_dimensions, concat_exprs);

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dyn_operand_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, dyn_operand_shape, "rhs"));
  HloInstruction* concat = builder.AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape, {lhs, rhs}, 2));

  builder.AddInstruction(HloInstruction::CreateBinary(
      concat_shape, HloOpcode::kAdd, concat, concat));

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  std::string filecheck_pattern = R"(
; CHECK-NOT: llvm.memcpy
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size]] to i64
; CHECK: icmp uge i64
)";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options,
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuDynamicShapeTest, StaticConcatUsesFastPath) {
  HloComputation::Builder builder(TestName());

  Shape operand_shape = ShapeUtil::MakeShape(F32, {128, 26, 4});
  Shape concat_shape = ShapeUtil::MakeShape(F32, {128, 26, 8});

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateParameter(0, operand_shape, "lhs"));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateParameter(1, operand_shape, "rhs"));
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(concat_shape, {lhs, rhs}, 2));

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  std::string filecheck_pattern = R"(
; CHECK: llvm.memcpy
)";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  hlo_module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_use_thunk_runtime(false);

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options,
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

}  // namespace
}  // namespace cpu
}  // namespace xla

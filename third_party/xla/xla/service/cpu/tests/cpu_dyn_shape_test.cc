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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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

CpuAotCompilationOptions GetAotOptions() {
  return CpuAotCompilationOptions{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};
}

Shape MakeDynamicDenseShape(absl::Span<const int64_t> dimensions,
                            absl::Span<const int64_t> minor_to_major,
                            int64_t dynamic_dimension,
                            const DExpr& dynamic_expression) {
  Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, dimensions, minor_to_major);
  shape.set_dynamic_dimension(dynamic_dimension, true);
  shape.set_expression(dynamic_dimension, dynamic_expression);
  return shape;
}

class CpuDynamicShapeTest : public CpuCodegenTest {
 protected:
  std::unique_ptr<HloModule> CreateDynamicFastConcatModule(
      absl::string_view name, bool add_consumer, bool three_operands = false) {
    HloComputation::Builder builder(name);

    const DExpr dynamic_batch = DExpr::Var(0);
    Shape operand_shape = MakeDynamicDenseShape({64, 26, 4}, {2, 1, 0},
                                                /*dynamic_dimension=*/0,
                                                dynamic_batch);
    Shape concat_shape =
        MakeDynamicDenseShape({64, 26, three_operands ? 12 : 8}, {2, 1, 0},
                              /*dynamic_dimension=*/0, dynamic_batch);

    HloInstruction* lhs = builder.AddInstruction(
        HloInstruction::CreateParameter(0, operand_shape, "lhs"));
    HloInstruction* rhs = builder.AddInstruction(
        HloInstruction::CreateParameter(1, operand_shape, "rhs"));

    HloInstruction* concat = nullptr;
    if (three_operands) {
      HloInstruction* third = builder.AddInstruction(
          HloInstruction::CreateParameter(2, operand_shape, "third"));
      concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {lhs, rhs, third}, /*dimension=*/2));
    } else {
      concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
          concat_shape, {lhs, rhs}, /*dimension=*/2));
    }

    if (add_consumer) {
      int64_t addend_param_number = three_operands ? 3 : 2;
      HloInstruction* addend =
          builder.AddInstruction(HloInstruction::CreateParameter(
              addend_param_number, concat_shape, "addend"));
      builder.AddInstruction(HloInstruction::CreateBinary(
          concat_shape, HloOpcode::kAdd, concat, addend));
    }

    auto hlo_module = CreateNewVerifiedModule();
    hlo_module->AddEntryComputation(builder.Build());
    auto& debug_options = hlo_module->mutable_config().mutable_debug_options();
    debug_options.set_xla_cpu_use_thunk_runtime(false);
    debug_options.add_xla_disable_hlo_passes("fusion");
    return hlo_module;
  }
};

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

TEST_F(CpuDynamicShapeTest, DynamicFastConcatenateCompilesWithoutFusion) {
  auto hlo_module =
      CreateDynamicFastConcatModule(TestName(), /*add_consumer=*/false);

  std::string filecheck_pattern = R"(
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size:.*]] to i64
; CHECK: br label %concatenate.loop_header.concat.1
; CHECK: call void @llvm.memcpy
  )";

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), GetAotOptions(),
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuDynamicShapeTest,
       DynamicFastConcatenateWithAddCompilesWithoutFusion) {
  auto hlo_module =
      CreateDynamicFastConcatModule(TestName(), /*add_consumer=*/true);

  std::string filecheck_pattern = R"(
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size:.*]] to i64
; CHECK: br label %concatenate.loop_header.concat.1
; CHECK: call void @llvm.memcpy
  )";

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), GetAotOptions(),
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

TEST_F(CpuDynamicShapeTest,
       DynamicFastConcatenateWithThreeOperandsCompilesWithoutFusion) {
  auto hlo_module = CreateDynamicFastConcatModule(
      TestName(), /*add_consumer=*/true, /*three_operands=*/true);

  std::string filecheck_pattern = R"(
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size:.*]] to i64
; CHECK: br label %concatenate.loop_header.concat.1
; CHECK: call void @llvm.memcpy
  )";

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), GetAotOptions(),
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

}  // namespace
}  // namespace cpu
}  // namespace xla

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

#include "xla/service/llvm_ir/llvm_util.h"

#include <gtest/gtest.h>
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "xla/shape_expr.h"

namespace xla {
namespace llvm_ir {
namespace {

TEST(LlvmUtilTest, DynamicExpressionDivisionIsSigned) {
  llvm::LLVMContext context;
  llvm::Module module("llvm_util_test", context);
  llvm::IRBuilder<> builder(context);
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), /*isVarArg=*/false);
  llvm::Function* function = llvm::Function::Create(
      function_type, llvm::Function::ExternalLinkage, "test", module);
  llvm::BasicBlock* block =
      llvm::BasicBlock::Create(context, "entry", function);
  builder.SetInsertPoint(block);

  llvm::Type* i64_type = builder.getInt64Ty();
  llvm::Value* batch_dim_address = builder.CreateAlloca(i64_type);
  builder.CreateLoad(i64_type, batch_dim_address, "bdim_value");

  DExpr expression = (DExpr::Var(1) - 5) / 2;
  llvm::Value* value = EmitExpression(&builder, expression);

  auto* division = llvm::dyn_cast<llvm::BinaryOperator>(value);
  ASSERT_NE(division, nullptr);
  EXPECT_EQ(division->getOpcode(), llvm::Instruction::SDiv);
}

}  // namespace
}  // namespace llvm_ir
}  // namespace xla

/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/kernels/xla_ops_internal.h"

#include <cstdint>
#include <string>
#include <vector>

#include "xla/shape_expr.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace xla_ops_internal {
namespace {

TEST(XlaInputMappingTest, UsesCompilationInputMapping) {
  const std::vector<int> input_mapping = {1, 3};

  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 0, /*num_constant_args=*/1,
                                 /*constants_omitted=*/false),
            1);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 1, /*num_constant_args=*/1,
                                 /*constants_omitted=*/false),
            3);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 0, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            0);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 1, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            2);
  EXPECT_EQ(GetRuntimeInputIndex(input_mapping, 2, /*num_constant_args=*/1,
                                 /*constants_omitted=*/true),
            -1);
}

XlaArgument MakeDynamicArgument(int64_t observed_size,
                                xla::DExpr expr = xla::DExpr::Var(1)) {
  XlaArgument arg;
  arg.kind = XlaArgument::kParameter;
  arg.type = DT_FLOAT;
  TensorShape shape({observed_size, 16});
  shape.set_expression(0, std::move(expr));
  shape.set_expression(1, xla::DExpr::Const(16));
  arg.shape = shape;
  return arg;
}

xla::Shape MakeDynamicXlaInput(xla::DExpr leading_expr) {
  xla::Shape shape;
  shape.set_element_type(xla::F32);
  shape.add_dimensions(1024, /*is_dynamic=*/true,
                       std::move(leading_expr));
  shape.add_dimensions(16, /*is_dynamic=*/false, xla::DExpr::Const(16));
  return shape;
}

DynamicBatchResolutionResult ResolveForTest(
    absl::Span<const xla::Shape> xla_shapes,
    absl::Span<const TensorShape> runtime_shapes,
    absl::Span<const int> runtime_input_indices,
    absl::Span<const std::string> runtime_input_names) {
  return ResolveDynamicBatchSizeFromRuntimeShapes(
      xla_shapes, runtime_shapes.size(),
      [&](int i) { return runtime_input_indices[i]; },
      [&](int i) -> const TensorShape& { return runtime_shapes[i]; },
      [&](int i) -> absl::string_view { return runtime_input_names[i]; });
}

TEST(DynamicRuntimeSolveTest, UsesMappedRuntimeShapesToSolveExpressions) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable / 2),
  };
  // XLA parameters are deliberately ordered differently from runtime inputs.
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({120, 16}),
      TensorShape({240, 16}),
  };
  const std::vector<int> runtime_input_indices = {1, 0};
  const std::vector<std::string> runtime_input_names = {"half", "full"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 240);
}

TEST(DynamicRuntimeSolveTest,
     PrefersNonSingletonEvidenceAcrossEquivalentExpressions) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable + 1),
  };
  // The first occurrence may be a broadcast singleton. A + 1 = 241 gives
  // the stronger runtime evidence A = 240.
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({1, 16}),
      TensorShape({241, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"singleton", "full"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 240);
}

TEST(DynamicRuntimeSolveTest, KeepsConsistentNonSingletonEvidence) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(variable),
      MakeDynamicXlaInput(variable / 3),
  };
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({720, 16}),
      TensorShape({240, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"full", "third"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_TRUE(result.can_run);
  ASSERT_TRUE(result.has_batch_size);
  EXPECT_EQ(result.batch_size, 720);
}

TEST(DynamicRuntimeSolveTest, RejectsConflictingNonSingletonEvidence) {
  const std::vector<xla::Shape> xla_shapes = {
      MakeDynamicXlaInput(xla::DExpr::Var(1)),
      MakeDynamicXlaInput(xla::DExpr::Var(1)),
  };
  const std::vector<TensorShape> runtime_shapes = {
      TensorShape({240, 16}),
      TensorShape({720, 16}),
  };
  const std::vector<int> runtime_input_indices = {0, 1};
  const std::vector<std::string> runtime_input_names = {"first", "second"};

  DynamicBatchResolutionResult result = ResolveForTest(
      xla_shapes, runtime_shapes, runtime_input_indices, runtime_input_names);

  EXPECT_FALSE(result.can_run);
  EXPECT_FALSE(result.has_batch_size);
  EXPECT_NE(result.diagnostic.find("conflicting non-singleton candidates"),
            std::string::npos);
}

TEST(DynamicSolveFilterTest, IgnoresSingletonWhenNonSingletonEvidenceExists) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1),
      MakeDynamicArgument(240),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  ASSERT_TRUE(decision.can_run);
  ASSERT_EQ(decision.ignored_occurrences.size(), 1);
  EXPECT_EQ(decision.ignored_occurrences[0].arg_index, 0);
  EXPECT_EQ(decision.ignored_occurrences[0].observed_value, 1);

  StripIgnoredDynamicArgumentOccurrences(decision.ignored_occurrences, &args);
  const TensorShape& singleton_shape = std::get<TensorShape>(args[0].shape);
  const TensorShape& nonsingleton_shape = std::get<TensorShape>(args[1].shape);
  EXPECT_FALSE(singleton_shape.get_expression(0));
  EXPECT_EQ(singleton_shape.dim_size(0), 1);
  EXPECT_TRUE(nonsingleton_shape.get_expression(0));
}

TEST(DynamicSolveFilterTest,
     IgnoresSingletonAcrossExpressionsOfTheSameVariable) {
  const xla::DExpr variable = xla::DExpr::Var(1);
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1, variable),
      MakeDynamicArgument(241, variable + 1),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  ASSERT_TRUE(decision.can_run);
  ASSERT_EQ(decision.ignored_occurrences.size(), 1);
  EXPECT_EQ(decision.ignored_occurrences[0].arg_index, 0);
  EXPECT_EQ(decision.ignored_occurrences[0].observed_value, 1);
  EXPECT_EQ(decision.ignored_occurrences[0].solved_value, 1);
}

TEST(DynamicSolveFilterTest, RejectsConflictingNonSingletonEvidence) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(240),
      MakeDynamicArgument(720),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  EXPECT_FALSE(decision.can_run);
  EXPECT_TRUE(decision.ignored_occurrences.empty());
  EXPECT_NE(decision.diagnostic.find("conflicting non-singleton candidates"),
            std::string::npos);
}

TEST(DynamicSolveFilterTest, KeepsConsistentSingletonOnlyEvidence) {
  std::vector<XlaArgument> args = {
      MakeDynamicArgument(1),
      MakeDynamicArgument(1),
  };

  DynamicSolveFilterDecision decision =
      AnalyzeIgnoredDynamicArgumentOccurrences(args);

  EXPECT_TRUE(decision.can_run);
  EXPECT_TRUE(decision.ignored_occurrences.empty());
}

TEST(DynamicSolveFilterTest, StaticArgumentsKeepConcreteRuntimeShapes) {
  XlaArgument dynamic_arg = MakeDynamicArgument(37);
  dynamic_arg.constant_value_expressions.resize(1);
  dynamic_arg.constant_value_expressions[0].set_variable_id(1);

  std::vector<XlaArgument> args = {dynamic_arg};
  std::vector<XlaArgument> static_args =
      BuildStaticCompilationArguments(args);

  ASSERT_EQ(static_args.size(), 1);
  const TensorShape& static_shape =
      std::get<TensorShape>(static_args[0].shape);
  EXPECT_EQ(static_shape.dim_size(0), 37);
  EXPECT_EQ(static_shape.dim_size(1), 16);
  EXPECT_TRUE(static_shape.get_expressions().empty());
  EXPECT_TRUE(static_args[0].constant_value_expressions.empty());
}

TEST(DynamicSolveFilterTest, StaticArgumentsClearXlaShapeDynamism) {
  xla::Shape dynamic_shape;
  dynamic_shape.set_element_type(xla::F32);
  dynamic_shape.add_dimensions(37, /*is_dynamic=*/true,
                               xla::DExpr::Var(1));

  XlaArgument dynamic_arg;
  dynamic_arg.kind = XlaArgument::kParameter;
  dynamic_arg.type = DT_FLOAT;
  dynamic_arg.shape = dynamic_shape;
  std::vector<XlaArgument> args = {dynamic_arg};

  std::vector<XlaArgument> static_args =
      BuildStaticCompilationArguments(args);

  const xla::Shape& static_shape = std::get<xla::Shape>(static_args[0].shape);
  EXPECT_EQ(static_shape.dimensions(0), 37);
  EXPECT_FALSE(static_shape.is_dynamic_dimension(0));
  ASSERT_EQ(static_shape.expressions().size(), 1);
  EXPECT_TRUE(static_shape.expressions(0)->is_constant());
  EXPECT_EQ(static_shape.expressions(0)->get_val(), 37);
}

}  // namespace
}  // namespace xla_ops_internal
}  // namespace tensorflow

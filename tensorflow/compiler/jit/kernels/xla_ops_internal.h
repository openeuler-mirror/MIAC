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

#ifndef TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_INTERNAL_H_
#define TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_INTERNAL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_argument.h"

namespace tensorflow {
namespace xla_ops_internal {

struct IgnoredDynamicArgumentOccurrence {
  enum class Source {
    kShapeDimension,
    kConstantValueElement,
  };

  Source source;
  int arg_index;
  int dim_or_index;
  int64_t observed_value;
  int64_t solved_value;
  std::string expr;
};

struct DynamicSolveFilterDecision {
  bool can_run = true;
  std::string diagnostic;
  std::vector<IgnoredDynamicArgumentOccurrence> ignored_occurrences;
};

struct DynamicBatchResolutionResult {
  bool can_run = true;
  bool has_batch_size = false;
  int64_t batch_size = 0;
  bool batch_size_resource_not_found = false;
  std::string diagnostic;
};

int GetRuntimeInputIndex(absl::Span<const int> input_mapping,
                         int xla_input_index, int num_constant_args,
                         bool constants_omitted);

DynamicBatchResolutionResult ResolveDynamicBatchSizeFromRuntimeShapes(
    absl::Span<const xla::Shape> xla_input_shapes,
    int runtime_input_count,
    absl::FunctionRef<int(int)> get_runtime_input_index,
    absl::FunctionRef<const TensorShape&(int)> get_runtime_input_shape,
    absl::FunctionRef<absl::string_view(int)> get_runtime_input_name,
    bool log_solves = false, absl::string_view cluster_name = {});

DynamicSolveFilterDecision AnalyzeIgnoredDynamicArgumentOccurrences(
    absl::Span<const XlaArgument> args);

std::vector<XlaArgument> BuildStaticCompilationArguments(
    absl::Span<const XlaArgument> args);

void StripIgnoredDynamicArgumentOccurrences(
    const std::vector<IgnoredDynamicArgumentOccurrence>& ignored_occurrences,
    std::vector<XlaArgument>* args);

}  // namespace xla_ops_internal
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_KERNELS_XLA_OPS_INTERNAL_H_

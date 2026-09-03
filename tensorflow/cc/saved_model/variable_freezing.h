/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_
#define TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace internal {

// Freezes allowlisted read-only variables into Const nodes.
// `max_tensor_bytes` controls the maximum tensor size in bytes to freeze;
// A negative value or zero uses the internal default (1 MiB).
absl::Status FreezeAllowlistedVariableReads(const std::string& export_dir,
                                            MetaGraphDef* meta_graph_def,
                                            int64_t max_tensor_bytes = -1);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_

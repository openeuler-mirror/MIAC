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
// a value of 0 means no limit. A negative value uses the internal default
// (1 MiB).
absl::Status FreezeAllowlistedVariableReads(const std::string& export_dir,
                                            MetaGraphDef* meta_graph_def,
                                            int64_t max_tensor_bytes = -1);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_

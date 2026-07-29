#ifndef TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_
#define TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_

#include <string>

#include "absl/status/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace internal {

absl::Status FreezeAllowlistedVariableReads(const std::string& export_dir,
                                            MetaGraphDef* meta_graph_def);

}  // namespace internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_VARIABLE_FREEZING_H_

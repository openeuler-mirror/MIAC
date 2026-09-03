/* Copyright 2026 The TensorFlow Authors.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_BATCH_SIZE_RESOURCE_H_
#define TENSORFLOW_CORE_FRAMEWORK_BATCH_SIZE_RESOURCE_H_

#include <cstddef>

#include "tensorflow/core/framework/resource_mgr.h"

namespace tensorflow {
const string BatchSizeResourceName = "BatchSizeResource_";
class BatchSizeResource : public ResourceBase {
 public:
  ~BatchSizeResource() override {
    VLOG(1) << "BatchSizeResource destroyed with batch size: " << batch_size_;
  }
  string DebugString() const override { return BatchSizeResourceName; }
  void SetBatchSize(size_t s) { batch_size_ = s; }
  size_t GetBatchSize() { return batch_size_; }

 private:
  size_t batch_size_ = 0;
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_BATCH_SIZE_RESOURCE_H_

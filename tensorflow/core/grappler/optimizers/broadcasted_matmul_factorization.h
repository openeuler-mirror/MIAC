/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_BROADCASTED_MATMUL_FACTORIZATION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_BROADCASTED_MATMUL_FACTORIZATION_H_

#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"

namespace tensorflow {
namespace grappler {

// Projects an input before it is repeated across rows in a frozen graph:
//
//   MatMul(Concat(X, Repeat(Q, L)), W)
//     -> MatMul(X, W_x) + Broadcast(MatMul(Q, W_q), L).
class BroadcastedMatMulFactorizationOptimizer : public GraphOptimizer {
 public:
  string name() const override { return "broadcasted_matmul_factorization"; }
  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_BROADCASTED_MATMUL_FACTORIZATION_H_

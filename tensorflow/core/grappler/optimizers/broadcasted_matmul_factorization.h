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

#include "tensorflow/core/grappler/optimizers/broadcasted_matmul_factorization.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef* FindNode(GraphDef* graph, const string& name) {
  for (NodeDef& node : *graph->mutable_node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

NodeDef* AddPlaceholder(GraphDef* graph, const string& name,
                        const TensorShape& shape) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Placeholder");
  (*node->mutable_attr())["dtype"].set_type(DT_FLOAT);
  *(*node->mutable_attr())["shape"].mutable_shape() = shape.AsProto();
  return node;
}

NodeDef* AddIntConst(GraphDef* graph, const string& name,
                     const std::vector<int32>& values, bool scalar = false) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Const");
  (*node->mutable_attr())["dtype"].set_type(DT_INT32);
  Tensor tensor(DT_INT32,
                scalar ? TensorShape({})
                       : TensorShape({static_cast<int64_t>(values.size())}));
  if (scalar) {
    tensor.scalar<int32>()() = values[0];
  } else {
    auto flat = tensor.flat<int32>();
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
      flat(i) = values[i];
    }
  }
  tensor.AsProtoTensorContent(
      (*node->mutable_attr())["value"].mutable_tensor());
  return node;
}

NodeDef* AddFloatConst(GraphDef* graph, const string& name,
                       const TensorShape& shape, int seed) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op("Const");
  (*node->mutable_attr())["dtype"].set_type(DT_FLOAT);
  Tensor tensor(DT_FLOAT, shape);
  auto flat = tensor.flat<float>();
  for (int64_t i = 0; i < tensor.NumElements(); ++i) {
    flat(i) = static_cast<float>((i + seed) % 17 - 8) * 0.01f;
  }
  tensor.AsProtoTensorContent(
      (*node->mutable_attr())["value"].mutable_tensor());
  return node;
}

GrapplerItem MakeGraph(int repeats, bool repeated_first = false,
                       bool frozen_weight_read = false) {
  constexpr int kBatch = 2;
  constexpr int kBehaviorWidth = 2;
  constexpr int kQueryWidth = 4;
  constexpr int kOutputWidth = 5;

  GrapplerItem item;
  GraphDef* graph = &item.graph;
  AddPlaceholder(graph, "inputs/history",
                 TensorShape({kBatch * repeats, kBehaviorWidth}));
  AddPlaceholder(graph, "inputs/candidate", TensorShape({kBatch, kQueryWidth}));

  AddIntConst(graph, "repeat/counts", {1, repeats});
  NodeDef* tile = graph->add_node();
  tile->set_name("repeat/candidate");
  tile->set_op("Tile");
  tile->add_input("inputs/candidate");
  tile->add_input("repeat/counts");
  (*tile->mutable_attr())["T"].set_type(DT_FLOAT);
  (*tile->mutable_attr())["Tmultiples"].set_type(DT_INT32);

  AddIntConst(graph, "repeat/shape", {-1, kQueryWidth});
  NodeDef* reshape = graph->add_node();
  reshape->set_name("repeat/rows");
  reshape->set_op("Reshape");
  reshape->add_input("repeat/candidate");
  reshape->add_input("repeat/shape");
  (*reshape->mutable_attr())["T"].set_type(DT_FLOAT);
  (*reshape->mutable_attr())["Tshape"].set_type(DT_INT32);

  AddIntConst(graph, "features/axis", {1}, true);
  NodeDef* concat = graph->add_node();
  concat->set_name("features/join");
  concat->set_op("ConcatV2");
  concat->add_input(repeated_first ? "repeat/rows" : "inputs/history");
  concat->add_input(repeated_first ? "inputs/history" : "repeat/rows");
  concat->add_input("features/axis");
  (*concat->mutable_attr())["N"].set_i(2);
  (*concat->mutable_attr())["T"].set_type(DT_FLOAT);
  (*concat->mutable_attr())["Tidx"].set_type(DT_INT32);

  // This is the shape of ANNC's post-BatchNorm fused weight. Deliberately do
  // not add _output_shapes: the optimizer must read the Const tensor itself.
  AddFloatConst(graph, "dense/projection/fused_weight",
                TensorShape({kBehaviorWidth + kQueryWidth, kOutputWidth}), 1);
  string weight_input = "dense/projection/fused_weight";
  if (frozen_weight_read) {
    NodeDef* read = graph->add_node();
    read->set_name("dense/projection/frozen_weight_read");
    read->set_op("Identity");
    read->add_input(weight_input);
    (*read->mutable_attr())["T"].set_type(DT_FLOAT);
    // FreezeSavedModel does not copy _output_shapes to this Identity.
    weight_input = read->name();
  }

  NodeDef* matmul = graph->add_node();
  matmul->set_name("dense/projection");
  matmul->set_op("MatMul");
  matmul->add_input("features/join");
  matmul->add_input(weight_input);
  (*matmul->mutable_attr())["T"].set_type(DT_FLOAT);
  (*matmul->mutable_attr())["transpose_a"].set_b(false);
  (*matmul->mutable_attr())["transpose_b"].set_b(false);

  AddFloatConst(graph, "dense/fused_bias", TensorShape({kOutputWidth}), 2);
  NodeDef* output = graph->add_node();
  output->set_name("result");
  output->set_op("BiasAdd");
  output->add_input("dense/projection");
  output->add_input("dense/fused_bias");
  (*output->mutable_attr())["T"].set_type(DT_FLOAT);
  (*output->mutable_attr())["data_format"].set_s("NHWC");

  item.fetch = {"result"};
  return item;
}

Tensor Input(const TensorShape& shape, int seed) {
  Tensor tensor(DT_FLOAT, shape);
  auto flat = tensor.flat<float>();
  for (int64_t i = 0; i < tensor.NumElements(); ++i) {
    flat(i) = static_cast<float>((i + seed) % 13 - 6) * 0.03f;
  }
  return tensor;
}

class BroadcastedMatMulFactorizationTest : public GrapplerTest {};

TEST_F(BroadcastedMatMulFactorizationTest,
       RewritesAnncFusedWeightWithoutOutputShapes) {
  GrapplerItem item = MakeGraph(/*repeats=*/3);
  const std::vector<std::pair<string, Tensor>> feeds = {
      {"inputs/history", Input(TensorShape({6, 2}), 1)},
      {"inputs/candidate", Input(TensorShape({2, 4}), 2)},
  };
  const std::vector<Tensor> expected =
      EvaluateNodes(item.graph, item.fetch, feeds);

  BroadcastedMatMulFactorizationOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  NodeDef* result = FindNode(&optimized, "result");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(NodeName(result->input(0)),
            "dense/projection/broadcast_factorization/output");
  EXPECT_EQ(NodeName(result->input(1)), "dense/fused_bias");

  const std::vector<Tensor> actual =
      EvaluateNodes(optimized, item.fetch, feeds);
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-5);

  GrapplerItem second_item = item;
  second_item.graph = optimized;
  GraphDef second;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, second_item, &second));
  EXPECT_EQ(second.node_size(), optimized.node_size());
}

TEST_F(BroadcastedMatMulFactorizationTest, SupportsRepeatedInputFirst) {
  GrapplerItem item = MakeGraph(/*repeats=*/3, /*repeated_first=*/true);
  const std::vector<std::pair<string, Tensor>> feeds = {
      {"inputs/history", Input(TensorShape({6, 2}), 1)},
      {"inputs/candidate", Input(TensorShape({2, 4}), 2)},
  };
  const std::vector<Tensor> expected =
      EvaluateNodes(item.graph, item.fetch, feeds);

  BroadcastedMatMulFactorizationOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));
  const std::vector<Tensor> actual =
      EvaluateNodes(optimized, item.fetch, feeds);
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-5);
}

TEST_F(BroadcastedMatMulFactorizationTest,
       RewritesFrozenIdentityWrappedWeight) {
  GrapplerItem item = MakeGraph(/*repeats=*/3, /*repeated_first=*/false,
                                /*frozen_weight_read=*/true);
  const std::vector<std::pair<string, Tensor>> feeds = {
      {"inputs/history", Input(TensorShape({6, 2}), 1)},
      {"inputs/candidate", Input(TensorShape({2, 4}), 2)},
  };
  const std::vector<Tensor> expected =
      EvaluateNodes(item.graph, item.fetch, feeds);

  BroadcastedMatMulFactorizationOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  NodeDef* split = FindNode(
      &optimized, "dense/projection/broadcast_factorization/split_weights");
  ASSERT_NE(split, nullptr);
  EXPECT_EQ(NodeName(split->input(0)), "dense/projection/frozen_weight_read");

  NodeDef* result = FindNode(&optimized, "result");
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(NodeName(result->input(0)),
            "dense/projection/broadcast_factorization/output");

  const std::vector<Tensor> actual =
      EvaluateNodes(optimized, item.fetch, feeds);
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-5);
}

TEST_F(BroadcastedMatMulFactorizationTest, SkipsUnfrozenIdentityWrappedWeight) {
  GrapplerItem item = MakeGraph(/*repeats=*/3, /*repeated_first=*/false,
                                /*frozen_weight_read=*/true);

  NodeDef* variable = item.graph.add_node();
  variable->set_name("dense/projection/unfrozen_weight");
  variable->set_op("VariableV2");
  (*variable->mutable_attr())["dtype"].set_type(DT_FLOAT);
  *(*variable->mutable_attr())["shape"].mutable_shape() =
      TensorShape({6, 5}).AsProto();
  (*variable->mutable_attr())["container"].set_s("");
  (*variable->mutable_attr())["shared_name"].set_s("");

  NodeDef* read = FindNode(&item.graph, "dense/projection/frozen_weight_read");
  ASSERT_NE(read, nullptr);
  read->set_input(0, variable->name());

  BroadcastedMatMulFactorizationOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));
  EXPECT_EQ(FindNode(&optimized,
                     "dense/projection/broadcast_factorization/split_weights"),
            nullptr);
}

TEST_F(BroadcastedMatMulFactorizationTest, SkipsUnprofitableRepeat) {
  GrapplerItem item = MakeGraph(/*repeats=*/1);
  BroadcastedMatMulFactorizationOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(
      FindNode(&output, "dense/projection/broadcast_factorization/output"),
      nullptr);
}

TEST_F(BroadcastedMatMulFactorizationTest, DoesNotRunAutomaticallyByDefault) {
  GrapplerItem item = MakeGraph(/*repeats=*/3);
  ConfigProto config;
  RewriterConfig* rewrite =
      config.mutable_graph_options()->mutable_rewrite_options();
  rewrite->set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewrite->set_min_graph_nodes(-1);

  GraphDef optimized;
  TF_ASSERT_OK(RunMetaOptimizer(std::move(item), config,
                                /*cpu_device=*/nullptr, /*cluster=*/nullptr,
                                &optimized));
  EXPECT_EQ(
      FindNode(&optimized, "dense/projection/broadcast_factorization/output"),
      nullptr);
}

TEST_F(BroadcastedMatMulFactorizationTest, RunsAutomaticallyWhenEnabled) {
  GrapplerItem item = MakeGraph(/*repeats=*/3);
  ConfigProto config;
  RewriterConfig* rewrite =
      config.mutable_graph_options()->mutable_rewrite_options();
  rewrite->set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewrite->set_min_graph_nodes(-1);
  rewrite->set_broadcasted_matmul_factorization(RewriterConfig::ON);

  GraphDef optimized;
  TF_ASSERT_OK(RunMetaOptimizer(std::move(item), config,
                                /*cpu_device=*/nullptr, /*cluster=*/nullptr,
                                &optimized));
  EXPECT_NE(
      FindNode(&optimized, "dense/projection/broadcast_factorization/output"),
      nullptr);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

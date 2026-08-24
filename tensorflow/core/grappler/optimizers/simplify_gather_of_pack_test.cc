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

#include "tensorflow/core/grappler/optimizers/simplify_gather_of_pack.h"

#include <utility>
#include <vector>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/meta_optimizer.h"
#include "tensorflow/core/grappler/optimizers/model_pruner.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

const NodeDef* FindNode(const GraphDef& graph, const string& name) {
  for (const NodeDef& node : graph.node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

NodeDef* FindMutableNode(GraphDef* graph, const string& name) {
  for (NodeDef& node : *graph->mutable_node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

void SetOutputRank(NodeDef* node, int rank) {
  AttrValue output_shapes;
  TensorShapeProto* shape = output_shapes.mutable_list()->add_shape();
  for (int i = 0; i < rank; ++i) {
    shape->add_dim()->set_size(-1);
  }
  (*node->mutable_attr())["_output_shapes"] = std::move(output_shapes);
}

GrapplerItem MakeGraph(const std::vector<int32>& indices = {0, 1},
                       int gather_axis = 1, bool add_pack_consumer = false) {
  Scope scope = Scope::NewRootScope();
  Output a = ops::Placeholder(scope.WithOpName("a"), DT_FLOAT,
                              ops::Placeholder::Shape({2, 128}));
  Output b = ops::Placeholder(scope.WithOpName("b"), DT_FLOAT,
                              ops::Placeholder::Shape({2, 128}));
  Output c = ops::Placeholder(scope.WithOpName("c"), DT_FLOAT,
                              ops::Placeholder::Shape({2, 128}));
  Output packed =
      ops::Stack(scope.WithOpName("packed"), {a, b, c}, ops::Stack::Axis(1));
  Output gathered = ops::GatherV2(
      scope.WithOpName("gathered"), packed,
      ops::Const(scope.WithOpName("indices"), test::AsTensor<int32>(indices)),
      ops::Const(scope.WithOpName("axis"), gather_axis));
  ops::Identity(scope.WithOpName("output"), gathered);
  if (add_pack_consumer) {
    ops::Identity(scope.WithOpName("packed_output"), packed);
  }

  GrapplerItem item;
  item.fetch = {"output"};
  if (add_pack_consumer) {
    item.fetch.push_back("packed_output");
  }
  TF_CHECK_OK(scope.ToGraphDef(&item.graph));
  return item;
}

void OptimizeAndPrune(SimplifyGatherOfPackOptimizer* optimizer,
                      GrapplerItem* item, GraphDef* output) {
  TF_ASSERT_OK(optimizer->Optimize(nullptr, *item, output));
  item->graph.Swap(output);
  output->Clear();
  TF_ASSERT_OK(ModelPruner().Optimize(nullptr, *item, output));
}

class SimplifyGatherOfPackOptimizerTest : public GrapplerTest {};

TEST_F(SimplifyGatherOfPackOptimizerTest, RewritesCvrPattern) {
  GrapplerItem item = MakeGraph();
  auto a = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto b = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto c = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  const auto expected =
      EvaluateNodes(item.graph, item.fetch, {{"a", a}, {"b", b}, {"c", c}});
  ASSERT_EQ(expected.size(), 1);

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  OptimizeAndPrune(&optimizer, &item, &optimized);

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->op(), "Pack");
  ASSERT_EQ(rewritten->input_size(), 2);
  EXPECT_EQ(rewritten->input(0), "a");
  EXPECT_EQ(rewritten->input(1), "b");
  EXPECT_EQ(FindNode(optimized, "packed"), nullptr);
  EXPECT_EQ(FindNode(optimized, "c"), nullptr);

  const auto actual =
      EvaluateNodes(optimized, item.fetch, {{"a", a}, {"b", b}});
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-6);
}

TEST_F(SimplifyGatherOfPackOptimizerTest, SkipsMismatchedAxis) {
  GrapplerItem item = MakeGraph(/*indices=*/{0, 1}, /*gather_axis=*/0);
  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  const NodeDef* gather = FindNode(optimized, "gathered");
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->op(), "GatherV2");
}

TEST_F(SimplifyGatherOfPackOptimizerTest,
       RewritesEquivalentNegativeGatherAxis) {
  GrapplerItem item = MakeGraph(/*indices=*/{0, 1}, /*gather_axis=*/-2);
  NodeDef* pack = FindMutableNode(&item.graph, "packed");
  ASSERT_NE(pack, nullptr);
  SetOutputRank(pack, /*rank=*/3);

  auto a = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto b = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto c = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  const auto expected =
      EvaluateNodes(item.graph, item.fetch, {{"a", a}, {"b", b}, {"c", c}});
  ASSERT_EQ(expected.size(), 1);

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  OptimizeAndPrune(&optimizer, &item, &optimized);

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->op(), "Pack");
  ASSERT_EQ(rewritten->input_size(), 2);
  EXPECT_EQ(rewritten->input(0), "a");
  EXPECT_EQ(rewritten->input(1), "b");
  EXPECT_EQ(rewritten->attr().at("axis").i(), 1);

  const auto actual =
      EvaluateNodes(optimized, item.fetch, {{"a", a}, {"b", b}});
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-6);
}

TEST_F(SimplifyGatherOfPackOptimizerTest, SkipsFedPack) {
  GrapplerItem item = MakeGraph();
  item.feed = {
      {"packed", GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 3, 128}))}};

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  const NodeDef* gather = FindNode(optimized, "gathered");
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->op(), "GatherV2");
}

TEST_F(SimplifyGatherOfPackOptimizerTest, SkipsNonzeroBatchDims) {
  GrapplerItem item = MakeGraph();
  NodeDef* gather = FindMutableNode(&item.graph, "gathered");
  ASSERT_NE(gather, nullptr);
  (*gather->mutable_attr())["batch_dims"].set_i(1);
  gather->mutable_attr()->erase("_output_shapes");

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  const NodeDef* optimized_gather = FindNode(optimized, "gathered");
  ASSERT_NE(optimized_gather, nullptr);
  EXPECT_EQ(optimized_gather->op(), "GatherV2");
  EXPECT_EQ(optimized_gather->attr().at("batch_dims").i(), 1);
}

TEST_F(SimplifyGatherOfPackOptimizerTest, SkipsOutOfRangeIndices) {
  GrapplerItem item = MakeGraph(/*indices=*/{0, 3});

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &optimized));

  const NodeDef* gather = FindNode(optimized, "gathered");
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->op(), "GatherV2");
  ASSERT_EQ(gather->input_size(), 3);
  EXPECT_EQ(gather->input(0), "packed");
}

TEST_F(SimplifyGatherOfPackOptimizerTest, PreservesPackWithAnotherConsumer) {
  GrapplerItem item = MakeGraph(/*indices=*/{0, 1}, /*gather_axis=*/1,
                                /*add_pack_consumer=*/true);
  auto a = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto b = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto c = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  const auto expected =
      EvaluateNodes(item.graph, item.fetch, {{"a", a}, {"b", b}, {"c", c}});
  ASSERT_EQ(expected.size(), 2);

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  OptimizeAndPrune(&optimizer, &item, &optimized);

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->op(), "Pack");
  ASSERT_EQ(rewritten->input_size(), 2);
  EXPECT_EQ(rewritten->input(0), "a");
  EXPECT_EQ(rewritten->input(1), "b");

  const NodeDef* original_pack = FindNode(optimized, "packed");
  ASSERT_NE(original_pack, nullptr);
  EXPECT_EQ(original_pack->op(), "Pack");
  ASSERT_EQ(original_pack->input_size(), 3);
  EXPECT_EQ(original_pack->input(2), "c");
  const NodeDef* other_consumer = FindNode(optimized, "packed_output");
  ASSERT_NE(other_consumer, nullptr);
  ASSERT_EQ(other_consumer->input_size(), 1);
  EXPECT_EQ(other_consumer->input(0), "packed");

  const auto actual =
      EvaluateNodes(optimized, item.fetch, {{"a", a}, {"b", b}, {"c", c}});
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-6);
  test::ExpectTensorNear<float>(actual[1], expected[1], 1e-6);
}

TEST_F(SimplifyGatherOfPackOptimizerTest, SupportsDuplicateIndices) {
  GrapplerItem item = MakeGraph(/*indices=*/{0, 0});
  auto a = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto b = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  auto c = GenerateRandomTensor<DT_FLOAT>(TensorShape({2, 128}));
  const auto expected =
      EvaluateNodes(item.graph, item.fetch, {{"a", a}, {"b", b}, {"c", c}});
  ASSERT_EQ(expected.size(), 1);

  SimplifyGatherOfPackOptimizer optimizer;
  GraphDef optimized;
  OptimizeAndPrune(&optimizer, &item, &optimized);

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  ASSERT_EQ(rewritten->input_size(), 2);
  EXPECT_EQ(rewritten->input(0), "a");
  EXPECT_EQ(rewritten->input(1), "a");

  const auto actual = EvaluateNodes(optimized, item.fetch, {{"a", a}});
  ASSERT_EQ(actual.size(), expected.size());
  test::ExpectTensorNear<float>(actual[0], expected[0], 1e-6);
}

TEST_F(SimplifyGatherOfPackOptimizerTest,
       RunsAutomaticallyWhenEnabledIndependentlyOfArithmetic) {
  GrapplerItem item = MakeGraph();
  ConfigProto config;
  RewriterConfig* rewrite =
      config.mutable_graph_options()->mutable_rewrite_options();
  rewrite->set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewrite->set_min_graph_nodes(-1);
  rewrite->set_arithmetic_optimization(RewriterConfig::OFF);
  rewrite->set_simplify_gather_of_pack(RewriterConfig::ON);

  GraphDef optimized;
  TF_ASSERT_OK(RunMetaOptimizer(std::move(item), config,
                                /*cpu_device=*/nullptr, /*cluster=*/nullptr,
                                &optimized));

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->op(), "Pack");
}

TEST_F(SimplifyGatherOfPackOptimizerTest, DoesNotRunWhenDisabled) {
  GrapplerItem item = MakeGraph();
  ConfigProto config;
  RewriterConfig* rewrite =
      config.mutable_graph_options()->mutable_rewrite_options();
  rewrite->set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewrite->set_min_graph_nodes(-1);
  rewrite->set_simplify_gather_of_pack(RewriterConfig::OFF);

  GraphDef optimized;
  TF_ASSERT_OK(RunMetaOptimizer(std::move(item), config,
                                /*cpu_device=*/nullptr, /*cluster=*/nullptr,
                                &optimized));

  const NodeDef* gather = FindNode(optimized, "gathered");
  ASSERT_NE(gather, nullptr);
  EXPECT_EQ(gather->op(), "GatherV2");
}

TEST_F(SimplifyGatherOfPackOptimizerTest, RunsWhenSelectedByName) {
  GrapplerItem item = MakeGraph();
  ConfigProto config;
  RewriterConfig* rewrite =
      config.mutable_graph_options()->mutable_rewrite_options();
  rewrite->set_meta_optimizer_iterations(RewriterConfig::ONE);
  rewrite->set_min_graph_nodes(-1);
  rewrite->set_simplify_gather_of_pack(RewriterConfig::OFF);
  rewrite->add_optimizers("simplify_gather_of_pack");

  GraphDef optimized;
  TF_ASSERT_OK(RunMetaOptimizer(std::move(item), config,
                                /*cpu_device=*/nullptr, /*cluster=*/nullptr,
                                &optimized));

  const NodeDef* rewritten = FindNode(optimized, "gathered");
  ASSERT_NE(rewritten, nullptr);
  EXPECT_EQ(rewritten->op(), "Pack");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow

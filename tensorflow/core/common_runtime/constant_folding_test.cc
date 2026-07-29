/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/constant_folding.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/cc/ops/array_ops_internal.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/nn_ops.h"
#include "tensorflow/cc/ops/sendrecv_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape_expr.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/null_file_system.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {

TensorShapeProto MakeDynamicShapeProto977x16() {
  TensorShapeProto proto;
  proto.add_dim()->set_size(977);
  proto.add_dim()->set_size(16);
  proto.add_expressions()->set_variable_id(0);
  proto.add_expressions()->set_constant_value(16);
  return proto;
}

class ConstantFoldingTest : public ::testing::Test {
 protected:
  template <typename T>
  void ExpectNodeClose(const Node* n, gtl::ArraySlice<T> values,
                       TensorShape shape) {
    EXPECT_TRUE(n->IsConstant());
    const TensorProto* tensor_proto;
    TF_EXPECT_OK(GetNodeAttr(n->attrs(), "value", &tensor_proto));
    DataType dtype;
    TF_EXPECT_OK(GetNodeAttr(n->attrs(), "dtype", &dtype));
    Tensor t(dtype);
    EXPECT_TRUE(t.FromProto(*tensor_proto));
    test::ExpectClose(t, test::AsTensor(values, shape));
  }

  template <typename T>
  void ExpectNodeEqual(const Node* n, gtl::ArraySlice<T> values,
                       TensorShape shape) {
    EXPECT_TRUE(n->IsConstant());
    const TensorProto* tensor_proto;
    TF_EXPECT_OK(GetNodeAttr(n->attrs(), "value", &tensor_proto));
    DataType dtype;
    TF_EXPECT_OK(GetNodeAttr(n->attrs(), "dtype", &dtype));
    Tensor t(dtype);
    EXPECT_TRUE(t.FromProto(*tensor_proto));
    test::ExpectTensorEqual<T>(t, test::AsTensor(values, shape));
  }

  // Constructs the following graph.
  /*
        s1  s2
        |    |
        m1   m2
        / \ / \
       a   b   c
  */
  void BuildSimpleGraph(Scope* scope) {
    Scope& s = *scope;
    auto a = ops::Const<float>(s, {1.0, 0.0, 0.0, 1.0}, {2, 2});
    auto b = ops::Const<float>(s, {1.0, 2.0, 3.0, 4.0}, {2, 2});
    auto c = ops::Const<float>(s, {0.0, 1.0, 1.0, 0.0}, {2, 2});
    auto m1 = ops::MatMul(s, a, b);
    auto s1 = ops::_Send(s.WithOpName("s1"), m1, "m1", "sender", 0, "receiver");
    auto m2 = ops::MatMul(s.WithOpName("m2"), b, c);
    auto s2 = ops::_Send(s.WithOpName("s2"), m2, "m2", "sender", 0, "receiver");
  }
};

class FakeDevice : public Device {
 private:
  explicit FakeDevice(const DeviceAttributes& device_attributes)
      : Device(nullptr, device_attributes) {}

 public:
  absl::Status Sync() override {
    return errors::Unimplemented("FakeDevice::Sync()");
  }

  Allocator* GetAllocator(AllocatorAttributes attr) override { return nullptr; }

  static std::unique_ptr<Device> Make(const string& name, const string& type) {
    DeviceAttributes device_attributes;
    device_attributes.set_name(name);
    device_attributes.set_device_type(DeviceType(type).type());
    return std::unique_ptr<Device>(new FakeDevice(device_attributes));
  }
};

TEST_F(ConstantFoldingTest, Basic) {
  Scope s = Scope::NewRootScope();
  BuildSimpleGraph(&s);
  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(&g));

  bool was_mutated;
  TF_ASSERT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* s1 = index.at("s1");
  Node* s2 = index.at("s2");
  // Nodes s1 and s2 now should now have a constant input
  EXPECT_EQ(1, s1->num_inputs());
  ExpectNodeClose<float>(*(s1->in_nodes().begin()), {1.0, 2.0, 3.0, 4.0},
                         {2, 2});
  EXPECT_EQ(1, s2->num_inputs());
  ExpectNodeClose<float>(*(s2->in_nodes().begin()), {2.0, 1.0, 4.0, 3.0},
                         {2, 2});
}

// Tests that different node creation ordering creates same graph after constant
// folding.
TEST_F(ConstantFoldingTest, DeterministicFolding) {
  auto build_graph_and_constant_folding = [](Graph& g,
                                             bool swap) -> absl::Status {
    Scope s = Scope::NewRootScope();
    auto a = ops::Const<float>(s, {1.0}, {});
    auto b = ops::Const<float>(s, {2.0}, {});

    if (swap) {
      auto add1 = ops::Add(s.WithOpName("add1"), a, b);
      auto add2 = ops::Add(s.WithOpName("add2"), a, b);
      auto s1 =
          ops::_Send(s.WithOpName("s1"), add1, "add1", "sender", 0, "receiver");
      auto s2 =
          ops::_Send(s.WithOpName("s2"), add2, "add2", "sender", 0, "receiver");
    } else {
      // Swap the order of node creation.
      auto add2 = ops::Add(s.WithOpName("add2"), a, b);
      auto add1 = ops::Add(s.WithOpName("add1"), a, b);
      auto s1 =
          ops::_Send(s.WithOpName("s1"), add1, "add1", "sender", 0, "receiver");
      auto s2 =
          ops::_Send(s.WithOpName("s2"), add2, "add2", "sender", 0, "receiver");
    }

    TF_CHECK_OK(s.ToGraph(&g));
    bool was_mutated;
    int64_t unique_id = 0;
    auto generate_new_name = [&unique_id](Graph* graph, string old_name) {
      return strings::StrCat(graph->NewName(old_name), "__cf__", unique_id++);
    };
    ConstantFoldingOptions opt{};
    opt.generate_new_name = generate_new_name;
    TF_CHECK_OK(
        ConstantFold(opt, nullptr, Env::Default(), nullptr, &g, &was_mutated));
    return absl::OkStatus();
  };

  Graph g1(OpRegistry::Global());
  TF_ASSERT_OK(build_graph_and_constant_folding(g1, false));
  Graph g2(OpRegistry::Global());
  TF_ASSERT_OK(build_graph_and_constant_folding(g2, true));
  EXPECT_EQ(g1.num_nodes(), g2.num_nodes());
  auto index = g2.BuildNodeNameIndex();

  // All the nodes in g1 are expected to be present in g2.
  for (int64_t i = 0; i < g1.num_nodes(); ++i) {
    Node* n1 = g1.FindNodeId(i);
    EXPECT_GT(index.count(n1->name()), 0);
  }
}

TEST_F(ConstantFoldingTest, ConsiderFunction) {
  Scope s = Scope::NewRootScope();
  BuildSimpleGraph(&s);
  Graph g(OpRegistry::Global());
  TF_ASSERT_OK(s.ToGraph(&g));

  ConstantFoldingOptions opts;
  // Do not allow constant folding of m2
  opts.consider = [](const Node* n) { return "m2" != n->name(); };
  bool was_mutated;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* s1 = index.at("s1");
  Node* s2 = index.at("s2");
  Node* m2 = index.at("m2");

  // Node s1 now should now have a constant input
  EXPECT_EQ(1, s1->num_inputs());
  ExpectNodeClose<float>(*(s1->in_nodes().begin()), {1.0, 2.0, 3.0, 4.0},
                         {2, 2});
  // s2's input should still be m2
  EXPECT_EQ(1, s2->num_inputs());
  EXPECT_EQ(*(s2->in_nodes().begin()), m2);
}

TEST_F(ConstantFoldingTest, TestNoReplaceAnotherConstant) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    BuildSimpleGraph(&s);
    auto d = ops::Const<float>(s.WithOpName("d"), {1.0, 0.0, 0.0, 1.0}, {2, 2});
    auto s3 = ops::_Send(s.WithOpName("s3"), d, "d", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  bool was_mutated;
  TF_ASSERT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* d = index.at("d");
  Node* s3 = index.at("s3");

  // Nodes s3 should still have d as input
  EXPECT_EQ(1, s3->num_inputs());
  EXPECT_EQ(*(s3->in_nodes().begin()), d);
}

TEST_F(ConstantFoldingTest, TwoOutputs) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto s0 = ops::Const<int>(s, {1}, {1});
    auto s1 = ops::Const<int>(s, {2, 2}, {2});
    auto b = ops::internal::BroadcastGradientArgs(s, s0, s1);
    auto b0 = ops::_Send(s.WithOpName("b0"), ops::Identity(s, b.r0), "b0",
                         "sender", 0, "receiver");
    auto b1 = ops::_Send(s.WithOpName("b1"), ops::Identity(s, b.r1), "b1",
                         "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  bool was_mutated;
  TF_ASSERT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* b0 = index.at("b0");
  Node* b1 = index.at("b1");

  EXPECT_EQ(1, b0->num_inputs());
  ExpectNodeEqual<int>(*(b0->in_nodes().begin()), {0, 1}, {2});
  EXPECT_EQ(1, b1->num_inputs());
  ExpectNodeEqual<int>(*(b1->in_nodes().begin()), {}, {0});
}

TEST_F(ConstantFoldingTest, TwoOutputsFoldOneOutput) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto s0 = ops::Const<int>(s, {1}, {1});
    auto s1 = ops::Const<int>(s, {2, 2}, {2});
    auto b = ops::internal::BroadcastGradientArgs(s, s0, s1);
    auto b0 = ops::_Send(s.WithOpName("b0"), ops::Identity(s, b.r0), "b0",
                         "sender", 0, "receiver");
    auto b1_ident = ops::Identity(s.WithOpName("b1_ident"), b.r1);
    auto b1 =
        ops::_Send(s.WithOpName("b1"), b1_ident, "b1", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  ConstantFoldingOptions opts;
  opts.consider = [](const Node* n) { return "b1_ident" != n->name(); };
  bool was_mutated;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* b0 = index.at("b0");
  Node* b1 = index.at("b1");
  Node* b1_ident = index.at("b1_ident");

  // 0th output of b should have been folded.
  ASSERT_EQ(1, b0->num_inputs());
  ExpectNodeEqual<int>(*(b0->in_nodes().begin()), {0, 1}, {2});
  // 1st output of b should still be b1_ident. However, b1_ident's input must
  // have been replaced with a constant.
  ASSERT_EQ(1, b1->num_inputs());
  EXPECT_EQ(*(b1->in_nodes().begin()), b1_ident);

  ASSERT_EQ(1, b1_ident->num_inputs());
  ExpectNodeEqual<int>(*(b1_ident->in_nodes().begin()), {}, {0});
}

TEST_F(ConstantFoldingTest, TestNoReplaceLargeConstant) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto s0 = ops::Const<int>(s, 0, {5 * 1024 * 256});
    auto s1 = ops::Const<int>(s, 0, {5 * 1024 * 256 + 1});
    auto concat_dim = ops::Const<int>(s, 0);
    auto concat = ops::Concat(s, {s0, s1}, concat_dim);
    auto concat_send = ops::_Send(s.WithOpName("concat_send"), concat,
                                  "concat_send", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  // The above concat should not have been constant folded.
  bool was_mutated;
  TF_EXPECT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_FALSE(was_mutated);

  // Increase the limit and the concat should now be constant folded.
  ConstantFoldingOptions opt;
  opt.max_constant_size_in_bytes = 10 * 1024 * 1024 + 4;
  TF_EXPECT_OK(
      ConstantFold(opt, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);
}

TEST_F(ConstantFoldingTest, TestNoReplaceFunctionCall) {
  FunctionDefLibrary flib;
  *flib.add_function() = test::function::XTimesTwo();

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph g(flib_def);
  {
    Scope s = Scope::NewRootScope();
    auto c = ops::Const<int32>(s.WithOpName("c"), {1}, {1});
    TF_EXPECT_OK(s.graph()->AddFunctionLibrary(flib));

    // TODO(phawkins): there is no way to make a function call using the C++
    // graph builder API.
    NodeDef def;
    TF_ASSERT_OK(
        NodeDefBuilder("times_two", "XTimesTwo", s.graph()->op_registry())
            .Input(c.name(), 0, DT_INT32)
            .Finalize(&def));
    absl::Status status;
    Node* times_two = s.graph()->AddNode(def, &status);
    TF_ASSERT_OK(status);
    TF_ASSERT_OK(s.DoShapeInference(times_two));
    s.graph()->AddEdge(c.node(), 0, times_two, 0);

    auto times_two_send =
        ops::_Send(s.WithOpName("times_two_send"), Output(times_two),
                   "times_two_send", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  // The above function call should not have been constant folded.
  bool was_mutated;
  TF_EXPECT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_FALSE(was_mutated);
}

REGISTER_OP("ConstantFoldingTestOp")
    .Input("a: int64")
    .Output("b: int64")
    .SetShapeFn(shape_inference::UnknownShape);

TEST_F(ConstantFoldingTest, TestNoReplaceNonCPUOp) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto aconst = ops::Const<int64_t>(s, 0, {5});

    NodeDef def;
    TF_ASSERT_OK(NodeDefBuilder("testop", "ConstantFoldingTestOp")
                     .Input(aconst.name(), 0, DT_INT64)
                     .Finalize(&def));
    absl::Status status;
    Node* non_cpu = s.graph()->AddNode(def, &status);
    TF_ASSERT_OK(status);
    TF_ASSERT_OK(s.DoShapeInference(non_cpu));

    auto non_cpu_send =
        ops::_Send(s.WithOpName("non_cpu_send"), Output(non_cpu),
                   "non_cpu_send", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }

  // The non-CPU op should not have been constant folded.
  bool was_mutated;
  TF_EXPECT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_FALSE(was_mutated);
}

TEST_F(ConstantFoldingTest, ControlDependencies) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto c0 = ops::Const<int>(s, 1);
    auto recv1 = ops::_Recv(s.WithOpName("recv1"), DT_FLOAT, "recv1", "sender",
                            0, "receiver");
    auto c1 = ops::Const<int>(s.WithControlDependencies(recv1), 2);
    auto recv2 = ops::_Recv(s.WithOpName("recv2"), DT_FLOAT, "recv2", "sender",
                            0, "receiver");
    auto c2 = ops::Const<int>(s.WithControlDependencies(recv2), 3);
    auto add = ops::Add(s.WithControlDependencies(c2), c0, c1);
    auto send =
        ops::_Send(s.WithOpName("send"), add, "send", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }
  bool was_mutated;
  TF_EXPECT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, Env::Default(),
                            nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* recv1 = index.at("recv1");
  Node* recv2 = index.at("recv2");
  Node* send = index.at("send");

  ASSERT_EQ(1, send->num_inputs());
  Node* p = *(send->in_nodes().begin());
  ExpectNodeEqual<int>(p, {3}, {});

  ASSERT_EQ(2, p->in_edges().size());
  for (const Edge* e : p->in_edges()) {
    EXPECT_TRUE(e->IsControlEdge());
    EXPECT_TRUE(e->src() == recv1 || e->src() == recv2) << e->src()->name();
  }
}

TEST_F(ConstantFoldingTest, SimpleShapeKnown) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    Output recv0 = ops::_Recv(s.WithOpName("recv0"), DT_FLOAT, "recv0",
                              "sender", 0, "receiver");
    auto shape = ops::Shape(s.WithOpName("shape"), recv0);
    Output recv1 = ops::_Recv(s.WithOpName("recv1"), DT_FLOAT, "recv1",
                              "sender", 0, "receiver");
    auto shape_n = ops::ShapeN(s.WithOpName("shape_n"), {recv0, recv1});
    auto rank = ops::Rank(s.WithOpName("rank"), recv0);
    auto size = ops::Size(s.WithOpName("size"), recv1);
    auto recv2 = ops::_Recv(s.WithOpName("recv2"), DT_FLOAT, "recv2", "sender",
                            0, "receiver");
    auto c = ops::Const<int>(s.WithControlDependencies(recv2), 3);
    auto add0 = ops::Add(s.WithControlDependencies(c), rank, size);
    auto add1 = ops::Add(s, shape, shape_n[0]);
    auto add2 = ops::Add(s, shape_n[1], shape_n[1]);
    auto send0 = ops::_Send(s.WithOpName("send0"), add0, "send0", "sender", 0,
                            "receiver");
    auto send1 = ops::_Send(s.WithOpName("send1"), add1, "send1", "sender", 0,
                            "receiver");
    auto send2 = ops::_Send(s.WithOpName("send2"), add2, "send2", "sender", 0,
                            "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }
  std::unordered_map<string, Node*> orig_index = g.BuildNodeNameIndex();
  Node* recv0 = orig_index.at("recv0");
  Node* recv1 = orig_index.at("recv1");
  PartialTensorShape ps0;
  int r0_dims[] = {1, 2};
  TF_EXPECT_OK(PartialTensorShape::MakePartialShape(r0_dims, 2, &ps0));
  PartialTensorShape ps1;
  int r1_dims[] = {2, 3, 4};
  TF_EXPECT_OK(PartialTensorShape::MakePartialShape<int>(r1_dims, 3, &ps1));
  std::unordered_map<string, std::vector<PartialTensorShape>> map;
  map[recv0->name()].push_back(ps0);
  map[recv1->name()].push_back(ps1);
  ConstantFoldingOptions opts;
  opts.shape_map = &map;
  bool was_mutated;
  TF_EXPECT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* recv2 = index.at("recv2");
  Node* send0 = index.at("send0");
  Node* send1 = index.at("send1");
  Node* send2 = index.at("send2");

  ASSERT_EQ(1, send0->num_inputs());
  Node* cf0 = *(send0->in_nodes().begin());
  ExpectNodeEqual<int>(cf0, {26}, {});

  ASSERT_EQ(1, send1->num_inputs());
  Node* cf1 = *(send1->in_nodes().begin());
  ExpectNodeEqual<int>(cf1, {2, 4}, {2});

  ASSERT_EQ(1, send2->num_inputs());
  Node* cf2 = *(send2->in_nodes().begin());
  ExpectNodeEqual<int>(cf2, {4, 6, 8}, {3});

  ASSERT_EQ(3, cf0->in_edges().size());
  for (const Edge* e : cf0->in_edges()) {
    EXPECT_TRUE(e->IsControlEdge());
    EXPECT_TRUE(e->src() == recv0 || e->src() == recv1 || e->src() == recv2)
        << e->src()->name();
  }

  ASSERT_EQ(2, cf1->in_edges().size());
  for (const Edge* e : cf1->in_edges()) {
    EXPECT_TRUE(e->IsControlEdge());
    EXPECT_TRUE(e->src() == recv0 || e->src() == recv1) << e->src()->name();
  }

  ASSERT_EQ(2, cf2->in_edges().size());
  for (const Edge* e : cf2->in_edges()) {
    EXPECT_TRUE(e->IsControlEdge());
    EXPECT_TRUE(e->src() == recv0 || e->src() == recv1) << e->src()->name();
  }
}

TEST_F(ConstantFoldingTest, PartialShape) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    Output recv0 = ops::_Recv(s.WithOpName("recv0"), DT_FLOAT, "recv0",
                              "sender", 0, "receiver");
    Output recv1 = ops::_Recv(s.WithOpName("recv1"), DT_FLOAT, "recv1",
                              "sender", 0, "receiver");
    auto shape = ops::Shape(s.WithOpName("shape"), recv0);
    auto rank0 = ops::Rank(s.WithOpName("rank0"), recv0);
    auto rank1 = ops::Rank(s.WithOpName("rank1"), recv1);
    auto size = ops::Size(s.WithOpName("size"), recv0);
    auto send0 = ops::_Send(s.WithOpName("send0"), rank0, "send0", "sender", 0,
                            "receiver");
    auto send1 = ops::_Send(s.WithOpName("send1"), shape, "send1", "sender", 0,
                            "receiver");
    auto send2 = ops::_Send(s.WithOpName("send2"), size, "send2", "sender", 0,
                            "receiver");
    auto send3 = ops::_Send(s.WithOpName("send3"), rank1, "send3", "sender", 0,
                            "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }
  std::unordered_map<string, Node*> orig_index = g.BuildNodeNameIndex();
  Node* recv0 = orig_index.at("recv0");
  Node* recv1 = orig_index.at("recv1");
  PartialTensorShape ps0;
  int r0_dims[] = {-1, -1};
  TF_EXPECT_OK(PartialTensorShape::MakePartialShape(r0_dims, 2, &ps0));
  PartialTensorShape ps1;
  std::unordered_map<string, std::vector<PartialTensorShape>> map;
  map[recv0->name()].push_back(ps0);
  map[recv1->name()].push_back(ps1);
  ConstantFoldingOptions opts;
  opts.shape_map = &map;
  bool was_mutated;
  TF_EXPECT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* shape = index.at("shape");
  Node* size = index.at("size");
  Node* rank1 = index.at("rank1");
  Node* send0 = index.at("send0");
  Node* send1 = index.at("send1");
  Node* send2 = index.at("send2");
  Node* send3 = index.at("send3");

  ASSERT_EQ(1, send0->num_inputs());
  Node* cf0 = *(send0->in_nodes().begin());
  ExpectNodeEqual<int>(cf0, {2}, {});

  ASSERT_EQ(1, send1->num_inputs());
  Node* ncf1 = *(send1->in_nodes().begin());
  EXPECT_EQ(ncf1, shape);

  ASSERT_EQ(1, send2->num_inputs());
  Node* ncf2 = *(send2->in_nodes().begin());
  EXPECT_EQ(ncf2, size);

  ASSERT_EQ(1, send3->num_inputs());
  Node* ncf3 = *(send3->in_nodes().begin());
  EXPECT_EQ(ncf3, rank1);
}

TEST_F(ConstantFoldingTest, ConstShapeKnown) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope();
    auto recv0 = ops::_Recv(s.WithOpName("recv0"), DT_FLOAT, "recv0", "sender",
                            0, "receiver");
    auto c0 =
        ops::Const<int>(s.WithOpName("c0").WithControlDependencies(recv0), 1);
    auto rank = ops::Rank(s.WithOpName("rank"), c0);
    auto add0 = ops::Add(s, rank, rank);
    auto send0 = ops::_Send(s.WithOpName("send0"), add0, "send0", "sender", 0,
                            "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }
  std::unordered_map<string, Node*> orig_index = g.BuildNodeNameIndex();
  Node* c0 = orig_index.at("c0");
  PartialTensorShape ps0;
  int c0_dims[] = {};
  TF_EXPECT_OK(PartialTensorShape::MakePartialShape(c0_dims, 0, &ps0));
  std::unordered_map<string, std::vector<PartialTensorShape>> map;
  map[c0->name()].push_back(ps0);
  ConstantFoldingOptions opts;
  opts.shape_map = &map;
  bool was_mutated;
  TF_EXPECT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);

  std::unordered_map<string, Node*> index = g.BuildNodeNameIndex();
  Node* recv0 = index.at("recv0");
  Node* send0 = index.at("send0");

  ASSERT_EQ(1, send0->num_inputs());
  Node* cf0 = *(send0->in_nodes().begin());
  ExpectNodeEqual<int>(cf0, {0}, {});

  ASSERT_EQ(1, cf0->in_edges().size());
  for (const Edge* e : cf0->in_edges()) {
    EXPECT_TRUE(e->IsControlEdge());
    EXPECT_TRUE(e->src() == recv0) << e->src()->name();
  }
}

TEST_F(ConstantFoldingTest, FoldShapeFromDynamicArgPreservesContents) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto send = ops::_Send(s.WithOpName("send"), shape, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ExpectNodeEqual<int32>(folded, {977, 16}, {2});

  // This test checks the stable contract we care about: after folding
  // Shape(arg), the replacement Const still carries symbolic contents.
  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 2);
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_FALSE(IsDynamicDimExpr(contents_proto.expressions(1)));
  EXPECT_EQ(contents_proto.dim(0).size(), 977);
  EXPECT_EQ(contents_proto.dim(1).size(), 16);
}

TEST_F(ConstantFoldingTest, FoldSliceOfDynamicShapePreservesContents) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto begin = ops::Const(s.WithOpName("begin"), {0}, {1});
  auto size = ops::Const(s.WithOpName("size"), {1}, {1});
  auto slice = ops::Slice(s.WithOpName("slice"), shape, begin, size);
  auto send = ops::_Send(s.WithOpName("send"), slice, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ExpectNodeEqual<int32>(folded, {977}, {1});

  // Folding Slice(Shape(arg), [0], [1]) should preserve the selected symbolic
  // content, not just the concrete value 977.
  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 1);
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_EQ(contents_proto.dim(0).size(), 977);
}

TEST_F(ConstantFoldingTest, FoldGatherOfDynamicShapePreservesContents) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto index = ops::Const(s.WithOpName("index"), 0);
  auto gather = ops::GatherV2(s.WithOpName("gather"), shape, index,
                              ops::Const(s.WithOpName("axis"), 0));
  auto send = ops::_Send(s.WithOpName("send"), gather, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ExpectNodeEqual<int32>(folded, {977}, {});

  // Folding Gather(Shape(arg), 0, axis=0) should preserve the selected
  // symbolic content, not just the concrete scalar value 977.
  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 1);
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_EQ(contents_proto.dim(0).size(), 977);
}

TEST_F(ConstantFoldingTest, FoldReshapeOfDynamicShapePreservesContents) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto begin = ops::Const(s.WithOpName("begin"), {0}, {1});
  auto size = ops::Const(s.WithOpName("size"), {1}, {1});
  auto slice = ops::Slice(s.WithOpName("slice"), shape, begin, size);
  auto scalar_shape = ops::Const<int32>(s.WithOpName("scalar_shape"), {});
  auto reshape =
      ops::Reshape(s.WithOpName("reshape"), slice, scalar_shape);
  auto send = ops::_Send(s.WithOpName("send"), reshape, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ExpectNodeEqual<int32>(folded, {977}, {});

  // Folding Reshape(Slice(Shape(arg), [0], [1]), []) should preserve the
  // selected symbolic content when the shape-vector result is scalarized.
  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 1);
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_EQ(contents_proto.dim(0).size(), 977);
}

TEST_F(ConstantFoldingTest, FoldPackWithRepeatedDynamicInputPreservesContents) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto first = ops::GatherV2(s.WithOpName("first"), shape,
                             ops::Const(s.WithOpName("index"), 0),
                             ops::Const(s.WithOpName("axis"), 0));
  // Reusing the same producer is a DAG, not a recursion cycle. Both visits must
  // preserve its symbolic content.
  OutputList pack_inputs = {first, first};
  auto pack = ops::Stack(s.WithOpName("pack"), pack_inputs,
                         ops::Stack::Axis(0));
  auto send = ops::_Send(s.WithOpName("send"), pack, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ASSERT_TRUE(folded->IsConstant());
  ExpectNodeEqual<int32>(folded, {977, 977}, {2});

  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 2);
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(1)));
  EXPECT_EQ(contents_proto.dim(0).size(), 977);
  EXPECT_EQ(contents_proto.dim(1).size(), 977);
}

TEST_F(ConstantFoldingTest, FoldPackKeepsSymbolicContentsAligned) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto dynamic_value = ops::GatherV2(
      s.WithOpName("dynamic_value"), shape,
      ops::Const(s.WithOpName("index"), 0),
      ops::Const(s.WithOpName("axis"), 0));
  auto static_value = ops::Const(s.WithOpName("static_value"), 16);
  auto pack = ops::Stack(s.WithOpName("pack"),
                         OutputList{static_value, dynamic_value},
                         ops::Stack::Axis(0));
  auto send = ops::_Send(s.WithOpName("send"), pack, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(index_by_name.at("send")->input_edge(0, &send_input));
  Node* folded = send_input->src();
  ASSERT_TRUE(folded->IsConstant());
  ExpectNodeEqual<int32>(folded, {16, 977}, {2});

  string serialized_contents_proto;
  TF_ASSERT_OK(GetNodeAttr(folded->attrs(),
                           "_user_inferred_value_contents",
                           &serialized_contents_proto));
  TensorShapeProto contents_proto;
  ASSERT_TRUE(contents_proto.ParseFromString(serialized_contents_proto));
  ASSERT_EQ(contents_proto.expressions_size(), 2);
  EXPECT_FALSE(IsDynamicDimExpr(contents_proto.expressions(0)));
  EXPECT_TRUE(IsDynamicDimExpr(contents_proto.expressions(1)));
  EXPECT_EQ(contents_proto.dim(0).size(), 16);
  EXPECT_EQ(contents_proto.dim(1).size(), 977);
}

TEST_F(ConstantFoldingTest, DoNotFoldUnsupportedDynamicContentsTransform) {
  Graph g(OpRegistry::Global());
  Scope s = Scope::NewRootScope();
  auto arg = ops::_Arg(s.WithOpName("arg"), DT_FLOAT, 0);
  auto shape = ops::Shape(s.WithOpName("shape"), arg);
  auto added = ops::Add(s.WithOpName("added"), shape, shape);
  auto send = ops::_Send(s.WithOpName("send"), added, "send", "sender", 0,
                         "receiver");
  TF_ASSERT_OK(s.ToGraph(&g));

  std::unordered_map<string, Node*> index_by_name = g.BuildNodeNameIndex();
  Node* arg_node = index_by_name.at("arg");
  arg_node->AddAttr("_output_shapes",
                    std::vector<TensorShapeProto>{MakeDynamicShapeProto977x16()});

  PartialTensorShape partial_shape({977, 16});
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  shape_map[arg_node->name()].push_back(partial_shape);

  ConstantFoldingOptions opts;
  opts.shape_map = &shape_map;
  bool was_mutated = false;
  TF_ASSERT_OK(
      ConstantFold(opts, nullptr, Env::Default(), nullptr, &g, &was_mutated));

  index_by_name = g.BuildNodeNameIndex();
  Node* send_node = index_by_name.at("send");
  const Edge* send_input = nullptr;
  TF_ASSERT_OK(send_node->input_edge(0, &send_input));
  EXPECT_FALSE(send_input->src()->IsConstant());
}

TEST_F(ConstantFoldingTest, NoReplacePartialOutput) {
  Graph g(OpRegistry::Global());
  {
    Scope s = Scope::NewRootScope().ExitOnError().WithAssignedDevice("/gpu:0");

    auto c0 = ops::Const<float>(s.WithOpName("c0"), {5.0, 2.0, 8.0, 1.0}, {4});
    auto k = ops::Const<int>(s.WithOpName("k"), 3);
    auto topK =
        ops::TopK(s.WithOpName("topK"), c0, k, ops::TopK::Sorted(false));
    auto send_values = ops::_Send(s.WithOpName("send_values"), topK.values,
                                  "send_values", "sender", 0, "receiver");
    auto send_indices = ops::_Send(s.WithOpName("send_indices"), topK.indices,
                                   "send_indices", "sender", 0, "receiver");
    TF_ASSERT_OK(s.ToGraph(&g));
  }
  bool was_mutated;
  TF_EXPECT_OK(ConstantFold(
      ConstantFoldingOptions{}, nullptr, Env::Default(),
      FakeDevice::Make("/job:tpu_worker/replica:0/task:0/device:GPU:0",
                       DEVICE_GPU)
          .get(),
      &g, &was_mutated));
  EXPECT_FALSE(was_mutated);
}

namespace {

const char kTestMemRegionName[] = "test://test";

class TestReadOnlyMemoryRegion : public ::tensorflow::ReadOnlyMemoryRegion {
 public:
  ~TestReadOnlyMemoryRegion() override = default;
  TestReadOnlyMemoryRegion(const void* data, uint64 length)
      : data_(data), length_(length) {}
  const void* data() override { return data_; }
  uint64 length() override { return length_; }

 protected:
  const void* data_;
  uint64 length_;
};

class TestTFFileSystem : public ::tensorflow::NullFileSystem {
 public:
  TestTFFileSystem()
      : ::tensorflow::NullFileSystem(),
        data_tensor_(test::AsTensor<double>({1., 2., 3., 4.}, {2, 2})) {}

  using ::tensorflow::NullFileSystem::NewReadOnlyMemoryRegionFromFile;

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, ::tensorflow::TransactionToken* token,
      std::unique_ptr<::tensorflow::ReadOnlyMemoryRegion>* result) override {
    if (fname != kTestMemRegionName) {
      return ::tensorflow::errors::Unimplemented(
          "NewReadOnlyMemoryRegionFromFile unimplemented");
    }
    const absl::string_view sp = data_tensor_.tensor_data();
    *result = std::unique_ptr<::tensorflow::ReadOnlyMemoryRegion>(
        new TestReadOnlyMemoryRegion(sp.data(), sp.size()));
    return absl::OkStatus();
  }

 protected:
  ::tensorflow::Tensor data_tensor_;
};

// A test TF environment that checks that the environment was used.
class TestTFEnvironment : public ::tensorflow::EnvWrapper {
 public:
  using tf_base = ::tensorflow::EnvWrapper;
  TestTFEnvironment() : ::tensorflow::EnvWrapper(Default()) {}
  absl::Status GetFileSystemForFile(
      const string& fname, ::tensorflow::FileSystem** result) override {
    was_used_ = true;
    if (fname == "test://test") {
      *result = &test_filesystem_;
      return absl::OkStatus();
    }
    return tf_base::GetFileSystemForFile(fname, result);
  }
  bool was_used() const { return was_used_; }

 protected:
  TestTFFileSystem test_filesystem_;
  bool was_used_ = false;
};
}  // namespace

TEST_F(ConstantFoldingTest, TestImmutableConst) {
  Graph g(OpRegistry::Global());
  Scope root = Scope::NewRootScope();

  auto a = ops::ImmutableConst(root, DT_DOUBLE, {2, 2}, kTestMemRegionName);
  auto b = ops::Const<double>(root, {1.0, 2.0, 3.0, 4.0}, {2, 2});
  auto c = ops::RandomGamma(root, {2, 2}, 2.0);
  auto result1 = ops::MatMul(root, a, b);
  auto result2 = ops::MatMul(root, result1, c);
  TF_ASSERT_OK(root.ToGraph(&g));
  TestTFEnvironment test_env;
  bool was_mutated;
  absl::Status status = ConstantFold(ConstantFoldingOptions{}, nullptr,
                                     Env::Default(), nullptr, &g, &was_mutated);
  EXPECT_FALSE(was_mutated);
  EXPECT_FALSE(status.ok());
  TF_EXPECT_OK(ConstantFold(ConstantFoldingOptions{}, nullptr, &test_env,
                            nullptr, &g, &was_mutated));
  EXPECT_TRUE(was_mutated);
}

}  // namespace
}  // namespace tensorflow

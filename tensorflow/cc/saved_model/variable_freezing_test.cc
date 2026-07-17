#include "tensorflow/cc/saved_model/variable_freezing.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace internal {
namespace {

std::string CurrentTestExportDir() {
  const ::testing::TestInfo* test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();
  return io::JoinPath(testing::TmpDir(),
                      absl::StrCat("variable_freezing_test_",
                                   test_info->test_suite_name(), "_",
                                   test_info->name()));
}

NodeDef* AddNode(GraphDef* graph, const std::string& name,
                 const std::string& op,
                 const std::vector<std::string>& inputs = {}) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  for (const std::string& input : inputs) {
    node->add_input(input);
  }
  return node;
}

void SetTypeAttr(NodeDef* node, const std::string& attr_name, DataType dtype) {
  (*node->mutable_attr())[attr_name].set_type(dtype);
}

void SetStringAttr(NodeDef* node, const std::string& attr_name,
                   const std::string& value) {
  (*node->mutable_attr())[attr_name].set_s(value);
}

void SetShapeAttr(NodeDef* node, const TensorShape& shape) {
  shape.AsProto((*node->mutable_attr())["shape"].mutable_shape());
}

void SetTensorAttr(NodeDef* node, const std::string& attr_name,
                   const Tensor& tensor) {
  SetTypeAttr(node, "dtype", tensor.dtype());
  tensor.AsProtoTensorContent(
      (*node->mutable_attr())[attr_name].mutable_tensor());
}

Tensor FloatTensor(const std::vector<float>& values) {
  Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(values.size())}));
  test::FillValues<float>(&tensor, values);
  return tensor;
}

Tensor StringTensor(const std::vector<tstring>& values) {
  Tensor tensor(DT_STRING, TensorShape({static_cast<int64_t>(values.size())}));
  auto flat = tensor.flat<tstring>();
  for (int i = 0; i < values.size(); ++i) {
    flat(i) = values[i];
  }
  return tensor;
}

std::string VariablesPrefix(const std::string& export_dir) {
  return io::JoinPath(export_dir, kSavedModelVariablesDirectory,
                      kSavedModelVariablesFilename);
}

void WriteCheckpoint(const std::string& export_dir,
                     const std::string& tensor_name, const Tensor& tensor) {
  TF_ASSERT_OK(Env::Default()->RecursivelyCreateDir(
      io::JoinPath(export_dir, kSavedModelVariablesDirectory)));
  BundleWriter writer(Env::Default(), VariablesPrefix(export_dir));
  TF_ASSERT_OK(writer.Add(tensor_name, tensor));
  TF_ASSERT_OK(writer.Finish());
}

const NodeDef* FindNode(const GraphDef& graph, const std::string& name) {
  for (const NodeDef& node : graph.node()) {
    if (node.name() == name) return &node;
  }
  return nullptr;
}

void ExpectConstTensorEquals(const NodeDef& node, const Tensor& expected) {
  ASSERT_EQ(node.op(), "Const");
  const auto dtype_it = node.attr().find("dtype");
  ASSERT_NE(dtype_it, node.attr().end());
  EXPECT_EQ(dtype_it->second.type(), expected.dtype());

  const auto value_it = node.attr().find("value");
  ASSERT_NE(value_it, node.attr().end());
  ASSERT_TRUE(value_it->second.has_tensor());

  Tensor actual;
  ASSERT_TRUE(actual.FromProto(value_it->second.tensor()));
  test::ExpectTensorEqual<float>(expected, actual);
}

void AddMutatingVarHandleReadGraph(GraphDef* graph) {
  NodeDef* handle = AddNode(graph, "model/bn/gamma", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetShapeAttr(handle, TensorShape({1}));
  SetStringAttr(handle, "shared_name", "model/bn/gamma");
  NodeDef* new_value = AddNode(graph, "new_gamma", "Placeholder");
  SetTypeAttr(new_value, "dtype", DT_FLOAT);
  NodeDef* assign = AddNode(graph, "assign_gamma", "AssignVariableOp",
                            {"model/bn/gamma", "new_gamma"});
  SetTypeAttr(assign, "dtype", DT_FLOAT);
  NodeDef* read = AddNode(graph, "read_gamma", "ReadVariableOp",
                          {"model/bn/gamma", "^assign_gamma"});
  SetTypeAttr(read, "dtype", DT_FLOAT);
}

void AddMutatingVariableV2ReadGraph(GraphDef* graph,
                                   const std::string& export_dir) {
  NodeDef* variable = AddNode(graph, "model/dense/bias", "VariableV2");
  SetTypeAttr(variable, "dtype", DT_FLOAT);

  Tensor tensor_names = StringTensor({"checkpoint/dense/bias"});
  NodeDef* tensor_names_node =
      AddNode(graph, "save/RestoreV2/tensor_names", "Const");
  SetTensorAttr(tensor_names_node, "value", tensor_names);

  Tensor shape_and_slices = StringTensor({""});
  NodeDef* shape_and_slices_node =
      AddNode(graph, "save/RestoreV2/shape_and_slices", "Const");
  SetTensorAttr(shape_and_slices_node, "value", shape_and_slices);

  NodeDef* prefix = AddNode(graph, "save/Const", "Const");
  SetTensorAttr(prefix, "value", StringTensor({VariablesPrefix(export_dir)}));

  AddNode(graph, "save/RestoreV2", "RestoreV2",
          {"save/Const", "save/RestoreV2/tensor_names",
           "save/RestoreV2/shape_and_slices"});
  AddNode(graph, "save/Assign", "Assign",
          {"model/dense/bias", "save/RestoreV2"});

  NodeDef* new_value = AddNode(graph, "new_bias", "Placeholder");
  SetTypeAttr(new_value, "dtype", DT_FLOAT);
  NodeDef* runtime_assign =
      AddNode(graph, "assign_bias", "Assign", {"model/dense/bias", "new_bias"});
  SetTypeAttr(runtime_assign, "T", DT_FLOAT);

  NodeDef* read =
      AddNode(graph, "model/dense/bias/read", "Identity",
              {"model/dense/bias", "^assign_bias"});
  SetTypeAttr(read, "T", DT_FLOAT);
}

void RunAssignThenRead(const GraphDef& graph, float new_value, Tensor* output) {
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  auto status = session->Create(graph);
  EXPECT_TRUE(status.ok()) << status.ToString();
  if (!status.ok()) return;

  std::vector<Tensor> outputs;
  status = session->Run({{"new_gamma", FloatTensor({new_value})}},
                        {"read_gamma"}, {}, &outputs);
  EXPECT_TRUE(status.ok()) << status.ToString();
  if (!status.ok()) return;

  if (outputs.size() != 1) {
    ADD_FAILURE() << "expected one read_gamma output, got " << outputs.size();
    return;
  }
  *output = outputs[0];
}

TEST(VariableFreezingTest, FreezesAllowlistedVarHandleRead) {
  const std::string export_dir = CurrentTestExportDir();
  const Tensor frozen = FloatTensor({1.0f, 2.0f});
  WriteCheckpoint(export_dir, "model/dense/kernel", frozen);

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "model/dense/kernel", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetStringAttr(handle, "shared_name", "model/dense/kernel");
  NodeDef* read = AddNode(graph, "model/dense/kernel/Read/ReadVariableOp",
                          "ReadVariableOp", {"model/dense/kernel"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_handle =
      FindNode(meta_graph_def.graph_def(), "model/dense/kernel");
  ASSERT_NE(rewritten_handle, nullptr);
  EXPECT_EQ(rewritten_handle->op(), "VarHandleOp");

  const NodeDef* rewritten_read = FindNode(
      meta_graph_def.graph_def(), "model/dense/kernel/Read/ReadVariableOp");
  ASSERT_NE(rewritten_read, nullptr);
  ExpectConstTensorEquals(*rewritten_read, frozen);
}

TEST(VariableFreezingTest, SkipsLinearWeightsPartZeroVarHandleRead) {
  // Names containing "/part_" are explicitly rejected to avoid freezing
  // sharded variables, even when the rest of the name looks like a weight.
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "linear/linear_model/I1/weights/part_0",
                  FloatTensor({1.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle =
      AddNode(graph, "linear/linear_model/I1/weights/part_0", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetStringAttr(handle, "shared_name", "linear/linear_model/I1/weights/part_0");
  NodeDef* read =
      AddNode(graph, "linear/linear_model/I1/weights/ReadVariableOp",
              "ReadVariableOp", {"linear/linear_model/I1/weights/part_0"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read = FindNode(
      meta_graph_def.graph_def(), "linear/linear_model/I1/weights/ReadVariableOp");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, SkipsLinearBiasWeightsPartZeroVarHandleRead) {
  // Names containing "/part_" are explicitly rejected to avoid freezing
  // sharded variables, even when the rest of the name looks like a bias.
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "linear/linear_model/bias_weights/part_0",
                  FloatTensor({2.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "linear/linear_model/bias_weights/part_0",
                            "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetStringAttr(handle, "shared_name",
                "linear/linear_model/bias_weights/part_0");
  NodeDef* read = AddNode(graph,
                          "linear/linear_model/bias_weights/ReadVariableOp",
                          "ReadVariableOp",
                          {"linear/linear_model/bias_weights/part_0"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read = FindNode(
      meta_graph_def.graph_def(),
      "linear/linear_model/bias_weights/ReadVariableOp");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, SkipsKernelPartZeroVarHandleRead) {
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "model/dense/kernel/part_0", FloatTensor({9.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "model/dense/kernel/part_0", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetStringAttr(handle, "shared_name", "model/dense/kernel/part_0");
  NodeDef* read = AddNode(graph, "model/dense/kernel/part_0/read",
                          "ReadVariableOp",
                          {"model/dense/kernel/part_0"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/dense/kernel/part_0/read");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, FreezesVariableV2IdentityReadFromRestoreV2Key) {
  const std::string export_dir = CurrentTestExportDir();
  const Tensor frozen = FloatTensor({3.0f, 4.0f});
  WriteCheckpoint(export_dir, "checkpoint/dense/bias", frozen);

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* variable = AddNode(graph, "model/dense/bias", "VariableV2");
  SetTypeAttr(variable, "dtype", DT_FLOAT);

  Tensor tensor_names = StringTensor({"checkpoint/dense/bias"});
  NodeDef* tensor_names_node =
      AddNode(graph, "save/RestoreV2/tensor_names", "Const");
  SetTensorAttr(tensor_names_node, "value", tensor_names);

  Tensor shape_and_slices = StringTensor({""});
  NodeDef* shape_and_slices_node =
      AddNode(graph, "save/RestoreV2/shape_and_slices", "Const");
  SetTensorAttr(shape_and_slices_node, "value", shape_and_slices);

  NodeDef* prefix = AddNode(graph, "save/Const", "Const");
  SetTensorAttr(prefix, "value", StringTensor({VariablesPrefix(export_dir)}));

  AddNode(graph, "save/RestoreV2", "RestoreV2",
          {"save/Const", "save/RestoreV2/tensor_names",
           "save/RestoreV2/shape_and_slices"});
  AddNode(graph, "save/Assign", "Assign",
          {"model/dense/bias", "save/RestoreV2"});
  NodeDef* read = AddNode(graph, "model/dense/bias/read", "Identity",
                          {"model/dense/bias"});
  SetTypeAttr(read, "T", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_variable =
      FindNode(meta_graph_def.graph_def(), "model/dense/bias");
  ASSERT_NE(rewritten_variable, nullptr);
  EXPECT_EQ(rewritten_variable->op(), "VariableV2");

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/dense/bias/read");
  ASSERT_NE(rewritten_read, nullptr);
  ExpectConstTensorEquals(*rewritten_read, frozen);
}

TEST(VariableFreezingTest, SkipsDisallowedEmbeddingVarHandleName) {
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "model/embedding/kernel", FloatTensor({5.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "model/embedding/kernel", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetStringAttr(handle, "shared_name", "model/embedding/kernel");
  NodeDef* read = AddNode(graph, "model/embedding/kernel/read",
                          "ReadVariableOp", {"model/embedding/kernel"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/embedding/kernel/read");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, FreezesVarHandleReadWithPreservedAssignVariableOp) {
  // A VarHandleOp consumed by an AssignVariableOp (even one fed by a runtime
  // Placeholder) is still frozen: freezing only rewrites read paths and leaves
  // the variable handle plus its write paths intact. The AssignVariableOp is
  // preserved so session init / restore semantics are unchanged.
  const std::string export_dir = CurrentTestExportDir();
  const Tensor frozen = FloatTensor({7.0f});
  WriteCheckpoint(export_dir, "model/bn/gamma", frozen);

  MetaGraphDef meta_graph_def;
  AddMutatingVarHandleReadGraph(meta_graph_def.mutable_graph_def());

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* assign = FindNode(meta_graph_def.graph_def(), "assign_gamma");
  ASSERT_NE(assign, nullptr);
  EXPECT_EQ(assign->op(), "AssignVariableOp");

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "read_gamma");
  ASSERT_NE(rewritten_read, nullptr);
  ExpectConstTensorEquals(*rewritten_read, frozen);
}

TEST(VariableFreezingTest, FreezesVariableV2ReadWithPreservedRuntimeAssign) {
  // A VariableV2 consumed by a runtime Assign (fed by a Placeholder) is still
  // frozen: freezing only rewrites read paths and leaves the variable plus its
  // write paths intact. Both the restore Assign and the runtime Assign are
  // preserved.
  const std::string export_dir = CurrentTestExportDir();
  const Tensor frozen = FloatTensor({3.0f, 4.0f});
  WriteCheckpoint(export_dir, "checkpoint/dense/bias", frozen);

  MetaGraphDef meta_graph_def;
  AddMutatingVariableV2ReadGraph(meta_graph_def.mutable_graph_def(), export_dir);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* restore_assign =
      FindNode(meta_graph_def.graph_def(), "save/Assign");
  ASSERT_NE(restore_assign, nullptr);
  EXPECT_EQ(restore_assign->op(), "Assign");

  const NodeDef* runtime_assign =
      FindNode(meta_graph_def.graph_def(), "assign_bias");
  ASSERT_NE(runtime_assign, nullptr);
  EXPECT_EQ(runtime_assign->op(), "Assign");

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/dense/bias/read");
  ASSERT_NE(rewritten_read, nullptr);
  ExpectConstTensorEquals(*rewritten_read, frozen);
}

TEST(VariableFreezingTest, DoesNotFreezeVarHandleReadWithDtypeMismatch) {
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "model/dense/kernel", FloatTensor({1.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "model/dense/kernel", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_INT32);
  SetStringAttr(handle, "shared_name", "model/dense/kernel");
  NodeDef* read = AddNode(graph, "model/dense/kernel/read", "ReadVariableOp",
                          {"model/dense/kernel"});
  SetTypeAttr(read, "dtype", DT_INT32);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/dense/kernel/read");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, DoesNotFreezeVarHandleReadWithShapeMismatch) {
  const std::string export_dir = CurrentTestExportDir();
  WriteCheckpoint(export_dir, "model/dense/gamma", FloatTensor({1.0f}));

  MetaGraphDef meta_graph_def;
  GraphDef* graph = meta_graph_def.mutable_graph_def();
  NodeDef* handle = AddNode(graph, "model/dense/gamma", "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetShapeAttr(handle, TensorShape({2}));
  SetStringAttr(handle, "shared_name", "model/dense/gamma");
  NodeDef* read = AddNode(graph, "model/dense/gamma/read", "ReadVariableOp",
                          {"model/dense/gamma"});
  SetTypeAttr(read, "dtype", DT_FLOAT);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "model/dense/gamma/read");
  ASSERT_NE(rewritten_read, nullptr);
  EXPECT_EQ(rewritten_read->op(), "ReadVariableOp");
}

TEST(VariableFreezingTest, FrozenReadReturnsCheckpointValueAfterFreeze) {
  // After freezing, the read path is rewritten to a Const carrying the
  // checkpoint value. The runtime AssignVariableOp is preserved but no longer
  // feeds the read, so the read returns the frozen checkpoint value regardless
  // of the runtime-assigned value.
  const std::string export_dir = CurrentTestExportDir();
  const Tensor checkpoint_value = FloatTensor({7.0f});
  WriteCheckpoint(export_dir, "model/bn/gamma", checkpoint_value);

  MetaGraphDef meta_graph_def;
  AddMutatingVarHandleReadGraph(meta_graph_def.mutable_graph_def());
  Tensor before_freeze_output;
  RunAssignThenRead(meta_graph_def.graph_def(), 11.0f, &before_freeze_output);
  test::ExpectTensorEqual<float>(FloatTensor({11.0f}), before_freeze_output);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  // The read node is now a Const; it no longer depends on the variable handle
  // or the AssignVariableOp, so feeding new_gamma has no effect on the output.
  const NodeDef* rewritten_read =
      FindNode(meta_graph_def.graph_def(), "read_gamma");
  ASSERT_NE(rewritten_read, nullptr);
  ExpectConstTensorEquals(*rewritten_read, checkpoint_value);
}

// Builds a graph mirroring the real presort moving_mean pattern: a VarHandleOp
// with VarIsInitializedOp, a Const-backed initializer Assign, a direct
// ReadVariableOp, and a RestoreV2->Identity->AssignVariableOp restore path.
// All write-path consumers must be preserved while the direct read path is
// frozen to Const.
void AddMovingMeanPatternGraph(GraphDef* graph, const std::string& export_dir) {
  const std::string var_name = "model/deep_1/deep_1_dense_1/deep_1_dense_1_bn/moving_mean";

  // VarHandleOp
  NodeDef* handle = AddNode(graph, var_name, "VarHandleOp");
  SetTypeAttr(handle, "dtype", DT_FLOAT);
  SetShapeAttr(handle, TensorShape({2}));
  SetStringAttr(handle, "shared_name", var_name);

  // VarIsInitializedOp consumer
  AddNode(graph, var_name + "/IsInitialized/VarIsInitializedOp",
          "VarIsInitializedOp", {var_name});

  // Const initializer -> AssignVariableOp (init path)
  NodeDef* zeros = AddNode(graph, var_name + "/Initializer/zeros", "Const");
  SetTensorAttr(zeros, "value", FloatTensor({0.0f, 0.0f}));
  NodeDef* init_assign =
      AddNode(graph, var_name + "/Assign", "AssignVariableOp",
              {var_name, var_name + "/Initializer/zeros"});
  SetTypeAttr(init_assign, "dtype", DT_FLOAT);

  // Direct ReadVariableOp consumer
  NodeDef* direct_read =
      AddNode(graph, var_name + "/Read/ReadVariableOp", "ReadVariableOp",
              {var_name});
  SetTypeAttr(direct_read, "dtype", DT_FLOAT);

  // RestoreV2 -> Identity -> AssignVariableOp (restore path)
  Tensor tensor_names = StringTensor({var_name});
  NodeDef* tensor_names_node =
      AddNode(graph, "save/RestoreV2/tensor_names", "Const");
  SetTensorAttr(tensor_names_node, "value", tensor_names);
  Tensor shape_and_slices = StringTensor({""});
  NodeDef* shape_and_slices_node =
      AddNode(graph, "save/RestoreV2/shape_and_slices", "Const");
  SetTensorAttr(shape_and_slices_node, "value", shape_and_slices);
  NodeDef* prefix = AddNode(graph, "save/Const", "Const");
  SetTensorAttr(prefix, "value", StringTensor({VariablesPrefix(export_dir)}));
  AddNode(graph, "save/RestoreV2", "RestoreV2",
          {"save/Const", "save/RestoreV2/tensor_names",
           "save/RestoreV2/shape_and_slices"});
  NodeDef* restore_identity =
      AddNode(graph, "save/Identity", "Identity", {"save/RestoreV2"});
  SetTypeAttr(restore_identity, "T", DT_FLOAT);
  NodeDef* restore_assign =
      AddNode(graph, "save/AssignVariableOp", "AssignVariableOp",
              {var_name, "save/Identity"});
  SetTypeAttr(restore_assign, "dtype", DT_FLOAT);
}

TEST(VariableFreezingTest, FreezesVarHandleWithRestoreIdentityAssign) {
  // Reproduces the presort moving_mean pattern minus the resource Switch
  // chain (Switch is not a recognized read-only consumer, so a variable that
  // feeds a Switch is rejected for freezing). Here a VarHandleOp's consumers
  // include VarIsInitializedOp, a Const-backed init Assign, a direct
  // ReadVariableOp, and a RestoreV2->Identity->AssignVariableOp restore path.
  // All write paths are preserved; the direct read path is frozen to the
  // checkpoint value.
  const std::string export_dir = CurrentTestExportDir();
  const Tensor frozen = FloatTensor({0.12f, -0.34f});
  const std::string var_name =
      "model/deep_1/deep_1_dense_1/deep_1_dense_1_bn/moving_mean";
  WriteCheckpoint(export_dir, var_name, frozen);

  MetaGraphDef meta_graph_def;
  AddMovingMeanPatternGraph(meta_graph_def.mutable_graph_def(), export_dir);

  TF_ASSERT_OK(FreezeAllowlistedVariableReads(export_dir, &meta_graph_def));

  // VarHandleOp is preserved.
  const NodeDef* handle = FindNode(meta_graph_def.graph_def(), var_name);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->op(), "VarHandleOp");

  // VarIsInitializedOp is preserved.
  const NodeDef* var_is_init = FindNode(
      meta_graph_def.graph_def(), var_name + "/IsInitialized/VarIsInitializedOp");
  ASSERT_NE(var_is_init, nullptr);
  EXPECT_EQ(var_is_init->op(), "VarIsInitializedOp");

  // Init AssignVariableOp is preserved.
  const NodeDef* init_assign =
      FindNode(meta_graph_def.graph_def(), var_name + "/Assign");
  ASSERT_NE(init_assign, nullptr);
  EXPECT_EQ(init_assign->op(), "AssignVariableOp");

  // Restore AssignVariableOp (fed by RestoreV2->Identity) is preserved.
  const NodeDef* restore_assign =
      FindNode(meta_graph_def.graph_def(), "save/AssignVariableOp");
  ASSERT_NE(restore_assign, nullptr);
  EXPECT_EQ(restore_assign->op(), "AssignVariableOp");

  // Direct ReadVariableOp is frozen to Const.
  const NodeDef* direct_read = FindNode(
      meta_graph_def.graph_def(), var_name + "/Read/ReadVariableOp");
  ASSERT_NE(direct_read, nullptr);
  ExpectConstTensorEquals(*direct_read, frozen);
}

}  // namespace
}  // namespace internal
}  // namespace tensorflow

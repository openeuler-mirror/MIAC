/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"

#include <gmock/gmock.h>
#include <optional>
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/data_flow_ops.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/list_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {

class XlaCompilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();

    XlaOpRegistry::RegisterCompilationKernels();

    FunctionDefLibrary flib;
    flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), flib));
  }

  XlaCompiler::Options DefaultOptions() {
    XlaCompiler::Options options;
    options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
    options.client = client_;
    options.flib_def = flib_def_.get();
    return options;
  }

  FunctionLibraryDefinition* LocalFlibDef(XlaCompiler* compiler) {
    return compiler->local_flib_def_.get();
  }

  xla::Client* client_;
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;
};

class ScopedTfXlaDynamicSizesFlag {
 public:
  ScopedTfXlaDynamicSizesFlag() {
    old_value_ = GetMarkForCompilationPassFlags()->tf_xla_enable_dynamic_sizes;
    GetMarkForCompilationPassFlags()->tf_xla_enable_dynamic_sizes = true;
    SetTensorShapeExpressionsEnabledForTesting(true);
  }

  ~ScopedTfXlaDynamicSizesFlag() {
    SetTensorShapeExpressionsEnabledForTesting(std::nullopt);
    GetMarkForCompilationPassFlags()->tf_xla_enable_dynamic_sizes = old_value_;
  }

 private:
  bool old_value_ = false;
};

class XlaCompilerDynamicSizesTest : public XlaCompilerTest {
 protected:
  void SetUp() override {
    dynamic_sizes_flag_ = std::make_unique<ScopedTfXlaDynamicSizesFlag>();
    XlaCompilerTest::SetUp();
  }

  void TearDown() override { dynamic_sizes_flag_.reset(); }

 private:
  std::unique_ptr<ScopedTfXlaDynamicSizesFlag> dynamic_sizes_flag_;
};

namespace {

// Helper class to test the ability to pass resources through to XLA
// compiled kernels.
class DummyResourceForTest : public ResourceBase {
 public:
  string DebugString() const override { return "dummy"; }
  void Increment() { ++value_; }
  int Get() { return value_; }

 private:
  int value_ = 0;
};

class DummyReadResourceOp : public XlaOpKernel {
 public:
  explicit DummyReadResourceOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ResourceMgr* rm = ctx->op_kernel_context()->resource_manager();
    OP_REQUIRES(ctx, rm, errors::Internal("No resource manager."));
    DummyResourceForTest* dummy;
    OP_REQUIRES_OK(ctx, rm->Lookup<DummyResourceForTest>(
                            rm->default_container(), "dummy", &dummy));
    dummy->Increment();
    dummy->Unref();

    ctx->SetOutput(0, ctx->Input(0));
    ctx->SetOutput(1, ctx->Input(0));
  }
};

class DummyReadResourceCC {
 public:
  DummyReadResourceCC(const Scope& scope, const Input& value) {
    if (!scope.ok()) return;
    auto _value = ops::AsNodeOut(scope, value);
    if (!scope.ok()) return;
    Node* ret;
    const auto unique_name = scope.GetUniqueNameForOp("DummyReadResource");
    auto builder = NodeBuilder(unique_name, "DummyReadResource").Input(_value);
    scope.UpdateBuilder(&builder);
    scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
    if (!scope.ok()) return;
    scope.UpdateStatus(scope.DoShapeInference(ret));
    if (!scope.ok()) return;
    this->output1_ = Output(ret, 0);
    this->output2_ = Output(ret, 1);
  }

  Output output1_;
  Output output2_;
};

REGISTER_OP("DummyReadResource")
    .Input("input: int32")
    .Output("output1: int32")
    .Output("output2: int32")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output1: dummy output.
output2: dummy output.
)doc");

REGISTER_XLA_OP(Name("DummyReadResource"), DummyReadResourceOp);

// DummyDuplicateOp is present purely to test multiple REGISTER_XLA_OP calls
// on the same Op name below.
class DummyDuplicateOp : public XlaOpKernel {
 public:
  explicit DummyDuplicateOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    ctx->SetOutput(0, ctx->Input(0));
  }
};

REGISTER_OP("DummyDuplicateOp")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
A dummy Op.

input: dummy input.
output: dummy output.
)doc");

REGISTER_XLA_OP(Name("DummyDuplicateOp").Device(DEVICE_CPU_XLA_JIT),
                DummyDuplicateOp);
REGISTER_XLA_OP(Name("DummyDuplicateOp").Device(DEVICE_GPU_XLA_JIT),
                DummyDuplicateOp);

// Tests compilation and execution of an empty graph.
TEST_F(XlaCompilerTest, EmptyReturnValues) {
  XlaCompiler compiler(DefaultOptions());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph),
                                     /*args=*/{}, &result));

  TF_ASSERT_OK(client_->Execute(*result.computation, {}).status());
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, Simple) {
  // Builds a graph that adds two Tensors.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  // Tests that the generated computation works.
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({4, 143});
  xla::Literal expected_literal = xla::LiteralUtil::MakeTuple({&expected0});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

absl::StatusOr<std::unique_ptr<xla::HloModule>> LoadModuleFromHloProto(
    const xla::HloModuleProto& module_proto) {
  TF_ASSIGN_OR_RETURN(auto module_config,
                      xla::HloModule::CreateModuleConfigFromProto(
                          module_proto, xla::GetDebugOptionsFromFlags()));
  return xla::CreateModuleFromProto(module_proto, module_config);
}

// Tests compilation and execution of a graph that adds two tensors with dynamic
// shape parameters.
TEST_F(XlaCompilerTest, SimpleDynamicShapeParameter) {
  // Builds a graph that adds two Tensors.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Add(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape =
      xla::ShapeUtil::MakeShape(/*element_type=*/xla::S32, /*dimensions=*/{2},
                                /*dynamic_dimensions=*/std::vector<bool>{true},
                                /*expressions=*/{});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape(/*dimensions=*/{2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  auto hlo = result.computation->proto();
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadModuleFromHloProto(hlo));
  EXPECT_EQ(module->computation_count(), 1);
  EXPECT_TRUE(module->mutable_computation(0)
                  ->parameter_instruction(0)
                  ->shape()
                  .is_dynamic());
}

// Tests compilation preserves dynamic shape expressions through scalar
// broadcasting in standard binary ops.
TEST_F(XlaCompilerTest, ScalarBroadcastPreservesDynamicShapeExpressionsInAdd) {
  auto* flags = GetMarkForCompilationPassFlags();
  bool old_dynamic_sizes = flags->tf_xla_enable_dynamic_sizes;
  flags->tf_xla_enable_dynamic_sizes = true;
  auto restore_dynamic_sizes = gtl::MakeCleanup([&] {
    flags->tf_xla_enable_dynamic_sizes = old_dynamic_sizes;
  });

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto scalar = ops::Const(scope.WithOpName("Scalar"), 7);
  auto add = ops::Add(scope.WithOpName("Add"), a, scalar);
  auto ret = ops::_Retval(scope.WithOpName("Ret"), add, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      /*element_type=*/xla::S32, /*dimensions=*/{2, 1},
      /*dynamic_dimensions=*/{true, false},
      /*expressions=*/{xla::DExpr::Var(1), xla::DExpr::Const(1)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  compile_options.is_entry_computation = false;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "broadcast_add",
                                     std::move(graph), args, &result));

  auto hlo = result.computation->proto();
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadModuleFromHloProto(hlo));
  EXPECT_EQ(module->computation_count(), 1);

  const xla::Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  ASSERT_EQ(root_shape.dimensions_size(), 2);
  ASSERT_EQ(root_shape.expressions().size(), 2);
  EXPECT_TRUE(root_shape.expressions(0) &&
              root_shape.expressions(0)->is_dynamic());
  EXPECT_TRUE(root_shape.expressions(1) &&
              root_shape.expressions(1)->is_constant());
}

// Tests compilation preserves dynamic shape expressions through scalar
// broadcasting in binary ops that use the explicit tf2xla broadcast helper.
TEST_F(XlaCompilerTest,
       ScalarBroadcastPreservesDynamicShapeExpressionsInFloorMod) {
  auto* flags = GetMarkForCompilationPassFlags();
  bool old_dynamic_sizes = flags->tf_xla_enable_dynamic_sizes;
  flags->tf_xla_enable_dynamic_sizes = true;
  auto restore_dynamic_sizes = gtl::MakeCleanup([&] {
    flags->tf_xla_enable_dynamic_sizes = old_dynamic_sizes;
  });

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto scalar = ops::Const(scope.WithOpName("Scalar"), 7);
  auto floormod = ops::FloorMod(scope.WithOpName("FloorMod"), a, scalar);
  auto ret = ops::_Retval(scope.WithOpName("Ret"), floormod, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      /*element_type=*/xla::S32, /*dimensions=*/{2, 1},
      /*dynamic_dimensions=*/{true, false},
      /*expressions=*/{xla::DExpr::Var(1), xla::DExpr::Const(1)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  compile_options.is_entry_computation = false;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "broadcast_floormod",
                                     std::move(graph), args, &result));

  auto hlo = result.computation->proto();
  TF_ASSERT_OK_AND_ASSIGN(auto module, LoadModuleFromHloProto(hlo));
  EXPECT_EQ(module->computation_count(), 1);

  const xla::Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  ASSERT_EQ(root_shape.dimensions_size(), 2);
  ASSERT_EQ(root_shape.expressions().size(), 2);
  EXPECT_TRUE(root_shape.expressions(0) &&
              root_shape.expressions(0)->is_dynamic());
  EXPECT_TRUE(root_shape.expressions(1) &&
              root_shape.expressions(1)->is_constant());
}

TEST_F(XlaCompilerDynamicSizesTest, DynamicShapeParameterPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto identity = ops::Identity(scope.WithOpName("identity"), input);
  auto retval = ops::_Retval(scope.WithOpName("retval"), identity, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6}, std::vector<xla::DExpr>{xla::DExpr::Var(1)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "identity",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(1)));

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          LoadModuleFromHloProto(result.computation->proto()));
  const xla::Shape& param_shape =
      module->entry_computation()->parameter_instruction(0)->shape();
  EXPECT_TRUE(
      xla::DynExpr::equal(param_shape.expressions(0), xla::DExpr::Var(1)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(1)));
}

TEST_F(XlaCompilerDynamicSizesTest, ReverseSequencePreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto seq_lens = ops::_Arg(scope.WithOpName("seq_lens"), DT_INT32, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("reverse_sequence", "ReverseSequence")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(seq_lens.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tlen", DT_INT32)
                   .Attr("batch_dim", 0)
                   .Attr("seq_dim", 1)
                   .Finalize(&def));
  absl::Status status;
  Node* reverse_sequence = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(reverse_sequence));
  scope.graph()->AddEdge(input.node(), 0, reverse_sequence, 0);
  scope.graph()->AddEdge(seq_lens.node(), 0, reverse_sequence, 1);

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(reverse_sequence), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {4, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(1), xla::DExpr::Var(2)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {4}, std::vector<xla::DExpr>{xla::DExpr::Var(1)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "reverse_sequence", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(1)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Var(2)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(1)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(1), xla::DExpr::Var(2)));
}

TEST_F(XlaCompilerDynamicSizesTest, UniquePreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("unique", "Unique")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("out_idx", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* unique = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(unique));
  scope.graph()->AddEdge(input.node(), 0, unique, 0);

  auto retval0 =
      ops::_Retval(scope.WithOpName("retval0"), Output(unique, 0), 0);
  auto retval1 =
      ops::_Retval(scope.WithOpName("retval1"), Output(unique, 1), 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7}, std::vector<xla::DExpr>{xla::DExpr::Var(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "unique",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 2);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(3)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[1].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(3)));

  const xla::Shape& values_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  const xla::Shape& indices_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {1});
  EXPECT_TRUE(
      xla::DynExpr::equal(values_shape.expressions(0), xla::DExpr::Var(3)));
  EXPECT_TRUE(
      xla::DynExpr::equal(indices_shape.expressions(0), xla::DExpr::Var(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, DynamicPartitionPreservesPartitionExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto data = ops::_Arg(scope.WithOpName("data"), DT_INT32, 0);
  auto partitions = ops::_Arg(scope.WithOpName("partitions"), DT_INT32, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("dynamic_partition", "DynamicPartition")
                   .Input(data.node()->name(), 0, DT_INT32)
                   .Input(partitions.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("num_partitions", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* dynamic_partition = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(dynamic_partition));
  scope.graph()->AddEdge(data.node(), 0, dynamic_partition, 0);
  scope.graph()->AddEdge(partitions.node(), 0, dynamic_partition, 1);

  auto retval0 = ops::_Retval(scope.WithOpName("retval0"),
                              Output(dynamic_partition, 0), 0);
  auto retval1 = ops::_Retval(scope.WithOpName("retval1"),
                              Output(dynamic_partition, 1), 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6}, std::vector<xla::DExpr>{xla::DExpr::Var(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6}, std::vector<xla::DExpr>{xla::DExpr::Var(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "dynamic_partition", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 2);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[1].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(4)));

  const xla::Shape& result0_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  const xla::Shape& result1_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {1});
  EXPECT_TRUE(
      xla::DynExpr::equal(result0_shape.expressions(0), xla::DExpr::Var(4)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result1_shape.expressions(0), xla::DExpr::Var(4)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       DynamicPartitionBroadcastPreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto data = ops::_Arg(scope.WithOpName("data"), DT_INT32, 0);
  auto partitions = ops::_Arg(scope.WithOpName("partitions"), DT_INT32, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("dynamic_partition", "DynamicPartition")
                   .Input(data.node()->name(), 0, DT_INT32)
                   .Input(partitions.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("num_partitions", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* dynamic_partition = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(dynamic_partition));
  scope.graph()->AddEdge(data.node(), 0, dynamic_partition, 0);
  scope.graph()->AddEdge(partitions.node(), 0, dynamic_partition, 1);

  auto retval0 = ops::_Retval(scope.WithOpName("retval0"),
                              Output(dynamic_partition, 0), 0);
  auto retval1 = ops::_Retval(scope.WithOpName("retval1"),
                              Output(dynamic_partition, 1), 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(40), xla::DExpr::Const(3)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8}, std::vector<xla::DExpr>{xla::DExpr::Var(40)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "dynamic_partition_broadcast",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_TRUE(xla::DynExpr::equal(
        result.outputs[i].shape.get_filled_expression(0), xla::DExpr::Var(40)));
    EXPECT_TRUE(xla::DynExpr::equal(
        result.outputs[i].shape.get_filled_expression(1), xla::DExpr::Const(3)));
    const xla::Shape& out_shape =
        xla::ShapeUtil::GetSubshape(result.xla_output_shape, {i});
    EXPECT_TRUE(
        xla::DynExpr::equal(out_shape.expressions(0), xla::DExpr::Var(40)));
    EXPECT_TRUE(
        xla::DynExpr::equal(out_shape.expressions(1), xla::DExpr::Const(3)));
  }
}

TEST_F(XlaCompilerDynamicSizesTest,
       DenseBincountMatrixPreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto weights = ops::_Arg(scope.WithOpName("weights"), DT_FLOAT, 1);
  auto size = ops::Const(scope.WithOpName("size"), 5);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("dense_bincount", "DenseBincount")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(size.node()->name(), 0, DT_INT32)
                   .Input(weights.node()->name(), 0, DT_FLOAT)
                   .Attr("Tidx", DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("binary_output", false)
                   .Finalize(&def));
  absl::Status status;
  Node* bincount = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, bincount, 0);
  scope.graph()->AddEdge(size.node(), 0, bincount, 1);
  scope.graph()->AddEdge(weights.node(), 0, bincount, 2);
  TF_ASSERT_OK(scope.DoShapeInference(bincount));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(bincount, 0), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {12, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(50), xla::DExpr::Const(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_FLOAT;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {12, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(50), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "dense_bincount", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(50)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));

  const xla::Shape& out_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(xla::DynExpr::equal(out_shape.expressions(0), xla::DExpr::Var(50)));
  EXPECT_TRUE(
      xla::DynExpr::equal(out_shape.expressions(1), xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       DynamicStitchEmptyPreservesTrailingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto data0 = ops::_Arg(scope.WithOpName("data0"), DT_INT32, 0);
  auto data1 = ops::_Arg(scope.WithOpName("data1"), DT_INT32, 1);
  Tensor empty_indices_tensor(DT_INT32, TensorShape({0}));
  auto indices0 = ops::Const(scope.WithOpName("indices0"), empty_indices_tensor);
  auto indices1 = ops::Const(scope.WithOpName("indices1"), empty_indices_tensor);

  NodeDef def;
  std::vector<NodeDefBuilder::NodeOut> indices_inputs = {
      {indices0.node()->name(), 0, DT_INT32},
      {indices1.node()->name(), 0, DT_INT32},
  };
  std::vector<NodeDefBuilder::NodeOut> data_inputs = {
      {data0.node()->name(), 0, DT_INT32},
      {data1.node()->name(), 0, DT_INT32},
  };
  TF_ASSERT_OK(NodeDefBuilder("dynamic_stitch_empty", "DynamicStitch")
                   .Input(indices_inputs)
                   .Input(data_inputs)
                   .Attr("N", 2)
                   .Attr("T", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* dynamic_stitch = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(indices0.node(), 0, dynamic_stitch, 0);
  scope.graph()->AddEdge(indices1.node(), 0, dynamic_stitch, 1);
  scope.graph()->AddEdge(data0.node(), 0, dynamic_stitch, 2);
  scope.graph()->AddEdge(data1.node(), 0, dynamic_stitch, 3);

  auto retval = ops::_Retval(scope.WithOpName("retval"),
                             Output(dynamic_stitch, 0), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {0, 7},
      std::vector<xla::DExpr>{xla::DExpr::Const(0), xla::DExpr::Var(61)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {0, 7},
      std::vector<xla::DExpr>{xla::DExpr::Const(0), xla::DExpr::Var(61)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "dynamic_stitch_empty", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Const(0)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Var(61)));

  const xla::Shape& out_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(out_shape.expressions(0), xla::DExpr::Const(0)));
  EXPECT_TRUE(
      xla::DynExpr::equal(out_shape.expressions(1), xla::DExpr::Var(61)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       TensorListPushBackStackPreservesElementExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto element = ops::_Arg(scope.WithOpName("element"), DT_INT32, 0);
  auto element_shape = ops::Const(scope.WithOpName("element_shape"),
                                  {-1, 3}, {2});
  auto max_num_elements = ops::Const(scope.WithOpName("max_num_elements"), 4);

  NodeDef empty_def;
  TF_ASSERT_OK(NodeDefBuilder("empty_list", "EmptyTensorList")
                   .Input(element_shape.node()->name(), 0, DT_INT32)
                   .Input(max_num_elements.node()->name(), 0, DT_INT32)
                   .Attr("element_dtype", DT_INT32)
                   .Attr("shape_type", DT_INT32)
                   .Finalize(&empty_def));
  absl::Status status;
  Node* empty_list = scope.graph()->AddNode(empty_def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(element_shape.node(), 0, empty_list, 0);
  scope.graph()->AddEdge(max_num_elements.node(), 0, empty_list, 1);
  TF_ASSERT_OK(scope.DoShapeInference(empty_list));

  NodeDef push_def;
  TF_ASSERT_OK(NodeDefBuilder("push_back", "TensorListPushBack")
                   .Input(empty_list->name(), 0, DT_VARIANT)
                   .Input(element.node()->name(), 0, DT_INT32)
                   .Attr("element_dtype", DT_INT32)
                   .Finalize(&push_def));
  Node* push_back = scope.graph()->AddNode(push_def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(empty_list, 0, push_back, 0);
  scope.graph()->AddEdge(element.node(), 0, push_back, 1);
  TF_ASSERT_OK(scope.DoShapeInference(push_back));

  NodeDef stack_def;
  TF_ASSERT_OK(NodeDefBuilder("stack", "TensorListStack")
                   .Input(push_back->name(), 0, DT_VARIANT)
                   .Input(element_shape.node()->name(), 0, DT_INT32)
                   .Attr("element_dtype", DT_INT32)
                   .Attr("num_elements", 4)
                   .Finalize(&stack_def));
  Node* stack = scope.graph()->AddNode(stack_def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(push_back, 0, stack, 0);
  scope.graph()->AddEdge(element_shape.node(), 0, stack, 1);
  TF_ASSERT_OK(scope.DoShapeInference(stack));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(stack, 0), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(70), xla::DExpr::Const(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "tensor_list_stack", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Var(70)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, ShapeThenReshapePreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto shape_source = ops::_Arg(scope.WithOpName("shape_source"), DT_INT32, 0);
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 1);
  auto shape = ops::Shape(scope.WithOpName("shape"), shape_source);
  auto reshaped = ops::Reshape(scope.WithOpName("reshape"), input, shape);
  auto retval = ops::_Retval(scope.WithOpName("retval"), reshaped, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 7},
      std::vector<xla::DExpr>{xla::DExpr::Var(41), xla::DExpr::Const(7)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 7},
      std::vector<xla::DExpr>{xla::DExpr::Var(41), xla::DExpr::Const(7)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "shape_then_reshape", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(41)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(7)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(41)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(1), xla::DExpr::Const(7)));
}

TEST_F(XlaCompilerDynamicSizesTest, ZerosLikePreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto zeros = ops::ZerosLike(scope.WithOpName("zeros_like"), input);
  auto retval = ops::_Retval(scope.WithOpName("retval"), zeros, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {9, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(43), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "zeros_like_exprs", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(43)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, OnesLikePreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto ones = ops::OnesLike(scope.WithOpName("ones_like"), input);
  auto retval = ops::_Retval(scope.WithOpName("retval"), ones, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {9, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(44), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "ones_like_exprs", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(44)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, MatrixDiagPreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("matrix_diag", "MatrixDiag")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* matrix_diag = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(matrix_diag));
  scope.graph()->AddEdge(input.node(), 0, matrix_diag, 0);

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(matrix_diag), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(45), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "matrix_diag_exprs", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(45)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, WhereBuildsDynamicIndexMatrixShape) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_BOOL, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("where", "Where")
                   .Input(input.node()->name(), 0, DT_BOOL)
                   .Finalize(&def));
  absl::Status status;
  Node* where = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(where));
  scope.graph()->AddEdge(input.node(), 0, where, 0);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(where), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_BOOL;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::PRED, {8, 4, 6},
      std::vector<xla::DExpr>{xla::DExpr::Var(46), xla::DExpr::Const(4),
                              xla::DExpr::Var(47)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "where",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_EQ(result_shape.dimensions_size(), 2);
  EXPECT_EQ(result_shape.dimensions(1), 3);
  // The number of true elements is data-dependent and cannot reuse an input
  // dimension expression. SetDimensionSize records the runtime length while
  // the physical dimension remains the static upper bound.
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
  EXPECT_TRUE(result_shape.expressions(0).is_unknown());
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(1), xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, DiagDuplicatesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("diag", "Diag")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* diag = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(diag));
  scope.graph()->AddEdge(input.node(), 0, diag, 0);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(diag), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5}, std::vector<xla::DExpr>{xla::DExpr::Var(42)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "diag",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(42)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Var(42)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(42)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(1), xla::DExpr::Var(42)));
}

TEST_F(XlaCompilerDynamicSizesTest, InTopKPreservesBatchExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto predictions = ops::_Arg(scope.WithOpName("predictions"), DT_FLOAT, 0);
  auto targets = ops::_Arg(scope.WithOpName("targets"), DT_INT32, 1);
  auto k = ops::Const<int32>(scope.WithOpName("k"), 3, {});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("in_topk", "InTopKV2")
                   .Input(predictions.node()->name(), 0, DT_FLOAT)
                   .Input(targets.node()->name(), 0, DT_INT32)
                   .Input(k.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* in_topk = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(predictions.node(), 0, in_topk, 0);
  scope.graph()->AddEdge(targets.node(), 0, in_topk, 1);
  scope.graph()->AddEdge(k.node(), 0, in_topk, 2);
  TF_ASSERT_OK(scope.DoShapeInference(in_topk));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(in_topk), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {5, 7},
      std::vector<xla::DExpr>{xla::DExpr::Var(43), xla::DExpr::Const(7)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5}, std::vector<xla::DExpr>{xla::DExpr::Var(43)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "in_topk",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(43)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(43)));
}

TEST_F(XlaCompilerDynamicSizesTest, ReshapeCollapsePreservesSymbolicExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {96}, {1});
  auto reshaped = ops::Reshape(scope.WithOpName("reshape"), input, shape);
  auto retval = ops::_Retval(scope.WithOpName("retval"), reshaped, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {3, 4, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(5), xla::DExpr::Const(4),
                              xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "reshape_collapse", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(5) * xla::DExpr::Const(32)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(0), expected));
}

TEST_F(XlaCompilerDynamicSizesTest, ReshapeSplitPreservesSymbolicExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {5, 16}, {2});
  auto reshaped = ops::Reshape(scope.WithOpName("reshape"), input, shape);
  auto retval = ops::_Retval(scope.WithOpName("retval"), reshaped, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {10, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(6), xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "reshape_split", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(6) / xla::DExpr::Const(2)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(0), expected));
}

TEST_F(XlaCompilerDynamicSizesTest, ReshapeSplitAndCollapsePreservesSymbolicExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {4, 64}, {2});
  auto reshaped = ops::Reshape(scope.WithOpName("reshape"), input, shape);
  auto retval = ops::_Retval(scope.WithOpName("retval"), reshaped, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 8, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(7), xla::DExpr::Const(8),
                              xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "reshape_split_collapse",
                                     std::move(graph), args, &result));

  xla::DExpr expected =
      (xla::DExpr::Var(7) / xla::DExpr::Const(2)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(0), expected));
}

TEST_F(XlaCompilerDynamicSizesTest, GatherV2PreservesUngatheredExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto params = ops::_Arg(scope.WithOpName("params"), DT_INT32, 0);
  auto indices = ops::Const<int32>(scope.WithOpName("indices"), {0, 2, 4}, {3});
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 1, {});
  auto gathered =
      ops::GatherV2(scope.WithOpName("gather"), params, indices, axis);
  auto retval = ops::_Retval(scope.WithOpName("retval"), gathered, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {9, 7},
      std::vector<xla::DExpr>{xla::DExpr::Var(8), xla::DExpr::Const(7)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "gather_preserve", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(8)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(8)));
}

TEST_F(XlaCompilerDynamicSizesTest, TransposePermutesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto perm = ops::Const<int32>(scope.WithOpName("perm"), {1, 0}, {2});
  auto transposed = ops::Transpose(scope.WithOpName("transpose"), input, perm);
  auto retval = ops::_Retval(scope.WithOpName("retval"), transposed, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5, 7},
      std::vector<xla::DExpr>{xla::DExpr::Var(9), xla::DExpr::Var(10)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "transpose_exprs", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(10)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Var(9)));
}

TEST_F(XlaCompilerDynamicSizesTest, ExpandDimsInsertsUnitExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto dim = ops::Const<int32>(scope.WithOpName("dim"), 1, {});
  auto expanded = ops::ExpandDims(scope.WithOpName("expand"), input, dim);
  auto retval = ops::_Retval(scope.WithOpName("retval"), expanded, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(11), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "expand_dims_exprs", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(11)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(1)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, SqueezeRemovesUnitExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("squeeze", "Squeeze")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("squeeze_dims", {1})
                   .Finalize(&def));
  absl::Status status;
  Node* squeeze = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, squeeze, 0);
  TF_ASSERT_OK(scope.DoShapeInference(squeeze));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(squeeze), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 1, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(12), xla::DExpr::Const(1),
                              xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "squeeze_exprs", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(12)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, SplitPreservesAndDividesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto split_dim = ops::Const<int32>(scope.WithOpName("split_dim"), 0, {});
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto split = ops::Split(scope.WithOpName("split"), split_dim, input, 2);
  auto retval0 = ops::_Retval(scope.WithOpName("retval0"), split.output[0], 0);
  auto retval1 = ops::_Retval(scope.WithOpName("retval1"), split.output[1], 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(13), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "split_exprs", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(13) / xla::DExpr::Const(2)).simplify();
  ASSERT_EQ(result.outputs.size(), 2);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[1].shape.get_filled_expression(
                                      0),
                                  expected));
}

TEST_F(XlaCompilerDynamicSizesTest, TileScalesExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto multiples =
      ops::Const<int32>(scope.WithOpName("multiples"), {3, 1}, {2});
  auto tiled = ops::Tile(scope.WithOpName("tile"), input, multiples);
  auto retval = ops::_Retval(scope.WithOpName("retval"), tiled, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {4, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(14), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "tile_exprs", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(14) * xla::DExpr::Const(3)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, PackInsertsAxisAndPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input0 = ops::_Arg(scope.WithOpName("input0"), DT_INT32, 0);
  auto input1 = ops::_Arg(scope.WithOpName("input1"), DT_INT32, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("pack", "Pack")
                   .Input({NodeDefBuilder::NodeOut(input0.node()->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(input1.node()->name(), 0,
                                                   DT_INT32)})
                   .Attr("T", DT_INT32)
                   .Attr("N", 2)
                   .Attr("axis", 1)
                   .Finalize(&def));
  absl::Status status;
  Node* pack = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input0.node(), 0, pack, 0);
  scope.graph()->AddEdge(input1.node(), 0, pack, 1);
  TF_ASSERT_OK(scope.DoShapeInference(pack));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(pack), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6}, std::vector<xla::DExpr>{xla::DExpr::Var(15)});
  args[1] = args[0];

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "pack",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(15)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(2)));
}

TEST_F(XlaCompilerDynamicSizesTest, UnpackRemovesAxisAndPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("unpack", "Unpack")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("num", 3)
                   .Attr("axis", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* unpack = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, unpack, 0);
  TF_ASSERT_OK(scope.DoShapeInference(unpack));

  auto retval0 = ops::_Retval(scope.WithOpName("retval0"), Output(unpack, 0), 0);
  auto retval1 = ops::_Retval(scope.WithOpName("retval1"), Output(unpack, 1), 1);
  auto retval2 = ops::_Retval(scope.WithOpName("retval2"), Output(unpack, 2), 2);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 4, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(16), xla::DExpr::Const(4),
                              xla::DExpr::Const(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "unpack",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(xla::DynExpr::equal(
        result.outputs[i].shape.get_filled_expression(0), xla::DExpr::Var(16)));
    EXPECT_TRUE(xla::DynExpr::equal(result.outputs[i].shape.get_filled_expression(
                                        1),
                                    xla::DExpr::Const(4)));
  }
}

TEST_F(XlaCompilerDynamicSizesTest, ConcatV2AddsLeadingExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 0, {});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("concat", "ConcatV2")
                   .Input({NodeDefBuilder::NodeOut(lhs.node()->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(rhs.node()->name(), 0,
                                                   DT_INT32)})
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Attr("N", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* concat = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(lhs.node(), 0, concat, 0);
  scope.graph()->AddEdge(rhs.node(), 0, concat, 1);
  scope.graph()->AddEdge(axis.node(), 0, concat, 2);
  TF_ASSERT_OK(scope.DoShapeInference(concat));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(concat), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(17), xla::DExpr::Const(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(18), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "concat",
                                     std::move(graph), args, &result));

  xla::DExpr expected =
      (xla::DExpr::Var(17) + xla::DExpr::Var(18)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, ConcatAddsLeadingExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 0, {});
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("concat", "Concat")
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Input({NodeDefBuilder::NodeOut(lhs.node()->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(rhs.node()->name(), 0,
                                                   DT_INT32)})
                   .Attr("T", DT_INT32)
                   .Attr("N", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* concat = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(axis.node(), 0, concat, 0);
  scope.graph()->AddEdge(lhs.node(), 0, concat, 1);
  scope.graph()->AddEdge(rhs.node(), 0, concat, 2);
  TF_ASSERT_OK(scope.DoShapeInference(concat));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(concat), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(22), xla::DExpr::Const(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(23), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "concat_legacy", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(22) + xla::DExpr::Var(23)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, ConcatAddsThreeLeadingExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto mid = ops::_Arg(scope.WithOpName("mid"), DT_INT32, 1);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 2);
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 0, {});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("concat0", "ConcatV2")
                   .Input({NodeDefBuilder::NodeOut(lhs.node()->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(mid.node()->name(), 0,
                                                   DT_INT32)})
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Attr("N", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* concat0 = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(lhs.node(), 0, concat0, 0);
  scope.graph()->AddEdge(mid.node(), 0, concat0, 1);
  scope.graph()->AddEdge(axis.node(), 0, concat0, 2);
  TF_ASSERT_OK(scope.DoShapeInference(concat0));

  NodeDef def1;
  TF_ASSERT_OK(NodeDefBuilder("concat1", "ConcatV2")
                   .Input({NodeDefBuilder::NodeOut(concat0->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(rhs.node()->name(), 0,
                                                   DT_INT32)})
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Attr("N", 2)
                   .Finalize(&def1));
  Node* concat1 = scope.graph()->AddNode(def1, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(concat0, 0, concat1, 0);
  scope.graph()->AddEdge(rhs.node(), 0, concat1, 1);
  scope.graph()->AddEdge(axis.node(), 0, concat1, 2);
  TF_ASSERT_OK(scope.DoShapeInference(concat1));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(concat1), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(3);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {3, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(31), xla::DExpr::Const(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {5, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(32), xla::DExpr::Const(4)});
  args[2].kind = XlaCompiler::Argument::kParameter;
  args[2].type = DT_INT32;
  args[2].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(33), xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "concat_three", std::move(graph), args,
                                     &result));

  xla::DExpr expected =
      (xla::DExpr::Var(31) + xla::DExpr::Var(32) + xla::DExpr::Var(33))
          .simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  expected));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(0), expected));
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, ConcatV2PreservesLeadingExpressionOnInnerAxis) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 1, {});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("concat", "ConcatV2")
                   .Input({NodeDefBuilder::NodeOut(lhs.node()->name(), 0,
                                                   DT_INT32),
                           NodeDefBuilder::NodeOut(rhs.node()->name(), 0,
                                                   DT_INT32)})
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Attr("N", 2)
                   .Finalize(&def));
  absl::Status status;
  Node* concat = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(lhs.node(), 0, concat, 0);
  scope.graph()->AddEdge(rhs.node(), 0, concat, 1);
  scope.graph()->AddEdge(axis.node(), 0, concat, 2);
  TF_ASSERT_OK(scope.DoShapeInference(concat));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(concat), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 4},
      std::vector<xla::DExpr>{xla::DExpr::Var(24), xla::DExpr::Const(4)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(24), xla::DExpr::Const(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "concat_inner", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(24)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(7)));
}

TEST_F(XlaCompilerDynamicSizesTest, AddPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto sum = ops::Add(scope.WithOpName("add"), lhs, rhs);
  auto retval = ops::_Retval(scope.WithOpName("retval"), sum, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(44), xla::DExpr::Const(5)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(44), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(44)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       AddSameRankBroadcastPreservesMappedExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto sum = ops::Add(scope.WithOpName("add"), lhs, rhs);
  auto retval = ops::_Retval(scope.WithOpName("retval"), sum, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 1, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(45), xla::DExpr::Const(1),
                              xla::DExpr::Const(3)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 4, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(45), xla::DExpr::Const(4),
                              xla::DExpr::Const(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "add_broadcast", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(45)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, AddDegenerateBroadcastPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto sum = ops::Add(scope.WithOpName("add"), lhs, rhs);
  auto retval = ops::_Retval(scope.WithOpName("retval"), sum, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {1, 5},
      std::vector<xla::DExpr>{xla::DExpr::Const(1), xla::DExpr::Const(5)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(46), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "add_degenerate", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(46)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       MulSameRankBroadcastPreservesMappedExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_INT32, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_INT32, 1);
  auto product = ops::Mul(scope.WithOpName("mul"), lhs, rhs);
  auto retval = ops::_Retval(scope.WithOpName("retval"), product, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 1, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(47), xla::DExpr::Const(1),
                              xla::DExpr::Const(3)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 4, 3},
      std::vector<xla::DExpr>{xla::DExpr::Var(47), xla::DExpr::Const(4),
                              xla::DExpr::Const(3)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "mul_broadcast", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(47)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, ReverseV2PreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), {1}, {1});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("reverse", "ReverseV2")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tidx", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* reverse = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, reverse, 0);
  scope.graph()->AddEdge(axis.node(), 0, reverse, 1);
  TF_ASSERT_OK(scope.DoShapeInference(reverse));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(reverse), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {8, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(19), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "reverse_v2",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(19)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, BatchMatMulPreservesBatchExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_FLOAT, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_FLOAT, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("batch_matmul", "BatchMatMul")
                   .Input(lhs.node()->name(), 0, DT_FLOAT)
                   .Input(rhs.node()->name(), 0, DT_FLOAT)
                   .Attr("T", DT_FLOAT)
                   .Attr("adj_x", false)
                   .Attr("adj_y", false)
                   .Finalize(&def));
  absl::Status status;
  Node* batch_matmul = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(lhs.node(), 0, batch_matmul, 0);
  scope.graph()->AddEdge(rhs.node(), 0, batch_matmul, 1);
  TF_ASSERT_OK(scope.DoShapeInference(batch_matmul));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(batch_matmul), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 4, 6},
      std::vector<xla::DExpr>{xla::DExpr::Var(25), xla::DExpr::Const(4),
                              xla::DExpr::Const(6)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_FLOAT;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 6, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(25), xla::DExpr::Const(6),
                              xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "batch_matmul", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(25)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, BatchMatMulV2BroadcastsBatchExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto lhs = ops::_Arg(scope.WithOpName("lhs"), DT_FLOAT, 0);
  auto rhs = ops::_Arg(scope.WithOpName("rhs"), DT_FLOAT, 1);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("batch_matmul_v2", "BatchMatMulV2")
                   .Input(lhs.node()->name(), 0, DT_FLOAT)
                   .Input(rhs.node()->name(), 0, DT_FLOAT)
                   .Attr("T", DT_FLOAT)
                   .Attr("adj_x", false)
                   .Attr("adj_y", false)
                   .Finalize(&def));
  absl::Status status;
  Node* batch_matmul = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(lhs.node(), 0, batch_matmul, 0);
  scope.graph()->AddEdge(rhs.node(), 0, batch_matmul, 1);
  TF_ASSERT_OK(scope.DoShapeInference(batch_matmul));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(batch_matmul), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {1, 4, 6},
      std::vector<xla::DExpr>{xla::DExpr::Const(1), xla::DExpr::Const(4),
                              xla::DExpr::Const(6)});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_FLOAT;
  args[1].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 6, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(26), xla::DExpr::Const(6),
                              xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "batch_matmul_v2", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(26)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, SlicePreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 2}, {2});
  auto size = ops::Const<int32>(scope.WithOpName("size"), {-1, 3}, {2});
  auto sliced = ops::Slice(scope.WithOpName("slice"), input, begin, size);
  auto retval = ops::_Retval(scope.WithOpName("retval"), sliced, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(20), xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "slice",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(20)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, SliceSubtractsFromLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {2, 1}, {2});
  auto size = ops::Const<int32>(scope.WithOpName("size"), {-1, 3}, {2});
  auto sliced = ops::Slice(scope.WithOpName("slice"), input, begin, size);
  auto retval = ops::_Retval(scope.WithOpName("retval"), sliced, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {9, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(27), xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "slice_subtract", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  (xla::DExpr::Var(27) - 2).simplify()));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       StridedSlicePreservesLeadingExpressionOnInnerAxis) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 1}, {2});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 7}, {2});
  auto strides = ops::Const<int32>(scope.WithOpName("strides"), {1, 2}, {2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 1)
                   .Attr("end_mask", 1)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0)
                   .Attr("shrink_axis_mask", 0)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(40), xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_inner", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(40)));
  EXPECT_EQ(result.outputs[0].shape.dim_size(1), 3);
}

TEST_F(XlaCompilerDynamicSizesTest, StridedSliceScalesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 0}, {2});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 4}, {2});
  auto strides = ops::Const<int32>(scope.WithOpName("strides"), {2, 1}, {2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 1)
                   .Attr("end_mask", 1)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0)
                   .Attr("shrink_axis_mask", 0)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 4},
      std::vector<xla::DExpr>{
          ((xla::DExpr::Const(2) * xla::DExpr::Var(41)) -
           xla::DExpr::Const(1))
              .simplify(),
          xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_leading", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(
      result.outputs[0].shape.get_filled_expression(0), xla::DExpr::Var(41)));
  EXPECT_EQ(result.outputs[0].shape.dim_size(0), 4);
  EXPECT_EQ(result.outputs[0].shape.dim_size(1), 4);
}

TEST_F(XlaCompilerDynamicSizesTest, StridedSliceNewAxisInsertsUnitExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 0}, {2});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 0}, {2});
  auto strides = ops::Const<int32>(scope.WithOpName("strides"), {1, 1}, {2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 0x1)
                   .Attr("end_mask", 0x1)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0x2)
                   .Attr("shrink_axis_mask", 0)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(42), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_new_axis",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(42)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(1)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       StridedSliceShrinkAxisPreservesRemainingExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 2, 0}, {3});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 2, 5}, {3});
  auto strides = ops::Const<int32>(scope.WithOpName("strides"), {1, 1, 1}, {3});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 0x3)
                   .Attr("end_mask", 0x3)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0)
                   .Attr("shrink_axis_mask", 0x2)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 5, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(43), xla::DExpr::Const(5),
                              xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_shrink", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(43)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       StridedSliceNegativeStridePreservesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 0}, {2});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 0}, {2});
  auto strides =
      ops::Const<int32>(scope.WithOpName("strides"), {-1, 1}, {2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 0x3)
                   .Attr("end_mask", 0x3)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0)
                   .Attr("shrink_axis_mask", 0)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(44), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_negative",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(44)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest,
       StridedSliceNegativeStrideTwoScalesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto begin = ops::Const<int32>(scope.WithOpName("begin"), {0, 0}, {2});
  auto end = ops::Const<int32>(scope.WithOpName("end"), {0, 0}, {2});
  auto strides =
      ops::Const<int32>(scope.WithOpName("strides"), {-2, 1}, {2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("strided_slice", "StridedSlice")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 0, DT_INT32)
                   .Input(end.node()->name(), 0, DT_INT32)
                   .Input(strides.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Index", DT_INT32)
                   .Attr("begin_mask", 0x3)
                   .Attr("end_mask", 0x3)
                   .Attr("ellipsis_mask", 0)
                   .Attr("new_axis_mask", 0)
                   .Attr("shrink_axis_mask", 0)
                   .Finalize(&def));
  absl::Status status;
  Node* strided_slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, strided_slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, strided_slice, 1);
  scope.graph()->AddEdge(end.node(), 0, strided_slice, 2);
  scope.graph()->AddEdge(strides.node(), 0, strided_slice, 3);
  TF_ASSERT_OK(scope.DoShapeInference(strided_slice));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(strided_slice), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(45), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "strided_slice_negative_two",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(
      result.outputs[0].shape.get_filled_expression(0),
      ((xla::DExpr::Var(45) + xla::DExpr::Const(1)) / xla::DExpr::Const(2))
          .simplify()));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, PadAddsToLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto paddings =
      ops::Const<int32>(scope.WithOpName("paddings"), {1, 2, 0, 0}, {2, 2});
  auto padded = ops::Pad(scope.WithOpName("pad"), input, paddings);
  auto retval = ops::_Retval(scope.WithOpName("retval"), padded, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {7, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(28), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "pad",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  (xla::DExpr::Var(28) + 3).simplify()));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

TEST_F(XlaCompilerDynamicSizesTest, SpaceToBatchNDScalesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_FLOAT, 0);
  auto block_shape =
      ops::Const<int32>(scope.WithOpName("block_shape"), {2}, {1});
  auto paddings =
      ops::Const<int32>(scope.WithOpName("paddings"), {0, 0}, {1, 2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("space_to_batch", "SpaceToBatchND")
                   .Input(input.node()->name(), 0, DT_FLOAT)
                   .Input(block_shape.node()->name(), 0, DT_INT32)
                   .Input(paddings.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("Tblock_shape", DT_INT32)
                   .Attr("Tpaddings", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* space_to_batch = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, space_to_batch, 0);
  scope.graph()->AddEdge(block_shape.node(), 0, space_to_batch, 1);
  scope.graph()->AddEdge(paddings.node(), 0, space_to_batch, 2);
  TF_ASSERT_OK(scope.DoShapeInference(space_to_batch));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(space_to_batch), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {4, 8},
      std::vector<xla::DExpr>{xla::DExpr::Var(29), xla::DExpr::Const(8)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "space_to_batch", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  (xla::DExpr::Const(2) * xla::DExpr::Var(29))
                                      .simplify()));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
}

TEST_F(XlaCompilerDynamicSizesTest, BatchToSpaceNDDividesLeadingExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_FLOAT, 0);
  auto block_shape =
      ops::Const<int32>(scope.WithOpName("block_shape"), {2}, {1});
  auto crops = ops::Const<int32>(scope.WithOpName("crops"), {0, 0}, {1, 2});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("batch_to_space", "BatchToSpaceND")
                   .Input(input.node()->name(), 0, DT_FLOAT)
                   .Input(block_shape.node()->name(), 0, DT_INT32)
                   .Input(crops.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_FLOAT)
                   .Attr("Tblock_shape", DT_INT32)
                   .Attr("Tcrops", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* batch_to_space = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, batch_to_space, 0);
  scope.graph()->AddEdge(block_shape.node(), 0, batch_to_space, 1);
  scope.graph()->AddEdge(crops.node(), 0, batch_to_space, 2);
  TF_ASSERT_OK(scope.DoShapeInference(batch_to_space));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(batch_to_space), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {8, 4},
      std::vector<xla::DExpr>{(xla::DExpr::Const(2) * xla::DExpr::Var(30))
                                  .simplify(),
                              xla::DExpr::Const(4)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "batch_to_space", std::move(graph), args,
                                     &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(30)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(8)));
}

TEST_F(XlaCompilerDynamicSizesTest, SpaceToDepthScalesDepthExpression) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_FLOAT, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("space_to_depth", "SpaceToDepth")
                   .Input(input.node()->name(), 0, DT_FLOAT)
                   .Attr("T", DT_FLOAT)
                   .Attr("block_size", 2)
                   .Attr("data_format", "NHWC")
                   .Finalize(&def));
  absl::Status status;
  Node* space_to_depth = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, space_to_depth, 0);
  TF_ASSERT_OK(scope.DoShapeInference(space_to_depth));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(space_to_depth), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {5, 8, 8, 3},
      std::vector<xla::DExpr>{xla::DExpr::Const(5), xla::DExpr::Const(8),
                              xla::DExpr::Const(8), xla::DExpr::Var(35)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "space_to_depth", std::move(graph), args,
                                     &result));

  xla::DExpr expected_depth =
      (xla::DExpr::Const(4) * xla::DExpr::Var(35)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Const(5)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      3),
                                  expected_depth));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Const(5)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(1), xla::DExpr::Const(4)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(2), xla::DExpr::Const(4)));
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(3), expected_depth));
}

TEST_F(XlaCompilerDynamicSizesTest, DepthToSpaceScalesSpatialExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_FLOAT, 0);

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("depth_to_space", "DepthToSpace")
                   .Input(input.node()->name(), 0, DT_FLOAT)
                   .Attr("T", DT_FLOAT)
                   .Attr("block_size", 2)
                   .Attr("data_format", "NHWC")
                   .Finalize(&def));
  absl::Status status;
  Node* depth_to_space = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, depth_to_space, 0);
  TF_ASSERT_OK(scope.DoShapeInference(depth_to_space));

  auto retval =
      ops::_Retval(scope.WithOpName("retval"), Output(depth_to_space), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_FLOAT;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::F32, {5, 4, 4, 12},
      std::vector<xla::DExpr>{xla::DExpr::Const(5), xla::DExpr::Var(37),
                              xla::DExpr::Const(4), xla::DExpr::Const(12)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "depth_to_space", std::move(graph), args,
                                     &result));

  xla::DExpr expected_height =
      (xla::DExpr::Const(2) * xla::DExpr::Var(37)).simplify();
  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Const(5)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  expected_height));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      2),
                                  xla::DExpr::Const(8)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      3),
                                  xla::DExpr::Const(3)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Const(5)));
  EXPECT_TRUE(xla::DynExpr::equal(result_shape.expressions(1), expected_height));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(2), xla::DExpr::Const(8)));
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(3), xla::DExpr::Const(3)));
}

TEST_F(XlaCompilerDynamicSizesTest, RollPreservesExpressions) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto input = ops::_Arg(scope.WithOpName("input"), DT_INT32, 0);
  auto shift = ops::Const<int32>(scope.WithOpName("shift"), 2, {});
  auto axis = ops::Const<int32>(scope.WithOpName("axis"), 1, {});

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("roll", "Roll")
                   .Input(input.node()->name(), 0, DT_INT32)
                   .Input(shift.node()->name(), 0, DT_INT32)
                   .Input(axis.node()->name(), 0, DT_INT32)
                   .Attr("T", DT_INT32)
                   .Attr("Tshift", DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* roll = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  scope.graph()->AddEdge(input.node(), 0, roll, 0);
  scope.graph()->AddEdge(shift.node(), 0, roll, 1);
  scope.graph()->AddEdge(axis.node(), 0, roll, 2);
  TF_ASSERT_OK(scope.DoShapeInference(roll));

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(roll), 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {6, 5},
      std::vector<xla::DExpr>{xla::DExpr::Var(21), xla::DExpr::Const(5)});

  XlaCompiler compiler(DefaultOptions());
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "roll",
                                     std::move(graph), args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(21)));
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      1),
                                  xla::DExpr::Const(5)));
}

// Tests compilation of a graph where the _Retval node is not necessarily last
// amongst the graph nodes in construction order, and always_return_tuple is
// false. Regression test for bug where the wrong value was returned.
TEST_F(XlaCompilerTest, OutOfOrderGraph) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  // The _Retval node is not last in construction order.
  auto d = ops::_Retval(scope.WithOpName("D"), a, 0);
  auto c = ops::Add(scope.WithOpName("C"), a, b);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompileOptions compile_options;
  compile_options.always_return_tuple = false;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  // Tests that the generated computation works.
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  EXPECT_TRUE(xla::LiteralTestUtil::Equal(param0_literal, actual_literal));
}

// Tests that the compiler can correctly propagate the layout assigned by
// shape_representation_fn_ to resource returns that have not been written to.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForUnwrittenResource) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType dt, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  options.shape_determination_fns = shape_determination_fns;
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = true;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S32, {2, 3}, {0, 1});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed}));
}

// Tests that the compiler can correctly propagate fast mem attribute for input
// resource variable.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForFastMemVar) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});
  args[0].fast_mem = true;

  auto options = DefaultOptions();
  int fast_mem_arg_count = 0;
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [&fast_mem_arg_count](
          const TensorShape& shape, DataType dt, bool use_fast_memory,
          XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    if (use_fast_memory) {
      fast_mem_arg_count++;
    }
    return xla_shape;
  };
  options.shape_determination_fns = shape_determination_fns;
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = true;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));
  // Count 2: one for argument, one for the return value.
  EXPECT_EQ(fast_mem_arg_count, 2);
}

// Tests that the compiler can correctly propagate the layout assigned by
// shape_representation_fn_ to return types.
TEST_F(XlaCompilerTest, HonorShapeRepresentationFnForRetVal) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType dt, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dt, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  options.shape_determination_fns = shape_determination_fns;
  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S32, {2, 3}, {0, 1});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
  EXPECT_EQ(result.computation->GetProgramShape().value().result(),
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
}

// The layout of resource variable shouldn't change after transpose
TEST_F(XlaCompilerTest, TransposeVariables) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto transposed_read = ops::Transpose(scope, read, {1, 0});
  auto reshape = ops::Reshape(scope, transposed_read, {2, 3});
  auto d = ops::_Retval(scope.WithOpName("D"), reshape, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 3});
  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "transpose",
                                     std::move(graph), args, &result));
  xla::Shape transposed =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S32, {2, 3}, {1, 0});
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape({transposed, transposed}));
}

// Unranked fake param returns a 0 shaped tensor.
TEST_F(XlaCompilerTest, UnrankedFakeParam) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  PartialTensorShape shape;
  auto a = ops::FakeParam(scope, DT_INT32, shape);
  auto ret = ops::_Retval(scope.WithOpName("D"), a, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "compile",
                                     std::move(graph), {}, &result));
  // Check that the return shapes are correctly tranposed.
  EXPECT_EQ(result.xla_output_shape,
            xla::ShapeUtil::MakeTupleShape(
                {xla::ShapeUtil::MakeShape(xla::S32, {0})}));
}

// Tests that the compiler doesn't reorder the parameters.
TEST_F(XlaCompilerTest, MixedOrderArguments) {
  for (bool swap_order : {false, true}) {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto var =
        ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, swap_order ? 0 : 1);
    auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, swap_order ? 1 : 0);
    // Adds an identity op around the resource to make sure identity ops
    // propagate resources correctly.
    auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
    auto write = ops::AssignAddVariableOp(scope, identity, a);
    auto read = ops::ReadVariableOp(
        scope.WithControlDependencies(std::vector<Operation>{write}), var,
        DT_INT32);
    auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
    auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_ASSERT_OK(scope.ToGraph(graph.get()));

    // Builds a description of the arguments.
    std::vector<XlaCompiler::Argument> args(2);
    args[0].kind = XlaCompiler::Argument::kParameter;
    args[0].type = DT_INT32;
    args[0].shape = TensorShape({2});
    args[1].kind = XlaCompiler::Argument::kResource;
    args[1].resource_kind = XlaResource::kVariable;
    args[1].initialized = true;
    args[1].type = DT_INT32;
    args[1].shape = TensorShape({2});

    if (swap_order) {
      // Even after swapping arguments, the compiler should maintain the new
      // ordering of parameters.
      std::swap(args[0], args[1]);
    }
    // Compiles the graph.
    XlaCompiler compiler(DefaultOptions());

    XlaCompiler::CompileOptions compile_options;
    compile_options.always_return_tuple = false;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                       args, &result));

    EXPECT_THAT(result.input_mapping, ::testing::ElementsAre(0, 1));
  }
}

TEST_F(XlaCompilerTest, HasSaneErrorOnNonCompileTimeConstantInputToReshape) {
  // Builds a graph that adds reshapes a tensor, but with the shape not
  // statically known.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Arg(scope.WithOpName("B"), DT_INT32, 1);
  auto c = ops::Reshape(scope.WithOpName("C"), a, b);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kParameter;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  absl::Status status =
      compiler.CompileGraph(XlaCompiler::CompileOptions(), "reshape",
                            std::move(graph), args, &result);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "depends on a parameter"))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "{{node C}}"))
      << status.message();
  EXPECT_TRUE(
      absl::StrContains(status.message(), "must be a compile-time constant"))
      << status.message();
}

// Tests handling of compile-time constant outputs.
TEST_F(XlaCompilerTest, ConstantOutputs) {
  // Builds a graph with one compile-time constant output and one data-dependent
  // output, i.e.,
  // func(a) { b=7; c=-a; return b, c; }
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::Const<int32>(scope.WithOpName("B"), 7);
  auto c = ops::Neg(scope.WithOpName("C"), a);
  auto d = ops::_Retval(scope.WithOpName("D"), b, 0);
  auto e = ops::_Retval(scope.WithOpName("E"), c, 1);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  XlaCompiler::Options options = DefaultOptions();
  XlaCompiler compiler(options);

  {
    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());

    XlaCompiler::CompileOptions compile_options;
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                       std::move(graph_copy), args, &result));

    ASSERT_EQ(2, result.outputs.size());
    EXPECT_FALSE(result.outputs[0].is_constant);
    EXPECT_FALSE(result.outputs[1].is_constant);

    // Tests that the generated computation works.
    xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
    std::unique_ptr<xla::GlobalData> param0_data =
        client_->TransferToServer(param0_literal).value();

    std::unique_ptr<xla::GlobalData> actual =
        client_->Execute(*result.computation, {param0_data.get()}).value();
    xla::Literal actual_literal = client_->Transfer(*actual).value();

    xla::Literal expected0 = xla::LiteralUtil::CreateR0<int32>(7);
    xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({-7, -42});
    xla::Literal expected =
        xla::LiteralUtil::MakeTuple({&expected0, &expected1});
    EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected, actual_literal));
  }
}

TEST_F(XlaCompilerTest, ConstantOutputsOfFunctionalNode) {
  // Define a function with one compile-time constant output and one
  // data-dependent output.
  // @function.Defun(noinline=True)
  // foo(a) {b=7; return b, a; }
  const Tensor seven = test::AsScalar<int>(7);
  FunctionDef fdef = FunctionDefHelper::Create(
      "foo", {"a_0:int32"}, {"const:int32", "a:int32"}, {},
      {
          {{"Const"}, "Const", {}, {{"dtype", DT_INT32}, {"value", seven}}},
      },
      {{"a", "a_0"}, {"const", "Const:output:0"}});
  (*fdef.mutable_attr())["_noinline"].set_b(true);
  FunctionDefLibrary fdef_lib;
  *(fdef_lib.add_function()) = fdef;
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));
    auto arg = ops::_Arg(scope.WithOpName("input_arg"), DT_INT32, 0);
    NodeDef foo;
    foo.set_name("foo");
    foo.set_op("foo");
    *foo.add_input() = "input_arg";
    absl::Status status;
    scope.graph()->AddNode(foo, &status);
    TF_ASSERT_OK(status);
    NodeDef retval_1;
    retval_1.set_name("retval_0");
    retval_1.set_op(FunctionLibraryDefinition::kRetOp);
    *retval_1.add_input() = "foo";
    (*retval_1.mutable_attr())["T"].set_type(DT_INT32);
    (*retval_1.mutable_attr())["index"].set_i(0);
    scope.graph()->AddNode(retval_1, &status);
    TF_ASSERT_OK(status);
    NodeDef retval_2;
    retval_2.set_name("retval_1");
    retval_2.set_op(FunctionLibraryDefinition::kRetOp);
    *retval_2.add_input() = "foo:1";
    (*retval_2.mutable_attr())["T"].set_type(DT_INT32);
    (*retval_2.mutable_attr())["index"].set_i(1);
    scope.graph()->AddNode(retval_2, &status);
    TF_ASSERT_OK(status);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
  }

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1});

  XlaCompiler::Options options = DefaultOptions();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "constants",
                                     std::move(graph), args, &result));

  ASSERT_EQ(2, result.outputs.size());
  EXPECT_FALSE(result.outputs[1].is_constant);
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, ResourceManager) {
  // Builds a graph that calls the dummy resource Op.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = DummyReadResourceCC(scope.WithOpName("B"), a);
  auto c = ops::Add(scope.WithOpName("C"), b.output2_, b.output1_);
  auto d = ops::_Retval(scope.WithOpName("D"), c, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  DummyResourceForTest* resource = new DummyResourceForTest();

  // Compiles the graph.
  auto options = DefaultOptions();
  std::function<absl::Status(ResourceMgr*)> populate_function =
      [resource](ResourceMgr* rm) {
        resource->Ref();
        return rm->Create(rm->default_container(), "dummy", resource);
      };
  options.populate_resource_manager = &populate_function;
  XlaCompiler compiler(options);

  EXPECT_EQ(0, resource->Get());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "dummy",
                                     std::move(graph), args, &result));

  EXPECT_EQ(1, resource->Get());

  resource->Unref();
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, DeterministicCompilation) {
  // Builds a graph that contains a node with two output edges. The compiler
  // should always traverse them in the same order.
  const int64_t test_count = 2;

  std::vector<XlaCompiler::CompilationResult> results(test_count);

  for (int64_t i = 0; i < test_count; ++i) {
    Scope scope = Scope::NewRootScope().ExitOnError();
    auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
    auto b = ops::Neg(scope.WithOpName("B"), a);
    auto c = ops::Neg(scope.WithOpName("C"), a);
    auto d = ops::Add(scope.WithOpName("D"), b, c);
    auto e = ops::_Retval(scope.WithOpName("E"), d, 0);
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    TF_ASSERT_OK(scope.ToGraph(graph.get()));

    // Builds a description of the argument.
    std::vector<XlaCompiler::Argument> args(1);
    args[0].kind = XlaCompiler::Argument::kParameter;
    args[0].type = DT_INT32;
    args[0].shape = TensorShape({2});

    // Compiles the graph.
    auto options = DefaultOptions();
    XlaCompiler compiler(options);

    TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "dummy",
                                       std::move(graph), args, &results[i]));
  }

  for (int64_t i = 1; i < test_count; ++i) {
    const auto& m1 = results[i - 1].computation->proto();
    const auto& m2 = results[i].computation->proto();
    ASSERT_EQ(m1.computations_size(), m2.computations_size());
    // Check if every hlo computation is the same.
    for (int k = 0; k < m1.computations_size(); k++) {
      const auto& c1 = m1.computations(k);
      const auto& c2 = m2.computations(k);
      ASSERT_EQ(c1.instructions_size(), c2.instructions_size());
      for (int j = 0; j < c1.instructions_size(); j++) {
        auto instr1 = c1.instructions(j);
        auto instr2 = c2.instructions(j);
        instr1.clear_name();
        instr1.clear_id();
        instr1.clear_operand_ids();
        instr2.clear_name();
        instr2.clear_id();
        instr2.clear_operand_ids();
        // The names of instructions were uniquified by the XlaBuilder and the
        // unique ids may be different, the rest of the fields should be
        // identical.
        string str1, str2;
        LOG(INFO) << "instr1 = " << instr1.DebugString();
        LOG(INFO) << "instr2 = " << instr2.DebugString();
        instr1.AppendPartialToString(&str1);
        instr2.AppendPartialToString(&str2);
        EXPECT_EQ(str1, str2);
      }
    }
  }
}

// Tests a computation that receives a TensorArray resource as input and
// updates it.
TEST_F(XlaCompilerTest, CanPassTensorArraysToAndFromComputation) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad1");
  auto grad2 = ops::TensorArrayGrad(scope, arg, grad1.flow_out, "grad2");
  auto index = ops::Const<int32>(scope, 1);
  auto write = ops::TensorArrayWrite(scope, grad1.grad_handle, index, index,
                                     grad2.flow_out);
  auto read = ops::TensorArrayRead(scope, arg, index, write.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].max_array_size = 2;
  args[0].tensor_array_gradients = {"grad2"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  ASSERT_EQ(1, result.resource_updates.size());
  const XlaCompiler::ResourceUpdate& update = result.resource_updates[0];
  EXPECT_EQ(0, update.input_index);
  EXPECT_EQ(DT_INT32, update.type);
  EXPECT_EQ((std::set<string>{"grad1", "grad2"}),
            update.tensor_array_gradients_accessed);

  // Tests that the generated computation works.
  xla::Literal input_base = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal input_grad2 = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  xla::Literal input = xla::LiteralUtil::MakeTuple({&input_base, &input_grad2});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(input).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_->Execute(*result.computation, {param0_data.get()}).value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal output_read = xla::LiteralUtil::CreateR0<int32>(42);
  xla::Literal output_base = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal output_grad1 = xla::LiteralUtil::CreateR1<int32>({0, 1});
  xla::Literal output_grad2 = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  xla::Literal output_resource =
      xla::LiteralUtil::MakeTuple({&output_base, &output_grad1, &output_grad2});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&output_read, &output_resource});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, UnwrittenTensorArrayGradientsAreNotComputationOutputs) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad1");
  auto index = ops::Const<int32>(scope, 1);
  auto read = ops::TensorArrayRead(scope, arg, index, grad1.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].max_array_size = 2;
  args[0].tensor_array_gradients = {"grad1"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  EXPECT_EQ(0, result.resource_updates.size());
}

// Tests compilation and execution of a graph that adds two tensors.
TEST_F(XlaCompilerTest, NewTensorArrayGradientsAreComputationOutputs) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_RESOURCE, 0);
  auto flow = ops::Const<float>(scope, {});
  auto grad1 = ops::TensorArrayGrad(scope, arg, flow, "grad2");
  auto index = ops::Const<int32>(scope, 1);
  auto read = ops::TensorArrayRead(scope, arg, index, grad1.flow_out, DT_INT32);
  auto retval = ops::_Retval(scope.WithOpName("retval"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kTensorArray;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[0].max_array_size = 2;
  args[0].tensor_array_gradients = {"grad1"};

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  EXPECT_EQ(1, result.resource_updates.size());
}

// Tests CompileFunction with undefined function fails.
TEST_F(XlaCompilerTest, UndefinedFunctionFails) {
  XlaCompiler compiler(DefaultOptions());

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  NameAttrList name_attr;
  name_attr.set_name("Function_NotDefined_");
  absl::Status status =
      compiler.CompileFunction(XlaCompiler::CompileOptions(), name_attr,
                               /*args=*/{}, &result);
  EXPECT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "is not defined."))
      << status.message();
}

FunctionDef FillFn() {
  return FunctionDefHelper::Define(
      // Name
      "FillFn",
      // Args
      {"x: T", "dims: int32"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {{{"y"}, "Fill", {"dims", "x"}, {{"T", "$T"}}}});
}

FunctionDef IdentityFn() {
  return FunctionDefHelper::Define(
      "IdentityFn", {"x: T"}, {"y: T"},
      {"T: {float, double, int32, int64}"},
      {{{"y"}, "Identity", {"x"}, {{"T", "$T"}}}});
}

TEST_F(XlaCompilerTest, FunctionCallWithConstants) {
  // Certain operations in a function, "Fill" for example, requires the
  // operator's argument to be a compile-time constant instead of a parameter.
  // This testcase tests if XlaCompiler can handle such operators inside
  // function calls.
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  *flib.add_function() = FillFn();

  TF_ASSERT_OK(flib_def_->AddFunctionDef(FillFn()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("value"), 1, {});
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("fill", "FillFn", flib_def_.get())
                   .Input(value.name(), 0, DT_INT32)
                   .Input(shape.name(), 1, DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* fill = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(fill));
  scope.graph()->AddEdge(value.node(), 0, fill, 0);
  scope.graph()->AddEdge(shape.node(), 0, fill, 1);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(fill), 0);

  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args;

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "fill",
                                     std::move(graph), args, &result));
}

TEST_F(XlaCompilerDynamicSizesTest, FunctionCallPreservesDynamicExpressions) {
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  *flib.add_function() = IdentityFn();
  TF_ASSERT_OK(flib_def_->AddFunctionDef(IdentityFn()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_INT32, 0);
  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("identity_fn", "IdentityFn", flib_def_.get())
                   .Input(arg.node()->name(), 0, DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* identity_fn = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(identity_fn));
  scope.graph()->AddEdge(arg.node(), 0, identity_fn, 0);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(identity_fn), 0);
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = xla::ShapeUtil::MakeShape(
      xla::S32, {9}, std::vector<xla::DExpr>{xla::DExpr::Var(2)});

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(),
                                     "identity_function", std::move(graph),
                                     args, &result));

  ASSERT_EQ(result.outputs.size(), 1);
  EXPECT_TRUE(xla::DynExpr::equal(result.outputs[0].shape.get_filled_expression(
                                      0),
                                  xla::DExpr::Var(2)));

  const xla::Shape& result_shape =
      xla::ShapeUtil::GetSubshape(result.xla_output_shape, {0});
  EXPECT_TRUE(
      xla::DynExpr::equal(result_shape.expressions(0), xla::DExpr::Var(2)));
}

// Tests CompileFunction with a local function lookup failing, fails with
// informative error about both lookups.
TEST_F(XlaCompilerTest, LocalFunctionWithWrongArgumentsFail) {
  XlaCompiler compiler(DefaultOptions());

  auto local_flib_def = LocalFlibDef(&compiler);
  TF_ASSERT_OK(local_flib_def->AddFunctionDef(test::function::XTimesTwo()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  XlaCompiler::CompilationResult result;
  NameAttrList name_attr;
  name_attr.set_name("XTimesTwo");
  absl::Status status =
      compiler.CompileFunction(XlaCompiler::CompileOptions(), name_attr,
                               /*args=*/{}, &result);

  ASSERT_FALSE(status.ok());
  // Flib lookup failure.
  EXPECT_TRUE(absl::StrContains(status.message(), "is not defined."))
      << status.message();
  // Local flib lookup failure.
  EXPECT_TRUE(absl::StrContains(status.message(), "Attr T is not found"))
      << status.message();
}

FunctionDef SliceFn() {
  return FunctionDefHelper::Define(
      // Name
      "SliceFn",
      // Args
      {"x: T", "begin: Index", "size: Index"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}", "Index: {int32,int64}"},
      // Nodes
      {{{"y"},
        "Slice",
        {"x", "begin", "size"},
        {{"T", "$T"}, {"Index", "$Index"}}}});
}

TEST_F(XlaCompilerTest, SliceWithDynamicBegins) {
  // Certain operations in a function, "Slice" for example, support both dynamic
  // inputs and static inputs. This test checks that dynamic inputs can also
  // be supported in a function call.
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  *flib.add_function() = SliceFn();

  TF_ASSERT_OK(flib_def_->AddFunctionDef(SliceFn()));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  auto begin = ops::_Arg(scope.WithOpName("arg"), DT_INT32, 0);
  auto size = ops::Const<int32>(scope.WithOpName("value"), {1}, {1});

  TF_EXPECT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("slice", "SliceFn", flib_def_.get())
                   .Input(value.name(), 0, DT_INT32)
                   .Input(begin.node()->name(), 1, DT_INT32)
                   .Input(size.name(), 2, DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* slice = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(slice));
  scope.graph()->AddEdge(value.node(), 0, slice, 0);
  scope.graph()->AddEdge(begin.node(), 0, slice, 1);
  scope.graph()->AddEdge(size.node(), 0, slice, 2);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(slice), 0);

  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the argument.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({1});

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "slice",
                                     std::move(graph), args, &result));
}

void RunAndCheckVariablesComputation(
    xla::Client* client, const XlaCompiler::CompilationResult& result) {
  xla::Literal param0_literal = xla::LiteralUtil::CreateR1<int32>({7, 42});
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param0_data =
      client->TransferToServer(param0_literal).value();
  std::unique_ptr<xla::GlobalData> param1_data =
      client->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .value();
  xla::Literal actual_literal = client->Transfer(*actual).value();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({5, 144});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({4, 143});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests a simple graph that reads and writes a variable.
TEST_F(XlaCompilerTest, Variables) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  RunAndCheckVariablesComputation(client_, result);
}

TEST_F(XlaCompilerTest, ResultLayoutSingle) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("RET"), a, 0);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  // Sets the representation function to return a non-default layout.
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType type, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  options.shape_determination_fns = shape_determination_fns;

  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  auto compile_options = XlaCompiler::CompileOptions();
  compile_options.always_return_tuple = false;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "id", std::move(graph),
                                     args, &result));
  EXPECT_TRUE(xla::ShapeUtil::Equal(
      result.xla_output_shape,
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S32, {2, 3}, {0, 1})));
  EXPECT_EQ(result.computation->GetProgramShape().value().result(),
            result.xla_output_shape);
}

TEST_F(XlaCompilerTest, ResultLayoutMultiple) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("RET1"), a, 0);
  auto c = ops::_Retval(scope.WithOpName("RET2"), a, 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 3});

  auto options = DefaultOptions();
  // Sets the representation function to return a non-default layout.
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType type, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::Shape xla_shape;
    TF_RETURN_IF_ERROR(TensorShapeToXLAShape(type, shape, &xla_shape));
    *xla_shape.mutable_layout() = xla::LayoutUtil::MakeLayout({0, 1});
    return xla_shape;
  };
  shape_determination_fns.layout_preference_fn = UseNoPreferenceLayoutFn();
  options.shape_determination_fns = shape_determination_fns;

  // Compiles the graph.
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "id",
                                     std::move(graph), args, &result));
  xla::Shape result_shape =
      xla::ShapeUtil::MakeShapeWithDenseLayout(xla::S32, {2, 3}, {0, 1});

  EXPECT_TRUE(xla::ShapeUtil::Equal(
      result.xla_output_shape,
      xla::ShapeUtil::MakeTupleShape({result_shape, result_shape})));
  EXPECT_EQ(result.computation->GetProgramShape().value().result(),
            result.xla_output_shape);
}

// Tests a simple graph that reads and writes a variable.
TEST_F(XlaCompilerTest, ReturnResourceHandleOnly) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), var, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));

  // Tests that the generated computation works.
  xla::Literal param1_literal = xla::LiteralUtil::CreateR1<int32>({-3, 101});
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_->Execute(*result.computation, {param1_data.get()}).value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal expected_literal = xla::LiteralUtil::MakeTuple({});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, ReturnResourceHandle) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  // Adds an identity op around the resource to make sure identity ops propagate
  // resources correctly.
  auto identity = ops::Identity(scope.WithOpName("VIdentity"), var);
  auto write = ops::AssignAddVariableOp(scope, identity, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto r = ops::_Retval(scope.WithOpName("R"), var, 0);
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 1);

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  RunAndCheckVariablesComputation(client_, result);
}

absl::StatusOr<std::unique_ptr<Graph>> BuildTestGraph() {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  auto write = ops::AssignAddVariableOp(scope, var, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto read_plus_one = ops::Add(scope, read, ops::Const<int32>(scope, 1));
  auto d = ops::_Retval(scope.WithOpName("D"), read_plus_one, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_RETURN_IF_ERROR(scope.ToGraph(graph.get()));
  return std::move(graph);
}

// Tests a simple graph that reads and writes a variable, with a
// shape_representation_fn passed to the compiler that flattens all
// variable tensors to vectors.
TEST_F(XlaCompilerTest, VariableRepresentationShapeFunction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> graph, BuildTestGraph());

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 2});

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType type, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::PrimitiveType ptype;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(type, &ptype));
    return xla::ShapeUtil::MakeShape(ptype, {shape.num_elements()});
  };
  options.shape_determination_fns = shape_determination_fns;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = false;  // Only reshape variables.

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::ProgramShape> program_shape,
                          client_->GetComputationShape(*result.computation));

  ASSERT_EQ(program_shape->parameters_size(), 2);
  EXPECT_TRUE(
      xla::ShapeUtil::Compatible(program_shape->parameters(0),
                                 xla::ShapeUtil::MakeShape(xla::S32, {2, 2})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(1), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->result(),
      xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::S32, {2, 2}),
           xla::ShapeUtil::MakeShape(xla::S32, {4})})));

  // Tests that the generated computation works.
  xla::Literal param0_literal =
      xla::LiteralUtil::CreateR2<int32>({{4, 55}, {1, -3}});
  xla::Literal param1_literal =
      xla::LiteralUtil::CreateR1<int32>({22, 11, 33, 404});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal expected0 =
      xla::LiteralUtil::CreateR2<int32>({{27, 67}, {35, 402}});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({26, 66, 34, 401});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, ArgRetvalShapeRepresentationFunction) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Graph> graph, BuildTestGraph());

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 2});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2, 2});

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns;
  shape_determination_fns.shape_representation_fn =
      [](const TensorShape& shape, DataType type, bool use_fast_memory,
         XlaLayoutPreference layout_preference) -> absl::StatusOr<xla::Shape> {
    xla::PrimitiveType ptype;
    TF_RETURN_IF_ERROR(DataTypeToPrimitiveType(type, &ptype));
    return xla::ShapeUtil::MakeShape(ptype, {shape.num_elements()});
  };
  options.shape_determination_fns = shape_determination_fns;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = true;  // Reshape args and retvals.

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::ProgramShape> program_shape,
                          client_->GetComputationShape(*result.computation));

  ASSERT_EQ(program_shape->parameters_size(), 2);
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(0), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->parameters(1), xla::ShapeUtil::MakeShape(xla::S32, {4})));
  EXPECT_TRUE(xla::ShapeUtil::Compatible(
      program_shape->result(),
      xla::ShapeUtil::MakeTupleShape(
          {xla::ShapeUtil::MakeShape(xla::S32, {4}),
           xla::ShapeUtil::MakeShape(xla::S32, {4})})));

  // Tests that the generated computation works.
  xla::Literal param0_literal =
      xla::LiteralUtil::CreateR1<int32>({4, 55, 1, -3});
  xla::Literal param1_literal =
      xla::LiteralUtil::CreateR1<int32>({22, 11, 33, 404});
  std::unique_ptr<xla::GlobalData> param0_data =
      client_->TransferToServer(param0_literal).value();
  std::unique_ptr<xla::GlobalData> param1_data =
      client_->TransferToServer(param1_literal).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation, {param0_data.get(), param1_data.get()})
          .value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal expected0 = xla::LiteralUtil::CreateR1<int32>({27, 67, 35, 402});
  xla::Literal expected1 = xla::LiteralUtil::CreateR1<int32>({26, 66, 34, 401});
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

// Tests a graph which has a function with an invalid op.
TEST_F(XlaCompilerTest, FunctionWithInvalidOp) {
  XlaCompiler compiler(DefaultOptions());

  FunctionDefLibrary flib;
  FunctionDef fn = FillFn();
  NodeDef* node = fn.add_node_def();
  node->set_name("Invalid");
  node->set_op("InvalidOp"); /* unsupported op */
  node = fn.add_node_def();
  node->set_name("Switch");
  node->set_op("Switch"); /* control flow node */
  *flib.add_function() = fn;

  TF_ASSERT_OK(flib_def_->AddFunctionDef(fn));

  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto value = ops::Const<int32>(scope.WithOpName("value"), 1, {});
  auto shape = ops::Const<int32>(scope.WithOpName("shape"), {5}, {1});
  TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(flib));

  NodeDef def;
  TF_ASSERT_OK(NodeDefBuilder("fill_fn", "FillFn", flib_def_.get())
                   .Input(value.name(), 0, DT_INT32)
                   .Input(shape.name(), 1, DT_INT32)
                   .Finalize(&def));
  absl::Status status;
  Node* fill = scope.graph()->AddNode(def, &status);
  TF_ASSERT_OK(status);
  TF_ASSERT_OK(scope.DoShapeInference(fill));
  scope.graph()->AddEdge(value.node(), 0, fill, 0);
  scope.graph()->AddEdge(shape.node(), 0, fill, 1);

  auto retval = ops::_Retval(scope.WithOpName("retval"), Output(fill), 0);

  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler::CompilationResult result;
  status = compiler.CompileGraph(XlaCompiler::CompileOptions(), "fill",
                                 std::move(graph), args, &result);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(), "InvalidOp"))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "{{node fill_fn}}"))
      << status.message();
}

// Tests a graph which has a node with invalid data type.
TEST_F(XlaCompilerTest, NodeWithInvalidDataType) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef shape;
  shape.set_name("Shape");
  shape.set_op("Shape");
  (*shape.mutable_attr())["T"].set_type(DT_INT32);
  (*shape.mutable_attr())["out_type"].set_type(DT_BOOL); /* invalid type */
  absl::Status status;
  Node* shape_node = graph->AddNode(shape, &status);
  TF_ASSERT_OK(status);
  graph->AddControlEdge(graph->source_node(), shape_node);

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler::CompilationResult result;
  XlaCompiler compiler(DefaultOptions());
  status = compiler.CompileGraph(XlaCompiler::CompileOptions(), "invalid_type",
                                 std::move(graph), args, &result);
  ASSERT_FALSE(status.ok());
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "is not in the list of allowed values"))
      << status.message();
  EXPECT_TRUE(absl::StrContains(status.message(), "{{node Shape}}"))
      << status.message();
}

TEST_F(XlaCompilerTest, SingleOpWithoutInputs) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef no_op;
  no_op.set_name("NoOp");
  no_op.set_op("NoOp");
  absl::Status status;
  graph->AddNode(no_op, &status);
  TF_ASSERT_OK(status);

  std::vector<XlaCompiler::Argument> args;
  XlaCompiler compiler(DefaultOptions());
  // No control edge linking NoOp with source/sink.
  {
    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "NoOp",
                                       std::move(graph_copy), args, &result));
  }
}

class DummySideEffectingOp : public XlaOpKernel {
 public:
  explicit DummySideEffectingOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {}
  void Compile(XlaOpKernelContext* ctx) override {
    OP_REQUIRES_OK(ctx, ctx->compiler()->SetNodeToken(
                            name(), xla::CreateToken(ctx->builder())));
  }
};

REGISTER_OP("DummySideEffectingOp");

REGISTER_XLA_OP(Name("DummySideEffectingOp"), DummySideEffectingOp);

TEST_F(XlaCompilerTest, TokenInputAndOutput) {
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  NodeDef side_effecting_op;
  side_effecting_op.set_name("DummySideEffectingOp");
  side_effecting_op.set_op("DummySideEffectingOp");
  AddNodeAttr(kXlaTokenInputNodesAttrName,
              std::vector<string>{kXlaTokenArgNodeName}, &side_effecting_op);
  AddNodeAttr(kXlaOriginalOutsideCompilationNodeName, side_effecting_op.name(),
              &side_effecting_op);
  absl::Status status;
  graph->AddNode(side_effecting_op, &status);
  TF_ASSERT_OK(status);
  EXPECT_TRUE(FixupSourceAndSinkEdges(graph.get()));

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kResource;
  args[0].resource_kind = XlaResource::kVariable;
  args[0].initialized = true;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2, 2});

  {
    // The case for entry computation: we don't add token input/output. Instead,
    // we use CreateToken HLO to create the entry token.
    XlaCompiler::CompileOptions options;
    options.is_entry_computation = true;
    options.add_token_input_output = false;
    options.return_updated_values_for_all_resources = true;
    XlaCompiler compiler(DefaultOptions());

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(options, "NoOp", std::move(graph_copy),
                                       args, &result));
    EXPECT_EQ(result.xla_input_shapes.size(), 1);
    EXPECT_TRUE(result.xla_output_shape.IsTuple());
    EXPECT_EQ(xla::ShapeUtil::TupleElementCount(result.xla_output_shape), 1);
  }
  {
    // The case for non-entry computation (e.g. while loop body). We add token
    // input/output.
    XlaCompiler::CompileOptions options;
    options.is_entry_computation = false;
    options.add_token_input_output = true;
    options.return_updated_values_for_all_resources = true;
    XlaCompiler compiler(DefaultOptions());

    std::unique_ptr<Graph> graph_copy(new Graph(OpRegistry::Global()));
    CopyGraph(*graph, graph_copy.get());
    XlaCompiler::CompilationResult result;
    TF_ASSERT_OK(compiler.CompileGraph(options, "NoOp", std::move(graph_copy),
                                       args, &result));
    EXPECT_EQ(result.xla_input_shapes.size(), 2);
    EXPECT_TRUE(result.xla_input_shapes[1].IsToken());
    EXPECT_TRUE(result.xla_output_shape.IsTuple());
    EXPECT_EQ(xla::ShapeUtil::TupleElementCount(result.xla_output_shape), 2);
    EXPECT_TRUE(xla::ShapeUtil::GetTupleElementShape(result.xla_output_shape, 1)
                    .IsToken());
  }
}

TEST_F(XlaCompilerTest, OpsWithTensorListInput) {
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  // Build cond fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    auto result = ops::Const<bool>(scope, {true}, {});
    ops::_Retval(scope.WithOpName("ret"), result, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "cond", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }
  // Build body fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
    ops::_Retval(scope.WithOpName("ret"), arg, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "body", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto element_shape = ops::Const<int32>(scope, {1}, {1});
  auto max_elements = ops::Const<int32>(scope, {10}, {});
  auto arg = ops::_Arg(scope.WithOpName("arg"), DT_VARIANT, 0);
  std::initializer_list<Output> out = {arg, arg};
  auto add_n = ops::AddN(scope, out);
  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op =
      ops::While(scope, std::initializer_list<Input>{arg}, cond_fn, body_fn);
  auto ret0 = ops::_Retval(scope.WithOpName("ret0"), add_n, 0);
  auto ret1 = ops::_Retval(scope.WithOpName("ret1"), while_op.output[0], 1);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kTensorList;
  xla::Shape tensor_list_element_shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_INT32, TensorShape{1},
                                     &tensor_list_element_shape));
  xla::Shape index_shape;
  TF_ASSERT_OK(TensorShapeToXLAShape(DT_INT32, TensorShape{}, &index_shape));
  std::vector<xla::Shape> shapes{tensor_list_element_shape, index_shape};
  xla::Shape arg_shape = xla::ShapeUtil::MakeTupleShape(shapes);
  args[0].shape = arg_shape;

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "add",
                                     std::move(graph), args, &result));
  ASSERT_EQ(result.outputs.size(), 2);
  const XlaCompiler::OutputDescription& output0 = result.outputs[0];
  ASSERT_TRUE(output0.is_tensor_list);
  const XlaCompiler::OutputDescription& output1 = result.outputs[1];
  ASSERT_TRUE(output1.is_tensor_list);
}

// Test the compiler supports WhileOp with a loop body where DT_RESOURCE
// variables are both inputs and outputs.
TEST_F(XlaCompilerTest, WhileWithResources) {
  FunctionDefLibrary fdef_lib;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), fdef_lib);
  // Build cond fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);
    auto less = ops::Less(scope, arg0, ops::Const<int32>(scope, 10));
    (void)ops::_Retval(scope.WithOpName("ret"), less, 0);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "cond", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }
  // Build body fn for While.
  {
    Scope scope = Scope::NewRootScope().ExitOnError();
    std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
    auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
    auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
    auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);
    auto read1 = ops::ReadVariableOp(scope.WithOpName("read1"), arg1, DT_INT32);
    auto plus_read1 = ops::Add(scope, arg0, read1);
    auto read2 = ops::ReadVariableOp(scope.WithOpName("read2"), arg2, DT_INT32);
    auto minus_read2 = ops::Sub(scope, plus_read1, read2);
    (void)ops::_Retval(scope.WithOpName("ret0"), minus_read2, 0);
    (void)ops::_Retval(scope.WithOpName("ret1"), arg1, 1);
    (void)ops::_Retval(scope.WithOpName("ret2"), arg2, 2);
    TF_ASSERT_OK(scope.ToGraph(graph.get()));
    FunctionDef fdef;
    TF_ASSERT_OK(GraphToFunctionDef(*graph, "body", &fdef));
    TF_ASSERT_OK(flib_def.AddFunctionDef(fdef));
  }

  Scope scope = Scope::NewRootScope().ExitOnError();
  auto arg0 = ops::_Arg(scope.WithOpName("arg0"), DT_INT32, 0);
  auto arg1 = ops::_Arg(scope.WithOpName("arg1"), DT_RESOURCE, 1);
  auto arg2 = ops::_Arg(scope.WithOpName("arg2"), DT_RESOURCE, 2);

  NameAttrList cond_fn, body_fn;
  cond_fn.set_name("cond");
  body_fn.set_name("body");
  auto while_op = ops::While(
      scope, std::initializer_list<Input>{arg0, arg1, arg2}, cond_fn, body_fn);

  (void)ops::_Retval(scope.WithOpName("ret0"), while_op.output[0], 0);
  (void)ops::_Retval(scope.WithOpName("ret1"), while_op.output[1], 1);
  (void)ops::_Retval(scope.WithOpName("ret2"), while_op.output[2], 2);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(3);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({});
  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({});
  args[2].kind = XlaCompiler::Argument::kResource;
  args[2].resource_kind = XlaResource::kVariable;
  args[2].initialized = true;
  args[2].type = DT_INT32;
  args[2].shape = TensorShape({});

  // Compiles the graph.
  XlaCompiler::Options options = DefaultOptions();
  options.flib_def = &flib_def;
  XlaCompiler compiler(options);

  XlaCompiler::CompileOptions compile_options = XlaCompiler::CompileOptions();
  compile_options.return_updated_values_for_all_resources = true;
  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "tested_while_with_vars",
                                     std::move(graph), args, &result));
  ASSERT_EQ(result.outputs.size(), 3);
  const XlaCompiler::OutputDescription& output1 = result.outputs[1];
  ASSERT_EQ(output1.input_index, 1);
  const XlaCompiler::OutputDescription& output2 = result.outputs[2];
  ASSERT_EQ(output2.input_index, 2);

  // Tests that the generated computation works.
  xla::Literal literal0 = xla::LiteralUtil::CreateR0<int32>(0);
  xla::Literal literal1 = xla::LiteralUtil::CreateR0<int32>(2);
  xla::Literal literal2 = xla::LiteralUtil::CreateR0<int32>(1);
  std::unique_ptr<xla::GlobalData> data0 =
      client_->TransferToServer(literal0).value();
  std::unique_ptr<xla::GlobalData> data1 =
      client_->TransferToServer(literal1).value();
  std::unique_ptr<xla::GlobalData> data2 =
      client_->TransferToServer(literal2).value();

  std::unique_ptr<xla::GlobalData> actual =
      client_
          ->Execute(*result.computation,
                    {data0.get(), data1.get(), data2.get()})
          .value();
  xla::Literal actual_literal = client_->Transfer(*actual).value();

  xla::Literal expected0 = xla::LiteralUtil::CreateR0<int32>(10);
  xla::Literal expected1 = xla::LiteralUtil::CreateR0<int32>(2);
  xla::Literal expected2 = xla::LiteralUtil::CreateR0<int32>(1);
  xla::Literal expected_literal =
      xla::LiteralUtil::MakeTuple({&expected0, &expected1, &expected2});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

TEST_F(XlaCompilerTest, SetShardingForReturnedTuple) {
  // Builds a graph that returns its only argument.
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::_Arg(scope.WithOpName("A"), DT_INT32, 0);
  auto b = ops::_Retval(scope.WithOpName("B"), a, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Sets _XlaSharding attribute for the _Retval node.
  auto node_name_index = graph->BuildNodeNameIndex();
  Node* ret_node = node_name_index["B"];
  ASSERT_NE(ret_node, nullptr);
  xla::Array<int64_t> tile_assignment({2});
  tile_assignment.FillIota(0);
  xla::HloSharding sharding = xla::HloSharding::Tile(tile_assignment);
  ret_node->AddAttr("_XlaSharding", sharding.ToProto().SerializeAsString());

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kParameter;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});

  // Compiles the graph.
  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(XlaCompiler::CompileOptions(), "test",
                                     std::move(graph), args, &result));

  // Tests that we set sharding on the root TUPLE instruction.
  const auto& hlo_module_proto = result.computation->proto();
  ASSERT_EQ(hlo_module_proto.computations_size(), 1);
  const auto& hlo_computation_proto = hlo_module_proto.computations(0);
  std::optional<xla::HloInstructionProto> root_instruction_proto;
  for (const auto& inst : hlo_computation_proto.instructions()) {
    if (inst.id() == hlo_computation_proto.root_id()) {
      root_instruction_proto = inst;
      break;
    }
  }
  ASSERT_TRUE(root_instruction_proto);
  xla::Shape tuple_shape = xla::ShapeUtil::MakeTupleShape(
      {xla::ShapeUtil::MakeShape(xla::S32, {2})});
  xla::HloSharding tuple_sharding = xla::HloSharding::Tuple(
      tuple_shape, std::vector<xla::HloSharding>{sharding});
  EXPECT_EQ(root_instruction_proto->sharding().SerializeAsString(),
            tuple_sharding.ToProto().SerializeAsString());
}

TEST_F(XlaCompilerTest, AliasResourceUpdates) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto a = ops::Const<int32>(scope.WithOpName("A"), {1, 2});
  auto var = ops::_Arg(scope.WithOpName("V"), DT_RESOURCE, 1);
  auto write = ops::AssignAddVariableOp(scope, var, a);
  auto read = ops::ReadVariableOp(
      scope.WithControlDependencies(std::vector<Operation>{write}), var,
      DT_INT32);
  auto d = ops::_Retval(scope.WithOpName("D"), read, 0);
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));
  TF_ASSERT_OK(scope.ToGraph(graph.get()));

  // Builds a description of the arguments.
  std::vector<XlaCompiler::Argument> args(2);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_INT32;
  args[0].shape = TensorShape({2});
  args[0].constant_value = Tensor(DT_INT32, {1, 1});
  args[0].initialized = true;

  args[1].kind = XlaCompiler::Argument::kResource;
  args[1].resource_kind = XlaResource::kVariable;
  args[1].initialized = true;
  args[1].type = DT_INT32;
  args[1].shape = TensorShape({2});

  XlaCompiler compiler(DefaultOptions());

  XlaCompiler::CompileOptions compile_options;
  compile_options.alias_resource_update = true;

  XlaCompiler::CompilationResult result;
  TF_ASSERT_OK(compiler.CompileGraph(compile_options, "add", std::move(graph),
                                     args, &result));

  const xla::HloInputOutputAliasProto& alias =
      result.computation->proto().input_output_alias();
  EXPECT_EQ(alias.entries_size(), 1);
  EXPECT_EQ(alias.entries(0).parameter_number(), 0);
}

// Tests that passing in an exact duplicate input to SetDeviceToHostMetadata
// is not an error.
TEST_F(XlaCompilerTest, SetDeviceToHostMetadataExactDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};

  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
}

// Tests that passing in a mismatched duplicate input to
// SetDeviceToHostMeatadata is not an error.
TEST_F(XlaCompilerTest, SetDeviceToHostMetadataMismatchedDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};
  std::vector<DataType> types2{DT_FLOAT};
  std::vector<TensorShape> shapes2{TensorShape({1})};

  TF_ASSERT_OK(compiler.SetDeviceToHostMetadata(key, types, shapes));
  absl::Status status = compiler.SetDeviceToHostMetadata(key, types2, shapes2);
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
}

// Tests that passing in an exact duplicate input to SetHostToDeviceMetadata
// is not an error.
TEST_F(XlaCompilerTest, SetHostToDeviceMetadataExactDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};

  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
}

// Tests that passing in a mismatched duplicate input to
// SetHostToDeviceMeatadata is not an error.
TEST_F(XlaCompilerTest, SetHostToDeviceMetadataMismatchedDuplicate) {
  XlaCompiler compiler(DefaultOptions());

  const string& key = "comm_key";
  std::vector<DataType> types{DT_INT32};
  std::vector<TensorShape> shapes{TensorShape({2})};
  std::vector<DataType> types2{DT_FLOAT};
  std::vector<TensorShape> shapes2{TensorShape({1})};

  TF_ASSERT_OK(compiler.SetHostToDeviceMetadata(key, types, shapes));
  absl::Status status = compiler.SetHostToDeviceMetadata(key, types2, shapes2);
  EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
}

TEST_F(XlaCompilerTest, GetChannelHandleIndependently) {
  XlaCompiler compiler1(DefaultOptions());
  XlaCompiler compiler2(DefaultOptions());
  int num_channels = 3;
  std::vector<int> channel_ids1, channel_ids2;
  for (int j = 0; j < num_channels; ++j) {
    xla::ChannelHandle channel_handle;
    TF_ASSERT_OK(
        compiler1.GetChannelHandle(/*key=*/absl::StrCat(j), &channel_handle));
    channel_ids1.push_back(channel_handle.handle());
  }
  for (int j = 0; j < num_channels; ++j) {
    xla::ChannelHandle channel_handle;
    TF_ASSERT_OK(
        compiler2.GetChannelHandle(/*key=*/absl::StrCat(j), &channel_handle));
    channel_ids2.push_back(channel_handle.handle());
  }
  EXPECT_THAT(channel_ids1, ::testing::UnorderedElementsAreArray({1, 2, 3}));
  EXPECT_THAT(channel_ids2, ::testing::UnorderedElementsAreArray({1, 2, 3}));
}

TEST_F(OpsTestBase, BuildSingleOpCompileArgument) {
  TF_EXPECT_OK(NodeDefBuilder("identity_op", "Identity")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DT_FLOAT)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2}), {0, 1});
  TF_EXPECT_OK(RunOpKernel());

  XlaCompiler::SingleOpCompileArgument arg(*context_);

  EXPECT_THAT(arg.output_dtypes, ::testing::ElementsAreArray({DT_FLOAT}));
  EXPECT_EQ(arg.node_def.SerializeAsString(),
            context_->op_kernel().def().SerializeAsString());
  EXPECT_EQ(arg.config_proto.ByteSizeLong(), 0);
}

TEST_F(OpsTestBase, CompileSingleOp) {
  TF_EXPECT_OK(NodeDefBuilder("identity_op", "Identity")
                   .Input(FakeInput(DT_FLOAT))
                   .Attr("T", DT_FLOAT)
                   .Finalize(node_def()));
  TF_EXPECT_OK(InitOp());
  AddInputFromArray<float>(TensorShape({1, 2}), {6.9, 4.2});
  TF_EXPECT_OK(RunOpKernel());

  XlaCompiler::SingleOpCompileArgument single_op_arg(*context_);

  xla::Client* client = xla::ClientLibrary::LocalClientOrDie();
  XlaOpRegistry::RegisterCompilationKernels();
  FunctionDefLibrary flib;
  std::unique_ptr<FunctionLibraryDefinition> flib_def(
      new FunctionLibraryDefinition(OpRegistry::Global(), flib));

  XlaCompiler::Options options;
  options.device_type = DeviceType(DEVICE_CPU_XLA_JIT);
  options.client = client;
  options.flib_def = flib_def.get();

  XlaCompiler compiler(options);

  std::vector<XlaCompiler::Argument> args(1);
  args[0].kind = XlaCompiler::Argument::kConstant;
  args[0].type = DT_FLOAT;
  args[0].shape = TensorShape({1, 2});
  args[0].constant_value = GetInput(0);
  args[0].initialized = true;

  XlaCompiler::CompilationResult result;
  TF_EXPECT_OK(compiler.CompileSingleOp(XlaCompiler::CompileOptions(),
                                        single_op_arg, args, &result));

  // Tests that the generated computation works.
  std::unique_ptr<xla::GlobalData> actual =
      client->Execute(*result.computation, {}).value();
  xla::Literal actual_literal = client->Transfer(*actual).value();

  xla::Literal expected0 = xla::LiteralUtil::CreateR2<float>({{6.9, 4.2}});
  xla::Literal expected_literal = xla::LiteralUtil::MakeTuple({&expected0});
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(expected_literal, actual_literal));
}

}  // namespace
}  // namespace tensorflow

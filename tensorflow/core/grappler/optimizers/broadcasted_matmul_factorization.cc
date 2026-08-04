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

#include "tensorflow/core/grappler/optimizers/broadcasted_matmul_factorization.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace {

using NodeIndex = absl::flat_hash_map<string, const NodeDef*>;

struct RepeatedInput {
  string query;
  int64_t repeats;
  int64_t width;
};

struct Candidate {
  const NodeDef* matmul;
  string varying;
  string query;
  string weights;
  int varying_index;
  int repeated_index;
  int32_t varying_width;
  int32_t query_width;
  int32_t output_width;
  int32_t repeats;
  DataType dtype;
};

const NodeDef* FindNode(const NodeIndex& nodes, const string& input) {
  const auto it = nodes.find(NodeName(input));
  return it == nodes.end() ? nullptr : it->second;
}

const NodeDef* ResolveIdentitySource(string input, const NodeIndex& nodes) {
  const NodeDef* node = FindNode(nodes, input);
  std::size_t remaining = nodes.size();

  while (node != nullptr && node->op() == "Identity") {
    // Following more Identity nodes than the graph contains implies a cycle.
    if (remaining == 0 || node->input_size() < 1) return nullptr;
    --remaining;
    node = FindNode(nodes, node->input(0));
  }

  return node;
}

bool HasType(const NodeDef& node, DataType type) {
  const auto it = node.attr().find("T");
  return it != node.attr().end() && it->second.type() == type;
}

bool IsUntransposed(const NodeDef& matmul) {
  const auto a = matmul.attr().find("transpose_a");
  const auto b = matmul.attr().find("transpose_b");
  return a != matmul.attr().end() && b != matmul.attr().end() &&
         !a->second.b() && !b->second.b();
}

bool ReadIntConst(const NodeDef* node, std::vector<int64_t>* values) {
  if (node == nullptr || (node->op() != "Const" && node->op() != "HostConst"))
    return false;
  const auto value = node->attr().find("value");
  if (value == node->attr().end()) return false;

  Tensor tensor;
  if (!tensor.FromProto(value->second.tensor())) return false;
  values->clear();
  values->reserve(tensor.NumElements());
  if (tensor.dtype() == DT_INT32) {
    const auto flat = tensor.flat<int32>();
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
      values->push_back(flat(i));
    }
    return true;
  }
  if (tensor.dtype() == DT_INT64) {
    const auto flat = tensor.flat<int64_t>();
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
      values->push_back(flat(i));
    }
    return true;
  }
  return false;
}

bool OutputShape(const NodeDef& node, int port,
                 const GraphProperties* properties, TensorShapeProto* shape) {
  const auto value = node.attr().find("value");
  // ANNC fused weights carry their shape in Const.value only.
  if (port == 0 && (node.op() == "Const" || node.op() == "HostConst")) {
    if (value == node.attr().end()) return false;

    Tensor tensor;
    if (!tensor.FromProto(value->second.tensor())) {
      VLOG(2) << "Could not decode constant tensor " << node.name();
      return false;
    }

    tensor.shape().AsProto(shape);
    return true;
  }

  const auto outputs = node.attr().find("_output_shapes");
  if (outputs != node.attr().end() && port >= 0 &&
      port < outputs->second.list().shape_size()) {
    *shape = outputs->second.list().shape(port);
    return !shape->unknown_rank();
  }

  if (properties != nullptr && properties->HasOutputProperties(node.name())) {
    const auto& inferred = properties->GetOutputProperties(node.name());
    if (port >= 0 && port < static_cast<int>(inferred.size())) {
      *shape = inferred[port].shape();
      return !shape->unknown_rank();
    }
  }

  const auto declared = node.attr().find("shape");
  if (port == 0 && declared != node.attr().end()) {
    *shape = declared->second.shape();
    return !shape->unknown_rank();
  }
  return false;
}

bool InputShape(string input, const NodeIndex& nodes,
                const GraphProperties* properties, TensorShapeProto* shape) {
  for (std::size_t remaining = nodes.size(); remaining > 0; --remaining) {
    const NodeDef* node = FindNode(nodes, input);
    if (node == nullptr) return false;

    const int port = NodePosition(input);
    if (OutputShape(*node, port, properties, shape)) return true;

    if (port != 0 || node->op() != "Identity" || node->input_size() < 1) {
      return false;
    }
    input = node->input(0);
  }
  return false;
}

bool Rank2Width(const string& input, const NodeIndex& nodes,
                const GraphProperties* properties, int64_t* width) {
  TensorShapeProto shape;
  if (!InputShape(input, nodes, properties, &shape) || shape.dim_size() != 2 ||
      shape.dim(1).size() <= 0) {
    return false;
  }
  *width = shape.dim(1).size();
  return true;
}

bool IsPreserved(const NodeDef* node,
                 const std::unordered_set<string>& preserved) {
  return node != nullptr && preserved.find(node->name()) != preserved.end();
}

bool MatchRepeatedInput(const string& input, DataType dtype,
                        const NodeIndex& nodes,
                        const GraphProperties* properties,
                        const std::unordered_set<string>& preserved,
                        RepeatedInput* match) {
  const NodeDef* reshape = FindNode(nodes, input);
  if (reshape == nullptr || reshape->op() != "Reshape" ||
      reshape->input_size() != 2 || !HasType(*reshape, dtype) ||
      IsPreserved(reshape, preserved)) {
    return false;
  }

  const NodeDef* tile = FindNode(nodes, reshape->input(0));
  const NodeDef* reshape_shape = FindNode(nodes, reshape->input(1));
  if (tile == nullptr || tile->op() != "Tile" || tile->input_size() != 2 ||
      !HasType(*tile, dtype) || IsPreserved(tile, preserved)) {
    return false;
  }
  const NodeDef* multiples = FindNode(nodes, tile->input(1));
  if (IsPreserved(multiples, preserved) ||
      IsPreserved(reshape_shape, preserved)) {
    return false;
  }

  std::vector<int64_t> tile_values;
  std::vector<int64_t> reshape_values;
  int64_t query_width;
  if (!ReadIntConst(multiples, &tile_values) || tile_values.size() != 2 ||
      tile_values[0] != 1 || tile_values[1] <= 1 ||
      !ReadIntConst(reshape_shape, &reshape_values) ||
      reshape_values.size() != 2 ||
      !Rank2Width(tile->input(0), nodes, properties, &query_width) ||
      reshape_values[1] != query_width) {
    return false;
  }

  TensorShapeProto query_shape;
  if (reshape_values[0] != -1 &&
      (!InputShape(tile->input(0), nodes, properties, &query_shape) ||
       query_shape.dim(0).size() < 0 ||
       reshape_values[0] != query_shape.dim(0).size() * tile_values[1])) {
    return false;
  }

  match->query = tile->input(0);
  match->repeats = tile_values[1];
  match->width = query_width;
  return true;
}

bool FitsInt32(int64_t value) {
  return value > 0 && value <= std::numeric_limits<int32_t>::max();
}

bool Analyze(const NodeDef& matmul, const NodeIndex& nodes,
             const GraphProperties* properties,
             const std::unordered_set<string>& preserved,
             Candidate* candidate) {
  if (matmul.op() != "MatMul" || matmul.input_size() != 2 ||
      IsPreserved(&matmul, preserved) || !IsUntransposed(matmul)) {
    return false;
  }

  const auto type = matmul.attr().find("T");
  if (type == matmul.attr().end() || type->second.type() != DT_FLOAT) {
    return false;
  }
  const DataType dtype = type->second.type();

  const NodeDef* concat = FindNode(nodes, matmul.input(0));
  if (concat == nullptr || concat->op() != "ConcatV2" ||
      concat->input_size() != 3 || !HasType(*concat, dtype) ||
      IsPreserved(concat, preserved)) {
    return false;
  }
  const auto n = concat->attr().find("N");
  if (n == concat->attr().end() || n->second.i() != 2) return false;

  const NodeDef* axis_node = FindNode(nodes, concat->input(2));
  std::vector<int64_t> axis;
  if (IsPreserved(axis_node, preserved) || !ReadIntConst(axis_node, &axis) ||
      axis.size() != 1 || (axis[0] != 1 && axis[0] != -1)) {
    return false;
  }

  const NodeDef* weight_source = ResolveIdentitySource(matmul.input(1), nodes);
  TensorShapeProto weight_shape;
  if (weight_source == nullptr ||
      (weight_source->op() != "Const" && weight_source->op() != "HostConst") ||
      !InputShape(matmul.input(1), nodes, properties, &weight_shape) ||
      weight_shape.dim_size() != 2) {
    VLOG(2) << "Rejected broadcasted MatMul candidate " << matmul.name()
            << ": weight must resolve to a rank-2 Const or HostConst";
    return false;
  }

  for (int repeated_index = 0; repeated_index < 2; ++repeated_index) {
    RepeatedInput repeated;
    if (!MatchRepeatedInput(concat->input(repeated_index), dtype, nodes,
                            properties, preserved, &repeated)) {
      continue;
    }

    const int varying_index = 1 - repeated_index;
    int64_t varying_width;
    const int64_t output_width = weight_shape.dim(1).size();
    if (!Rank2Width(concat->input(varying_index), nodes, properties,
                    &varying_width) ||
        weight_shape.dim(0).size() != varying_width + repeated.width ||
        !FitsInt32(varying_width) || !FitsInt32(repeated.width) ||
        !FitsInt32(output_width) || !FitsInt32(repeated.repeats) ||
        repeated.width <= repeated.repeats / (repeated.repeats - 1)) {
      continue;
    }

    *candidate = {&matmul,
                  concat->input(varying_index),
                  repeated.query,
                  matmul.input(1),
                  varying_index,
                  repeated_index,
                  static_cast<int32_t>(varying_width),
                  static_cast<int32_t>(repeated.width),
                  static_cast<int32_t>(output_width),
                  static_cast<int32_t>(repeated.repeats),
                  dtype};
    return true;
  }
  VLOG(2) << "Rejected broadcasted MatMul candidate " << matmul.name()
          << ": no profitable repeated Tile/Reshape input with compatible "
             "dimensions";
  return false;
}

NodeDef* AddNode(GraphDef* graph, const string& name, const char* op,
                 const string& device) {
  NodeDef* node = graph->add_node();
  node->set_name(name);
  node->set_op(op);
  node->set_device(device);
  return node;
}

string AddIntConst(GraphDef* graph, const string& name,
                   const std::vector<int32>& values, bool scalar,
                   const string& device) {
  NodeDef* node = AddNode(graph, name, "Const", device);
  (*node->mutable_attr())["dtype"].set_type(DT_INT32);
  Tensor tensor(DT_INT32,
                scalar ? TensorShape({})
                       : TensorShape({static_cast<int64_t>(values.size())}));
  if (scalar) {
    DCHECK_EQ(values.size(), 1);
    tensor.scalar<int32>()() = values[0];
  } else {
    auto flat = tensor.flat<int32>();
    for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i) {
      flat(i) = values[i];
    }
  }
  tensor.AsProtoTensorContent(
      (*node->mutable_attr())["value"].mutable_tensor());
  return name;
}

string SplitOutput(const string& split, int index) {
  return index == 0 ? split : absl::StrCat(split, ":", index);
}

string Rewrite(const Candidate& candidate, GraphDef* graph) {
  const string prefix =
      absl::StrCat(candidate.matmul->name(), "/broadcast_factorization");
  const string& device = candidate.matmul->device();
  const string sizes = AddIntConst(
      graph, absl::StrCat(prefix, "/split_sizes"),
      candidate.repeated_index == 0
          ? std::vector<int32>{candidate.query_width, candidate.varying_width}
          : std::vector<int32>{candidate.varying_width, candidate.query_width},
      false, device);
  const string split_axis = AddIntConst(
      graph, absl::StrCat(prefix, "/split_axis"), {0}, true, device);

  const string split_name = absl::StrCat(prefix, "/split_weights");
  NodeDef* split = AddNode(graph, split_name, "SplitV", device);
  split->add_input(candidate.weights);
  split->add_input(sizes);
  split->add_input(split_axis);
  (*split->mutable_attr())["T"].set_type(candidate.dtype);
  (*split->mutable_attr())["Tlen"].set_type(DT_INT32);
  (*split->mutable_attr())["num_split"].set_i(2);

  auto add_matmul = [&](const string& name, const string& lhs,
                        int weight_index) {
    NodeDef* node = AddNode(graph, name, "MatMul", device);
    node->add_input(lhs);
    node->add_input(SplitOutput(split_name, weight_index));
    (*node->mutable_attr())["T"].set_type(candidate.dtype);
    (*node->mutable_attr())["transpose_a"].set_b(false);
    (*node->mutable_attr())["transpose_b"].set_b(false);
  };

  const string varying_matmul = absl::StrCat(prefix, "/varying_matmul");
  add_matmul(varying_matmul, candidate.varying, candidate.varying_index);
  const string query_matmul = absl::StrCat(prefix, "/query_matmul");
  add_matmul(query_matmul, candidate.query, candidate.repeated_index);

  const string varying_shape = AddIntConst(
      graph, absl::StrCat(prefix, "/varying_shape"),
      {-1, candidate.repeats, candidate.output_width}, false, device);
  const string varying_3d = absl::StrCat(prefix, "/varying_3d");
  NodeDef* reshape = AddNode(graph, varying_3d, "Reshape", device);
  reshape->add_input(varying_matmul);
  reshape->add_input(varying_shape);
  (*reshape->mutable_attr())["T"].set_type(candidate.dtype);
  (*reshape->mutable_attr())["Tshape"].set_type(DT_INT32);

  const string expand_axis = AddIntConst(
      graph, absl::StrCat(prefix, "/expand_axis"), {1}, true, device);
  const string query_3d = absl::StrCat(prefix, "/query_3d");
  NodeDef* expand = AddNode(graph, query_3d, "ExpandDims", device);
  expand->add_input(query_matmul);
  expand->add_input(expand_axis);
  (*expand->mutable_attr())["T"].set_type(candidate.dtype);
  (*expand->mutable_attr())["Tdim"].set_type(DT_INT32);

  const string sum_name = absl::StrCat(prefix, "/sum");
  NodeDef* sum = AddNode(graph, sum_name, "AddV2", device);
  sum->add_input(varying_3d);
  sum->add_input(query_3d);
  (*sum->mutable_attr())["T"].set_type(candidate.dtype);

  const string flat_shape =
      AddIntConst(graph, absl::StrCat(prefix, "/flat_shape"),
                  {-1, candidate.output_width}, false, device);
  const string output = absl::StrCat(prefix, "/output");
  NodeDef* flat = AddNode(graph, output, "Reshape", device);
  flat->add_input(sum_name);
  flat->add_input(flat_shape);
  (*flat->mutable_attr())["T"].set_type(candidate.dtype);
  (*flat->mutable_attr())["Tshape"].set_type(DT_INT32);
  return output;
}

void ReplaceFanouts(GraphDef* graph, const string& old_node,
                    const string& replacement) {
  for (NodeDef& node : *graph->mutable_node()) {
    for (string& input : *node.mutable_input()) {
      if (NodeName(input) != old_node) continue;
      input = !input.empty() && input[0] == '^' ? absl::StrCat("^", replacement)
                                                : replacement;
    }
  }
}

}  // namespace

absl::Status BroadcastedMatMulFactorizationOptimizer::Optimize(
    Cluster* /*cluster*/, const GrapplerItem& item, GraphDef* output) {
  *output = item.graph;

  NodeIndex nodes;
  std::unordered_set<string> referenced;
  for (const NodeDef& node : item.graph.node()) {
    nodes.emplace(node.name(), &node);
    for (const string& input : node.input()) referenced.insert(NodeName(input));
  }

  bool has_candidate = false;
  for (const NodeDef& node : item.graph.node()) {
    const NodeDef* input =
        node.input_size() > 0 ? FindNode(nodes, node.input(0)) : nullptr;
    // MatMul(Concat(X, Repeat(Q, L)), W) -> MatMul(X, W_x) +
    // Broadcast(MatMul(Q, W_q), L).
    if (node.op() == "MatMul" && input != nullptr &&
        input->op() == "ConcatV2" &&
        referenced.find(node.name()) != referenced.end()) {
      has_candidate = true;
      break;
    }
  }
  if (!has_candidate) return absl::OkStatus();

  GraphProperties properties(item);
  const absl::Status inferred = properties.InferStatically(
      /*assume_valid_feeds=*/false, /*aggressive_shape_inference=*/false,
      /*include_tensor_values=*/false);
  const GraphProperties* shapes = inferred.ok() ? &properties : nullptr;
  if (!inferred.ok()) {
    VLOG(1) << "Broadcasted MatMul shape inference failed: "
            << inferred.message();
  }

  const std::unordered_set<string> preserved = item.NodesToPreserve();
  for (const NodeDef& node : item.graph.node()) {
    if (referenced.find(node.name()) == referenced.end()) continue;
    Candidate candidate;
    if (!Analyze(node, nodes, shapes, preserved, &candidate)) continue;

    const string replacement = Rewrite(candidate, output);
    ReplaceFanouts(output, node.name(), replacement);
    VLOG(1) << "Factorized broadcasted input of " << node.name()
            << " (repeat=" << candidate.repeats
            << ", width=" << candidate.query_width << ")";
  }
  return absl::OkStatus();
}

}  // namespace grappler
}  // namespace tensorflow

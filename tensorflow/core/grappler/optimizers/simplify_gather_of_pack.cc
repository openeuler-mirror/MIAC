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

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace grappler {
namespace {

bool TensorFromConst(const string& input, const NodeMap& node_map,
                     const std::unordered_set<string>& feed_nodes,
                     Tensor* tensor, const NodeDef** const_node) {
  const NodeDef* node = node_map.GetNode(input);
  if (node == nullptr || !IsConstant(*node) ||
      feed_nodes.find(node->name()) != feed_nodes.end()) {
    return false;
  }
  const auto value = node->attr().find("value");
  if (value == node->attr().end() ||
      !tensor->FromProto(value->second.tensor())) {
    return false;
  }
  *const_node = node;
  return true;
}

bool ScalarIntValue(const Tensor& tensor, int64_t* value) {
  if (tensor.dims() != 0) return false;
  if (tensor.dtype() == DT_INT32) {
    *value = tensor.flat<int32>()(0);
    return true;
  }
  if (tensor.dtype() == DT_INT64) {
    *value = tensor.flat<int64_t>()(0);
    return true;
  }
  return false;
}

bool IntVectorValues(const Tensor& tensor, std::vector<int64_t>* values) {
  if (tensor.dims() != 1 || tensor.NumElements() <= 0) return false;
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

void ForwardControlDependencies(const NodeDef& source, NodeDef* target,
                                NodeMap* node_map) {
  for (int i = source.input_size() - 1; i >= 0; --i) {
    if (!IsControlInput(source.input(i))) break;
    target->add_input(source.input(i));
    node_map->AddOutput(NodeName(source.input(i)), target->name());
  }
}

bool TrySimplify(NodeDef* gather,
                 const std::unordered_set<string>& nodes_to_preserve,
                 const std::unordered_set<string>& feed_nodes,
                 NodeMap* node_map) {
  if (gather->op() != "GatherV2" && gather->op() != "Gather") {
    return false;
  }

  LOG(INFO) << "[SimplifyGatherOfPack] inspecting Gather node: name="
            << gather->name() << ", op=" << gather->op()
            << ", inputs=" << gather->input_size();
  if (gather->input_size() < 2) {
    LOG(INFO) << "[SimplifyGatherOfPack] skipping " << gather->name()
              << ": Gather has fewer than two inputs; node="
              << gather->ShortDebugString();
    return false;
  }

  NodeDef* source = node_map->GetNode(gather->input(0));
  if (source == nullptr) {
    LOG(INFO) << "[SimplifyGatherOfPack] skipping " << gather->name()
              << ": input 0 could not be resolved: " << gather->input(0)
              << "; node=" << gather->ShortDebugString();
    return false;
  }
  if (source->op() != "Pack") {
    LOG(INFO) << "[SimplifyGatherOfPack] Gather " << gather->name()
              << " is not a Gather-of-Pack candidate: input 0="
              << gather->input(0) << " resolves to op=" << source->op();
    return false;
  }

  LOG(INFO) << "[SimplifyGatherOfPack] found candidate: gather="
            << gather->ShortDebugString()
            << "; source_pack=" << source->ShortDebugString();

  const auto skip = [&](const char* reason) {
    LOG(INFO) << "[SimplifyGatherOfPack] skipping candidate " << gather->name()
              << " over " << source->name() << ": " << reason;
    return false;
  };

  if (nodes_to_preserve.find(gather->name()) != nodes_to_preserve.end()) {
    return skip("Gather node must be preserved");
  }
  if (feed_nodes.find(source->name()) != feed_nodes.end()) {
    return skip("source Pack is fed");
  }
  if (!source->device().empty() && !gather->device().empty() &&
      source->device() != gather->device()) {
    return skip("Pack and Gather have different devices");
  }

  int batch_dims = 0;
  if (gather->op() == "GatherV2") {
    const auto batch_dims_attr = gather->attr().find("batch_dims");
    if (batch_dims_attr != gather->attr().end()) {
      batch_dims = batch_dims_attr->second.i();
    }
    if (batch_dims != 0) return skip("batch_dims is not zero");
    if (gather->input_size() < 3) return skip("GatherV2 has no axis input");
  }

  int64_t axis = 0;
  const NodeDef* axis_node = nullptr;
  if (gather->op() == "GatherV2") {
    Tensor axis_tensor;
    if (!TensorFromConst(gather->input(2), *node_map, feed_nodes, &axis_tensor,
                         &axis_node) ||
        !ScalarIntValue(axis_tensor, &axis)) {
      const NodeDef* unresolved_axis = node_map->GetNode(gather->input(2));
      LOG(INFO) << "[SimplifyGatherOfPack] axis diagnostic for "
                << gather->name() << ": input=" << gather->input(2)
                << ", resolved_node="
                << (unresolved_axis == nullptr
                        ? "<null>"
                        : unresolved_axis->ShortDebugString())
                << ", is_feed="
                << (unresolved_axis != nullptr &&
                    feed_nodes.find(unresolved_axis->name()) !=
                        feed_nodes.end());
      return skip("axis is not an unfed scalar int32/int64 Const");
    }
    LOG(INFO) << "[SimplifyGatherOfPack] axis for " << gather->name() << " is "
              << axis << "; axis_node=" << axis_node->ShortDebugString();
  }
  const auto pack_axis_attr = source->attr().find("axis");
  const auto pack_n_attr = source->attr().find("N");
  const auto pack_type_attr = source->attr().find("T");
  if (pack_axis_attr == source->attr().end() ||
      pack_n_attr == source->attr().end() ||
      pack_type_attr == source->attr().end()) {
    return skip("Pack is missing axis, N, or T");
  }
  const int64_t pack_axis = pack_axis_attr->second.i();
  // Exact comparison handles the common positive-axis form and equal negative
  // axes without requiring whole-graph shape inference.
  if (pack_axis != axis) {
    LOG(INFO) << "[SimplifyGatherOfPack] skipping candidate " << gather->name()
              << " over " << source->name() << ": axes differ (Pack axis "
              << pack_axis << ", Gather axis " << axis << ")";
    return false;
  }

  const int64_t source_input_count = pack_n_attr->second.i();
  if (source_input_count <= 0 || source_input_count > source->input_size()) {
    return skip("Pack has an invalid N attribute");
  }

  Tensor indices_tensor;
  const NodeDef* indices_node = nullptr;
  if (!TensorFromConst(gather->input(1), *node_map, feed_nodes, &indices_tensor,
                       &indices_node)) {
    const NodeDef* unresolved_indices = node_map->GetNode(gather->input(1));
    LOG(INFO) << "[SimplifyGatherOfPack] indices diagnostic for "
              << gather->name() << ": input=" << gather->input(1)
              << ", resolved_node="
              << (unresolved_indices == nullptr
                      ? "<null>"
                      : unresolved_indices->ShortDebugString())
              << ", is_feed="
              << (unresolved_indices != nullptr &&
                  feed_nodes.find(unresolved_indices->name()) !=
                      feed_nodes.end());
    return skip("indices are not an unfed Const");
  }
  std::vector<int64_t> selected_indices;
  if (!IntVectorValues(indices_tensor, &selected_indices)) {
    LOG(INFO) << "[SimplifyGatherOfPack] invalid indices tensor for "
              << gather->name() << ": dims=" << indices_tensor.dims()
              << ", elements=" << indices_tensor.NumElements()
              << ", dtype=" << DataTypeString(indices_tensor.dtype())
              << "; indices_node=" << indices_node->ShortDebugString();
    return skip("indices are not a non-empty int32/int64 vector");
  }

  LOG(INFO) << "[SimplifyGatherOfPack] candidate values for " << gather->name()
            << ": Pack axis=" << pack_axis << ", Pack N=" << source_input_count
            << ", Gather indices=" << indices_tensor.SummarizeValue(16);

  std::vector<string> selected_inputs;
  selected_inputs.reserve(selected_indices.size());
  std::unordered_set<int64_t> unique_indices;
  for (const int64_t index : selected_indices) {
    if (index < 0 || index >= source_input_count) {
      LOG(INFO) << "[SimplifyGatherOfPack] skipping candidate "
                << gather->name() << " over " << source->name() << ": index "
                << index << " is outside [0, " << source_input_count << ")";
      return false;
    }
    selected_inputs.push_back(source->input(index));
    unique_indices.insert(index);
  }
  if (unique_indices.size() >= static_cast<std::size_t>(source_input_count)) {
    LOG(INFO) << "[SimplifyGatherOfPack] skipping candidate " << gather->name()
              << " over " << source->name()
              << ": Gather uses every Pack operand (Pack operands "
              << source_input_count << ", indices " << selected_indices.size()
              << ", unique indices " << unique_indices.size() << ")";
    return false;
  }

  const NodeDef original_gather = *gather;
  node_map->RemoveInputs(gather->name());
  gather->set_op("Pack");
  gather->clear_input();
  EraseRegularNodeAttributes(gather);
  gather->mutable_attr()->erase("_kernel");
  (*gather->mutable_attr())["N"].set_i(
      static_cast<int64_t>(selected_inputs.size()));
  (*gather->mutable_attr())["T"] = pack_type_attr->second;
  (*gather->mutable_attr())["axis"] = pack_axis_attr->second;

  for (const string& input : selected_inputs) {
    gather->add_input(input);
    node_map->AddOutput(NodeName(input), gather->name());
  }
  ForwardControlDependencies(original_gather, gather, node_map);
  ForwardControlDependencies(*source, gather, node_map);
  ForwardControlDependencies(*indices_node, gather, node_map);
  if (axis_node != nullptr) {
    ForwardControlDependencies(*axis_node, gather, node_map);
  }
  DedupControlInputs(gather);

  LOG(INFO) << "[SimplifyGatherOfPack] simplified node " << gather->name()
            << " over " << source->name() << ": selected "
            << selected_inputs.size() << " outputs from "
            << unique_indices.size() << " of " << source_input_count
            << " Pack operands; rewritten_node=" << gather->ShortDebugString();
  return true;
}

}  // namespace

absl::Status SimplifyGatherOfPackOptimizer::Optimize(Cluster* /*cluster*/,
                                                     const GrapplerItem& item,
                                                     GraphDef* output) {
  LOG(INFO) << "[SimplifyGatherOfPack] optimizer entered: item_id=" << item.id
            << ", graph_nodes=" << item.graph.node_size()
            << ", fetches=" << item.fetch.size()
            << ", feeds=" << item.feed.size()
            << ", keep_ops=" << item.keep_ops.size()
            << ", init_ops=" << item.init_ops.size();
  *output = item.graph;

  const std::unordered_set<string> nodes_to_preserve = item.NodesToPreserve();
  std::unordered_set<string> feed_nodes;
  feed_nodes.reserve(item.feed.size());
  for (const auto& feed : item.feed) {
    feed_nodes.insert(NodeName(feed.first));
  }

  NodeMap node_map(output);
  int gather_nodes = 0;
  int gather_of_pack_candidates = 0;
  int simplified_nodes = 0;
  for (int i = 0; i < output->node_size(); ++i) {
    GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    NodeDef* node = output->mutable_node(i);
    if (node->op() == "Gather" || node->op() == "GatherV2") {
      ++gather_nodes;
      if (node->input_size() > 0) {
        const NodeDef* source = node_map.GetNode(node->input(0));
        if (source != nullptr && source->op() == "Pack") {
          ++gather_of_pack_candidates;
        }
      }
    }
    if (TrySimplify(node, nodes_to_preserve, feed_nodes, &node_map)) {
      ++simplified_nodes;
    }
  }
  LOG(INFO) << "[SimplifyGatherOfPack] optimizer finished: item_id=" << item.id
            << ", Gather/GatherV2 nodes=" << gather_nodes
            << ", Gather-of-Pack candidates=" << gather_of_pack_candidates
            << ", simplified=" << simplified_nodes;
  return absl::OkStatus();
}

}  // namespace grappler
}  // namespace tensorflow

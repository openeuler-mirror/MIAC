#include "tensorflow/cc/saved_model/variable_freezing.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

namespace tensorflow {
namespace internal {
namespace {

constexpr int64_t kDefaultMaxTensorBytes = 1LL * 1024 * 1024;

using FrozenValueMap = absl::flat_hash_map<std::string, Tensor>;
using NodeMap = absl::flat_hash_map<std::string, const NodeDef*>;
using Fanouts = absl::flat_hash_map<std::string, std::vector<const NodeDef*>>;

bool HasAnyToken(absl::string_view value,
                 std::initializer_list<absl::string_view> tokens) {
  for (absl::string_view token : tokens) {
    if (!token.empty() && value.find(token) != absl::string_view::npos) {
      return true;
    }
  }
  return false;
}

bool HasAnySuffix(absl::string_view value,
                  std::initializer_list<absl::string_view> suffixes) {
  for (absl::string_view suffix : suffixes) {
    if (!suffix.empty() && absl::EndsWith(value, suffix)) {
      return true;
    }
  }
  return false;
}

// Allowed promotions for freezing variables are based on variable name
// patterns, which is admittedly a bit hacky but is the most practical way to
// avoid freezing non-variable tensors without doing an expensive static
// analysis of the graph.
bool IsAllowlistedVariableName(absl::string_view name) {
  const std::string lowered_name = absl::AsciiStrToLower(std::string(name));
  const absl::string_view lowered_view(lowered_name);
  if (HasAnyToken(lowered_view,
                  {"embedding", "lookup_table", "/part_", "hash_table"})) {
    return false;
  }
  return HasAnySuffix(lowered_view, {"weight", "bias", "kernel", "_w", "/w",
                                     "_b", "/b", "beta", "gamma", "mean",
                                     "variance"});
}

// Checks if the given op mutates a variable. This is used to determine 
// if a variable is read-only and can be frozen.
bool IsMutatingVariableOp(absl::string_view op) {
  return op == "Assign" || op == "AssignAdd" || op == "AssignSub" ||
         op == "AssignVariableOp" || op == "AssignAddVariableOp" ||
         op == "AssignSubVariableOp" || op == "DestroyResourceOp" ||
         absl::StartsWith(op, "ResourceApply") ||
         absl::StartsWith(op, "ResourceScatter") ||
         absl::StartsWith(op, "Scatter");
}

bool ShouldFreezeTensor(const Tensor& tensor, int64_t max_tensor_bytes) {
  return max_tensor_bytes <= 0 || tensor.TotalBytes() <= max_tensor_bytes;
}

bool GraphContainsOp(const GraphDef& graph_def, absl::string_view op_name) {
  for (const NodeDef& node : graph_def.node()) {
    if (node.op() == op_name) {
      return true;
    }
  }
  return false;
}

// Normalizes a graph input string like "foo", "foo:0", or "^foo" down to
// the producer node name "foo".
std::string BaseNodeName(absl::string_view input) {
  return std::string(ParseTensorName(input).node());
}

// Returns the first non-control input for a node, normalized to its producer
// node name. Control inputs like "^foo" are skipped.
bool GetFirstDataInput(const NodeDef& node, std::string* input_name) {
  for (const std::string& input : node.input()) {
    const TensorId tensor_id = ParseTensorName(input);
    if (tensor_id.index() == Graph::kControlSlot) continue;
    *input_name = std::string(tensor_id.node());
    return true;
  }
  return false;
}

std::vector<std::string> GetControlInputs(const NodeDef& node) {
  std::vector<std::string> controls;
  for (const std::string& input : node.input()) {
    const TensorId tensor_id = ParseTensorName(input);
    if (tensor_id.index() == Graph::kControlSlot) {
      controls.push_back(input);
    }
  }
  return controls;
}

bool IsNodeInStack(const std::vector<std::string>& stack,
                   absl::string_view key) {
  for (const std::string& existing : stack) {
    if (absl::string_view(existing) == key) return true;
  }
  return false;
}

bool ForEachDataInputFrom(const NodeDef& consumer, absl::string_view producer,
                         const std::function<bool(int)>& visit) {
  int data_input = 0;
  for (const std::string& input : consumer.input()) {
    if (ParseTensorName(input).index() == Graph::kControlSlot) continue;
    if (BaseNodeName(input) == producer && !visit(data_input)) {
      return false;
    }
    ++data_input;
  }
  return true;
}

bool ResolveInputType(const NodeDef& node, int input_index,
                      DataType* input_type) {
  const OpDef* op_def = nullptr;
  return OpRegistry::Global()->LookUpOpDef(node.op(), &op_def).ok() &&
         InputTypeForNode(node, *op_def, input_index, input_type).ok();
}

bool IsTensorCompatibleWithNode(const Tensor& frozen_value,
                                const NodeDef& node) {
  const auto dtype_it = node.attr().find("dtype");
  if (dtype_it != node.attr().end()) {
    if (dtype_it->second.type() != frozen_value.dtype()) {
      return false;
    }
  }
  const auto type_it = node.attr().find("T");
  if (type_it != node.attr().end()) {
    if (type_it->second.type() != frozen_value.dtype()) {
      return false;
    }
  }
  if (dtype_it == node.attr().end() && type_it == node.attr().end()) {
    return false;
  }

  const auto shape_it = node.attr().find("shape");
  if (shape_it == node.attr().end()) {
    return true;
  }

  PartialTensorShape expected_shape;
  if (!GetNodeAttr(node, "shape", &expected_shape).ok()) {
    return false;
  }
  return expected_shape.IsCompatibleWith(frozen_value.shape());
}

Fanouts BuildDataFanouts(const GraphDef& graph_def) {
  Fanouts fanouts;
  for (const NodeDef& node : graph_def.node()) {
    for (const std::string& input : node.input()) {
      if (ParseTensorName(input).index() == Graph::kControlSlot) continue;
      fanouts[BaseNodeName(input)].push_back(&node);
    }
  }
  return fanouts;
}

void PreserveInternalAttrs(const NodeDef& original, NodeDef* replacement) {
  absl::flat_hash_map<std::string, AttrValue> internal_attrs;
  for (const auto& attr : original.attr()) {
    if (!attr.first.empty() && attr.first[0] == '_') {
      internal_attrs.insert(attr);
    }
  }

  replacement->clear_attr();
  for (const auto& attr : internal_attrs) {
    (*replacement->mutable_attr())[attr.first] = attr.second;
  }
}

void ReplaceNodeWithConst(const Tensor& frozen_value, NodeDef* node) {
  const std::vector<std::string> controls = GetControlInputs(*node);
  const NodeDef original = *node;

  node->set_op("Const");
  node->clear_input();
  for (const std::string& control : controls) {
    node->add_input(control);
  }

  PreserveInternalAttrs(original, node);
  (*node->mutable_attr())["dtype"].set_type(frozen_value.dtype());
  frozen_value.AsProtoTensorContent(
      (*node->mutable_attr())["value"].mutable_tensor());
}

// Decodes a Const string tensor node into an in-memory vector of C++ strings.
// RestoreV2 uses this form to store the checkpoint tensor names for each
// output slot.
absl::StatusOr<std::vector<std::string>> GetConstStringValues(
    const NodeDef& node) {
  if (node.op() != "Const") {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected Const node but found ", node.op()));
  }
  const auto dtype_it = node.attr().find("dtype");
  const auto value_it = node.attr().find("value");
  if (dtype_it == node.attr().end() || value_it == node.attr().end() ||
      dtype_it->second.type() != DT_STRING ||
      !value_it->second.has_tensor()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Const node ", node.name(),
                     " is not a string tensor const"));
  }

  const TensorProto& tensor_proto = value_it->second.tensor();
  if (!tensor_proto.string_val().empty()) {
    return std::vector<std::string>(tensor_proto.string_val().begin(),
                                    tensor_proto.string_val().end());
  }

  Tensor tensor;
  if (!tensor.FromProto(tensor_proto)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unable to deserialize const tensor ", node.name()));
  }
  if (tensor.dtype() != DT_STRING) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected string const tensor ", node.name()));
  }

  std::vector<std::string> values;
  values.reserve(tensor.NumElements());
  auto flat = tensor.flat<tstring>();
  for (int i = 0; i < flat.size(); ++i) {
    values.push_back(std::string(flat(i)));
  }
  return values;
}

absl::StatusOr<NodeMap> BuildNodeMap(const GraphDef& graph_def) {
  NodeMap node_map;
  node_map.reserve(graph_def.node_size());
  for (const NodeDef& node : graph_def.node()) {
    node_map[node.name()] = &node;
  }
  return node_map;
}

// Follows chains of Identity nodes until it reaches a real producer, so graph
// rewrites can reason about the underlying source variable instead of its
// forwarding wrappers.
std::string ResolveForwardedInputName(
    absl::string_view input_name,
    const NodeMap& node_map) {
  std::string current = BaseNodeName(input_name);
  absl::flat_hash_set<std::string> visited;
  while (true) {
    // Guard against malformed graphs with Identity cycles by refusing to
    // revisit a node we've already walked through.
    if (!visited.insert(current).second) {
      return current;
    }
    const auto it = node_map.find(current);
    if (it == node_map.end() || it->second->op() != "Identity") {
      return current;
    }

    std::string forwarded_name;
    if (!GetFirstDataInput(*it->second, &forwarded_name)) {
      return current;
    }
    current = forwarded_name;
  }
}

bool IsReadOnlyGraphResourceInput(absl::string_view producer_name,
                                  const Tensor& frozen_value,
                                  const Fanouts& fanouts,
                                  std::vector<std::string>* stack) {
  if (IsNodeInStack(*stack, producer_name)) {
    return false;
  }
  stack->push_back(std::string(producer_name));

  const auto fanout_it = fanouts.find(std::string(producer_name));
  if (fanout_it == fanouts.end()) {
    stack->pop_back();
    return true;
  }

  for (const NodeDef* consumer : fanout_it->second) {
    bool ok = true;
    ForEachDataInputFrom(*consumer, producer_name, [&](int dst_input) {
      if (consumer->op() == "VarIsInitializedOp") {
        ok = dst_input == 0;
        return ok;
      }
      if (consumer->op() == "ReadVariableOp") {
        ok = dst_input == 0 &&
             IsTensorCompatibleWithNode(frozen_value, *consumer);
        return ok;
      }
      if (consumer->op() == "Identity") {
        const auto type_it = consumer->attr().find("T");
        ok = dst_input == 0 && type_it != consumer->attr().end() &&
             type_it->second.type() == DT_RESOURCE &&
             IsReadOnlyGraphResourceInput(consumer->name(), frozen_value,
                                          fanouts, stack);
        return ok;
      }

      if (IsMutatingVariableOp(consumer->op())) {
        ok = dst_input == 0;
        return ok;
      }
      ok = false;
      return false;
    });
    if (!ok) {
      stack->pop_back();
      return false;
    }
  }

  stack->pop_back();
  return true;
}

bool IsSafeValueIdentityChain(absl::string_view producer_name,
                              const Fanouts& fanouts,
                              const Tensor& frozen_value,
                              std::vector<std::string>* stack) {
  if (IsNodeInStack(*stack, producer_name)) {
    return false;
  }
  stack->push_back(std::string(producer_name));

  const auto fanout_it = fanouts.find(std::string(producer_name));
  if (fanout_it == fanouts.end()) {
    stack->pop_back();
    return true;
  }

  for (const NodeDef* consumer : fanout_it->second) {
    bool ok = true;
    ForEachDataInputFrom(*consumer, producer_name, [&](int dst_input) {
      if (consumer->op() == "Identity") {
        ok = dst_input == 0 &&
             IsTensorCompatibleWithNode(frozen_value, *consumer) &&
             IsSafeValueIdentityChain(consumer->name(), fanouts, frozen_value,
                                      stack);
        return ok;
      }
      if (consumer->op() == "Save" || consumer->op() == "SaveV2" ||
          IsMutatingVariableOp(consumer->op())) {
        ok = false;
        return false;
      }

      DataType input_type = DT_INVALID;
      ok = ResolveInputType(*consumer, dst_input, &input_type) &&
           !IsRefType(input_type) &&
           BaseType(input_type) == frozen_value.dtype();
      return ok;
    });
    if (!ok) {
      stack->pop_back();
      return false;
    }
  }

  stack->pop_back();
  return true;
}

bool IsSafeVariableV2ToFreeze(const NodeDef& variable, const Tensor& frozen_value,
                              const Fanouts& fanouts) {
  if (!IsTensorCompatibleWithNode(frozen_value, variable)) {
    return false;
  }

  const auto fanout_it = fanouts.find(variable.name());
  if (fanout_it == fanouts.end()) {
    return true;
  }

  for (const NodeDef* consumer : fanout_it->second) {
    bool ok = true;
    ForEachDataInputFrom(*consumer, variable.name(), [&](int dst_input) {
      if (consumer->op() == "Save" || consumer->op() == "SaveV2") {
        return true;
      }

      if (IsMutatingVariableOp(consumer->op())) {
        ok = dst_input == 0;
        return ok;
      }
      if (consumer->op() == "Identity") {
        if (dst_input != 0 ||
            !IsTensorCompatibleWithNode(frozen_value, *consumer)) {
          ok = false;
          return false;
        }
        std::vector<std::string> value_stack;
        ok = IsSafeValueIdentityChain(consumer->name(), fanouts, frozen_value,
                                      &value_stack);
        return ok;
      }
      ok = false;
      return false;
    });
    if (!ok) {
      return false;
    }
  }

  return true;
}

bool IsSafeVarHandleToFreeze(const NodeDef& variable,
                             const Tensor& frozen_value,
                             const Fanouts& fanouts) {
  if (!IsTensorCompatibleWithNode(frozen_value, variable)) {
    return false;
  }

  const auto fanout_it = fanouts.find(variable.name());
  if (fanout_it == fanouts.end()) {
    return true;
  }

  std::vector<std::string> stack;
  return IsReadOnlyGraphResourceInput(variable.name(), frozen_value, fanouts,
                                      &stack);
}

std::vector<std::string> CandidateCheckpointKeys(const NodeDef& node) {
  std::vector<std::string> keys;
  const auto shared_name_it = node.attr().find("shared_name");
  const std::string shared_name =
      shared_name_it != node.attr().end() ? shared_name_it->second.s() : "";
  if (!shared_name.empty()) {
    keys.push_back(shared_name);
    keys.push_back(absl::StrCat(shared_name, "/.ATTRIBUTES/VARIABLE_VALUE"));
  }
  keys.push_back(node.name());
  keys.push_back(absl::StrCat(node.name(), "/.ATTRIBUTES/VARIABLE_VALUE"));
  return keys;
}

absl::StatusOr<std::unique_ptr<BundleReader>> OpenVariablesBundleReader(
    const std::string& export_dir) {
  const std::string variables_prefix = io::JoinPath(
      export_dir, kSavedModelVariablesDirectory, kSavedModelVariablesFilename);
  auto reader = std::make_unique<BundleReader>(Env::Default(), variables_prefix);
  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      reader->status(), "Unable to load SavedModel variables checkpoint from ",
      variables_prefix);
  return reader;
}

absl::StatusOr<bool> LookupTensorForCheckpointKeys(
    BundleReader* reader, const std::vector<std::string>& candidate_keys,
    Tensor* tensor) {
  for (const std::string& candidate_key : candidate_keys) {
    absl::Status status = reader->Lookup(candidate_key, tensor);
    if (status.ok()) {
      return true;
    }
    if (status.code() != absl::StatusCode::kNotFound) {
      return status;
    }
  }
  return false;
}

// Scans SavedModel graph that uses VariableV2, find variables that look safe to
// freeze, loads their checkpoint values.
absl::StatusOr<FrozenValueMap> LoadFrozenVariableV2Values(
    BundleReader* reader, const GraphDef& graph_def, const NodeMap& node_map,
    const Fanouts& fanouts, int64_t max_tensor_bytes) {
  FrozenValueMap frozen_values;
  // The pattern we are looking is:
  // RestoreV2 -> Assign -> VariableV2
  for (const NodeDef& node : graph_def.node()) {
    if (node.op() != "Assign" || node.input_size() < 2) continue;

    // Input 0 corresponds to destination variable, input 1 corresponds to
    // source tensor from checkpoint.
    const std::string variable_name = BaseNodeName(node.input(0));
    const auto variable_it = node_map.find(variable_name);
    if (variable_it == node_map.end() ||
        variable_it->second->op() != "VariableV2") {
      continue;
    }
    if (!IsAllowlistedVariableName(variable_name)) continue;
    if (frozen_values.find(variable_name) != frozen_values.end()) continue;

    // Default to the graph variable name, then refine it when the Assign input
    // comes from RestoreV2 by mapping that output index back to the matching
    // checkpoint tensor name listed in RestoreV2's tensor-names input.
    std::string tensor_name = variable_name;
    const std::string restore_input = node.input(1);
    const TensorId restore_tensor = ParseTensorName(restore_input);
    const int restore_output_index = restore_tensor.index();
    const std::string restore_name(restore_tensor.node());
    const auto restore_it = node_map.find(restore_name);
    if (restore_it != node_map.end() &&
        restore_it->second->op() == "RestoreV2" &&
        restore_it->second->input_size() >= 2) {
      const std::string tensor_names_const_name =
          BaseNodeName(restore_it->second->input(1));
      const auto tensor_names_const_it = node_map.find(tensor_names_const_name);
      if (tensor_names_const_it != node_map.end()) {
        // RestoreV2 input(1) is a string Const listing checkpoint keys in
        // output order, so decode it and use the restore output index to pick
        // the matching checkpoint tensor name.
        TF_ASSIGN_OR_RETURN(
            std::vector<std::string> tensor_names,
            GetConstStringValues(*tensor_names_const_it->second));
        if (restore_output_index >= 0 &&
            restore_output_index < tensor_names.size()) {
          tensor_name = tensor_names[restore_output_index];
        } else if (!tensor_names.empty()) {
          tensor_name = tensor_names[0];
        }
      }
    }

    Tensor tensor;
    // Try the resolved tensor name plus the standard V1/V2 checkpoint key
    // suffixes, tolerating NotFound so a single unmatched variable does not
    // abort the whole freezing pass. This mirrors the resource-variable path.
    std::vector<std::string> candidate_keys = {
        tensor_name,
        absl::StrCat(tensor_name, "/.ATTRIBUTES/VARIABLE_VALUE"),
        variable_name,
        absl::StrCat(variable_name, "/.ATTRIBUTES/VARIABLE_VALUE")};
    TF_ASSIGN_OR_RETURN(
        bool found,
        LookupTensorForCheckpointKeys(reader, candidate_keys, &tensor));
    if (!found || !ShouldFreezeTensor(tensor, max_tensor_bytes)) {
      continue;
    }
    if (!IsSafeVariableV2ToFreeze(*variable_it->second, tensor, fanouts)) {
      VLOG(2) << "[variable_freezing] skip VariableV2 graph_node="
              << variable_name << " reason=unsafe_or_incompatible";
      continue;
    }
    VLOG(2) << "[variable_freezing] matched VariableV2 checkpoint key="
              << tensor_name << " graph_node=" << variable_name;
    frozen_values[variable_name] = tensor;
  }
  return frozen_values;
}

// Scans resource-variable graphs for VarHandleOp nodes whose names look like
// freezeable model parameters, then loads their checkpoint values by trying
// the common checkpoint key conventions derived from each handle.
absl::StatusOr<FrozenValueMap> LoadFrozenVarHandleValues(
    BundleReader* reader, const GraphDef& graph_def, const Fanouts& fanouts,
    int64_t max_tensor_bytes) {
  FrozenValueMap frozen_values;
  for (const NodeDef& node : graph_def.node()) {
    // Only VarHandleOp nodes represent resource variables in this path.
    if (node.op() != "VarHandleOp") {
      continue;
    }
    // Resource variables may carry both a graph node name and a shared_name;
    // accept the handle when either one looks like an allowlisted weight/bias.
    const auto shared_name_it = node.attr().find("shared_name");
    const std::string shared_name =
        shared_name_it != node.attr().end() ? shared_name_it->second.s() : "";
    if ((!node.name().empty() && !IsAllowlistedVariableName(node.name())) &&
        (shared_name.empty() || !IsAllowlistedVariableName(shared_name))) {
      continue;
    }

    Tensor tensor;
    // Try the common checkpoint naming conventions for this handle until one
    // matches, filling `tensor` when successful.
    TF_ASSIGN_OR_RETURN(
      bool found,
      LookupTensorForCheckpointKeys(reader, CandidateCheckpointKeys(node),
                      &tensor));
    // Skip handles that have no checkpoint entry or whose tensor is too large
    // to freeze into the graph.
    if (!found || !ShouldFreezeTensor(tensor, max_tensor_bytes)) {
      continue;
    }
    if (!IsSafeVarHandleToFreeze(node, tensor, fanouts)) {
      VLOG(2) << "[variable_freezing] skip VarHandleOp graph_node="
              << node.name() << " reason=unsafe_or_incompatible";
      continue;
    }
    VLOG(2) << "[variable_freezing] matched VarHandleOp checkpoint graph_node="
            << node.name() << " shared_name=" << shared_name;
    // Key the frozen map by graph node name because later graph rewrites match
    // consumers against VarHandleOp node names, not checkpoint keys.
    frozen_values[node.name()] = tensor;
  }
  return frozen_values;
}

// Rewrites top-level variable-read nodes into Const nodes when their resolved
// source variable already has a frozen checkpoint value.
absl::Status RewriteTopLevelReadNodes(GraphDef* graph_def,
                                      const FrozenValueMap& frozen_values) {
  TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(*graph_def));
  for (NodeDef& node : *graph_def->mutable_node()) {
    // Identity is included because graphs often forward a variable read through
    // an Identity before the value reaches real consumers.
    if (node.op() != "ReadVariableOp" && node.op() != "Identity") {
      continue;
    }

    std::string source_name;
    if (!GetFirstDataInput(node, &source_name)) continue;
    source_name = ResolveForwardedInputName(source_name, node_map);
    const auto frozen_it = frozen_values.find(source_name);
    if (frozen_it == frozen_values.end()) continue;
    if (!IsTensorCompatibleWithNode(frozen_it->second, node)) continue;
    VLOG(2) << "[variable_freezing] rewrite top-level node=" << node.name()
            << " op=" << node.op() << " source=" << source_name;
    ReplaceNodeWithConst(frozen_it->second, &node);
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status FreezeAllowlistedVariableReads(const std::string& export_dir,
                                                 MetaGraphDef* meta_graph_def) {
  if (meta_graph_def == nullptr) {
    return absl::OkStatus();
  }
  // Snapshot the graph before any in-place rewrites for debugging.
  VLOG(1) << "[variable_freezing] graph before freeze:"
          << meta_graph_def->graph_def().DebugString();

  GraphDef* graph_def = meta_graph_def->mutable_graph_def();
  const bool has_variable_v2 = GraphContainsOp(*graph_def, "VariableV2");
  const bool has_var_handle = GraphContainsOp(*graph_def, "VarHandleOp");
  if (!has_variable_v2 && !has_var_handle) {
    VLOG(2) << "[variable_freezing] export_dir=" << export_dir
              << " matched_v1=0 matched_total=0";
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<BundleReader> reader,
                      OpenVariablesBundleReader(export_dir));
  const Fanouts fanouts = BuildDataFanouts(*graph_def);

  // Collect freezeable values from both legacy VariableV2 graphs and
  // resource-variable graphs, then merge them into one lookup table.
  // OPs logic can be found here tensorflow/core/ops/state_ops.cc
  FrozenValueMap v1_frozen_values;
  if (has_variable_v2) {
    TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(*graph_def));
    TF_ASSIGN_OR_RETURN(
        FrozenValueMap loaded_v1_frozen_values,
        LoadFrozenVariableV2Values(reader.get(), *graph_def, node_map, fanouts,
                                   kDefaultMaxTensorBytes));
    v1_frozen_values = std::move(loaded_v1_frozen_values);
  }

  FrozenValueMap frozen_values;
  if (has_var_handle) {
    TF_ASSIGN_OR_RETURN(
        FrozenValueMap loaded_var_handle_values,
        LoadFrozenVarHandleValues(reader.get(), *graph_def, fanouts,
                                  kDefaultMaxTensorBytes));
    frozen_values = std::move(loaded_var_handle_values);
  }
  for (auto& entry : v1_frozen_values) {
    frozen_values[entry.first] = std::move(entry.second);
  }
  LOG(INFO) << "[variable_freezing] export_dir=" << export_dir
      << " matched_v1=" << v1_frozen_values.size()
      << " matched_total=" << frozen_values.size();
  // Leave the graph untouched when no allowlisted checkpoint tensors matched.
  if (frozen_values.empty()) {
    return absl::OkStatus();
  }
  // Rewrite direct top-level reads.
  TF_RETURN_IF_ERROR(RewriteTopLevelReadNodes(graph_def, frozen_values));
  VLOG(1) << "[variable_freezing] graph after freeze:\n"
          << meta_graph_def->graph_def().DebugString();
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorflow
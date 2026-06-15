#include "tensorflow/cc/saved_model/variable_freezing.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.h"
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
using FrozenInputMap = absl::flat_hash_map<std::string, const Tensor*>;

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
                                     "_b", "/b", "beta", "gamma", "moving_mean",
                                     "moving_variance", "mean", "variance"});
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

bool IsCallNode(const NodeDef& node) {
	return node.op() == "PartitionedCall" ||
				 node.op() == "StatefulPartitionedCall";
}

std::string GetCalledFunctionName(const NodeDef& node) {
	const auto it = node.attr().find("f");
	if (it == node.attr().end() || !it->second.has_func()) return std::string();
	return it->second.func().name();
}

absl::StatusOr<std::string> GetConstStringValue(const NodeDef& node) {
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
		return tensor_proto.string_val(0);
	}

	Tensor tensor;
	if (!tensor.FromProto(tensor_proto)) {
		return absl::InvalidArgumentError(
				absl::StrCat("Unable to deserialize const tensor ", node.name()));
	}
	if (tensor.dtype() != DT_STRING || tensor.NumElements() != 1) {
		return absl::InvalidArgumentError(
				absl::StrCat("Expected single-element string const ", node.name()));
	}
	return std::string(tensor.flat<tstring>()(0));
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

absl::StatusOr<absl::flat_hash_map<std::string, const NodeDef*>> BuildNodeMap(
		const GraphDef& graph_def) {
	absl::flat_hash_map<std::string, const NodeDef*> node_map;
	node_map.reserve(graph_def.node_size());
	for (const NodeDef& node : graph_def.node()) {
		node_map[node.name()] = &node;
	}
	return node_map;
}

absl::flat_hash_map<std::string, const NodeDef*> BuildFunctionNodeMap(
		const FunctionDef& function) {
	absl::flat_hash_map<std::string, const NodeDef*> node_map;
	node_map.reserve(function.node_def_size());
	for (const NodeDef& node : function.node_def()) {
		node_map[node.name()] = &node;
	}
	return node_map;
}

// Follows chains of Identity nodes until it reaches a real producer, so graph
// rewrites can reason about the underlying source variable instead of its
// forwarding wrappers.
std::string ResolveForwardedInputName(
    absl::string_view input_name,
    const absl::flat_hash_map<std::string, const NodeDef*>& node_map) {
  std::string current = BaseNodeName(input_name);
  while (true) {
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

// Scans SavedModel graph that uses VariableV2, find variables that look safe to
// freeze, loads their checkpoint values.
absl::StatusOr<FrozenValueMap> LoadFrozenVariableV1Values(
    const std::string& export_dir, const GraphDef& graph_def,
    int64_t max_tensor_bytes) {
  // Skip the legacy-variable path entirely when the graph has no VariableV2
  // nodes to inspect.
  if (!GraphContainsOp(graph_def, "VariableV2")) return FrozenValueMap();

  // Open the variables checkpoint reader. This will be used to load tensor
  // values for candidate variables if their checkpoint keys can be identified
  // from the graph.
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BundleReader> reader,
                      OpenVariablesBundleReader(export_dir));

  TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(graph_def));
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
    if (frozen_values.contains(variable_name)) continue;

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
    TF_RETURN_IF_ERROR(reader->Lookup(tensor_name, &tensor));
    if (!ShouldFreezeTensor(tensor, max_tensor_bytes)) {
      continue;
    }
    VLOG(2) << "[variable_freezing] matched VariableV2 checkpoint key="
            << tensor_name << " graph_node=" << variable_name;
    frozen_values[variable_name] = tensor;
  }
  return frozen_values;
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

// Maps a call node's frozen actual arguments onto the callee function's formal
// input names. Only non-control inputs that resolve to already-frozen tensors
// are forwarded into the returned map.
template <typename FrozenLookupFn>
FrozenInputMap BuildCalleeInputsForCallNode(
    const NodeDef& call_node,
    const absl::flat_hash_map<std::string, const NodeDef*>& caller_node_map,
    const FunctionDef& callee, FrozenLookupFn&& frozen_lookup) {
  FrozenInputMap callee_inputs;
  const int arg_count =
      std::min(call_node.input_size(), callee.signature().input_arg_size());
  for (int i = 0; i < arg_count; ++i) {
	const TensorId tensor_id = ParseTensorName(call_node.input(i));
	if (tensor_id.index() == Graph::kControlSlot) continue;
    const std::string input_name =
        ResolveForwardedInputName(call_node.input(i), caller_node_map);
    const Tensor* frozen_tensor = frozen_lookup(input_name);
    if (frozen_tensor == nullptr) continue;
    callee_inputs[callee.signature().input_arg(i).name()] = frozen_tensor;
  }
  return callee_inputs;
}

// Scans resource-variable graphs for VarHandleOp nodes whose names look like
// freezeable model parameters, then loads their checkpoint values by trying
// the common checkpoint key conventions derived from each handle.
absl::StatusOr<FrozenValueMap> LoadFrozenVarHandleValues(
    const std::string& export_dir, const GraphDef& graph_def,
    int64_t max_tensor_bytes) {
  if (!GraphContainsOp(graph_def, "VarHandleOp")) return FrozenValueMap();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BundleReader> reader,
                      OpenVariablesBundleReader(export_dir));

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
        bool found, LookupTensorForCheckpointKeys(
                        reader.get(), CandidateCheckpointKeys(node), &tensor));
    // Skip handles that have no checkpoint entry or whose tensor is too large
    // to freeze into the graph.
    if (!found || !ShouldFreezeTensor(tensor, max_tensor_bytes)) {
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
    VLOG(2) << "[variable_freezing] rewrite top-level node=" << node.name()
            << " op=" << node.op() << " source=" << source_name;
    ReplaceNodeWithConst(frozen_it->second, &node);
  }
  return absl::OkStatus();
}

std::string BuildVisitedKey(absl::string_view function_name,
														const FrozenInputMap& frozen_inputs) {
	std::vector<std::string> keys;
	keys.reserve(frozen_inputs.size());
	for (const auto& entry : frozen_inputs) {
		keys.push_back(entry.first);
	}
	std::sort(keys.begin(), keys.end());
  // Canonicalize the frozen input names so revisiting the same function with
  // the same logical frozen-input set produces the same cache key.
  std::string visited_key(function_name);
  for (const std::string& key : keys) {
    absl::StrAppend(&visited_key, "|", key);
  }
  return visited_key;
}

void BuildFunctionMap(GraphDef* graph_def,
                      absl::flat_hash_map<std::string, FunctionDef*>* map) {
  map->clear();
  map->reserve(graph_def->library().function_size());
  for (FunctionDef& function :
       *graph_def->mutable_library()->mutable_function()) {
    (*map)[function.signature().name()] = &function;
  }
}

absl::Status RewriteFunctionAndDescendants(
    FunctionDef* function, const FrozenInputMap& frozen_inputs,
    absl::flat_hash_map<std::string, FunctionDef*>* function_map,
    absl::flat_hash_map<std::string, bool>* visited) {
  // Guard against revisiting the same function with the same set of frozen
  // inputs while walking nested call graphs.
  const std::string visited_key =
      BuildVisitedKey(function->signature().name(), frozen_inputs);
  if ((*visited)[visited_key]) {
    return absl::OkStatus();
  }
  (*visited)[visited_key] = true;

  // Build local name lookup for nodes inside this specific function body.
  const absl::flat_hash_map<std::string, const NodeDef*> function_node_map =
      BuildFunctionNodeMap(*function);

  for (NodeDef& node : *function->mutable_node_def()) {
    if (node.op() == "ReadVariableOp") {
      // If this read consumes one of the function inputs that is already known
      // to be frozen, rewrite the read directly into a Const node.
      std::string input_name;
      if (GetFirstDataInput(node, &input_name)) {
        input_name = ResolveForwardedInputName(input_name, function_node_map);
        const auto frozen_it = frozen_inputs.find(input_name);
        if (frozen_it != frozen_inputs.end()) {
          VLOG(2) << "[variable_freezing] rewrite function node="
                    << node.name()
                    << " function=" << function->signature().name()
                    << " source=" << input_name;
          ReplaceNodeWithConst(*frozen_it->second, &node);
        }
      }
      continue;
    }

    if (!IsCallNode(node)) continue;

    const std::string callee_name = GetCalledFunctionName(node);
    if (callee_name.empty()) continue;
    const auto callee_it = function_map->find(callee_name);
    if (callee_it == function_map->end()) continue;

    // Translate any frozen actual arguments at this call site into the callee's
    // formal input names, then recurse only when something frozen is actually
    // being passed through.
    FrozenInputMap callee_inputs = BuildCalleeInputsForCallNode(
        node, function_node_map, *callee_it->second,
        [&frozen_inputs](absl::string_view input_name) -> const Tensor* {
          const auto frozen_it = frozen_inputs.find(std::string(input_name));
          return frozen_it == frozen_inputs.end() ? nullptr : frozen_it->second;
        });

    if (!callee_inputs.empty()) {
      TF_RETURN_IF_ERROR(RewriteFunctionAndDescendants(
          callee_it->second, callee_inputs, function_map, visited));
    }
  }

  return absl::OkStatus();
}

// Seeds function-body rewrites from top-level PartitionedCall nodes by mapping
// frozen top-level inputs onto each callee's formal arguments, then recursively
// rewriting ReadVariableOp nodes inside those called functions.
absl::Status RewriteCapturedFunctionReads(GraphDef* graph_def,
                                          const FrozenValueMap& frozen_values) {
  // Build quick lookup tables for top-level nodes and library functions.
  TF_ASSIGN_OR_RETURN(const auto node_map, BuildNodeMap(*graph_def));
  absl::flat_hash_map<std::string, FunctionDef*> function_map;
  BuildFunctionMap(graph_def, &function_map);

  absl::flat_hash_map<std::string, bool> visited;
  for (const NodeDef& node : graph_def->node()) {
    // Only call nodes can pass frozen top-level values into function bodies.
    if (!IsCallNode(node)) continue;

    const std::string callee_name = GetCalledFunctionName(node);
    if (callee_name.empty()) continue;
    const auto callee_it = function_map.find(callee_name);
    if (callee_it == function_map.end()) continue;

    // Convert any frozen top-level call arguments into the callee's formal
    // input names so RewriteFunctionAndDescendants can rewrite reads inside
    // that function body.
    FrozenInputMap callee_inputs = BuildCalleeInputsForCallNode(
        node, node_map, *callee_it->second,
        [&frozen_values](absl::string_view input_name) -> const Tensor* {
          const auto frozen_it = frozen_values.find(std::string(input_name));
          return frozen_it == frozen_values.end() ? nullptr
                                                  : &frozen_it->second;
        });

    if (!callee_inputs.empty()) {
      TF_RETURN_IF_ERROR(RewriteFunctionAndDescendants(
          callee_it->second, callee_inputs, &function_map, &visited));
    }
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
  // Collect freezeable values from both legacy VariableV2 graphs and
  // resource-variable graphs, then merge them into one lookup table.
  // OPs logic can be found here tensorflow/core/ops/state_ops.cc
  TF_ASSIGN_OR_RETURN(FrozenValueMap v1_frozen_values,
                      LoadFrozenVariableV1Values(export_dir, *graph_def,
                                                 kDefaultMaxTensorBytes));
  TF_ASSIGN_OR_RETURN(FrozenValueMap frozen_values,
                      LoadFrozenVarHandleValues(export_dir, *graph_def,
                                                kDefaultMaxTensorBytes));
  for (auto& entry : v1_frozen_values) {
    frozen_values[entry.first] = std::move(entry.second);
  }
  VLOG(1) << "[variable_freezing] export_dir=" << export_dir
          << " matched_v1=" << v1_frozen_values.size()
          << " matched_total=" << frozen_values.size();
  // Leave the graph untouched when no allowlisted checkpoint tensors matched.
  if (frozen_values.empty()) {
    return absl::OkStatus();
  }
  // Rewrite direct top-level reads first, then propagate the same frozen
  // values into function bodies reached through PartitionedCall nodes.
  TF_RETURN_IF_ERROR(RewriteTopLevelReadNodes(graph_def, frozen_values));
  TF_RETURN_IF_ERROR(RewriteCapturedFunctionReads(graph_def, frozen_values));
  VLOG(1) << "[variable_freezing] graph after freeze:\n"
          << meta_graph_def->graph_def().DebugString();
  return absl::OkStatus();
}

}  // namespace internal
}  // namespace tensorflow

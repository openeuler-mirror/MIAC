/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"

#include "absl/strings/str_cat.h"
#include <string>
#include <utility>
#include <variant>

namespace tensorflow {
namespace {
using Signature = DeviceCompilationClusterSignature;
using ConstantTensor = Signature::ConstantTensor;
using TensorTypeAndShape = Signature::TensorTypeAndShape;

// Functor that converts a Signature's arg to a human readable string.
struct SignatureHumanStringAppender {
  explicit SignatureHumanStringAppender(std::string* dest) : dest(dest) {}
  std::string* dest;
  void operator()(const ConstantTensor& arg) {
    absl::StrAppend(dest, "; ", arg.value.DebugString());
    if (!arg.contents.empty()) {
      absl::StrAppend(dest, " contents=[");
      for (int i = 0; i < arg.contents.size(); ++i) {
        if (i > 0) absl::StrAppend(dest, ",");
        absl::StrAppend(dest, arg.contents[i].DebugString());
      }
      absl::StrAppend(dest, "]");
    }
  }
  void operator()(const TensorTypeAndShape& arg) {
    absl::StrAppend(dest, ",", DataTypeString(arg.first));
    absl::StrAppend(dest, " [", absl::StrJoin(arg.second, ","), "]");
  }
};

// Functor that compares the arg values of two different signatures. Returns
// true when the args are not equal.
struct SignatureNotEqual {
  bool operator()(const ConstantTensor& arg, const ConstantTensor& other) {
    if (arg.value.dtype() != other.value.dtype() ||
        arg.value.shape() != other.value.shape() ||
        arg.value.tensor_data() != other.value.tensor_data() ||
        arg.contents.size() != other.contents.size()) {
      return true;
    }
    for (int i = 0; i < arg.contents.size(); ++i) {
      if (arg.contents[i].SerializeAsString() !=
          other.contents[i].SerializeAsString()) {
        return true;
      }
    }
    return false;
  }
  bool operator()(const TensorTypeAndShape& arg,
                  const TensorTypeAndShape& other) {
    return arg.first != other.first || arg.second != other.second;
  }
  bool operator()(const ConstantTensor& arg, const TensorTypeAndShape& other) {
    return true;
  }
  bool operator()(const TensorTypeAndShape& arg, const ConstantTensor& other) {
    return true;
  }
};

// Functor that incrementally computes a Signature's hash given its current hash
// and one of its args.
struct SignatureHashCombiner {
  explicit SignatureHashCombiner(const uint64 h) : h(h) {}
  uint64 h;
  uint64 operator()(const ConstantTensor& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.value.dtype())));
    h = Hash64Combine(
        h, Hash64(arg.value.tensor_data().data(), arg.value.tensor_data().size()));
    for (int dim = 0; dim < arg.value.dims(); ++dim) {
      h = Hash64Combine(h, std::hash<int>()(arg.value.dim_size(dim)));
    }
    for (const xla::ExpressionProto& expr : arg.contents) {
      std::string serialized = expr.SerializeAsString();
      h = Hash64Combine(h, Hash64(serialized.data(), serialized.size()));
    }
    return h;
  }
  uint64 operator()(const TensorTypeAndShape& arg) {
    h = Hash64Combine(h, std::hash<int>()(static_cast<int>(arg.first)));
    h = Hash64Combine(h, std::hash<int>()(arg.second.size()));
    for (int dim : arg.second) {
      h = Hash64Combine(h, std::hash<int>()(dim));
    }
    return h;
  }
};
}  // namespace

// Compute a string signature which encodes the shapes of the
// arguments in the supplied list.
std::string Signature::HumanString() const {
  std::string result = name;
  for (const auto& arg : args) {
    std::visit(SignatureHumanStringAppender(&result), arg);
  }
  return result;
}

bool Signature::operator==(const Signature& other) const {
  if (name != other.name) return false;
  if (args.size() != other.args.size()) return false;
  for (int i = 0, end = args.size(); i < end; ++i) {
    if (std::visit(SignatureNotEqual(), args[i], other.args[i])) {
      return false;
    }
  }
  return true;
}

uint64 Signature::Hash::operator()(const Signature& signature) const {
  uint64 h = std::hash<string>()(signature.name);
  for (const auto& arg : signature.args) {
    h = std::visit(SignatureHashCombiner(h), arg);
  }
  return h;
}

absl::StatusOr<Signature> Signature::Build(
    const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args) {
  Signature signature;
  signature.name = Canonicalize(function.name(), AttrSlice(&function.attr()));

  for (const XlaCompiler::Argument& arg : args) {
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstant:
      case XlaCompiler::Argument::kConstantResource:
        signature.args.push_back(
            ConstantTensor{arg.constant_value, arg.constant_value_expressions});
        break;
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kResource:
        signature.args.push_back(
            TensorTypeAndShape(arg.type, arg.DimensionSizesAsInlinedVector()));
        break;
      default:
        return errors::InvalidArgument(
            "Unhandled argument kind in XlaCompilationCache: ",
            arg.HumanString());
    }
  }
  return std::move(signature);
}

}  // namespace tensorflow

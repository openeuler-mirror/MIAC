/* Copyright 2026 The TensorFlow Authors.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "xla/shape_expr.h"

namespace tensorflow {

// TensorFlow shape inference and XLA use the same owning expression value.
// TensorFlow-specific helpers below only bridge TensorShapeProto's protobuf.
using DimExpr = xla::DExpr;

DimExpr DimExprFromProto(const ExpressionProto& proto);
void DimExprToProto(const DimExpr& expr, ExpressionProto* proto);
std::string DimExprDebugString(const DimExpr& expr);

// Simplifies through xla::DExpr and stores the returned value in `arena`.
DimExpr* SimplifyExpr(DimExpr* expr,
                      std::vector<std::unique_ptr<DimExpr>>* arena);

bool TensorShapeExpressionsEnabled();
void SetTensorShapeExpressionsEnabledForTesting(std::optional<bool> enabled);
bool IsDynamicDimExpr(const ExpressionProto& proto);
bool HasDynamicDimExprs(const TensorShapeProto& proto);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_SHAPE_EXPR_H_

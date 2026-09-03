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

#include "tensorflow/compiler/jit/xla_batch_matcher.h"

#include <vector>

#include <gtest/gtest.h>

namespace tensorflow {
namespace {

TEST(XlaBatchMatcherTest, ReusesGeneratedFactorBasedBucket) {
  XlaBatchMatcher matcher;

  EXPECT_EQ(matcher.get_xla_compile_batch("cluster", 100), 200);
  EXPECT_EQ(matcher.get_xla_compile_batch("cluster", 101), 200);
  EXPECT_EQ(matcher.get_all_batches("cluster"),
            (std::vector<int64_t>{200}));
}

TEST(XlaBatchMatcherTest, KeepsGeneratedBucketsPerCluster) {
  XlaBatchMatcher matcher;

  EXPECT_EQ(matcher.get_xla_compile_batch("first", 100), 200);
  EXPECT_EQ(matcher.get_xla_compile_batch("second", 101), 202);
  EXPECT_EQ(matcher.get_all_batches("first"),
            (std::vector<int64_t>{200}));
  EXPECT_EQ(matcher.get_all_batches("second"),
            (std::vector<int64_t>{202}));
}

}  // namespace
}  // namespace tensorflow

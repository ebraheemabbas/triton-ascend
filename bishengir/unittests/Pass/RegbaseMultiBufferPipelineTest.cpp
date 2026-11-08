//===- RegbaseMultiBufferPipelineTest.cpp - buffer policy propagation
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/HIVM/Pipelines/regbase/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <string>

using namespace mlir;

namespace {

class RegbaseMultiBufferPipelineTest
    : public ::testing::TestWithParam<unsigned> {};

// Test the public pipeline builders, rather than the standalone marker: this
// catches losing a CLI/fallback policy while constructing either pipeline
// stage.
TEST_P(RegbaseMultiBufferPipelineTest, ForwardsEveryPerSpaceCombination) {
  const unsigned mask = GetParam();
  hivm::regbase::HIVMPipelineOptions options;
  options.target = hacc::TargetDevice::Ascend950PR_9579;
  options.enableMixedCV = true;
  options.enableAutoMultiBuffer = true;
  options.enablePreload = true;
  options.setCVPipelineMode = CVPipelineMode::Skew;
  options.enableAutoBindSubBlock = true;
  options.enableAutoBlockifyLoop = true;
  options.enableTritonKernelCompile = true;
  options.limitAutoMultiBufferOfLocalBuffer = MultiBufferStrategy::NO_LIMIT;
  options.limitAutoMultiBufferBuffer = MultiBufferStrategy::NO_LIMIT;
  options.disableMultiBufferOnUB = (mask & 1) != 0;
  options.disableMultiBufferOnL0C = (mask & 2) != 0;
  options.disableMultiBufferOnL1 = (mask & 4) != 0;

  // The normal compiler allows the builders to nest func.func passes.
  OpPassManager manager("builtin.module", OpPassManager::Nesting::Implicit);
  hivm::regbase::buildHIVMTensorOptimizations(manager, options);
  hivm::regbase::buildLowerHIVMPipelines(manager, options);
  std::string pipeline;
  llvm::raw_string_ostream stream(pipeline);
  manager.printAsTextualPipeline(stream);
  stream.flush();

  llvm::StringRef remaining(pipeline);
  unsigned markers = 0;
  while (true) {
    const size_t begin = remaining.find("hivm-mark-multi-buffer");
    if (begin == llvm::StringRef::npos)
      break;
    remaining = remaining.drop_front(begin);
    const size_t end = remaining.find('}');
    ASSERT_NE(end, llvm::StringRef::npos) << remaining.str();
    llvm::StringRef marker = remaining.take_front(end + 1);
    auto isEnabled = [&](llvm::StringRef name) {
      const std::string prefix = name.str() + "=";
      return marker.contains(prefix + "true") || marker.contains(prefix + "1");
    };
    EXPECT_EQ(isEnabled("disable-multi-buffer-on-ub"), (mask & 1) != 0)
        << marker.str();
    EXPECT_EQ(isEnabled("disable-multi-buffer-on-l0c"), (mask & 2) != 0)
        << marker.str();
    EXPECT_EQ(isEnabled("disable-multi-buffer-on-l1"), (mask & 4) != 0)
        << marker.str();
    EXPECT_TRUE(isEnabled("enable-auto")) << marker.str();
    EXPECT_TRUE(marker.contains("limit-mix-auto-multi-buffer-buffer=no-limit"))
        << marker.str();
    ++markers;
    remaining = remaining.drop_front(end + 1);
  }
  // This explicit Skew configuration marks before CVP, after CVP, and again
  // after bufferization. Check every occurrence, including the repeated stage.
  EXPECT_EQ(markers, 3u) << pipeline;
}

INSTANTIATE_TEST_SUITE_P(AllPerSpacePolicies, RegbaseMultiBufferPipelineTest,
                         ::testing::Values(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u));

} // namespace

//===- CVPCubeResourcePlanTest.cpp ----------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "bishengir/Dialect/Utils/CVPCubeResourcePlan.h"
#include "gtest/gtest.h"
#include <limits>

using namespace bishengir::cvp;

TEST(CVPCubeResourcePlan, AdmittedGeometryUsesUnusedHalfBankRegions) {
  auto p = deriveQCacheResources(65536, 65536, 128, 256, 64, 2);
  ASSERT_TRUE(p.valid);
  EXPECT_EQ(p.qkChunkK, 64u);
  EXPECT_EQ(p.pvChunkK, 128u);
  EXPECT_EQ(p.pvChunks, 2u);
  EXPECT_EQ(p.qBytes, 16384u);
  EXPECT_EQ(p.pvBBytes, 16384u);
  EXPECT_EQ(p.pvB[0].begin, 0u);
  EXPECT_EQ(p.pvB[1].begin, 32768u);
  EXPECT_EQ(p.q[0].begin, 16384u);
  EXPECT_EQ(p.q[1].begin, 49152u);
  EXPECT_EQ(p.q[1].end, 65536u);
}

TEST(CVPCubeResourcePlan, EveryQRegionIsDisjointFromBothPVRegions) {
  auto p = kA5HD64QCacheResources;
  for (unsigned q = 0; q < 2; ++q)
    for (unsigned pv = 0; pv < 2; ++pv)
      EXPECT_FALSE(p.q[q].overlaps(p.pvB[pv]));
  EXPECT_FALSE(p.q[0].overlaps(p.q[1]));
  EXPECT_FALSE(p.a[0].overlaps(p.a[1]));
}

TEST(CVPCubeResourcePlan, InsufficientBDoesNotDropACacheBank) {
  EXPECT_FALSE(deriveQCacheResources(65536, 32768, 128, 256, 64, 2).valid);
  EXPECT_FALSE(deriveQCacheResources(65536, 49152, 128, 256, 64, 2).valid);
}

TEST(CVPCubeResourcePlan, QMustFitOneMatrixChunk) {
  EXPECT_FALSE(deriveQCacheResources(32768, 65536, 128, 256, 64, 2).valid);
}

TEST(CVPCubeResourcePlan, HD128IsNotSilentlyAdmitted) {
  EXPECT_FALSE(supportedQCacheGeometry(128, 256, 128, 2));
  EXPECT_FALSE(deriveQCacheResources(65536, 65536, 128, 256, 128, 2).valid);
}

TEST(CVPCubeResourcePlan, TemplateTargetMustAgreeEvenWhenCapacityFits) {
  EXPECT_TRUE(deriveQCacheResources(65536, 131072, 128, 256, 64, 2).valid);
  EXPECT_FALSE(supportedQCacheTarget(65536, 131072));
  EXPECT_TRUE(supportedQCacheTarget(65536, 65536));
}

TEST(CVPCubeResourcePlan, RejectsZeroAndMisalignedInputs) {
  EXPECT_FALSE(deriveQCacheResources(0, 65536, 128, 256, 64, 2).valid);
  EXPECT_FALSE(deriveQCacheResources(65536, 0, 128, 256, 64, 2).valid);
  EXPECT_FALSE(deriveQCacheResources(65535, 65536, 128, 256, 64, 2).valid);
  EXPECT_FALSE(deriveQCacheResources(65536, 65536, 127, 256, 64, 2).valid);
  EXPECT_FALSE(deriveQCacheResources(65536, 65536, 128, 256, 64, 0).valid);
}

TEST(CVPCubeResourcePlan, CheckedArithmeticRejectsOverflow) {
  const uint64_t maximum = std::numeric_limits<uint64_t>::max();
  uint64_t result = 0;
  EXPECT_FALSE(checkedMul(maximum, 2, result));
  EXPECT_FALSE(alignBytes(maximum, result));
  EXPECT_FALSE(
      deriveQCacheResources(65536, 65536, maximum - 15, 256, 64, 2).valid);
}

TEST(CVPCubeResourcePlan, TouchingAndEmptyRangesDoNotOverlap) {
  EXPECT_FALSE((ByteRange{0, 16}.overlaps({16, 32})));
  EXPECT_TRUE((ByteRange{0, 17}.overlaps({16, 32})));
  EXPECT_FALSE((ByteRange{1, 1}.overlaps({0, 32})));
  EXPECT_FALSE((ByteRange{8, 4}.overlaps({0, 32})));
}

TEST(CVPCubeResourcePlan, GeometryAdmissionIsExplicit) {
  EXPECT_TRUE(supportedQCacheGeometry(128, 256, 64, 2));
  EXPECT_FALSE(supportedQCacheGeometry(128, 128, 64, 2));
  EXPECT_FALSE(supportedQCacheGeometry(64, 256, 64, 2));
  EXPECT_FALSE(supportedQCacheGeometry(128, 256, 64, 4));
}

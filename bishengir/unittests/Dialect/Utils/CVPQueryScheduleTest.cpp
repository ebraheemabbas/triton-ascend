//===- CVPQueryScheduleTest.cpp - Logical query schedule tests
//--------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/HIVM/Transforms/regbase/CVPQuerySchedule.h"
#include "gtest/gtest.h"
#include <array>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

using namespace mlir::hivm;

namespace {
using Work = std::pair<int64_t, int64_t>;

TEST(CVPQueryScheduleTest, PreservesOriginalAssignmentsAndCompleteStageTails) {
  for (int64_t bn : {128, 256}) {
    for (int64_t tiles : {2, 3, 4, 5, 6, 8}) {
      for (auto [start, upper, step] :
           {std::tuple<int64_t, int64_t, int64_t>{9, 9, 3},
            {12, 9, 3},
            {7, 8, 3},
            {2, 6, 3},
            {1, 12, 3},
            {0, 11, 4},
            {3, 11, 4},
            {-5, 6, 4}}) {
        std::string reason;
        auto schedule =
            CVPQuerySchedule::create(tiles * bn, bn, upper, step, reason);
        ASSERT_TRUE(schedule) << reason;
        std::vector<Work> expected;
        int64_t queries = 0;
        for (int64_t task = start; task < upper; task += step) {
          ++queries;
          for (int64_t key = 0; key < tiles * bn; key += bn)
            expected.emplace_back(task, key);
        }
        EXPECT_EQ(schedule->getQueryCount(start), queries);
        EXPECT_EQ(schedule->getWorkEnd(start), int64_t(expected.size()) * bn);
        auto end = schedule->getActiveIssueEnd(start);
        ASSERT_TRUE(end);
        std::array<std::vector<Work>, CVPQuerySchedule::kStageCount> actual;
        std::array<int64_t, CVPQuerySchedule::kStageCount> first{}, last{};
        for (int64_t tick = -4; tick < *end + 4; ++tick) {
          for (unsigned stage = 0; stage < CVPQuerySchedule::kStageCount;
               ++stage) {
            auto coordinate = schedule->getStageCoordinate(
                start, tick, static_cast<CVPQueryStage>(stage));
            if (!coordinate)
              continue;
            EXPECT_GE(tick, 0);
            EXPECT_LT(tick, *end);
            EXPECT_EQ(coordinate->querySlot, coordinate->queryOrdinal % 2);
            EXPECT_EQ(coordinate->localKey, coordinate->keyTile * bn);
            first[stage] += coordinate->firstKey;
            last[stage] += coordinate->lastKey;
            actual[stage].emplace_back(coordinate->queryTask,
                                       coordinate->localKey);
          }
        }
        for (unsigned stage = 0; stage < CVPQuerySchedule::kStageCount;
             ++stage) {
          EXPECT_EQ(actual[stage], expected);
          EXPECT_EQ(first[stage], queries);
          EXPECT_EQ(last[stage], queries);
        }
        EXPECT_EQ(*end == 0, expected.empty());
      }
    }
  }
}

TEST(CVPQueryScheduleTest, UnequalCoresCoverEachOriginalTaskOnce) {
  std::string reason;
  auto schedule = CVPQuerySchedule::create(6 * 128, 128, 11, 4, reason);
  ASSERT_TRUE(schedule);
  std::array<int, 11> seen{};
  for (int64_t core = 0; core < 4; ++core) {
    auto count = schedule->getQueryCount(core);
    ASSERT_TRUE(count);
    EXPECT_EQ(*count, core == 3 ? 2 : 3);
    for (int64_t ordinal = 0; ordinal < *count; ++ordinal) {
      auto first = schedule->getKeyLane(core, ordinal, 0, 0, 1);
      ASSERT_TRUE(first);
      ASSERT_GE(first->queryTask, 0);
      ASSERT_LT(first->queryTask, int64_t(seen.size()));
      ++seen[first->queryTask];
    }
  }
  for (int count : seen)
    EXPECT_EQ(count, 1);
}

TEST(CVPQueryScheduleTest, KeyUnrollLanesPreservePartialBodyBoundaries) {
  // These are logical lane tests, not evidence of an ordinary Unroll emitter.
  for (int64_t tiles : {2, 3, 4, 5, 6, 8}) {
    std::string reason;
    auto schedule = CVPQuerySchedule::create(tiles * 128, 128, 9, 4, reason);
    ASSERT_TRUE(schedule);
    for (unsigned unroll : {1u, 2u, 4u}) {
      for (int64_t ordinal = 0; ordinal < 3; ++ordinal) {
        std::vector<int64_t> visited;
        unsigned first = 0, last = 0;
        for (int64_t base = 0; base < tiles; base += unroll) {
          auto validLanes = schedule->getValidKeyLanes(base * 128, unroll);
          ASSERT_TRUE(validLanes);
          unsigned visitedInBody = 0;
          for (unsigned lane = 0; lane < unroll; ++lane) {
            auto coordinate =
                schedule->getKeyLane(0, ordinal, base, lane, unroll);
            if (base + lane >= tiles) {
              EXPECT_FALSE(coordinate);
              continue;
            }
            ASSERT_TRUE(coordinate);
            EXPECT_EQ(coordinate->queryOrdinal, ordinal);
            EXPECT_EQ(coordinate->queryTask, 4 * ordinal);
            visited.push_back(coordinate->localKey);
            ++visitedInBody;
            first += coordinate->firstKey;
            last += coordinate->lastKey;
          }
          EXPECT_EQ(*validLanes, visitedInBody);
        }
        std::vector<int64_t> expected;
        for (int64_t key = 0; key < tiles * 128; key += 128)
          expected.push_back(key);
        EXPECT_EQ(visited, expected);
        EXPECT_EQ(first, 1u);
        EXPECT_EQ(last, 1u);
      }
    }
  }
}

TEST(CVPQueryScheduleTest, QueryOverlapKeepsOutputHandoverInOrder) {
  for (int64_t tiles : {2, 3, 4, 5, 6, 8}) {
    std::string reason;
    auto schedule = CVPQuerySchedule::create(tiles * 256, 256, 2, 1, reason);
    ASSERT_TRUE(schedule);
    auto nextQK = schedule->getStageCoordinate(0, tiles, CVPQueryStage::QK);
    auto nextP =
        schedule->getStageCoordinate(0, tiles + 1, CVPQueryStage::Probability);
    auto previousLastO =
        schedule->getStageCoordinate(0, tiles + 2, CVPQueryStage::Output);
    auto nextFirstO =
        schedule->getStageCoordinate(0, tiles + 3, CVPQueryStage::Output);
    ASSERT_TRUE(nextQK && nextP && previousLastO && nextFirstO);
    EXPECT_EQ(nextQK->queryOrdinal, 1);
    EXPECT_TRUE(nextQK->firstKey);
    EXPECT_EQ(nextP->queryOrdinal, 1);
    EXPECT_TRUE(nextP->firstKey);
    EXPECT_EQ(previousLastO->queryOrdinal, 0);
    EXPECT_TRUE(previousLastO->lastKey);
    EXPECT_EQ(nextFirstO->queryOrdinal, 1);
    EXPECT_TRUE(nextFirstO->firstKey);
    // This is issue order only. DMA completion/release is a separate proof.
  }
}

TEST(CVPQueryScheduleTest, RejectsUnsupportedOrMalformedDomains) {
  std::string reason;
  EXPECT_FALSE(CVPQuerySchedule::create(128, 128, 8, 1, reason));
  EXPECT_EQ(reason, "two query-state slots require at least two key tiles");
  for (auto [keys, bn, upper, step] : {std::array<int64_t, 4>{0, 128, 8, 1},
                                       {257, 128, 8, 1},
                                       {256, 0, 8, 1},
                                       {256, 128, 0, 1},
                                       {256, 128, 8, 0},
                                       {256, 128, 8, -1},
                                       {int64_t{1} << 32, 128, 8, 1}}) {
    EXPECT_FALSE(CVPQuerySchedule::create(keys, bn, upper, step, reason));
    EXPECT_FALSE(reason.empty());
  }
  auto schedule = CVPQuerySchedule::create(768, 128, 8, 1, reason);
  ASSERT_TRUE(schedule);
  EXPECT_TRUE(reason.empty());
  EXPECT_FALSE(schedule->getQueryCount(int64_t{1} << 32));
  EXPECT_FALSE(
      schedule->getStageCoordinate(0, 0, static_cast<CVPQueryStage>(4)));
  EXPECT_FALSE(schedule->getKeyLane(0, -1, 0, 0, 1));
  EXPECT_FALSE(schedule->getKeyLane(0, 8, 0, 0, 1));
  EXPECT_FALSE(schedule->getKeyLane(0, 0, 0, 1, 1));
  EXPECT_FALSE(schedule->getKeyLane(0, 0, 0, 0, 3));
  EXPECT_FALSE(schedule->getKeyLane(0, 0, 1, 0, 4));
  EXPECT_FALSE(schedule->getKeyLane(0, 0, 6, 0, 2));
  EXPECT_FALSE(schedule->getValidKeyLanes(0, 3));
  EXPECT_FALSE(schedule->getValidKeyLanes(1, 2));
  EXPECT_FALSE(schedule->getValidKeyLanes(128, 4));
  EXPECT_FALSE(schedule->getValidKeyLanes(768, 2));
}

TEST(CVPQueryScheduleTest, WidensSignedI32AssignmentBeforeCounting) {
  std::string reason;
  int64_t minimum = std::numeric_limits<int32_t>::min();
  int64_t maximum = std::numeric_limits<int32_t>::max();
  auto schedule = CVPQuerySchedule::create(256, 128, maximum, maximum, reason);
  ASSERT_TRUE(schedule) << reason;
  EXPECT_EQ(schedule->getQueryCount(minimum), 3);
  auto last = schedule->getKeyLane(minimum, 2, 0, 0, 1);
  ASSERT_TRUE(last);
  EXPECT_EQ(last->queryTask, maximum - 1);
  EXPECT_EQ(schedule->getWorkEnd(minimum), 3 * 256);
}

TEST(CVPQueryScheduleTest, QueryUnrollFactorsHaveWidenedFinalGroupBounds) {
  std::string reason;
  for (int64_t tiles : {2, 3, 4, 5, 6, 8}) {
    auto schedule = CVPQuerySchedule::create(tiles * 128, 128, 11, 4, reason);
    ASSERT_TRUE(schedule);
    for (unsigned factor : {1u, 2u, 4u})
      EXPECT_TRUE(schedule->canEmitQueryUnroll(factor));
    for (unsigned factor : {0u, 3u, 8u})
      EXPECT_FALSE(schedule->canEmitQueryUnroll(factor));
  }
  int64_t maximum = std::numeric_limits<int32_t>::max();
  auto wide = CVPQuerySchedule::create(maximum - 1, (maximum - 1) / 2, maximum,
                                       1, reason);
  ASSERT_TRUE(wide) << reason;
  EXPECT_TRUE(wide->canEmitQueryUnroll(4));
}
} // namespace

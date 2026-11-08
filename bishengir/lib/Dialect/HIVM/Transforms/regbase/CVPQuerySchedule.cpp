//===- CVPQuerySchedule.cpp - Logical query/key coordinates
//----------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/HIVM/Transforms/regbase/CVPQuerySchedule.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <algorithm>
#include <limits>

using namespace mlir::hivm;

namespace {
bool isI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}
} // namespace

std::optional<CVPQuerySchedule>
CVPQuerySchedule::create(int64_t keyCount, int64_t keyStep, int64_t queryUpper,
                         int64_t queryStep, std::string &reason) {
  reason.clear();
  if (keyCount <= 0 || keyStep <= 0 || queryUpper <= 0 || queryStep <= 0 ||
      !isI32(keyCount) || !isI32(keyStep) || !isI32(queryUpper) ||
      !isI32(queryStep) || keyCount % keyStep) {
    reason = "unproved static key domain or persistent query stride";
    return std::nullopt;
  }
  // Probability leads O by two key ticks. A one-tile query crosses two query
  // boundaries during that lag and requires a different state-storage plan.
  if (keyCount / keyStep < 2) {
    reason = "two query-state slots require at least two key tiles";
    return std::nullopt;
  }
  CVPQuerySchedule schedule(keyCount, keyStep, queryUpper, queryStep);
  auto maximumWork = schedule.getWorkEnd(std::numeric_limits<int32_t>::min());
  auto conservativeDrain = llvm::checkedMul(int64_t{kStageCount}, keyStep);
  if (!maximumWork || !conservativeDrain ||
      !llvm::checkedAdd(*maximumWork, *conservativeDrain)) {
    reason = "continuous-query iteration arithmetic can overflow i64";
    return std::nullopt;
  }
  return schedule;
}

std::optional<int64_t>
CVPQuerySchedule::getQueryCount(int64_t queryStart) const {
  if (!isI32(queryStart))
    return std::nullopt;
  if (queryStart >= queryUpper)
    return 0;
  int64_t remaining = queryUpper - queryStart;
  return remaining / queryStep + (remaining % queryStep != 0);
}

std::optional<int64_t> CVPQuerySchedule::getWorkEnd(int64_t queryStart) const {
  auto count = getQueryCount(queryStart);
  return count ? llvm::checkedMul(*count, keyCount) : std::nullopt;
}

std::optional<int64_t>
CVPQuerySchedule::getActiveIssueEnd(int64_t queryStart) const {
  auto count = getQueryCount(queryStart);
  if (!count)
    return std::nullopt;
  if (*count == 0)
    return 0;
  auto tiles = llvm::checkedMul(*count, getKeyTiles());
  return tiles ? llvm::checkedAdd(*tiles, int64_t{kStageCount - 1})
               : std::nullopt;
}

std::optional<CVPQueryCoordinate>
CVPQuerySchedule::getCoordinate(int64_t queryStart, int64_t queryOrdinal,
                                int64_t keyTile) const {
  auto count = getQueryCount(queryStart);
  if (!count || queryOrdinal < 0 || queryOrdinal >= *count || keyTile < 0 ||
      keyTile >= getKeyTiles())
    return std::nullopt;
  auto offset = llvm::checkedMul(queryOrdinal, queryStep);
  auto task = offset ? llvm::checkedAdd(queryStart, *offset) : std::nullopt;
  if (!task || !isI32(*task))
    return std::nullopt;
  return CVPQueryCoordinate{queryOrdinal,
                            *task,
                            keyTile,
                            keyTile * keyStep,
                            queryOrdinal % kQuerySlots,
                            keyTile == 0,
                            keyTile == getKeyTiles() - 1};
}

std::optional<CVPQueryCoordinate>
CVPQuerySchedule::getStageCoordinate(int64_t queryStart, int64_t issueTick,
                                     CVPQueryStage stage) const {
  unsigned lag = static_cast<unsigned>(stage);
  if (lag >= kStageCount || issueTick < int64_t{lag})
    return std::nullopt;
  int64_t tile = issueTick - lag;
  return getCoordinate(queryStart, tile / getKeyTiles(), tile % getKeyTiles());
}

std::optional<CVPQueryCoordinate>
CVPQuerySchedule::getKeyLane(int64_t queryStart, int64_t queryOrdinal,
                             int64_t baseKeyTile, unsigned lane,
                             unsigned keyUnrollFactor) const {
  if ((keyUnrollFactor != 1 && keyUnrollFactor != 2 && keyUnrollFactor != 4) ||
      lane >= keyUnrollFactor || baseKeyTile < 0 ||
      baseKeyTile >= getKeyTiles() || baseKeyTile % keyUnrollFactor)
    return std::nullopt;
  return getCoordinate(queryStart, queryOrdinal, baseKeyTile + lane);
}

std::optional<unsigned>
CVPQuerySchedule::getValidKeyLanes(int64_t baseKey,
                                   unsigned emittedLanes) const {
  if ((emittedLanes != 1 && emittedLanes != 2 && emittedLanes != 4) ||
      baseKey < 0 || baseKey >= keyCount || baseKey % keyStep ||
      (baseKey / keyStep) % emittedLanes)
    return std::nullopt;
  return static_cast<unsigned>(
      std::min<int64_t>((keyCount - baseKey) / keyStep, emittedLanes));
}

bool CVPQuerySchedule::canEmitQueryUnroll(unsigned factor) const {
  if (factor != 1 && factor != 2 && factor != 4)
    return false;
  auto maximumWork = getWorkEnd(std::numeric_limits<int32_t>::min());
  auto extension = llvm::checkedMul(int64_t{kStageCount + factor - 1}, keyStep);
  return maximumWork && extension &&
         llvm::checkedAdd(*maximumWork, *extension).has_value();
}

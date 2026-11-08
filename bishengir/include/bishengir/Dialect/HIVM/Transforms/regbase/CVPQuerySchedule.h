//===- CVPQuerySchedule.h - Logical query/key coordinates ---------*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_HIVM_REGBASE_CVPQUERYSCHEDULE_H
#define BISHENGIR_HIVM_REGBASE_CVPQUERYSCHEDULE_H

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::hivm {

enum class CVPQueryStage : unsigned { QK, Probability, PV, Output };

/// Coordinates exist only for valid work. A physical ring index is deliberately
/// separate from the monotonically increasing logical query generation.
struct CVPQueryCoordinate {
  int64_t queryOrdinal;
  int64_t queryTask;
  int64_t keyTile;
  int64_t localKey;
  int64_t querySlot;
  bool firstKey;
  bool lastKey;
};

/// Logical contract shared by query scheduling and its tests. Counts/offsets
/// retain the original i32 query assignment and BN-sized key loop; flattened
/// arithmetic uses i64. This contract does not certify asynchronous ownership
/// and accepting a lane factor does not imply an Unroll emitter exists.
class CVPQuerySchedule {
public:
  static constexpr unsigned kStageCount = 4;
  static constexpr unsigned kQuerySlots = 2;

  static std::optional<CVPQuerySchedule>
  create(int64_t keyCount, int64_t keyStep, int64_t queryUpper,
         int64_t queryStep, std::string &reason);

  int64_t getKeyCount() const { return keyCount; }
  int64_t getKeyStep() const { return keyStep; }
  int64_t getKeyTiles() const { return keyCount / keyStep; }
  int64_t getLastKey() const { return keyCount - keyStep; }
  int64_t getQueryUpper() const { return queryUpper; }
  int64_t getQueryStep() const { return queryStep; }

  /// Preserve task = originalStart + queryOrdinal * originalStep, including
  /// zero-task cores. A start outside the original signed i32 domain rejects.
  std::optional<int64_t> getQueryCount(int64_t queryStart) const;
  std::optional<int64_t> getWorkEnd(int64_t queryStart) const;

  /// Exclusive issue-tick bound containing all valid stages. Empty streams
  /// have no work. Existing CreatePreload may retain conservative empty ticks;
  /// using this bound to remove them requires the separate event/drain proof.
  std::optional<int64_t> getActiveIssueEnd(int64_t queryStart) const;

  /// issueTick and stage lag are measured in key tiles, not element offsets.
  /// Inactive prologue/epilogue work has no coordinate.
  std::optional<CVPQueryCoordinate>
  getStageCoordinate(int64_t queryStart, int64_t issueTick,
                     CVPQueryStage stage) const;

  /// A key-unrolled body is local to one query. Invalid lanes in its final
  /// partial body must not become the next query's first key. U1/U2/U4 are
  /// logical lane contracts, independent of CVPipelineMode and buffer depth.
  std::optional<CVPQueryCoordinate>
  getKeyLane(int64_t queryStart, int64_t queryOrdinal, int64_t baseKeyTile,
             unsigned lane, unsigned keyUnrollFactor) const;

  /// Match the real ordinary-Unroll emitter's capped body bound. baseKey is
  /// an element offset; this count alone does not create cross-query overlap.
  std::optional<unsigned> getValidKeyLanes(int64_t baseKey,
                                           unsigned emittedLanes) const;
  /// Include a conservative four-stage drain and the final partial group in
  /// the widened arithmetic bound. No physical/event proof is implied.
  bool canEmitQueryUnroll(unsigned factor) const;

private:
  CVPQuerySchedule(int64_t keyCount, int64_t keyStep, int64_t queryUpper,
                   int64_t queryStep)
      : keyCount(keyCount), keyStep(keyStep), queryUpper(queryUpper),
        queryStep(queryStep) {}
  std::optional<CVPQueryCoordinate> getCoordinate(int64_t queryStart,
                                                  int64_t queryOrdinal,
                                                  int64_t keyTile) const;

  int64_t keyCount, keyStep, queryUpper, queryStep;
};

} // namespace mlir::hivm
#endif

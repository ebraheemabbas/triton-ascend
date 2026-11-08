//===- CVPCubeResourcePlan.h - Shared CVPipeline Cube resources ---*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_DIALECT_UTILS_CVPCUBERESOURCEPLAN_H
#define BISHENGIR_DIALECT_UTILS_CVPCUBERESOURCEPLAN_H

// This header is shared with the device template. Keep it independent of
// LLVM/MLIR, allocations and host-only library facilities.
#include <cstdint>

namespace bishengir::cvp {

constexpr uint64_t kCubeResourceVersion = 1;
constexpr uint64_t kA5L0ABytes = 64 * 1024;
constexpr uint64_t kA5L0BBytes = 64 * 1024;
constexpr uint64_t kCubeAlignment = 32;
constexpr uint64_t kFractalElements = 16;
constexpr uint64_t kCubeBanks = 2;

struct ByteRange {
  uint64_t begin = 0;
  uint64_t end = 0;
  constexpr uint64_t size() const { return end - begin; }
  constexpr bool overlaps(ByteRange other) const {
    return begin < end && other.begin < other.end && begin < other.end &&
           other.begin < end;
  }
};

struct QCacheResources {
  bool valid = false;
  uint64_t aHalfBytes = 0, bHalfBytes = 0;
  uint64_t qkChunkK = 0, pvChunkK = 0, pvChunks = 0;
  uint64_t qBytes = 0, pvBBytes = 0;
  ByteRange a[2], pvB[2], q[2];
};

constexpr bool checkedMul(uint64_t a, uint64_t b, uint64_t &result) {
  if (b && a > (~uint64_t{0}) / b)
    return false;
  result = a * b;
  return true;
}

constexpr bool alignBytes(uint64_t value, uint64_t &result) {
  if (value > (~uint64_t{0}) - (kCubeAlignment - 1))
    return false;
  result = (value + kCubeAlignment - 1) / kCubeAlignment * kCubeAlignment;
  return true;
}

constexpr uint64_t minimum(uint64_t a, uint64_t b) { return a < b ? a : b; }

constexpr bool supportedQCacheGeometry(uint64_t bm, uint64_t bn, uint64_t hd,
                                       uint64_t elementBytes) {
  return bm == 128 && bn == 256 && hd == 64 && elementBytes == 2;
}

constexpr bool supportedQCacheTarget(uint64_t aBytes, uint64_t bBytes) {
  return aBytes == kA5L0ABytes && bBytes == kA5L0BBytes;
}

// Derive complete ranges from capacities and the existing double-buffered
// template's K tiling. The caller additionally checks the supported target and
// geometry. A capacity alone never proves asynchronous ownership.
constexpr QCacheResources deriveQCacheResources(uint64_t aBytes,
                                                uint64_t bBytes, uint64_t bm,
                                                uint64_t bn, uint64_t hd,
                                                uint64_t elementBytes) {
  QCacheResources result;
  if (!bm || !bn || !hd || !elementBytes || bm % kFractalElements ||
      bn % kFractalElements || hd % kFractalElements ||
      aBytes % (kCubeBanks * kCubeAlignment) ||
      bBytes % (kCubeBanks * kCubeAlignment))
    return result;
  uint64_t mBytes = 0, nBytes = 0, hdBytes = 0;
  if (!checkedMul(bm, elementBytes, mBytes) ||
      !checkedMul(bn, elementBytes, nBytes) ||
      !checkedMul(hd, elementBytes, hdBytes))
    return result;
  result.aHalfBytes = aBytes / kCubeBanks;
  result.bHalfBytes = bBytes / kCubeBanks;
  result.qkChunkK =
      minimum(result.aHalfBytes / nBytes, result.bHalfBytes / mBytes) /
      kFractalElements * kFractalElements;
  result.pvChunkK =
      minimum(result.aHalfBytes / mBytes, result.bHalfBytes / hdBytes) /
      kFractalElements * kFractalElements;
  // Cached Q occupies one complete B fragment. No generic whole-L0 fallback
  // or repeated cache load into the same region is permitted here.
  if (!result.qkChunkK || result.qkChunkK < hd || !result.pvChunkK ||
      bn % result.pvChunkK)
    return result;
  result.pvChunks = bn / result.pvChunkK;
  uint64_t qRaw = 0, pvRaw = 0;
  if (!checkedMul(hd, mBytes, qRaw) ||
      !checkedMul(result.pvChunkK, hdBytes, pvRaw) ||
      !alignBytes(qRaw, result.qBytes) || !alignBytes(pvRaw, result.pvBBytes) ||
      result.pvBBytes > result.bHalfBytes ||
      result.qBytes > result.bHalfBytes - result.pvBBytes)
    return result;
  for (uint64_t bank = 0; bank < kCubeBanks; ++bank) {
    const uint64_t aBase = bank * result.aHalfBytes;
    const uint64_t bBase = bank * result.bHalfBytes;
    result.a[bank] = {aBase, aBase + result.aHalfBytes};
    result.pvB[bank] = {bBase, bBase + result.pvBBytes};
    result.q[bank] = {bBase + result.pvBBytes,
                      bBase + result.pvBBytes + result.qBytes};
  }
  result.valid = true;
  return result;
}

constexpr QCacheResources kA5HD64QCacheResources =
    deriveQCacheResources(kA5L0ABytes, kA5L0BBytes, 128, 256, 64, 2);
static_assert(kA5HD64QCacheResources.valid);
static_assert(kA5HD64QCacheResources.qkChunkK == 64);
static_assert(kA5HD64QCacheResources.pvChunkK == 128);
static_assert(kA5HD64QCacheResources.pvChunks == 2);
static_assert(
    !kA5HD64QCacheResources.q[0].overlaps(kA5HD64QCacheResources.pvB[0]));
static_assert(
    !kA5HD64QCacheResources.q[1].overlaps(kA5HD64QCacheResources.pvB[1]));

} // namespace bishengir::cvp
#endif

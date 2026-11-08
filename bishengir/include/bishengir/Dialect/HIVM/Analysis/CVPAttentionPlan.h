//===- CVPAttentionPlan.h - Typed attention recognition ------------*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BISHENGIR_DIALECT_HIVM_ANALYSIS_CVPATTENTIONPLAN_H
#define BISHENGIR_DIALECT_HIVM_ANALYSIS_CVPATTENTIONPLAN_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace mlir::hivm {
inline constexpr llvm::StringLiteral kCVPAttentionPlanAttr =
    "hivm.cvp_attention.plan";
inline constexpr llvm::StringLiteral kCVPAttentionRoleAttr =
    "hivm.cvp_attention.role";
inline constexpr llvm::StringLiteral kCVPAttentionAppliedAttr =
    "hivm.cvp_attention.applied";

inline constexpr llvm::StringLiteral kCVPAttentionExpectedAttr =
    "hivm.cvp_attention.expected_count";
inline constexpr llvm::StringLiteral kCVPAttentionVerifiedAttr =
    "hivm.cvp_attention.verified";

enum class CVPAttentionRole {
  QK,
  QKTransfer,
  Scale,
  TileMax,
  Max,
  P32,
  P16,
  PV,
  PVTransfer,
  Alpha,
  TileSum,
  Sum,
  Output
};
StringRef stringifyCVPAttentionRole(CVPAttentionRole role);

enum class CVPAttentionScoreLayout { QueryMajor, KeyMajor };

/// A read-only description of the original tensor recurrence. This is not a
/// physical-lifetime or numerical-reassociation certificate. Later transforms
/// must revalidate operands and obtain the additional proofs they require.
struct CVPAttentionPlan {
  scf::ForOp loop;
  int64_t blockM = 0, blockN = 0, headDim = 0, keyCount = 0;
  unsigned outputState = 0, sumState = 0, maxState = 0;
  MmadL1Op qk, pv;
  Value rawScore, scaledScore, p32, p16, alpha;
  FloatAttr scale, initialMax, initialSum, initialOutput;
  CVPAttentionScoreLayout scoreLayout = CVPAttentionScoreLayout::QueryMajor;
  SmallVector<std::pair<Operation *, CVPAttentionRole>> roles;
  DictionaryAttr getSummary(MLIRContext *context) const;
};

FailureOr<CVPAttentionPlan> analyzeCVPAttention(
    scf::ForOp loop, std::string &reason,
    CVPAttentionScoreLayout layout = CVPAttentionScoreLayout::QueryMajor);
} // namespace mlir::hivm
#endif

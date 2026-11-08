//===- CVPQCacheContract.h - Verified Cube template resources -----*- C++
//-*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef BISHENGIR_DIALECT_HIVM_IR_CVPQCACHECONTRACT_H
#define BISHENGIR_DIALECT_HIVM_IR_CVPQCACHECONTRACT_H

#include "bishengir/Dialect/Utils/CVPCubeResourcePlan.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include <utility>

namespace mlir::hivm {
class MmadL1Op;
constexpr llvm::StringLiteral kCVPQCacheResources =
    "hivm.cvp_q_cache.resources";
constexpr llvm::StringLiteral kCVPCubeBankedReady =
    "hivm.cvp_cube.banked_ready";
constexpr llvm::StringLiteral kCVPCubeReadiness = "hivm.cvp_cube.readiness";
constexpr llvm::StringLiteral kCVPCubeOverlap = "hivm.cvp_cube.overlap";
constexpr llvm::StringLiteral kCVPCubeDerivedDrain =
    "hivm.cvp_cube.derived_drain";
constexpr llvm::StringLiteral kCVPCubeCreditDrainLoop =
    "hivm.cvp_cube.credit_drain_loop";

DictionaryAttr buildCVPQCacheResources(MLIRContext *context, StringRef mode,
                                       uint64_t aBytes, uint64_t bBytes);
FailureOr<std::pair<uint64_t, uint64_t>>
readCVPQCacheTargetCapacity(Operation *op);
FailureOr<bishengir::cvp::QCacheResources>
readCVPQCacheResources(func::FuncOp function);
LogicalResult verifyCVPQCacheMmad(MmadL1Op op);
} // namespace mlir::hivm
#endif

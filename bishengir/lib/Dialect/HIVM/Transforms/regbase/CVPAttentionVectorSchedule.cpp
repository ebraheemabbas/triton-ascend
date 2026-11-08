//===- CVPAttentionVectorSchedule.cpp - Independent attention rows
//---------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONVECTORSCHEDULE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
constexpr StringLiteral kRowsPerIteration =
    "hivm.cvp_attention.rows_per_iteration";
constexpr StringLiteral kRowUnrollCount = "hivm.cvp_attention.row_unrolls";

std::optional<int64_t> constantIndex(Value value) {
  APInt integer;
  if (!value.getType().isIndex() ||
      !matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

// A non-overlapping static root layout makes separate values of the row
// coordinate separate physical elements. Padding is allowed, including the
// blocked P destination. No address or symbol from an experiment is required.
bool hasInjectiveLayout(MemRefType type) {
  if (!type.hasStaticShape() || type.getRank() < 1 || type.getRank() > 4 ||
      type.getMemorySpace() !=
          AddressSpaceAttr::get(type.getContext(), AddressSpace::UB))
    return false;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(type, strides, offset)))
    return false;
  SmallVector<unsigned> axes;
  for (unsigned axis = 0; axis < type.getRank(); ++axis) {
    if (type.getDimSize(axis) < 1 || type.getDimSize(axis) > 4096 ||
        strides[axis] < 1 || strides[axis] > (int64_t{1} << 30))
      return false;
    axes.push_back(axis);
  }
  llvm::sort(axes,
             [&](unsigned a, unsigned b) { return strides[a] < strides[b]; });
  int64_t span = 1;
  for (unsigned axis : axes) {
    if (type.getDimSize(axis) == 1)
      continue;
    if (strides[axis] < span)
      return false;
    span += (type.getDimSize(axis) - 1) * strides[axis];
  }
  return true;
}

bool dependsOn(Value value, Value row) {
  SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (current == row)
      return true;
    Operation *definition = current.getDefiningOp();
    if (definition && visited.insert(definition).second)
      llvm::append_range(worklist, definition->getOperands());
  }
  return false;
}

bool isRowAccess(Value memory, ValueRange indices, scf::ForOp rows,
                 func::FuncOp function, std::optional<VectorType> storedType) {
  auto view = memory.getDefiningOp<memref::SubViewOp>();
  if (!view || !llvm::all_of(indices, [](Value index) {
        return constantIndex(index) == 0;
      }))
    return false;
  auto root = dyn_cast<BlockArgument>(view.getSource());
  if (!root || root.getOwner() != &function.getBody().front())
    return false;
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType || !hasInjectiveLayout(rootType))
    return false;
  auto offsets = view.getMixedOffsets();
  auto sizes = view.getMixedSizes();
  auto strides = view.getMixedStrides();
  if (offsets.size() != static_cast<size_t>(rootType.getRank()))
    return false;
  unsigned rowDimensions = 0;
  for (unsigned axis = 0; axis < offsets.size(); ++axis) {
    if (getConstantIntValue(strides[axis]) != 1 ||
        !getConstantIntValue(sizes[axis]))
      return false;
    auto value = dyn_cast<Value>(offsets[axis]);
    if (!value)
      continue;
    if (value == rows.getInductionVar()) {
      if (getConstantIntValue(sizes[axis]) != 1 ||
          rootType.getDimSize(axis) != *constantIndex(rows.getUpperBound()))
        return false;
      ++rowDimensions;
    } else if (dependsOn(value, rows.getInductionVar())) {
      return false;
    }
  }
  if (rowDimensions != 1)
    return false;
  if (storedType) {
    if (storedType->isScalable() || storedType->getRank() != sizes.size())
      return false;
    for (unsigned axis = 0; axis < sizes.size(); ++axis)
      if (getConstantIntValue(sizes[axis]) != storedType->getDimSize(axis))
        return false;
  }
  return true;
}

bool hasSupportedRowWork(func::FuncOp function) {
  bool reduction = false, multiply = false, add = false, normalization = false;
  function.walk([&](Operation *op) {
    reduction |= isa<vector::MultiDimReductionOp>(op);
    multiply |= isa<arith::MulFOp>(op);
    add |= isa<arith::AddFOp>(op);
    normalization |= isa<arith::DivFOp, math::LogOp>(op);
  });
  auto boundary =
      function->getAttrOfType<StringAttr>("hivm.cvp_attention.boundary");
  bool finalUpdate = boundary && (boundary.getValue() == "last_output" ||
                                  boundary.getValue() == "first_last_output");
  return (!normalization || finalUpdate) &&
         !function->hasAttr("hfusion.has_fill") &&
         (reduction || (multiply && add));
}

bool canUnrollRows(scf::ForOp rows, func::FuncOp function) {
  auto upper = constantIndex(rows.getUpperBound());
  if (rows->getParentOp() != function || rows.getNumResults() != 0 ||
      constantIndex(rows.getLowerBound()) != 0 ||
      constantIndex(rows.getStep()) != 1 || !upper || *upper < 2 ||
      *upper > 128 || *upper % 2 || rows->hasAttr("reductionLoop"))
    return false;
  bool valid = true;
  unsigned operations = 0, stores = 0;
  rows.walk([&](Operation *op) {
    if (!valid || op == rows.getOperation())
      return;
    if (++operations > 256) {
      valid = false;
      return;
    }
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      valid =
          !read.getMask() && isRowAccess(read.getSource(), read.getIndices(),
                                         rows, function, std::nullopt);
      return;
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      ++stores;
      valid = !write.getMask() && write.getPermutationMap().isIdentity() &&
              isRowAccess(write.getSource(), write.getIndices(), rows, function,
                          cast<VectorType>(write.getVector().getType()));
      return;
    }
    if (isa<scf::ForOp, scf::YieldOp>(op))
      return;
    if (isa<func::CallOp>(op) || op->getNumRegions() || !isMemoryEffectFree(op))
      valid = false;
  });
  return valid && stores;
}

bool isVerifiedCaller(func::FuncOp function) {
  auto plans = function->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
  if (function->hasAttr("hivm.backup_function") || !plans ||
      plans.size() != 1 || !function->hasAttr("hivm.part_of_mix") ||
      function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(function.getContext(), TFuncCoreType::AIV))
    return false;
  auto plan = dyn_cast<DictionaryAttr>(plans[0]);
  auto bm = plan ? plan.getAs<IntegerAttr>("block_m") : IntegerAttr();
  auto bn = plan ? plan.getAs<IntegerAttr>("block_n") : IntegerAttr();
  auto hd = plan ? plan.getAs<IntegerAttr>("head_dim") : IntegerAttr();
  return bm && bm.getInt() == 128 && bn &&
         (bn.getInt() == 128 || bn.getInt() == 256) && hd &&
         (hd.getInt() == 64 || hd.getInt() == 128);
}

void unrollTwoRows(scf::ForOp loop) {
  SmallVector<Operation *> original;
  for (Operation &op : loop.getBody()->without_terminator())
    original.push_back(&op);
  OpBuilder builder(loop);
  Value two = builder.create<arith::ConstantIndexOp>(loop.getLoc(), 2);
  Value one = loop.getStep();
  loop.getStepMutable().assign(two);
  builder.setInsertionPoint(loop.getBody()->getTerminator());
  Value next =
      builder.create<arith::AddIOp>(loop.getLoc(), loop.getInductionVar(), one);
  IRMapping mapping;
  mapping.map(loop.getInductionVar(), next);
  // Keep complete original row bodies in their original order. In particular,
  // nested reduction iter_args and their merge order are cloned, not regrouped.
  for (Operation *op : original)
    builder.clone(*op, mapping);
  loop->setAttr(kRowsPerIteration, builder.getI64IntegerAttr(2));
}

struct CVPAttentionVectorSchedulePass
    : impl::CVPAttentionVectorScheduleBase<CVPAttentionVectorSchedulePass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<scf::ForOp> candidates;
    llvm::SmallPtrSet<Operation *, 8> visited;
    unsigned alreadyApplied = 0;
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (!isVerifiedCaller(caller))
        continue;
      caller.walk([&](func::CallOp call) {
        auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
        if (!callee || callee.isExternal() ||
            callee->hasAttr("hivm.backup_function") ||
            !callee->hasAttr("hivm.vector_function") ||
            !hasSupportedRowWork(callee) || !visited.insert(callee).second)
          return;
        auto uses = SymbolTable::getSymbolUses(callee, module);
        if (!uses ||
            !llvm::all_of(*uses, [&](const SymbolTable::SymbolUse &use) {
              return use.getUser() == call.getOperation();
            }))
          return;
        for (scf::ForOp loop : callee.getBody().front().getOps<scf::ForOp>()) {
          if (auto factor =
                  loop->getAttrOfType<IntegerAttr>(kRowsPerIteration)) {
            if (factor.getInt() == 2 && constantIndex(loop.getStep()) == 2)
              ++alreadyApplied;
            continue;
          }
          if (canUnrollRows(loop, callee))
            candidates.push_back(loop);
        }
      });
    }
    if (candidates.empty() && !alreadyApplied && requireMatch) {
      module.emitError("CVP rows2 requires a verified route and an independent "
                       "even row loop without external effects");
      signalPassFailure();
      return;
    }
    for (scf::ForOp loop : candidates)
      unrollTwoRows(loop);
    if (!candidates.empty())
      module->setAttr(kRowUnrollCount,
                      IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                       candidates.size() + alreadyApplied));
    if (!retainProvenance)
      for (func::FuncOp function : module.getOps<func::FuncOp>())
        if (!function->hasAttr("hivm.backup_function"))
          function->removeAttr(kCVPAttentionVerifiedAttr);
  }
};
} // namespace

std::unique_ptr<Pass> createCVPAttentionVectorSchedulePass(
    const CVPAttentionVectorScheduleOptions &options) {
  return std::make_unique<CVPAttentionVectorSchedulePass>(options);
}
} // namespace hivm
} // namespace mlir

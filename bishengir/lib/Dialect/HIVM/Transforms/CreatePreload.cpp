//===- CreatePreload.cpp - Create preload for CV pipelining --------------===//
//
// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/regbase/CVPQuerySchedule.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Scope/Transforms/Passes.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "bishengir/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DECL_CREATEPRELOAD
#define GEN_PASS_DEF_CREATEPRELOAD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"

} // namespace mlir

#define DEBUG_TYPE "hivm-create-preload"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define DBGSNL() (llvm::dbgs() << "\n")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::hivm;

struct PreloadInfo {
  // Preload info
  size_t preloadNum;
  size_t maxPreloadNum;
  Value maxPreloadValue;
  // Loop info
  Value indVar;
  Value lb;
  Value ub;
  Value step;
  // Kernel info
  TFuncCoreType funcCoreType;
  // Planned pointer roots for preload-local buffers and their physical buffer
  // counts. The annotation may be attached to a view of the planned root.
  DenseMap<Value, size_t> preloadLocalBuffers;
  // IRMapping for rewriting
  SmallVector<IRMapping> mappings;
};

struct CreatePreloadPass : public impl::CreatePreloadBase<CreatePreloadPass> {
  void runOnOperation() override;
};

static std::optional<hivm::PointerCastOp> getLocalBuffer(Value v) {
  if (auto pointerCastOp = v.getDefiningOp<hivm::PointerCastOp>()) {
    if (utils::getAnnotateOpWithAttr(v, hivm::PreloadLocalBufferAttr::name)
            .has_value())
      return pointerCastOp;
    return std::nullopt;
  }
  if (auto scopeOp = v.getDefiningOp<scope::ScopeOp>()) {
    auto &scopeBody = scopeOp.getRegion().front();
    auto returnOp = cast<scope::ReturnOp>(scopeBody.getTerminator());
    auto resIdx = cast<OpResult>(v).getResultNumber();
    return getLocalBuffer(returnOp.getResults()[resIdx]);
  }
  if (auto blockArgument = dyn_cast<BlockArgument>(v)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArgument.getOwner()->getParentOp());
    if (!forOp)
      return std::nullopt;
    return getLocalBuffer(forOp.getTiedLoopInit(blockArgument)->get());
  }
  return std::nullopt;
}

static void collectPreloadLocalBuffers(func::FuncOp funcOp, PreloadInfo &info) {
  auto isPointerCast = [](Value value) {
    return static_cast<bool>(value.getDefiningOp<hivm::PointerCastOp>());
  };

  funcOp.walk([&](annotation::MarkOp markOp) {
    if (!utils::isAnnotationWithAttr(markOp,
                                     hivm::PreloadLocalBufferAttr::name))
      return;

    auto roots =
        utils::tracebackMemRefVecByTargetFn(markOp.getSrc(), isPointerCast);
    if (roots.size() != 1)
      return;

    auto pointerCastOp = roots.front().getDefiningOp<hivm::PointerCastOp>();
    if (!pointerCastOp || pointerCastOp.getAddrs().empty())
      return;

    // PlanMemory materializes one address per physical buffer, so the address
    // list is authoritative even when max_preload_num is larger.
    info.preloadLocalBuffers[pointerCastOp.getResult()] =
        pointerCastOp.getAddrs().size();
  });
}

static std::optional<size_t> getPreloadLocalBufferNum(hivm::PointerCastOp op,
                                                      const PreloadInfo &info) {
  auto it = info.preloadLocalBuffers.find(op.getResult());
  if (it != info.preloadLocalBuffers.end())
    return it->second;

  // Preserve support for the legacy form that stores the marker directly on
  // the pointer_cast operation instead of using annotation.mark.
  if (op->hasAttr(hivm::PreloadLocalBufferAttr::name) && !op.getAddrs().empty())
    return op.getAddrs().size();

  return std::nullopt;
}

static Value cloneLocalBuffer(Value oldValue, hivm::PointerCastOp op,
                              size_t preloadNum, PreloadInfo &info,
                              OpBuilder &b) {
  auto maybeBufferNum = getPreloadLocalBufferNum(op, info);
  assert(maybeBufferNum && *maybeBufferNum > 0 &&
         "expected a preload-local buffer with planned addresses");
  size_t bufferNum = *maybeBufferNum;
  SmallVector<Value> addrs(bufferNum);

  size_t preloadIdx = preloadNum % bufferNum;
  for (auto [i, addr] : llvm::enumerate(op.getAddrs())) {
    auto shiftedIdx = (bufferNum - preloadIdx - 1 + i) % bufferNum;
    addrs[shiftedIdx] = addr;
  }
  auto newPointerCastOp =
      b.create<hivm::PointerCastOp>(op.getLoc(), op.getType(), addrs);
  info.mappings[preloadNum].map(oldValue, newPointerCastOp);
  return newPointerCastOp;
}

static bool hasPreloadWorkspaceMark(Value value) {
  if (!value)
    return false;

  if (utils::getAnnotateOpWithAttr(value, hivm::PreloadWorkspaceAttr::name)
          .has_value()) {
    return true;
  }

  if (Operation *defOp = value.getDefiningOp()) {
    return defOp->hasAttr(hivm::PreloadWorkspaceAttr::name);
  }

  return false;
}

static bool isPreloadWorkspaceSubview(memref::SubViewOp subviewOp) {
  return hasPreloadWorkspaceMark(subviewOp.getSource());
}

static Value cloneWorkspaceSubview(memref::SubViewOp subviewOp,
                                   size_t preloadNum, PreloadInfo &info,
                                   OpBuilder &b) {
  auto offsets = subviewOp.getMixedOffsets();
  auto loc = subviewOp.getLoc();

  auto indVar = info.mappings[preloadNum].lookup(info.indVar);

  Value offset = b.create<arith::DivSIOp>(indVar.getLoc(), indVar, info.step);
  offset =
      b.create<arith::RemSIOp>(offset.getLoc(), offset, info.maxPreloadValue);
  offset =
      b.create<arith::IndexCastOp>(offset.getLoc(), b.getIndexType(), offset);
  offsets[0] = offset;

  auto src = info.mappings[preloadNum].lookupOrDefault(subviewOp.getSource());
  Value newResult;

  if (subviewOp.getSourceType().getRank() == subviewOp.getType().getRank()) {
    newResult = b.create<memref::SubViewOp>(loc, src, offsets,
                                            subviewOp.getMixedSizes(),
                                            subviewOp.getMixedStrides());
  } else {
    auto newResType = memref::SubViewOp::inferRankReducedResultType(
        subviewOp.getType().getShape(), subviewOp.getSourceType(), offsets,
        subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
    newResult = b.create<memref::SubViewOp>(
        loc, cast<MemRefType>(newResType), src, offsets,
        subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
  }

  auto adaptorOp =
      b.create<UnrealizedConversionCastOp>(loc, subviewOp.getType(), newResult);

  adaptorOp->setAttr("preload_adaptor", b.getUnitAttr());
  LDBG("Mapping " << subviewOp << '\n' << adaptorOp);
  info.mappings[preloadNum].map(subviewOp.getResult(), adaptorOp->getResult(0));

  return adaptorOp->getResult(0);
}

static Value
preprocessLoopArgs(scf::ForOp forOp, SmallVector<Value> &newInitArgs,
                   DenseMap<Value, hivm::PointerCastOp> &valueToAdapt,
                   const PreloadInfo &info, OpBuilder &b) {
  for (auto [initArg, iterArg] :
       llvm::zip_equal(forOp.getInitArgs(), forOp.getRegionIterArgs())) {
    if (auto pointerCastOp = initArg.getDefiningOp<hivm::PointerCastOp>();
        pointerCastOp && getPreloadLocalBufferNum(pointerCastOp, info)) {
      valueToAdapt[iterArg] = pointerCastOp;
    } else {
      newInitArgs.push_back(initArg);
    }
  }
  Value newUpperbound = b.create<arith::MulIOp>(
      info.ub.getLoc(), info.maxPreloadValue, info.step);
  newUpperbound =
      b.create<arith::AddIOp>(info.ub.getLoc(), info.ub, newUpperbound);
  return newUpperbound;
}

static Value getPreloadCondition(const PreloadInfo &info, Location loc,
                                 OpBuilder &b) {
  auto &mapping = info.mappings[info.preloadNum];

  Value indVar = mapping.lookup(info.indVar);
  Value lb = mapping.lookupOrDefault(info.lb);
  Value ub = mapping.lookupOrDefault(info.ub);

  Value lowerCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, indVar, lb);
  Value upperCond =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, indVar, ub);

  return b.create<arith::AndIOp>(loc, lowerCond, upperCond);
}

static bool isRematerializableAliasOp(Operation *op) {
  // PointerCastOp is not ViewLikeOpInterface because it creates a memref from
  // raw addresses, but it can still be cloned as the root of a view chain.
  return isa_and_nonnull<ViewLikeOpInterface, hivm::PointerCastOp>(op);
}

static void rewriteScopeReturnOp(ValueRange returnResults,
                                 const PreloadInfo &info, Location loc,
                                 OpBuilder &b) {
  SmallVector<Value> newReturns;
  for (auto res : returnResults) {
    if (!getLocalBuffer(res)) {
      res = info.mappings[info.preloadNum].lookupOrDefault(res);
      newReturns.push_back(res);
    }
  }
  b.create<scope::ReturnOp>(loc, newReturns);
}

static void rewriteBody(Block *body, PreloadInfo &info, OpBuilder &b) {
  for (auto &op : body->without_terminator()) {
    if (auto pointerCastOp = dyn_cast<hivm::PointerCastOp>(&op);
        pointerCastOp && getPreloadLocalBufferNum(pointerCastOp, info)) {
      cloneLocalBuffer(pointerCastOp, pointerCastOp, info.preloadNum, info, b);
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(&op);
               subviewOp && isPreloadWorkspaceSubview(subviewOp)) {
      cloneWorkspaceSubview(subviewOp, info.preloadNum, info, b);
    } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      auto &mapping = info.mappings[info.preloadNum];
      auto lb = mapping.lookupOrDefault(forOp.getLowerBound());
      auto ub = mapping.lookupOrDefault(forOp.getUpperBound());
      auto step = mapping.lookupOrDefault(forOp.getStep());
      SmallVector<Value> inits;
      for (auto init : forOp.getInits()) {
        inits.push_back(mapping.lookupOrDefault(init));
      }
      auto newForOp = b.create<scf::ForOp>(
          op.getLoc(), lb, ub, step, inits,
          [&](OpBuilder &b, Location loc, Value indVar, ValueRange args) {
            mapping.map(forOp.getInductionVar(), indVar);
            mapping.map(forOp.getRegionIterArgs(), args);
            rewriteBody(forOp.getBody(), info, b);
            b.clone(*forOp.getBody()->getTerminator(),
                    info.mappings[info.preloadNum]);
          });
      info.mappings[info.preloadNum].map(forOp->getResults(),
                                         newForOp->getResults());
      // This is source provenance only. Logical iteration values remain in
      // the per-access binding marks cloned through the stage's IRMapping.
    } else {
      b.clone(op, info.mappings[info.preloadNum]);
    }
  }
}

static Value lookThroughScopeResult(Value value) {
  while (auto scopeOp = value.getDefiningOp<scope::ScopeOp>()) {
    auto returnOp =
        cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator());
    value = returnOp.getResults()[cast<OpResult>(value).getResultNumber()];
  }
  return value;
}

// Bufferization can return a selected state buffer through copies instead of
// directly yielding the scope result. An inactive pipeline stage must preserve
// that state's previous iteration value, not select a newly allocated buffer
// whose producer did not execute. This is logical value forwarding only; it
// neither changes the planned addresses nor establishes physical lifetimes.
static bool hasSameStaticElements(Value lhs, Value rhs) {
  auto a = dyn_cast<MemRefType>(lhs.getType());
  auto b = dyn_cast<MemRefType>(rhs.getType());
  return a && b && a.hasStaticShape() && b.hasStaticShape() &&
         a.getElementType() == b.getElementType() &&
         a.getNumElements() == b.getNumElements();
}

static Value getWholeBufferViewSource(Operation *op) {
  if (!isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(op))
    return {};
  Value source = op->getOperand(0);
  return hasSameStaticElements(source, op->getResult(0)) ? source : Value{};
}

static std::pair<Value, Value> getWholeBufferCopy(Operation *op) {
  Value source, target;
  if (auto copy = dyn_cast<hivm::CopyOp>(op)) {
    source = copy.getSrc();
    target = copy.getDst();
  } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    source = copy.getSource();
    target = copy.getTarget();
  }
  if (!source || !hasSameStaticElements(source, target))
    return {};
  return {source, target};
}

static SmallVector<Value> getWholeBufferAliases(Value value) {
  SmallVector<Value> aliases{value};
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value current = aliases[i];
    auto add = [&](Value candidate) {
      if (candidate && !llvm::is_contained(aliases, candidate))
        aliases.push_back(candidate);
    };
    if (Operation *def = current.getDefiningOp())
      add(getWholeBufferViewSource(def));
    for (Operation *user : current.getUsers())
      if (getWholeBufferViewSource(user) == current)
        add(user->getResult(0));
  }
  return aliases;
}

static Operation *getLoopBodyAncestor(Operation *op, scf::ForOp loop) {
  while (op && op->getBlock() != loop.getBody())
    op = op->getParentOp();
  return op;
}

static SmallVector<Value> getAliasesForWriteCheck(Value value,
                                                  scf::ForOp loop) {
  SmallVector<Value> aliases{value};
  for (unsigned i = 0; i < aliases.size(); ++i) {
    Value current = aliases[i];
    auto add = [&](Value candidate) {
      if (candidate && isa<BaseMemRefType>(candidate.getType()) &&
          !llvm::is_contained(aliases, candidate))
        aliases.push_back(candidate);
    };
    if (auto view = current.getDefiningOp<ViewLikeOpInterface>())
      add(view.getViewSource());
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == current)
          for (Value result : user->getResults())
            add(result);
      } else if (auto select = dyn_cast<arith::SelectOp>(user)) {
        add(select.getResult());
      } else if (isa<scope::ReturnOp>(user)) {
        add(user->getParentOp()->getResult(use.getOperandNumber()));
      } else if (isa<scf::YieldOp>(user) &&
                 user->getParentOp() != loop.getOperation()) {
        if (isa<scf::IfOp>(user->getParentOp()))
          add(user->getParentOp()->getResult(use.getOperandNumber()));
      }
    }
  }
  return aliases;
}

static bool isUnmodifiedBetween(Value value, Operation *from, Operation *until,
                                scf::ForOp loop) {
  for (Value alias : getAliasesForWriteCheck(value, loop)) {
    for (Operation *user : alias.getUsers()) {
      Operation *position = getLoopBodyAncestor(user, loop);
      if (!position || position == from || position == until ||
          !from->isBeforeInBlock(position) || !position->isBeforeInBlock(until))
        continue;
      if (isa<ViewLikeOpInterface, arith::SelectOp, annotation::MarkOp,
              scf::YieldOp, scope::ReturnOp>(user))
        continue;
      if (auto [source, target] = getWholeBufferCopy(user); source) {
        if (target == alias)
          return false;
        continue;
      }
      auto effects = dyn_cast<MemoryEffectOpInterface>(user);
      if (!effects)
        return false;
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const auto &effect : instances)
        if ((!effect.getValue() || effect.getValue() == alias) &&
            !isa<MemoryEffects::Read>(effect.getEffect()))
          return false;
    }
  }
  return true;
}

static bool reachesOnlyStateYield(Value result, scope::ScopeOp scope,
                                  scf::ForOp loop, unsigned stateIndex) {
  SmallVector<std::pair<Value, Operation *>> work{{result, scope}};
  bool found = false;
  for (unsigned i = 0; i < work.size(); ++i) {
    auto [value, origin] = work[i];
    for (Value alias : getWholeBufferAliases(value)) {
      for (OpOperand &use : alias.getUses()) {
        Operation *user = use.getOwner();
        if (user == loop.getBody()->getTerminator()) {
          if (use.getOperandNumber() != stateIndex ||
              !isUnmodifiedBetween(value, origin, user, loop))
            return false;
          found = true;
          continue;
        }
        if (user->getBlock() != loop.getBody() ||
            !origin->isBeforeInBlock(user))
          continue;
        auto [source, target] = getWholeBufferCopy(user);
        if (source != alias || !isUnmodifiedBetween(value, origin, user, loop))
          continue;
        auto next = std::make_pair(target, user);
        if (!llvm::is_contained(work, next))
          work.push_back(next);
      }
    }
  }
  return found;
}

static Value getSkippedLoopState(Value result, scope::ScopeOp scope) {
  auto loop = dyn_cast<scf::ForOp>(scope->getParentOp());
  if (!loop)
    return {};
  Value direct;
  for (OpOperand &use : result.getUses()) {
    if (use.getOwner() != loop.getBody()->getTerminator())
      continue;
    Value previous = loop.getRegionIterArgs()[use.getOperandNumber()];
    if (direct && direct != previous)
      return {};
    direct = previous;
  }
  if (direct)
    return direct;

  auto selected =
      lookThroughScopeResult(result).getDefiningOp<arith::SelectOp>();
  if (!selected)
    return {};
  auto trueArg = dyn_cast<BlockArgument>(selected.getTrueValue());
  auto falseArg = dyn_cast<BlockArgument>(selected.getFalseValue());
  bool trueState = trueArg && trueArg.getOwner() == loop.getBody() &&
                   trueArg != loop.getInductionVar();
  bool falseState = falseArg && falseArg.getOwner() == loop.getBody() &&
                    falseArg != loop.getInductionVar();
  if (trueState == falseState)
    return {};
  BlockArgument previous = trueState ? trueArg : falseArg;
  unsigned index = previous.getArgNumber() - 1;
  if (previous.getType() != result.getType() ||
      !reachesOnlyStateYield(result, scope, loop, index))
    return {};
  return previous;
}

static bool canRematerializeViewResult(Value value, scope::ScopeOp scope) {
  value = lookThroughScopeResult(value);
  Region *region = value.getParentRegion();
  if (!region || !scope.getRegion().isAncestor(region))
    return true;
  Operation *def = value.getDefiningOp();
  return isRematerializableAliasOp(def) &&
         llvm::all_of(def->getOperands(), [&](Value operand) {
           return canRematerializeViewResult(operand, scope);
         });
}

static void emitRematerializationError(Operation *definingOp,
                                       scope::ScopeOp scopeOp) {
  if (!definingOp) {
    scopeOp.emitOpError(
        "cannot rematerialize a preload scope result through an internal "
        "block argument");
  } else {
    definingOp->emitOpError(
        "is not a supported view while rematerializing a preload scope "
        "result");
  }
}

static FailureOr<Value>
rematerializeViewLikeResult(Value value, scope::ScopeOp scopeOp,
                            const IRMapping &preloadMapping,
                            IRMapping &viewMapping, OpBuilder &builder) {
  if (Value mapped = viewMapping.lookupOrNull(value))
    return mapped;

  Value source = lookThroughScopeResult(value);
  if (source != value) {
    auto rematerialized = rematerializeViewLikeResult(
        source, scopeOp, preloadMapping, viewMapping, builder);
    if (failed(rematerialized))
      return failure();
    viewMapping.map(value, *rematerialized);
    return *rematerialized;
  }

  Region *definingRegion = value.getParentRegion();
  if (!definingRegion || !scopeOp.getRegion().isAncestor(definingRegion)) {
    Value mapped = preloadMapping.lookupOrDefault(value);
    viewMapping.map(value, mapped);
    return mapped;
  }

  Operation *definingOp = value.getDefiningOp();
  if (!isRematerializableAliasOp(definingOp)) {
    emitRematerializationError(definingOp, scopeOp);
    return failure();
  }

  for (Value operand : definingOp->getOperands()) {
    if (failed(rematerializeViewLikeResult(operand, scopeOp, preloadMapping,
                                           viewMapping, builder)))
      return failure();
  }

  builder.clone(*definingOp, viewMapping);
  return viewMapping.lookup(value);
}

static scf::IfOp rewriteScopeOp(Value cond, scope::ScopeOp scopeOp,
                                PreloadInfo &info, Location loc,
                                OpBuilder &builder) {
  auto &scopeBody = scopeOp.getRegion().front();
  auto returnResults =
      cast<scope::ReturnOp>(scopeBody.getTerminator()).getResults();
  return builder.create<scf::IfOp>(
      loc, cond,
      [&](OpBuilder &b, Location loc) {
        SmallVector<Type> newScopeResultTypes;
        for (auto res : returnResults) {
          if (!getLocalBuffer(res))
            newScopeResultTypes.push_back(res.getType());
        }
        auto newOp =
            b.create<scope::ScopeOp>(scopeOp.getLoc(), newScopeResultTypes);
        newOp->setAttrs(scopeOp->getAttrs());
        newOp.setNoInline(false);

        Region &region = newOp.getRegion();
        Block *bodyBlock = b.createBlock(&region);

        b.setInsertionPointToEnd(bodyBlock);
        rewriteBody(&scopeBody, info, b);
        rewriteScopeReturnOp(returnResults, info, loc, b);
        b.setInsertionPointAfter(newOp);
        b.create<scf::YieldOp>(loc, newOp->getResults());
      },
      [&](OpBuilder &b, Location loc) {
        SmallVector<Value> newYields;
        IRMapping viewMapping;
        for (auto [res, retRes] :
             llvm::zip_equal(scopeOp->getResults(), returnResults)) {
          if (!getLocalBuffer(retRes)) {
            if (Value previous = getSkippedLoopState(res, scopeOp)) {
              newYields.push_back(
                  info.mappings[info.preloadNum].lookupOrDefault(previous));
            } else if (auto pointerCastOp =
                           retRes.getDefiningOp<hivm::PointerCastOp>()) {
              auto newRes = b.clone(*pointerCastOp)->getResult(0);
              newYields.push_back(newRes);
            } else {
              // Rematerialize the view chain independently in the skip branch.
              auto rematerialized = rematerializeViewLikeResult(
                  retRes, scopeOp, info.mappings[info.preloadNum], viewMapping,
                  b);
              if (failed(rematerialized))
                llvm::report_fatal_error("Unhandled scope result case");
              newYields.push_back(*rematerialized);
            }
          }
        }
        b.create<scf::YieldOp>(loc, newYields);
      });
}

static bool isSynchronizationOp(Operation *op) {
  return isa<hivm::SetFlagOp, hivm::WaitFlagOp, hivm::PipeBarrierOp,
             hivm::SyncBlockOp, hivm::SyncBlockSetOp, hivm::SyncBlockWaitOp>(
      op);
}

// One logical issue tick. Both emitters use the same checked inactive-state,
// view rematerialization and physical-ring selection rules. A new mapping per
// tick prevents the next Unroll lane from reading the preceding lane's SSA.
static SmallVector<Value> materializeScheduledTick(
    scf::ForOp forOp, DenseMap<Operation *, size_t> &scopeToIdx,
    DenseMap<Value, hivm::PointerCastOp> &valueToAdapt, PreloadInfo &info,
    Value indVar, ValueRange args, OpBuilder &b, Location loc,
    bool filterAdaptedState = false) {
  const size_t maxPreloadNum = info.maxPreloadNum;
  info.mappings.clear();
  info.mappings.resize(maxPreloadNum);
  Value curIndVar = indVar;
  info.indVar = forOp.getInductionVar();
  for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0; preloadNum--) {
    info.mappings[preloadNum].map(info.indVar, curIndVar);
    auto newArgIter = args.begin();
    for (auto oldArg : forOp.getRegionIterArgs()) {
      if (auto it = valueToAdapt.find(oldArg); it != valueToAdapt.end()) {
        auto pointerCastOp = it->second;
        cloneLocalBuffer(oldArg, pointerCastOp, preloadNum, info, b);
      } else {
        info.mappings[preloadNum].map(oldArg, *newArgIter);
        ++newArgIter;
      }
    }
    curIndVar =
        b.createOrFold<arith::SubIOp>(curIndVar.getLoc(), curIndVar, info.step);
  }

  for (auto &bodyOp : forOp.getBody()->without_terminator()) {
    if (!scopeToIdx.contains(&bodyOp)) {
      if (isSynchronizationOp(&bodyOp)) {
        b.clone(bodyOp, info.mappings[maxPreloadNum - 1]);
        continue;
      }
      if (auto pointerCastOp = dyn_cast<hivm::PointerCastOp>(&bodyOp)) {
        if (getPreloadLocalBufferNum(pointerCastOp, info)) {
          for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0;
               preloadNum--) {
            cloneLocalBuffer(pointerCastOp, pointerCastOp, preloadNum, info, b);
          }
          continue;
        }
      } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(&bodyOp);
                 subviewOp && isPreloadWorkspaceSubview(subviewOp)) {
        for (int64_t preloadNum = maxPreloadNum - 1; preloadNum >= 0;
             preloadNum--) {
          cloneWorkspaceSubview(subviewOp, preloadNum, info, b);
        }
        continue;
      }
      for (auto &mapping : info.mappings)
        b.clone(bodyOp, mapping);
      continue;
    }
    auto scopeOp = cast<scope::ScopeOp>(&bodyOp);
    info.preloadNum = scopeToIdx.at(&bodyOp);

    LDBG("Handling preload scope: " << info.preloadNum << "\n" << bodyOp);

    auto loc = scopeOp.getLoc();
    Value cond = getPreloadCondition(info, loc, b);

    auto newOp = rewriteScopeOp(cond, scopeOp, info, loc, b);
    auto returnResults =
        cast<scope::ReturnOp>(scopeOp.getRegion().front().getTerminator())
            .getResults();
    auto scopeResIter = scopeOp.result_begin();
    SmallVector<IRMapping> viewMappings(info.mappings.size());
    for (auto newRes : newOp->getResults()) {
      auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
      while (maybeLocalBuffer.has_value()) {
        for (auto &mapping : info.mappings)
          mapping.map(*scopeResIter, mapping.lookup(maybeLocalBuffer.value()));
        ++scopeResIter;
        maybeLocalBuffer = getLocalBuffer(*scopeResIter);
      }

      Value scopeRes = *scopeResIter;
      auto resultNumber = cast<OpResult>(scopeRes).getResultNumber();
      Value returnResult = returnResults[resultNumber];
      Value source = lookThroughScopeResult(returnResult);
      if (!isa_and_nonnull<ViewLikeOpInterface>(source.getDefiningOp())) {
        for (auto &mapping : info.mappings)
          mapping.map(scopeRes, newRes);
      } else {
        for (auto [mapping, viewMapping] :
             llvm::zip_equal(info.mappings, viewMappings)) {
          auto rematerialized = rematerializeViewLikeResult(
              returnResult, scopeOp, mapping, viewMapping, b);
          assert(succeeded(rematerialized) &&
                 "Failed to rematerialize checked scope result");
          mapping.map(scopeRes, *rematerialized);
        }
      }
      ++scopeResIter;
    }
    for (; scopeResIter != scopeOp.result_end(); ++scopeResIter) {
      auto maybeLocalBuffer = getLocalBuffer(*scopeResIter);
      for (auto &mapping : info.mappings)
        mapping.map(*scopeResIter, mapping.lookup(maybeLocalBuffer.value()));
    }
  }

  SmallVector<Value> newYields;
  if (!args.empty()) {
    newYields = llvm::map_to_vector(forOp.getYieldedValues(), [&](Value yield) {
      return info.mappings[maxPreloadNum - 1].lookupOrDefault(yield);
    });
  }
  if (filterAdaptedState && !valueToAdapt.empty()) {
    SmallVector<Value> filtered;
    for (auto [index, argument] : llvm::enumerate(forOp.getRegionIterArgs()))
      if (!valueToAdapt.contains(argument))
        filtered.push_back(newYields[index]);
    newYields = std::move(filtered);
  }
  return newYields;
}

static void rewritePreloadLoop(scf::ForOp forOp,
                               SmallVector<scope::ScopeOp, 4> scopes,
                               size_t maxPreloadNum) {
  IRRewriter rewriter(forOp.getContext());
  PreloadInfo info;
  info.funcCoreType = queryFuncCoreType(forOp->getParentOfType<func::FuncOp>())
                          .value_or(TFuncCoreType::AIC_OR_AIV);
  info.lb = forOp.getLowerBound();
  info.ub = forOp.getUpperBound();
  info.step = forOp.getStep();
  info.maxPreloadNum = maxPreloadNum;
  rewriter.setInsertionPoint(forOp);
  info.maxPreloadValue = rewriter.create<arith::ConstantOp>(
      info.ub.getLoc(),
      rewriter.getIntegerAttr(info.ub.getType(), info.maxPreloadNum));
  info.mappings.resize(maxPreloadNum);
  collectPreloadLocalBuffers(forOp->getParentOfType<func::FuncOp>(), info);

  SmallVector<Value> newInitArgs;
  DenseMap<Value, hivm::PointerCastOp> valueToAdapt;
  Value newUpperbound =
      preprocessLoopArgs(forOp, newInitArgs, valueToAdapt, info, rewriter);

  DenseMap<Operation *, size_t> scopeToIdx;
  for (size_t preloadNum = 0; preloadNum < scopes.size(); preloadNum++) {
    auto scopeOp = scopes[preloadNum];
    if (!scopeOp)
      continue;
    scopeToIdx[scopeOp] = preloadNum;
  }

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), info.lb, newUpperbound, info.step, newInitArgs,
      [&](OpBuilder &b, Location loc, Value indVar, ValueRange args) {
        auto newYields = materializeScheduledTick(
            forOp, scopeToIdx, valueToAdapt, info, indVar, args, b, loc);
        b.create<scf::YieldOp>(loc, newYields);
      });

  // Do not stamp the original local clock onto the shifted IV. The paired
  // origin can repeat across clones; binding SSA establishes each generation.
  LDBG("New for loop:\n" << newForOp);

  SmallVector<Value> newRes;
  for (auto [res, yield] :
       llvm::zip_equal(newForOp->getResults(), newForOp.getYieldedValues())) {
    if (auto maybeLocalBuffer = getLocalBuffer(yield);
        maybeLocalBuffer.has_value()) {
      newRes.push_back(maybeLocalBuffer.value());
    } else {
      newRes.push_back(res);
    }
  }
  rewriter.replaceOp(forOp, newRes);
}

static LogicalResult
validateContinuousQueryUnroll(scf::ForOp loop, ArrayRef<scope::ScopeOp> scopes,
                              unsigned factor) {
  auto reject = [&](StringRef message) {
    loop.emitError("continuous Unroll: ") << message;
    return failure();
  };
  auto stream = loop->getAttrOfType<DictionaryAttr>("hivm.cvp_query.stream");
  auto mode = stream ? stream.getAs<StringAttr>("pipeline_mode") : StringAttr();
  auto lanes = stream ? stream.getAs<IntegerAttr>("key_unroll") : IntegerAttr();
  auto physical = stream ? stream.getAs<BoolAttr>("physical_lifetimes_verified")
                         : BoolAttr();
  if (!mode || mode.getValue() != "unroll" || !lanes ||
      lanes.getInt() != factor || !physical || physical.getValue())
    return reject("missing typed stream or unsupported ownership certificate");
  auto keys = stream.getAs<IntegerAttr>("key_count");
  auto bn = stream.getAs<IntegerAttr>("key_step");
  auto queryUpper = stream.getAs<IntegerAttr>("query_upper");
  auto queryStep = stream.getAs<IntegerAttr>("query_step");
  if (!keys || !bn || !queryUpper || !queryStep)
    return reject("missing original query/key geometry");
  std::string reason;
  auto schedule =
      CVPQuerySchedule::create(keys.getInt(), bn.getInt(), queryUpper.getInt(),
                               queryStep.getInt(), reason);
  if (!schedule || !schedule->canEmitQueryUnroll(factor))
    return reject("unsupported query domain or overflowing group bound");
  if (scopes.size() != CVPQuerySchedule::kStageCount ||
      !loop.getInductionVar().getType().isInteger(64) ||
      getConstantIntValue(loop.getLowerBound()) != 0 ||
      getConstantIntValue(loop.getStep()) != bn.getInt() ||
      loop->hasAttr("unsignedCmp"))
    return reject("expected the signed widened stream and original key step");
  auto core = queryFuncCoreType(loop->getParentOfType<func::FuncOp>())
                  .value_or(TFuncCoreType::AIC_OR_AIV);
  for (auto [index, scope] : llvm::enumerate(scopes)) {
    bool cubeStage = index % 2 == 1;
    bool expected = core == TFuncCoreType::AIC   ? cubeStage
                    : core == TFuncCoreType::AIV ? !cubeStage
                                                 : true;
    if (static_cast<bool>(scope) != expected)
      return reject("missing or wrong-core stage after MIX splitting");
    if (!scope)
      continue;
    auto value =
        scope->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
    if (!value || value.getInt() != factor)
      return reject("stage lane factors disagree");
    auto stageCore = scope->getAttrOfType<TCoreTypeAttr>("hivm.loop_core_type");
    if (!stageCore || stageCore.getTcoretype() !=
                          (cubeStage ? TCoreType::CUBE : TCoreType::VECTOR))
      return reject("stage core assignment disagrees with logical role");
  }
  for (Value value : loop.getInitArgs())
    if (isa<TensorType>(value.getType()))
      return reject("query state must be bufferized before direct emission");
  // Do not duplicate an unknown effect outside its guarded stage. Whole-buffer
  // state copies and planned pointer/view operations use the existing rules.
  for (Operation &operation : loop.getBody()->without_terminator()) {
    if (isa<scope::ScopeOp>(operation) || isSynchronizationOp(&operation) ||
        isMemoryEffectFree(&operation) || isa<hivm::PointerCastOp>(operation) ||
        getWholeBufferCopy(&operation).first)
      continue;
    return reject("unknown effect outside a scheduled stage");
  }
  return success();
}

// Direct continuous-query Unroll specialization. This emits ordinary U1/U2/U4
// issue groups from the typed stages; it does not run the Skew emitter or apply
// a generic loop-unroll pass to its output. Lanes execute in issue order and
// explicitly thread all carried state. Keeping complete issue ticks together
// preserves the two-query storage contract even when one group crosses several
// short query boundaries. The unrelated legacy Unroll emitter remains
// unchanged.
static LogicalResult rewriteContinuousQueryUnroll(
    scf::ForOp forOp, SmallVector<scope::ScopeOp, 4> scopes, unsigned factor) {
  if (failed(validateContinuousQueryUnroll(forOp, scopes, factor)))
    return failure();
  IRRewriter rewriter(forOp.getContext());
  PreloadInfo info;
  info.funcCoreType = queryFuncCoreType(forOp->getParentOfType<func::FuncOp>())
                          .value_or(TFuncCoreType::AIC_OR_AIV);
  info.lb = forOp.getLowerBound();
  info.ub = forOp.getUpperBound();
  info.step = forOp.getStep();
  info.maxPreloadNum = CVPQuerySchedule::kStageCount;
  rewriter.setInsertionPoint(forOp);
  info.maxPreloadValue = rewriter.create<arith::ConstantOp>(
      forOp.getLoc(), rewriter.getI64IntegerAttr(info.maxPreloadNum));
  collectPreloadLocalBuffers(forOp->getParentOfType<func::FuncOp>(), info);
  SmallVector<Value> initialState;
  DenseMap<Value, hivm::PointerCastOp> adapted;
  Value conservativeEnd =
      preprocessLoopArgs(forOp, initialState, adapted, info, rewriter);
  for (auto [index, argument] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (adapted.contains(argument) && !forOp.getResult(index).use_empty()) {
      forOp.emitError("continuous Unroll cannot expose an adapted local-ring "
                      "result after the stream");
      return failure();
    }
  }
  // Preserve the common conservative stage drain; trimming its synchronization
  // is Stage 9. Empty assignments have no issue ticks, including no empty tail.
  Value nonEmpty = rewriter.create<arith::CmpIOp>(
      forOp.getLoc(), arith::CmpIPredicate::slt, info.lb, info.ub);
  Value issueEnd = rewriter.create<arith::SelectOp>(forOp.getLoc(), nonEmpty,
                                                    conservativeEnd, info.lb);
  int64_t keyStep = *getConstantIntValue(info.step);
  Value groupStep = rewriter.create<arith::ConstantOp>(
      forOp.getLoc(), rewriter.getI64IntegerAttr(keyStep * factor));
  DenseMap<Operation *, size_t> scopeToIdx;
  for (auto [index, scope] : llvm::enumerate(scopes))
    if (scope)
      scopeToIdx[scope] = index;
  auto grouped = rewriter.create<scf::ForOp>(
      forOp.getLoc(), info.lb, issueEnd, groupStep, initialState,
      [&](OpBuilder &builder, Location loc, Value base, ValueRange arguments) {
        SmallVector<Value> state(arguments.begin(), arguments.end());
        for (unsigned lane = 0; lane < factor; ++lane) {
          Value offset = builder.create<arith::ConstantOp>(
              loc, builder.getI64IntegerAttr(lane * keyStep));
          Value tick = builder.create<arith::AddIOp>(loc, base, offset);
          Value valid = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::slt, tick, issueEnd);
          auto guarded = builder.create<scf::IfOp>(
              loc, valid,
              [&](OpBuilder &body, Location bodyLoc) {
                auto next =
                    materializeScheduledTick(forOp, scopeToIdx, adapted, info,
                                             tick, state, body, bodyLoc,
                                             /*filterAdaptedState=*/true);
                body.create<scf::YieldOp>(bodyLoc, next);
              },
              [&](OpBuilder &body, Location bodyLoc) {
                body.create<scf::YieldOp>(bodyLoc, state);
              });
          guarded->setAttr("hivm.cvp_query.unroll_lane",
                           builder.getI32IntegerAttr(lane));
          state.assign(guarded.getResults().begin(),
                       guarded.getResults().end());
        }
        builder.create<scf::YieldOp>(loc, state);
      });
  grouped->setAttr(hivm::kCVUnrolledLoopName, rewriter.getUnitAttr());
  grouped->setAttr("hivm.cvp_attention.applied",
                   rewriter.getStringAttr("unroll"));
  grouped->setAttr("hivm.cvp_query.stream",
                   forOp->getAttr("hivm.cvp_query.stream"));
  grouped->setAttr(
      "hivm.cvp_query.unroll_emitted",
      rewriter.getDictionaryAttr(
          {rewriter.getNamedAttr("lanes", rewriter.getI32IntegerAttr(factor)),
           rewriter.getNamedAttr("lane_order", rewriter.getStringAttr("issue")),
           rewriter.getNamedAttr("ownership_required",
                                 rewriter.getBoolAttr(true))}));
  // Preserve Stage 6's symbolic origin without stamping it onto shifted IVs.
  SmallVector<Value> replacements;
  unsigned resultIndex = 0;
  for (auto argument : forOp.getRegionIterArgs()) {
    if (auto found = adapted.find(argument); found != adapted.end())
      replacements.push_back(found->second);
    else
      replacements.push_back(grouped.getResult(resultIndex++));
  }
  rewriter.replaceOp(forOp, replacements);
  return success();
}

struct PropagateAdaptor : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern<UnrealizedConversionCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("preload_adaptor"))
      return failure();
    auto input = op.getInputs()[0];
    for (auto &use : op->getUses()) {
      auto *user = use.getOwner();
      auto loc = user->getLoc();
      rewriter.setInsertionPoint(user);
      if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(user)) {
        auto maybeNewType = memref::ExpandShapeOp::computeExpandedType(
            cast<MemRefType>(input.getType()),
            expandShapeOp.getResultType().getShape(),
            expandShapeOp.getReassociationIndices());
        if (failed(maybeNewType))
          return expandShapeOp->emitOpError("Cannot expand shape");
        auto outputShape =
            getMixedValues(expandShapeOp.getStaticOutputShape(),
                           expandShapeOp.getOutputShape(), rewriter);
        auto newOp = rewriter.create<memref::ExpandShapeOp>(
            loc, maybeNewType.value(), input,
            expandShapeOp.getReassociationIndices(), outputShape);
        auto newAdaptorOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, expandShapeOp.getResultType(), newOp.getResult());
        newAdaptorOp->setAttr("preload_adaptor", rewriter.getUnitAttr());
        rewriter.replaceOp(expandShapeOp, newAdaptorOp);
        return success();
      }
      if (auto collapseShapeOp = dyn_cast<memref::CollapseShapeOp>(user)) {
        auto newOp = rewriter.create<memref::CollapseShapeOp>(
            loc, input, collapseShapeOp.getReassociationIndices());
        auto newAdaptorOp = rewriter.create<UnrealizedConversionCastOp>(
            loc, collapseShapeOp.getResultType(), newOp.getResult());
        newAdaptorOp->setAttr("preload_adaptor", rewriter.getUnitAttr());
        rewriter.replaceOp(collapseShapeOp, newAdaptorOp);
        return success();
      }
      rewriter.modifyOpInPlace(user, [&]() { use.set(input); });
      if (failed(verify(user))) {
        rewriter.modifyOpInPlace(user, [&]() { use.set(op->getResult(0)); });
        return user->emitOpError("Unhandled propagation");
      }
      return success();
    }
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    return failure();
  }
};

static void cleanupPreloadWorkspaceMarks(Operation &op) {
  op.walk([&](annotation::MarkOp markOp) {
    if (utils::isAnnotationWithAttr(markOp, hivm::PreloadWorkspaceAttr::name))
      markOp->removeAttr(hivm::PreloadWorkspaceAttr::name);
  });
}

void CreatePreloadPass::runOnOperation() {
  ModuleOp original = cast<ModuleOp>(getOperation());
  bool directUnroll = false;
  original.walk([&](scope::ScopeOp scope) {
    directUnroll |= scope->hasAttr("hivm.cvp_query.unroll_factor");
  });
  // An unsupported function or late materialization failure must not leave
  // another function partly transformed. The disabled/legacy path keeps its
  // existing operation identity and emitted IR.
  OwningOpRef<ModuleOp> candidate;
  ModuleOp moduleOp = original;
  const bool transactional = directUnroll;
  if (transactional) {
    candidate = OwningOpRef<ModuleOp>(cast<ModuleOp>(original->clone()));
    moduleOp = *candidate;
  }
  bool invalid = false;
  if (transactional) {
    moduleOp.walk([&](scope::ScopeOp scope) {
      bool marked = scope->hasAttr("hivm.cvp_query.unroll_factor");
      if (!marked && !scope->hasAttr(hivm::PreloadNumAttr::name) &&
          !scope->hasAttr(hivm::MaxPreloadNumAttr::name))
        return;
      auto parent = dyn_cast<scf::ForOp>(scope->getParentOp());
      auto lanes =
          scope->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
      auto maximum =
          scope->getAttrOfType<IntegerAttr>(hivm::MaxPreloadNumAttr::name);
      auto index =
          scope->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name);
      bool badMode = marked && (!lanes ||
                                (lanes.getInt() != 1 && lanes.getInt() != 2 &&
                                 lanes.getInt() != 4) ||
                                !maximum || maximum.getInt() != 4);
      if (!parent || badMode || !maximum || maximum.getInt() <= 0 ||
          maximum.getInt() > 4096 || !index || index.getInt() < 0 ||
          index.getInt() >= maximum.getInt()) {
        scope.emitOpError("malformed continuous-Unroll stage metadata");
        invalid = true;
      }
    });
    if (invalid)
      return signalPassFailure();
  }
  DenseMap<scf::ForOp, SmallVector<scope::ScopeOp, 4>> preload;
  moduleOp->walk([&](scope::ScopeOp scopeOp) {
    if (auto maxPreloadNumAttr = scopeOp->getAttrOfType<IntegerAttr>(
            hivm::MaxPreloadNumAttr::name)) {
      auto parentForOp = cast<scf::ForOp>(scopeOp->getParentOp());
      preload[parentForOp].resize(maxPreloadNumAttr.getInt(), nullptr);
    }
    if (auto preloadNumAttr =
            scopeOp->getAttrOfType<IntegerAttr>(hivm::PreloadNumAttr::name)) {
      auto parentForOp = cast<scf::ForOp>(scopeOp->getParentOp());
      auto preloadNum = preloadNumAttr.getInt();
      auto &preloadVec = preload[parentForOp];
      assert(preloadNum >= 0 && "PreloadNum must be non-negative integer");
      assert(preloadNum < static_cast<int64_t>(preloadVec.size()) &&
             "MaxPreloadNumAttr must be set");
      if (transactional && preloadVec[preloadNum]) {
        scopeOp.emitOpError("duplicate stage in continuous-Unroll module");
        invalid = true;
      }
      preloadVec[preloadNum] = scopeOp;
    }
  });
  if (invalid)
    return signalPassFailure();

  if (preload.empty())
    return;

  // Validate every inactive branch before mutating any preload loop. Do not
  // replace an unsupported selected state with an uninitialized pointer.
  for (auto &[forOp, scopes] : preload) {
    for (scope::ScopeOp scope : scopes) {
      if (!scope)
        continue;
      for (Value result : scope.getResults()) {
        Value returned = lookThroughScopeResult(result);
        if (getLocalBuffer(returned) || getSkippedLoopState(result, scope) ||
            canRematerializeViewResult(returned, scope))
          continue;
        scope.emitOpError("cannot preserve a preload scope result in the "
                          "inactive stage: expected a carried state through "
                          "unmodified whole-buffer copies or a supported view");
        return signalPassFailure();
      }
    }
  }

  for (auto &[forOp, scopes] : preload) {
    LDBG("Processing preload:\n" << forOp);
    auto scopeIt = llvm::find_if(scopes, [](scope::ScopeOp scopeOp) {
      return static_cast<bool>(scopeOp);
    });
    assert(scopeIt != scopes.end() && "Expected at least one preload scope");
    auto maxPreloadNumAttr =
        (*scopeIt)->getAttrOfType<IntegerAttr>(hivm::MaxPreloadNumAttr::name);
    assert(maxPreloadNumAttr && "MaxPreloadNumAttr must be set");
    IntegerAttr unroll;
    for (auto scope : scopes)
      if (scope && scope->hasAttr("hivm.cvp_query.unroll_factor")) {
        unroll =
            scope->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
        break;
      }
    if (unroll) {
      if (failed(rewriteContinuousQueryUnroll(forOp, scopes, unroll.getInt())))
        return signalPassFailure();
    } else {
      rewritePreloadLoop(forOp, scopes,
                         static_cast<size_t>(maxPreloadNumAttr.getInt()));
    }
  }

  cleanupPreloadWorkspaceMarks(*moduleOp);

  RewritePatternSet patterns(&getContext());
  patterns.add<PropagateAdaptor>(&getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  LDBG("After processing preload:\n" << *moduleOp);

  PassManager pm(&getContext());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(scope::createInlineScopePass());

  if (failed(pm.run(moduleOp))) {
    signalPassFailure();
    return;
  }
  if (transactional) {
    if (failed(verify(moduleOp)))
      return signalPassFailure();
    original.getBodyRegion().takeBody(moduleOp.getBodyRegion());
  }
}

std::unique_ptr<Pass> mlir::hivm::createCreatePreloadPass() {
  return std::make_unique<CreatePreloadPass>();
}

//===- CVPAttentionLayout.cpp - Closed attention layout rewrite
//------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <tuple>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONLAYOUT
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {

constexpr StringLiteral kScoreLayout = "score_layout";

/// Analyze a disposable flattened copy. The executable scopes and their lag
/// values are never flattened or rescheduled by this transformation.
FailureOr<CVPAttentionPlan> analyzeScoped(scf::ForOp loop,
                                          CVPAttentionScoreLayout layout,
                                          std::string &reason) {
  auto function = loop->getParentOfType<func::FuncOp>();
  auto copy = cast<func::FuncOp>(function->clone());
  auto cleanup = llvm::make_scope_exit([&] { copy.erase(); });
  SmallVector<Operation *> originals, copies;
  function.walk([&](Operation *op) { originals.push_back(op); });
  copy.walk([&](Operation *op) { copies.push_back(op); });
  assert(originals.size() == copies.size());
  llvm::DenseMap<Operation *, Operation *> toOriginal, toCopy;
  for (auto [original, cloned] : llvm::zip(originals, copies)) {
    toOriginal[cloned] = original;
    toCopy[original] = cloned;
  }
  auto clonedLoop = cast<scf::ForOp>(toCopy.lookup(loop));
  SmallVector<scope::ScopeOp> scopes(
      clonedLoop.getBody()->getOps<scope::ScopeOp>());
  if (scopes.size() != 4) {
    reason = "expected four Skew scopes";
    return failure();
  }
  int64_t previous = INT64_MAX;
  for (auto [index, scopeOp] : llvm::enumerate(scopes)) {
    auto lag = scopeOp->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto expected = TCoreTypeAttr::get(
        loop.getContext(), index % 2 ? TCoreType::VECTOR : TCoreType::CUBE);
    if (!lag || lag.getInt() < 0 || lag.getInt() >= previous ||
        scopeOp->getAttr(kPipelinedLoopCoreTypeAttrName) != expected ||
        !scopeOp->getRegion(0).hasOneBlock()) {
      reason = "unsupported scope lag, core assignment or region";
      return failure();
    }
    previous = lag.getInt();
  }
  SmallVector<Operation *> originalScopes;
  for (scope::ScopeOp scopeOp : scopes)
    originalScopes.push_back(toOriginal.lookup(scopeOp));
  for (scope::ScopeOp scopeOp : scopes) {
    Block &body = scopeOp->getRegion(0).front();
    auto terminator = dyn_cast<scope::ReturnOp>(body.getTerminator());
    if (!terminator || body.getNumArguments() || scopeOp->getNumOperands() ||
        terminator->getNumOperands() != scopeOp->getNumResults()) {
      reason = "unsupported scope interface";
      return failure();
    }
    scopeOp->replaceAllUsesWith(terminator.getOperands());
    for (Operation &op : llvm::make_early_inc_range(body.without_terminator()))
      op.moveBefore(scopeOp);
    scopeOp.erase();
  }
  auto plan = analyzeCVPAttention(clonedLoop, reason, layout);
  if (failed(plan))
    return failure();
  auto mapValue = [&](Value value) -> Value {
    auto result = cast<OpResult>(value);
    return toOriginal.lookup(result.getOwner())
        ->getResult(result.getResultNumber());
  };
  plan->loop = loop;
  plan->qk = cast<MmadL1Op>(toOriginal.lookup(plan->qk));
  plan->pv = cast<MmadL1Op>(toOriginal.lookup(plan->pv));
  plan->rawScore = mapValue(plan->rawScore);
  plan->scaledScore = mapValue(plan->scaledScore);
  plan->p32 = mapValue(plan->p32);
  plan->p16 = mapValue(plan->p16);
  plan->alpha = mapValue(plan->alpha);
  for (auto &entry : plan->roles)
    entry.first = toOriginal.lookup(entry.first);
  auto role = [&](CVPAttentionRole wanted) -> Operation * {
    for (auto [op, current] : plan->roles)
      if (current == wanted)
        return op;
    return nullptr;
  };
  for (auto [index, op] : llvm::enumerate(SmallVector<Operation *>{
           plan->qk.getOperation(), role(CVPAttentionRole::P32),
           plan->pv.getOperation(), role(CVPAttentionRole::Output)})) {
    if (!op || op->getParentOp() != originalScopes[index]) {
      reason = "recurrence roles do not belong to the expected Skew scopes";
      return failure();
    }
  }
  return plan;
}

bool tensorShape(Value value, ArrayRef<int64_t> shape, Type element) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && !type.getEncoding() &&
         type.getShape() == shape && type.getElementType() == element;
}

bool onlyUsers(Value value, ArrayRef<Operation *> allowed) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return llvm::is_contained(allowed, user);
  });
}

struct LayoutChain {
  CVPAttentionPlan plan;
  FixpipeOp qkTransfer;
  VMulOp scale;
  VReduceOp maximum, sum;
  VSubOp shift;
  VExpOp probability;
  VCastOp cast;
  CopyOp pCopy;
  bufferization::ToTensorOp scoreView, pView;
  memref::AllocOp scoreBuffer, pBuffer;
  tensor::ExpandShapeOp pExpand, pNZ;
  VTransposeOp pTranspose;
  SmallVector<memref::MemorySpaceCastOp> deadSpaceCasts;
};

FailureOr<LayoutChain> matchChain(CVPAttentionPlan plan, std::string &reason) {
  LayoutChain chain;
  chain.plan = plan;
  for (auto [op, role] : plan.roles) {
    if (role == CVPAttentionRole::QKTransfer)
      chain.qkTransfer = dyn_cast<FixpipeOp>(op);
    if (role == CVPAttentionRole::TileMax)
      chain.maximum = dyn_cast<VReduceOp>(op);
    if (role == CVPAttentionRole::TileSum)
      chain.sum = dyn_cast<VReduceOp>(op);
  }
  chain.scale = plan.scaledScore.getDefiningOp<VMulOp>();
  chain.probability = plan.p32.getDefiningOp<VExpOp>();
  chain.cast = plan.p16.getDefiningOp<VCastOp>();
  chain.shift = chain.probability->getOperand(0).getDefiningOp<VSubOp>();
  chain.scoreView = plan.rawScore.getDefiningOp<bufferization::ToTensorOp>();
  chain.pView = plan.pv.getA().getDefiningOp<bufferization::ToTensorOp>();
  auto reject = [&](StringRef why) -> FailureOr<LayoutChain> {
    reason = why.str();
    return failure();
  };
  if (!chain.qkTransfer || !chain.maximum || !chain.sum || !chain.scale ||
      !chain.shift || !chain.cast || !chain.scoreView || !chain.pView)
    return reject("unsupported score or probability view chain");
  chain.scoreBuffer =
      chain.scoreView->getOperand(0).getDefiningOp<memref::AllocOp>();
  chain.pBuffer = chain.pView->getOperand(0).getDefiningOp<memref::AllocOp>();
  if (!chain.scoreBuffer || !chain.pBuffer)
    return reject("layout requires direct private score/P allocations");
  if (chain.scoreBuffer.getType().getMemorySpace() !=
          AddressSpaceAttr::get(plan.loop.getContext(), AddressSpace::UB) ||
      chain.pBuffer.getType().getMemorySpace() !=
          AddressSpaceAttr::get(plan.loop.getContext(), AddressSpace::L1))
    return reject("score/P buffers are not in the expected UB/L1 spaces");
  for (Operation *user : chain.pBuffer->getResult(0).getUsers())
    if (auto copy = dyn_cast<CopyOp>(user)) {
      if (chain.pCopy)
        return reject("ambiguous P copy");
      chain.pCopy = copy;
    }
  if (!chain.pCopy)
    return reject("missing P copy");
  chain.pNZ = chain.pCopy.getSrc().getDefiningOp<tensor::ExpandShapeOp>();
  if (chain.pNZ)
    chain.pTranspose = chain.pNZ.getSrc().getDefiningOp<VTransposeOp>();
  if (chain.pTranspose)
    chain.pExpand =
        chain.pTranspose->getOperand(0).getDefiningOp<tensor::ExpandShapeOp>();
  if (!chain.pNZ || !chain.pTranspose || !chain.pExpand ||
      chain.pExpand.getSrc() != plan.p16 ||
      chain.pTranspose.getPermutation() != ArrayRef<int64_t>{1, 0, 2})
    return reject("unsupported normalized probability mapping");
  if (!onlyUsers(plan.rawScore, {chain.scale}) ||
      !onlyUsers(plan.scaledScore, {chain.maximum, chain.shift}) ||
      !onlyUsers(chain.shift->getResult(0), {chain.probability}) ||
      !onlyUsers(plan.p32, {chain.cast, chain.sum}) ||
      !onlyUsers(plan.p16, {chain.pExpand}) ||
      !onlyUsers(chain.pView->getResult(0), {plan.pv}) ||
      !chain.pExpand->getResult(0).hasOneUse() ||
      !chain.pTranspose->getResult(0).hasOneUse() ||
      !chain.pNZ->getResult(0).hasOneUse() ||
      !plan.qk->getResult(0).hasOneUse())
    return reject("score/P data escapes the complete layout chain");
  for (auto [allocation, writer, view] :
       {std::make_tuple(chain.scoreBuffer, chain.qkTransfer.getOperation(),
                        chain.scoreView.getOperation()),
        std::make_tuple(chain.pBuffer, chain.pCopy.getOperation(),
                        chain.pView.getOperation())}) {
    if (!allocation.getType().hasStaticShape() ||
        !allocation.getType().getLayout().isIdentity() ||
        allocation->getNumOperands())
      return reject("unsupported private buffer layout");
    for (Operation *user : allocation->getResult(0).getUsers()) {
      if (user == writer || user == view || isa<annotation::MarkOp>(user))
        continue;
      // Staged Unroll may leave an unused space cast beside the real FIX
      // destination. It has no memory effect or consumer; remove it before
      // changing the allocation shape. A used alias still escapes this proof.
      auto spaceCast = dyn_cast<memref::MemorySpaceCastOp>(user);
      if (spaceCast && spaceCast->use_empty()) {
        chain.deadSpaceCasts.push_back(spaceCast);
        continue;
      }
      return reject("private score/P buffer has another use or alias");
    }
  }
  Builder b(plan.loop.getContext());
  const int64_t m = plan.blockM, n = plan.blockN, d = plan.headDim;
  const bool keyMajor = plan.scoreLayout == CVPAttentionScoreLayout::KeyMajor;
  SmallVector<int64_t> score =
      keyMajor ? SmallVector<int64_t>{n, m} : SmallVector<int64_t>{m, n};
  SmallVector<int64_t> nz = {score[1] / 16, score[0] / 16, 16, 16};
  if (!tensorShape(plan.qk.getA(), {d / 16, score[0] / 16, 16, 16},
                   b.getF16Type()) ||
      !tensorShape(plan.qk.getB(), {d / 16, score[1] / 16, 16, 16},
                   b.getF16Type()) ||
      !tensorShape(plan.qk.getC(), nz, b.getF32Type()) ||
      !tensorShape(plan.qk->getResult(0), nz, b.getF32Type()) ||
      !tensorShape(chain.pExpand->getResult(0), {score[0], score[1] / 16, 16},
                   b.getF16Type()) ||
      !tensorShape(chain.pTranspose->getResult(0),
                   {score[1] / 16, score[0], 16}, b.getF16Type()) ||
      !tensorShape(chain.pNZ->getResult(0), nz, b.getF16Type()) ||
      chain.scoreBuffer.getType().getShape() != ArrayRef<int64_t>(score) ||
      chain.pBuffer.getType().getShape() != ArrayRef<int64_t>(nz) ||
      chain.qkTransfer.getDmaMode() != FixpipeDMAMode::NZ2ND ||
      chain.qkTransfer.getDualDstMode() !=
          FixpipeDualDstModeAttr::get(plan.loop.getContext(),
                                      FixpipeDualDstMode::NO_DUAL) ||
      chain.qkTransfer->getNumOperands() != 2)
    return reject("normalized matmul/fixpipe/P geometry is unsupported");
  for (VReduceOp reduce : {chain.maximum, chain.sum}) {
    auto init = reduce->getOperand(1).getDefiningOp<tensor::ExpandShapeOp>();
    if (!init || !tensorShape(init.getSrc(), {m}, b.getF32Type()))
      return reject("reduction initializer is not a complete query row");
    for (Operation *user : reduce->getResult(0).getUsers()) {
      auto collapse = dyn_cast<tensor::CollapseShapeOp>(user);
      if (!collapse ||
          !tensorShape(collapse->getResult(0), {m}, b.getF32Type()))
        return reject("reduced result has a layout-sensitive extra user");
    }
  }
  auto maximumView =
      chain.shift->getOperand(1).getDefiningOp<tensor::ExpandShapeOp>();
  auto broadcast = chain.shift->getAttrOfType<DenseI64ArrayAttr>("broadcast");
  if (!maximumView || !tensorShape(maximumView.getSrc(), {m}, b.getF32Type()) ||
      !broadcast ||
      broadcast.asArrayRef() != ArrayRef<int64_t>{keyMajor ? 0 : 1})
    return reject("probability broadcast is not a complete query row");
  for (Operation *op :
       {chain.scale.getOperation(), chain.shift.getOperation(),
        chain.probability.getOperation(), chain.cast.getOperation(),
        chain.pTranspose.getOperation()}) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(op);
    if (!dps || dps.getNumDpsInits() != 1 ||
        !dps.getDpsInits()[0].getDefiningOp<tensor::EmptyOp>())
      return reject("layout-sensitive destination is not an empty tensor");
  }
  if (!plan.qk.getC().getDefiningOp<tensor::EmptyOp>())
    return reject("QK destination is not an empty tensor");
  return chain;
}

void rewriteChain(LayoutChain &chain, IRRewriter &rewriter) {
  // This is the pass's disposable module copy. Failure in any later chain
  // leaves the original module, including all aliases, unchanged.
  for (memref::MemorySpaceCastOp spaceCast : chain.deadSpaceCasts)
    rewriter.eraseOp(spaceCast);
  auto &plan = chain.plan;
  const int64_t m = plan.blockM, n = plan.blockN;
  MLIRContext *context = rewriter.getContext();
  auto tensorType = [&](Value old, ArrayRef<int64_t> shape) {
    return RankedTensorType::get(
        shape, cast<RankedTensorType>(old.getType()).getElementType());
  };
  llvm::DenseMap<Value, Value> emptyTensors;
  SmallVector<Operation *> oldViews;
  auto resizeInit = [&](Operation *op, ArrayRef<int64_t> shape) {
    auto dps = cast<DestinationStyleOpInterface>(op);
    Value old = dps.getDpsInits()[0];
    Value &replacement = emptyTensors[old];
    if (!replacement) {
      rewriter.setInsertionPoint(op);
      replacement = rewriter.create<tensor::EmptyOp>(
          op->getLoc(), shape,
          cast<RankedTensorType>(old.getType()).getElementType());
    }
    dps.getDpsInitsMutable().assign(ValueRange{replacement});
    op->getResult(0).setType(tensorType(op->getResult(0), shape));
  };
  auto resizeBuffer = [&](memref::AllocOp allocation, ArrayRef<int64_t> shape,
                          bufferization::ToTensorOp view) {
    auto old = allocation.getType();
    allocation->getResult(0).setType(MemRefType::get(
        shape, old.getElementType(), old.getLayout(), old.getMemorySpace()));
    view->getResult(0).setType(
        RankedTensorType::get(shape, old.getElementType()));
  };
  auto rowView = [&](tensor::ExpandShapeOp old) -> Value {
    oldViews.push_back(old);
    rewriter.setInsertionPoint(old);
    return rewriter.create<tensor::ExpandShapeOp>(
        old.getLoc(), RankedTensorType::get({1, m}, rewriter.getF32Type()),
        old.getSrc(),
        ArrayRef<ReassociationIndices>{ReassociationIndices{0, 1}});
  };
  Value a = plan.qk.getA(), b = plan.qk.getB();
  Value realM = plan.qk.getRealM(), realN = plan.qk.getRealN();
  plan.qk.getAMutable().assign(b);
  plan.qk.getBMutable().assign(a);
  plan.qk.getRealMMutable().assign(realN);
  plan.qk.getRealNMutable().assign(realM);
  resizeInit(plan.qk, {m / 16, n / 16, 16, 16});
  resizeBuffer(chain.scoreBuffer, {n, m}, chain.scoreView);
  for (Operation *op :
       {chain.scale.getOperation(), chain.shift.getOperation(),
        chain.probability.getOperation(), chain.cast.getOperation()})
    resizeInit(op, {n, m});
  for (VReduceOp reduce : {chain.maximum, chain.sum}) {
    auto old = reduce->getOperand(1).getDefiningOp<tensor::ExpandShapeOp>();
    reduce->setOperand(1, rowView(old));
    reduce.setReduceDimsAttr(rewriter.getDenseI64ArrayAttr({0}));
    reduce->getResult(0).setType(
        RankedTensorType::get({1, m}, rewriter.getF32Type()));
  }
  auto oldMaximum =
      chain.shift->getOperand(1).getDefiningOp<tensor::ExpandShapeOp>();
  chain.shift->setOperand(1, rowView(oldMaximum));
  chain.shift.setBroadcastAttr(rewriter.getDenseI64ArrayAttr({0}));
  chain.pExpand->getResult(0).setType(
      tensorType(chain.pExpand->getResult(0), {n, m / 16, 16}));
  chain.pExpand.setStaticOutputShapeAttr(
      rewriter.getDenseI64ArrayAttr({n, m / 16, 16}));
  resizeInit(chain.pTranspose, {m / 16, n, 16});
  chain.pNZ->getResult(0).setType(
      tensorType(chain.pNZ->getResult(0), {m / 16, n / 16, 16, 16}));
  chain.pNZ.setStaticOutputShapeAttr(
      rewriter.getDenseI64ArrayAttr({m / 16, n / 16, 16, 16}));
  resizeBuffer(chain.pBuffer, {m / 16, n / 16, 16, 16}, chain.pView);
  plan.pv->setAttr("a_transpose", UnitAttr::get(context));
  llvm::SmallPtrSet<Operation *, 4> visitedViews;
  for (Operation *op : oldViews)
    if (visitedViews.insert(op).second && op->use_empty())
      rewriter.eraseOp(op);
  for (auto [old, replacement] : emptyTensors) {
    (void)replacement;
    if (old.use_empty())
      rewriter.eraseOp(old.getDefiningOp());
  }
}

bool summaryMatches(DictionaryAttr recorded, const CVPAttentionPlan &plan) {
  auto actual = plan.getSummary(plan.loop->getContext());
  for (StringRef name :
       {"version", "block_m", "block_n", "head_dim", "key_count", "key_tiles",
        "output_state", "sum_state", "max_state", "scale", "initial_max",
        "initial_sum", "initial_output"})
    if (recorded.get(name) != actual.get(name))
      return false;
  auto layout = recorded.getAs<StringAttr>(kScoreLayout);
  return plan.scoreLayout == CVPAttentionScoreLayout::KeyMajor
             ? layout && layout.getValue() == "key-major"
             : !layout || layout.getValue() == "query-major";
}

struct CVPAttentionLayoutPass
    : impl::CVPAttentionLayoutBase<CVPAttentionLayoutPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto copy = cast<ModuleOp>(module->clone());
    auto cleanup = llvm::make_scope_exit([&] { copy.erase(); });
    SmallVector<func::FuncOp> originals(module.getOps<func::FuncOp>());
    SmallVector<func::FuncOp> functions(copy.getOps<func::FuncOp>());
    SmallVector<unsigned> changed;
    unsigned matched = 0;
    IRRewriter rewriter(&getContext());
    for (auto indexedFunction : llvm::enumerate(functions)) {
      unsigned index = indexedFunction.index();
      func::FuncOp function = indexedFunction.value();
      auto verified =
          function->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
      if (!verified || verified.empty())
        continue;
      auto fail = [&](StringRef why) {
        originals[index].emitError()
            << "CVP attention layout rejected: " << why;
        signalPassFailure();
      };
      if (function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(&getContext(), TFuncCoreType::MIX)) {
        fail("requires the mixed kernel before splitting");
        return;
      }
      bool bound = false;
      function.walk([&](GetSubBlockIdxOp) { bound = true; });
      if (bound) {
        fail("requires layout selection before subblock binding");
        return;
      }
      SmallVector<scf::ForOp> loops;
      function.walk([&](scf::ForOp loop) {
        if (llvm::range_size(loop.getBody()->getOps<scope::ScopeOp>()) == 4)
          loops.push_back(loop);
      });
      if (loops.size() != verified.size()) {
        fail("verified attention-loop count changed");
        return;
      }
      SmallVector<Attribute> updated;
      unsigned rewrites = 0;
      for (auto [loopIndex, loop] : llvm::enumerate(loops)) {
        auto recorded = dyn_cast<DictionaryAttr>(verified[loopIndex]);
        if (!recorded) {
          fail("malformed verified plan");
          return;
        }
        auto storage = recorded.getAs<StringAttr>(kScoreLayout);
        bool keyMajor = storage && storage.getValue() == "key-major";
        std::string reason;
        auto plan =
            analyzeScoped(loop,
                          keyMajor ? CVPAttentionScoreLayout::KeyMajor
                                   : CVPAttentionScoreLayout::QueryMajor,
                          reason);
        if (failed(plan) || !summaryMatches(recorded, *plan)) {
          fail(failed(plan)
                   ? StringRef(reason)
                   : StringRef("typed recurrence differs from provenance"));
          return;
        }
        auto chain = matchChain(*plan, reason);
        if (failed(chain)) {
          fail(reason);
          return;
        }
        ++matched;
        if (!keyMajor) {
          auto applied =
              loop->getAttrOfType<StringAttr>(kCVPAttentionAppliedAttr);
          auto stagedFactor =
              loop->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
          auto mode = recorded.getAs<StringAttr>("pipeline_mode");
          bool stagedUnroll = applied &&
                              applied.getValue() == "unroll-staged" && mode &&
                              mode.getValue() == "unroll" && stagedFactor &&
                              recorded.get("key_unroll") == stagedFactor;
          if (!applied || (applied.getValue() != "skew" && !stagedUnroll)) {
            fail("actual Skew CVPipeline application is required");
            return;
          }
          if (!allowFP32SumReassociation) {
            fail("key-major storage requires explicit FP32 sum reassociation "
                 "permission");
            return;
          }
          rewriteChain(*chain, rewriter);
          auto check =
              analyzeScoped(loop, CVPAttentionScoreLayout::KeyMajor, reason);
          if (failed(check) || failed(matchChain(*check, reason))) {
            fail("rewritten layout did not preserve the complete recurrence: " +
                 reason);
            return;
          }
          ++rewrites;
        }
        NamedAttrList summary(recorded);
        summary.set(kScoreLayout, rewriter.getStringAttr("key-major"));
        summary.set("numerical_policy",
                    rewriter.getStringAttr("reassociate-fp32-sum"));
        summary.set("physical_score_shape", rewriter.getDenseI64ArrayAttr(
                                                {plan->blockN, plan->blockM}));
        summary.set("physical_lifetimes_verified", rewriter.getBoolAttr(false));
        updated.push_back(summary.getDictionary(&getContext()));
      }
      if (!rewrites)
        continue;
      function.walk([](Operation *op) {
        op->removeAttr(kCVPAttentionPlanAttr);
        op->removeAttr(kCVPAttentionRoleAttr);
        op->removeAttr(kCVPAttentionAppliedAttr);
        op->removeAttr(kCVPAttentionExpectedAttr);
      });
      function->setAttr(kCVPAttentionVerifiedAttr,
                        rewriter.getArrayAttr(updated));
      function->setAttr("hivm.cvp_attention.layout_rewrites",
                        rewriter.getI64IntegerAttr(rewrites));
      if (failed(verify(function))) {
        fail("rewritten function failed IR verification");
        return;
      }
      changed.push_back(index);
    }
    if (!matched && requireMatch) {
      module.emitError(
          "CVP attention layout requires a verified mixed Skew recurrence");
      signalPassFailure();
      return;
    }
    // Commit only after every requested function has passed. Other functions,
    // module attributes, scope schedules and external signatures are untouched.
    for (unsigned index : changed) {
      originals[index].getBody().takeBody(functions[index].getBody());
      originals[index]->setAttrs(functions[index]->getAttrs());
    }
  }
};
} // namespace

std::unique_ptr<Pass>
createCVPAttentionLayoutPass(const CVPAttentionLayoutOptions &options) {
  return std::make_unique<CVPAttentionLayoutPass>(options);
}
} // namespace hivm
} // namespace mlir

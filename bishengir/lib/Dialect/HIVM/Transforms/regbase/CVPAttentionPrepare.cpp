//===- CVPAttentionPrepare.cpp - Describe and verify attention route
//--------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/StringMap.h"

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONPREPARE
#define GEN_PASS_DEF_VERIFYCVPATTENTION
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
void clearAnnotations(func::FuncOp function) {
  function.walk([](Operation *op) {
    op->removeAttr(kCVPAttentionPlanAttr);
    op->removeAttr(kCVPAttentionRoleAttr);
    op->removeAttr(kCVPAttentionAppliedAttr);
    op->removeAttr(kCVPAttentionExpectedAttr);
    op->removeAttr(kCVPAttentionVerifiedAttr);
  });
}
struct CVPAttentionPreparePass
    : impl::CVPAttentionPrepareBase<CVPAttentionPreparePass> {
  using Base::Base;
  void runOnOperation() override {
    auto function = getOperation();
    // User-supplied or stale annotations never bypass recognition.
    clearAnnotations(function);
    unsigned recognized = 0;
    function.walk([&](scf::ForOp loop) {
      unsigned count = 0;
      for (auto op : loop.getBody()->getOps<MmadL1Op>()) {
        (void)op;
        ++count;
      }
      if (count != 2)
        return;
      std::string reason;
      auto plan = analyzeCVPAttention(loop, reason);
      if (failed(plan)) {
        if (emitRemarks)
          loop.emitRemark() << "CVP attention skipped: " << reason;
        return;
      }
      ++recognized;
      loop->setAttr(kCVPAttentionPlanAttr, plan->getSummary(&getContext()));
      for (auto [op, role] : plan->roles)
        op->setAttr(
            kCVPAttentionRoleAttr,
            StringAttr::get(&getContext(), stringifyCVPAttentionRole(role)));
      if (emitRemarks)
        loop.emitRemark() << "CVP attention recognized: BM=" << plan->blockM
                          << " BN=" << plan->blockN << " HD=" << plan->headDim
                          << "; effective features: none";
    });
    function->setAttr(
        kCVPAttentionExpectedAttr,
        IntegerAttr::get(IntegerType::get(&getContext(), 64), recognized));
  }
};
struct VerifyCVPAttentionPass
    : impl::VerifyCVPAttentionBase<VerifyCVPAttentionPass> {
  using Base::Base;
  void runOnOperation() override {
    auto function = getOperation();
    bool invalid = false;
    unsigned checked = 0;
    SmallVector<Attribute> verifiedPlans;
    auto expected =
        function->getAttrOfType<IntegerAttr>(kCVPAttentionExpectedAttr);
    if (expected &&
        (!expected.getType().isInteger(64) || expected.getInt() < 0)) {
      function.emitError(
          "CVP attention route invalid: malformed expected count");
      signalPassFailure();
      return;
    }
    function.walk([&](scf::ForOp loop) {
      if (!loop->hasAttr(kCVPAttentionPlanAttr))
        return;
      ++checked;
      auto fail = [&](StringRef message) {
        loop.emitError() << "CVP attention route invalid: " << message;
        invalid = true;
      };
      auto applied = loop->getAttrOfType<StringAttr>(kCVPAttentionAppliedAttr);
      bool stagedUnroll = applied && applied.getValue() == "unroll-staged";
      auto unroll =
          loop->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
      if (stagedUnroll &&
          (!unroll || (unroll.getInt() != 1 && unroll.getInt() != 2 &&
                       unroll.getInt() != 4))) {
        fail("malformed continuous Unroll lane factor");
        return;
      }
      if (!applied || (applied.getValue() != "skew" && !stagedUnroll)) {
        fail("successful Skew CVPipeline application was not recorded");
        return;
      }
      llvm::StringMap<Operation *> roles;
      bool duplicate = false;
      loop.walk([&](Operation *op) {
        if (auto role = op->getAttrOfType<StringAttr>(kCVPAttentionRoleAttr)) {
          if (!roles.try_emplace(role.getValue(), op).second)
            duplicate = true;
        }
      });
      if (duplicate) {
        fail("ambiguous cloned role");
        return;
      }
      Operation *qk = roles.lookup("qk"), *p = roles.lookup("p32"),
                *pv = roles.lookup("pv"), *output = roles.lookup("output");
      if (!qk || !p || !pv || !output || !isa<MmadL1Op>(qk) ||
          !isa<VExpOp>(p) || !isa<MmadL1Op>(pv) || !isa<VAddOp>(output)) {
        fail("missing or mistyped QK/P/PV/O roles");
        return;
      }
      int64_t previous = INT64_MAX;
      unsigned index = 0;
      for (Operation *op : {qk, p, pv, output}) {
        auto scope = op->getParentOfType<scope::ScopeOp>();
        if (!scope || scope->getParentOp() != loop) {
          fail("role is outside the transformed key-loop scope");
          return;
        }
        if (stagedUnroll &&
            scope->getAttr("hivm.cvp_query.unroll_factor") != unroll) {
          fail("continuous Unroll stage lane factors disagree");
          return;
        }
        auto offset = scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
        auto core = scope->getAttr(kPipelinedLoopCoreTypeAttrName);
        auto expected = TCoreTypeAttr::get(
            &getContext(), index % 2 ? TCoreType::VECTOR : TCoreType::CUBE);
        if (!offset || offset.getInt() < 0 || offset.getInt() >= previous ||
            core != expected) {
          fail("invalid QK/P/PV/O scope order or core assignment");
          return;
        }
        previous = offset.getInt();
        ++index;
      }
      auto plan = loop->getAttrOfType<DictionaryAttr>(kCVPAttentionPlanAttr);
      if (!plan) {
        fail("malformed typed plan");
        return;
      }
      if (stagedUnroll) {
        NamedAttrList staged(plan);
        staged.set("pipeline_mode", StringAttr::get(&getContext(), "unroll"));
        staged.set("key_unroll", unroll);
        verifiedPlans.push_back(staged.getDictionary(&getContext()));
      } else {
        verifiedPlans.push_back(plan);
      }
      if (emitRemarks)
        loop.emitRemark(stagedUnroll
                            ? "CVP attention stages verified for direct "
                              "ordinary-Unroll emission; ownership pending"
                            : "CVP attention route verified: Skew QK/P/PV/O; "
                              "features none");
    });
    if ((expected && static_cast<int64_t>(checked) != expected.getInt()) ||
        (!expected && (checked || retainProvenance))) {
      function.emitError("CVP attention route invalid: recognized plan count "
                         "changed or was not recorded");
      invalid = true;
    }
    if (invalid) {
      signalPassFailure();
      return;
    }
    // Analysis-only full compilation strips metadata at this boundary, so it
    // cannot accidentally change downstream CSE or memory planning behavior.
    if (dropAnnotations)
      clearAnnotations(function);
    if (retainProvenance && !verifiedPlans.empty())
      function->setAttr(kCVPAttentionVerifiedAttr,
                        ArrayAttr::get(&getContext(), verifiedPlans));
  }
};
} // namespace
std::unique_ptr<Pass>
createCVPAttentionPreparePass(const CVPAttentionPrepareOptions &options) {
  return std::make_unique<CVPAttentionPreparePass>(options);
}
std::unique_ptr<Pass>
createVerifyCVPAttentionPass(const VerifyCVPAttentionOptions &options) {
  return std::make_unique<VerifyCVPAttentionPass>(options);
}
} // namespace hivm
} // namespace mlir

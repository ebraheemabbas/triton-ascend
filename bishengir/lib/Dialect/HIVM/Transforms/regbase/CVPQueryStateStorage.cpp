//===- CVPQueryStateStorage.cpp - Stage-local query state storage
//-----------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include <array>
#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CVPQUERYSTATESTORAGE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr StringLiteral kStream = "hivm.cvp_query.stream";
constexpr StringLiteral kLowered = "hivm.cvp_query.state_storage";
constexpr StringLiteral kBank = "hivm.cvp_query.state_bank";
constexpr StringLiteral kWrite = "hivm.cvp_query.state_write";

struct StatePlan {
  scf::ForOp loop;
  scope::ScopeOp producer, output;
  scf::IfOp finalization;
  Value odd, sum, maximum, accumulator;
  RankedTensorType rowType, outputType;
};

static std::optional<int64_t> field(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<IntegerAttr>(name) : IntegerAttr();
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

static bool selectIs(arith::SelectOp op, Value condition, Value onTrue,
                     Value onFalse) {
  return op && op.getCondition() == condition && op.getTrueValue() == onTrue &&
         op.getFalseValue() == onFalse;
}

static bool queryParity(Value condition, scf::ForOp loop, int64_t keys) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq ||
      getConstantIntValue(compare.getRhs()) != 1)
    return false;
  auto remainder = compare.getLhs().getDefiningOp<arith::RemSIOp>();
  if (!remainder || getConstantIntValue(remainder.getRhs()) != 2)
    return false;
  auto ordinal = remainder.getLhs().getDefiningOp<arith::DivSIOp>();
  return ordinal && ordinal.getLhs() == loop.getInductionVar() &&
         getConstantIntValue(ordinal.getRhs()) == keys;
}

static bool isLastKey(Value condition, scf::ForOp loop, int64_t keys,
                      int64_t step) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq ||
      getConstantIntValue(compare.getRhs()) != keys - step)
    return false;
  auto local = compare.getLhs().getDefiningOp<arith::RemSIOp>();
  return local && local.getLhs() == loop.getInductionVar() &&
         getConstantIntValue(local.getRhs()) == keys;
}

static bool in(Operation *op, Operation *parent) {
  return op == parent || parent->isAncestor(op);
}

static FailureOr<StatePlan> analyze(scf::ForOp loop, std::string &reason) {
  auto reject = [&](StringRef message) -> FailureOr<StatePlan> {
    reason = message.str();
    return failure();
  };
  auto stream = loop->getAttrOfType<DictionaryAttr>(kStream);
  auto mode = stream ? stream.getAs<StringAttr>("pipeline_mode") : StringAttr();
  auto keys = field(stream, "key_count"), step = field(stream, "key_step");
  if (!loop->hasAttr("hivm.cvp_query.output_storage") || !mode ||
      mode.getValue() != "skew" || field(stream, "query_slots") != 2 || !keys ||
      !step || *step <= 0 || *keys / *step < 2 || *keys % *step ||
      getConstantIntValue(loop.getLowerBound()) != 0 ||
      getConstantIntValue(loop.getStep()) != step ||
      !loop.getInductionVar().getType().isInteger(64) ||
      loop->hasAttr("unsignedCmp") || loop.getInitArgs().size() != 5 ||
      !loop->use_empty())
    return reject("expected the closed five-state Skew query stream");

  StatePlan plan;
  plan.loop = loop;
  plan.outputType = dyn_cast<RankedTensorType>(loop.getInitArgs()[0].getType());
  plan.rowType = dyn_cast<RankedTensorType>(loop.getInitArgs()[1].getType());
  if (!plan.outputType || !plan.rowType || !plan.outputType.hasStaticShape() ||
      !plan.rowType.hasStaticShape() || plan.outputType.getRank() != 2 ||
      plan.rowType.getRank() != 1 ||
      !plan.outputType.getElementType().isF32() ||
      !plan.rowType.getElementType().isF32() ||
      plan.rowType.getDimSize(0) <= 0 ||
      plan.outputType.getDimSize(0) != plan.rowType.getDimSize(0) ||
      plan.outputType.getDimSize(1) <= 0)
    return reject("unsupported local accumulator/row geometry");
  for (Value value : loop.getInitArgs().drop_front())
    if (value.getType() != plan.rowType)
      return reject("row-state types disagree");

  for (Operation &op : loop.getBody()->without_terminator()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (!scope)
      continue;
    auto number = scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto depth = scope->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
    if (!number || !depth || depth.getInt() != 4)
      return reject("unproved stage schedule");
    if (number.getInt() != 2 && number.getInt() != 0)
      continue;
    auto &stage = number.getInt() == 2 ? plan.producer : plan.output;
    if (stage || scope->getAttr("hivm.loop_core_type") !=
                     TCoreTypeAttr::get(loop.getContext(), TCoreType::VECTOR))
      return reject("ambiguous or wrong-core state stage");
    stage = scope;
  }
  if (!plan.producer || !plan.output ||
      !plan.producer->isBeforeInBlock(plan.output) ||
      plan.producer.getNumResults() != 5 || plan.output.getNumResults() != 1)
    return reject("expected distinct P and O state stages");
  auto args = loop.getRegionIterArgs();
  for (unsigned i = 0; i < args.size(); ++i)
    for (Operation *user : args[i].getUsers())
      if (!in(user, i == 0 ? plan.output : plan.producer))
        return reject("loop state escapes its owning stage");
  auto pReturn =
      cast<scope::ReturnOp>(plan.producer.getRegion().front().getTerminator());
  auto evenSum = pReturn.getOperand(2).getDefiningOp<arith::SelectOp>();
  auto oddSum = pReturn.getOperand(3).getDefiningOp<arith::SelectOp>();
  auto evenMax = pReturn.getOperand(1).getDefiningOp<arith::SelectOp>();
  auto oddMax = pReturn.getOperand(4).getDefiningOp<arith::SelectOp>();
  if (!evenSum || !oddSum || !evenMax || !oddMax)
    return reject("missing paired state selectors");
  plan.odd = evenSum.getCondition();
  plan.sum = evenSum.getFalseValue();
  plan.maximum = evenMax.getFalseValue();
  if (!queryParity(plan.odd, loop, *keys) ||
      !selectIs(evenSum, plan.odd, args[1], plan.sum) ||
      !selectIs(oddSum, plan.odd, plan.sum, args[3]) ||
      !selectIs(evenMax, plan.odd, args[2], plan.maximum) ||
      !selectIs(oddMax, plan.odd, plan.maximum, args[4]))
    return reject("unproved active/pass-through query-bank selectors");
  const std::array<Operation *, 4> outgoing = {
      evenSum.getOperation(), evenMax.getOperation(), oddSum.getOperation(),
      oddMax.getOperation()};
  // Reads must select the active bank. The other appearances are the checked
  // pass-through selectors whose values disappear with the tensor recurrence.
  for (unsigned i = 1; i < 5; ++i)
    for (Operation *user : args[i].getUsers()) {
      if (user == outgoing[i - 1])
        continue;
      auto select = dyn_cast<arith::SelectOp>(user);
      bool sum = i == 1 || i == 3;
      if (!selectIs(select, plan.odd, args[sum ? 3 : 4], args[sum ? 1 : 2]))
        return reject("state read does not select the active query bank");
    }

  auto oReturn =
      cast<scope::ReturnOp>(plan.output.getRegion().front().getTerminator());
  plan.accumulator = oReturn.getOperand(0);
  if (!plan.accumulator.getDefiningOp() ||
      plan.accumulator.getDefiningOp()->getBlock() !=
          &plan.output.getRegion().front())
    return reject("O accumulator is not defined directly in its stage");
  for (Operation &op : plan.output.getRegion().front().without_terminator())
    if (auto branch = dyn_cast<scf::IfOp>(op))
      if (isLastKey(branch.getCondition(), loop, *keys, *step)) {
        if (plan.finalization || branch.getNumResults() || !branch.elseBlock())
          return reject("ambiguous or value-carrying finalization");
        plan.finalization = branch;
      }
  if (!plan.finalization ||
      !plan.accumulator.getDefiningOp()->isBeforeInBlock(plan.finalization))
    return reject("missing finalization after the O update");
  for (unsigned i = 1; i < 5; ++i)
    for (Operation *user : plan.producer.getResult(i).getUsers()) {
      if (user == loop.getBody()->getTerminator())
        continue;
      auto select = dyn_cast<arith::SelectOp>(user);
      bool sum = i == 2 || i == 3;
      if (!in(user, plan.finalization) ||
          !selectIs(select, plan.odd, plan.producer.getResult(sum ? 3 : 4),
                    plan.producer.getResult(sum ? 2 : 1)))
        return reject("row result escapes final active-bank selection");
    }
  for (Operation *user : plan.producer.getResult(0).getUsers())
    if (!in(user, plan.output))
      return reject("alpha escapes the O stage");
  for (Operation *user : plan.output.getResult(0).getUsers())
    if (user != loop.getBody()->getTerminator())
      return reject("O result escapes its recurrence");
  auto yielded = loop.getYieldedValues();
  if (yielded[0] != plan.output.getResult(0) ||
      yielded[1] != plan.producer.getResult(2) ||
      yielded[2] != plan.producer.getResult(1) ||
      yielded[3] != plan.producer.getResult(3) ||
      yielded[4] != plan.producer.getResult(4))
    return reject("unexpected state recurrence order");
  return plan;
}

static void materialize(OpBuilder &builder, Location loc, Value tensor,
                        Value destination, StringRef role = {}) {
  OperationState state(
      loc, bufferization::MaterializeInDestinationOp::getOperationName());
  state.addOperands({tensor, destination});
  state.addAttribute("writable", builder.getUnitAttr());
  if (!role.empty())
    state.addAttribute(kWrite, builder.getStringAttr(role));
  builder.create(state);
}

// Take a real stage-local snapshot. A fresh buffer has one restricted tensor
// boundary; the persistent bank itself is never exposed through multiple
// restrict to_tensor operations or hidden mutable tensor aliases.
static Value readBank(OpBuilder &builder, Location loc, Value bank,
                      RankedTensorType type, bool writable = false) {
  auto memory = cast<MemRefType>(bank.getType());
  auto snapshot = builder.create<memref::AllocOp>(loc, memory);
  builder.create<CopyOp>(loc, TypeRange{}, bank, snapshot.getResult());
  return builder.create<bufferization::ToTensorOp>(
      loc, type, snapshot.getResult(), builder.getUnitAttr(),
      writable ? builder.getUnitAttr() : UnitAttr());
}

static Value makeBank(OpBuilder &builder, Location loc, RankedTensorType type,
                      StringRef role, unsigned slot, unsigned count) {
  auto memory = MemRefType::get(
      type.getShape(), type.getElementType(), MemRefLayoutAttrInterface{},
      AddressSpaceAttr::get(builder.getContext(), AddressSpace::UB));
  auto bank = builder.create<memref::AllocOp>(loc, memory);
  bank->setAttr(
      kBank, builder.getDictionaryAttr(
                 {builder.getNamedAttr("role", builder.getStringAttr(role)),
                  builder.getNamedAttr("bank", builder.getI32IntegerAttr(slot)),
                  builder.getNamedAttr("required_slots",
                                       builder.getI32IntegerAttr(count)),
                  builder.getNamedAttr(
                      "clock", builder.getStringAttr(
                                   role == "accumulator" ? "O" : "query"))}));
  OperationState mark(loc, annotation::MarkOp::getOperationName());
  mark.addOperands(bank.getResult());
  mark.addAttribute(HIVMMemoryUniqueAttr::getMnemonic(), builder.getUnitAttr());
  builder.create(mark);
  return bank;
}

static Value selected(OpBuilder &builder, Location loc, Value odd,
                      Value evenBank, Value oddBank) {
  return builder.create<arith::SelectOp>(loc, odd, oddBank, evenBank);
}

static void rewrite(StatePlan &plan) {
  OpBuilder builder(plan.loop);
  Location loc = plan.loop.getLoc();
  Value accumulator =
      makeBank(builder, loc, plan.outputType, "accumulator", 0, 1);
  std::array<Value, 2> sums, maxima;
  for (unsigned i = 0; i < 2; ++i) {
    sums[i] = makeBank(builder, loc, plan.rowType, "sum", i, 2);
    maxima[i] = makeBank(builder, loc, plan.rowType, "maximum", i, 2);
  }
  // Initial writes are before the stream, never shared operations in its body.
  // The existing first-key reset remains in the cloned math for every query.
  materialize(builder, loc, plan.loop.getInitArgs()[0], accumulator);
  for (unsigned i = 0; i < 2; ++i) {
    materialize(builder, loc, plan.loop.getInitArgs()[1], sums[i]);
    materialize(builder, loc, plan.loop.getInitArgs()[2], maxima[i]);
  }
  auto loop = builder.create<scf::ForOp>(loc, plan.loop.getLowerBound(),
                                         plan.loop.getUpperBound(),
                                         plan.loop.getStep());
  loop->setAttrs(plan.loop->getAttrs());
  loop->setAttr(kLowered, builder.getUnitAttr());
  loop.getBody()->getTerminator()->erase();
  builder.setInsertionPointToStart(loop.getBody());
  IRMapping mapping;
  mapping.map(plan.loop.getInductionVar(), loop.getInductionVar());
  for (Operation &op : plan.loop.getBody()->without_terminator()) {
    if (&op != plan.producer.getOperation() &&
        &op != plan.output.getOperation()) {
      builder.clone(op, mapping);
      continue;
    }
    bool producer = &op == plan.producer.getOperation();
    auto old = cast<scope::ScopeOp>(op);
    SmallVector<Type> types;
    if (producer)
      types.push_back(old.getResult(0).getType());
    auto stage = builder.create<scope::ScopeOp>(loc, types);
    stage->setAttrs(old->getAttrs());
    Block *body = builder.createBlock(&stage.getRegion());
    builder.setInsertionPointToStart(body);
    IRMapping local(mapping);
    auto args = plan.loop.getRegionIterArgs();
    Value odd = mapping.lookupOrDefault(plan.odd);
    if (producer) {
      Value sumBank = selected(builder, loc, odd, sums[0], sums[1]);
      Value maxBank = selected(builder, loc, odd, maxima[0], maxima[1]);
      Value sum = readBank(builder, loc, sumBank, plan.rowType);
      Value maximum = readBank(builder, loc, maxBank, plan.rowType);
      local.map(args[1], sum);
      local.map(args[3], sum);
      local.map(args[2], maximum);
      local.map(args[4], maximum);
      for (Operation &inner : old.getRegion().front().without_terminator())
        builder.clone(inner, local);
      materialize(builder, loc, local.lookupOrDefault(plan.sum), sumBank,
                  "sum");
      materialize(builder, loc, local.lookupOrDefault(plan.maximum), maxBank,
                  "maximum");
      auto returned =
          cast<scope::ReturnOp>(old.getRegion().front().getTerminator());
      builder.create<scope::ReturnOp>(
          loc, ValueRange{local.lookupOrDefault(returned.getOperand(0))});
    } else {
      // The snapshot owns a full private copy. It may be reused in place by
      // bufferization; the persistent accumulator never has a tensor alias.
      local.map(args[0], readBank(builder, loc, accumulator, plan.outputType,
                                  /*writable=*/true));
      for (Operation &inner : old.getRegion().front().without_terminator()) {
        if (&inner != plan.finalization.getOperation()) {
          builder.clone(inner, local);
          if (&inner == plan.accumulator.getDefiningOp())
            materialize(builder, loc, local.lookupOrDefault(plan.accumulator),
                        accumulator, "accumulator");
          continue;
        }
        auto finish = builder.create<scf::IfOp>(
            loc, TypeRange{},
            local.lookupOrDefault(plan.finalization.getCondition()), true);
        finish->setAttrs(inner.getAttrs());
        finish.thenBlock()->getTerminator()->erase();
        finish.elseBlock()->getTerminator()->erase();
        builder.setInsertionPointToStart(finish.thenBlock());
        IRMapping tail(local);
        Value sum = readBank(builder, loc,
                             selected(builder, loc, odd, sums[0], sums[1]),
                             plan.rowType);
        Value maximum = readBank(
            builder, loc, selected(builder, loc, odd, maxima[0], maxima[1]),
            plan.rowType);
        for (unsigned i = 1; i < 5; ++i)
          tail.map(plan.producer.getResult(i),
                   i == 2 || i == 3 ? sum : maximum);
        for (Operation &final :
             plan.finalization.thenBlock()->without_terminator())
          builder.clone(final, tail);
        builder.create<scf::YieldOp>(loc);
        builder.setInsertionPointToStart(finish.elseBlock());
        for (Operation &empty :
             plan.finalization.elseBlock()->without_terminator())
          builder.clone(empty, local);
        builder.create<scf::YieldOp>(loc);
        builder.setInsertionPointAfter(finish);
      }
      builder.create<scope::ReturnOp>(loc);
    }
    builder.setInsertionPointAfter(stage);
    if (producer)
      mapping.map(old.getResult(0), stage.getResult(0));
  }
  builder.create<scf::YieldOp>(loc);
  plan.loop.erase();
}

static bool verifyLowered(scf::ForOp loop) {
  if (!loop.getInitArgs().empty() || !loop->use_empty())
    return false;
  unsigned sums = 0, maxima = 0, accumulators = 0;
  bool valid = true;
  loop.walk([&](bufferization::MaterializeInDestinationOp op) {
    auto role = op->getAttrOfType<StringAttr>(kWrite);
    if (!role)
      return;
    auto stage = op->getParentOfType<scope::ScopeOp>();
    auto phase = stage ? stage->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                       : IntegerAttr();
    if (!stage || stage->getParentOp() != loop ||
        op->getBlock() != &stage.getRegion().front() || !phase) {
      valid = false;
      return;
    }
    if (role.getValue() == "sum") {
      ++sums;
      valid &= phase.getInt() == 2;
    } else if (role.getValue() == "maximum") {
      ++maxima;
      valid &= phase.getInt() == 2;
    } else if (role.getValue() == "accumulator") {
      ++accumulators;
      valid &= phase.getInt() == 0;
    } else {
      valid = false;
    }
  });
  return valid && sums == 1 && maxima == 1 && accumulators == 1;
}

struct CVPQueryStateStoragePass
    : impl::CVPQueryStateStorageBase<CVPQueryStateStoragePass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    SmallVector<scf::ForOp> loops;
    copy->walk([&](scf::ForOp loop) {
      auto func = loop->getParentOfType<func::FuncOp>();
      if (func && func->hasAttr("hacc.entry") && func->hasAttr(kStream) &&
          func->getAttr("hivm.func_core_type") ==
              TFuncCoreTypeAttr::get(loop.getContext(), TFuncCoreType::AIV) &&
          loop->hasAttr(kStream))
        loops.push_back(loop);
    });
    if (loops.empty()) {
      if (requireMatch) {
        original.emitError(
            "CVP state storage requires a partitioned AIV query stream");
        signalPassFailure();
      }
      return;
    }
    for (scf::ForOp loop : loops) {
      if (loop->hasAttr(kLowered)) {
        if (verifyLowered(loop))
          continue;
        original.emitError(
            "invalid previously lowered CVP query state storage");
        return signalPassFailure();
      }
      std::string reason;
      auto plan = analyze(loop, reason);
      if (failed(plan)) {
        if (requireMatch) {
          original.emitError("CVP state storage: ") << reason;
          signalPassFailure();
        }
        return;
      }
      rewrite(*plan);
    }
    bool valid = succeeded(verify(*copy));
    copy->walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kLowered))
        valid &= verifyLowered(loop);
    });
    if (!valid) {
      original.emitError("CVP query state-storage rewrite failed verification");
      return signalPassFailure();
    }
    original.getBodyRegion().takeBody(copy->getBodyRegion());
    original->setAttrs((*copy)->getAttrs());
  }
};
} // namespace

std::unique_ptr<Pass> mlir::hivm::createCVPQueryStateStoragePass(
    const CVPQueryStateStorageOptions &options) {
  return std::make_unique<CVPQueryStateStoragePass>(options);
}

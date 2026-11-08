//===- CVPQueryPipeline.cpp - Continuous logical query scheduling
//-----------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Transforms/regbase/CVPQuerySchedule.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include <array>
#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CVPQUERYPIPELINE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr StringLiteral streamAttr = "hivm.cvp_query.stream";
constexpr StringLiteral storageAttr = "hivm.cvp_query.storage";

struct QueryPlan {
  scf::ForOp query, key;
  std::array<scope::ScopeOp, 4> stages;
  std::array<unsigned, 3> stateIndices;
  std::array<OpResult, 3> stateResults;
  std::array<VBrcOp, 3> initializers;
  std::array<arith::AddIOp, 2> counters;
  ND2NZOp qLoad;
  memref::AllocOp qAlloc;
  bufferization::ToTensorOp qTensor;
  MmadL1Op qk, pv;
  Operation *outputDivide = nullptr, *outputCast = nullptr;
  Operation *mValue = nullptr;
  Value outputEmpty, rowDestination, outputDestination;
  SmallVector<Operation *> prefix, suffix;
  std::optional<CVPQuerySchedule> schedule;
  unsigned keyUnroll = 1;
  bool ordinaryUnroll = false;
  DictionaryAttr geometry;
};

static std::optional<int64_t> integer(DictionaryAttr dict, StringRef key) {
  auto value = dict.getAs<IntegerAttr>(key);
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

static Value input(Operation *op, unsigned index) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  return dps && index < dps.getNumDpsInputs()
             ? dps.getDpsInputOperand(index)->get()
             : Value();
}

static Value destination(Operation *op) {
  auto dps = dyn_cast<DestinationStyleOpInterface>(op);
  return dps && dps.getNumDpsInits() == 1 ? dps.getDpsInitOperand(0)->get()
                                          : Value();
}

static bool usedOnlyBy(Value value, Operation *user) {
  return llvm::all_of(value.getUsers(),
                      [&](Operation *op) { return op == user; });
}

// Match actual preload scopes and their SSA recurrence. Printed names, physical
// offsets, event numbers and a fixed device/core count are not inputs here.
static FailureOr<QueryPlan> analyze(scf::ForOp key, DictionaryAttr geometry,
                                    std::string &reason) {
  auto reject = [&](StringRef text) -> FailureOr<QueryPlan> {
    reason = text.str();
    return failure();
  };
  QueryPlan plan;
  plan.key = key;
  plan.query = dyn_cast<scf::ForOp>(key->getParentOp());
  plan.geometry = geometry;
  if (!plan.query || !plan.query.getInitArgs().empty() ||
      key.getInitArgs().size() != 5 ||
      !key.getInductionVar().getType().isInteger(32) ||
      !plan.query.getInductionVar().getType().isInteger(32) ||
      key->hasAttr("unsignedCmp") || plan.query->hasAttr("unsignedCmp"))
    return reject("expected the supported persistent i32 query/key loop nest");
  auto lower = getConstantIntValue(key.getLowerBound());
  auto count = getConstantIntValue(key.getUpperBound());
  auto step = getConstantIntValue(key.getStep());
  auto qUpper = getConstantIntValue(plan.query.getUpperBound());
  auto qStep = getConstantIntValue(plan.query.getStep());
  if (!lower || *lower != 0 || !count || !step || !qUpper || !qStep ||
      integer(geometry, "key_count") != count ||
      integer(geometry, "block_n") != step)
    return reject("unproved static key domain or persistent query stride");
  plan.schedule =
      CVPQuerySchedule::create(*count, *step, *qUpper, *qStep, reason);
  if (!plan.schedule)
    return failure();
  if (integer(geometry, "key_tiles") != plan.schedule->getKeyTiles())
    return reject(
        "verified key-tile count disagrees with the logical key domain");
  if (auto mode = geometry.getAs<StringAttr>("pipeline_mode")) {
    if (mode.getValue() != "unroll" && mode.getValue() != "skew")
      return reject("unknown verified query emitter");
    plan.ordinaryUnroll = mode.getValue() == "unroll";
  }
  if (plan.ordinaryUnroll) {
    auto factor = integer(geometry, "key_unroll");
    auto pending =
        key->getAttrOfType<IntegerAttr>("hivm.cvp_query.unroll_factor");
    if (!factor || !pending || *factor != pending.getInt() ||
        (*factor != 1 && *factor != 2 && *factor != 4) ||
        !plan.schedule->canEmitQueryUnroll(*factor))
      return reject(
          "unproved continuous Unroll lane factor or arithmetic bound");
    plan.keyUnroll = *factor;
  } else if (key->hasAttr("hivm.cvp_query.unroll_factor")) {
    return reject("Unroll stage provenance disagrees with verified query mode");
  }

  for (Operation &op : key.getBody()->without_terminator()) {
    auto scope = dyn_cast<scope::ScopeOp>(op);
    if (!scope)
      continue;
    if (plan.ordinaryUnroll && scope->getAttr("hivm.cvp_query.unroll_factor") !=
                                   key->getAttr("hivm.cvp_query.unroll_factor"))
      return reject("continuous Unroll stage lane factors disagree");
    auto index = scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto maximum = scope->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
    if (!index || !maximum || maximum.getInt() != 4 || index.getInt() < 0 ||
        index.getInt() >= 4 || plan.stages[3 - index.getInt()])
      return reject("expected four uniquely indexed CVPipeline preload scopes");
    plan.stages[3 - index.getInt()] = scope;
  }
  if (llvm::any_of(plan.stages, [](auto scope) { return !scope; }))
    return reject("missing CVPipeline stage");
  unsigned scopePosition = 0;
  for (Operation &op : key.getBody()->without_terminator())
    if (auto scope = dyn_cast<scope::ScopeOp>(op))
      if (scope != plan.stages[scopePosition++])
        return reject(
            "CVPipeline stages are not in their logical dependency order");
  for (unsigned i = 0; i < 4; ++i) {
    auto core =
        plan.stages[i]->getAttrOfType<TCoreTypeAttr>("hivm.loop_core_type");
    if (!core ||
        core.getTcoretype() != (i % 2 ? TCoreType::VECTOR : TCoreType::CUBE))
      return reject("unexpected Cube/Vector stage order");
  }
  for (auto [i, scope] : llvm::enumerate(plan.stages)) {
    SmallVector<MmadL1Op> matrices;
    scope.walk([&](MmadL1Op op) { matrices.push_back(op); });
    if ((i % 2 == 0 && matrices.size() != 1) ||
        (i % 2 != 0 && !matrices.empty()))
      return reject("unproved matrix stage boundaries");
    if (i == 0)
      plan.qk = matrices.front();
    if (i == 2)
      plan.pv = matrices.front();
  }

  auto yield = key.getBody()->getTerminator();
  const StringRef stateNames[] = {"output_state", "sum_state", "max_state"};
  for (auto [i, name] : llvm::enumerate(stateNames)) {
    auto index = integer(geometry, name);
    if (!index || *index < 0 || *index >= 3)
      return reject("invalid attention state binding");
    plan.stateIndices[i] = *index;
    auto value = dyn_cast<OpResult>(yield->getOperand(*index));
    auto expectedScope = i == 0 ? plan.stages[3] : plan.stages[1];
    if (!value || value.getOwner() != expectedScope)
      return reject("state yield does not come from the expected stage");
    plan.stateResults[i] = value;
    auto init = key.getInitArgs()[*index].getDefiningOp<VBrcOp>();
    if (!init || init->getNumResults() != 1 ||
        !isa<RankedTensorType>(value.getType()) || plan.query->isAncestor(init))
      return reject("state initializer is not a complete scalar broadcast");
    plan.initializers[i] = init;
  }
  if (plan.stateIndices != std::array<unsigned, 3>{0, 1, 2})
    return reject("unsupported state permutation");
  if (plan.stages[3].getNumResults() != 1 ||
      plan.stateResults[0].getResultNumber() != 0 ||
      !usedOnlyBy(plan.stages[3].getResult(0), yield) ||
      !key.getResult(3).use_empty() || !key.getResult(4).use_empty())
    return reject("private counter/output results escape their recurrence");
  // The carried O value after a query is dead: the next query resets O before
  // running this stage. Require every read of that state to be inside the stage
  // where rewrite() inserts the first-local-key reset; an external read would
  // invalidate dropping the normalized value from the recurrence.
  for (Operation *user : key.getRegionIterArgs()[0].getUsers())
    if (!plan.stages[3]->isAncestor(user))
      return reject("O state escapes its first-key reset stage");
  auto outputType = dyn_cast<RankedTensorType>(key.getInitArgs()[0].getType());
  auto rowType = dyn_cast<RankedTensorType>(key.getInitArgs()[1].getType());
  if (!outputType || !rowType || outputType.getRank() != 2 ||
      rowType.getRank() != 1 || !outputType.hasStaticShape() ||
      !rowType.hasStaticShape() || !outputType.getElementType().isF32() ||
      !rowType.getElementType().isF32() ||
      key.getInitArgs()[2].getType() != rowType ||
      integer(geometry, "block_m") != outputType.getDimSize(0) ||
      integer(geometry, "head_dim") != outputType.getDimSize(1) ||
      rowType.getDimSize(0) != outputType.getDimSize(0))
    return reject("state types do not match verified attention geometry");
  for (unsigned i = 0; i < 2; ++i) {
    unsigned arg = 3 + i;
    auto scope = plan.stages[i * 2];
    if (scope.getNumResults() != 1 ||
        yield->getOperand(arg) != scope.getResult(0) ||
        !usedOnlyBy(scope.getResult(0), yield) ||
        getConstantIntValue(key.getInitArgs()[arg]) != 0)
      return reject("unproved private Cube key counter");
    auto ret = scope.getRegion().front().getTerminator();
    auto add = ret->getOperand(0).getDefiningOp<arith::AddIOp>();
    auto counter = key.getRegionIterArgs()[arg];
    if (!add || !usedOnlyBy(add.getResult(), ret) ||
        !((add.getLhs() == counter &&
           getConstantIntValue(add.getRhs()) == step) ||
          (add.getRhs() == counter &&
           getConstantIntValue(add.getLhs()) == step)))
      return reject("Cube counter is not the logical key induction");
    for (Operation *user : counter.getUsers())
      if (!scope->isAncestor(user))
        return reject("Cube key counter escapes its stage");
    plan.counters[i] = add;
  }

  bool before = true;
  for (Operation &op : plan.query.getBody()->without_terminator()) {
    if (&op == key.getOperation()) {
      before = false;
      continue;
    }
    (before ? plan.prefix : plan.suffix).push_back(&op);
    if (before && isa<ND2NZOp>(op)) {
      if (plan.qLoad)
        return reject("multiple per-query Cube loads");
      plan.qLoad = cast<ND2NZOp>(op);
    }
  }
  if (!plan.qLoad || plan.qLoad->getNumResults() != 0 ||
      plan.qLoad->getNumOperands() != 2)
    return reject("unsupported per-query Q load");
  plan.qAlloc = plan.qLoad.getDst().getDefiningOp<memref::AllocOp>();
  if (!plan.qAlloc || !plan.qAlloc.getType().hasStaticShape())
    return reject("Q storage must have a static descriptor");
  auto sourceType = dyn_cast<MemRefType>(plan.qLoad.getSrc().getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      sourceType.getRank() != 2 ||
      sourceType.getDimSize(0) != outputType.getDimSize(0) ||
      sourceType.getDimSize(1) != outputType.getDimSize(1) ||
      sourceType.getElementType() != plan.qAlloc.getType().getElementType())
    return reject("Q source must match the static query/head geometry");
  for (Operation *user : plan.qAlloc.getResult().getUsers()) {
    if (user == plan.qLoad)
      continue;
    auto tensor = dyn_cast<bufferization::ToTensorOp>(user);
    if (!tensor || plan.qTensor)
      return reject("Q storage has an escaping alias/use");
    plan.qTensor = tensor;
  }
  if (!plan.qTensor || (plan.qk.getA() != plan.qTensor.getResult() &&
                        plan.qk.getB() != plan.qTensor.getResult()))
    return reject("Q load is not the invariant QK operand");
  if (!usedOnlyBy(plan.qTensor.getResult(), plan.qk.getOperation()))
    return reject("Q tensor must be a private read of the QK matrix operation");
  for (Operation *op : plan.prefix)
    if (op != plan.qLoad && op != plan.qAlloc && op != plan.qTensor &&
        (!isMemoryEffectFree(op) || op->getNumRegions()))
      return reject("unsupported effect in query address preparation");

  SmallVector<StoreOp> stores;
  for (Operation *op : plan.suffix) {
    if (auto store = dyn_cast<StoreOp>(op))
      stores.push_back(store);
    else {
      StringRef name = op->getName().getStringRef();
      bool finalMath =
          name == "hivm.hir.vln" || isa<VAddOp, VDivOp, VCastOp>(op);
      if (!finalMath && (!isMemoryEffectFree(op) ||
                         op->getDialect()->getNamespace() == "hivm"))
        return reject("unsupported effect in query finalization");
    }
    if (op->getNumRegions())
      return reject("nested query finalization is not supported");
  }
  if (stores.size() != 2)
    return reject("expected complete M and O stores");
  for (auto store : stores) {
    auto type = dyn_cast<RankedTensorType>(store.getSrc().getType());
    if (!type)
      return reject("expected tensor finalization before bufferization");
    if (type.getRank() == 2 && type.getElementType().isF16()) {
      plan.outputCast = store.getSrc().getDefiningOp();
      if (!isa_and_nonnull<VCastOp>(plan.outputCast))
        return reject("final O is not an FP16 conversion");
      plan.outputDivide = input(plan.outputCast, 0).getDefiningOp();
      if (!isa_and_nonnull<VDivOp>(plan.outputDivide) ||
          input(plan.outputDivide, 0) != key.getResult(plan.stateIndices[0]))
        return reject("final normalization does not consume the last O");
      if (plan.outputDivide->getNumResults() != 1 ||
          !usedOnlyBy(plan.outputDivide->getResult(0), plan.outputCast))
        return reject("normalized O escapes the final output conversion");
      auto denominator =
          input(plan.outputDivide, 1).getDefiningOp<tensor::ExpandShapeOp>();
      if (!denominator ||
          denominator.getSrc() != key.getResult(plan.stateIndices[1]) ||
          type.getShape() != outputType.getShape())
        return reject("final normalization does not use the final row sum");
      plan.outputEmpty = destination(plan.outputCast);
      plan.outputDestination = destination(plan.outputDivide);
      if (!plan.outputEmpty.getDefiningOp<tensor::EmptyOp>())
        return reject("final output storage escapes its conversion");
    } else if (type.getRank() == 1 && type.getElementType().isF32()) {
      plan.mValue = store.getSrc().getDefiningOp();
      if (!isa_and_nonnull<VAddOp>(plan.mValue) || !destination(plan.mValue))
        return reject("unproved M finalization");
      Value lhs = input(plan.mValue, 0), rhs = input(plan.mValue, 1);
      Value logarithm = lhs == key.getResult(2)   ? rhs
                        : rhs == key.getResult(2) ? lhs
                                                  : Value();
      auto *log = logarithm ? logarithm.getDefiningOp() : nullptr;
      if (!log || log->getName().getStringRef() != "hivm.hir.vln" ||
          input(log, 0) != key.getResult(1) || type != rowType)
        return reject("M is not the original log(sum)+maximum value");
      plan.rowDestination = destination(plan.mValue);
    } else
      return reject("unsupported final store geometry");
  }
  if (!plan.outputCast || !plan.mValue)
    return reject("missing distinct M/O finalization");
  return plan;
}

static Value emptyBank(OpBuilder &b, Location loc, RankedTensorType type,
                       StringRef role, unsigned bank) {
  auto op =
      b.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType());
  op->setAttr(storageAttr,
              b.getDictionaryAttr(
                  {b.getNamedAttr("role", b.getStringAttr(role)),
                   b.getNamedAttr("bank", b.getI32IntegerAttr(bank)),
                   b.getNamedAttr("required_slots", b.getI32IntegerAttr(2)),
                   b.getNamedAttr("clock", b.getStringAttr("query"))}));
  return op;
}

static Value choose(OpBuilder &b, Location loc, Value odd, Value evenValue,
                    Value oddValue) {
  return b.create<arith::SelectOp>(loc, odd, oddValue, evenValue);
}

static Value initialize(OpBuilder &b, VBrcOp original, Value destination) {
  IRMapping mapping;
  mapping.map(::destination(original), destination);
  return b.clone(*original.getOperation(), mapping)->getResult(0);
}

static void clonePrefix(QueryPlan &plan, OpBuilder &b, IRMapping &map,
                        Value queryTask) {
  map.map(plan.query.getInductionVar(), queryTask);
  for (Operation *op : plan.prefix)
    if (op != plan.qAlloc && op != plan.qLoad && op != plan.qTensor)
      b.clone(*op, map);
}

static void eraseYield(Block *block) {
  if (!block->empty() && isa<scf::YieldOp>(block->back()))
    block->back().erase();
}

static void tagCoordinate(OpBuilder &b, Value value, StringRef kind) {
  value.getDefiningOp()->setAttr("hivm.cvp_query.coordinate",
                                 b.getStringAttr(kind));
}

// Transform in a cloned module. The caller commits only after verification.
static LogicalResult rewrite(QueryPlan &plan) {
  const CVPQuerySchedule &schedule = *plan.schedule;
  OpBuilder b(plan.query);
  Location loc = plan.query.getLoc();
  auto c64 = [&](int64_t value) -> Value {
    return b.create<arith::ConstantIntOp>(loc, value, 64);
  };
  Value zero = c64(0), one = c64(1), two = c64(2);
  Value keyCount = c64(schedule.getKeyCount());
  Value keyStep = c64(schedule.getKeyStep());
  Value queryStep = c64(schedule.getQueryStep());
  Value queryUpper = c64(schedule.getQueryUpper());
  Value queryStart =
      b.create<arith::ExtSIOp>(loc, b.getI64Type(), plan.query.getLowerBound());
  Value remaining = b.create<arith::SubIOp>(loc, queryUpper, queryStart);
  Value positive = b.create<arith::MaxSIOp>(loc, remaining, zero);
  Value queryCount = b.create<arith::CeilDivSIOp>(loc, positive, queryStep);
  Value workEnd = b.create<arith::MulIOp>(loc, queryCount, keyCount);
  tagCoordinate(b, queryStart, "query_start");
  tagCoordinate(b, queryCount, "query_count");
  tagCoordinate(b, workEnd, "work_end");
  auto rowType = cast<RankedTensorType>(plan.key.getInitArgs()[1].getType());
  auto qType = cast<RankedTensorType>(plan.qTensor.getResult().getType());
  auto outType =
      cast<RankedTensorType>(plan.outputCast->getResult(0).getType());
  SmallVector<Value> init(plan.key.getInitArgs().take_front(3));
  init.push_back(emptyBank(b, loc, rowType, "sum", 1));
  init.push_back(emptyBank(b, loc, rowType, "maximum", 1));
  // Q has persistent buffer identity across key ticks. Carrying snapshots as
  // tensor loop/scope results makes generic copy insertion generate L1-to-L1
  // transfers, which have no DMA pipe. Keep two allocator-managed query banks;
  // the QK stage owns their real load/read effects and no Q view escapes it.
  auto originalQType = plan.qAlloc.getType();
  auto qBufferType =
      MemRefType::get(originalQType.getShape(), originalQType.getElementType(),
                      originalQType.getLayout(),
                      AddressSpaceAttr::get(b.getContext(), AddressSpace::L1));
  std::array<Value, 2> qBanks;
  for (unsigned i = 0; i < 2; ++i) {
    auto allocation = b.create<memref::AllocOp>(loc, qBufferType);
    allocation->setAttrs(plan.qAlloc->getAttrs());
    allocation->setAttr(
        storageAttr,
        b.getDictionaryAttr(
            {b.getNamedAttr("role", b.getStringAttr("Q")),
             b.getNamedAttr("bank", b.getI32IntegerAttr(i)),
             b.getNamedAttr("required_slots", b.getI32IntegerAttr(2)),
             b.getNamedAttr("clock", b.getStringAttr("query"))}));
    qBanks[i] = allocation;
  }
  for (unsigned i = 0; i < 2; ++i)
    init.push_back(emptyBank(b, loc, outType, "output", i));
  for (unsigned i = 0; i < 2; ++i)
    init.push_back(emptyBank(b, loc, rowType, "M", i));
  auto stream = b.create<scf::ForOp>(loc, zero, workEnd, keyStep, init);
  stream->setAttrs(plan.key->getAttrs());
  stream->setAttr(
      streamAttr,
      b.getDictionaryAttr(
          {b.getNamedAttr("key_count",
                          b.getI64IntegerAttr(schedule.getKeyCount())),
           b.getNamedAttr("key_step",
                          b.getI64IntegerAttr(schedule.getKeyStep())),
           b.getNamedAttr("query_step",
                          b.getI64IntegerAttr(schedule.getQueryStep())),
           b.getNamedAttr(
               "pipeline_mode",
               b.getStringAttr(plan.ordinaryUnroll ? "unroll" : "skew")),
           b.getNamedAttr("key_unroll", b.getI32IntegerAttr(plan.keyUnroll)),
           b.getNamedAttr("query_slots",
                          b.getI32IntegerAttr(CVPQuerySchedule::kQuerySlots)),
           b.getNamedAttr("physical_lifetimes_verified",
                          b.getBoolAttr(false))}));
  if (plan.ordinaryUnroll) {
    NamedAttrList streamPlan(stream->getAttrOfType<DictionaryAttr>(streamAttr));
    streamPlan.set("query_upper",
                   b.getI64IntegerAttr(schedule.getQueryUpper()));
    streamPlan.set("ownership_required", b.getBoolAttr(true));
    stream->setAttr(streamAttr, streamPlan.getDictionary(b.getContext()));
  }
  eraseYield(stream.getBody());
  b.setInsertionPointToStart(stream.getBody());
  auto args = stream.getRegionIterArgs();
  Value ordinal =
      b.create<arith::DivSIOp>(loc, stream.getInductionVar(), keyCount);
  Value local64 =
      b.create<arith::RemSIOp>(loc, stream.getInductionVar(), keyCount);
  Value local = b.create<arith::TruncIOp>(loc, b.getI32Type(), local64);
  Value taskOffset = b.create<arith::MulIOp>(loc, ordinal, queryStep);
  Value task64 = b.create<arith::AddIOp>(loc, queryStart, taskOffset);
  Value task = b.create<arith::TruncIOp>(loc, b.getI32Type(), task64);
  Value parity = b.create<arith::RemSIOp>(loc, ordinal, two);
  Value odd =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, parity, one);
  Value first =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, local64, zero);
  Value lastIndex = c64(schedule.getLastKey());
  Value last = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, local64,
                                       lastIndex);
  tagCoordinate(b, ordinal, "query_ordinal");
  tagCoordinate(b, local, "local_key");
  tagCoordinate(b, task, "query_task");
  tagCoordinate(b, parity, "query_slot");
  tagCoordinate(b, first, "first_key");
  tagCoordinate(b, last, "last_key");

  IRMapping global;
  global.map(plan.key.getInductionVar(), local);
  for (unsigned i = 0; i < 3; ++i)
    global.map(plan.key.getRegionIterArgs()[i], args[i]);
  for (unsigned i = 3; i < 5; ++i)
    global.map(plan.key.getRegionIterArgs()[i], local);
  SmallVector<Value> next(args.begin(), args.end());
  for (Operation &bodyOp : plan.key.getBody()->without_terminator()) {
    auto old = dyn_cast<scope::ScopeOp>(bodyOp);
    if (!old) {
      b.clone(bodyOp, global);
      continue;
    }
    unsigned stage = llvm::find(plan.stages, old) - plan.stages.begin();
    SmallVector<Type> resultTypes;
    if (stage == 1) {
      resultTypes.append(old.getResultTypes().begin(),
                         old.getResultTypes().end());
      resultTypes.push_back(rowType);
      resultTypes.push_back(rowType);
    } else if (stage == 3) {
      resultTypes = {args[0].getType(), outType, outType, rowType, rowType};
    }
    auto scope = b.create<scope::ScopeOp>(loc, resultTypes);
    scope->setAttrs(old->getAttrs());
    Block *body = b.createBlock(&scope.getRegion());
    b.setInsertionPointToStart(body);
    IRMapping map(global);
    clonePrefix(plan, b, map, task);
    Value currentQ, currentSum, currentMax;
    if (stage == 0) {
      Value slot = choose(b, loc, odd, qBanks[0], qBanks[1]);
      auto load = b.create<scf::IfOp>(loc, TypeRange{}, first, false);
      eraseYield(&load.getThenRegion().front());
      b.setInsertionPointToStart(&load.getThenRegion().front());
      auto qLoad = b.create<ND2NZOp>(loc, TypeRange{},
                                     map.lookupOrDefault(plan.qLoad.getSrc()),
                                     slot, plan.qLoad.getDstContinuousAttr());
      qLoad->setAttrs(plan.qLoad->getAttrs());
      b.create<scf::YieldOp>(loc);
      b.setInsertionPointAfter(load);
      // The view is read-only and local to this QK stage. The original
      // memref-form transfer writes the selected bank on its first key only.
      currentQ = b.create<bufferization::ToTensorOp>(
          loc, qType, slot, b.getUnitAttr(), UnitAttr());
      map.map(plan.qTensor.getResult(), currentQ);
    } else if (stage == 1) {
      currentSum = choose(b, loc, odd, args[1], args[3]);
      currentMax = choose(b, loc, odd, args[2], args[4]);
      auto reset =
          b.create<scf::IfOp>(loc, TypeRange{rowType, rowType}, first, true);
      b.setInsertionPointToStart(&reset.getThenRegion().front());
      Value sum = initialize(b, plan.initializers[1], currentSum);
      Value maximum = initialize(b, plan.initializers[2], currentMax);
      b.create<scf::YieldOp>(loc, ValueRange{sum, maximum});
      b.setInsertionPointToStart(&reset.getElseRegion().front());
      b.create<scf::YieldOp>(loc, ValueRange{currentSum, currentMax});
      b.setInsertionPointAfter(reset);
      map.map(plan.key.getRegionIterArgs()[1], reset.getResult(0));
      map.map(plan.key.getRegionIterArgs()[2], reset.getResult(1));
    } else if (stage == 3) {
      auto reset =
          b.create<scf::IfOp>(loc, TypeRange{args[0].getType()}, first, true);
      b.setInsertionPointToStart(&reset.getThenRegion().front());
      b.create<scf::YieldOp>(loc, initialize(b, plan.initializers[0], args[0]));
      b.setInsertionPointToStart(&reset.getElseRegion().front());
      b.create<scf::YieldOp>(loc, args[0]);
      b.setInsertionPointAfter(reset);
      map.map(plan.key.getRegionIterArgs()[0], reset.getResult(0));
      map.map(destination(plan.initializers[0]), reset.getResult(0));
    }
    for (Operation &op : old.getRegion().front().without_terminator()) {
      if (&op == plan.counters[0].getOperation() ||
          &op == plan.counters[1].getOperation())
        continue;
      b.clone(op, map);
    }
    SmallVector<Value> results;
    auto oldReturn = old.getRegion().front().getTerminator();
    if (stage == 1) {
      for (Value value : oldReturn->getOperands())
        results.push_back(map.lookupOrDefault(value));
      unsigned sumIndex = plan.stateResults[1].getResultNumber();
      unsigned maxIndex = plan.stateResults[2].getResultNumber();
      Value sum = results[sumIndex], maximum = results[maxIndex];
      results[sumIndex] = choose(b, loc, odd, sum, args[1]);
      results[maxIndex] = choose(b, loc, odd, maximum, args[2]);
      results.push_back(choose(b, loc, odd, args[3], sum));
      results.push_back(choose(b, loc, odd, args[4], maximum));
    } else if (stage == 3) {
      Value output = map.lookupOrDefault(oldReturn->getOperand(0));
      Value outSlot = choose(b, loc, odd, args[5], args[6]);
      Value mSlot = choose(b, loc, odd, args[7], args[8]);
      auto finish =
          b.create<scf::IfOp>(loc, TypeRange{outType, rowType}, last, true);
      b.setInsertionPointToStart(&finish.getThenRegion().front());
      IRMapping tail(map);
      tail.map(plan.key.getResult(0), output);
      tail.map(plan.key.getResult(1), choose(b, loc, odd, next[1], next[3]));
      tail.map(plan.key.getResult(2), choose(b, loc, odd, next[2], next[4]));
      tail.map(plan.outputEmpty, outSlot);
      tail.map(plan.rowDestination, mSlot);
      tail.map(plan.outputDestination, output);
      for (Operation *op : plan.suffix)
        if (op != plan.outputEmpty.getDefiningOp())
          b.clone(*op, tail);
      b.create<scf::YieldOp>(
          loc, ValueRange{tail.lookup(plan.outputCast->getResult(0)),
                          tail.lookup(plan.mValue->getResult(0))});
      b.setInsertionPointToStart(&finish.getElseRegion().front());
      b.create<scf::YieldOp>(loc, ValueRange{outSlot, mSlot});
      b.setInsertionPointAfter(finish);
      // Keep the ordinary accumulator pointer/value in the recurrence. The
      // normalized FP32 value is consumed only by the final FP16 conversion;
      // exposing it as loop state forces a redundant VF output and copies.
      // The next valid O stage is the next query's first key and resets O, as
      // proved above. Final normalization, M and both GM stores are unchanged.
      results = {output, choose(b, loc, odd, finish.getResult(0), args[5]),
                 choose(b, loc, odd, args[6], finish.getResult(0)),
                 choose(b, loc, odd, finish.getResult(1), args[7]),
                 choose(b, loc, odd, args[8], finish.getResult(1))};
    }
    b.create<scope::ReturnOp>(loc, results);
    b.setInsertionPointAfter(scope);
    if (stage == 1) {
      for (unsigned i = 0; i < old.getNumResults(); ++i)
        global.map(old.getResult(i), scope.getResult(i));
      next[1] = scope.getResult(plan.stateResults[1].getResultNumber());
      next[2] = scope.getResult(plan.stateResults[2].getResultNumber());
      next[3] = scope.getResult(old.getNumResults());
      next[4] = scope.getResult(old.getNumResults() + 1);
    } else if (stage == 3) {
      next[0] = scope.getResult(0);
      for (unsigned i = 0; i < 4; ++i)
        next[5 + i] = scope.getResult(1 + i);
    }
  }
  b.create<scf::YieldOp>(loc, next);
  plan.query.erase();
  return success();
}

struct CVPQueryPipelinePass : impl::CVPQueryPipelineBase<CVPQueryPipelinePass> {
  explicit CVPQueryPipelinePass(const CVPQueryPipelineOptions &options)
      : CVPQueryPipelineBase(options) {}
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // A later function may reject an explicit request. Keep the entire module
    // unchanged in that case, including earlier functions that did match.
    OwningOpRef<ModuleOp> candidate(cast<ModuleOp>(module->clone()));
    SmallVector<func::FuncOp> functions;
    for (func::FuncOp func : candidate->getOps<func::FuncOp>()) {
      bool ordinaryUnroll = false;
      func.walk([&](scf::ForOp loop) {
        ordinaryUnroll |= loop->hasAttr("hivm.cvp_query.unroll_domain");
      });
      if (func->hasAttr(kCVPAttentionVerifiedAttr) || ordinaryUnroll)
        functions.push_back(func);
    }
    unsigned transformed = 0;
    for (func::FuncOp func : functions) {
      if (func->hasAttr(streamAttr)) {
        ++transformed;
        continue;
      }
      bool ordinaryUnroll = false;
      func.walk([&](scf::ForOp loop) {
        ordinaryUnroll |= loop->hasAttr("hivm.cvp_query.unroll_domain");
      });
      if (ordinaryUnroll) {
        const char *reason =
            "ordinary Unroll emitted per-query keys; continuous "
            "query state and cross-stage forwarding are not implemented "
            "for already-emitted per-query loops; prepare "
            "continuous-query=true first";
        if (requireMatch) {
          func.emitError("query pipeline: ") << reason;
          return signalPassFailure();
        }
        func.emitRemark("query pipeline retained original schedule: ")
            << reason;
        continue;
      }
      auto summaries =
          func->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
      std::string reason = "expected one verified attention recurrence";
      if (!summaries || summaries.size() != 1 ||
          !isa<DictionaryAttr>(summaries[0])) {
        if (requireMatch) {
          func.emitError(reason);
          return signalPassFailure();
        }
        continue;
      }
      SmallVector<scf::ForOp> loops;
      func.walk([&](scf::ForOp loop) {
        if (llvm::any_of(loop.getBody()->getOperations(), [](Operation &op) {
              return isa<scope::ScopeOp>(op) &&
                     op.hasAttr(PreloadNumAttr::name);
            }))
          loops.push_back(loop);
      });
      FailureOr<QueryPlan> plan = failure();
      if (loops.size() == 1)
        plan =
            analyze(loops.front(), cast<DictionaryAttr>(summaries[0]), reason);
      else
        reason =
            "expected one Skew preload loop; ordinary Unroll query emission "
            "is not implemented";
      if (failed(plan)) {
        if (requireMatch) {
          func.emitError("query pipeline: ") << reason;
          return signalPassFailure();
        }
        func.emitRemark("query pipeline retained original schedule: ")
            << reason;
        continue;
      }
      if (failed(rewrite(*plan)) || failed(verify(*candidate))) {
        func.emitError(
            "query pipeline candidate failed verification; original retained");
        return signalPassFailure();
      }
      func->setAttr(streamAttr, UnitAttr::get(&getContext()));
      if (plan->ordinaryUnroll)
        func->setAttr("hivm.cvp_query.ownership_pending",
                      UnitAttr::get(&getContext()));
      ++transformed;
    }
    if (requireMatch && transformed == 0) {
      module.emitError(
          "query pipeline requires a verified CVPipeline attention kernel");
      signalPassFailure();
      return;
    }
    if (transformed)
      module.getBodyRegion().takeBody(candidate->getBodyRegion());
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::hivm::createCVPQueryPipelinePass(const CVPQueryPipelineOptions &options) {
  return std::make_unique<CVPQueryPipelinePass>(options);
}

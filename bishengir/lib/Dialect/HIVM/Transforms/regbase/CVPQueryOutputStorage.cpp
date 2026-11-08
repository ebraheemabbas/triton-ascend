//===- CVPQueryOutputStorage.cpp - Persistent final query storage
//----------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <array>
#include <limits>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPQUERYOUTPUTSTORAGE
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
constexpr StringLiteral kStream = "hivm.cvp_query.stream";
constexpr StringLiteral kStorage = "hivm.cvp_query.storage";
constexpr StringLiteral kLowered = "hivm.cvp_query.output_storage";

std::optional<int64_t> integer(Value value) {
  APInt result;
  if (!value || !matchPattern(value, m_ConstantInt(&result)))
    return std::nullopt;
  return result.getSExtValue();
}

bool onlyUsers(Value value, ArrayRef<Operation *> expected) {
  for (Operation *user : value.getUsers())
    if (!llvm::is_contained(expected, user))
      return false;
  return llvm::all_of(expected, [&](Operation *op) {
    return llvm::is_contained(value.getUsers(), op);
  });
}

bool constant(OpFoldResult value, int64_t expected) {
  return getConstantIntValue(value) == expected;
}

bool coordinate(OpFoldResult value, Value expected) {
  return dyn_cast<Value>(value) == expected;
}

Value comparedValue(Value condition, int64_t expected) {
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::eq)
    return {};
  if (integer(cmp.getRhs()) == expected)
    return cmp.getLhs();
  if (integer(cmp.getLhs()) == expected)
    return cmp.getRhs();
  return {};
}

bool queryParity(Value odd, scf::ForOp loop, int64_t keys) {
  Value compared = comparedValue(odd, 1);
  auto parity =
      compared ? compared.getDefiningOp<arith::RemSIOp>() : arith::RemSIOp();
  if (!parity || integer(parity.getRhs()) != 2)
    return false;
  auto ordinal = parity.getLhs().getDefiningOp<arith::DivSIOp>();
  return ordinal && ordinal.getLhs() == loop.getInductionVar() &&
         integer(ordinal.getRhs()) == keys;
}

bool lastKey(Value condition, scf::ForOp loop, int64_t keys, int64_t step) {
  Value compared = comparedValue(condition, keys - step);
  auto local =
      compared ? compared.getDefiningOp<arith::RemSIOp>() : arith::RemSIOp();
  return local && local.getLhs() == loop.getInductionVar() &&
         integer(local.getRhs()) == keys;
}

// Recognize the actual widened query-count calculation, including the clamp
// for empty work. Check its worst i32 query start rather than trusting tags.
bool completeQueryDomain(scf::ForOp loop, int64_t keys, int64_t step,
                         int64_t queryStep) {
  if (!loop.getInductionVar().getType().isInteger(64) ||
      loop->hasAttr("unsignedCmp") || integer(loop.getLowerBound()) != 0 ||
      integer(loop.getStep()) != step || keys <= 0 || step <= 0 ||
      keys % step || keys / step < 2 || queryStep <= 0)
    return false;
  auto end = loop.getUpperBound().getDefiningOp<arith::MulIOp>();
  if (!end)
    return false;
  Value count = integer(end.getRhs()) == keys   ? end.getLhs()
                : integer(end.getLhs()) == keys ? end.getRhs()
                                                : Value();
  auto ceil =
      count ? count.getDefiningOp<arith::CeilDivSIOp>() : arith::CeilDivSIOp();
  if (!ceil || integer(ceil.getRhs()) != queryStep)
    return false;
  auto positive = ceil.getLhs().getDefiningOp<arith::MaxSIOp>();
  if (!positive)
    return false;
  Value remaining = integer(positive.getRhs()) == 0   ? positive.getLhs()
                    : integer(positive.getLhs()) == 0 ? positive.getRhs()
                                                      : Value();
  auto subtract =
      remaining ? remaining.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  if (!subtract)
    return false;
  auto upper = integer(subtract.getLhs());
  auto start = subtract.getRhs().getDefiningOp<arith::ExtSIOp>();
  if (!upper || *upper <= 0 || *upper > std::numeric_limits<int32_t>::max() ||
      !start || !start.getIn().getType().isInteger(32))
    return false;
  int64_t maximumQueries =
      (*upper - int64_t{std::numeric_limits<int32_t>::min()}) / queryStep + 1;
  auto work = llvm::checkedMul(maximumQueries, keys);
  auto drain = llvm::checkedMul(int64_t{4}, step);
  return work && drain && llvm::checkedAdd(*work, *drain).has_value();
}

bool scalarLoop(scf::ForOp loop, int64_t upper, int64_t step) {
  return loop && loop.getInitArgs().size() == 1 &&
         !loop->hasAttr("unsignedCmp") && integer(loop.getLowerBound()) == 0 &&
         integer(loop.getUpperBound()) == upper &&
         integer(loop.getStep()) == step;
}

Value yielded(scf::ForOp loop) {
  auto terminator = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  return terminator.getNumOperands() == 1 ? terminator.getOperand(0) : Value();
}

bool fullTensorWrite(vector::TransferWriteOp write) {
  auto type = dyn_cast<RankedTensorType>(write.getSource().getType());
  auto vector = dyn_cast<VectorType>(write.getVector().getType());
  return type && type.hasStaticShape() && !type.getEncoding() && vector &&
         !vector.isScalable() && vector.getShape() == type.getShape() &&
         vector.getElementType() == type.getElementType() && !write.getMask() &&
         write.getPermutationMap().isIdentity() &&
         write->getNumResults() == 1 &&
         llvm::all_of(write.getIndices(),
                      [](Value index) { return integer(index) == 0; });
}

bool tileOffsets(ArrayRef<OpFoldResult> offsets, Value row, Value column) {
  return offsets.size() == 2 && coordinate(offsets[0], row) &&
         (column ? coordinate(offsets[1], column) : constant(offsets[1], 0));
}

bool tileSizes(ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  return sizes.size() == 2 && strides.size() == 2 && constant(sizes[0], 1) &&
         constant(sizes[1], 64) && constant(strides[0], 1) &&
         constant(strides[1], 1);
}

bool completeTile(scf::ForOp loop, Value row, Value column,
                  vector::TransferWriteOp writer) {
  Value result = yielded(loop);
  auto insert = result ? result.getDefiningOp<tensor::InsertSliceOp>()
                       : tensor::InsertSliceOp();
  auto extract = writer.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  if (!insert || !extract || writer->getBlock() != loop.getBody() ||
      insert.getSource() != writer->getResult(0) ||
      insert.getDest() != loop.getRegionIterArgs()[0] ||
      extract.getSource() != loop.getRegionIterArgs()[0] ||
      !tileOffsets(insert.getMixedOffsets(), row, column) ||
      !tileOffsets(extract.getMixedOffsets(), row, column) ||
      !tileSizes(insert.getMixedSizes(), insert.getMixedStrides()) ||
      !tileSizes(extract.getMixedSizes(), extract.getMixedStrides()) ||
      !fullTensorWrite(writer))
    return false;
  return onlyUsers(loop.getRegionIterArgs()[0],
                   {extract.getOperation(), insert.getOperation()}) &&
         onlyUsers(extract.getResult(), {writer.getOperation()}) &&
         onlyUsers(writer->getResult(0), {insert.getOperation()}) &&
         onlyUsers(insert.getResult(), {loop.getBody()->getTerminator()});
}

// The existing final VFs are not changed. Prove that their destination input
// contributes no old data and that every element is written before publication.
bool completeFinalVF(func::CallOp call, RankedTensorType type,
                     ModuleOp module) {
  auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
  if (!function || function.isExternal() ||
      !function->hasAttr("hivm.vector_function") ||
      function.getNumResults() != 1 || function.getResultTypes()[0] != type ||
      !llvm::hasSingleElement(function.getBody()) || call.getNumResults() != 1)
    return false;
  for (Type argument : function.getArgumentTypes())
    if (!isa<RankedTensorType>(argument))
      return false;
  SmallVector<vector::TransferWriteOp> writes;
  bool safe = true;
  function.walk([&](Operation *op) {
    if (!safe || op == function.getOperation())
      return;
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      safe = isa<RankedTensorType>(write.getSource().getType());
      writes.push_back(write);
    } else if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      safe = isa<RankedTensorType>(read.getSource().getType());
    } else if (!isa<scf::ForOp, scf::YieldOp, func::ReturnOp>(op) &&
               (isa<func::CallOp>(op) || op->getNumRegions() ||
                !isMemoryEffectFree(op))) {
      safe = false;
    }
  });
  if (!safe || writes.size() != 1)
    return false;
  auto ret =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 1)
    return false;
  auto writer = writes.front();
  BlockArgument destination;
  if (type.getRank() == 1) {
    destination = dyn_cast<BlockArgument>(writer.getSource());
    if (!destination || destination.getOwner() != &function.getBody().front() ||
        ret.getOperand(0) != writer->getResult(0) || !fullTensorWrite(writer) ||
        !onlyUsers(destination, {writer.getOperation()}) ||
        !onlyUsers(writer->getResult(0), {ret.getOperation()}))
      return false;
  } else if (type.getRank() == 2 && type.getDimSize(1) % 64 == 0) {
    auto rows = ret.getOperand(0).getDefiningOp<scf::ForOp>();
    if (!scalarLoop(rows, type.getDimSize(0), 1) ||
        rows->getBlock() != &function.getBody().front() ||
        !onlyUsers(rows.getResult(0), {ret.getOperation()}))
      return false;
    destination = dyn_cast<BlockArgument>(rows.getInitArgs()[0]);
    if (!destination || destination.getOwner() != &function.getBody().front() ||
        !onlyUsers(destination, {rows.getOperation()}))
      return false;
    if (type.getDimSize(1) == 64) {
      if (!completeTile(rows, rows.getInductionVar(), {}, writer))
        return false;
    } else {
      Value rowResult = yielded(rows);
      auto columns =
          rowResult ? rowResult.getDefiningOp<scf::ForOp>() : scf::ForOp();
      if (!scalarLoop(columns, type.getDimSize(1), 64) ||
          columns->getBlock() != rows.getBody() ||
          columns.getInitArgs()[0] != rows.getRegionIterArgs()[0] ||
          !onlyUsers(rows.getRegionIterArgs()[0], {columns.getOperation()}) ||
          !onlyUsers(columns.getResult(0), {rows.getBody()->getTerminator()}) ||
          !completeTile(columns, rows.getInductionVar(),
                        columns.getInductionVar(), writer))
        return false;
    }
  } else {
    return false;
  }
  unsigned index = destination.getArgNumber();
  return index < call.getNumOperands() &&
         bool(call.getOperand(index).getDefiningOp<tensor::EmptyOp>());
}

struct StoragePlan {
  scf::ForOp loop;
  scope::ScopeOp scope;
  scf::IfOp finalization;
  Value odd;
  RankedTensorType outputType, rowType;
  std::array<StoreOp, 2> stores;
  SmallVector<Operation *, 6> selectors;
};

std::optional<int64_t> field(DictionaryAttr attr, StringRef name) {
  auto value = attr ? attr.getAs<IntegerAttr>(name) : IntegerAttr();
  return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
}

bool selectIs(arith::SelectOp select, Value condition, Value onTrue,
              Value onFalse) {
  return select && select.getCondition() == condition &&
         select.getTrueValue() == onTrue && select.getFalseValue() == onFalse;
}

FailureOr<StoragePlan> analyze(scf::ForOp loop, ModuleOp module,
                               std::string &reason) {
  auto reject = [&](StringRef text) -> FailureOr<StoragePlan> {
    reason = text.str();
    return failure();
  };
  auto stream = loop->getAttrOfType<DictionaryAttr>(kStream);
  auto keys = field(stream, "key_count");
  auto step = field(stream, "key_step");
  auto queryStep = field(stream, "query_step");
  if (!keys || !step || !queryStep || loop.getInitArgs().size() != 9 ||
      !loop->use_empty() ||
      !completeQueryDomain(loop, *keys, *step, *queryStep))
    return reject("expected a closed nine-state complete query stream");
  auto args = loop.getRegionIterArgs();
  auto output = dyn_cast<RankedTensorType>(args[5].getType());
  auto row = dyn_cast<RankedTensorType>(args[7].getType());
  auto accumulator = dyn_cast<RankedTensorType>(args[0].getType());
  if (!output || !row || !accumulator || output.getRank() != 2 ||
      row.getRank() != 1 || !output.hasStaticShape() || !row.hasStaticShape() ||
      output.getEncoding() || row.getEncoding() || accumulator.getEncoding() ||
      !output.getElementType().isF16() || !row.getElementType().isF32() ||
      !accumulator.getElementType().isF32() ||
      output.getShape() != accumulator.getShape() ||
      output.getDimSize(0) != row.getDimSize(0) || row.getDimSize(0) <= 0 ||
      (output.getDimSize(1) != 64 && output.getDimSize(1) != 128) ||
      args[6].getType() != output || args[8].getType() != row)
    return reject("unsupported physical vector output/M shapes");
  for (unsigned i = 1; i < 5; ++i)
    if (args[i].getType() != row)
      return reject("query max/sum state shape changed");

  auto caller = loop->getParentOfType<func::FuncOp>();
  auto plans = caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
  auto geometry = plans && plans.size() == 1
                      ? dyn_cast<DictionaryAttr>(plans[0])
                      : DictionaryAttr();
  auto blockM = field(geometry, "block_m");
  if (!blockM || *blockM <= 0 || *blockM % row.getDimSize(0) ||
      field(geometry, "head_dim") != output.getDimSize(1) ||
      field(geometry, "key_count") != keys ||
      field(geometry, "block_n") != step ||
      field(geometry, "output_state") != 0 ||
      field(geometry, "sum_state") != 1 || field(geometry, "max_state") != 2)
    return reject("physical shapes or state bindings disagree with attention");

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto outputResult = dyn_cast<OpResult>(yield.getOperand(0));
  auto scope = outputResult ? dyn_cast<scope::ScopeOp>(outputResult.getOwner())
                            : scope::ScopeOp();
  auto phase = scope ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto depth = scope
                   ? scope->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
                   : IntegerAttr();
  if (!scope || scope->getParentOp() != loop ||
      outputResult.getResultNumber() != 0 || scope.getNumResults() != 5 ||
      !phase || phase.getInt() != 0 || !depth || depth.getInt() != 4 ||
      scope->getAttr("hivm.loop_core_type") !=
          TCoreTypeAttr::get(module.getContext(), TCoreType::VECTOR))
    return reject("missing protected O stage with five results");
  auto ret = cast<scope::ReturnOp>(scope.getRegion().front().getTerminator());
  for (unsigned i = 0; i < 5; ++i)
    if (!onlyUsers(scope.getResult(i), {yield.getOperation()}) ||
        yield.getOperand(i == 0 ? 0 : i + 4) != scope.getResult(i))
      return reject("O/output/M scope results escape their carried slots");

  auto evenOutput = ret.getOperand(1).getDefiningOp<arith::SelectOp>();
  if (!evenOutput)
    return reject("missing query-bank output selector");
  Value odd = evenOutput.getCondition();
  auto result = dyn_cast<OpResult>(evenOutput.getFalseValue());
  auto finish = result ? dyn_cast<scf::IfOp>(result.getOwner()) : scf::IfOp();
  if (!finish || result.getResultNumber() != 0 ||
      finish->getParentOp() != scope || finish.getNumResults() != 2 ||
      !finish.elseBlock() || finish.getResult(0).getType() != output ||
      finish.getResult(1).getType() != row || !queryParity(odd, loop, *keys) ||
      !lastKey(finish.getCondition(), loop, *keys, *step))
    return reject("unproved query parity or local last-key finalization");
  auto thenYield = cast<scf::YieldOp>(finish.thenBlock()->getTerminator());
  auto elseYield = cast<scf::YieldOp>(finish.elseBlock()->getTerminator());
  for (Operation &op : finish.elseBlock()->without_terminator())
    if (!isa<AnchorOp>(op))
      return reject("non-final branch has effects or extra state computation");

  StoragePlan plan{loop, scope, finish, odd, output, row, {}, {}};
  for (unsigned role = 0; role < 2; ++role) {
    Value even = args[5 + 2 * role], oddBank = args[6 + 2 * role];
    auto slot = elseYield.getOperand(role).getDefiningOp<arith::SelectOp>();
    auto nextEven =
        ret.getOperand(1 + 2 * role).getDefiningOp<arith::SelectOp>();
    auto nextOdd =
        ret.getOperand(2 + 2 * role).getDefiningOp<arith::SelectOp>();
    if (!selectIs(slot, odd, oddBank, even) ||
        !selectIs(nextEven, odd, even, finish.getResult(role)) ||
        !selectIs(nextOdd, odd, finish.getResult(role), oddBank) ||
        slot->getParentOp() != scope || nextEven->getParentOp() != scope ||
        nextOdd->getParentOp() != scope ||
        !onlyUsers(even, {slot.getOperation(), nextEven.getOperation()}) ||
        !onlyUsers(oddBank, {slot.getOperation(), nextOdd.getOperation()}) ||
        !onlyUsers(slot.getResult(), {elseYield.getOperation()}) ||
        !onlyUsers(nextEven.getResult(), {ret.getOperation()}) ||
        !onlyUsers(nextOdd.getResult(), {ret.getOperation()}) ||
        !onlyUsers(finish.getResult(role),
                   {nextEven.getOperation(), nextOdd.getOperation()}))
      return reject("output/M state is not a closed pass-through bank graph");
    plan.selectors.append(
        {slot.getOperation(), nextEven.getOperation(), nextOdd.getOperation()});
    Value value = thenYield.getOperand(role);
    auto call = value.getDefiningOp<func::CallOp>();
    if (!call || call->getBlock() != finish.thenBlock() ||
        !completeFinalVF(call, role == 0 ? output : row, module))
      return reject(
          "final VF does not prove a complete destination-only write");
    for (Operation *user : value.getUsers())
      if (auto store = dyn_cast<StoreOp>(user)) {
        if (plan.stores[role] || store.getSrc() != value ||
            store->getBlock() != finish.thenBlock() ||
            store->getNumOperands() != 2 || store->getNumResults())
          return reject("ambiguous or non-local final store");
        auto destination = dyn_cast<MemRefType>(store.getDst().getType());
        auto type = role == 0 ? output : row;
        if (!destination || !destination.hasStaticShape() ||
            destination.getShape() != type.getShape() ||
            destination.getElementType() != type.getElementType())
          return reject("final store has partial or mismatched shape");
        plan.stores[role] = store;
      }
    if (!plan.stores[role] ||
        !onlyUsers(value, {thenYield.getOperation(),
                           plan.stores[role].getOperation()}))
      return reject("final tensor escapes its store and pass-through yield");
  }
  unsigned stores = 0;
  loop.walk([&](StoreOp) { ++stores; });
  if (stores != 2)
    return reject("unexpected stores in the query stream");
  return plan;
}

void eraseTerminator(Block *block) {
  if (!block->empty() && block->back().hasTrait<OpTrait::IsTerminator>())
    block->back().erase();
}

void rewrite(StoragePlan &plan) {
  OpBuilder builder(plan.loop);
  Location loc = plan.loop.getLoc();
  std::array<std::array<Value, 2>, 2> banks;
  for (unsigned role = 0; role < 2; ++role) {
    auto type = role == 0 ? plan.outputType : plan.rowType;
    auto memory = MemRefType::get(
        type.getShape(), type.getElementType(), MemRefLayoutAttrInterface{},
        AddressSpaceAttr::get(builder.getContext(), AddressSpace::UB));
    for (unsigned bank = 0; bank < 2; ++bank) {
      auto allocation = builder.create<memref::AllocOp>(loc, memory);
      allocation->setAttr(
          kStorage,
          builder.getDictionaryAttr(
              {builder.getNamedAttr(
                   "role", builder.getStringAttr(role == 0 ? "output" : "M")),
               builder.getNamedAttr("bank", builder.getI32IntegerAttr(bank)),
               builder.getNamedAttr("required_slots",
                                    builder.getI32IntegerAttr(2)),
               builder.getNamedAttr("clock", builder.getStringAttr("query"))}));
      banks[role][bank] = allocation;
    }
  }
  auto loop = builder.create<scf::ForOp>(
      loc, plan.loop.getLowerBound(), plan.loop.getUpperBound(),
      plan.loop.getStep(), plan.loop.getInitArgs().take_front(5));
  loop->setAttrs(plan.loop->getAttrs());
  loop->setAttr(kLowered, builder.getUnitAttr());
  eraseTerminator(loop.getBody());
  builder.setInsertionPointToStart(loop.getBody());
  IRMapping mapping;
  mapping.map(plan.loop.getInductionVar(), loop.getInductionVar());
  mapping.map(plan.loop.getRegionIterArgs().take_front(5),
              loop.getRegionIterArgs());
  for (Operation &op : plan.loop.getBody()->without_terminator()) {
    if (&op != plan.scope.getOperation()) {
      builder.clone(op, mapping);
      continue;
    }
    auto scope = builder.create<scope::ScopeOp>(
        loc, TypeRange{plan.scope.getResult(0).getType()});
    scope->setAttrs(plan.scope->getAttrs());
    Block *body = builder.createBlock(&scope.getRegion());
    builder.setInsertionPointToStart(body);
    IRMapping local(mapping);
    for (Operation &inner :
         plan.scope.getRegion().front().without_terminator()) {
      if (llvm::is_contained(plan.selectors, &inner))
        continue;
      if (&inner != plan.finalization.getOperation()) {
        builder.clone(inner, local);
        continue;
      }
      auto finish = builder.create<scf::IfOp>(
          inner.getLoc(), TypeRange{},
          local.lookupOrDefault(plan.finalization.getCondition()), true);
      finish->setAttrs(plan.finalization->getAttrs());
      eraseTerminator(finish.thenBlock());
      eraseTerminator(finish.elseBlock());
      builder.setInsertionPointToStart(finish.thenBlock());
      for (Operation &finalOp :
           plan.finalization.thenBlock()->without_terminator()) {
        auto store = dyn_cast<StoreOp>(finalOp);
        if (!store) {
          builder.clone(finalOp, local);
          continue;
        }
        unsigned role = store == plan.stores[0] ? 0 : 1;
        Value selected = builder.create<arith::SelectOp>(
            store.getLoc(), local.lookupOrDefault(plan.odd), banks[role][1],
            banks[role][0]);
        // Explicit materialization has supported tensor-to-memref semantics.
        // Do not add restrict: later unrolling may create aliasing copies of
        // this operation, and zero-copy placement is not required here.
        OperationState materialize(
            store.getLoc(),
            bufferization::MaterializeInDestinationOp::getOperationName());
        materialize.addOperands(
            {local.lookupOrDefault(store.getSrc()), selected});
        materialize.addAttribute("writable", builder.getUnitAttr());
        builder.create(materialize);
        auto published =
            builder.create<StoreOp>(store.getLoc(), TypeRange{}, selected,
                                    local.lookupOrDefault(store.getDst()));
        published->setAttrs(store->getAttrs());
      }
      builder.create<scf::YieldOp>(inner.getLoc());
      builder.setInsertionPointToStart(finish.elseBlock());
      for (Operation &empty :
           plan.finalization.elseBlock()->without_terminator())
        builder.clone(empty, local);
      builder.create<scf::YieldOp>(inner.getLoc());
      builder.setInsertionPointAfter(finish);
    }
    auto oldReturn =
        cast<scope::ReturnOp>(plan.scope.getRegion().front().getTerminator());
    builder.create<scope::ReturnOp>(
        loc, ValueRange{local.lookupOrDefault(oldReturn.getOperand(0))});
    builder.setInsertionPointAfter(scope);
    mapping.map(plan.scope.getResult(0), scope.getResult(0));
  }
  SmallVector<Value> results;
  for (Value value : plan.loop.getYieldedValues().take_front(5))
    results.push_back(mapping.lookupOrDefault(value));
  builder.create<scf::YieldOp>(loc, results);
  plan.loop.erase();
}

bool verifyLowered(scf::ForOp loop) {
  auto stream = loop->getAttrOfType<DictionaryAttr>(kStream);
  auto keys = field(stream, "key_count"), step = field(stream, "key_step");
  auto queryStep = field(stream, "query_step");
  if (!keys || !step || !queryStep || loop.getInitArgs().size() != 5 ||
      !completeQueryDomain(loop, *keys, *step, *queryStep) ||
      !loop->use_empty())
    return false;
  SmallVector<bufferization::MaterializeInDestinationOp> materializations;
  loop.walk([&](bufferization::MaterializeInDestinationOp op) {
    materializations.push_back(op);
  });
  if (materializations.size() != 2)
    return false;
  llvm::SmallPtrSet<Operation *, 4> roots;
  scf::IfOp finalization;
  unsigned matrices = 0, rows = 0;
  for (auto op : materializations) {
    auto branch = dyn_cast<scf::IfOp>(op->getParentOp());
    auto scope = branch ? dyn_cast<scope::ScopeOp>(branch->getParentOp())
                        : scope::ScopeOp();
    if (!branch || branch.getNumResults() ||
        op->getBlock() != branch.thenBlock() ||
        (finalization && finalization != branch) || !scope ||
        scope->getParentOp() != loop || scope.getNumResults() != 1 ||
        !lastKey(branch.getCondition(), loop, *keys, *step) ||
        op->getNumResults() || !op.getWritable() || op.getRestrict())
      return false;
    finalization = branch;
    auto select = op.getDest().getDefiningOp<arith::SelectOp>();
    auto type = dyn_cast<RankedTensorType>(op.getSource().getType());
    auto memory = dyn_cast<MemRefType>(op.getDest().getType());
    if (!select || !type || !memory ||
        !queryParity(select.getCondition(), loop, *keys) ||
        memory.getShape() != type.getShape() ||
        memory.getElementType() != type.getElementType() ||
        memory.getMemorySpace() !=
            AddressSpaceAttr::get(loop.getContext(), AddressSpace::UB))
      return false;
    auto call = op.getSource().getDefiningOp<func::CallOp>();
    if (!call || call->getBlock() != branch.thenBlock() ||
        !completeFinalVF(call, type, loop->getParentOfType<ModuleOp>()) ||
        !onlyUsers(op.getSource(), {op.getOperation()}))
      return false;
    if (type.getRank() == 2 && type.getElementType().isF16())
      ++matrices;
    else if (type.getRank() == 1 && type.getElementType().isF32())
      ++rows;
    else
      return false;
    unsigned bank = 0;
    for (Value value : {select.getFalseValue(), select.getTrueValue()}) {
      auto alloc = value.getDefiningOp<memref::AllocOp>();
      if (!alloc || alloc->getBlock() != loop->getBlock() ||
          !alloc->isBeforeInBlock(loop) || !roots.insert(alloc).second ||
          !onlyUsers(value, {select.getOperation()}))
        return false;
      auto storage = alloc->getAttrOfType<DictionaryAttr>(kStorage);
      auto role = storage ? storage.getAs<StringAttr>("role") : StringAttr();
      auto clock = storage ? storage.getAs<StringAttr>("clock") : StringAttr();
      if (!role || role.getValue() != (type.getRank() == 2 ? "output" : "M") ||
          !clock || clock.getValue() != "query" ||
          field(storage, "bank") != bank++ ||
          field(storage, "required_slots") != 2)
        return false;
    }
    StoreOp store;
    for (Operation *user : select.getResult().getUsers())
      if (auto candidate = dyn_cast<StoreOp>(user)) {
        if (store || candidate.getSrc() != select.getResult() ||
            candidate->getBlock() != op->getBlock() ||
            !op->isBeforeInBlock(candidate))
          return false;
        store = candidate;
      }
    if (!store || !onlyUsers(select.getResult(),
                             {op.getOperation(), store.getOperation()}))
      return false;
  }
  unsigned stores = 0;
  loop.walk([&](StoreOp) { ++stores; });
  if (finalization.elseBlock())
    for (Operation &op : finalization.elseBlock()->without_terminator())
      if (!isa<AnchorOp>(op))
        return false;
  return matrices == 1 && rows == 1 && roots.size() == 4 && stores == 2;
}

struct CVPQueryOutputStoragePass
    : impl::CVPQueryOutputStorageBase<CVPQueryOutputStoragePass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    ModuleOp module = *copy;
    SmallVector<scf::ForOp> candidates;
    module.walk([&](scf::ForOp loop) {
      auto caller = loop->getParentOfType<func::FuncOp>();
      if (caller && caller->hasAttr("hacc.entry") && caller->hasAttr(kStream) &&
          caller->getAttr("hivm.func_core_type") ==
              TFuncCoreTypeAttr::get(module.getContext(), TFuncCoreType::AIV) &&
          loop->getAttrOfType<DictionaryAttr>(kStream))
        candidates.push_back(loop);
    });
    if (candidates.empty()) {
      if (requireMatch) {
        original.emitError(
            "CVP output storage requires a partitioned AIV query stream");
        signalPassFailure();
      }
      return;
    }
    for (scf::ForOp loop : candidates) {
      if (loop->hasAttr(kLowered)) {
        if (!verifyLowered(loop)) {
          original.emitError("invalid previously lowered CVP output storage");
          return signalPassFailure();
        }
        continue;
      }
      std::string reason;
      auto plan = analyze(loop, module, reason);
      if (failed(plan)) {
        if (requireMatch) {
          original.emitError("CVP output storage: ") << reason;
          signalPassFailure();
        }
        return;
      }
      rewrite(*plan);
    }
    bool valid = succeeded(verify(module));
    module.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kLowered) && !verifyLowered(loop))
        valid = false;
    });
    if (!valid) {
      original.emitError(
          "CVP output-storage rewrite failed structural verification");
      return signalPassFailure();
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

} // namespace
std::unique_ptr<Pass>
createCVPQueryOutputStoragePass(const CVPQueryOutputStorageOptions &options) {
  return std::make_unique<CVPQueryOutputStoragePass>(options);
}
} // namespace hivm
} // namespace mlir

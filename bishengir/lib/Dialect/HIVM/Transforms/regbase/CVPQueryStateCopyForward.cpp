//===- CVPQueryStateCopyForward.cpp - Forward owned state snapshots
//---------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <algorithm>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPQUERYSTATECOPYFORWARD
#define GEN_PASS_DEF_CVPQUERYFINALOUTPUTFORWARD
#define GEN_PASS_DEF_CVPQUERYOUTPUTCOPYFORWARD
#define GEN_PASS_DEF_CVPQUERYSCALARINPUTFORWARD
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::hivm;

namespace {
constexpr StringLiteral kStream = "hivm.cvp_query.stream";
constexpr StringLiteral kStateStorage = "hivm.cvp_query.state_storage";
constexpr StringLiteral kBank = "hivm.cvp_query.state_bank";
constexpr StringLiteral kForwarded = "hivm.cvp_query.snapshot_copies_forwarded";

// These operations only form aliases. No tensor boundary is introduced or
// changed: this pass runs after bufferization and VF operand substitution.
static Value viewSource(Operation *operation) {
  if (auto cast = dyn_cast<memref::CastOp>(operation))
    return cast.getSource();
  if (!isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp,
           memref::ReinterpretCastOp>(operation))
    return {};
  auto view = dyn_cast<ViewLikeOpInterface>(operation);
  return view ? view.getViewSource() : Value();
}

static std::optional<int64_t> elementCount(MemRefType type) {
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  std::optional<int64_t> count = 1;
  for (int64_t size : type.getShape()) {
    count = llvm::checkedMul(*count, size);
    if (!count)
      return std::nullopt;
  }
  return count;
}

static Value completeViewRoot(Value value) {
  while (Operation *operation = value.getDefiningOp()) {
    if (!isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(
            operation))
      break;
    Value source = viewSource(operation);
    auto from = source ? dyn_cast<MemRefType>(source.getType()) : MemRefType();
    auto to = dyn_cast<MemRefType>(value.getType());
    auto fromCount = elementCount(from);
    auto toCount = elementCount(to);
    if (!from || !to || !from.hasStaticShape() || !to.hasStaticShape() ||
        from.getElementType() != to.getElementType() ||
        from.getMemorySpace() != to.getMemorySpace() || !fromCount ||
        !toCount || fromCount != toCount)
      return {};
    value = source;
  }
  return value;
}

static bool copyValues(CopyOp copy, Value &source, Value &destination) {
  if (copy.getDpsInputs().size() != 1 || copy.getDpsInits().size() != 1)
    return false;
  source = copy.getDpsInputs().front();
  destination = copy.getDpsInits().front();
  return isa<MemRefType>(source.getType()) &&
         isa<MemRefType>(destination.getType());
}

static bool fromFormal(Value value, func::FuncOp function,
                       llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner() == &function.getBody().front();
  Operation *operation = value.getDefiningOp();
  if (!operation)
    return false;
  if (Value source = viewSource(operation))
    return fromFormal(source, function, visited);
  if (auto select = dyn_cast<arith::SelectOp>(operation))
    return fromFormal(select.getTrueValue(), function, visited) &&
           fromFormal(select.getFalseValue(), function, visited);
  return false;
}

// A read-only formal is insufficient if the VF can write the bank through a
// hidden global/pointer or an unknown nested call. All memory bases must come
// from actual formals and only the recognized vector/scalar accesses are used.
static bool closedVFMemory(func::FuncOp function) {
  bool valid = true;
  function.walk([&](Operation *operation) {
    if (!valid || operation == function.getOperation())
      return;
    for (Value operand : operation->getOperands()) {
      if (!isa<BaseMemRefType>(operand.getType()))
        continue;
      llvm::DenseSet<Value> visited;
      if (!fromFormal(operand, function, visited)) {
        valid = false;
        return;
      }
    }
    for (Value result : operation->getResults())
      if (isa<BaseMemRefType>(result.getType()) && !viewSource(operation) &&
          !isa<arith::SelectOp>(operation)) {
        valid = false;
        return;
      }
    if (isa<vector::TransferReadOp, vector::TransferWriteOp, vector::LoadOp,
            vector::StoreOp, memref::LoadOp, memref::StoreOp, scf::ForOp,
            scf::IfOp>(operation))
      return;
    if (!isMemoryEffectFree(operation))
      valid = false;
  });
  return valid;
}

// Check the actual callee argument, including its derived views. A declaration,
// nested call, write, return or unrecognized use is not a read-only proof.
static bool readOnlyVFArgument(func::CallOp call, unsigned argument) {
  auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!function || function.isExternal() ||
      !function->hasAttr("hivm.vector_function") ||
      argument >= function.getNumArguments() || !closedVFMemory(function))
    return false;
  SmallVector<Value> pending{function.getArgument(argument)};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (viewSource(operation) == value) {
        pending.push_back(operation->getResult(0));
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        if (use.getOperandNumber() == 0)
          return false;
        pending.push_back(select.getResult());
        continue;
      }
      if (auto read = dyn_cast<vector::TransferReadOp>(operation)) {
        if (read.getSource() == value)
          continue;
      }
      if (auto read = dyn_cast<vector::LoadOp>(operation)) {
        if (read.getBase() == value)
          continue;
      }
      if (auto read = dyn_cast<memref::LoadOp>(operation)) {
        if (read.getMemref() == value)
          continue;
      }
      if (isa<memref::DimOp>(operation))
        continue;
      return false;
    }
  }
  return true;
}

// Scope movement is performed later by CreatePreload. Readers and recurrent
// writers must stay in this one O stage; ordinary block order between different
// pipeline stages is not used as an ownership proof.
static bool onlyIfAncestors(Operation *operation, Operation *stage) {
  for (Operation *parent = operation->getParentOp(); parent != stage;
       parent = parent ? parent->getParentOp() : nullptr) {
    if (!parent || !isa<scf::IfOp>(parent))
      return false;
  }
  return true;
}

static Operation *inBlock(Operation *operation, Block *block) {
  while (operation && operation->getBlock() != block)
    operation = operation->getParentOp();
  return operation;
}

static bool exclusiveBranches(Operation *a, Operation *b, Operation *stage) {
  for (Operation *parent = a->getParentOp(); parent && parent != stage;
       parent = parent->getParentOp()) {
    auto conditional = dyn_cast<scf::IfOp>(parent);
    if (!conditional)
      continue;
    Region *aRegion = a->getParentRegion();
    while (aRegion && aRegion->getParentOp() != parent)
      aRegion = aRegion->getParentOp()->getParentRegion();
    Region *bRegion = b->getParentRegion();
    while (bRegion && bRegion->getParentOp() != parent)
      bRegion = bRegion->getParentOp()->getParentRegion();
    if (aRegion && bRegion && aRegion != bRegion)
      return true;
  }
  return false;
}

static bool readBeforeWrite(Operation *reader, Operation *writer,
                            Operation *stage) {
  if (exclusiveBranches(reader, writer, stage))
    return true;
  for (Block *block = reader->getBlock(); block;) {
    Operation *read = inBlock(reader, block);
    Operation *write = inBlock(writer, block);
    if (read && write && read != write)
      return read->isBeforeInBlock(write);
    Operation *parent = block->getParentOp();
    if (parent == stage)
      break;
    block = parent ? parent->getBlock() : nullptr;
  }
  return false;
}

struct ForwardPlan {
  CopyOp copy;
  memref::AllocOp snapshot;
  Value bank;
  scope::ScopeOp stage;
  scf::ForOp stream;
};

static bool collectSnapshotReaders(ForwardPlan &plan,
                                   SmallVectorImpl<Operation *> &readers) {
  SmallVector<Value> pending{plan.snapshot.getResult()};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (operation == plan.copy.getOperation())
        continue;
      if (!plan.stage->isAncestor(operation) ||
          !onlyIfAncestors(operation, plan.stage))
        return false;
      if (viewSource(operation) == value) {
        pending.push_back(operation->getResult(0));
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        if (use.getOperandNumber() == 0)
          return false;
        pending.push_back(select.getResult());
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(operation)) {
        if (!readOnlyVFArgument(call, use.getOperandNumber()))
          return false;
        readers.push_back(operation);
        continue;
      }
      if (auto copy = dyn_cast<CopyOp>(operation)) {
        Value source, destination;
        if (copyValues(copy, source, destination) && source == value &&
            destination != value) {
          readers.push_back(operation);
          continue;
        }
      }
      if (isa<memref::DimOp>(operation))
        continue;
      return false;
    }
  }
  return !readers.empty();
}

static bool bankWritesAfterReads(ForwardPlan &plan,
                                 ArrayRef<Operation *> readers) {
  SmallVector<Value> pending{plan.bank};
  llvm::DenseSet<Value> visited;
  bool unique = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (viewSource(operation) == value) {
        pending.push_back(operation->getResult(0));
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(operation)) {
        if (use.getOperandNumber() == 0)
          return false;
        pending.push_back(select.getResult());
        continue;
      }
      if (auto mark = dyn_cast<annotation::MarkOp>(operation)) {
        if (value != plan.bank ||
            !mark->hasAttr(HIVMMemoryUniqueAttr::getMnemonic()))
          return false;
        unique = true;
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(operation)) {
        if (readOnlyVFArgument(call, use.getOperandNumber()))
          continue;
        return false;
      }
      auto copy = dyn_cast<CopyOp>(operation);
      Value source, destination;
      if (!copy || !copyValues(copy, source, destination))
        return false;
      if (source == value && destination != value)
        continue;
      if (destination != value)
        return false;
      // Initialization/final writes outside the stream must be in its own
      // block. No asynchronous or unrecognized enclosing region is accepted.
      if (!plan.stream->isAncestor(operation)) {
        if (operation->getBlock() == plan.stream->getBlock())
          continue;
        return false;
      }
      if (!plan.stage->isAncestor(operation) ||
          !onlyIfAncestors(operation, plan.stage))
        return false;
      Operation *top = inBlock(operation, plan.copy->getBlock());
      if (top && top != plan.copy && top->isBeforeInBlock(plan.copy))
        continue;
      for (Operation *reader : readers)
        if (!readBeforeWrite(reader, operation, plan.stage))
          return false;
    }
  }
  return unique;
}

static FailureOr<ForwardPlan> analyze(CopyOp copy) {
  Value source, destination;
  if (!copyValues(copy, source, destination))
    return failure();
  Value bank = completeViewRoot(source);
  Value temporary = completeViewRoot(destination);
  auto allocation = temporary ? temporary.getDefiningOp<memref::AllocOp>()
                              : memref::AllocOp();
  auto bankAllocation =
      bank ? bank.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  auto type = bank ? dyn_cast<MemRefType>(bank.getType()) : MemRefType();
  if (!allocation || !bankAllocation || bank == temporary || !type ||
      type != temporary.getType() || type.getRank() != 2 ||
      !type.hasStaticShape() || !type.getElementType().isF32() ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      !type.getLayout().isIdentity())
    return failure();
  for (NamedAttribute attribute : allocation->getAttrs())
    if (attribute.getName() != "alignment" &&
        attribute.getName() != "operandSegmentSizes")
      return failure();
  if (auto alignment = allocation->getAttrOfType<IntegerAttr>("alignment")) {
    auto bankAlignment =
        bankAllocation->getAttrOfType<IntegerAttr>("alignment");
    if (!bankAlignment || alignment.getInt() <= 0 ||
        bankAlignment.getInt() < alignment.getInt())
      return failure();
  }
  auto address = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  auto descriptor = bankAllocation->getAttrOfType<DictionaryAttr>(kBank);
  auto role = descriptor ? descriptor.getAs<StringAttr>("role") : StringAttr();
  auto clock =
      descriptor ? descriptor.getAs<StringAttr>("clock") : StringAttr();
  auto slots = descriptor ? descriptor.getAs<IntegerAttr>("required_slots")
                          : IntegerAttr();
  auto number =
      descriptor ? descriptor.getAs<IntegerAttr>("bank") : IntegerAttr();
  if (!address || address.getAddressSpace() != AddressSpace::UB || !role ||
      role.getValue() != "accumulator" || !clock || clock.getValue() != "O" ||
      !slots || slots.getInt() != 1 || !number || number.getInt() != 0)
    return failure();
  auto stage = copy->getParentOfType<scope::ScopeOp>();
  auto stream =
      stage ? dyn_cast<scf::ForOp>(stage->getParentOp()) : scf::ForOp();
  auto phase = stage ? stage->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto depth = stage
                   ? stage->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
                   : IntegerAttr();
  if (!stage || !stream || !stream->hasAttr(kStateStorage) ||
      !stream->hasAttr(kStream) || !stream.getInitArgs().empty() || !phase ||
      phase.getInt() != 0 || !depth || depth.getInt() != 4 ||
      stage->getAttr("hivm.loop_core_type") !=
          TCoreTypeAttr::get(copy.getContext(), TCoreType::VECTOR) ||
      copy->getBlock() != &stage.getRegion().front() ||
      allocation->getBlock() != copy->getBlock() ||
      !allocation->isBeforeInBlock(copy) ||
      bankAllocation->getBlock() != stream->getBlock() ||
      !bankAllocation->isBeforeInBlock(stream))
    return failure();
  auto function = stream->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("hacc.entry") ||
      function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(copy.getContext(), TFuncCoreType::AIV))
    return failure();
  ForwardPlan plan{copy, allocation, bank, stage, stream};
  SmallVector<Operation *> readers;
  if (!collectSnapshotReaders(plan, readers) ||
      !bankWritesAfterReads(plan, readers))
    return failure();
  return plan;
}

struct CVPQueryStateCopyForwardPass
    : impl::CVPQueryStateCopyForwardBase<CVPQueryStateCopyForwardPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(original->clone()));
    ModuleOp module = *cloned;
    SmallVector<CopyOp> candidates;
    module.walk([&](CopyOp copy) { candidates.push_back(copy); });
    unsigned count = 0;
    for (CopyOp copy : candidates) {
      auto plan = analyze(copy);
      if (failed(plan))
        continue;
      plan->copy.erase();
      plan->snapshot.getResult().replaceAllUsesWith(plan->bank);
      plan->snapshot.erase();
      ++count;
    }
    if (!count && requireMatch) {
      original.emitError("CVP state copy forwarding requires a proved "
                         "read-only full O snapshot");
      return signalPassFailure();
    }
    if (failed(verify(module))) {
      original.emitError(
          "CVP state copy forwarding failed structural verification");
      return signalPassFailure();
    }
    module->setAttr(
        kForwarded,
        IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};
// Final output forwarding stays at the same abstract-buffer level as input
// forwarding. It does not move or remove a wait: the normal planner and sync
// pipeline run after it, including dependencies on an earlier output DMA.
static bool finalConstantIs(Value value, int64_t expected) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto attr =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
  return attr && attr.getInt() == expected;
}

// A complete row may be emitted in fixed-width chunks. Both loops must cover
// their exact static extent, with no carried values or overlapping chunks.
static bool completeOutputChunkLoops(scf::ForOp rows, scf::ForOp columns,
                                     func::FuncOp function, MemRefType type,
                                     int64_t width) {
  return rows && columns && width > 0 && type.getDimSize(0) > 0 &&
         type.getDimSize(1) > width && type.getDimSize(1) % width == 0 &&
         rows->getParentOp() == function && columns->getParentOp() == rows &&
         rows.getInitArgs().empty() && columns.getInitArgs().empty() &&
         finalConstantIs(rows.getLowerBound(), 0) &&
         finalConstantIs(rows.getUpperBound(), type.getDimSize(0)) &&
         finalConstantIs(rows.getStep(), 1) &&
         finalConstantIs(columns.getLowerBound(), 0) &&
         finalConstantIs(columns.getUpperBound(), type.getDimSize(1)) &&
         finalConstantIs(columns.getStep(), width);
}

static bool completeOutputChunkView(memref::SubViewOp view, Value formal,
                                    scf::ForOp rows, scf::ForOp columns,
                                    int64_t width) {
  if (!view || view.getSource() != formal || view->getParentOp() != columns ||
      view.getOffsets().size() != 2 ||
      view.getOffsets()[0] != rows.getInductionVar() ||
      view.getOffsets()[1] != columns.getInductionVar())
    return false;
  auto offsets = view.getStaticOffsets();
  auto sizes = view.getStaticSizes();
  auto strides = view.getStaticStrides();
  return offsets.size() == 2 && offsets[0] == ShapedType::kDynamic &&
         offsets[1] == ShapedType::kDynamic && sizes.size() == 2 &&
         sizes[0] == 1 && sizes[1] == width && strides.size() == 2 &&
         strides[0] == 1 && strides[1] == 1;
}

static bool finalOutputArgument(func::CallOp call, unsigned argument) {
  auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!function || function.isExternal() ||
      !function->hasAttr("hivm.vector_function") ||
      argument >= function.getNumArguments() || !closedVFMemory(function))
    return false;
  Value formal = function.getArgument(argument);
  auto type = dyn_cast<MemRefType>(formal.getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
      !type.getElementType().isF16() || !type.getLayout().isIdentity())
    return false;
  SmallVector<Value> pending{formal};
  llvm::DenseSet<Value> visited;
  SmallVector<vector::TransferWriteOp> writes;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (viewSource(operation) == value) {
        pending.push_back(operation->getResult(0));
        continue;
      }
      if (auto write = dyn_cast<vector::TransferWriteOp>(operation)) {
        if (write.getSource() != value)
          return false;
        writes.push_back(write);
        continue;
      }
      if (isa<memref::DimOp>(operation))
        continue;
      // A read, escape, nested call or another write is not write-only output.
      return false;
    }
  }
  if (writes.size() != 1)
    return false;
  auto write = writes.front();
  auto vectorType = dyn_cast<VectorType>(write.getVector().getType());
  if (!vectorType || vectorType.isScalable() || vectorType.getRank() != 2 ||
      write.getMask() || !write.getPermutationMap().isIdentity() ||
      write.getIndices().size() != 2 ||
      !llvm::all_of(write.getIndices(),
                    [](Value v) { return finalConstantIs(v, 0); }))
    return false;
  // One complete vector, one full row per iteration, or exact nonoverlapping
  // column chunks in every row. No partial/masked/conditional coverage.
  if (write.getSource() == formal && write->getParentOp() == function)
    return llvm::equal(vectorType.getShape(), type.getShape());
  auto row = write.getSource().getDefiningOp<memref::SubViewOp>();
  auto loop = dyn_cast<scf::ForOp>(write->getParentOp());
  auto rows = loop ? dyn_cast<scf::ForOp>(loop->getParentOp()) : scf::ForOp();
  int64_t width = vectorType.getDimSize(1);
  if (completeOutputChunkLoops(rows, loop, function, type, width) &&
      completeOutputChunkView(row, formal, rows, loop, width))
    return vectorType.getDimSize(0) == 1;
  if (!row || !loop || loop->getParentOp() != function ||
      completeViewRoot(row.getSource()) != formal ||
      !loop.getInitArgs().empty() ||
      !finalConstantIs(loop.getLowerBound(), 0) ||
      !finalConstantIs(loop.getUpperBound(), type.getDimSize(0)) ||
      !finalConstantIs(loop.getStep(), 1) ||
      row.getStaticOffsets().size() != 2 ||
      row.getStaticOffsets()[0] != ShapedType::kDynamic ||
      row.getStaticOffsets()[1] != 0 || row.getOffsets().size() != 1 ||
      row.getOffsets().front() != loop.getInductionVar() ||
      row.getStaticSizes().size() != 2 || row.getStaticSizes()[0] != 1 ||
      row.getStaticSizes()[1] != type.getDimSize(1) ||
      row.getStaticStrides().size() != 2 || row.getStaticStrides()[0] != 1 ||
      row.getStaticStrides()[1] != 1)
    return false;
  return vectorType.getDimSize(0) == 1 &&
         vectorType.getDimSize(1) == type.getDimSize(1);
}

struct FinalForwardPlan {
  CopyOp copy;
  memref::AllocOp temporary;
  func::CallOp writer;
  unsigned argument = 0;
  arith::SelectOp destination;
  SmallVector<memref::AllocOp> banks;
  SmallVector<Operation *> temporaryViews;
  int64_t alignment = 1;
};

static bool finalTemporaryUses(FinalForwardPlan &plan) {
  SmallVector<Value> pending{plan.temporary.getResult()};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *operation = use.getOwner();
      if (isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(
              operation) &&
          viewSource(operation) == value &&
          completeViewRoot(operation->getResult(0)) ==
              plan.temporary.getResult()) {
        plan.temporaryViews.push_back(operation);
        pending.push_back(operation->getResult(0));
        continue;
      }
      if (operation == plan.copy) {
        Value source, destination;
        if (!copyValues(plan.copy, source, destination) || source != value)
          return false;
        continue;
      }
      auto call = dyn_cast<func::CallOp>(operation);
      if (!call || plan.writer || value != plan.temporary.getResult() ||
          !call->hasAttr("hivm.vector_function") ||
          !finalOutputArgument(call, use.getOperandNumber()))
        return false;
      plan.writer = call;
      plan.argument = use.getOperandNumber();
    }
  }
  return bool(plan.writer);
}

// A ScopeOp result aliases exactly its corresponding return operand. Follow
// only that identity; an arbitrary region result is not an allocation proof.
static Value scopeResultSource(Value value) {
  auto producer = value.getDefiningOp<scope::ScopeOp>();
  auto result = dyn_cast<OpResult>(value);
  if (!producer || !result || !llvm::hasSingleElement(producer.getRegion()))
    return {};
  auto returned =
      dyn_cast<scope::ReturnOp>(producer.getRegion().front().getTerminator());
  if (!returned || returned.getNumOperands() != producer.getNumResults())
    return {};
  Value source = returned.getOperand(result.getResultNumber());
  return source.getType() == value.getType() ? source : Value();
}

static bool finalDisjointFromBanks(Value value, const FinalForwardPlan &plan,
                                   llvm::DenseSet<Value> &visited) {
  if (!isa<BaseMemRefType>(value.getType()))
    return true;
  if (!visited.insert(value).second)
    return true;
  for (memref::AllocOp bank : plan.banks)
    if (bank.getResult() == value)
      return false;
  if (value.getDefiningOp<memref::AllocOp>())
    return true; // Different fresh allocations have distinct abstract storage.
  if (Operation *operation = value.getDefiningOp()) {
    if (Value source = viewSource(operation))
      return finalDisjointFromBanks(source, plan, visited);
    if (auto select = dyn_cast<arith::SelectOp>(operation))
      return finalDisjointFromBanks(select.getTrueValue(), plan, visited) &&
             finalDisjointFromBanks(select.getFalseValue(), plan, visited);
    if (Value source = scopeResultSource(value))
      return finalDisjointFromBanks(source, plan, visited);
  }
  if (isa<BlockArgument>(value)) {
    auto type = dyn_cast<MemRefType>(value.getType());
    auto space = type
                     ? dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace())
                     : AddressSpaceAttr();
    return space && space.getAddressSpace() != AddressSpace::UB;
  }
  return false;
}

static bool finalInterveningEffects(const FinalForwardPlan &plan) {
  for (Operation *op = plan.writer->getNextNode(); op != plan.copy;
       op = op->getNextNode()) {
    if (!op || op->getNumRegions() || isa<func::CallOp>(op))
      return false;
    if (isa<CopyOp, StoreOp>(op)) {
      for (Value operand : op->getOperands()) {
        llvm::DenseSet<Value> visited;
        if (!finalDisjointFromBanks(operand, plan, visited))
          return false;
      }
      continue;
    }
    // Never move a write across a wait, DMA fence or other unknown effect.
    if (!isa<AnchorOp>(op) && !isMemoryEffectFree(op))
      return false;
  }
  return true;
}

static bool finalBankUses(const FinalForwardPlan &plan) {
  SmallVector<Value> pending;
  for (memref::AllocOp bank : plan.banks)
    pending.push_back(bank.getResult());
  llvm::DenseSet<Value> visited;
  bool stored = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (viewSource(op) == value) {
        pending.push_back(op->getResult(0));
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(op)) {
        if (use.getOperandNumber() == 0)
          return false;
        pending.push_back(select.getResult());
        continue;
      }
      if (op == plan.copy) {
        Value source, destination;
        if (!copyValues(plan.copy, source, destination) || destination != value)
          return false;
        continue;
      }
      if (auto store = dyn_cast<StoreOp>(op)) {
        if (store.getDpsInputs().size() != 1 ||
            store.getDpsInits().size() != 1 ||
            store.getDpsInputs().front() != value)
          return false;
        auto target =
            dyn_cast<MemRefType>(store.getDpsInits().front().getType());
        auto space =
            target ? dyn_cast_or_null<AddressSpaceAttr>(target.getMemorySpace())
                   : AddressSpaceAttr();
        if (!space || space.getAddressSpace() != AddressSpace::GM)
          return false;
        stored = true;
        continue;
      }
      if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
        if (!mark->hasAttr(HIVMMemoryUniqueAttr::getMnemonic()) ||
            mark->hasAttr("hivm.multi_buffer"))
          return false;
        continue;
      }
      if (isa<memref::DimOp>(op))
        continue;
      return false;
    }
  }
  return stored;
}

static FailureOr<FinalForwardPlan>
analyzeFinalOutput(CopyOp copy, DominanceInfo &dominance) {
  Value source, destination;
  if (!copyValues(copy, source, destination))
    return failure();
  Value root = completeViewRoot(source);
  auto temporary =
      root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  Value target = completeViewRoot(destination);
  auto select =
      target ? target.getDefiningOp<arith::SelectOp>() : arith::SelectOp();
  auto type =
      temporary ? dyn_cast<MemRefType>(temporary.getType()) : MemRefType();
  if (!temporary || !select || !type || type != target.getType() ||
      type.getRank() != 2 || !type.hasStaticShape() ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      !type.getElementType().isF16() || !type.getLayout().isIdentity())
    return failure();
  auto address = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  if (!address || address.getAddressSpace() != AddressSpace::UB)
    return failure();
  auto stage = copy->getParentOfType<scope::ScopeOp>();
  auto stream =
      stage ? dyn_cast<scf::ForOp>(stage->getParentOp()) : scf::ForOp();
  auto phase = stage ? stage->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto depth = stage
                   ? stage->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
                   : IntegerAttr();
  if (!stage || !stream || !stream->hasAttr(kStream) ||
      !stream->hasAttr("hivm.cvp_query.output_storage") ||
      !stream.getInitArgs().empty() || !phase || phase.getInt() != 0 ||
      !depth || depth.getInt() != 4 ||
      stage->getAttr("hivm.loop_core_type") !=
          TCoreTypeAttr::get(copy.getContext(), TCoreType::VECTOR) ||
      !onlyIfAncestors(copy, stage))
    return failure();
  auto function = stream->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("hacc.entry") ||
      function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(copy.getContext(), TFuncCoreType::AIV))
    return failure();
  FinalForwardPlan plan;
  plan.copy = copy;
  plan.temporary = temporary;
  plan.destination = select;
  for (NamedAttribute attr : temporary->getAttrs())
    if (attr.getName() != "alignment" &&
        attr.getName() != "operandSegmentSizes")
      return failure();
  if (auto alignment = temporary->getAttrOfType<IntegerAttr>("alignment")) {
    if (alignment.getInt() <= 0 ||
        (alignment.getInt() & (alignment.getInt() - 1)))
      return failure();
    plan.alignment = alignment.getInt();
  }
  unsigned slotMask = 0;
  for (Value operand : {select.getTrueValue(), select.getFalseValue()}) {
    Value base = completeViewRoot(operand);
    auto bank =
        base ? base.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
    auto descriptor =
        bank ? bank->getAttrOfType<DictionaryAttr>("hivm.cvp_query.storage")
             : DictionaryAttr();
    auto role =
        descriptor ? descriptor.getAs<StringAttr>("role") : StringAttr();
    auto clock =
        descriptor ? descriptor.getAs<StringAttr>("clock") : StringAttr();
    auto slots = descriptor ? descriptor.getAs<IntegerAttr>("required_slots")
                            : IntegerAttr();
    auto number =
        descriptor ? descriptor.getAs<IntegerAttr>("bank") : IntegerAttr();
    if (!bank || bank.getType() != type || !role ||
        role.getValue() != "output" || !clock || clock.getValue() != "query" ||
        !slots || slots.getInt() != 2 || !number ||
        (number.getInt() != 0 && number.getInt() != 1) ||
        bank->getBlock() != stream->getBlock() ||
        !bank->isBeforeInBlock(stream))
      return failure();
    if (auto alignment = bank->getAttrOfType<IntegerAttr>("alignment")) {
      if (alignment.getInt() <= 0 ||
          (alignment.getInt() & (alignment.getInt() - 1)))
        return failure();
    }
    slotMask |= 1u << number.getInt();
    plan.banks.push_back(bank);
  }
  if (slotMask != 3 || plan.banks[0] == plan.banks[1] ||
      !finalTemporaryUses(plan) ||
      plan.writer->getBlock() != copy->getBlock() ||
      !plan.writer->isBeforeInBlock(copy) ||
      temporary->getBlock() != copy->getBlock() ||
      !temporary->isBeforeInBlock(plan.writer) ||
      !dominance.dominates(select.getCondition(), plan.writer))
    return failure();
  for (unsigned index = 0; index < plan.writer.getNumOperands(); ++index) {
    if (index == plan.argument)
      continue;
    llvm::DenseSet<Value> visited;
    if (!finalDisjointFromBanks(plan.writer.getOperand(index), plan, visited))
      return failure();
  }
  if (!finalInterveningEffects(plan) || !finalBankUses(plan))
    return failure();
  return plan;
}

struct CVPQueryFinalOutputForwardPass
    : impl::CVPQueryFinalOutputForwardBase<CVPQueryFinalOutputForwardPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(original->clone()));
    ModuleOp module = *cloned;
    DominanceInfo dominance(module);
    SmallVector<CopyOp> candidates;
    module.walk([&](CopyOp copy) { candidates.push_back(copy); });
    unsigned count = 0;
    for (CopyOp copy : candidates) {
      auto plan = analyzeFinalOutput(copy, dominance);
      if (failed(plan))
        continue;
      OpBuilder builder(plan->writer);
      for (memref::AllocOp bank : plan->banks) {
        auto current = bank->getAttrOfType<IntegerAttr>("alignment");
        if (!current || current.getInt() < plan->alignment)
          bank->setAttr("alignment",
                        builder.getI64IntegerAttr(plan->alignment));
      }
      auto destination = builder.create<arith::SelectOp>(
          copy.getLoc(), plan->destination.getCondition(),
          plan->banks[0].getResult(), plan->banks[1].getResult());
      plan->writer->setOperand(plan->argument, destination.getResult());
      plan->copy.erase();
      for (Operation *view : llvm::reverse(plan->temporaryViews))
        if (view->use_empty())
          view->erase();
      assert(plan->temporary->use_empty() &&
             "validated temporary unexpectedly escapes");
      plan->temporary.erase();
      ++count;
    }
    if (!count && requireMatch) {
      original.emitError("CVP final output forwarding requires a proved full "
                         "write into query output banks");
      return signalPassFailure();
    }
    if (failed(verify(module))) {
      original.emitError(
          "CVP final output forwarding failed structural verification");
      return signalPassFailure();
    }
    module->setAttr(
        "hivm.cvp_query.final_output_copies_forwarded",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};
// The O-update VF has separate PV, old-O and output formals even when its
// caller aliases PV and output. Only this closed row-local contract permits
// changing the output to old-O's persistent bank without a copyback.
static bool outputForwardRow(memref::SubViewOp row, Value formal,
                             scf::ForOp loop, int64_t columns, bool scalar) {
  if (!row || row.getSource() != formal || row->getParentOp() != loop ||
      row.getOffsets().size() != 1 ||
      row.getOffsets().front() != loop.getInductionVar())
    return false;
  auto offsets = row.getStaticOffsets();
  auto sizes = row.getStaticSizes();
  auto strides = row.getStaticStrides();
  if (scalar)
    return offsets.size() == 1 && offsets[0] == ShapedType::kDynamic &&
           sizes.size() == 1 && sizes[0] == 1 && strides.size() == 1 &&
           strides[0] == 1;
  return offsets.size() == 2 && offsets[0] == ShapedType::kDynamic &&
         offsets[1] == 0 && sizes.size() == 2 && sizes[0] == 1 &&
         sizes[1] == columns && strides.size() == 2 && strides[0] == 1 &&
         strides[1] == 1;
}

static bool outputForwardVector(VectorType type, int64_t columns) {
  return type && !type.isScalable() && type.getRank() == 2 &&
         type.getDimSize(0) == 1 && type.getDimSize(1) == columns &&
         type.getElementType().isF32();
}

static bool outputForwardVF(func::CallOp call, bool first = false) {
  auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      call, call.getCalleeAttr());
  if (!function || function.isExternal() ||
      !function->hasAttr("hivm.vector_function") ||
      function.getNumArguments() != (first ? 3u : 4u) ||
      call.getNumOperands() != function.getNumArguments() ||
      function.getFunctionType().getNumResults() || call.getNumResults() ||
      !closedVFMemory(function))
    return false;
  if (first) {
    auto kind =
        function->getAttrOfType<StringAttr>("hivm.cvp_attention.boundary");
    if (!kind || kind.getValue() != "query_first_output")
      return false;
  }
  auto type = dyn_cast<MemRefType>(function.getArgument(1).getType());
  if (!type || !type.hasStaticShape() || type.getRank() != 2 ||
      !type.getLayout().isIdentity() || !type.getElementType().isF32() ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      function.getArgument(2).getType() != type ||
      (!first && function.getArgument(3).getType() != type))
    return false;
  auto alpha = dyn_cast<MemRefType>(function.getArgument(0).getType());
  if (!alpha || !alpha.hasStaticShape() || alpha.getRank() != 1 ||
      alpha.getDimSize(0) != type.getDimSize(0) ||
      !alpha.getElementType().isF32() || !alpha.getLayout().isIdentity() ||
      alpha.getMemorySpace() != type.getMemorySpace())
    return false;
  // No noalias/restrict/alignment contract may be silently invalidated.
  for (unsigned i = 0; i != function.getNumArguments(); ++i)
    if (auto attrs = function.getArgAttrDict(i); attrs && !attrs.empty())
      return false;
  SmallVector<scf::ForOp> loops;
  function.walk([&](scf::ForOp loop) { loops.push_back(loop); });
  if (loops.size() != 1 && loops.size() != 2)
    return false;
  scf::ForOp loop = loops.front(), columns;
  int64_t width = type.getDimSize(1);
  if (loops.size() == 2) {
    // Operation walking need not visit a parent before its nested loop.
    if (loops[0]->getParentOp() == function) {
      loop = loops[0];
      columns = loops[1];
    } else if (loops[1]->getParentOp() == function) {
      loop = loops[1];
      columns = loops[0];
    } else {
      return false;
    }
    auto step = columns.getStep().getDefiningOp<arith::ConstantOp>();
    auto integer =
        step ? dyn_cast<IntegerAttr>(step.getValue()) : IntegerAttr();
    if (!integer)
      return false;
    width = integer.getInt();
    if (!completeOutputChunkLoops(loop, columns, function, type, width))
      return false;
  } else if (loop->getParentOp() != function || !loop.getInitArgs().empty() ||
             !finalConstantIs(loop.getLowerBound(), 0) ||
             !finalConstantIs(loop.getUpperBound(), type.getDimSize(0)) ||
             !finalConstantIs(loop.getStep(), 1)) {
    return false;
  }
  SmallVector<vector::TransferReadOp> reads;
  vector::TransferWriteOp write;
  for (unsigned i = 0; i != function.getNumArguments(); ++i) {
    Value formal = function.getArgument(i);
    if (!formal.hasOneUse())
      return false;
    auto row = dyn_cast<memref::SubViewOp>(*formal.getUsers().begin());
    bool complete =
        columns && i != 0
            ? completeOutputChunkView(row, formal, loop, columns, width)
            : outputForwardRow(row, formal, loop, width, i == 0);
    if (!complete || !row.getResult().hasOneUse())
      return false;
    Operation *access = *row.getResult().getUsers().begin();
    Operation *expectedParent =
        columns && i != 0 ? columns.getOperation() : loop.getOperation();
    if (access->getParentOp() != expectedParent)
      return false;
    if (i == function.getNumArguments() - 1) {
      write = dyn_cast<vector::TransferWriteOp>(access);
      if (!write || write.getSource() != row.getResult() || write.getMask() ||
          !write.getPermutationMap().isIdentity() ||
          !outputForwardVector(write.getVectorType(), width) ||
          write.getIndices().size() != 2 ||
          !llvm::all_of(write.getIndices(),
                        [](Value v) { return finalConstantIs(v, 0); }))
        return false;
      continue;
    }
    auto read = dyn_cast<vector::TransferReadOp>(access);
    if (!read || read.getSource() != row.getResult() || read.getMask() ||
        !outputForwardVector(read.getVectorType(), width) ||
        !llvm::all_of(read.getIndices(),
                      [](Value v) { return finalConstantIs(v, 0); }))
      return false;
    if (i) {
      if (read.getIndices().size() != 2 ||
          !read.getPermutationMap().isIdentity())
        return false;
    } else {
      AffineMap map = read.getPermutationMap();
      if (read.getIndices().size() != 1 || map.getNumDims() != 1 ||
          map.getNumSymbols() || map.getNumResults() != 2 ||
          map.getResult(0) != getAffineDimExpr(0, call.getContext()) ||
          map.getResult(1) != getAffineConstantExpr(0, call.getContext()))
        return false;
    }
    reads.push_back(read);
  }
  for (unsigned i = 0; i != reads.size(); ++i) {
    // Alpha belongs to the row and must be ready before any column chunk.
    // PV/old-O reads and the matching write are in the same chunk block.
    Operation *consumer =
        columns && i == 0 ? columns.getOperation() : write.getOperation();
    if (!reads[i]->isBeforeInBlock(consumer))
      return false;
  }
  // The first helper has no old-O input. Its output can use a disjoint bank
  // after the closed full-row input reads, regardless of its pure arithmetic.
  // Do not replace its retained alpha*0/add with a copy: that is a separate
  // floating-point proof. The ordinary aliased old-O update stays exact below.
  if (first)
    return true;
  auto sum = write.getVector().getDefiningOp<arith::AddFOp>();
  auto product =
      sum ? sum.getRhs().getDefiningOp<arith::MulFOp>() : arith::MulFOp();
  return sum && product && sum.getLhs() == reads[1].getResult() &&
         product.getLhs() == reads[2].getResult() &&
         product.getRhs() == reads[0].getResult();
}

struct OutputCopyForwardPlan {
  CopyOp copy;
  memref::AllocOp bank;
  memref::AllocOp pv;
  func::CallOp writer;
  func::CallOp epilogue;
  arith::SelectOp oldO;
  scope::ScopeOp stage;
  scf::ForOp stream;
  func::CallOp firstWriter;
};

static bool outputCopyBankUses(OutputCopyForwardPlan &plan) {
  SmallVector<Value> pending{plan.bank.getResult()};
  llvm::DenseSet<Value> visited;
  bool unique = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (viewSource(op) == value &&
          completeViewRoot(op->getResult(0)) == plan.bank.getResult()) {
        pending.push_back(op->getResult(0));
        continue;
      }
      if (op == plan.copy) {
        Value source, destination;
        if (!copyValues(plan.copy, source, destination) || destination != value)
          return false;
        continue;
      }
      // Preserve the existing full initialization before the stream. It is
      // not a recurrent writer and its source is the unchanged first-key bank.
      if (auto init = dyn_cast<CopyOp>(op)) {
        Value source, destination;
        if (!copyValues(init, source, destination) || destination != value ||
            source.getType() != destination.getType() ||
            init->getBlock() != plan.stream->getBlock() ||
            !init->isBeforeInBlock(plan.stream))
          return false;
        Value root = completeViewRoot(source);
        if (plan.oldO) {
          if (root != completeViewRoot(plan.oldO.getTrueValue()))
            return false;
        } else {
          // Boundary first-O no longer passes the zero select to the ordinary
          // writer. Preserve the original private initialization unchanged.
          auto initial =
              root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
          if (!plan.firstWriter || !initial || initial == plan.bank ||
              initial == plan.pv || initial.getType() != plan.bank.getType() ||
              initial->getBlock() != plan.stream->getBlock() ||
              !initial->isBeforeInBlock(plan.stream))
            return false;
        }
        continue;
      }
      if (plan.oldO && op == plan.oldO && use.getOperandNumber() == 2 &&
          value == plan.bank.getResult())
        continue;
      if (plan.firstWriter && op == plan.writer &&
          use.getOperandNumber() == 2 && value == plan.bank.getResult())
        continue;
      if (op == plan.epilogue && plan.epilogue.getNumOperands() == 8 &&
          use.getOperandNumber() == 4 && value == plan.bank.getResult())
        continue; // Proved readonly in the opposite finalization branch.
      if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
        if (value != plan.bank.getResult() ||
            !mark->hasAttr(HIVMMemoryUniqueAttr::getMnemonic()) ||
            mark->hasAttr("hivm.multi_buffer") ||
            mark->hasAttr("hivm.tightly_coupled_buffer"))
          return false;
        unique = true;
        continue;
      }
      return false;
    }
  }
  return unique && (!plan.oldO ||
                    (plan.oldO.getResult().hasOneUse() &&
                     *plan.oldO.getResult().getUsers().begin() == plan.writer));
}

static bool outputCopyPVUses(OutputCopyForwardPlan &plan,
                             DominanceInfo &dominance) {
  SmallVector<Value> pending{plan.pv.getResult()};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (viewSource(op) == value &&
          completeViewRoot(op->getResult(0)) == plan.pv.getResult()) {
        pending.push_back(op->getResult(0));
        continue;
      }
      if (op == plan.copy) {
        Value source, destination;
        if (!copyValues(plan.copy, source, destination) || source != value)
          return false;
        continue;
      }
      if (op == plan.writer && value == plan.pv.getResult() &&
          (use.getOperandNumber() == 1 || use.getOperandNumber() == 3))
        continue;
      if (plan.firstWriter && op == plan.firstWriter &&
          value == plan.pv.getResult() &&
          (use.getOperandNumber() == 1 || use.getOperandNumber() == 2))
        continue;
      if (isa<annotation::MarkOp>(op))
        continue; // Existing TCB/multibuffer marks remain untouched.
      auto call = dyn_cast<func::CallOp>(op);
      if (!call || plan.epilogue || value != plan.pv.getResult() ||
          call.getNumResults() || !finalOutputArgument(call, 5) ||
          !plan.stage->isAncestor(call) || !onlyIfAncestors(call, plan.stage))
        return false;
      if (call.getNumOperands() == 6) {
        if (use.getOperandNumber() != 4 || !readOnlyVFArgument(call, 4) ||
            plan.copy->getBlock() != &plan.stage.getRegion().front() ||
            !dominance.dominates(plan.copy.getOperation(), call.getOperation()))
          return false;
      } else if (call.getNumOperands() == 8) {
        auto function = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
            call, call.getCalleeAttr());
        auto kind = function ? function->getAttrOfType<StringAttr>(
                                   "hivm.cvp_attention.boundary")
                             : StringAttr();
        auto finalization = dyn_cast<scf::IfOp>(plan.copy->getParentOp());
        if (!kind || kind.getValue() != "query_last_output" ||
            function.getNumArguments() != 8 ||
            function.getFunctionType().getNumResults() ||
            use.getOperandNumber() != 7 ||
            call.getOperand(4) != plan.bank.getResult() ||
            !readOnlyVFArgument(call, 4) || !readOnlyVFArgument(call, 6) ||
            !readOnlyVFArgument(call, 7) || !finalization ||
            finalization.getNumResults() ||
            finalization->getParentOp() != plan.stage ||
            plan.copy->getBlock() != finalization.elseBlock() ||
            call->getBlock() != finalization.thenBlock() ||
            !exclusiveBranches(plan.copy, call, plan.stage))
          return false;
      } else {
        return false;
      }
      plan.epilogue = call;
    }
  }
  return bool(plan.epilogue);
}

// A producer scope may return a fresh alpha allocation. Use the same exact
// return-identity proof as final-output forwarding, retaining unknown UB roots.
static bool outputCopyDisjoint(Value value, memref::AllocOp bank,
                               llvm::DenseSet<Value> &visited) {
  if (!isa<BaseMemRefType>(value.getType()))
    return true;
  if (!visited.insert(value).second)
    return true;
  if (value == bank.getResult())
    return false;
  if (value.getDefiningOp<memref::AllocOp>())
    return true;
  if (Operation *op = value.getDefiningOp()) {
    if (Value source = viewSource(op))
      return outputCopyDisjoint(source, bank, visited);
    if (auto select = dyn_cast<arith::SelectOp>(op))
      return outputCopyDisjoint(select.getTrueValue(), bank, visited) &&
             outputCopyDisjoint(select.getFalseValue(), bank, visited);
    if (Value source = scopeResultSource(value))
      return outputCopyDisjoint(source, bank, visited);
  }
  if (isa<BlockArgument>(value)) {
    auto type = dyn_cast<MemRefType>(value.getType());
    auto space = type
                     ? dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace())
                     : AddressSpaceAttr();
    return space && space.getAddressSpace() != AddressSpace::UB;
  }
  return false;
}

// Boundary first-O emits one closed VF in each branch. Other live operations
// or nested regions are not a complete writer proof, even if they are pure.
static func::CallOp singleOutputWriter(Block *block) {
  if (!block)
    return {};
  func::CallOp writer;
  for (Operation &op : block->without_terminator()) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (writer)
        return {};
      writer = call;
      continue;
    }
    if (op.getNumRegions() || (!isa<AnchorOp>(op) && !isMemoryEffectFree(&op)))
      return {};
  }
  return writer;
}

static FailureOr<OutputCopyForwardPlan>
analyzeOutputCopy(CopyOp copy, DominanceInfo &dominance) {
  Value source, destination;
  if (!copyValues(copy, source, destination) ||
      source.getType() != destination.getType())
    return failure();
  Value pvValue = completeViewRoot(source);
  Value bankValue = completeViewRoot(destination);
  auto pv =
      pvValue ? pvValue.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  auto bank = bankValue ? bankValue.getDefiningOp<memref::AllocOp>()
                        : memref::AllocOp();
  auto type = bank ? bank.getType() : MemRefType();
  if (!bank || !pv || bank == pv || pv.getType() != type ||
      type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getLayout().isIdentity() || !type.getElementType().isF32() ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0)
    return failure();
  auto address = dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace());
  auto descriptor = bank->getAttrOfType<DictionaryAttr>(kBank);
  auto role = descriptor ? descriptor.getAs<StringAttr>("role") : StringAttr();
  auto clock =
      descriptor ? descriptor.getAs<StringAttr>("clock") : StringAttr();
  auto slots = descriptor ? descriptor.getAs<IntegerAttr>("required_slots")
                          : IntegerAttr();
  auto number =
      descriptor ? descriptor.getAs<IntegerAttr>("bank") : IntegerAttr();
  if (!address || address.getAddressSpace() != AddressSpace::UB || !role ||
      role.getValue() != "accumulator" || !clock || clock.getValue() != "O" ||
      !slots || slots.getInt() != 1 || !number || number.getInt() != 0)
    return failure();
  auto stage = copy->getParentOfType<scope::ScopeOp>();
  auto stream =
      stage ? dyn_cast<scf::ForOp>(stage->getParentOp()) : scf::ForOp();
  auto phase = stage ? stage->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto depth = stage
                   ? stage->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
                   : IntegerAttr();
  auto streamInfo = stream ? stream->getAttrOfType<DictionaryAttr>(kStream)
                           : DictionaryAttr();
  auto mode =
      streamInfo ? streamInfo.getAs<StringAttr>("pipeline_mode") : StringAttr();
  auto lanes =
      streamInfo ? streamInfo.getAs<IntegerAttr>("key_unroll") : IntegerAttr();
  if (!stage || !stream || !stream->hasAttr(kStateStorage) ||
      !stream->hasAttr("hivm.cvp_query.output_storage") ||
      !stream.getInitArgs().empty() || !mode || mode.getValue() != "skew" ||
      !lanes || lanes.getInt() != 1 || !phase || phase.getInt() != 0 ||
      !depth || depth.getInt() != 4 ||
      stage->getAttr("hivm.loop_core_type") !=
          TCoreTypeAttr::get(copy.getContext(), TCoreType::VECTOR) ||
      !onlyIfAncestors(copy, stage) || pv->getBlock() != stream.getBody() ||
      !pv->isBeforeInBlock(stage) || bank->getBlock() != stream->getBlock() ||
      !bank->isBeforeInBlock(stream))
    return failure();
  auto function = stream->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("hacc.entry") ||
      function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(copy.getContext(), TFuncCoreType::AIV))
    return failure();
  func::CallOp writer;
  func::CallOp firstWriter;
  for (Operation *op = copy->getPrevNode(); op; op = op->getPrevNode()) {
    if (auto call = dyn_cast<func::CallOp>(op)) {
      writer = call;
      break;
    }
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults())
        return failure();
      firstWriter = singleOutputWriter(branch.thenBlock());
      writer = singleOutputWriter(branch.elseBlock());
      if (!firstWriter || !writer)
        return failure();
      break;
    }
    if (op->getNumRegions() || (!isa<AnchorOp>(op) && !isMemoryEffectFree(op)))
      return failure(); // Do not move the bank write across a fence/effect.
  }
  if (!writer || !outputForwardVF(writer) || writer.getOperand(1) != pvValue ||
      writer.getOperand(3) != pvValue)
    return failure();
  auto oldO = writer.getOperand(2).getDefiningOp<arith::SelectOp>();
  Value zeroValue = oldO ? completeViewRoot(oldO.getTrueValue()) : Value();
  auto zero = zeroValue ? zeroValue.getDefiningOp<memref::AllocOp>()
                        : memref::AllocOp();
  if (firstWriter) {
    if (!outputForwardVF(firstWriter, true) ||
        firstWriter.getOperand(0) != writer.getOperand(0) ||
        firstWriter.getOperand(1) != pvValue ||
        firstWriter.getOperand(2) != pvValue ||
        writer.getOperand(2) != bankValue || oldO)
      return failure();
  } else if (!oldO || oldO.getFalseValue() != bankValue || !zero ||
             zero == bank || zero == pv || zero.getType() != type ||
             zero->getBlock() != stream->getBlock() ||
             !zero->isBeforeInBlock(stream)) {
    return failure();
  }
  OutputCopyForwardPlan plan{copy, bank,  pv,     writer,     {},
                             oldO, stage, stream, firstWriter};
  // Find and prove the final reader before admitting any old-O use by it.
  if (!outputCopyPVUses(plan, dominance) || !outputCopyBankUses(plan))
    return failure();
  // Different abstract allocations plus mem_unique keep the bank disjoint
  // from PV, alpha, scalar state and final FP16 output during stock replanning.
  for (auto call : {writer, plan.epilogue})
    for (unsigned i = 0; i != call.getNumOperands(); ++i) {
      if ((call == writer && i == 2) || (call == plan.epilogue && i == 4))
        continue;
      llvm::DenseSet<Value> visited;
      if (!outputCopyDisjoint(call.getOperand(i), bank, visited))
        return failure();
    }
  if (firstWriter)
    for (Value operand : firstWriter.getOperands()) {
      llvm::DenseSet<Value> visited;
      if (!outputCopyDisjoint(operand, bank, visited))
        return failure();
    }
  return plan;
}

struct CVPQueryOutputCopyForwardPass
    : impl::CVPQueryOutputCopyForwardBase<CVPQueryOutputCopyForwardPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(original->clone()));
    ModuleOp module = *cloned;
    DominanceInfo dominance(module);
    SmallVector<CopyOp> candidates;
    module.walk([&](CopyOp copy) { candidates.push_back(copy); });
    unsigned count = 0;
    for (CopyOp copy : candidates) {
      auto plan = analyzeOutputCopy(copy, dominance);
      if (failed(plan))
        continue;
      plan->writer->setOperand(3, plan->bank.getResult());
      if (plan->firstWriter)
        plan->firstWriter->setOperand(2, plan->bank.getResult());
      if (plan->epilogue.getNumOperands() == 6)
        plan->epilogue->setOperand(4, plan->bank.getResult());
      // The final eight-argument helper still reads raw PV at argument 7.
      // Its old-O argument 4 already names the persistent bank unchanged.
      plan->copy.erase();
      ++count;
    }
    if (!count && requireMatch) {
      original.emitError("CVP output copy forwarding requires a proved "
                         "row-local full O update and closed final read");
      return signalPassFailure();
    }
    if (failed(verify(module))) {
      original.emitError(
          "CVP output copy forwarding failed structural verification");
      return signalPassFailure();
    }
    module->setAttr(
        "hivm.cvp_query.output_copies_forwarded",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

// Forward the paired readonly l/m snapshots inside one stored P stage. The
// normal target memory/sync pipeline still runs after this abstract-buffer
// rewrite; issue order between different stages is not a completion proof.
struct ScalarInputChain {
  arith::SelectOp oldValue;
  arith::SelectOp selected;
  SmallVector<memref::AllocOp, 3> temporaries;
  SmallVector<CopyOp, 3> copies; // Last copy first, toward the persistent bank.
  SmallVector<memref::AllocOp, 2> banks;
  SmallVector<int64_t, 2> bankNumbers;
  int64_t alignment = 1;
};

struct ScalarInputForwardPlan {
  func::CallOp reader;
  scope::ScopeOp stage;
  scf::ForOp stream;
  SmallVector<ScalarInputChain, 2> chains;
};

static bool scalarAllocation(memref::AllocOp allocation, MemRefType type,
                             int64_t &alignment, bool bank = false) {
  if (!allocation || allocation.getType() != type ||
      allocation->getNumOperands())
    return false;
  for (NamedAttribute attr : allocation->getAttrs())
    if (attr.getName() != "alignment" &&
        attr.getName() != "operandSegmentSizes" &&
        (!bank || attr.getName() != kBank))
      return false;
  if (auto required = allocation->getAttrOfType<IntegerAttr>("alignment")) {
    int64_t value = required.getInt();
    if (value <= 0 || (value & (value - 1)))
      return false;
    alignment = std::max(alignment, value);
  }
  return true;
}

// A readonly initial value may be hoisted before the ordinary outer
// autoblockify loop. Its pointer and initialization are never rewritten.
static bool scalarSeedBeforeStream(memref::AllocOp seed, scf::ForOp stream) {
  Operation *contained = stream;
  while (contained->getBlock() != seed->getBlock()) {
    auto enclosing = dyn_cast<scf::ForOp>(contained->getParentOp());
    if (!enclosing)
      return false;
    contained = enclosing;
  }
  return seed->isBeforeInBlock(contained);
}

static FailureOr<ScalarInputChain>
analyzeScalarChain(func::CallOp reader, unsigned argument, StringRef role,
                   scope::ScopeOp stage, scf::ForOp stream) {
  ScalarInputChain chain;
  chain.oldValue = reader.getOperand(argument).getDefiningOp<arith::SelectOp>();
  if (!chain.oldValue || chain.oldValue->getBlock() != reader->getBlock() ||
      !chain.oldValue->isBeforeInBlock(reader))
    return failure();
  auto type = dyn_cast<MemRefType>(chain.oldValue.getType());
  auto space = type ? dyn_cast_or_null<AddressSpaceAttr>(type.getMemorySpace())
                    : AddressSpaceAttr();
  if (!type || type.getRank() != 1 || !type.hasStaticShape() ||
      type.getDimSize(0) <= 0 || !type.getElementType().isF32() ||
      !type.getLayout().isIdentity() || !space ||
      space.getAddressSpace() != AddressSpace::UB)
    return failure();
  auto seed = chain.oldValue.getTrueValue().getDefiningOp<memref::AllocOp>();
  if (!seed || seed.getType() != type || !scalarSeedBeforeStream(seed, stream))
    return failure();

  Value current = chain.oldValue.getFalseValue();
  Operation *consumer = chain.oldValue;
  for (unsigned index = 0; index != 3; ++index) {
    auto temporary = current.getDefiningOp<memref::AllocOp>();
    if (!scalarAllocation(temporary, type, chain.alignment) ||
        temporary->getBlock() != &stage.getRegion().front())
      return failure();
    CopyOp incoming;
    for (OpOperand &use : current.getUses()) {
      auto copy = dyn_cast<CopyOp>(use.getOwner());
      Value source, destination;
      if (copy && copyValues(copy, source, destination) &&
          destination == current && source != current) {
        if (incoming)
          return failure();
        incoming = copy;
      } else if (use.getOwner() != consumer) {
        return failure(); // No extra reader, alias, annotation or escape.
      }
    }
    Value source, destination;
    if (!incoming || !copyValues(incoming, source, destination) ||
        source.getType() != type || destination != current ||
        incoming->getBlock() != reader->getBlock() ||
        !temporary->isBeforeInBlock(incoming) ||
        !incoming->isBeforeInBlock(consumer))
      return failure();
    for (NamedAttribute attr : incoming->getAttrs())
      if (attr.getName() != "operandSegmentSizes" &&
          attr.getName() != "operand_segment_sizes")
        return failure();
    // The consumer must read this value exactly once, as a whole input.
    if (consumer == chain.oldValue.getOperation()) {
      if (chain.oldValue.getTrueValue() == current)
        return failure();
    } else {
      Value read, written;
      if (!copyValues(cast<CopyOp>(consumer), read, written) ||
          read != current || written == current)
        return failure();
    }
    chain.temporaries.push_back(temporary);
    chain.copies.push_back(incoming);
    current = source;
    consumer = incoming;
  }
  chain.selected = current.getDefiningOp<arith::SelectOp>();
  if (!chain.selected || chain.selected->getBlock() != reader->getBlock() ||
      !chain.selected->isBeforeInBlock(chain.copies.back()))
    return failure();
  unsigned mask = 0;
  for (Value value :
       {chain.selected.getTrueValue(), chain.selected.getFalseValue()}) {
    auto bank = value.getDefiningOp<memref::AllocOp>();
    if (!scalarAllocation(bank, type, chain.alignment, true) || bank == seed ||
        bank->getBlock() != stream->getBlock() ||
        !bank->isBeforeInBlock(stream))
      return failure();
    auto descriptor = bank->getAttrOfType<DictionaryAttr>(kBank);
    auto bankRole =
        descriptor ? descriptor.getAs<StringAttr>("role") : StringAttr();
    auto clock =
        descriptor ? descriptor.getAs<StringAttr>("clock") : StringAttr();
    auto slots = descriptor ? descriptor.getAs<IntegerAttr>("required_slots")
                            : IntegerAttr();
    auto number =
        descriptor ? descriptor.getAs<IntegerAttr>("bank") : IntegerAttr();
    if (!bankRole || bankRole.getValue() != role || !clock ||
        clock.getValue() != "query" || !slots || slots.getInt() != 2 ||
        !number || (number.getInt() != 0 && number.getInt() != 1))
      return failure();
    mask |= 1u << number.getInt();
    chain.banks.push_back(bank);
    chain.bankNumbers.push_back(number.getInt());
  }
  if (mask != 3 || chain.banks[0] == chain.banks[1])
    return failure();
  return chain;
}

static bool scalarBankUses(ScalarInputForwardPlan &plan, memref::AllocOp bank) {
  SmallVector<Value> pending{bank.getResult()};
  llvm::DenseSet<Value> visited;
  bool unique = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (viewSource(op) == value) {
        if (completeViewRoot(op->getResult(0)) != bank.getResult())
          return false;
        pending.push_back(op->getResult(0));
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(op)) {
        if (use.getOperandNumber() == 0)
          return false;
        pending.push_back(select.getResult());
        continue;
      }
      if (auto mark = dyn_cast<annotation::MarkOp>(op)) {
        // The inherent default write effect is not another annotation.
        auto effects = mark.getEffectsAttr();
        if (value != bank.getResult() ||
            !mark->hasAttr(HIVMMemoryUniqueAttr::getMnemonic()) ||
            mark.getAttrNum() != 1 || mark->getNumOperands() != 1 ||
            effects.size() != 1 ||
            cast<StringAttr>(effects[0]).getValue() != "write")
          return false;
        unique = true;
        continue;
      }
      if (auto call = dyn_cast<func::CallOp>(op)) {
        if (!readOnlyVFArgument(call, use.getOperandNumber()))
          return false;
        continue;
      }
      if (isa<memref::DimOp>(op))
        continue;
      auto copy = dyn_cast<CopyOp>(op);
      Value source, destination;
      if (!copy || !copyValues(copy, source, destination))
        return false;
      if (source == value && destination != value)
        continue;
      if (destination != value || source == value)
        return false;
      if (!plan.stream->isAncestor(op)) {
        if (op->getBlock() != plan.stream->getBlock() ||
            !op->isBeforeInBlock(plan.stream))
          return false;
        continue;
      }
      // Keep all recurrent writers in this stage after the ultimate VF read.
      // The unchanged target pipeline supplies its normal VF/PIPE_V contract.
      if (!plan.stage->isAncestor(op) || !onlyIfAncestors(op, plan.stage) ||
          !readBeforeWrite(plan.reader, op, plan.stage))
        return false;
    }
  }
  return unique;
}

static bool scalarInputWindow(ScalarInputForwardPlan &plan) {
  Operation *first = plan.chains[0].copies.back();
  llvm::SmallPtrSet<Operation *, 8> copies;
  for (ScalarInputChain &chain : plan.chains)
    for (CopyOp copy : chain.copies) {
      copies.insert(copy);
      if (copy->isBeforeInBlock(first))
        first = copy;
    }
  for (Operation *op = first; op != plan.reader; op = op->getNextNode()) {
    if (!op)
      return false;
    bool valid = true;
    op->walk([&](Operation *nested) {
      if (isa<scf::IfOp>(nested))
        return;
      if (nested->getNumRegions() || isa<func::CallOp>(nested)) {
        valid = false;
        return;
      }
      if (isa<CopyOp>(nested)) {
        valid &= copies.contains(nested);
        return;
      }
      if (isa<AnchorOp, memref::AllocOp, annotation::MarkOp>(nested))
        return;
      if (!isMemoryEffectFree(nested))
        valid = false;
    });
    if (!valid)
      return false;
  }
  return true;
}

static FailureOr<ScalarInputForwardPlan>
analyzeScalarInputs(func::CallOp reader) {
  if (reader.getNumOperands() != 7 || reader.getNumResults() ||
      !reader->hasAttr("hivm.vector_function") ||
      !readOnlyVFArgument(reader, 2) || !readOnlyVFArgument(reader, 3))
    return failure();
  auto stage = dyn_cast<scope::ScopeOp>(reader->getParentOp());
  auto stream =
      stage ? dyn_cast<scf::ForOp>(stage->getParentOp()) : scf::ForOp();
  auto info = stream ? stream->getAttrOfType<DictionaryAttr>(kStream)
                     : DictionaryAttr();
  auto mode = info ? info.getAs<StringAttr>("pipeline_mode") : StringAttr();
  auto lanes = info ? info.getAs<IntegerAttr>("key_unroll") : IntegerAttr();
  auto slots = info ? info.getAs<IntegerAttr>("query_slots") : IntegerAttr();
  auto phase = stage ? stage->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto depth = stage
                   ? stage->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
                   : IntegerAttr();
  if (!stage || !stream || !stream->hasAttr(kStateStorage) ||
      !stream.getInitArgs().empty() || !mode || mode.getValue() != "skew" ||
      !lanes || lanes.getInt() != 1 || !slots || slots.getInt() != 2 ||
      !phase || phase.getInt() != 2 || !depth || depth.getInt() != 4 ||
      stage->getAttr("hivm.loop_core_type") !=
          TCoreTypeAttr::get(reader.getContext(), TCoreType::VECTOR))
    return failure();
  auto function = stream->getParentOfType<func::FuncOp>();
  if (!function || !function->hasAttr("hacc.entry") ||
      function->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(reader.getContext(), TFuncCoreType::AIV))
    return failure();
  auto sum = analyzeScalarChain(reader, 2, "sum", stage, stream);
  auto maximum = analyzeScalarChain(reader, 3, "maximum", stage, stream);
  if (failed(sum) || failed(maximum) ||
      sum->oldValue.getType() != maximum->oldValue.getType() ||
      sum->oldValue.getCondition() != maximum->oldValue.getCondition() ||
      sum->selected.getCondition() != maximum->selected.getCondition() ||
      sum->bankNumbers != maximum->bankNumbers)
    return failure();
  ScalarInputForwardPlan plan{reader, stage, stream, {}};
  plan.chains.push_back(*sum);
  plan.chains.push_back(*maximum);
  llvm::DenseSet<Value> distinct;
  for (ScalarInputChain &chain : plan.chains) {
    for (memref::AllocOp temporary : chain.temporaries)
      if (!distinct.insert(temporary.getResult()).second)
        return failure();
    for (memref::AllocOp bank : chain.banks) {
      if (!distinct.insert(bank.getResult()).second ||
          !scalarBankUses(plan, bank))
        return failure();
      // The old-input selectors may contain only distinct readonly seeds and
      // the matched temporaries; every other actual must be disjoint from all
      // four persistent roots, including the produced l/m/alpha/P operands.
      for (unsigned argument = 0; argument != reader.getNumOperands();
           ++argument) {
        Value actual = reader.getOperand(argument);
        if (argument == 2 || argument == 3)
          actual = actual.getDefiningOp<arith::SelectOp>().getTrueValue();
        llvm::DenseSet<Value> visited;
        if (!outputCopyDisjoint(actual, bank, visited))
          return failure();
      }
    }
  }
  if (!scalarInputWindow(plan))
    return failure();
  return plan;
}

struct CVPQueryScalarInputForwardPass
    : impl::CVPQueryScalarInputForwardBase<CVPQueryScalarInputForwardPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> cloned(cast<ModuleOp>(original->clone()));
    ModuleOp module = *cloned;
    SmallVector<func::CallOp> readers;
    module.walk([&](func::CallOp call) { readers.push_back(call); });
    unsigned count = 0;
    for (func::CallOp reader : readers) {
      auto plan = analyzeScalarInputs(reader);
      if (failed(plan))
        continue;
      for (ScalarInputChain &chain : plan->chains) {
        for (memref::AllocOp bank : chain.banks)
          bank->setAttr("alignment",
                        IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                         chain.alignment));
        chain.oldValue->setOperand(2, chain.selected.getResult());
        for (auto pair : llvm::zip(chain.copies, chain.temporaries)) {
          std::get<0>(pair).erase();
          std::get<1>(pair).erase();
          ++count;
        }
      }
    }
    if (!count && requireMatch) {
      original.emitError("CVP scalar input forwarding requires a proved paired "
                         "six-copy readonly P-state contract");
      return signalPassFailure();
    }
    if (failed(verify(module))) {
      original.emitError("CVP scalar input forwarding failed structural "
                         "verification");
      return signalPassFailure();
    }
    module->setAttr(
        "hivm.cvp_query.scalar_input_copies_forwarded",
        IntegerAttr::get(IntegerType::get(&getContext(), 64), count));
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

} // namespace

std::unique_ptr<Pass> mlir::hivm::createCVPQueryStateCopyForwardPass(
    const CVPQueryStateCopyForwardOptions &options) {
  return std::make_unique<CVPQueryStateCopyForwardPass>(options);
}

std::unique_ptr<Pass> mlir::hivm::createCVPQueryFinalOutputForwardPass(
    const CVPQueryFinalOutputForwardOptions &options) {
  return std::make_unique<CVPQueryFinalOutputForwardPass>(options);
}

std::unique_ptr<Pass> mlir::hivm::createCVPQueryOutputCopyForwardPass(
    const CVPQueryOutputCopyForwardOptions &options) {
  return std::make_unique<CVPQueryOutputCopyForwardPass>(options);
}

std::unique_ptr<Pass> mlir::hivm::createCVPQueryScalarInputForwardPass(
    const CVPQueryScalarInputForwardOptions &options) {
  return std::make_unique<CVPQueryScalarInputForwardPass>(options);
}

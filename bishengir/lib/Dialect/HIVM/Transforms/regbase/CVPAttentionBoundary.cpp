//===- CVPAttentionBoundary.cpp - Specialize attention boundaries
//----------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include <functional>
#include <iterator>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONBOUNDARY
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
constexpr StringLiteral kBoundaryKind = "hivm.cvp_attention.boundary";

std::optional<int64_t> integer(Value value) {
  APInt result;
  if (!matchPattern(value, m_ConstantInt(&result)))
    return std::nullopt;
  return result.getSExtValue();
}

BlockArgument rootArgument(Value value, func::FuncOp function) {
  while (auto view = value.getDefiningOp<memref::SubViewOp>())
    value = view.getSource();
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return {};
  return argument;
}

bool isUB(Type type, ArrayRef<int64_t> shape, unsigned width) {
  auto memory = dyn_cast<MemRefType>(type);
  auto element =
      memory ? dyn_cast<FloatType>(memory.getElementType()) : FloatType();
  return memory && memory.hasStaticShape() && memory.getShape() == shape &&
         memory.getLayout().isIdentity() && element &&
         element.getWidth() == width &&
         memory.getMemorySpace() ==
             AddressSpaceAttr::get(type.getContext(), AddressSpace::UB);
}

bool zeroIndices(ValueRange indices) {
  return llvm::all_of(indices, [](Value value) { return integer(value) == 0; });
}

bool isChunk(Type type, unsigned width) {
  auto vector = dyn_cast<VectorType>(type);
  auto element =
      vector ? dyn_cast<FloatType>(vector.getElementType()) : FloatType();
  return vector && !vector.isScalable() &&
         vector.getShape() == ArrayRef<int64_t>{1, 64} && element &&
         element.getWidth() == width;
}

bool simpleFunction(func::FuncOp function, int64_t denseKeys = 0) {
  if (!function || function.isExternal() || function.getNumResults() ||
      !function->hasAttr("hivm.vector_function"))
    return false;
  for (NamedAttribute attr : function->getAttrs()) {
    StringRef name = attr.getName().getValue();
    // This geometry metadata has no argument/result indices to invalidate
    // when removing known initial-state inputs from a cloned producer.
    if (denseKeys && name == "hivm.cvp_attention.dense_p") {
      auto geometry = dyn_cast<DictionaryAttr>(attr.getValue());
      auto keys =
          geometry ? geometry.getAs<IntegerAttr>("keys") : IntegerAttr();
      auto rows =
          geometry ? geometry.getAs<IntegerAttr>("query_rows") : IntegerAttr();
      auto padding =
          geometry ? geometry.getAs<BoolAttr>("padding_required") : BoolAttr();
      if (!geometry || geometry.size() != 3 || !keys ||
          !keys.getType().isInteger(64) || keys.getInt() != denseKeys ||
          !rows || !rows.getType().isInteger(64) || rows.getInt() != 64 ||
          !padding)
        return false;
      continue;
    }
    if (denseKeys && name == "hfusion.has_fill" &&
        !isa<UnitAttr>(attr.getValue()))
      return false;
    if (name != "sym_name" && name != "function_type" &&
        name != "sym_visibility" && name != "arg_attrs" &&
        name != "res_attrs" && name != "hivm.func_core_type" &&
        name != "hivm.vector_function" && name != "no_inline" &&
        name != "hfusion.has_fill" && name != "hivm.vf_mode" &&
        name != kBoundaryKind)
      return false;
  }
  auto interface = cast<FunctionOpInterface>(function.getOperation());
  for (unsigned index = 0; index < function.getNumArguments(); ++index)
    for (NamedAttribute attr : interface.getArgAttrs(index))
      if (attr.getName().getValue() != "alignment" &&
          attr.getName().getValue() != "llvm.align")
        return false;
  bool valid = true;
  function.walk([&](Operation *op) {
    if (!valid || op == function.getOperation())
      return;
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      valid = !read.getMask() && zeroIndices(read.getIndices()) &&
              bool(rootArgument(read.getSource(), function));
      return;
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      valid = !write.getMask() && zeroIndices(write.getIndices()) &&
              bool(rootArgument(write.getSource(), function));
      return;
    }
    if (isa<scf::ForOp, scf::YieldOp, func::ReturnOp>(op))
      return;
    if (isa<func::CallOp>(op) || op->getNumRegions() || !isMemoryEffectFree(op))
      valid = false;
  });
  return valid;
}

bool simpleUniqueCall(func::CallOp call, func::FuncOp function,
                      ModuleOp module) {
  for (NamedAttribute attr : call->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "callee" && name != "hivm.vector_function" &&
        name != "no_inline" && name != "ptc_simdvf")
      return false;
  }
  auto uses = SymbolTable::getSymbolUses(function, module);
  return uses && llvm::hasSingleElement(*uses) &&
         uses->begin()->getUser() == call.getOperation();
}

bool vectorEvent(Operation *op, bool set, PIPE from, PIPE to) {
  if (!op || (set ? !isa<SyncBlockSetOp>(op) : !isa<SyncBlockWaitOp>(op)))
    return false;
  auto core = op->getAttrOfType<TCoreTypeAttr>("tcore_type");
  auto source = op->getAttrOfType<PipeAttr>("tpipe");
  auto destination = op->getAttrOfType<PipeAttr>("pipe");
  auto flag = op->getAttrOfType<IntegerAttr>("static_flag_id");
  auto mode = set ? cast<SyncBlockSetOp>(op).getTsyncInstrModeAttr()
                  : cast<SyncBlockWaitOp>(op).getTsyncInstrModeAttr();
  return core && core.getTcoretype() == TCoreType::VECTOR && source &&
         destination && source.getPipe() == from &&
         destination.getPipe() == to && flag && flag.getInt() >= 0 &&
         !op->getNumOperands() &&
         (!mode || mode.getSyncInstrMode() ==
                       SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
}

// All new PV accesses occupy exactly the old update call's protected scope.
// First-state accesses stay inside the old P scope. Preserve every event and
// reject moved releases, nested synchronization, dynamic events or other modes.
// Final physical interference and generated event lifetimes are rechecked by
// the ordinary planner/GSS pipeline and the combined native qualification.
bool protectedVectorScope(scope::ScopeOp stage, bool probability) {
  auto core =
      stage->getAttrOfType<TCoreTypeAttr>(kPipelinedLoopCoreTypeAttrName);
  if (!core || core.getTcoretype() != TCoreType::VECTOR)
    return false;
  bool nestedEvent = false;
  stage.walk([&](Operation *op) {
    nestedEvent |= isa<SyncBlockOp, SyncBlockSetOp, SyncBlockWaitOp, SetFlagOp,
                       WaitFlagOp, PipeBarrierOp>(op);
  });
  if (nestedEvent)
    return false;
  Operation *before = stage->getPrevNode();
  while (before && isa<AnchorOp>(before))
    before = before->getPrevNode();
  if (!vectorEvent(before, false, PIPE::PIPE_FIX, PIPE::PIPE_V))
    return false;
  Operation *after = stage->getNextNode();
  while (after && isa<AnchorOp>(after))
    after = after->getNextNode();
  if (probability) {
    if (!vectorEvent(after, true, PIPE::PIPE_MTE3, PIPE::PIPE_MTE1))
      return false;
    after = after->getNextNode();
    while (after && isa<AnchorOp>(after))
      after = after->getNextNode();
  }
  return vectorEvent(after, true, PIPE::PIPE_V, PIPE::PIPE_FIX);
}

bool sameSlice(memref::SubViewOp a, memref::SubViewOp b) {
  return a && b && a.getMixedOffsets() == b.getMixedOffsets() &&
         a.getMixedSizes() == b.getMixedSizes() &&
         a.getMixedStrides() == b.getMixedStrides();
}

bool fullRowChunks(memref::SubViewOp view, func::FuncOp function, int64_t hd) {
  if (!view || view.getMixedOffsets().size() != 2 ||
      getConstantIntValue(view.getMixedSizes()[0]) != 1 ||
      getConstantIntValue(view.getMixedSizes()[1]) != 64 ||
      getConstantIntValue(view.getMixedStrides()[0]) != 1 ||
      getConstantIntValue(view.getMixedStrides()[1]) != 1)
    return false;
  auto row = dyn_cast<Value>(view.getMixedOffsets()[0]);
  auto argument = row ? dyn_cast<BlockArgument>(row) : BlockArgument();
  auto loop = argument
                  ? dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp())
                  : scf::ForOp();
  if (!loop || argument != loop.getInductionVar() ||
      loop->getParentOp() != function || integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != 64 || integer(loop.getStep()) != 1 ||
      loop.getNumResults())
    return false;
  if (getConstantIntValue(view.getMixedOffsets()[1]) == 0)
    return hd == 64;
  auto column = dyn_cast<Value>(view.getMixedOffsets()[1]);
  auto columnArg = column ? dyn_cast<BlockArgument>(column) : BlockArgument();
  auto columns = columnArg
                     ? dyn_cast<scf::ForOp>(columnArg.getOwner()->getParentOp())
                     : scf::ForOp();
  return columns && columnArg == columns.getInductionVar() &&
         columns->getParentOp() == loop &&
         integer(columns.getLowerBound()) == 0 &&
         integer(columns.getUpperBound()) == hd &&
         integer(columns.getStep()) == 64 && !columns.getNumResults();
}

struct Update {
  func::FuncOp function;
  vector::TransferReadOp oldRead, pvRead, alphaRead;
  vector::TransferWriteOp store;
  arith::MulFOp multiply;
  arith::AddFOp add;
  unsigned oldIndex, pvIndex, alphaIndex, outputIndex;
};

std::optional<Update> matchUpdate(func::FuncOp function, int64_t hd) {
  if (!simpleFunction(function) || function.getNumArguments() != 4)
    return std::nullopt;
  SmallVector<vector::TransferWriteOp> stores;
  function.walk(
      [&](vector::TransferWriteOp store) { stores.push_back(store); });
  if (stores.size() != 1)
    return std::nullopt;
  auto store = stores.front();
  auto output = rootArgument(store.getSource(), function);
  auto add = store.getVector().getDefiningOp<arith::AddFOp>();
  if (!output || !isUB(output.getType(), {64, hd}, 32) || !add ||
      !isChunk(store.getVector().getType(), 32) ||
      !fullRowChunks(store.getSource().getDefiningOp<memref::SubViewOp>(),
                     function, hd))
    return std::nullopt;
  for (unsigned side : {0u, 1u}) {
    auto multiply = add->getOperand(side).getDefiningOp<arith::MulFOp>();
    auto pvRead =
        add->getOperand(1 - side).getDefiningOp<vector::TransferReadOp>();
    if (!multiply || !pvRead)
      continue;
    auto pv = rootArgument(pvRead.getSource(), function);
    for (unsigned productSide : {0u, 1u}) {
      auto oldRead = multiply->getOperand(productSide)
                         .getDefiningOp<vector::TransferReadOp>();
      auto alphaRead = multiply->getOperand(1 - productSide)
                           .getDefiningOp<vector::TransferReadOp>();
      if (!oldRead || !alphaRead)
        continue;
      auto old = rootArgument(oldRead.getSource(), function);
      auto alpha = rootArgument(alphaRead.getSource(), function);
      auto oldView = oldRead.getSource().getDefiningOp<memref::SubViewOp>();
      auto alphaView = alphaRead.getSource().getDefiningOp<memref::SubViewOp>();
      if (!old || !alpha || !pv || !isUB(old.getType(), {64, hd}, 32) ||
          !isChunk(oldRead.getResult().getType(), 32) ||
          !isChunk(pvRead.getResult().getType(), 32) ||
          !isChunk(alphaRead.getResult().getType(), 32) ||
          !isUB(pv.getType(), {64, hd}, 32) ||
          !isUB(alpha.getType(), {64}, 32) ||
          !sameSlice(oldView,
                     pvRead.getSource().getDefiningOp<memref::SubViewOp>()) ||
          !sameSlice(oldView,
                     store.getSource().getDefiningOp<memref::SubViewOp>()) ||
          !alphaView || alphaView.getMixedOffsets().size() != 1 ||
          alphaView.getMixedOffsets()[0] != oldView.getMixedOffsets()[0] ||
          getConstantIntValue(alphaView.getMixedSizes()[0]) != 1)
        continue;
      llvm::DenseSet<unsigned> indices{old.getArgNumber(), pv.getArgNumber(),
                                       alpha.getArgNumber(),
                                       output.getArgNumber()};
      if (indices.size() != 4)
        continue;
      unsigned reads = 0;
      function.walk([&](vector::TransferReadOp) { ++reads; });
      if (reads != 3 || !multiply->getResult(0).hasOneUse() ||
          !add->getResult(0).hasOneUse())
        continue;
      return Update{function,
                    oldRead,
                    pvRead,
                    alphaRead,
                    store,
                    multiply,
                    add,
                    old.getArgNumber(),
                    pv.getArgNumber(),
                    alpha.getArgNumber(),
                    output.getArgNumber()};
    }
  }
  return std::nullopt;
}

struct Normalizer {
  func::FuncOp function;
  vector::TransferReadOp oldRead, denominatorRead;
  arith::DivFOp divide;
  unsigned oldIndex;
  SmallVector<unsigned> reads, writes;
};

std::optional<Normalizer> matchNormalizer(func::FuncOp function, int64_t hd) {
  if (!simpleFunction(function) || function.getNumArguments() != 6)
    return std::nullopt;
  SmallVector<arith::DivFOp> divides;
  function.walk([&](arith::DivFOp op) { divides.push_back(op); });
  if (divides.size() != 1)
    return std::nullopt;
  auto divide = divides.front();
  auto oldRead = divide.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto denominator = divide.getRhs().getDefiningOp<vector::TransferReadOp>();
  if (!oldRead || !denominator || !divide.getResult().hasOneUse())
    return std::nullopt;
  auto old = rootArgument(oldRead.getSource(), function);
  auto sum = rootArgument(denominator.getSource(), function);
  auto oldView = oldRead.getSource().getDefiningOp<memref::SubViewOp>();
  auto sumView = denominator.getSource().getDefiningOp<memref::SubViewOp>();
  if (!old || !sum || !isUB(old.getType(), {64, hd}, 32) ||
      !isChunk(oldRead.getResult().getType(), 32) ||
      !isChunk(denominator.getResult().getType(), 32) ||
      !isUB(sum.getType(), {64}, 32) || !fullRowChunks(oldView, function, hd) ||
      !sumView || sumView.getMixedOffsets().size() != 1 ||
      sumView.getMixedOffsets()[0] != oldView.getMixedOffsets()[0])
    return std::nullopt;
  auto conversion =
      dyn_cast<arith::TruncFOp>(*divide.getResult().getUsers().begin());
  if (!conversion || !conversion.getResult().hasOneUse())
    return std::nullopt;
  auto output = dyn_cast<vector::TransferWriteOp>(
      *conversion.getResult().getUsers().begin());
  if (!output || !isChunk(output.getVector().getType(), 16) ||
      !sameSlice(oldView,
                 output.getSource().getDefiningOp<memref::SubViewOp>()))
    return std::nullopt;
  auto target = rootArgument(output.getSource(), function);
  if (!target || !isUB(target.getType(), {64, hd}, 16))
    return std::nullopt;
  llvm::DenseSet<unsigned> reads, writes;
  function.walk([&](vector::TransferReadOp read) {
    reads.insert(rootArgument(read.getSource(), function).getArgNumber());
  });
  function.walk([&](vector::TransferWriteOp write) {
    writes.insert(rootArgument(write.getSource(), function).getArgNumber());
  });
  if (writes.size() != 2 ||
      reads.size() + writes.size() != function.getNumArguments() ||
      llvm::any_of(writes, [&](unsigned i) { return reads.contains(i); }))
    return std::nullopt;
  Normalizer result{function,           oldRead, denominator, divide,
                    old.getArgNumber(), {},      {}};
  llvm::append_range(result.reads, reads);
  llvm::append_range(result.writes, writes);
  llvm::sort(result.reads);
  llvm::sort(result.writes);
  return result;
}

FloatAttr splat(Value value) {
  Attribute constant;
  if (!matchPattern(value, m_Constant(&constant)))
    return {};
  if (auto scalar = dyn_cast<FloatAttr>(constant))
    return scalar;
  auto dense = dyn_cast<DenseElementsAttr>(constant);
  if (!dense || !dense.isSplat() || !dense.getElementType().isF32())
    return {};
  return FloatAttr::get(dense.getElementType(), dense.getSplatValue<APFloat>());
}

bool readOnlyArgument(func::FuncOp function, unsigned index) {
  SmallVector<Value> values{function.getArgument(index)};
  while (!values.empty()) {
    Value value = values.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(use.getOwner())) {
        if (read.getSource() != value)
          return false;
      } else if (auto view = dyn_cast<memref::SubViewOp>(use.getOwner())) {
        if (view.getSource() != value)
          return false;
        values.push_back(view.getResult());
      } else {
        return false;
      }
    }
  }
  return true;
}

std::optional<FloatAttr> fillValue(func::FuncOp function, Type type,
                                   int64_t hd) {
  if (!simpleFunction(function) || !function->hasAttr("hfusion.has_fill") ||
      function.getNumArguments() != 1 ||
      function.getArgument(0).getType() != type)
    return std::nullopt;
  unsigned reads = 0;
  SmallVector<vector::TransferWriteOp> stores;
  function.walk([&](vector::TransferReadOp) { ++reads; });
  function.walk(
      [&](vector::TransferWriteOp write) { stores.push_back(write); });
  if (reads || stores.size() != 1)
    return std::nullopt;
  auto store = stores.front();
  FloatAttr constant = splat(store.getVector());
  if (!constant || !constant.getType().isF32())
    return std::nullopt;
  if (isUB(type, {64}, 32)) {
    auto vector = dyn_cast<VectorType>(store.getVector().getType());
    if (store.getSource() != function.getArgument(0) || !vector ||
        vector.getShape() != ArrayRef<int64_t>{64})
      return std::nullopt;
  } else if (!isUB(type, {64, hd}, 32) ||
             !isChunk(store.getVector().getType(), 32) ||
             !fullRowChunks(
                 store.getSource().getDefiningOp<memref::SubViewOp>(), function,
                 hd)) {
    return std::nullopt;
  }
  return constant;
}

struct Initialization {
  Operation *writer;
  FloatAttr value;
};

std::optional<Initialization> initialization(Value memory, Operation *consumer,
                                             ModuleOp module, int64_t hd,
                                             bool allowCopy = true) {
  auto allocation = memory.getDefiningOp<memref::AllocOp>();
  if (!allocation)
    return std::nullopt;
  DominanceInfo dominance(module);
  SmallVector<Initialization> writers;
  for (OpOperand &use : memory.getUses()) {
    Operation *op = use.getOwner();
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
      auto value = fillValue(function, memory.getType(), hd);
      if (!value || call.getNumOperands() != 1 || use.getOperandNumber() != 0 ||
          !dominance.dominates(op, consumer))
        return std::nullopt;
      writers.push_back({op, *value});
    } else if (auto copy = dyn_cast<CopyOp>(op)) {
      if (copy.getDst() != memory) {
        // A top-level initializer is erased by specialization. Its contents
        // must not escape to another reader. Recursive constant-source proofs
        // retain that source's writer, so other source copies remain valid.
        if (allowCopy)
          return std::nullopt;
        continue;
      }
      if (!allowCopy || !dominance.dominates(op, consumer))
        return std::nullopt;
      auto source = initialization(copy.getSrc(), op, module, hd, false);
      if (!source)
        return std::nullopt;
      writers.push_back({op, source->value});
    } else if (op != consumer && !isa<annotation::MarkOp>(op)) {
      return std::nullopt;
    }
  }
  if (writers.size() != 1)
    return std::nullopt;
  return writers.front();
}

void copyCallAttrs(func::CallOp source, func::CallOp destination) {
  for (NamedAttribute attr : source->getAttrs())
    if (attr.getName().getValue() != "callee")
      destination->setAttr(attr.getName(), attr.getValue());
}

func::FuncOp cloneFunction(func::FuncOp source, ModuleOp module,
                           StringRef kind) {
  auto function = cast<func::FuncOp>(source->clone());
  function.setSymName((llvm::Twine("__cvp_attention_") + kind).str());
  function->setAttr(kBoundaryKind, StringAttr::get(module.getContext(), kind));
  SymbolTable(module).insert(function);
  return function;
}

void replaceInitialReads(
    func::FuncOp function,
    ArrayRef<std::pair<unsigned, FloatAttr>> initialValues) {
  SmallVector<vector::TransferReadOp> reads;
  function.walk([&](vector::TransferReadOp read) { reads.push_back(read); });
  for (vector::TransferReadOp read : reads) {
    auto root = rootArgument(read.getSource(), function);
    if (!root)
      continue;
    auto initial = llvm::find_if(initialValues, [&](const auto &item) {
      return item.first == root.getArgNumber();
    });
    if (initial == initialValues.end())
      continue;
    OpBuilder builder(read);
    auto vector = cast<VectorType>(read.getResult().getType());
    Value value = builder.create<arith::ConstantOp>(
        read.getLoc(),
        DenseElementsAttr::get(vector, initial->second.getValue()));
    read.getResult().replaceAllUsesWith(value);
    read.erase();
  }
  // Only address calculations made dead by the replaced reads can disappear.
  bool changed;
  do {
    changed = false;
    SmallVector<memref::SubViewOp> dead;
    function.walk([&](memref::SubViewOp view) {
      if (view.getResult().use_empty())
        dead.push_back(view);
    });
    for (auto view : llvm::reverse(dead)) {
      view.erase();
      changed = true;
    }
  } while (changed);
  SmallVector<unsigned> indices;
  for (auto [index, initial] : initialValues) {
    (void)initial;
    indices.push_back(index);
  }
  llvm::sort(indices, std::greater<unsigned>());
  for (unsigned index : indices) {
    assert(function.getArgument(index).use_empty());
    function.eraseArgument(index);
  }
}

func::FuncOp makeLastFunction(const Normalizer &normal, Update &update,
                              ModuleOp module) {
  auto function = cloneFunction(normal.function, module, "last_output");
  auto matched = matchNormalizer(
      function,
      cast<MemRefType>(update.function.getArgument(update.oldIndex).getType())
          .getDimSize(1));
  // The clone retains the same signature and complete row geometry.
  assert(matched);
  auto normalCopy = *matched;
  unsigned alphaIndex = function.getNumArguments();
  function.insertArgument(
      alphaIndex, update.function.getArgument(update.alphaIndex).getType(),
      DictionaryAttr(), function.getLoc());
  unsigned pvIndex = function.getNumArguments();
  function.insertArgument(pvIndex,
                          update.function.getArgument(update.pvIndex).getType(),
                          DictionaryAttr(), function.getLoc());
  auto oldView =
      normalCopy.oldRead.getSource().getDefiningOp<memref::SubViewOp>();
  OpBuilder builder(normalCopy.denominatorRead);
  auto row = oldView.getMixedOffsets()[0];
  SmallVector<OpFoldResult> offsets{row}, sizes{builder.getIndexAttr(1)},
      strides{builder.getIndexAttr(1)};
  auto alphaView = builder.create<memref::SubViewOp>(
      normalCopy.denominatorRead.getLoc(), function.getArgument(alphaIndex),
      offsets, sizes, strides);
  IRMapping mapping;
  mapping.map(normalCopy.denominatorRead.getSource(), alphaView.getResult());
  auto alphaRead = cast<vector::TransferReadOp>(
      builder.clone(*normalCopy.denominatorRead, mapping));
  builder.setInsertionPoint(normalCopy.divide);
  mapping.clear();
  mapping.map(oldView.getSource(), function.getArgument(pvIndex));
  auto pvView = cast<memref::SubViewOp>(builder.clone(*oldView, mapping));
  mapping.clear();
  mapping.map(normalCopy.oldRead.getSource(), pvView.getResult());
  auto pvRead =
      cast<vector::TransferReadOp>(builder.clone(*normalCopy.oldRead, mapping));
  mapping.clear();
  mapping.map(update.oldRead.getResult(), normalCopy.oldRead.getResult());
  mapping.map(update.alphaRead.getResult(), alphaRead.getResult());
  auto multiply = cast<arith::MulFOp>(builder.clone(*update.multiply, mapping));
  mapping.map(update.multiply.getResult(), multiply.getResult());
  mapping.map(update.pvRead.getResult(), pvRead.getResult());
  auto add = cast<arith::AddFOp>(builder.clone(*update.add, mapping));
  normalCopy.divide->setOperand(0, add.getResult());
  return function;
}

struct OutputSite {
  func::FuncOp caller;
  func::CallOp updateCall, normalCall;
  Update update;
  std::optional<Normalizer> normal;
  scf::ForOp loop;
  scope::ScopeOp scope;
  unsigned stateIndex;
  int64_t tiles;
  std::optional<Initialization> initial;
  SmallVector<memref::AllocOp> finalAllocations;
};

std::optional<OutputSite> matchOutputSite(ModuleOp module, func::FuncOp caller,
                                          func::CallOp call, int64_t hd,
                                          int64_t bn, int64_t keys, bool first,
                                          bool last) {
  auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
  auto update = matchUpdate(function, hd);
  if (!update || !simpleUniqueCall(call, function, module))
    return std::nullopt;
  auto scope = dyn_cast<scope::ScopeOp>(call->getParentOp());
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp();
  auto phase = scope ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  if (!loop || !phase || phase.getInt() != 0 ||
      !protectedVectorScope(scope, false) ||
      integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != keys || integer(loop.getStep()) != bn ||
      keys < bn || keys % bn || scope->getNumResults() != 1)
    return std::nullopt;
  for (Operation &operation : scope->getRegion(0).front().without_terminator())
    if (&operation != call.getOperation() &&
        !isa<memref::AllocOp, annotation::MarkOp, AnchorOp>(operation) &&
        (operation.getNumRegions() || !isMemoryEffectFree(&operation)))
      return std::nullopt;
  auto old = dyn_cast<BlockArgument>(call.getOperand(update->oldIndex));
  if (!old || old.getOwner() != loop.getBody() || old.getArgNumber() == 0)
    return std::nullopt;
  unsigned index = old.getArgNumber() - 1;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto ret = cast<scope::ReturnOp>(scope->getRegion(0).front().getTerminator());
  if (yield.getOperand(index) != scope->getResult(0) ||
      ret.getNumOperands() != 1 ||
      ret.getOperand(0) != call.getOperand(update->outputIndex) ||
      !scope->getResult(0).hasOneUse())
    return std::nullopt;
  for (OpOperand &use : old.getUses())
    if (use.getOwner() != call && use.getOwner() != ret)
      return std::nullopt;
  OutputSite site{caller,       call,         {},    *update,
                  std::nullopt, loop,         scope, index,
                  keys / bn,    std::nullopt, {}};
  if (first) {
    site.initial = initialization(loop.getInitArgs()[index], loop, module, hd);
    if (!site.initial || !site.initial->value.getValue().isPosZero() ||
        !readOnlyArgument(function, update->oldIndex))
      return std::nullopt;
  }
  if (!last)
    return site;
  if (!loop.getResult(index).hasOneUse())
    return std::nullopt;
  auto normalCall =
      dyn_cast<func::CallOp>(*loop.getResult(index).getUsers().begin());
  if (!normalCall || normalCall->getBlock() != loop->getBlock() ||
      !loop->isBeforeInBlock(normalCall))
    return std::nullopt;
  auto normalFunction =
      module.lookupSymbol<func::FuncOp>(normalCall.getCallee());
  auto normal = matchNormalizer(normalFunction, hd);
  if (!normal || !simpleUniqueCall(normalCall, normalFunction, module) ||
      normalCall.getOperand(normal->oldIndex) != loop.getResult(index) ||
      (first && !readOnlyArgument(normalFunction, normal->oldIndex)))
    return std::nullopt;
  for (unsigned input : normal->reads) {
    auto result = dyn_cast<OpResult>(normalCall.getOperand(input));
    if (!result || result.getOwner() != loop)
      return std::nullopt;
  }
  llvm::SmallPtrSet<Operation *, 4> allocations;
  for (unsigned output : normal->writes) {
    auto allocation =
        normalCall.getOperand(output).getDefiningOp<memref::AllocOp>();
    if (!allocation || allocation->getBlock() != loop->getBlock() ||
        !loop->isBeforeInBlock(allocation) ||
        !allocation->isBeforeInBlock(normalCall) ||
        !allocation.getDynamicSizes().empty())
      return std::nullopt;
    if (allocations.insert(allocation).second)
      site.finalAllocations.push_back(allocation);
  }
  for (Operation *op = loop->getNextNode(); op != normalCall;
       op = op->getNextNode())
    if (!isa<memref::AllocOp, annotation::MarkOp, AnchorOp>(op) &&
        !isMemoryEffectFree(op))
      return std::nullopt;
  site.normalCall = normalCall;
  site.normal = *normal;
  return site;
}

func::CallOp callLike(OpBuilder &builder, func::CallOp original,
                      func::FuncOp function, ValueRange values) {
  auto call = builder.create<func::CallOp>(
      original.getLoc(), function.getSymName(), TypeRange{}, values);
  copyCallAttrs(original, call);
  return call;
}

void rewriteOutput(OutputSite &site, ModuleOp module, bool first, bool last) {
  OpBuilder builder(site.updateCall);
  func::FuncOp firstFunction, lastFunction;
  if (first) {
    firstFunction = cloneFunction(site.update.function, module, "first_output");
    replaceInitialReads(firstFunction,
                        {{site.update.oldIndex, site.initial->value}});
  }
  if (last) {
    lastFunction = makeLastFunction(*site.normal, site.update, module);
    if (first && site.tiles == 1) {
      replaceInitialReads(lastFunction,
                          {{site.normal->oldIndex, site.initial->value}});
      lastFunction->setAttr(kBoundaryKind,
                            builder.getStringAttr("first_last_output"));
    }
    for (memref::AllocOp allocation : site.finalAllocations) {
      allocation->moveBefore(site.loop);
      SmallVector<Operation *> marks;
      for (Operation *user : allocation.getResult().getUsers())
        if (isa<annotation::MarkOp>(user))
          marks.push_back(user);
      for (Operation *mark : marks)
        mark->moveBefore(site.loop);
    }
  }
  auto emitOrdinaryOrFirst = [&](OpBuilder &where) {
    SmallVector<Value> ordinary(site.updateCall.getOperands());
    if (!first) {
      callLike(where, site.updateCall, site.update.function, ordinary);
      return;
    }
    SmallVector<Value> initialized = ordinary;
    initialized.erase(initialized.begin() + site.update.oldIndex);
    if (site.tiles == 1) {
      callLike(where, site.updateCall, firstFunction, initialized);
      return;
    }
    Value condition = where.create<arith::CmpIOp>(
        site.updateCall.getLoc(), arith::CmpIPredicate::eq,
        site.loop.getInductionVar(), site.loop.getLowerBound());
    auto branch =
        where.create<scf::IfOp>(site.updateCall.getLoc(), condition, true);
    OpBuilder thenBuilder = branch.getThenBodyBuilder();
    callLike(thenBuilder, site.updateCall, firstFunction, initialized);
    OpBuilder elseBuilder = branch.getElseBodyBuilder();
    callLike(elseBuilder, site.updateCall, site.update.function, ordinary);
  };
  if (last) {
    auto yield = cast<scf::YieldOp>(site.loop.getBody()->getTerminator());
    SmallVector<Value> arguments(site.normalCall.getOperands());
    for (unsigned input : site.normal->reads) {
      auto result = cast<OpResult>(arguments[input]);
      arguments[input] = input == site.normal->oldIndex
                             ? site.updateCall.getOperand(site.update.oldIndex)
                             : yield.getOperand(result.getResultNumber());
    }
    arguments.push_back(site.updateCall.getOperand(site.update.alphaIndex));
    arguments.push_back(site.updateCall.getOperand(site.update.pvIndex));
    if (first && site.tiles == 1)
      arguments.erase(arguments.begin() + site.normal->oldIndex);
    Value lastIndex = builder.create<arith::SubIOp>(site.updateCall.getLoc(),
                                                    site.loop.getUpperBound(),
                                                    site.loop.getStep());
    Value condition = builder.create<arith::CmpIOp>(
        site.updateCall.getLoc(), arith::CmpIPredicate::eq,
        site.loop.getInductionVar(), lastIndex);
    auto branch =
        builder.create<scf::IfOp>(site.updateCall.getLoc(), condition, true);
    OpBuilder thenBuilder = branch.getThenBodyBuilder();
    callLike(thenBuilder, site.normalCall, lastFunction, arguments);
    OpBuilder elseBuilder = branch.getElseBodyBuilder();
    emitOrdinaryOrFirst(elseBuilder);
    // The final FP32 buffer pointer may still flow through scope.return and
    // scf.yield, but its contents have no reader: the unique final normalizer
    // is removed and this is the last logical iteration. Keep the existing
    // pointer recurrence rather than introduce a new memref alias/select.
    site.normalCall.erase();
  } else {
    emitOrdinaryOrFirst(builder);
  }
  site.updateCall.erase();
  if (first)
    site.initial->writer->erase();
}

struct StateSite {
  func::CallOp combineCall, stateCall;
  func::FuncOp combineFunction, stateFunction;
  scf::ForOp loop;
  unsigned oldMaxInCombine, oldMaxInState, oldSumInState;
  Initialization maximum, sum;
};

struct InitialStateCall {
  func::CallOp call;
  func::FuncOp function;
  SmallVector<std::pair<unsigned, FloatAttr>, 2> values;
};

struct InitialStateGroup {
  scf::ForOp loop;
  Initialization maximum, sum;
  SmallVector<InitialStateCall, 2> calls;
};

// A uniform initializer may replace only complete, unmasked reads. In
// particular, a padded transfer_read could otherwise return its padding value
// outside the initialized allocation, rather than the initializer's value.
bool completeStateReads(func::FuncOp function, unsigned index) {
  if (!readOnlyArgument(function, index))
    return false;
  Value argument = function.getArgument(index);
  bool found = false;
  for (OpOperand &use : argument.getUses()) {
    auto read = dyn_cast<vector::TransferReadOp>(use.getOwner());
    auto vector = read ? dyn_cast<VectorType>(read.getType()) : VectorType();
    auto bounds =
        read ? read->getAttrOfType<ArrayAttr>("in_bounds") : ArrayAttr();
    if (!read || read.getSource() != argument || !vector ||
        vector.isScalable() || vector.getShape() != ArrayRef<int64_t>{64} ||
        !vector.getElementType().isF32() || read.getMask() ||
        !zeroIndices(read.getIndices()) ||
        !read.getPermutationMap().isIdentity() || !bounds ||
        bounds.size() != 1 || !isa<BoolAttr>(bounds[0]) ||
        !cast<BoolAttr>(bounds[0]).getValue())
      return false;
    found = true;
  }
  return found;
}

std::optional<InitialStateGroup>
matchInitialStateGroup(ModuleOp module, scf::ForOp loop, int64_t keys,
                       int64_t bn, int64_t maxSlot, int64_t sumSlot) {
  if (integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != keys || integer(loop.getStep()) != bn ||
      maxSlot < 0 || sumSlot < 0 || maxSlot == sumSlot ||
      static_cast<uint64_t>(maxSlot) >= loop.getInitArgs().size() ||
      static_cast<uint64_t>(sumSlot) >= loop.getInitArgs().size())
    return std::nullopt;
  Value maximum = loop.getRegionIterArg(maxSlot);
  Value sum = loop.getRegionIterArg(sumSlot);
  if (!isUB(maximum.getType(), {64}, 32) || !isUB(sum.getType(), {64}, 32) ||
      maximum.use_empty() || sum.use_empty())
    return std::nullopt;
  auto maxInit = initialization(loop.getInitArgs()[maxSlot], loop, module, 64);
  auto sumInit = initialization(loop.getInitArgs()[sumSlot], loop, module, 64);
  if (!maxInit || !maxInit->value.getValue().isInfinity() ||
      !maxInit->value.getValue().isNegative() || !sumInit ||
      sumInit->value.getValueAsDouble() != 1.0 ||
      maxInit->writer == sumInit->writer)
    return std::nullopt;
  InitialStateGroup group{loop, *maxInit, *sumInit, {}};
  scope::ScopeOp commonScope;
  for (auto [state, initial] :
       {std::pair<Value, FloatAttr>{maximum, maxInit->value},
        std::pair<Value, FloatAttr>{sum, sumInit->value}}) {
    for (OpOperand &use : state.getUses()) {
      auto call = dyn_cast<func::CallOp>(use.getOwner());
      auto scope = call ? dyn_cast<scope::ScopeOp>(call->getParentOp())
                        : scope::ScopeOp();
      auto phase = scope
                       ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                       : IntegerAttr();
      if (!scope || scope->getParentOp() != loop || !phase ||
          phase.getInt() != 2 || !protectedVectorScope(scope, true) ||
          (commonScope && commonScope != scope))
        return std::nullopt;
      commonScope = scope;
      auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
      unsigned index = use.getOperandNumber();
      if (!simpleFunction(function, bn) ||
          index >= function.getNumArguments() ||
          function->getAttr("hivm.func_core_type") !=
              TFuncCoreTypeAttr::get(module.getContext(), TFuncCoreType::AIV) ||
          !simpleUniqueCall(call, function, module) ||
          !isUB(function.getArgument(index).getType(), {64}, 32) ||
          !completeStateReads(function, index))
        return std::nullopt;
      auto site = llvm::find_if(group.calls, [&](const InitialStateCall &item) {
        return item.call == call;
      });
      if (site == group.calls.end()) {
        group.calls.push_back({call, function, {}});
        site = std::prev(group.calls.end());
      }
      site->values.emplace_back(index, initial);
    }
  }
  llvm::sort(group.calls,
             [](const InitialStateCall &first, const InitialStateCall &second) {
               return first.call->isBeforeInBlock(second.call);
             });
  return group;
}

std::optional<StateSite> matchState(ModuleOp module, func::FuncOp caller,
                                    func::CallOp stateCall, int64_t keys,
                                    int64_t bn) {
  auto function = module.lookupSymbol<func::FuncOp>(stateCall.getCallee());
  if (!simpleFunction(function) ||
      !simpleUniqueCall(stateCall, function, module) ||
      !llvm::all_of(function.getArgumentTypes(),
                    [](Type t) { return isUB(t, {64}, 32); }))
    return std::nullopt;
  SmallVector<math::ExpOp> exps;
  SmallVector<arith::MulFOp> multiplies;
  function.walk([&](math::ExpOp op) { exps.push_back(op); });
  function.walk([&](arith::MulFOp op) { multiplies.push_back(op); });
  if (exps.size() != 1 || multiplies.size() != 1)
    return std::nullopt;
  auto sub = exps.front().getOperand().getDefiningOp<arith::SubFOp>();
  auto oldMaxRead = sub ? sub.getLhs().getDefiningOp<vector::TransferReadOp>()
                        : vector::TransferReadOp();
  auto newMaxRead = sub ? sub.getRhs().getDefiningOp<vector::TransferReadOp>()
                        : vector::TransferReadOp();
  if (!oldMaxRead || !newMaxRead)
    return std::nullopt;
  auto oldMax = rootArgument(oldMaxRead.getSource(), function);
  auto newMax = rootArgument(newMaxRead.getSource(), function);
  auto multiply = multiplies.front();
  Value sumValue =
      multiply.getLhs() == exps.front().getResult()   ? multiply.getRhs()
      : multiply.getRhs() == exps.front().getResult() ? multiply.getLhs()
                                                      : Value();
  auto sumRead = sumValue ? sumValue.getDefiningOp<vector::TransferReadOp>()
                          : vector::TransferReadOp();
  auto oldSum =
      sumRead ? rootArgument(sumRead.getSource(), function) : BlockArgument();
  if (!oldMax || !newMax || !oldSum || oldMax == oldSum ||
      !readOnlyArgument(function, oldMax.getArgNumber()) ||
      !readOnlyArgument(function, oldSum.getArgNumber()))
    return std::nullopt;
  auto scope = dyn_cast<scope::ScopeOp>(stateCall->getParentOp());
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp();
  auto phase = scope ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  if (!loop || !phase || phase.getInt() != 2 ||
      !protectedVectorScope(scope, true) ||
      integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != keys || integer(loop.getStep()) != bn)
    return std::nullopt;
  auto maximumArg =
      dyn_cast<BlockArgument>(stateCall.getOperand(oldMax.getArgNumber()));
  auto sumArg =
      dyn_cast<BlockArgument>(stateCall.getOperand(oldSum.getArgNumber()));
  if (!maximumArg || !sumArg || maximumArg.getOwner() != loop.getBody() ||
      sumArg.getOwner() != loop.getBody() || !maximumArg.getArgNumber() ||
      !sumArg.getArgNumber())
    return std::nullopt;
  auto maxInit = initialization(
      loop.getInitArgs()[maximumArg.getArgNumber() - 1], loop, module, 64);
  auto sumInit = initialization(loop.getInitArgs()[sumArg.getArgNumber() - 1],
                                loop, module, 64);
  if (!maxInit || !maxInit->value.getValue().isInfinity() ||
      !maxInit->value.getValue().isNegative() || !sumInit ||
      sumInit->value.getValueAsDouble() != 1.0)
    return std::nullopt;
  func::CallOp combineCall;
  func::FuncOp combineFunction;
  unsigned combineOld = 0;
  for (func::CallOp call : scope->getRegion(0).front().getOps<func::CallOp>()) {
    auto callee = module.lookupSymbol<func::FuncOp>(call.getCallee());
    if (call == stateCall || !simpleFunction(callee) ||
        !simpleUniqueCall(call, callee, module) ||
        callee.getNumArguments() != 3)
      continue;
    SmallVector<arith::MaximumFOp> maxima;
    SmallVector<vector::TransferWriteOp> stores;
    callee.walk([&](arith::MaximumFOp op) { maxima.push_back(op); });
    callee.walk([&](vector::TransferWriteOp op) { stores.push_back(op); });
    if (maxima.size() != 1 || stores.size() != 1 ||
        stores.front().getVector() != maxima.front().getResult())
      continue;
    auto destination = rootArgument(stores.front().getSource(), callee);
    if (!destination || call.getOperand(destination.getArgNumber()) !=
                            stateCall.getOperand(newMax.getArgNumber()))
      continue;
    for (Value operand : maxima.front()->getOperands()) {
      auto read = operand.getDefiningOp<vector::TransferReadOp>();
      auto root =
          read ? rootArgument(read.getSource(), callee) : BlockArgument();
      if (root && call.getOperand(root.getArgNumber()) == maximumArg &&
          readOnlyArgument(callee, root.getArgNumber())) {
        if (combineCall)
          return std::nullopt;
        combineCall = call;
        combineFunction = callee;
        combineOld = root.getArgNumber();
      }
    }
  }
  if (!combineCall || !combineCall->isBeforeInBlock(stateCall))
    return std::nullopt;
  for (Operation *user : maximumArg.getUsers())
    if (user != combineCall && user != stateCall)
      return std::nullopt;
  for (Operation *user : sumArg.getUsers())
    if (user != stateCall)
      return std::nullopt;
  return StateSite{
      combineCall, stateCall,  combineFunction,       function,
      loop,        combineOld, oldMax.getArgNumber(), oldSum.getArgNumber(),
      *maxInit,    *sumInit};
}

void specializeCall(func::CallOp call, func::FuncOp original,
                    func::FuncOp first, ArrayRef<unsigned> removed,
                    scf::ForOp loop) {
  OpBuilder builder(call);
  Value condition = builder.create<arith::CmpIOp>(
      call.getLoc(), arith::CmpIPredicate::eq, loop.getInductionVar(),
      loop.getLowerBound());
  auto branch = builder.create<scf::IfOp>(call.getLoc(), condition, true);
  SmallVector<Value> firstOperands(call.getOperands());
  SmallVector<unsigned> indices(removed);
  llvm::sort(indices, std::greater<unsigned>());
  for (unsigned index : indices)
    firstOperands.erase(firstOperands.begin() + index);
  OpBuilder thenBuilder = branch.getThenBodyBuilder();
  callLike(thenBuilder, call, first, firstOperands);
  OpBuilder elseBuilder = branch.getElseBodyBuilder();
  callLike(elseBuilder, call, original, call.getOperands());
  call.erase();
}

void rewriteState(StateSite &site, ModuleOp module) {
  auto firstMaximum =
      cloneFunction(site.combineFunction, module, "first_maximum");
  replaceInitialReads(firstMaximum,
                      {{site.oldMaxInCombine, site.maximum.value}});
  auto firstState = cloneFunction(site.stateFunction, module, "first_state");
  replaceInitialReads(firstState, {{site.oldMaxInState, site.maximum.value},
                                   {site.oldSumInState, site.sum.value}});
  specializeCall(site.combineCall, site.combineFunction, firstMaximum,
                 {site.oldMaxInCombine}, site.loop);
  specializeCall(site.stateCall, site.stateFunction, firstState,
                 {site.oldMaxInState, site.oldSumInState}, site.loop);
  site.maximum.writer->erase();
  site.sum.writer->erase();
}

void rewriteInitialStateGroup(InitialStateGroup &group, ModuleOp module) {
  for (InitialStateCall &site : group.calls) {
    auto first = cloneFunction(site.function, module, "first_state");
    replaceInitialReads(first, site.values);
    SmallVector<unsigned, 2> removed;
    for (auto [index, value] : site.values) {
      (void)value;
      removed.push_back(index);
    }
    specializeCall(site.call, site.function, first, removed, group.loop);
  }
  group.maximum.writer->erase();
  group.sum.writer->erase();
}

// The continuous query stream uses local-key boundaries. Its output storage
// has already been made persistent before bufferization. Match that distinct
// graph without weakening the ordinary, single-query matcher above.
struct QueryContext {
  scf::ForOp loop;
  Value first, last;
  bool stored = false;
};

bool localKeyEquals(Value condition, scf::ForOp loop, int64_t keys,
                    int64_t wanted) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq)
    return false;
  for (unsigned side : {0u, 1u}) {
    auto remainder = compare->getOperand(side).getDefiningOp<arith::RemSIOp>();
    if (remainder && remainder.getLhs() == loop.getInductionVar() &&
        integer(remainder.getRhs()) == keys &&
        integer(compare->getOperand(1 - side)) == wanted)
      return true;
  }
  return false;
}

std::optional<QueryContext> queryContext(scf::ForOp loop, int64_t keys,
                                         int64_t bn, int64_t hd) {
  auto stream =
      loop ? loop->getAttrOfType<DictionaryAttr>("hivm.cvp_query.stream")
           : DictionaryAttr();
  auto count = stream ? stream.getAs<IntegerAttr>("key_count") : IntegerAttr();
  auto step = stream ? stream.getAs<IntegerAttr>("key_step") : IntegerAttr();
  if (!loop || !count || count.getInt() != keys || !step ||
      step.getInt() != bn || !loop->hasAttr("hivm.cvp_query.output_storage") ||
      keys < 2 * bn || keys % bn || integer(loop.getLowerBound()) != 0 ||
      integer(loop.getStep()) != bn ||
      !loop.getInductionVar().getType().isInteger(64) ||
      llvm::any_of(loop.getResults(), [](Value v) { return !v.use_empty(); }))
    return std::nullopt;
  bool stored = loop->hasAttr("hivm.cvp_query.state_storage");
  if (stored) {
    if (loop.getNumResults() || !loop.getInitArgs().empty())
      return std::nullopt;
  } else {
    if (loop.getNumResults() != 5 ||
        !isUB(loop.getResult(0).getType(), {64, hd}, 32))
      return std::nullopt;
    for (unsigned i = 1; i < 5; ++i)
      if (!isUB(loop.getResult(i).getType(), {64}, 32))
        return std::nullopt;
  }
  auto work = loop.getUpperBound().getDefiningOp<arith::MulIOp>();
  if (!work ||
      (integer(work.getLhs()) != keys && integer(work.getRhs()) != keys))
    return std::nullopt;
  QueryContext context{loop, {}, {}, stored};
  for (auto compare : loop.getBody()->getOps<arith::CmpIOp>()) {
    if (localKeyEquals(compare, loop, keys, 0)) {
      if (context.first)
        return std::nullopt;
      context.first = compare;
    }
    if (localKeyEquals(compare, loop, keys, keys - bn)) {
      if (context.last)
        return std::nullopt;
      context.last = compare;
    }
  }
  if (!context.first || !context.last)
    return std::nullopt;
  return context;
}

bool sameWholeMemory(Type a, Type b) {
  auto x = dyn_cast<MemRefType>(a), y = dyn_cast<MemRefType>(b);
  return x && y && x.hasStaticShape() && y.hasStaticShape() &&
         x.getLayout().isIdentity() && y.getLayout().isIdentity() &&
         x.getNumElements() == y.getNumElements() &&
         x.getElementType() == y.getElementType() &&
         x.getMemorySpace() == y.getMemorySpace();
}

Value wholeMemoryRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(
            op))
      break;
    if (!sameWholeMemory(value.getType(), op->getOperand(0).getType()))
      return {};
    value = op->getOperand(0);
  }
  return value;
}

std::optional<FloatAttr> retainedFillArgument(func::FuncOp function,
                                              unsigned index, Type type,
                                              int64_t hd) {
  if (!simpleFunction(function) || index >= function.getNumArguments() ||
      function.getArgument(index).getType() != type)
    return std::nullopt;
  bool reads = false;
  SmallVector<vector::TransferWriteOp> writes;
  function.walk([&](vector::TransferReadOp) { reads = true; });
  function.walk([&](vector::TransferWriteOp write) {
    if (rootArgument(write.getSource(), function) ==
        function.getArgument(index))
      writes.push_back(write);
  });
  if (reads || writes.size() != 1)
    return std::nullopt;
  auto write = writes.front();
  auto value = splat(write.getVector());
  if (!value || !value.getType().isF32())
    return std::nullopt;
  if (isUB(type, {64}, 32)) {
    auto vector = dyn_cast<VectorType>(write.getVector().getType());
    if (write.getSource() != function.getArgument(index) || !vector ||
        vector.getShape() != ArrayRef<int64_t>{64})
      return std::nullopt;
  } else if (!isUB(type, {64, hd}, 32) ||
             !isChunk(write.getVector().getType(), 32) ||
             !fullRowChunks(
                 write.getSource().getDefiningOp<memref::SubViewOp>(), function,
                 hd)) {
    return std::nullopt;
  }
  return value;
}

// Unlike initialization(), this proof never authorizes erasing the writer.
// A merged sum/O fill is allowed, but every alias of this particular output
// must remain read-only after its unique dominating full write.
std::optional<FloatAttr> retainedConstant(Value memory, Operation *consumer,
                                          ModuleOp module, int64_t hd,
                                          int64_t denseKeys) {
  auto allocation = memory.getDefiningOp<memref::AllocOp>();
  if (!allocation || !allocation.getDynamicSizes().empty())
    return std::nullopt;
  DominanceInfo dominance(module);
  std::optional<FloatAttr> initial;
  SmallVector<Value> pending{memory};
  llvm::DenseSet<Value> visited;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (isa<annotation::MarkOp>(op))
        continue;
      if (auto call = dyn_cast<func::CallOp>(op)) {
        auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
        auto fill = value == memory
                        ? retainedFillArgument(function, use.getOperandNumber(),
                                               memory.getType(), hd)
                        : std::nullopt;
        if (fill && dominance.dominates(op, consumer)) {
          if (initial)
            return std::nullopt;
          initial = *fill;
        } else if (!simpleFunction(function, denseKeys) ||
                   !readOnlyArgument(function, use.getOperandNumber())) {
          return std::nullopt;
        }
      } else if (auto copy = dyn_cast<CopyOp>(op)) {
        if (copy.getSrc() != value || copy.getDst() == value)
          return std::nullopt;
      } else if (auto select = dyn_cast<arith::SelectOp>(op)) {
        if (use.getOperandNumber() == 0)
          return std::nullopt;
        pending.push_back(select.getResult());
      } else if (isa<memref::SubViewOp, memref::CollapseShapeOp,
                     memref::ExpandShapeOp, memref::CastOp>(op)) {
        if (use.getOperandNumber() != 0)
          return std::nullopt;
        pending.push_back(op->getResult(0));
      } else {
        return std::nullopt;
      }
    }
  }
  return initial;
}

bool metadataOnly(Block *block) {
  return !block || llvm::all_of(block->without_terminator(), [](Operation &op) {
    return isa<AnchorOp, annotation::MarkOp>(op);
  });
}

struct QueryOutputSite {
  QueryContext context;
  scope::ScopeOp scope;
  func::CallOp updateCall, normalCall;
  Update update;
  Normalizer normal;
  scf::IfOp finalization;
  Value carried;
  FloatAttr initial;
  SmallVector<Operation *> tail;
  bool stored = false;
};

// Explicit state storage has no loop-carried memref and no O scope result.
// Prove its real bank/snapshot/copy graph instead of relaxing the legacy SSA
// recurrence matcher. Only whole, identity-layout views can join this graph.
bool closedWholeMemory(Value root,
                       llvm::function_ref<bool(OpOperand &)> allowed) {
  if (!root)
    return false;
  SmallVector<Value> pending{root};
  llvm::DenseSet<Value> seen;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (isa<annotation::MarkOp>(op))
        continue;
      if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(
              op) &&
          use.getOperandNumber() == 0 &&
          wholeMemoryRoot(op->getResult(0)) == root) {
        pending.push_back(op->getResult(0));
        continue;
      }
      if (!allowed(use))
        return false;
    }
  }
  return true;
}

bool accumulatorBank(Value value, scf::ForOp loop, int64_t hd) {
  auto alloc = value.getDefiningOp<memref::AllocOp>();
  auto bank =
      alloc ? alloc->getAttrOfType<DictionaryAttr>("hivm.cvp_query.state_bank")
            : DictionaryAttr();
  if (!alloc || !isUB(value.getType(), {64, hd}, 32) ||
      !alloc.getDynamicSizes().empty() ||
      alloc->getBlock() != loop->getBlock() || !alloc->isBeforeInBlock(loop) ||
      !bank || bank.size() != 4)
    return false;
  auto role = bank.getAs<StringAttr>("role");
  auto clock = bank.getAs<StringAttr>("clock");
  auto slot = bank.getAs<IntegerAttr>("bank");
  auto slots = bank.getAs<IntegerAttr>("required_slots");
  bool unique = llvm::any_of(value.getUsers(), [](Operation *op) {
    return isa<annotation::MarkOp>(op) &&
           isa_and_nonnull<UnitAttr>(op->getAttr("mem_unique"));
  });
  return unique && role && role.getValue() == "accumulator" && clock &&
         clock.getValue() == "O" && slot && slot.getInt() == 0 && slots &&
         slots.getInt() == 1;
}

struct WholeCopyChain {
  Value source;
  SmallVector<Value> roots;
  SmallVector<CopyOp> copies;
};

// Walk a unique sequence of whole copies backwards, from the consumer's
// snapshot to a persistent bank, or from that bank's write back to raw O.
std::optional<WholeCopyChain> wholeCopyChain(Value destination, Value source,
                                             Operation *consumer,
                                             scope::ScopeOp scope,
                                             scf::ForOp loop, int64_t hd) {
  WholeCopyChain chain;
  Value current = wholeMemoryRoot(destination);
  llvm::DenseSet<Value> seen;
  while (current && current != source &&
         (source || !accumulatorBank(current, loop, hd))) {
    if (!seen.insert(current).second ||
        !current.getDefiningOp<memref::AllocOp>() ||
        !sameWholeMemory(current.getType(), destination.getType()))
      return std::nullopt;
    CopyOp writer;
    bool valid = closedWholeMemory(current, [&](OpOperand &use) {
      auto copy = dyn_cast<CopyOp>(use.getOwner());
      if (!copy || wholeMemoryRoot(copy.getDst()) != current ||
          copy->getBlock() != &scope.getRegion().front())
        return true;
      if (writer && writer != copy)
        return false;
      writer = copy;
      return true;
    });
    if (!valid || !writer ||
        !sameWholeMemory(writer.getSrc().getType(), destination.getType()) ||
        !sameWholeMemory(writer.getDst().getType(), destination.getType()) ||
        !writer->isBeforeInBlock(consumer))
      return std::nullopt;
    if (current != wholeMemoryRoot(destination) || !source) {
      Operation *alloc = current.getDefiningOp();
      if (alloc->getBlock() != &scope.getRegion().front() ||
          !alloc->isBeforeInBlock(writer))
        return std::nullopt;
    }
    chain.roots.push_back(current);
    chain.copies.push_back(writer);
    consumer = writer;
    current = wholeMemoryRoot(writer.getSrc());
  }
  if (!current || chain.copies.empty())
    return std::nullopt;
  chain.source = current;
  return chain;
}

std::optional<QueryOutputSite>
matchStoredQueryOutput(ModuleOp module, func::CallOp call, Update update,
                       scope::ScopeOp scope, QueryContext context, int64_t bn,
                       int64_t hd) {
  if (scope.getNumResults() ||
      cast<scope::ReturnOp>(scope.getRegion().front().getTerminator())
          .getNumOperands())
    return std::nullopt;
  auto reset =
      call.getOperand(update.oldIndex).getDefiningOp<arith::SelectOp>();
  if (!reset || reset.getCondition() != context.first ||
      reset->getBlock() != &scope.getRegion().front())
    return std::nullopt;
  auto initial =
      retainedConstant(reset.getTrueValue(), context.loop, module, hd, bn);
  if (!initial || !initial->getValue().isPosZero() ||
      !readOnlyArgument(update.function, update.oldIndex))
    return std::nullopt;
  auto snapshot =
      wholeCopyChain(reset.getFalseValue(), {}, reset, scope, context.loop, hd);
  if (!snapshot || !accumulatorBank(snapshot->source, context.loop, hd))
    return std::nullopt;
  Value bank = snapshot->source;
  scf::IfOp finalization;
  for (auto branch : scope.getRegion().front().getOps<scf::IfOp>())
    if (branch.getCondition() == context.last) {
      if (finalization)
        return std::nullopt;
      finalization = branch;
    }
  if (!finalization || finalization.getNumResults() ||
      !call->isBeforeInBlock(finalization) ||
      !metadataOnly(finalization.elseBlock()))
    return std::nullopt;
  func::CallOp normalCall;
  std::optional<Normalizer> normal;
  unsigned stores = 0;
  for (Operation &op : finalization.thenBlock()->without_terminator()) {
    if (auto candidate = dyn_cast<func::CallOp>(op)) {
      if (normalCall)
        return std::nullopt;
      normalCall = candidate;
      auto callee = module.lookupSymbol<func::FuncOp>(candidate.getCallee());
      normal = matchNormalizer(callee, hd);
      if (!normal || !simpleUniqueCall(candidate, callee, module))
        return std::nullopt;
    } else if (isa<StoreOp>(op)) {
      ++stores;
    } else if (!isa<memref::AllocOp, CopyOp, AnchorOp, annotation::MarkOp>(
                   op) &&
               (op.getNumRegions() || !isMemoryEffectFree(&op))) {
      return std::nullopt;
    }
  }
  Value raw = call.getOperand(update.outputIndex);
  Value rawRoot = wholeMemoryRoot(raw);
  auto rawAllocation =
      rawRoot ? rawRoot.getDefiningOp<memref::AllocOp>() : memref::AllocOp();
  if (!rawAllocation || !rawAllocation.getDynamicSizes().empty() || !normal ||
      normalCall.getOperand(normal->oldIndex) != raw || stores != 2)
    return std::nullopt;
  if (rawAllocation->getBlock() == &scope.getRegion().front()) {
    if (!rawAllocation->isBeforeInBlock(call))
      return std::nullopt;
  } else if (rawAllocation->getBlock() != context.loop.getBody() ||
             !rawAllocation->isBeforeInBlock(scope)) {
    return std::nullopt;
  }
  auto write =
      wholeCopyChain(bank, rawRoot, finalization, scope, context.loop, hd);
  if (!write || write->source != rawRoot ||
      llvm::any_of(write->copies,
                   [&](CopyOp copy) { return !call->isBeforeInBlock(copy); }))
    return std::nullopt;
  llvm::SmallPtrSet<Operation *, 16> copies;
  for (CopyOp copy : snapshot->copies)
    copies.insert(copy);
  for (CopyOp copy : write->copies)
    copies.insert(copy);
  auto privateSnapshot = [&](Value root) {
    return closedWholeMemory(root, [&](OpOperand &use) {
      return copies.contains(use.getOwner()) ||
             (use.getOwner() == reset && use.getOperandNumber() == 2 &&
              wholeMemoryRoot(reset.getFalseValue()) == root);
    });
  };
  if (llvm::any_of(snapshot->roots,
                   [&](Value root) { return !privateSnapshot(root); }) ||
      llvm::any_of(write->roots, [&](Value root) {
        return root != bank && !privateSnapshot(root);
      }))
    return std::nullopt;
  CopyOp initialize;
  if (!closedWholeMemory(
          bank,
          [&](OpOperand &use) {
            Operation *op = use.getOwner();
            if (copies.contains(op))
              return true;
            auto copy = dyn_cast<CopyOp>(op);
            if (!copy || wholeMemoryRoot(copy.getDst()) != bank || initialize ||
                copy->getBlock() != context.loop->getBlock() ||
                !copy->isBeforeInBlock(context.loop) ||
                !sameWholeMemory(copy.getSrc().getType(), bank.getType()) ||
                !sameWholeMemory(copy.getDst().getType(), bank.getType()))
              return false;
            auto value = retainedConstant(wholeMemoryRoot(copy.getSrc()),
                                          context.loop, module, hd, bn);
            if (!value || !value->getValue().isPosZero())
              return false;
            initialize = copy;
            return true;
          }) ||
      !initialize)
    return std::nullopt;
  // Raw O may share the PV destination. Earlier copies in this same
  // iteration are unchanged. Do not allow an unknown call or alias-producing
  // operation to capture the buffer before the update. Every later use must
  // be the checked bank write or normalizer, since final O elides the update.
  if (!closedWholeMemory(rawRoot, [&](OpOperand &use) {
        Operation *op = use.getOwner();
        if (op == call || op == normalCall || copies.contains(op))
          return true;
        if (!isa<CopyOp>(op))
          return false;
        if (op->getBlock() == &scope.getRegion().front())
          return op->isBeforeInBlock(call);
        while (op && op->getBlock() != context.loop.getBody())
          op = op->getParentOp();
        return op && op != scope && op->isBeforeInBlock(scope);
      }))
    return std::nullopt;
  QueryOutputSite site{context,  scope,   call,         normalCall,
                       update,   *normal, finalization, reset.getFalseValue(),
                       *initial, {},      true};
  for (Operation *op = call->getNextNode(); op != finalization;
       op = op->getNextNode()) {
    if (!op ||
        (!copies.contains(op) &&
         !isa<memref::AllocOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
              memref::CastOp, AnchorOp, annotation::MarkOp>(op)))
      return std::nullopt;
    site.tail.push_back(op);
  }
  // Moving this write into the non-final branch must not move a definition
  // that is needed by the final branch or outside the O stage.
  for (Operation *op : site.tail)
    for (Value value : op->getResults())
      for (Operation *user : value.getUsers())
        if (!llvm::is_contained(site.tail, user))
          return std::nullopt;
  for (Operation &op : scope.getRegion().front().without_terminator()) {
    if (&op == call || &op == finalization || copies.contains(&op) ||
        llvm::is_contained(site.tail, &op))
      continue;
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults() || !metadataOnly(branch.thenBlock()) ||
          !metadataOnly(branch.elseBlock()))
        return std::nullopt;
    } else if (!isa<memref::AllocOp, AnchorOp, annotation::MarkOp>(op) &&
               (op.getNumRegions() || !isMemoryEffectFree(&op))) {
      return std::nullopt;
    }
  }
  return site;
}

std::optional<QueryOutputSite> matchQueryOutput(ModuleOp module,
                                                func::CallOp call, int64_t keys,
                                                int64_t bn, int64_t hd,
                                                bool first, bool last) {
  auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
  auto update = matchUpdate(function, hd);
  auto scope = dyn_cast<scope::ScopeOp>(call->getParentOp());
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp();
  auto context = queryContext(loop, keys, bn, hd);
  auto phase = scope ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  if (!update || !context || !phase || phase.getInt() != 0 ||
      !protectedVectorScope(scope, false) ||
      !simpleUniqueCall(call, function, module))
    return std::nullopt;
  if (context->stored) {
    auto site =
        matchStoredQueryOutput(module, call, *update, scope, *context, bn, hd);
    if (site && last) {
      bool anchored = false;
      call->getParentOfType<func::FuncOp>().walk(
          [&](AnchorOp) { anchored = true; });
      Block *otherwise = site->finalization.elseBlock();
      // Frozen MIX must already cover the false-region consumers introduced
      // by last-output specialization. An empty SCF else is insufficient.
      if (anchored && (!otherwise || otherwise->getOps<AnchorOp>().empty()))
        return std::nullopt;
    }
    return site;
  }
  if (scope.getNumResults() != 1)
    return std::nullopt;
  auto reset =
      call.getOperand(update->oldIndex).getDefiningOp<arith::SelectOp>();
  Value carried = loop.getRegionIterArgs()[0];
  if (!reset || reset.getCondition() != context->first ||
      reset.getFalseValue() != carried ||
      !llvm::all_of(carried.getUsers(),
                    [&](Operation *op) { return op == reset; }))
    return std::nullopt;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto ret = cast<scope::ReturnOp>(scope->getRegion(0).front().getTerminator());
  if (yield.getOperand(0) != scope.getResult(0) ||
      !scope.getResult(0).hasOneUse() || ret.getNumOperands() != 1)
    return std::nullopt;
  FloatAttr initial;
  if (first) {
    auto value = retainedConstant(reset.getTrueValue(), loop, module, hd, bn);
    if (!value || !value->getValue().isPosZero() ||
        !readOnlyArgument(function, update->oldIndex))
      return std::nullopt;
    initial = *value;
  }
  scf::IfOp finalization;
  for (auto branch : scope->getRegion(0).front().getOps<scf::IfOp>())
    if (branch.getCondition() == context->last) {
      if (finalization)
        return std::nullopt;
      finalization = branch;
    }
  if (!finalization || finalization.getNumResults() ||
      !call->isBeforeInBlock(finalization) ||
      !metadataOnly(finalization.elseBlock()))
    return std::nullopt;
  func::CallOp normalCall;
  std::optional<Normalizer> normal;
  unsigned stores = 0;
  for (Operation &op : finalization.thenBlock()->without_terminator()) {
    if (auto candidate = dyn_cast<func::CallOp>(op)) {
      if (normalCall)
        return std::nullopt;
      normalCall = candidate;
      auto callee = module.lookupSymbol<func::FuncOp>(candidate.getCallee());
      normal = matchNormalizer(callee, hd);
      if (!normal || !simpleUniqueCall(candidate, callee, module))
        return std::nullopt;
    } else if (isa<StoreOp>(op)) {
      ++stores;
    } else if (!isa<memref::AllocOp, CopyOp, AnchorOp, annotation::MarkOp>(
                   op) &&
               (op.getNumRegions() || !isMemoryEffectFree(&op))) {
      return std::nullopt;
    }
  }
  Value raw = call.getOperand(update->outputIndex);
  if (!normal || normalCall.getOperand(normal->oldIndex) != raw || stores != 2)
    return std::nullopt;
  QueryOutputSite site{*context, scope,        call,    normalCall, *update,
                       *normal,  finalization, carried, initial,    {}};
  llvm::SmallPtrSet<Operation *, 16> copies;
  for (Operation *op = finalization->getNextNode(); op != ret;
       op = op->getNextNode()) {
    if (!op ||
        !isa<memref::AllocOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
             memref::CastOp, CopyOp, AnchorOp, annotation::MarkOp>(op))
      return std::nullopt;
    site.tail.push_back(op);
    if (isa<CopyOp>(op))
      copies.insert(op);
  }
  Value current = wholeMemoryRoot(ret.getOperand(0));
  Value source = wholeMemoryRoot(raw);
  llvm::DenseSet<Value> roots;
  llvm::SmallPtrSet<Operation *, 16> chain;
  Operation *consumer = ret;
  while (current && current != source) {
    if (!roots.insert(current).second ||
        !current.getDefiningOp<memref::AllocOp>() ||
        !sameWholeMemory(current.getType(), raw.getType()))
      return std::nullopt;
    CopyOp writer;
    for (Operation *op : copies) {
      auto copy = cast<CopyOp>(op);
      if (wholeMemoryRoot(copy.getDst()) != current)
        continue;
      if (writer || !sameWholeMemory(copy.getSrc().getType(), raw.getType()) ||
          !sameWholeMemory(copy.getDst().getType(), raw.getType()) ||
          !op->isBeforeInBlock(consumer))
        return std::nullopt;
      writer = copy;
    }
    if (!writer)
      return std::nullopt;
    chain.insert(writer);
    consumer = writer;
    current = wholeMemoryRoot(writer.getSrc());
  }
  if (!current || !source || chain.size() != copies.size())
    return std::nullopt;
  roots.insert(source);
  llvm::SmallPtrSet<Operation *, 32> allowed;
  allowed.insert(call);
  allowed.insert(normalCall);
  allowed.insert(ret);
  for (Operation *op : copies)
    allowed.insert(op);
  for (Value root : roots) {
    SmallVector<Value> pending{root};
    llvm::DenseSet<Value> seen;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (!seen.insert(value).second)
        continue;
      for (Operation *op : value.getUsers()) {
        if (allowed.contains(op) || isa<annotation::MarkOp>(op))
          continue;
        if (isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::CastOp>(
                op) &&
            wholeMemoryRoot(op->getResult(0)) == root) {
          pending.push_back(op->getResult(0));
          continue;
        }
        return std::nullopt;
      }
    }
  }
  for (Operation &op : scope->getRegion(0).front().without_terminator()) {
    if (&op == call || &op == finalization ||
        llvm::is_contained(site.tail, &op))
      continue;
    if (auto branch = dyn_cast<scf::IfOp>(op)) {
      if (branch.getNumResults() || !metadataOnly(branch.thenBlock()) ||
          !metadataOnly(branch.elseBlock()))
        return std::nullopt;
    } else if (!isa<memref::AllocOp, AnchorOp, annotation::MarkOp>(op) &&
               (op.getNumRegions() || !isMemoryEffectFree(&op))) {
      return std::nullopt;
    }
  }
  return site;
}

struct QueryStateSite {
  QueryContext context;
  func::CallOp call;
  func::FuncOp function;
  SmallVector<std::pair<unsigned, FloatAttr>, 2> initial;
};

std::optional<QueryStateSite> matchQueryState(ModuleOp module,
                                              func::CallOp call, int64_t keys,
                                              int64_t bn, int64_t hd) {
  auto scope = dyn_cast<scope::ScopeOp>(call->getParentOp());
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp();
  auto context = queryContext(loop, keys, bn, hd);
  auto phase = scope ? scope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                     : IntegerAttr();
  auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
  if (!context || !phase || phase.getInt() != 2 ||
      !protectedVectorScope(scope, true) || !simpleFunction(function, bn) ||
      !simpleUniqueCall(call, function, module))
    return std::nullopt;
  QueryStateSite site{*context, call, function, {}};
  bool sum = false, maximum = false;
  for (auto [index, operand] : llvm::enumerate(call.getOperands())) {
    auto reset = operand.getDefiningOp<arith::SelectOp>();
    if (!reset || reset.getCondition() != context->first ||
        !isUB(operand.getType(), {64}, 32))
      continue;
    auto value = retainedConstant(reset.getTrueValue(), loop, module, hd, bn);
    if (!value || !readOnlyArgument(function, index))
      return std::nullopt;
    const APFloat &constant = value->getValue();
    if (constant.isExactlyValue(1.0) && !sum)
      sum = true;
    else if (constant.isInfinity() && constant.isNegative() && !maximum)
      maximum = true;
    else
      return std::nullopt;
    site.initial.push_back({static_cast<unsigned>(index), *value});
  }
  if (!sum || !maximum || site.initial.size() != 2)
    return std::nullopt;
  return site;
}

void rewriteQueryState(QueryStateSite &site, ModuleOp module) {
  auto first = cloneFunction(site.function, module, "query_first_state");
  replaceInitialReads(first, site.initial);
  SmallVector<Value> ordinary(site.call.getOperands());
  SmallVector<Value> initial = ordinary;
  SmallVector<unsigned> removed;
  for (auto [index, constant] : site.initial) {
    (void)constant;
    ordinary[index] =
        ordinary[index].getDefiningOp<arith::SelectOp>().getFalseValue();
    removed.push_back(index);
  }
  llvm::sort(removed, std::greater<unsigned>());
  for (unsigned index : removed)
    initial.erase(initial.begin() + index);
  OpBuilder builder(site.call);
  auto branch =
      builder.create<scf::IfOp>(site.call.getLoc(), site.context.first, true);
  auto thenBuilder = branch.getThenBodyBuilder();
  auto elseBuilder = branch.getElseBodyBuilder();
  callLike(thenBuilder, site.call, first, initial);
  callLike(elseBuilder, site.call, site.function, ordinary);
  site.call.erase();
}

void rewriteQueryOutput(QueryOutputSite &site, ModuleOp module, bool first,
                        bool last) {
  func::FuncOp firstFunction;
  if (first) {
    firstFunction =
        cloneFunction(site.update.function, module, "query_first_output");
    replaceInitialReads(firstFunction, {{site.update.oldIndex, site.initial}});
  }
  auto emitUpdate = [&](OpBuilder &builder) {
    SmallVector<Value> ordinary(site.updateCall.getOperands());
    if (!first) {
      callLike(builder, site.updateCall, site.update.function, ordinary);
      return;
    }
    SmallVector<Value> initial = ordinary;
    initial.erase(initial.begin() + site.update.oldIndex);
    ordinary[site.update.oldIndex] = site.carried;
    auto branch = builder.create<scf::IfOp>(site.updateCall.getLoc(),
                                            site.context.first, true);
    auto thenBuilder = branch.getThenBodyBuilder();
    auto elseBuilder = branch.getElseBodyBuilder();
    callLike(thenBuilder, site.updateCall, firstFunction, initial);
    callLike(elseBuilder, site.updateCall, site.update.function, ordinary);
  };
  if (!last) {
    OpBuilder builder(site.updateCall);
    emitUpdate(builder);
    site.updateCall.erase();
    return;
  }
  auto lastFunction = makeLastFunction(site.normal, site.update, module);
  lastFunction->setAttr(
      kBoundaryKind, StringAttr::get(module.getContext(), "query_last_output"));
  if (site.stored) {
    // Delayed GSS pairs these anchors with the frozen MIX graph. Reuse the
    // original finalization and preserve every marker's order and region.
    // The closed tail-use proof permits moving only its data operations.
    SmallVector<Value> arguments(site.normalCall.getOperands());
    arguments[site.normal.oldIndex] = site.carried;
    arguments.push_back(site.updateCall.getOperand(site.update.alphaIndex));
    arguments.push_back(site.updateCall.getOperand(site.update.pvIndex));
    OpBuilder normalBuilder(site.normalCall);
    callLike(normalBuilder, site.normalCall, lastFunction, arguments);
    site.normalCall.erase();
    OpBuilder elseBuilder(module.getContext());
    Block *otherwise = site.finalization.elseBlock();
    if (!otherwise) {
      otherwise = elseBuilder.createBlock(&site.finalization.getElseRegion());
      elseBuilder.create<scf::YieldOp>(site.finalization.getLoc());
    }
    // Keep the existing else metadata after the newly moved data consumers.
    // Live memory effects, including the PV reads, are rebuilt by delayed GSS.
    elseBuilder.setInsertionPointToStart(otherwise);
    emitUpdate(elseBuilder);
    for (Operation *op : site.tail)
      if (!isa<AnchorOp>(op))
        op->moveBefore(elseBuilder.getInsertionBlock(),
                       elseBuilder.getInsertionPoint());
    site.updateCall.erase();
    return;
  }
  auto ret =
      cast<scope::ReturnOp>(site.scope->getRegion(0).front().getTerminator());
  Value ordinaryCarry = ret.getOperand(0);
  OpBuilder builder(site.finalization);
  auto branch = builder.create<scf::IfOp>(site.finalization.getLoc(),
                                          TypeRange{site.carried.getType()},
                                          site.context.last, true);
  auto thenBuilder = branch.getThenBodyBuilder();
  for (Operation &op : llvm::make_early_inc_range(
           site.finalization.thenBlock()->without_terminator()))
    op.moveBefore(thenBuilder.getInsertionBlock(),
                  thenBuilder.getInsertionPoint());
  SmallVector<Value> arguments(site.normalCall.getOperands());
  arguments[site.normal.oldIndex] = site.carried;
  arguments.push_back(site.updateCall.getOperand(site.update.alphaIndex));
  arguments.push_back(site.updateCall.getOperand(site.update.pvIndex));
  OpBuilder normalBuilder(site.normalCall);
  callLike(normalBuilder, site.normalCall, lastFunction, arguments);
  site.normalCall.erase();
  thenBuilder.create<scf::YieldOp>(site.finalization.getLoc(), site.carried);
  auto elseBuilder = branch.getElseBodyBuilder();
  emitUpdate(elseBuilder);
  if (site.finalization.elseBlock())
    for (Operation &op : llvm::make_early_inc_range(
             site.finalization.elseBlock()->without_terminator()))
      op.moveBefore(elseBuilder.getInsertionBlock(),
                    elseBuilder.getInsertionPoint());
  for (Operation *op : site.tail)
    op->moveBefore(elseBuilder.getInsertionBlock(),
                   elseBuilder.getInsertionPoint());
  elseBuilder.create<scf::YieldOp>(site.finalization.getLoc(), ordinaryCarry);
  ret.setOperand(0, branch.getResult(0));
  site.updateCall.erase();
  site.finalization.erase();
}

struct CVPAttentionBoundaryPass
    : impl::CVPAttentionBoundaryBase<CVPAttentionBoundaryPass> {
  using Base::Base;
  void runOnOperation() override {
    if (!enableFirstOutput && !enableLastOutput && !enableFirstState)
      return;
    ModuleOp original = getOperation();
    bool hasQuery = llvm::any_of(
        original.getOps<func::FuncOp>(), [](func::FuncOp function) {
          return function->hasAttr("hivm.cvp_query.stream");
        });
    OwningOpRef<ModuleOp> candidate(hasQuery ? cast<ModuleOp>(original->clone())
                                             : ModuleOp());
    ModuleOp module = candidate ? *candidate : original;
    SmallVector<OutputSite, 1> outputs;
    SmallVector<StateSite> states;
    SmallVector<InitialStateGroup, 1> stateGroups;
    SmallVector<QueryOutputSite, 1> queryOutputs;
    SmallVector<QueryStateSite, 1> queryStates;
    bool incompleteQuery = false;
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      auto plans = caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
      if (caller->hasAttr("hivm.backup_function") || !plans ||
          plans.size() != 1 || !caller->hasAttr("hivm.part_of_mix") ||
          caller->getAttr("hivm.func_core_type") !=
              TFuncCoreTypeAttr::get(&getContext(), TFuncCoreType::AIV))
        continue;
      auto plan = dyn_cast<DictionaryAttr>(plans[0]);
      auto bm = plan ? plan.getAs<IntegerAttr>("block_m") : IntegerAttr();
      auto bn = plan ? plan.getAs<IntegerAttr>("block_n") : IntegerAttr();
      auto hd = plan ? plan.getAs<IntegerAttr>("head_dim") : IntegerAttr();
      auto keys = plan ? plan.getAs<IntegerAttr>("key_count") : IntegerAttr();
      if (!bm || bm.getInt() != 128 || !bn ||
          (bn.getInt() != 128 && bn.getInt() != 256) || !hd ||
          (hd.getInt() != 64 && hd.getInt() != 128) || !keys ||
          keys.getInt() < bn.getInt() || keys.getInt() % bn.getInt())
        continue;
      auto layout = plan.getAs<StringAttr>("score_layout");
      bool keyMajor = layout && layout.getValue() == "key-major";
      if (caller->hasAttr("hivm.cvp_query.stream")) {
        size_t previousOutputs = queryOutputs.size();
        size_t previousStates = queryStates.size();
        if (keyMajor)
          caller.walk([&](func::CallOp call) {
            if (enableFirstOutput || enableLastOutput)
              if (auto site = matchQueryOutput(
                      module, call, keys.getInt(), bn.getInt(), hd.getInt(),
                      enableFirstOutput, enableLastOutput))
                queryOutputs.push_back(std::move(*site));
            if (enableFirstState)
              if (auto site = matchQueryState(module, call, keys.getInt(),
                                              bn.getInt(), hd.getInt()))
                queryStates.push_back(std::move(*site));
          });
        incompleteQuery |=
            ((enableFirstOutput || enableLastOutput) &&
             queryOutputs.size() != previousOutputs + 1) ||
            (enableFirstState && queryStates.size() != previousStates + 1);
        continue;
      }
      if (enableFirstState && keyMajor) {
        auto maxSlot = plan.getAs<IntegerAttr>("max_state");
        auto sumSlot = plan.getAs<IntegerAttr>("sum_state");
        auto version = plan.getAs<IntegerAttr>("version");
        if (maxSlot && sumSlot && version && version.getInt() == 1)
          caller.walk([&](scf::ForOp loop) {
            if (auto group = matchInitialStateGroup(
                    module, loop, keys.getInt(), bn.getInt(), maxSlot.getInt(),
                    sumSlot.getInt()))
              stateGroups.push_back(std::move(*group));
          });
      }
      caller.walk([&](func::CallOp call) {
        if (enableFirstOutput || enableLastOutput)
          if (auto site = matchOutputSite(module, caller, call, hd.getInt(),
                                          bn.getInt(), keys.getInt(),
                                          enableFirstOutput, enableLastOutput))
            outputs.push_back(*site);
        if (enableFirstState && !keyMajor)
          if (auto site =
                  matchState(module, caller, call, keys.getInt(), bn.getInt()))
            states.push_back(*site);
      });
    }
    if (incompleteQuery ||
        (requireMatch && (((enableFirstOutput || enableLastOutput) &&
                           outputs.empty() && queryOutputs.empty()) ||
                          (enableFirstState && states.empty() &&
                           stateGroups.empty() && queryStates.empty())))) {
      if (requireMatch) {
        original.emitError(
            "CVP boundary specialization requires a proved Skew "
            "logical boundary, private initialization, and complete "
            "output/state use graph");
        signalPassFailure();
      }
      return;
    }
    for (OutputSite &site : outputs)
      rewriteOutput(site, module, enableFirstOutput, enableLastOutput);
    for (StateSite &site : states)
      rewriteState(site, module);
    for (InitialStateGroup &group : stateGroups)
      rewriteInitialStateGroup(group, module);
    for (QueryOutputSite &site : queryOutputs)
      rewriteQueryOutput(site, module, enableFirstOutput, enableLastOutput);
    for (QueryStateSite &site : queryStates)
      rewriteQueryState(site, module);
    auto count = [&](StringRef name, size_t value) {
      if (value)
        module->setAttr(
            name, IntegerAttr::get(IntegerType::get(&getContext(), 64), value));
    };
    if (enableFirstOutput)
      count("hivm.cvp_attention.first_output_specializations",
            outputs.size() + queryOutputs.size());
    if (enableLastOutput)
      count("hivm.cvp_attention.last_output_specializations",
            outputs.size() + queryOutputs.size());
    if (enableFirstState)
      count("hivm.cvp_attention.first_state_specializations",
            states.size() + stateGroups.size() + queryStates.size());
    if (!retainProvenance)
      for (func::FuncOp function : module.getOps<func::FuncOp>())
        if (!function->hasAttr("hivm.backup_function"))
          function->removeAttr(kCVPAttentionVerifiedAttr);
    if (failed(verify(module))) {
      signalPassFailure();
      return;
    }
    if (candidate) {
      original.getBodyRegion().takeBody(module.getBodyRegion());
      original->setAttrs(module->getAttrs());
    }
  }
};
} // namespace

std::unique_ptr<Pass>
createCVPAttentionBoundaryPass(const CVPAttentionBoundaryOptions &options) {
  return std::make_unique<CVPAttentionBoundaryPass>(options);
}
} // namespace hivm
} // namespace mlir

//===- CVPAttentionVector.cpp - In-place score rematerialization
//------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONVECTOR
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
constexpr StringLiteral kRematerialized =
    "hivm.cvp_attention.score_store_removals";
constexpr StringLiteral kMaxScaled = "hivm.cvp_attention.max_scalings";
constexpr StringLiteral kDistinctRemoved =
    "hivm.cvp_attention.distinct_scratch_removals";

std::optional<int64_t> constantInt(Value value) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getSExtValue();
}

BlockArgument entryArgument(Value value, func::FuncOp function) {
  while (auto view = value.getDefiningOp<memref::SubViewOp>())
    value = view.getSource();
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front())
    return {};
  return argument;
}

bool isTileType(Type type, int64_t columns) {
  auto memory = dyn_cast<MemRefType>(type);
  return memory && memory.hasStaticShape() && memory.getRank() == 2 &&
         memory.getShape() == ArrayRef<int64_t>{64, columns} &&
         memory.getElementType().isF32() && memory.getLayout().isIdentity() &&
         memory.getMemorySpace() ==
             AddressSpaceAttr::get(type.getContext(), AddressSpace::UB);
}

bool isTileVector(Type type) {
  auto vector = dyn_cast<VectorType>(type);
  return vector && !vector.isScalable() &&
         vector.getShape() == ArrayRef<int64_t>{1, 64} &&
         vector.getElementType().isF32();
}

bool directIndices(ValueRange indices) {
  return indices.size() == 2 &&
         llvm::all_of(indices, [](Value v) { return constantInt(v) == 0; });
}

bool allInBounds(Operation *op) {
  auto inBounds = op->getAttrOfType<ArrayAttr>("in_bounds");
  return inBounds && inBounds.size() == 2 &&
         llvm::all_of(inBounds, [](Attribute attr) {
           auto flag = dyn_cast<BoolAttr>(attr);
           return flag && flag.getValue();
         });
}

struct LoopOffset {
  scf::ForOp loop;
  int64_t factor = 1;
};

/// Admit only the two forms present in normalized dense vector loops:
/// the induction variable itself and affine.apply (iv * positive_constant).
std::optional<LoopOffset> loopOffset(OpFoldResult offset) {
  auto value = dyn_cast<Value>(offset);
  if (!value)
    return std::nullopt;
  int64_t factor = 1;
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumDims() != 1 || map.getNumSymbols() ||
        map.getNumResults() != 1 || apply.getNumOperands() != 1)
      return std::nullopt;
    AffineExpr expr = map.getResult(0);
    if (auto product = dyn_cast<AffineBinaryOpExpr>(expr)) {
      auto dimension = dyn_cast<AffineDimExpr>(product.getLHS());
      auto constant = dyn_cast<AffineConstantExpr>(product.getRHS());
      if (product.getKind() != AffineExprKind::Mul || !dimension ||
          dimension.getPosition() != 0 || !constant || constant.getValue() <= 0)
        return std::nullopt;
      factor = constant.getValue();
    } else if (!isa<AffineDimExpr>(expr) ||
               cast<AffineDimExpr>(expr).getPosition() != 0) {
      return std::nullopt;
    }
    value = apply.getOperand(0);
  }
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || argument.getArgNumber() != 0)
    return std::nullopt;
  auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
  if (!loop || argument != loop.getInductionVar())
    return std::nullopt;
  return LoopOffset{loop, factor};
}

bool coversTile(memref::SubViewOp view, Operation *access, int64_t columns) {
  if (!view || view.getMixedOffsets().size() != 2 ||
      view.getMixedSizes().size() != 2 || view.getMixedStrides().size() != 2)
    return false;
  auto sizes = view.getMixedSizes();
  auto strides = view.getMixedStrides();
  if (getConstantIntValue(sizes[0]) != 1 ||
      getConstantIntValue(sizes[1]) != 64 ||
      getConstantIntValue(strides[0]) != 1 ||
      getConstantIntValue(strides[1]) != 1)
    return false;
  auto row = loopOffset(view.getMixedOffsets()[0]);
  auto col = loopOffset(view.getMixedOffsets()[1]);
  auto function = access->getParentOfType<func::FuncOp>();
  if (!row || !col || row->factor != 1 || col->factor > 64 || !function ||
      row->loop->getParentOp() != function ||
      col->loop->getParentOp() != row->loop ||
      access->getParentOp() != col->loop)
    return false;
  auto rowLower = constantInt(row->loop.getLowerBound());
  auto rowUpper = constantInt(row->loop.getUpperBound());
  auto rowStep = constantInt(row->loop.getStep());
  auto colLower = constantInt(col->loop.getLowerBound());
  auto colUpper = constantInt(col->loop.getUpperBound());
  auto colStep = constantInt(col->loop.getStep());
  return rowLower == 0 && rowUpper == 64 && rowStep == 1 && colLower == 0 &&
         colUpper && colStep && *colStep > 0 &&
         *colUpper == columns / col->factor && columns % col->factor == 0 &&
         *colStep == 64 / col->factor && 64 % col->factor == 0;
}

/// Reject calls, absolute-memory construction and unmodelled effects. All
/// vector memory accesses must be through the callee's memref arguments.
bool hasOnlyArgumentMemory(func::FuncOp function) {
  if (function.isExternal() || function.getNumResults() != 0 ||
      !function->hasAttr("hivm.vector_function"))
    return false;
  bool valid = true;
  function.walk([&](Operation *op) {
    if (!valid || op == function.getOperation())
      return;
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      valid = bool(entryArgument(read.getSource(), function));
      return;
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      valid = bool(entryArgument(write.getSource(), function));
      return;
    }
    if (isa<func::CallOp>(op)) {
      valid = false;
      return;
    }
    if (isa<scf::ForOp, scf::YieldOp, func::ReturnOp>(op))
      return;
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      valid = false;
  });
  return valid;
}

bool uniqueCall(func::FuncOp function, func::CallOp call, ModuleOp module) {
  auto uses = SymbolTable::getSymbolUses(function, module);
  if (!uses)
    return false;
  unsigned count = 0;
  for (const auto &use : *uses) {
    if (use.getUser() != call.getOperation())
      return false;
    ++count;
  }
  return count == 1;
}

struct Producer {
  func::FuncOp function;
  vector::TransferReadOp read;
  arith::MulFOp multiply;
  vector::TransferWriteOp store;
  DenseElementsAttr factor;
  unsigned inputIndex;
  unsigned outputIndex;
};

std::optional<Producer> matchProducer(func::FuncOp function, int64_t columns) {
  if (!hasOnlyArgumentMemory(function))
    return std::nullopt;
  SmallVector<Producer> matches;
  function.walk([&](vector::TransferWriteOp store) {
    auto multiply = store.getVector().getDefiningOp<arith::MulFOp>();
    if (!multiply || !isTileVector(multiply.getType()) || store.getMask() ||
        !allInBounds(store) || !directIndices(store.getIndices()) ||
        !store.getPermutationMap().isIdentity())
      return;
    vector::TransferReadOp read;
    DenseElementsAttr factor;
    for (unsigned index : {0u, 1u}) {
      auto constant =
          multiply->getOperand(index).getDefiningOp<arith::ConstantOp>();
      auto candidate = multiply->getOperand(1 - index)
                           .getDefiningOp<vector::TransferReadOp>();
      if (!constant || !candidate)
        continue;
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (dense && dense.isSplat() && dense.getElementType().isF32() &&
          dense.getSplatValue<APFloat>().convertToDouble() == 0.5) {
        read = candidate;
        factor = dense;
      }
    }
    if (!read || read.getMask() || !allInBounds(read) ||
        !directIndices(read.getIndices()) ||
        !read.getPermutationMap().isIdentity() ||
        !read->getResult(0).hasOneUse())
      return;
    auto src = read.getSource().getDefiningOp<memref::SubViewOp>();
    auto dst = store.getSource().getDefiningOp<memref::SubViewOp>();
    auto input =
        src ? dyn_cast<BlockArgument>(src.getSource()) : BlockArgument();
    auto output =
        dst ? dyn_cast<BlockArgument>(dst.getSource()) : BlockArgument();
    if (!input || !output || input.getOwner() != &function.getBody().front() ||
        output.getOwner() != input.getOwner() || input == output ||
        !isTileType(input.getType(), columns) ||
        input.getType() != output.getType() || !input.hasOneUse() ||
        !output.hasOneUse() || !src.getResult().hasOneUse() ||
        !dst.getResult().hasOneUse() || !coversTile(src, read, columns) ||
        !coversTile(dst, store, columns) ||
        src.getMixedOffsets() != dst.getMixedOffsets() ||
        src.getMixedSizes() != dst.getMixedSizes() ||
        src.getMixedStrides() != dst.getMixedStrides())
      return;
    unsigned reductions = 0, stores = 0;
    for (Operation *user : multiply->getResult(0).getUsers()) {
      if (user == store.getOperation())
        ++stores;
      else if (isa<arith::MaximumFOp>(user))
        ++reductions;
      else
        return;
    }
    if (stores != 1 || reductions != 1)
      return;
    matches.push_back({function, read, multiply, store, factor,
                       input.getArgNumber(), output.getArgNumber()});
  });
  if (matches.size() != 1)
    return std::nullopt;
  return matches.front();
}

/// Stage 1B must prove the complete maximum chain. Stage 1A alone does not
/// require these additional constraints and retains its original matching.
struct MaximumChain {
  arith::MaximumFOp laneMaximum;
  vector::MultiDimReductionOp horizontalMaximum;
  vector::TransferWriteOp store;
};

bool isNegativeInfinity(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  auto dense = constant ? dyn_cast<DenseElementsAttr>(constant.getValue())
                        : DenseElementsAttr();
  if (!dense || !dense.isSplat() || !dense.getElementType().isF32())
    return false;
  APFloat number = dense.getSplatValue<APFloat>();
  return number.isInfinity() && number.isNegative();
}

std::optional<MaximumChain> matchMaximumChain(Producer producer,
                                              bool enableTritonKernelCompile) {
  arith::MaximumFOp maximum;
  for (Operation *user : producer.multiply->getResult(0).getUsers())
    if (auto candidate = dyn_cast<arith::MaximumFOp>(user))
      maximum = candidate;
  auto columnLoop = producer.read->getParentOfType<scf::ForOp>();
  auto rowLoop =
      columnLoop ? columnLoop->getParentOfType<scf::ForOp>() : scf::ForOp();
  if (!maximum || !columnLoop || !rowLoop ||
      maximum->getParentOp() != columnLoop || columnLoop.getNumResults() != 1 ||
      columnLoop.getInitArgs().size() != 1 ||
      !isNegativeInfinity(columnLoop.getInitArgs()[0]))
    return std::nullopt;
  Value carry = columnLoop.getRegionIterArgs()[0];
  Value scaled = producer.multiply->getResult(0);
  if (!carry.hasOneUse() ||
      !((maximum.getLhs() == scaled && maximum.getRhs() == carry) ||
        (maximum.getRhs() == scaled && maximum.getLhs() == carry)) ||
      !maximum->getResult(0).hasOneUse())
    return std::nullopt;
  auto yield = dyn_cast<scf::YieldOp>(columnLoop.getBody()->getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != maximum->getResult(0) ||
      !columnLoop.getResult(0).hasOneUse())
    return std::nullopt;
  auto horizontal = dyn_cast<vector::MultiDimReductionOp>(
      *columnLoop.getResult(0).user_begin());
  auto resultType = horizontal ? dyn_cast<VectorType>(horizontal.getDestType())
                               : VectorType();
  if (!horizontal || horizontal->getParentOp() != rowLoop ||
      horizontal.getSource() != columnLoop.getResult(0) ||
      horizontal.getKind() != vector::CombiningKind::MAXIMUMF ||
      horizontal.getReductionDims().size() != 1 ||
      cast<IntegerAttr>(horizontal.getReductionDims()[0]).getInt() != 1 ||
      !resultType || resultType.isScalable() ||
      resultType.getShape() != ArrayRef<int64_t>{1} ||
      !resultType.getElementType().isF32() ||
      !horizontal->getResult(0).hasOneUse())
    return std::nullopt;
  // NormalizeVector forwards this attribute. VectorToHIVMAVE ignores the
  // accumulator only when BOTH this marker and actual Triton mode are set.
  if (!(enableTritonKernelCompile &&
        horizontal->hasAttr("withoutInitMergeOp")) &&
      !isNegativeInfinity(horizontal.getAcc()))
    return std::nullopt;
  auto store =
      dyn_cast<vector::TransferWriteOp>(*horizontal->getResult(0).user_begin());
  if (!store || store->getParentOp() != rowLoop || store.getMask() ||
      store.getVector() != horizontal->getResult(0) ||
      !store.getPermutationMap().isIdentity() ||
      store.getIndices().size() != 1 || constantInt(store.getIndices()[0]) != 0)
    return std::nullopt;
  auto bounds = store->getAttrOfType<ArrayAttr>("in_bounds");
  if (!bounds || bounds.size() != 1 || !isa<BoolAttr>(bounds[0]) ||
      !cast<BoolAttr>(bounds[0]).getValue())
    return std::nullopt;
  auto view = store.getSource().getDefiningOp<memref::SubViewOp>();
  auto output =
      view ? dyn_cast<BlockArgument>(view.getSource()) : BlockArgument();
  auto outputType =
      output ? dyn_cast<MemRefType>(output.getType()) : MemRefType();
  if (!view || !output ||
      output.getOwner() != &producer.function.getBody().front() ||
      !output.hasOneUse() || !outputType ||
      outputType.getShape() != ArrayRef<int64_t>{64} ||
      !outputType.getElementType().isF32() ||
      !outputType.getLayout().isIdentity() ||
      outputType.getMemorySpace() !=
          AddressSpaceAttr::get(outputType.getContext(), AddressSpace::UB) ||
      view.getMixedOffsets().size() != 1 ||
      view.getMixedOffsets()[0] != OpFoldResult(rowLoop.getInductionVar()) ||
      getConstantIntValue(view.getMixedSizes()[0]) != 1 ||
      getConstantIntValue(view.getMixedStrides()[0]) != 1)
    return std::nullopt;
  // The only optional other use of this row output is the original horizontal
  // accumulator load. Preserve that load even when downstream ignores it.
  for (Operation *user : view.getResult().getUsers()) {
    if (user == store.getOperation())
      continue;
    auto read = dyn_cast<vector::TransferReadOp>(user);
    if (!read || read->getParentOp() != rowLoop ||
        read->getResult(0) != horizontal.getAcc() ||
        !read->getResult(0).hasOneUse())
      return std::nullopt;
  }
  return MaximumChain{maximum, horizontal, store};
}

struct Consumer {
  func::FuncOp function;
  vector::TransferReadOp read;
  arith::SubFOp subtract;
  unsigned inputIndex;
};

std::optional<Consumer> matchConsumer(func::FuncOp function, int64_t columns) {
  if (!hasOnlyArgumentMemory(function))
    return std::nullopt;
  SmallVector<Consumer> matches;
  function.walk([&](vector::TransferReadOp read) {
    if (!isTileVector(read.getType()) || read.getMask() || !allInBounds(read) ||
        !directIndices(read.getIndices()) ||
        !read.getPermutationMap().isIdentity() ||
        !read->getResult(0).hasOneUse())
      return;
    auto view = read.getSource().getDefiningOp<memref::SubViewOp>();
    auto input =
        view ? dyn_cast<BlockArgument>(view.getSource()) : BlockArgument();
    if (!input || input.getOwner() != &function.getBody().front() ||
        !isTileType(input.getType(), columns) || !input.hasOneUse() ||
        !view.getResult().hasOneUse() || !coversTile(view, read, columns))
      return;
    auto subtract = dyn_cast<arith::SubFOp>(*read->getResult(0).user_begin());
    if (!subtract || subtract.getLhs() != read->getResult(0) ||
        !subtract->getResult(0).hasOneUse())
      return;
    auto exponential =
        dyn_cast<math::ExpOp>(*subtract->getResult(0).user_begin());
    if (!exponential)
      return;
    bool hasSum = false, hasCast = false;
    for (Operation *user : exponential->getResult(0).getUsers()) {
      if (isa<arith::AddFOp>(user))
        hasSum = true;
      else if (auto cast = dyn_cast<arith::TruncFOp>(user)) {
        auto type = dyn_cast<VectorType>(cast.getType());
        hasCast = type && type.getElementType().isF16();
      } else
        return;
    }
    if (hasSum && hasCast)
      matches.push_back({function, read, subtract, input.getArgNumber()});
  });
  if (matches.size() != 1)
    return std::nullopt;
  return matches.front();
}

bool privateUses(Value allocation, func::CallOp producer, const Producer &p,
                 func::CallOp consumer, const Consumer &c) {
  if (!allocation.getDefiningOp<memref::AllocOp>())
    return false;
  for (OpOperand &use : allocation.getUses()) {
    if (use.getOwner() == producer.getOperation()) {
      if (use.getOperandNumber() != p.inputIndex &&
          use.getOperandNumber() != p.outputIndex)
        return false;
    } else if (use.getOwner() == consumer.getOperation()) {
      if (use.getOperandNumber() != c.inputIndex)
        return false;
    } else if (!isa<annotation::MarkOp>(use.getOwner())) {
      return false;
    }
  }
  return true;
}

bool safeInterval(func::CallOp producer, func::CallOp consumer,
                  ModuleOp module) {
  if (producer->getBlock() != consumer->getBlock() ||
      !producer->isBeforeInBlock(consumer))
    return false;
  auto scope = producer->getParentOfType<scope::ScopeOp>();
  if (!scope || consumer->getParentOp() != scope ||
      producer->getParentOp() != scope ||
      scope->getAttr(kPipelinedLoopCoreTypeAttrName) !=
          TCoreTypeAttr::get(module.getContext(), TCoreType::VECTOR))
    return false;
  for (Operation *op = producer->getNextNode(); op != consumer;
       op = op->getNextNode()) {
    if (!op)
      return false;
    if (auto call = dyn_cast<func::CallOp>(op)) {
      auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
      if (!function || !hasOnlyArgumentMemory(function))
        return false;
      continue;
    }
    if (isa<memref::AllocOp, annotation::MarkOp, AnchorOp>(op))
      continue;
    // In particular, never cross an event, copy or unknown effect boundary.
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      return false;
  }
  return true;
}

// A distinct scratch adds a new raw reader, so prove its complete slot/phase
// relation. The in-place route above intentionally keeps its original proof.
struct SlotMarks {
  HIVMTightlyCoupledBufferAttr slot;
  DictionaryAttr coupling;
  DictionaryAttr buffering;
};

std::optional<SlotMarks> slotMarks(Value value, bool tiled) {
  SlotMarks result;
  for (Operation *user : value.getUsers()) {
    auto mark = dyn_cast<annotation::MarkOp>(user);
    if (!mark)
      continue;
    if (auto slot = mark->getAttrOfType<HIVMTightlyCoupledBufferAttr>(
            HIVMTightlyCoupledBufferAttr::name)) {
      if (result.slot || !slot.getId())
        return std::nullopt;
      result.slot = slot;
      result.coupling = mark->getAttrDictionary();
      if (tiled) {
        auto dim = mark->getAttrOfType<IntegerAttr>("hivm.tiling_dim");
        if (!dim || dim.getInt() != 0 || !mark->hasAttr("tiledAlloc"))
          return std::nullopt;
      }
    } else if (mark->hasAttr(MultiBufferAttr::name) ||
               mark->hasAttr(PreloadLocalBufferAttr::name)) {
      if (result.buffering)
        return std::nullopt;
      auto banks = mark->getAttrOfType<IntegerAttr>(MultiBufferAttr::name);
      auto preload =
          mark->getAttrOfType<IntegerAttr>(PreloadLocalBufferAttr::name);
      if (!banks || banks.getInt() != 2 || !preload || preload.getInt() != 1)
        return std::nullopt;
      result.buffering = mark->getAttrDictionary();
    } else {
      return std::nullopt;
    }
  }
  if (!result.slot || !result.buffering)
    return std::nullopt;
  return result;
}

/// Anchor identity includes all enclosing query/key loops and scopes. Equal
/// inner bounds alone cannot identify the two halves of a mixed kernel.
std::optional<SmallVector<int64_t>> anchorLayout(func::FuncOp function) {
  SmallVector<int64_t> layout;
  llvm::SmallDenseSet<int64_t> seen;
  bool valid = true;
  function.walk([&](AnchorOp anchor) {
    auto id = anchor->getAttrOfType<IntegerAttr>("id");
    if (!id || !seen.insert(id.getInt()).second) {
      valid = false;
      return;
    }
    layout.push_back(id.getInt());
    SmallVector<Operation *> parents;
    for (Operation *parent = anchor->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
      if (!isa<scf::ForOp, scope::ScopeOp>(parent)) {
        valid = false;
        return;
      }
      parents.push_back(parent);
    }
    layout.push_back(parents.size());
    for (Operation *parent : llvm::reverse(parents)) {
      auto anchors = parent->getRegion(0).front().getOps<AnchorOp>();
      if (anchors.empty()) {
        valid = false;
        return;
      }
      auto first = (*anchors.begin())->getAttrOfType<IntegerAttr>("id");
      if (!first) {
        valid = false;
        return;
      }
      layout.push_back(isa<scf::ForOp>(parent) ? 0 : 1);
      layout.push_back(first.getInt());
      if (isa<scope::ScopeOp>(parent)) {
        auto phase = parent->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
        auto maximum =
            parent->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
        auto core = parent->getAttrOfType<TCoreTypeAttr>(
            kPipelinedLoopCoreTypeAttrName);
        if (!phase || !maximum || !core) {
          valid = false;
          return;
        }
        layout.push_back(phase.getInt());
        layout.push_back(maximum.getInt());
        layout.push_back(
            core == TCoreTypeAttr::get(function.getContext(), TCoreType::CUBE)
                ? 0
                : 1);
      }
    }
  });
  if (!valid || layout.empty())
    return std::nullopt;
  return layout;
}

SmallVector<scope::ScopeOp> orderedScopes(scf::ForOp loop) {
  SmallVector<scope::ScopeOp> scopes(loop.getBody()->getOps<scope::ScopeOp>());
  if (scopes.size() != 4)
    return {};
  for (unsigned i = 0; i != 4; ++i) {
    auto phase = scopes[i]->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto maximum =
        scopes[i]->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name);
    if (!phase || phase.getInt() != 3 - i || !maximum ||
        maximum.getInt() != 4 ||
        scopes[i]->getAttr(kPipelinedLoopCoreTypeAttrName) !=
            TCoreTypeAttr::get(loop.getContext(),
                               i % 2 ? TCoreType::VECTOR : TCoreType::CUBE))
      return {};
  }
  return scopes;
}

struct ChannelEvent {
  bool set;
  unsigned gap;
  PIPE from, to;
  IntegerAttr flag;
  bool operator==(const ChannelEvent &other) const {
    return set == other.set && gap == other.gap && from == other.from &&
           to == other.to && flag == other.flag;
  }
};

std::optional<SmallVector<ChannelEvent>> channels(scf::ForOp loop,
                                                  TCoreType core) {
  SmallVector<ChannelEvent> events;
  bool nested = false;
  loop.walk([&](Operation *op) {
    if (isa<SyncBlockSetOp, SyncBlockWaitOp>(op) && op->getParentOp() != loop)
      nested = true;
  });
  if (nested)
    return std::nullopt;
  unsigned gap = 0;
  for (Operation &op : *loop.getBody()) {
    if (isa<scope::ScopeOp>(op)) {
      ++gap;
      continue;
    }
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      continue;
    auto from = op.getAttrOfType<PipeAttr>("tpipe");
    auto to = op.getAttrOfType<PipeAttr>("pipe");
    if (!from || !to)
      return std::nullopt;
    bool ready =
        from.getPipe() == PIPE::PIPE_FIX && to.getPipe() == PIPE::PIPE_V;
    bool credit =
        from.getPipe() == PIPE::PIPE_V && to.getPipe() == PIPE::PIPE_FIX;
    if (!ready && !credit)
      continue;
    auto mode = isa<SyncBlockSetOp>(op)
                    ? cast<SyncBlockSetOp>(&op).getTsyncInstrModeAttr()
                    : cast<SyncBlockWaitOp>(&op).getTsyncInstrModeAttr();
    // The optional attribute defaults to INTRA_BLOCK_SYNCHRONIZATION, exactly
    // as eraseOldIntraBlockSyncOps interprets it before final regeneration.
    if (mode && mode.getSyncInstrMode() !=
                    SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION)
      return std::nullopt;
    auto actualCore = op.getAttrOfType<TCoreTypeAttr>("tcore_type");
    auto expectedCore = TCoreTypeAttr::get(loop.getContext(), core);
    if (actualCore != expectedCore) {
      bool backup = loop->getParentOfType<func::FuncOp>()->hasAttr(
          "hivm.backup_function");
      if (!backup || (actualCore != TCoreTypeAttr::get(loop.getContext(),
                                                       TCoreType::CUBE) &&
                      actualCore != TCoreTypeAttr::get(loop.getContext(),
                                                       TCoreType::VECTOR)))
        return std::nullopt;
      continue;
    }
    auto flag = op.getAttrOfType<IntegerAttr>("static_flag_id");
    if (!flag || flag.getInt() < 0 || op.getNumOperands() != 0)
      return std::nullopt;
    events.push_back(
        {isa<SyncBlockSetOp>(op), gap, from.getPipe(), to.getPipe(), flag});
  }
  return events;
}

/// Cross-core IDs share one namespace per originating core even when the pipe
/// pairs differ (SyncSolverBase::getEventIdSolverRef drops both pipe keys).
/// Check the whole function, including initial credits and final drains, so a
/// foreign MTE event cannot consume or replace a selected slot's credit/ready.
bool isolatedOwnershipEvents(func::FuncOp function, scf::ForOp loop,
                             ArrayRef<ChannelEvent> cubeEvents) {
  auto core = function->getAttrOfType<TFuncCoreTypeAttr>("hivm.func_core_type");
  bool backup = function->hasAttr("hivm.backup_function");
  bool hasCube =
      backup || (core && core.getFuncCoreType() == TFuncCoreType::AIC);
  bool hasVector =
      backup || (core && core.getFuncCoreType() == TFuncCoreType::AIV);
  if (!hasCube && !hasVector)
    return false;
  Operation *outerLoop = loop;
  while (outerLoop->getParentOp() != function) {
    outerLoop = outerLoop->getParentOp();
    if (!isa<scf::ForOp>(outerLoop))
      return false;
  }
  unsigned initialCredits[2] = {0, 0};
  unsigned finalCredits[2] = {0, 0};
  bool valid = true;
  function.walk([&](Operation *op) {
    if (isa<SyncBlockOp>(op)) {
      // Generic barriers do not identify the origin/slot relation used here.
      valid = false;
      return;
    }
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return;
    bool set = isa<SyncBlockSetOp>(op);
    auto actualCore = op->getAttrOfType<TCoreTypeAttr>("tcore_type");
    auto from = op->getAttrOfType<PipeAttr>("tpipe");
    auto to = op->getAttrOfType<PipeAttr>("pipe");
    auto flag = op->getAttrOfType<IntegerAttr>("static_flag_id");
    auto mode = set ? cast<SyncBlockSetOp>(op).getTsyncInstrModeAttr()
                    : cast<SyncBlockWaitOp>(op).getTsyncInstrModeAttr();
    if (!actualCore || !from || !to || !flag || flag.getInt() < 0 ||
        op->getNumOperands() != 0 ||
        (mode && mode.getSyncInstrMode() !=
                     SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION)) {
      valid = false;
      return;
    }
    bool onCube = actualCore.getTcoretype() == TCoreType::CUBE;
    bool onVector = actualCore.getTcoretype() == TCoreType::VECTOR;
    if ((!onCube && !onVector) || (onCube && !hasCube) ||
        (onVector && !hasVector)) {
      valid = false;
      return;
    }
    // A wait runs on the receiving core, so invert its core for the key.
    bool originCube = set ? onCube : onVector;
    bool ready =
        from.getPipe() == PIPE::PIPE_FIX && to.getPipe() == PIPE::PIPE_V;
    bool credit =
        from.getPipe() == PIPE::PIPE_V && to.getPipe() == PIPE::PIPE_FIX;
    bool selected = false;
    for (unsigned i = originCube ? 1 : 0; i < 4; i += 2)
      selected |= flag.getInt() == cubeEvents[i].flag.getInt();
    if (!ready && !credit) {
      if (selected)
        valid = false;
      return;
    }
    if ((ready && !originCube) || (credit && originCube) || !selected) {
      valid = false;
      return;
    }
    if (op->getParentOp() == loop)
      return; // The complete ordered stream is checked by channels().
    if (!credit || op->getParentOp() != function) {
      valid = false;
      return;
    }
    unsigned slot = flag.getInt() == cubeEvents[0].flag.getInt() ? 0 : 1;
    if (set && onVector && op->isBeforeInBlock(outerLoop))
      ++initialCredits[slot];
    else if (!set && onCube && outerLoop->isBeforeInBlock(op))
      ++finalCredits[slot];
    else
      valid = false;
  });
  return valid && initialCredits[0] == unsigned(hasVector) &&
         initialCredits[1] == unsigned(hasVector) &&
         finalCredits[0] == unsigned(hasCube) &&
         finalCredits[1] == unsigned(hasCube);
}

Value peelTensorShape(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

std::optional<unsigned> matrixInputArgument(Value value) {
  value = peelTensorShape(value);
  if (auto tensor = value.getDefiningOp<bufferization::ToTensorOp>())
    value = tensor->getOperand(0);
  if (!value.getDefiningOp<memref::AllocOp>())
    return std::nullopt;
  ND2NZOp load;
  for (Operation *user : value.getUsers()) {
    if (auto candidate = dyn_cast<ND2NZOp>(user)) {
      if (load || candidate.getDst() != value)
        return std::nullopt;
      load = candidate;
    } else if (!isa<annotation::MarkOp, bufferization::ToTensorOp, MmadL1Op>(
                   user)) {
      return std::nullopt;
    }
  }
  if (!load)
    return std::nullopt;
  value = load.getSrc();
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::ReinterpretCastOp, memref::CastOp,
             memref::MemorySpaceCastOp>(op))
      return std::nullopt;
    value = op->getOperand(0);
  }
  auto argument = dyn_cast<BlockArgument>(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!argument || !isa<func::FuncOp>(argument.getOwner()->getParentOp()) ||
      !type || !type.getElementType().isF16())
    return std::nullopt;
  return argument.getArgNumber();
}

std::optional<int64_t> firstAnchor(Block &block) {
  auto anchors = block.getOps<AnchorOp>();
  if (anchors.empty())
    return std::nullopt;
  auto id = (*anchors.begin())->getAttrOfType<IntegerAttr>("id");
  return id ? std::optional<int64_t>(id.getInt()) : std::nullopt;
}

bool sameGenerationValue(Value left, Value right, unsigned depth = 0) {
  if (depth > 64 || left.getType() != right.getType())
    return false;
  auto leftArg = dyn_cast<BlockArgument>(left);
  auto rightArg = dyn_cast<BlockArgument>(right);
  if (leftArg || rightArg) {
    if (!leftArg || !rightArg ||
        leftArg.getArgNumber() != rightArg.getArgNumber())
      return false;
    auto *leftOwner = leftArg.getOwner()->getParentOp();
    auto *rightOwner = rightArg.getOwner()->getParentOp();
    if (isa<func::FuncOp>(leftOwner) && isa<func::FuncOp>(rightOwner))
      return isa<IntegerType>(left.getType()) || left.getType().isIndex();
    auto leftLoop = dyn_cast<scf::ForOp>(leftOwner);
    auto rightLoop = dyn_cast<scf::ForOp>(rightOwner);
    if (!leftLoop || !rightLoop || leftArg != leftLoop.getInductionVar() ||
        rightArg != rightLoop.getInductionVar())
      return false;
    auto anchor = firstAnchor(*leftLoop.getBody());
    return anchor && firstAnchor(*rightLoop.getBody()) == anchor;
  }
  auto *a = left.getDefiningOp();
  auto *b = right.getDefiningOp();
  if (!a || !b || a->getName() != b->getName() ||
      a->getAttrDictionary() != b->getAttrDictionary() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getNumResults() != b->getNumResults() ||
      cast<OpResult>(left).getResultNumber() !=
          cast<OpResult>(right).getResultNumber() ||
      !isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
           arith::DivSIOp, arith::RemSIOp, arith::IndexCastOp, arith::ExtSIOp,
           arith::TruncIOp, arith::MaxSIOp, arith::MinSIOp,
           affine::AffineApplyOp, GetBlockIdxOp>(a))
    return false;
  for (auto [x, y] : llvm::zip(a->getOperands(), b->getOperands()))
    if (!sameGenerationValue(x, y, depth + 1))
      return false;
  return true;
}

bool sameLoopGenerations(func::FuncOp left, func::FuncOp right) {
  SmallVector<scf::ForOp> a, b;
  left.walk([&](scf::ForOp loop) { a.push_back(loop); });
  right.walk([&](scf::ForOp loop) { b.push_back(loop); });
  if (a.size() != b.size())
    return false;
  for (auto [x, y] : llvm::zip(a, b)) {
    auto anchor = firstAnchor(*x.getBody());
    if (!anchor || firstAnchor(*y.getBody()) != anchor ||
        !sameGenerationValue(x.getLowerBound(), y.getLowerBound()) ||
        !sameGenerationValue(x.getUpperBound(), y.getUpperBound()) ||
        !sameGenerationValue(x.getStep(), y.getStep()))
      return false;
  }
  return true;
}

/// Bind QK by the retained tensor graph, not equal QK/PV dimensions or slot
/// ordinal: QK -> scale -> max/sub/exp -> FP16 P -> copy -> the other matmul.
bool backupQKRole(Value raw, FixpipeOp fixpipe,
                  ArrayRef<scope::ScopeOp> scopes) {
  bufferization::ToTensorOp tensor;
  for (Operation *user : raw.getUsers()) {
    if (isa<annotation::MarkOp>(user) || user == fixpipe)
      continue;
    auto candidate = dyn_cast<bufferization::ToTensorOp>(user);
    if (!candidate || tensor || !candidate->hasAttr("restrict"))
      return false;
    tensor = candidate;
  }
  if (!tensor || !tensor->getResult(0).hasOneUse())
    return false;
  auto scale = dyn_cast<VMulOp>(*tensor->getResult(0).user_begin());
  if (!scale || scale->getParentOp() != scopes[1] ||
      scale.getSrc().size() != 2 || scale.getDst().size() != 1 ||
      scale->getNumResults() != 1)
    return false;
  bool half = false;
  for (unsigned i : {0u, 1u}) {
    auto constant = scale.getSrc()[i].getDefiningOp<arith::ConstantOp>();
    auto attr =
        constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
    half |= attr && attr.getType().isF32() && attr.getValueAsDouble() == 0.5 &&
            scale.getSrc()[1 - i] == tensor->getResult(0);
  }
  VReduceOp maximum;
  VSubOp subtract;
  for (Operation *user : scale->getResult(0).getUsers()) {
    if (auto reduce = dyn_cast<VReduceOp>(user)) {
      if (maximum || reduce->getNumResults() != 1 ||
          reduce.getSrc() != scale->getResult(0) ||
          reduce.getArith().getReduceOp() != ReduceOperation::max)
        return false;
      maximum = reduce;
    } else if (auto sub = dyn_cast<VSubOp>(user)) {
      if (subtract || sub.getSrc().size() != 2 || sub.getDst().size() != 1 ||
          sub->getNumResults() != 1 || sub.getSrc()[0] != scale->getResult(0))
        return false;
      subtract = sub;
    } else {
      return false;
    }
  }
  if (!half || !maximum || !subtract || !subtract->getResult(0).hasOneUse())
    return false;
  auto combine = peelTensorShape(subtract.getSrc()[1]).getDefiningOp<VMaxOp>();
  if (!combine || combine.getSrc().size() != 2 ||
      combine.getDst().size() != 1 || combine->getNumResults() != 1 ||
      !llvm::any_of(combine.getSrc(), [&](Value operand) {
        return peelTensorShape(operand) == maximum->getResult(0);
      }))
    return false;
  auto exp = dyn_cast<VExpOp>(*subtract->getResult(0).user_begin());
  if (!exp || exp.getSrc().size() != 1 || exp.getDst().size() != 1 ||
      exp->getNumResults() != 1 || exp.getSrc()[0] != subtract->getResult(0))
    return false;
  VCastOp cast;
  for (Operation *user : exp->getResult(0).getUsers()) {
    if (auto candidate = dyn_cast<VCastOp>(user)) {
      if (cast || candidate.getSrc().size() != 1 ||
          candidate.getDst().size() != 1 || candidate->getNumResults() != 1 ||
          candidate.getSrc()[0] != exp->getResult(0))
        return false;
      cast = candidate;
    } else if (auto sum = dyn_cast<VReduceOp>(user)) {
      if (sum->getNumResults() != 1 || sum.getSrc() != exp->getResult(0) ||
          sum.getArith().getReduceOp() != ReduceOperation::sum)
        return false;
    } else {
      return false;
    }
  }
  auto probabilityType =
      cast ? dyn_cast<RankedTensorType>(cast->getResult(0).getType())
           : RankedTensorType();
  if (!probabilityType || !probabilityType.getElementType().isF16())
    return false;
  auto pvScope = scopes[2];
  SmallVector<MmadL1Op> matrices(
      pvScope.getRegion().front().getOps<MmadL1Op>());
  if (matrices.size() != 1)
    return false;
  for (Value input : {matrices[0].getA(), matrices[0].getB()}) {
    auto read =
        peelTensorShape(input).getDefiningOp<bufferization::ToTensorOp>();
    if (!read)
      continue;
    for (Operation *user : read->getOperand(0).getUsers()) {
      auto copy = dyn_cast<CopyOp>(user);
      if (!copy || copy->getNumResults() != 0 ||
          copy.getDst() != read->getOperand(0) ||
          copy->getParentOp() != scopes[1])
        continue;
      auto transpose =
          peelTensorShape(copy.getSrc()).getDefiningOp<VTransposeOp>();
      if (transpose && transpose->getNumResults() == 1 &&
          peelTensorShape(transpose.getSrc()) == cast->getResult(0))
        return true;
    }
  }
  return false;
}

bool distinctSlotLifetime(ModuleOp module, func::FuncOp caller,
                          func::CallOp producer, func::CallOp consumer,
                          Value raw, int64_t keyCount) {
  auto readerMarks = slotMarks(raw, true);
  auto scope = producer->getParentOfType<scope::ScopeOp>();
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp();
  if (!readerMarks || !loop || consumer->getParentOp() != scope ||
      raw.getDefiningOp()->getParentOp() != loop)
    return false;
  auto vectorScopes = orderedScopes(loop);
  if (vectorScopes.empty() || vectorScopes[1] != scope)
    return false;
  func::FuncOp cube, backup;
  unsigned vectorCount = 0;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function->hasAttr("hivm.backup_function")) {
      if (backup ||
          function->getAttr("hivm.func_core_type") !=
              TFuncCoreTypeAttr::get(module.getContext(), TFuncCoreType::MIX))
        return false;
      backup = function;
    } else if (function->hasAttr("hivm.part_of_mix")) {
      auto core = function->getAttr("hivm.func_core_type");
      if (core ==
          TFuncCoreTypeAttr::get(module.getContext(), TFuncCoreType::AIC)) {
        if (cube)
          return false;
        cube = function;
      } else if (core == TFuncCoreTypeAttr::get(module.getContext(),
                                                TFuncCoreType::AIV)) {
        if (function != caller)
          return false;
        ++vectorCount;
      } else {
        return false;
      }
    }
  }
  if (!cube || !backup || vectorCount != 1 ||
      cube->getAttr(kCVPAttentionVerifiedAttr) !=
          caller->getAttr(kCVPAttentionVerifiedAttr))
    return false;
  auto layout = anchorLayout(caller);
  if (!layout || anchorLayout(cube) != layout ||
      anchorLayout(backup) != layout || !sameLoopGenerations(caller, cube) ||
      !sameLoopGenerations(caller, backup))
    return false;
  auto findWriter = [&](func::FuncOp function, bool tiled) -> FixpipeOp {
    FixpipeOp writer;
    bool valid = true;
    unsigned roots = 0;
    function.walk([&](memref::AllocOp allocation) {
      bool target = llvm::any_of(
          allocation->getResult(0).getUsers(), [&](Operation *user) {
            return isa<annotation::MarkOp>(user) &&
                   user->getAttr(HIVMTightlyCoupledBufferAttr::name) ==
                       readerMarks->slot;
          });
      if (!target)
        return;
      ++roots;
      auto marks = slotMarks(allocation.getResult(), tiled);
      if (!marks || marks->slot != readerMarks->slot ||
          marks->buffering != readerMarks->buffering) {
        valid = false;
        return;
      }
      if (tiled && (allocation.getType() != raw.getType() ||
                    marks->coupling != readerMarks->coupling ||
                    marks->buffering != readerMarks->buffering)) {
        valid = false;
        return;
      }
      if (!tiled) {
        auto type = allocation.getType();
        NamedAttrList expected(readerMarks->coupling);
        expected.erase("hivm.tiling_dim");
        expected.erase("tiledAlloc");
        if (type.getShape() != ArrayRef<int64_t>{128, 128} ||
            !type.getElementType().isF32() || !type.getLayout().isIdentity() ||
            type.getMemorySpace() !=
                AddressSpaceAttr::get(module.getContext(), AddressSpace::UB) ||
            marks->coupling != expected.getDictionary(module.getContext())) {
          valid = false;
          return;
        }
      }
      for (Operation *user : allocation->getResult(0).getUsers()) {
        if (isa<annotation::MarkOp>(user) ||
            (!tiled && isa<bufferization::ToTensorOp>(user)))
          continue;
        auto fixpipe = dyn_cast<FixpipeOp>(user);
        if (!fixpipe || fixpipe.getDst() != allocation.getResult() || writer) {
          valid = false;
          return;
        }
        writer = fixpipe;
      }
    });
    return valid && roots == 1 ? writer : FixpipeOp();
  };
  auto writer = findWriter(cube, true);
  auto reference = findWriter(backup, false);
  if (!writer || !reference ||
      writer.getDualDstModeAttr() !=
          FixpipeDualDstModeAttr::get(module.getContext(),
                                      FixpipeDualDstMode::ROW_SPLIT))
    return false;
  bool uniqueRoots = true;
  module.walk([&](memref::AllocOp allocation) {
    for (Operation *user : allocation->getResult(0).getUsers())
      if (isa<annotation::MarkOp>(user) &&
          user->getAttr(HIVMTightlyCoupledBufferAttr::name) ==
              readerMarks->slot &&
          allocation.getResult() != raw &&
          allocation.getResult() != writer.getDst() &&
          allocation.getResult() != reference.getDst())
        uniqueRoots = false;
  });
  if (!uniqueRoots)
    return false;
  auto cubeScope = writer->getParentOfType<scope::ScopeOp>();
  auto backupScope = reference->getParentOfType<scope::ScopeOp>();
  auto cubeLoop =
      cubeScope ? dyn_cast<scf::ForOp>(cubeScope->getParentOp()) : scf::ForOp();
  auto backupLoop = backupScope
                        ? dyn_cast<scf::ForOp>(backupScope->getParentOp())
                        : scf::ForOp();
  if (!cubeLoop || !backupLoop)
    return false;
  auto cubeScopes = orderedScopes(cubeLoop);
  auto backupScopes = orderedScopes(backupLoop);
  if (cubeScopes.empty() || backupScopes.empty() ||
      cubeScopes[0] != cubeScope || backupScopes[0] != backupScope ||
      !backupQKRole(reference.getDst(), reference, backupScopes))
    return false;
  auto loopAnchor = firstAnchor(*loop.getBody());
  if (!loopAnchor || firstAnchor(*cubeLoop.getBody()) != loopAnchor ||
      firstAnchor(*backupLoop.getBody()) != loopAnchor)
    return false;
  for (unsigned i = 0; i != 4; ++i) {
    auto anchor = firstAnchor(vectorScopes[i].getRegion().front());
    if (!anchor || firstAnchor(cubeScopes[i].getRegion().front()) != anchor ||
        firstAnchor(backupScopes[i].getRegion().front()) != anchor)
      return false;
  }
  SmallVector<MmadL1Op> matrices(
      cubeScope.getRegion().front().getOps<MmadL1Op>());
  auto referenceMatrix = reference.getSrc().getDefiningOp<MmadL1Op>();
  if (matrices.size() != 1 || matrices[0].getC() != writer.getSrc() ||
      !referenceMatrix ||
      matrices[0]->getAttrDictionary() != referenceMatrix->getAttrDictionary())
    return false;
  auto left = matrixInputArgument(matrices[0].getA());
  auto right = matrixInputArgument(matrices[0].getB());
  if (!left || !right || left == right ||
      matrixInputArgument(referenceMatrix.getA()) != left ||
      matrixInputArgument(referenceMatrix.getB()) != right)
    return false;
  for (scf::ForOp candidate : {loop, cubeLoop, backupLoop})
    if (constantInt(candidate.getLowerBound()) != 0 ||
        constantInt(candidate.getUpperBound()) != keyCount ||
        constantInt(candidate.getStep()) != 128)
      return false;
  auto c = channels(cubeLoop, TCoreType::CUBE);
  auto v = channels(loop, TCoreType::VECTOR);
  if (!c || !v || c->size() != 4 || v->size() != 4 ||
      channels(backupLoop, TCoreType::CUBE) != c ||
      channels(backupLoop, TCoreType::VECTOR) != v)
    return false;
  for (unsigned i = 0; i != 4; ++i) {
    bool cubeReady = i % 2;
    bool vectorCredit = i % 2;
    if ((*c)[i].set != cubeReady || (*c)[i].gap != i ||
        (*c)[i].from != (cubeReady ? PIPE::PIPE_FIX : PIPE::PIPE_V) ||
        (*c)[i].to != (cubeReady ? PIPE::PIPE_V : PIPE::PIPE_FIX) ||
        (*v)[i].set != vectorCredit || (*v)[i].gap != i + 1 ||
        (*v)[i].from != (vectorCredit ? PIPE::PIPE_V : PIPE::PIPE_FIX) ||
        (*v)[i].to != (vectorCredit ? PIPE::PIPE_FIX : PIPE::PIPE_V))
      return false;
  }
  return (*c)[0].flag == (*v)[1].flag && (*c)[1].flag == (*v)[0].flag &&
         (*c)[2].flag == (*v)[3].flag && (*c)[3].flag == (*v)[2].flag &&
         (*c)[0].flag != (*c)[2].flag &&
         isolatedOwnershipEvents(cube, cubeLoop, *c) &&
         isolatedOwnershipEvents(caller, loop, *c) &&
         isolatedOwnershipEvents(backup, backupLoop, *c);
}

bool removableArgumentMetadata(func::FuncOp function, func::CallOp call) {
  for (NamedAttribute attr : function->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "sym_name" && name != "sym_visibility" &&
        name != "function_type" && name != "arg_attrs" && name != "res_attrs" &&
        name != "hivm.func_core_type" && name != "hivm.vector_function" &&
        name != "no_inline")
      return false;
  }
  auto functionInterface = cast<FunctionOpInterface>(function.getOperation());
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    for (NamedAttribute attr : functionInterface.getArgAttrs(i)) {
      StringRef name = attr.getName().getValue();
      if (name != "llvm.align" && name != "llvm.noalias" &&
          name != "llvm.readonly" && name != "llvm.writeonly")
        return false;
    }
  for (NamedAttribute attr : call->getAttrs()) {
    StringRef name = attr.getName().getValue();
    if (name != "callee" && name != "hivm.vector_function" &&
        name != "no_inline")
      return false;
  }
  return true;
}

struct DistinctScratch {
  func::CallOp producerCall, consumerCall;
  memref::AllocOp allocation;
  memref::SubViewOp destination;
};

std::optional<DistinctScratch>
matchDistinctScratch(ModuleOp module, func::FuncOp caller, func::CallOp first,
                     Producer producer, func::CallOp second, Consumer consumer,
                     int64_t keyCount) {
  Value raw = first.getOperand(producer.inputIndex);
  Value scratch = first.getOperand(producer.outputIndex);
  auto allocation = scratch.getDefiningOp<memref::AllocOp>();
  auto destination =
      producer.store.getSource().getDefiningOp<memref::SubViewOp>();
  if (raw == scratch || !raw.getDefiningOp<memref::AllocOp>() || !allocation ||
      !destination || second.getOperand(consumer.inputIndex) != scratch ||
      !removableArgumentMetadata(producer.function, first) ||
      !safeInterval(first, second, module))
    return std::nullopt;
  unsigned uses = 0;
  for (OpOperand &use : scratch.getUses()) {
    if ((use.getOwner() != first ||
         use.getOperandNumber() != producer.outputIndex) &&
        (use.getOwner() != second ||
         use.getOperandNumber() != consumer.inputIndex))
      return std::nullopt;
    ++uses;
  }
  if (uses != 2)
    return std::nullopt;
  for (OpOperand &use : raw.getUses())
    if (!isa<annotation::MarkOp>(use.getOwner()) &&
        (use.getOwner() != first ||
         use.getOperandNumber() != producer.inputIndex))
      return std::nullopt;
  for (Operation *op = first->getNextNode(); op != second;
       op = op->getNextNode())
    if (isa<SyncBlockOp, SyncBlockSetOp, SyncBlockWaitOp, SetFlagOp, WaitFlagOp,
            PipeBarrierOp>(op))
      return std::nullopt;
  if (!distinctSlotLifetime(module, caller, first, second, raw, keyCount))
    return std::nullopt;
  return DistinctScratch{first, second, allocation, destination};
}

struct Rewrite {
  Producer producer;
  Consumer consumer;
  std::optional<MaximumChain> maximum;
  std::optional<DistinctScratch> distinct;
};

struct CVPAttentionVectorPass
    : impl::CVPAttentionVectorBase<CVPAttentionVectorPass> {
  using Base::Base;
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (enableMaxScaling && !enableScoreRematerialization) {
      module.emitError(
          "CVP attention max scaling requires score rematerialization");
      signalPassFailure();
      return;
    }
    if (!enableScoreRematerialization)
      return;
    SmallVector<Rewrite> rewrites;
    llvm::SmallPtrSet<Operation *, 8> touched;
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      auto proof = caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
      if (caller->hasAttr("hivm.backup_function") || !proof ||
          proof.size() != 1 ||
          caller->getAttr("hivm.func_core_type") !=
              TFuncCoreTypeAttr::get(&getContext(), TFuncCoreType::AIV) ||
          !caller->hasAttr("hivm.part_of_mix"))
        continue;
      auto plan = dyn_cast<DictionaryAttr>(proof[0]);
      auto bm = plan ? plan.getAs<IntegerAttr>("block_m") : IntegerAttr();
      auto bn = plan ? plan.getAs<IntegerAttr>("block_n") : IntegerAttr();
      if (!bm || bm.getInt() != 128 || !bn ||
          (bn.getInt() != 128 && bn.getInt() != 256))
        continue;
      SmallVector<func::CallOp> calls;
      caller.walk([&](func::CallOp call) { calls.push_back(call); });
      for (func::CallOp first : calls) {
        auto producerFunction =
            module.lookupSymbol<func::FuncOp>(first.getCallee());
        if (!producerFunction ||
            producerFunction->hasAttr("hivm.backup_function") ||
            !uniqueCall(producerFunction, first, module))
          continue;
        auto p = matchProducer(producerFunction, bn.getInt());
        if (!p)
          continue;
        bool inPlace =
            first.getOperand(p->inputIndex) == first.getOperand(p->outputIndex);
        auto hd = plan.getAs<IntegerAttr>("head_dim");
        auto keys = plan.getAs<IntegerAttr>("key_count");
        if (!inPlace && (bn.getInt() != 128 || !hd || hd.getInt() != 128 ||
                         !keys || keys.getInt() <= 0 || keys.getInt() % 128))
          continue;
        std::optional<MaximumChain> maximum;
        if (enableMaxScaling) {
          maximum = matchMaximumChain(*p, enableTritonKernelCompile);
          if (!maximum)
            continue;
        }
        Value raw = first.getOperand(p->inputIndex);
        for (func::CallOp second : calls) {
          if (second == first)
            continue;
          auto consumerFunction =
              module.lookupSymbol<func::FuncOp>(second.getCallee());
          if (!consumerFunction ||
              consumerFunction->hasAttr("hivm.backup_function") ||
              !uniqueCall(consumerFunction, second, module))
            continue;
          auto c = matchConsumer(consumerFunction, bn.getInt());
          if (!c || touched.contains(producerFunction) ||
              touched.contains(consumerFunction))
            continue;
          std::optional<DistinctScratch> distinct;
          if (inPlace) {
            if (second.getOperand(c->inputIndex) != raw ||
                !privateUses(raw, first, *p, second, *c) ||
                !safeInterval(first, second, module))
              continue;
          } else {
            distinct = matchDistinctScratch(module, caller, first, *p, second,
                                            *c, keys.getInt());
            if (!distinct)
              continue;
          }
          rewrites.push_back({*p, *c, maximum, distinct});
          touched.insert(producerFunction);
          touched.insert(consumerFunction);
        }
      }
    }
    if (rewrites.empty() && requireMatch) {
      module.emitError(
          enableMaxScaling
              ? "CVP attention max scaling requires a verified private score/P "
                "chain and complete negative-infinity maximum reduction"
              : "CVP score rematerialization requires a verified private "
                "in-place score/P call chain or a proved distinct-scratch "
                "lifetime");
      signalPassFailure();
      return;
    }
    unsigned distinctCount = 0;
    for (Rewrite &rewrite : rewrites) {
      OpBuilder builder(rewrite.consumer.subtract);
      auto constant = builder.create<arith::ConstantOp>(
          rewrite.consumer.read.getLoc(), rewrite.producer.factor);
      IRMapping mapping;
      mapping.map(rewrite.producer.read->getResult(0),
                  rewrite.consumer.read->getResult(0));
      for (Value operand : rewrite.producer.multiply->getOperands())
        if (operand != rewrite.producer.read->getResult(0))
          mapping.map(operand, constant->getResult(0));
      auto multiply = cast<arith::MulFOp>(
          builder.clone(*rewrite.producer.multiply, mapping));
      rewrite.consumer.subtract->setOperand(0, multiply->getResult(0));
      if (rewrite.distinct)
        rewrite.distinct->consumerCall->setOperand(
            rewrite.consumer.inputIndex,
            rewrite.distinct->producerCall.getOperand(
                rewrite.producer.inputIndex));
      rewrite.producer.store.erase();
      if (rewrite.maximum) {
        MaximumChain &chain = *rewrite.maximum;
        chain.laneMaximum->replaceUsesOfWith(
            rewrite.producer.multiply->getResult(0),
            rewrite.producer.read->getResult(0));
        builder.setInsertionPoint(chain.store);
        auto reducedType =
            cast<VectorType>(chain.horizontalMaximum.getDestType());
        auto factor = DenseElementsAttr::get(
            reducedType, rewrite.producer.factor.getSplatValue<APFloat>());
        auto scalarConstant = builder.create<arith::ConstantOp>(
            rewrite.producer.multiply.getLoc(), factor);
        IRMapping reducedMapping;
        reducedMapping.map(rewrite.producer.read->getResult(0),
                           chain.horizontalMaximum->getResult(0));
        for (Value operand : rewrite.producer.multiply->getOperands())
          if (operand != rewrite.producer.read->getResult(0))
            reducedMapping.map(operand, scalarConstant->getResult(0));
        // Cloning preserves operand order and every original multiply
        // attribute.
        auto reducedMultiply = cast<arith::MulFOp>(
            builder.clone(*rewrite.producer.multiply, reducedMapping));
        reducedMultiply->getResult(0).setType(reducedType);
        chain.store->setOperand(0, reducedMultiply->getResult(0));
        rewrite.producer.multiply.erase();
      }
      if (rewrite.distinct) {
        auto &distinct = *rewrite.distinct;
        // The prospective proof established these are the only destination
        // uses. FunctionOpInterface updates standard argument attributes too.
        assert(distinct.destination.getResult().use_empty());
        distinct.destination.erase();
        assert(
            rewrite.producer.function.getArgument(rewrite.producer.outputIndex)
                .use_empty());
        distinct.producerCall.getOperandsMutable().erase(
            rewrite.producer.outputIndex);
        rewrite.producer.function.eraseArgument(rewrite.producer.outputIndex);
        assert(distinct.allocation.getResult().use_empty());
        distinct.allocation.erase();
        ++distinctCount;
      }
    }
    if (distinctCount)
      module->setAttr(
          kDistinctRemoved,
          IntegerAttr::get(IntegerType::get(&getContext(), 64), distinctCount));
    if (!rewrites.empty())
      module->setAttr(kRematerialized,
                      IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                       rewrites.size()));
    if (enableMaxScaling && !rewrites.empty())
      module->setAttr(kMaxScaled,
                      IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                       rewrites.size()));
    if (!retainProvenance)
      for (func::FuncOp function : module.getOps<func::FuncOp>())
        // The delayed solver consumes the frozen backup, including its attrs.
        if (!function->hasAttr("hivm.backup_function"))
          function->removeAttr(kCVPAttentionVerifiedAttr);
    if (distinctCount && failed(verify(module)))
      signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass>
createCVPAttentionVectorPass(const CVPAttentionVectorOptions &options) {
  return std::make_unique<CVPAttentionVectorPass>(options);
}
} // namespace hivm
} // namespace mlir

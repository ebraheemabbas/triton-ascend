//===- CVPAttentionPStream.cpp - Publish dense P through half scratch
//-------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONPSTREAM
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {
constexpr StringLiteral kDense = "hivm.cvp_attention.dense_p";
constexpr StringLiteral kPhase = "hivm.cvp_attention.half_p_phase";
constexpr StringLiteral kCount = "hivm.cvp_attention.half_p_streams";

bool onlyAttrs(Operation *op, ArrayRef<StringRef> names) {
  return llvm::all_of(op->getAttrs(), [&](NamedAttribute attr) {
    return llvm::is_contained(names, attr.getName().getValue());
  });
}

enum class Producer { Full, FirstHalf, SecondHalf };

MemRefType memory(OpBuilder &b, ArrayRef<int64_t> shape, Type element) {
  return MemRefType::get(
      shape, element, MemRefLayoutAttrInterface{},
      AddressSpaceAttr::get(b.getContext(), AddressSpace::UB));
}

MemRefType probabilityType(OpBuilder &b, bool half) {
  return MemRefType::get(
      {8, half ? 64 : 128, 16}, b.getF16Type(),
      StridedLayoutAttr::get(b.getContext(), 0, {half ? 1040 : 2064, 16, 1}),
      AddressSpaceAttr::get(b.getContext(), AddressSpace::UB));
}

Value offset(OpBuilder &b, Location loc, Value key, int64_t amount) {
  if (!amount)
    return key;
  auto map = AffineMap::get(0, 1, b.getAffineSymbolExpr(0) + amount);
  return b.create<affine::AffineApplyOp>(loc, map, ValueRange{key});
}

Value read(OpBuilder &b, Location loc, VectorType type, Value source,
           Value zero, Value padding) {
  auto rank = cast<MemRefType>(source.getType()).getRank();
  SmallVector<Value> indices(rank, zero);
  SmallVector<bool> bounds(type.getRank(), true);
  return b.create<vector::TransferReadOp>(
      loc, type, source, indices, padding,
      std::optional<ArrayRef<bool>>(bounds));
}

void write(OpBuilder &b, Location loc, Value value, Value destination,
           Value zero) {
  SmallVector<Value> indices(cast<MemRefType>(destination.getType()).getRank(),
                             zero);
  SmallVector<bool> bounds(cast<VectorType>(value.getType()).getRank(), true);
  b.create<vector::TransferWriteOp>(loc, value, destination, indices,
                                    std::optional<ArrayRef<bool>>(bounds));
}

Value rawRow(OpBuilder &b, Location loc, Value raw, Value key, Value zero,
             Value padding) {
  SmallVector<OpFoldResult> offsets{key, b.getIndexAttr(0)};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(1), b.getIndexAttr(64)};
  SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
  Value view = b.create<memref::SubViewOp>(loc, raw, offsets, sizes, strides);
  return read(b, loc, VectorType::get({1, 64}, b.getF32Type()), view, zero,
              padding);
}

// Build the supported full producer and both replacements from typed math.
// The full version is a structural reference: SSA operands, attributes,
// rounding, memory accesses and all regions must match, not merely op counts.
func::FuncOp buildProducer(OpBuilder &outer, Location loc, StringRef name,
                           Producer mode) {
  auto *ctx = outer.getContext();
  bool last = mode == Producer::SecondHalf;
  bool half = mode != Producer::Full;
  auto state = memory(outer, {64}, outer.getF32Type());
  SmallVector<Type> arguments{memory(outer, {256, 64}, outer.getF32Type()),
                              probabilityType(outer, half)};
  arguments.append(last ? 2 : 5, state);
  auto function = outer.create<func::FuncOp>(
      loc, name, outer.getFunctionType(arguments, {}));
  function->setAttr("hivm.func_core_type",
                    TFuncCoreTypeAttr::get(ctx, TFuncCoreType::AIV));
  function->setAttr("hivm.vector_function", outer.getUnitAttr());
  function->setAttr("no_inline", outer.getUnitAttr());
  if (!last)
    function->setAttr(
        kDense,
        outer.getDictionaryAttr(
            {outer.getNamedAttr("keys", outer.getI64IntegerAttr(256)),
             outer.getNamedAttr("padding_required", outer.getBoolAttr(true)),
             outer.getNamedAttr("query_rows", outer.getI64IntegerAttr(64))}));
  if (half)
    function->setAttr(kPhase, outer.getI64IntegerAttr(last ? 1 : 0));
  Block *entry = function.addEntryBlock();
  OpBuilder b(entry, entry->begin());
  auto flat = VectorType::get({64}, b.getF32Type());
  auto row = VectorType::get({1, 64}, b.getF32Type());
  auto constant = [&](VectorType type, FloatAttr value) -> Value {
    return b.create<arith::ConstantOp>(
        loc, type, DenseElementsAttr::get(type, ArrayRef<Attribute>{value}));
  };
  Value sumZero = constant(flat, b.getF32FloatAttr(0.0));
  Value scale, minusInf;
  if (!last) {
    scale = constant(flat, b.getF32FloatAttr(0.5));
    minusInf = constant(
        flat, FloatAttr::get(b.getF32Type(),
                             APFloat::getInf(APFloat::IEEEsingle(), true)));
  }
  Value rowScale = constant(row, b.getF32FloatAttr(0.5));
  Value padding =
      b.create<arith::ConstantOp>(loc, b.getF32Type(), b.getF32FloatAttr(0.0));
  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value sixtyFour = b.create<arith::ConstantIndexOp>(loc, 64);
  Value limit =
      half ? b.create<arith::ConstantIndexOp>(loc, 32).getResult() : sixtyFour;
  Value raw = function.getArgument(0), p = function.getArgument(1);
  Value oldM, newM;
  if (!last) {
    auto maximum = b.create<scf::ForOp>(
        loc, zero, sixtyFour, one, ValueRange{minusInf, minusInf},
        [&](OpBuilder &body, Location at, Value key, ValueRange carried) {
          auto maximumWith = [&](Value previous, Value index) -> Value {
            Value loaded = rawRow(body, at, raw, index, zero, padding);
            Value value = body.create<vector::ShapeCastOp>(at, flat, loaded);
            return body.create<arith::MaximumFOp>(at, previous, value);
          };
          Value a = maximumWith(carried[0], key);
          Value k64 = offset(body, at, key, 64);
          Value b0 = maximumWith(carried[1], k64);
          Value k128 = offset(body, at, key, 128);
          a = maximumWith(a, k128);
          Value k192 = offset(body, at, key, 192);
          b0 = maximumWith(b0, k192);
          body.create<scf::YieldOp>(at, ValueRange{a, b0});
        });
    maximum->setAttr("hivm.cvp_attention.reduction_chains",
                     b.getI64IntegerAttr(2));
    maximum->setAttr("hivm.cvp_attention.reduction_key_groups",
                     b.getI64IntegerAttr(4));
    Value tileMax = b.create<arith::MaximumFOp>(loc, maximum.getResult(0),
                                                maximum.getResult(1));
    Value scaled = b.create<arith::MulFOp>(loc, tileMax, scale);
    oldM = read(b, loc, flat, function.getArgument(3), zero, padding);
    newM = b.create<arith::MaximumFOp>(loc, oldM, scaled);
    write(b, loc, newM, function.getArgument(6), zero);
  } else {
    newM = read(b, loc, flat, function.getArgument(3), zero, padding);
  }
  Value maxRow = b.create<vector::ShapeCastOp>(loc, row, newM);
  auto probability = b.create<scf::ForOp>(
      loc, zero, limit, one, ValueRange{sumZero, sumZero},
      [&](OpBuilder &body, Location at, Value key, ValueRange carried) {
        auto value = [&](Value index) -> std::pair<Value, Value> {
          Value loaded = rawRow(body, at, raw, index, zero, padding);
          Value scaled = body.create<arith::MulFOp>(at, loaded, rowScale);
          Value shifted = body.create<arith::SubFOp>(at, scaled, maxRow);
          Value exp = body.create<math::ExpOp>(at, shifted);
          Value fp32 = body.create<vector::ShapeCastOp>(at, flat, exp);
          auto cast = body.create<arith::TruncFOp>(
              at, VectorType::get({1, 64}, body.getF16Type()), exp);
          cast->setAttr("round_mode", hfusion::RoundModeAttr::get(
                                          ctx, hfusion::RoundMode::RINT));
          Value fp16 = body.create<vector::ShapeCastOp>(
              at, VectorType::get({64}, body.getF16Type()), cast);
          return {fp32, fp16};
        };
        auto pair = [&](Value aKey, int64_t secondOffset, Value column,
                        Value initial) -> Value {
          auto a = value(aKey);
          Value bKey = offset(body, at, key, secondOffset);
          auto b0 = value(bKey);
          Value sum = body.create<arith::AddFOp>(at, a.first, b0.first);
          Value total = body.create<arith::AddFOp>(at, initial, sum);
          auto packedType = VectorType::get({128}, body.getF16Type());
          auto packed = body.create<hivmave::VFDeInterleaveOp>(
              at, packedType, packedType, a.second, b0.second,
              hivmave::Layout_Change::DENSE);
          Value vector = body.create<vector::ShapeCastOp>(
              at, VectorType::get({8, 1, 16}, body.getF16Type()),
              packed.getRes1());
          SmallVector<OpFoldResult> offsets{body.getIndexAttr(0), column,
                                            body.getIndexAttr(0)};
          SmallVector<OpFoldResult> sizes{body.getIndexAttr(8),
                                          body.getIndexAttr(1),
                                          body.getIndexAttr(16)};
          SmallVector<OpFoldResult> strides(3, body.getIndexAttr(1));
          Value view =
              body.create<memref::SubViewOp>(at, p, offsets, sizes, strides);
          write(body, at, vector, view, zero);
          return total;
        };
        int64_t base = last ? 64 : 0;
        int64_t distance = half ? 32 : 64;
        Value firstKey = offset(body, at, key, base);
        Value a = pair(firstKey, base + 128, key, carried[0]);
        Value secondKey = offset(body, at, key, base + distance);
        Value column =
            half && last ? offset(body, at, key, distance) : secondKey;
        Value b0 = pair(secondKey, base + distance + 128, column, carried[1]);
        body.create<scf::YieldOp>(at, ValueRange{a, b0});
      });
  probability->setAttr("hivm.cvp_attention.reduction_chains",
                       b.getI64IntegerAttr(2));
  probability->setAttr("hivm.cvp_attention.reduction_key_groups",
                       b.getI64IntegerAttr(2));
  Value sum = b.create<arith::AddFOp>(loc, probability.getResult(0),
                                      probability.getResult(1));
  Value previous;
  if (!last) {
    Value difference = b.create<arith::SubFOp>(loc, oldM, newM);
    Value alpha = b.create<math::ExpOp>(loc, difference);
    write(b, loc, alpha, function.getArgument(5), zero);
    Value oldL = read(b, loc, flat, function.getArgument(2), zero, padding);
    previous = b.create<arith::MulFOp>(loc, oldL, alpha);
  } else {
    previous = read(b, loc, flat, function.getArgument(2), zero, padding);
  }
  Value nextL = b.create<arith::AddFOp>(loc, previous, sum);
  write(b, loc, nextL, function.getArgument(last ? 2 : 4), zero);
  b.create<func::ReturnOp>(loc);
  return function;
}

bool sameProducer(func::FuncOp actual) {
  OwningOpRef<ModuleOp> reference = ModuleOp::create(actual.getLoc());
  OpBuilder inside(actual.getContext());
  inside.setInsertionPointToEnd(reference->getBody());
  auto expected = buildProducer(inside, actual.getLoc(), actual.getSymName(),
                                Producer::Full);
  if (failed(verify(expected)))
    return false;
  IRMapping values;
  return OperationEquivalence::isEquivalentTo(
      expected, actual,
      [&](Value a, Value b) { return success(values.lookupOrDefault(a) == b); },
      [&](Value a, Value b) { values.map(a, b); },
      OperationEquivalence::IgnoreLocations);
}

std::optional<int64_t> integer(Value value) {
  APInt n;
  if (!matchPattern(value, m_ConstantInt(&n)))
    return std::nullopt;
  return n.getSExtValue();
}

bool slice(memref::SubViewOp op, ArrayRef<int64_t> offsets,
           ArrayRef<int64_t> sizes) {
  if (!op || op.getMixedOffsets().size() != offsets.size() ||
      op.getMixedSizes().size() != sizes.size())
    return false;
  for (auto [i, value] : llvm::enumerate(op.getMixedOffsets()))
    if (getConstantIntValue(value) != offsets[i])
      return false;
  for (auto [i, value] : llvm::enumerate(op.getMixedSizes()))
    if (getConstantIntValue(value) != sizes[i])
      return false;
  return llvm::all_of(op.getMixedStrides(), [](OpFoldResult value) {
    return getConstantIntValue(value) == 1;
  });
}

bool isUB(Type type, ArrayRef<int64_t> shape, Type element) {
  auto memory = dyn_cast<MemRefType>(type);
  return memory && memory.getShape() == shape &&
         memory.getElementType() == element &&
         memory.getLayout().isIdentity() &&
         memory.getMemorySpace() ==
             AddressSpaceAttr::get(type.getContext(), AddressSpace::UB);
}

struct Publication {
  CopyOp copy;
  memref::SubViewOp source, destination;
  memref::ExpandShapeOp expand;
  memref::CollapseShapeOp flatSource, flatDestination;
};

struct Stream {
  func::FuncOp function;
  func::CallOp call;
  scope::ScopeOp scope;
  memref::AllocOp allocation;
  memref::SubViewOp view;
  Value l1;
  SmallVector<Publication> publications;
};

std::optional<Stream> matchStream(ModuleOp module, func::FuncOp function) {
  OpBuilder b(module.getContext());
  if (!sameProducer(function))
    return std::nullopt;
  auto uses = SymbolTable::getSymbolUses(function, module);
  if (!uses || !llvm::hasSingleElement(*uses))
    return std::nullopt;
  auto call = dyn_cast<func::CallOp>(uses->begin()->getUser());
  auto scope =
      call ? dyn_cast<scope::ScopeOp>(call->getParentOp()) : scope::ScopeOp{};
  auto loop = scope ? dyn_cast<scf::ForOp>(scope->getParentOp()) : scf::ForOp{};
  if (!call || call.getNumOperands() != 7 || call.getNumResults() || !scope ||
      !loop ||
      !onlyAttrs(call, {"callee", "hivm.vector_function", "no_inline"}) ||
      scope->getAttrOfType<TCoreTypeAttr>("hivm.loop_core_type") !=
          TCoreTypeAttr::get(b.getContext(), TCoreType::VECTOR) ||
      scope->getAttrOfType<IntegerAttr>("hivm.preload_num") !=
          b.getI32IntegerAttr(2) ||
      scope->getAttrOfType<IntegerAttr>("hivm.max_preload_num") !=
          b.getI32IntegerAttr(4) ||
      loop.getNumRegionIterArgs() != 3 || integer(loop.getLowerBound()) != 0 ||
      integer(loop.getStep()) != 256 || !integer(loop.getUpperBound()) ||
      *integer(loop.getUpperBound()) < 256 ||
      *integer(loop.getUpperBound()) % 256 ||
      call.getOperand(2) != loop.getRegionIterArgs()[1] ||
      call.getOperand(3) != loop.getRegionIterArgs()[2] ||
      !isUB(loop.getRegionIterArgs()[0].getType(), {64, 128}, b.getF32Type()))
    return std::nullopt;
  auto raw = call.getOperand(0).getDefiningOp<memref::AllocOp>();
  auto view = call.getOperand(1).getDefiningOp<memref::SubViewOp>();
  auto allocation = view ? view.getSource().getDefiningOp<memref::AllocOp>()
                         : memref::AllocOp{};
  if (!raw || raw->getParentOp() != loop || !allocation ||
      allocation->getParentOp() != scope ||
      allocation.getDynamicSizes().size() ||
      !allocation->getResult(0).hasOneUse() ||
      !onlyAttrs(allocation, {"alignment", "operandSegmentSizes"}) ||
      !isUB(allocation.getType(), {8, 129, 16, 1}, b.getF16Type()) ||
      !onlyAttrs(view, {"operandSegmentSizes", "static_offsets", "static_sizes",
                        "static_strides"}) ||
      !slice(view, {0, 0, 0, 0}, {8, 128, 16, 1}) ||
      view.getType() != probabilityType(b, false))
    return std::nullopt;
  auto initialL = loop.getInitArgs()[1].getDefiningOp<memref::AllocOp>();
  auto initialM = loop.getInitArgs()[2].getDefiningOp<memref::AllocOp>();
  if (!initialL || !initialM || initialL == initialM)
    return std::nullopt;
  auto ret =
      dyn_cast<scope::ReturnOp>(scope->getRegion(0).front().getTerminator());
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!ret || ret.getNumOperands() != 3 || scope.getNumResults() != 3 ||
      !yield || ret.getOperand(0) != call.getOperand(5) ||
      ret.getOperand(1) != call.getOperand(6) ||
      ret.getOperand(2) != call.getOperand(4) || yield.getNumOperands() != 3 ||
      yield.getOperand(1) != scope.getResult(2) ||
      yield.getOperand(2) != scope.getResult(1))
    return std::nullopt;
  llvm::SmallPtrSet<Operation *, 32> allowed{
      call.getOperation(), allocation.getOperation(), view.getOperation(),
      ret.getOperation()};
  for (unsigned index : {4u, 5u, 6u}) {
    auto state = call.getOperand(index).getDefiningOp<memref::AllocOp>();
    if (!state || state->getParentOp() != scope ||
        !isUB(state.getType(), {64}, b.getF32Type()))
      return std::nullopt;
    allowed.insert(state);
  }
  if (call.getOperand(4) == call.getOperand(5) ||
      call.getOperand(4) == call.getOperand(6) ||
      call.getOperand(5) == call.getOperand(6))
    return std::nullopt;
  Stream result{function, call, scope, allocation, view, {}, {}};
  for (Operation *user : view.getResult().getUsers()) {
    if (user == call)
      continue;
    auto source = dyn_cast<memref::SubViewOp>(user);
    if (!source || !source.getResult().hasOneUse() ||
        (!slice(source, {0, 0, 0}, {4, 128, 16}) &&
         !slice(source, {4, 0, 0}, {4, 128, 16})))
      return std::nullopt;
    auto expand =
        dyn_cast<memref::ExpandShapeOp>(*source.getResult().getUsers().begin());
    if (!expand || !expand.getResult().hasOneUse() ||
        expand.getReassociationIndices() !=
            SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}})
      return std::nullopt;
    auto flatSource = dyn_cast<memref::CollapseShapeOp>(
        *expand.getResult().getUsers().begin());
    if (!flatSource || !flatSource.getResult().hasOneUse() ||
        flatSource.getReassociationIndices() !=
            SmallVector<ReassociationIndices>{{0}, {1, 2, 3}})
      return std::nullopt;
    auto copy = dyn_cast<CopyOp>(*flatSource.getResult().getUsers().begin());
    if (!copy || copy.getSrc() != flatSource.getResult() ||
        copy->getParentOp() != scope ||
        !onlyAttrs(copy,
                   {"operandSegmentSizes", "hivm.inserted-copy", "tiled_op"}) ||
        !call->isBeforeInBlock(copy))
      return std::nullopt;
    auto flatDestination =
        copy.getDst().getDefiningOp<memref::CollapseShapeOp>();
    auto destination =
        flatDestination
            ? flatDestination.getSrc().getDefiningOp<memref::SubViewOp>()
            : memref::SubViewOp{};
    if (!flatDestination || !destination ||
        !destination.getResult().hasOneUse() ||
        !flatDestination.getResult().hasOneUse() ||
        flatDestination.getReassociationIndices() !=
            SmallVector<ReassociationIndices>{{0}, {1, 2, 3}})
      return std::nullopt;
    if (result.l1 && result.l1 != destination.getSource())
      return std::nullopt;
    result.l1 = destination.getSource();
    for (Operation *op :
         {source.getOperation(), expand.getOperation(),
          flatSource.getOperation(), flatDestination.getOperation(),
          destination.getOperation()})
      if (!onlyAttrs(op, {"operandSegmentSizes", "static_offsets",
                          "static_sizes", "static_strides", "reassociation",
                          "static_output_shape"}))
        return std::nullopt;
    result.publications.push_back(
        {copy, source, destination, expand, flatSource, flatDestination});
    for (Operation *op :
         {source.getOperation(), expand.getOperation(),
          flatSource.getOperation(), copy.getOperation(),
          flatDestination.getOperation(), destination.getOperation()})
      allowed.insert(op);
  }
  if (result.publications.size() != 2)
    return std::nullopt;
  llvm::sort(result.publications, [](Publication &a, Publication &b) {
    return *getConstantIntValue(a.source.getMixedOffsets()[0]) <
           *getConstantIntValue(b.source.getMixedOffsets()[0]);
  });
  auto target = dyn_cast<MemRefType>(result.l1.getType());
  SmallVector<int64_t> targetStrides;
  int64_t targetOffset;
  if (!target || target.getShape() != ArrayRef<int64_t>{4, 16, 16, 16} ||
      target.getElementType() != b.getF16Type() ||
      target.getMemorySpace() !=
          AddressSpaceAttr::get(b.getContext(), AddressSpace::L1) ||
      failed(getStridesAndOffset(target, targetStrides, targetOffset)) ||
      targetStrides != SmallVector<int64_t>{4096, 256, 16, 1})
    return std::nullopt;
  for (auto [plane, publication] : llvm::enumerate(result.publications)) {
    if (!slice(publication.source, {static_cast<int64_t>(plane) * 4, 0, 0},
               {4, 128, 16}) ||
        !slice(publication.destination,
               {0, static_cast<int64_t>(plane) * 8, 0, 0}, {4, 8, 16, 16}))
      return std::nullopt;
  }
  if (!result.publications[0].copy->isBeforeInBlock(
          result.publications[1].copy))
    return std::nullopt;
  auto l1View = result.l1.getDefiningOp<memref::SubViewOp>();
  if (!l1View || !l1View.getSource().getDefiningOp<memref::AllocOp>())
    return std::nullopt;
  for (Operation &op : scope->getRegion(0).front()) {
    if (allowed.contains(&op) ||
        op.getName().getStringRef() == "hivm.hir.anchor")
      continue;
    auto mark = dyn_cast<annotation::MarkOp>(op);
    if (!mark || mark->getNumOperands() != 1 ||
        (mark->getOperand(0) != call.getOperand(5) &&
         mark->getOperand(0) != l1View.getSource()))
      return std::nullopt;
    allowed.insert(mark);
  }
  for (unsigned index : {4u, 5u, 6u})
    for (Operation *user : call.getOperand(index).getUsers())
      if (!allowed.contains(user))
        return std::nullopt;
  return result;
}

void applyStream(Stream &stream, ModuleOp module) {
  OpBuilder b(module.getContext());
  b.setInsertionPoint(stream.function);
  auto first = buildProducer(
      b, stream.function.getLoc(),
      (stream.function.getSymName() + "_cvp_half0").str(), Producer::FirstHalf);
  auto second =
      buildProducer(b, stream.function.getLoc(),
                    (stream.function.getSymName() + "_cvp_half1").str(),
                    Producer::SecondHalf);
  b.setInsertionPoint(stream.allocation);
  auto allocation = b.create<memref::AllocOp>(
      stream.allocation.getLoc(), memory(b, {8, 65, 16, 1}, b.getF16Type()));
  if (Attribute alignment = stream.allocation->getAttr("alignment"))
    allocation->setAttr("alignment", alignment);
  SmallVector<OpFoldResult> zeros(4, b.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(8), b.getIndexAttr(64),
                                  b.getIndexAttr(16), b.getIndexAttr(1)};
  SmallVector<OpFoldResult> ones(4, b.getIndexAttr(1));
  Value view = b.create<memref::SubViewOp>(stream.view.getLoc(),
                                           probabilityType(b, true), allocation,
                                           zeros, sizes, ones);
  auto invoke = [&](func::FuncOp function, ValueRange operands) {
    auto call =
        b.create<func::CallOp>(stream.call.getLoc(), function, operands);
    call->setAttr("hivm.vector_function", b.getUnitAttr());
    call->setAttr("no_inline", b.getUnitAttr());
  };
  SmallVector<Value> operands(stream.call.getOperands());
  operands[1] = view;
  b.setInsertionPoint(stream.call);
  invoke(first, operands);
  b.setInsertionPoint(stream.publications[0].copy);
  for (int64_t phase = 0; phase < 2; ++phase) {
    if (phase)
      invoke(second, ValueRange{operands[0], view, operands[4], operands[6]});
    for (int64_t plane = 0; plane < 2; ++plane) {
      auto &old = stream.publications[plane];
      auto source = b.create<memref::SubViewOp>(
          old.source.getLoc(), view, ArrayRef<int64_t>{plane * 4, 0, 0},
          ArrayRef<int64_t>{4, 64, 16}, ArrayRef<int64_t>{1, 1, 1});
      auto expanded = b.create<memref::ExpandShapeOp>(
          old.expand.getLoc(), ArrayRef<int64_t>{4, 4, 16, 16}, source,
          ArrayRef<ReassociationIndices>{{0}, {1, 2}, {3}});
      auto destination = b.create<memref::SubViewOp>(
          old.destination.getLoc(), stream.l1,
          ArrayRef<int64_t>{0, plane * 8 + phase * 4, 0, 0},
          ArrayRef<int64_t>{4, 4, 16, 16}, ArrayRef<int64_t>{1, 1, 1, 1});
      auto src = b.create<memref::CollapseShapeOp>(
          old.flatSource.getLoc(), expanded,
          ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}});
      auto dst = b.create<memref::CollapseShapeOp>(
          old.flatDestination.getLoc(), destination,
          ArrayRef<ReassociationIndices>{{0}, {1, 2, 3}});
      IRMapping mapping;
      mapping.map(old.copy.getSrc(), src);
      mapping.map(old.copy.getDst(), dst);
      b.clone(*old.copy, mapping);
    }
  }
  stream.call.erase();
  for (Publication &p : stream.publications) {
    p.copy.erase();
    p.flatSource.erase();
    p.flatDestination.erase();
    p.expand.erase();
    p.source.erase();
    p.destination.erase();
  }
  stream.view.erase();
  stream.allocation.erase();
  stream.function.erase();
  module->setAttr(kCount, b.getI64IntegerAttr(1));
  if (auto count = module->getAttrOfType<IntegerAttr>(
          "hivm.cvp_attention.dense_two_chain_loops"))
    module->setAttr("hivm.cvp_attention.dense_two_chain_loops",
                    b.getI64IntegerAttr(count.getInt() + 1));
}

struct CVPAttentionPStreamPass
    : impl::CVPAttentionPStreamBase<CVPAttentionPStreamPass> {
  using impl::CVPAttentionPStreamBase<
      CVPAttentionPStreamPass>::CVPAttentionPStreamBase;
  void runOnOperation() override {
    ModuleOp original = getOperation();
    if (!allowSumReassociation) {
      original.emitError(
          "CVP half-P streaming requires explicit FP32 sum reassociation");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> staged = cast<ModuleOp>(original->clone());
    SmallVector<func::FuncOp> candidates;
    for (auto function : staged->getOps<func::FuncOp>())
      if (function->hasAttr(kDense) && !function->hasAttr(kPhase))
        candidates.push_back(function);
    if (candidates.empty() && !requireMatch)
      return;
    auto match = candidates.size() == 1
                     ? matchStream(*staged, candidates.front())
                     : std::optional<Stream>{};
    if (!match || (*staged)->hasAttr(kCount) ||
        staged->lookupSymbol(
            (match->function.getSymName() + "_cvp_half0").str()) ||
        staged->lookupSymbol(
            (match->function.getSymName() + "_cvp_half1").str())) {
      original.emitError("CVP half-P requires one closed full-tile dense "
                         "max/P/state producer and publication graph");
      return signalPassFailure();
    }
    applyStream(*match, *staged);
    if (failed(verify(*staged))) {
      original.emitError(
          "CVP half-P replacement failed verification before commit");
      return signalPassFailure();
    }
    original->setAttrs((*staged)->getAttrs());
    original.getBodyRegion().takeBody(staged->getBodyRegion());
  }
};
} // namespace
std::unique_ptr<Pass>
createCVPAttentionPStreamPass(const CVPAttentionPStreamOptions &options) {
  return std::make_unique<CVPAttentionPStreamPass>(options);
}
} // namespace hivm
} // namespace mlir

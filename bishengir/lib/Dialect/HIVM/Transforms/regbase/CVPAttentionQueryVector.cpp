//===- CVPAttentionQueryVector.cpp - Key-major attention VF fusion --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <cassert>
#include <optional>
#include <string>

namespace mlir {
#define GEN_PASS_DEF_CVPATTENTIONQUERYVECTOR
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {

constexpr StringLiteral kProbabilityFusions =
    "hivm.cvp_attention.p32_materializations_removed";
constexpr StringLiteral kScoreRematerializations =
    "hivm.cvp_attention.query_score_materializations_removed";
constexpr StringLiteral kQueryMaxScalings =
    "hivm.cvp_attention.query_max_scalings";
constexpr StringLiteral kStateFusions =
    "hivm.cvp_attention.sum_materializations_removed";
constexpr StringLiteral kMaximumFusions =
    "hivm.cvp_attention.tile_max_materializations_removed";
constexpr StringLiteral kReductionChainLoops =
    "hivm.cvp_attention.four_chain_loops";
constexpr StringLiteral kDenseTwoChainLoops =
    "hivm.cvp_attention.dense_two_chain_loops";
constexpr StringLiteral kDensePLayouts = "hivm.cvp_attention.dense_p_layouts";

bool shape(Type type, ArrayRef<int64_t> dimensions, Type element) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && !tensor.getEncoding() && tensor.hasStaticShape() &&
         tensor.getShape() == dimensions && tensor.getElementType() == element;
}

bool vectorShape(Type type, ArrayRef<int64_t> dimensions, Type element) {
  auto vector = dyn_cast<VectorType>(type);
  return vector && !vector.isScalable() && vector.getShape() == dimensions &&
         vector.getElementType() == element;
}

std::optional<int64_t> integer(Value value) {
  APInt number;
  if (!matchPattern(value, m_ConstantInt(&number)))
    return std::nullopt;
  return number.getSExtValue();
}

bool constantFold(OpFoldResult value, int64_t expected) {
  return getConstantIntValue(value) == expected;
}

bool zeroIndices(ValueRange indices) {
  return llvm::all_of(indices, [](Value value) { return integer(value) == 0; });
}

bool inBounds(Operation *op, unsigned rank) {
  auto bounds = op->getAttrOfType<ArrayAttr>("in_bounds");
  return bounds && bounds.size() == rank &&
         llvm::all_of(bounds, [](Attribute value) {
           auto flag = dyn_cast<BoolAttr>(value);
           return flag && flag.getValue();
         });
}

bool keyLoop(scf::ForOp loop, int64_t keys, unsigned carried) {
  return loop && integer(loop.getLowerBound()) == 0 &&
         integer(loop.getUpperBound()) == keys &&
         integer(loop.getStep()) == 1 && loop.getInitArgs().size() == carried;
}

bool keySlice(tensor::ExtractSliceOp slice, Value source, Value key) {
  if (!slice || slice.getSource() != source ||
      slice.getMixedOffsets().size() != 2)
    return false;
  auto offsets = slice.getMixedOffsets();
  auto sizes = slice.getMixedSizes();
  auto strides = slice.getMixedStrides();
  return dyn_cast<Value>(offsets[0]) == key && constantFold(offsets[1], 0) &&
         constantFold(sizes[0], 1) && constantFold(sizes[1], 64) &&
         constantFold(strides[0], 1) && constantFold(strides[1], 1);
}

bool sameSlice(tensor::InsertSliceOp insert, tensor::ExtractSliceOp extract,
               Value source) {
  return insert && insert.getSource() == source &&
         insert.getDest() == extract.getSource() &&
         insert.getMixedOffsets() == extract.getMixedOffsets() &&
         insert.getMixedSizes() == extract.getMixedSizes() &&
         insert.getMixedStrides() == extract.getMixedStrides();
}

bool plainTensorVF(func::FuncOp function, bool allowFillMarker = false) {
  if (!function || function.isExternal() || !function.getBody().hasOneBlock() ||
      !function->hasAttr(VectorFunctionAttr::name) ||
      function->hasAttr("hacc.entry") ||
      function->hasAttr("hivm.backup_function"))
    return false;
  // Inserting an argument updates the standard function interface. Decline
  // unknown index-based metadata rather than silently leaving it stale.
  for (NamedAttribute attr : function->getAttrs()) {
    // This known marker does not encode argument/result indices. The caller
    // still proves tensor-only semantics and any body pattern being rewritten.
    if (allowFillMarker && attr.getName().strref() == "hfusion.has_fill" &&
        isa<UnitAttr>(attr.getValue()))
      continue;
    if (!llvm::is_contained(
            ArrayRef<StringRef>{"sym_name", "sym_visibility", "function_type",
                                "arg_attrs", "res_attrs", "no_inline",
                                "hivm.vector_function", "hivm.func_core_type"},
            attr.getName().strref()))
      return false;
  }
  for (unsigned i = 0; i < function.getNumArguments(); ++i)
    if (auto attrs = function.getArgAttrDict(i); attrs && !attrs.empty())
      return false;
  return true;
}

template <typename Op> Op onlyOp(func::FuncOp function) {
  SmallVector<Op> found;
  function.walk([&](Op op) { found.push_back(op); });
  return found.size() == 1 ? found.front() : Op{};
}

bool onlyOperations(func::FuncOp function, ArrayRef<Operation *> operations) {
  llvm::SmallPtrSet<Operation *, 16> allowed(operations.begin(),
                                             operations.end());
  bool valid = true;
  function.walk([&](Operation *op) {
    if (op != function && !isa<arith::ConstantOp>(op) && !allowed.contains(op))
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

bool plainCallMetadata(func::CallOp call) {
  for (NamedAttribute attr : call->getAttrs())
    if (!llvm::is_contained(
            ArrayRef<StringRef>{"callee", "hivm.vector_function", "no_inline"},
            attr.getName().strref()))
      return false;
  return true;
}

struct ProbabilityProducer {
  func::FuncOp function;
  scf::ForOp loop;
  vector::TransferReadOp read, maximum;
  arith::SubFOp subtract;
  math::ExpOp exponential;
  unsigned inputIndex, maximumIndex;
};

std::optional<ProbabilityProducer>
matchProducer(func::FuncOp function, int64_t keys, std::string &reason) {
  reason = "producer function interface or metadata";
  if (!plainTensorVF(function) || function.getNumArguments() != 3 ||
      function.getFunctionType().getNumResults() != 1)
    return std::nullopt;
  Builder b(function.getContext());
  auto loop = onlyOp<scf::ForOp>(function);
  auto exp = onlyOp<math::ExpOp>(function);
  auto sub = onlyOp<arith::SubFOp>(function);
  auto store = onlyOp<vector::TransferWriteOp>(function);
  auto insert = onlyOp<tensor::InsertSliceOp>(function);
  auto ret =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  reason = "producer loop or return coverage";
  if (!keyLoop(loop, keys, 1) || !exp || !sub || !store || !insert || !ret ||
      ret.getNumOperands() != 1 || ret.getOperand(0) != loop.getResult(0) ||
      loop->getParentOp() != function)
    return std::nullopt;
  auto read = sub.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto maximum = sub.getRhs().getDefiningOp<vector::TransferReadOp>();
  auto source = read ? read.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                     : tensor::ExtractSliceOp{};
  auto destination = store.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto input =
      source ? dyn_cast<BlockArgument>(source.getSource()) : BlockArgument{};
  auto row =
      maximum ? dyn_cast<BlockArgument>(maximum.getSource()) : BlockArgument{};
  reason = "producer input, row broadcast, slice or transfer contract";
  if (!read || !maximum || !input || !row ||
      input.getOwner() != &function.getBody().front() ||
      row.getOwner() != &function.getBody().front() ||
      !shape(input.getType(), {keys, 64}, b.getF32Type()) ||
      !shape(row.getType(), {64}, b.getF32Type()) ||
      !shape(loop.getInitArgs()[0].getType(), {keys, 64}, b.getF32Type()) ||
      !keySlice(source, input, loop.getInductionVar()) ||
      !keySlice(destination, loop.getRegionIterArg(0),
                loop.getInductionVar()) ||
      !vectorShape(read.getType(), {1, 64}, b.getF32Type()) ||
      maximum.getType() != read.getType() || read.getMask() ||
      maximum.getMask() || store.getMask() || !zeroIndices(read.getIndices()) ||
      !zeroIndices(maximum.getIndices()) || !zeroIndices(store.getIndices()) ||
      !inBounds(read, 2) || !inBounds(maximum, 2) || !inBounds(store, 2) ||
      maximum->getParentOp() != function || read->getParentOp() != loop ||
      sub->getParentOp() != loop || exp->getParentOp() != loop ||
      exp.getOperand() != sub.getResult() ||
      store.getVector() != exp.getResult() || store->getNumResults() != 1 ||
      !sameSlice(insert, destination, store->getResult(0)))
    return std::nullopt;
  auto broadcast =
      AffineMap::get(1, 0, {b.getAffineConstantExpr(0), b.getAffineDimExpr(0)},
                     b.getContext());
  reason = "producer transfer permutation";
  if (maximum.getPermutationMap() != broadcast ||
      !read.getPermutationMap().isIdentity() ||
      !store.getPermutationMap().isIdentity())
    return std::nullopt;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  reason = "producer closed-use or operation set";
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != insert.getResult() ||
      !read.getResult().hasOneUse() || !maximum.getResult().hasOneUse() ||
      !sub.getResult().hasOneUse() || !exp.getResult().hasOneUse() ||
      !store->getResult(0).hasOneUse() || !insert.getResult().hasOneUse() ||
      !onlyOperations(function, {loop, source, destination, read, maximum, sub,
                                 exp, store, insert, yield, ret}))
    return std::nullopt;
  reason.clear();
  return ProbabilityProducer{
      function,          loop, read, maximum, sub, exp, input.getArgNumber(),
      row.getArgNumber()};
}

struct ProbabilityConsumer {
  func::FuncOp function;
  scf::ForOp loop;
  vector::TransferReadOp read;
  unsigned inputIndex;
};

// A row-only VF may lie between maximum production and the P expression.
// It is safe to keep that call in place while extending a tensor read across
// it only when its entire body has tensor/value semantics, without memory or
// synchronization effects.
bool tensorOnlyVF(func::FuncOp function, bool allowFillMarker = false) {
  if (!plainTensorVF(function, allowFillMarker) ||
      !llvm::all_of(function.getArgumentTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); }) ||
      !llvm::all_of(function.getResultTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); }))
    return false;
  bool valid = true;
  function.walk([&](Operation *op) {
    if (op == function || isa<scf::ForOp, scf::YieldOp, func::ReturnOp>(op))
      return;
    if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
      valid &= isa<RankedTensorType>(read.getSource().getType());
      return;
    }
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      valid &= isa<RankedTensorType>(write.getSource().getType());
      return;
    }
    StringRef dialect = op->getName().getDialectNamespace();
    if (op->getNumRegions() || !isMemoryEffectFree(op) ||
        !llvm::is_contained(
            ArrayRef<StringRef>{"arith", "math", "tensor", "vector"}, dialect))
      valid = false;
  });
  return valid;
}

struct ScoreProducer {
  func::FuncOp function;
  scf::ForOp scaleLoop, maxLoop;
  vector::TransferReadOp rawRead, maxRead;
  tensor::ExtractSliceOp maxSlice;
  vector::ShapeCastOp maxShape;
  arith::MulFOp multiply;
  arith::ConstantOp factor;
  vector::TransferWriteOp rowStore;
  unsigned inputIndex, destinationIndex;
};

std::optional<ScoreProducer> matchScoreProducer(func::FuncOp function,
                                                int64_t keys) {
  if (!plainTensorVF(function) || function.getNumArguments() != 3 ||
      function.getFunctionType().getNumResults() != 2)
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (auto attrs = function.getResultAttrDict(i); attrs && !attrs.empty())
      return std::nullopt;
  Builder b(function.getContext());
  auto ret =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  if (!ret || ret.getNumOperands() != 2)
    return std::nullopt;
  auto scaleLoop = ret.getOperand(0).getDefiningOp<scf::ForOp>();
  auto rowStore = ret.getOperand(1).getDefiningOp<vector::TransferWriteOp>();
  auto maxLoop = rowStore ? rowStore.getVector().getDefiningOp<scf::ForOp>()
                          : scf::ForOp{};
  auto multiply = onlyOp<arith::MulFOp>(function);
  auto maximum = onlyOp<arith::MaximumFOp>(function);
  auto insert = onlyOp<tensor::InsertSliceOp>(function);
  if (!keyLoop(scaleLoop, keys, 1) || !keyLoop(maxLoop, keys, 1) ||
      scaleLoop == maxLoop || scaleLoop->getParentOp() != function ||
      maxLoop->getParentOp() != function || !multiply || !maximum || !insert ||
      !rowStore ||
      !vectorShape(maxLoop.getInitArgs()[0].getType(), {64}, b.getF32Type()) ||
      !matchPattern(maxLoop.getInitArgs()[0], m_NegInfFloat()))
    return std::nullopt;
  auto rawRead = multiply.getLhs().getDefiningOp<vector::TransferReadOp>();
  Value factorValue = multiply.getRhs();
  if (!rawRead) {
    rawRead = multiply.getRhs().getDefiningOp<vector::TransferReadOp>();
    factorValue = multiply.getLhs();
  }
  auto factor = factorValue.getDefiningOp<arith::ConstantOp>();
  auto dense = factor ? dyn_cast<DenseElementsAttr>(factor.getValue())
                      : DenseElementsAttr{};
  if (!rawRead || !factor || factor->getParentOp() != function || !dense ||
      !dense.isSplat() ||
      !vectorShape(factor.getType(), {1, 64}, b.getF32Type()) ||
      !dense.getSplatValue<APFloat>().bitwiseIsEqual(APFloat(0.5f)))
    return std::nullopt;
  auto rawSlice = rawRead.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto input = rawSlice ? dyn_cast<BlockArgument>(rawSlice.getSource())
                        : BlockArgument{};
  auto destination = dyn_cast<BlockArgument>(scaleLoop.getInitArgs()[0]);
  auto scaleStore = insert.getSource().getDefiningOp<vector::TransferWriteOp>();
  auto outSlice =
      scaleStore
          ? scaleStore.getSource().getDefiningOp<tensor::ExtractSliceOp>()
          : tensor::ExtractSliceOp{};
  Value maxValue =
      maximum.getLhs() == maxLoop.getRegionIterArg(0)   ? maximum.getRhs()
      : maximum.getRhs() == maxLoop.getRegionIterArg(0) ? maximum.getLhs()
                                                        : Value{};
  auto maxShape = maxValue ? maxValue.getDefiningOp<vector::ShapeCastOp>()
                           : vector::ShapeCastOp{};
  auto maxRead =
      maxShape ? maxShape.getSource().getDefiningOp<vector::TransferReadOp>()
               : vector::TransferReadOp{};
  auto maxSlice =
      maxRead ? maxRead.getSource().getDefiningOp<tensor::ExtractSliceOp>()
              : tensor::ExtractSliceOp{};
  auto row = dyn_cast<BlockArgument>(rowStore.getSource());
  if (!input || !destination || !row || input == destination ||
      input.getOwner() != &function.getBody().front() ||
      destination.getOwner() != &function.getBody().front() ||
      row.getOwner() != &function.getBody().front() ||
      !shape(input.getType(), {keys, 64}, b.getF32Type()) ||
      !shape(destination.getType(), {keys, 64}, b.getF32Type()) ||
      !shape(row.getType(), {64}, b.getF32Type()) || !input.hasOneUse() ||
      !destination.hasOneUse() || !row.hasOneUse() ||
      !keySlice(rawSlice, input, scaleLoop.getInductionVar()) ||
      !keySlice(outSlice, scaleLoop.getRegionIterArg(0),
                scaleLoop.getInductionVar()) ||
      !scaleStore || scaleStore.getVector() != multiply.getResult() ||
      !sameSlice(insert, outSlice, scaleStore->getResult(0)) || !maxShape ||
      !maxRead ||
      !keySlice(maxSlice, scaleLoop.getResult(0), maxLoop.getInductionVar()) ||
      !vectorShape(rawRead.getType(), {1, 64}, b.getF32Type()) ||
      maxRead.getType() != rawRead.getType() ||
      !vectorShape(maxShape.getType(), {64}, b.getF32Type()))
    return std::nullopt;
  for (vector::TransferReadOp read : {rawRead, maxRead})
    if (read.getMask() || !zeroIndices(read.getIndices()) ||
        !inBounds(read, 2) || !read.getPermutationMap().isIdentity())
      return std::nullopt;
  if (scaleStore.getMask() || !zeroIndices(scaleStore.getIndices()) ||
      !inBounds(scaleStore, 2) ||
      !scaleStore.getPermutationMap().isIdentity() || rowStore.getMask() ||
      !zeroIndices(rowStore.getIndices()) || !inBounds(rowStore, 1) ||
      !rowStore.getPermutationMap().isIdentity() ||
      multiply->getParentOp() != scaleLoop ||
      rawRead->getParentOp() != scaleLoop ||
      maxRead->getParentOp() != maxLoop || maximum->getParentOp() != maxLoop ||
      !maxLoop.getRegionIterArg(0).hasOneUse() ||
      !rawRead.getResult().hasOneUse() || !multiply.getResult().hasOneUse() ||
      !maxRead.getResult().hasOneUse() || !maxShape.getResult().hasOneUse() ||
      !maximum.getResult().hasOneUse() || !maxLoop.getResult(0).hasOneUse())
    return std::nullopt;
  for (Operation *user : scaleLoop.getResult(0).getUsers())
    if (user != maxSlice && user != ret)
      return std::nullopt;
  auto scaleYield =
      dyn_cast<scf::YieldOp>(scaleLoop.getBody()->getTerminator());
  auto maxYield = dyn_cast<scf::YieldOp>(maxLoop.getBody()->getTerminator());
  if (!scaleYield || scaleYield.getNumOperands() != 1 ||
      scaleYield.getOperand(0) != insert.getResult() || !maxYield ||
      maxYield.getNumOperands() != 1 ||
      maxYield.getOperand(0) != maximum.getResult() ||
      !onlyOperations(function,
                      {scaleLoop, rawSlice, outSlice, rawRead, multiply,
                       scaleStore, insert, scaleYield, maxLoop, maxSlice,
                       maxRead, maxShape, maximum, maxYield, rowStore, ret}))
    return std::nullopt;
  return ScoreProducer{function,
                       scaleLoop,
                       maxLoop,
                       rawRead,
                       maxRead,
                       maxSlice,
                       maxShape,
                       multiply,
                       factor,
                       rowStore,
                       input.getArgNumber(),
                       destination.getArgNumber()};
}

struct ProbabilityExpression {
  func::FuncOp function;
  scf::ForOp loop;
  vector::TransferReadOp read;
  arith::SubFOp subtract;
  unsigned inputIndex;
};

std::optional<ProbabilityExpression>
matchProbabilityExpression(func::FuncOp function, int64_t keys) {
  if (!tensorOnlyVF(function))
    return std::nullopt;
  auto loop = onlyOp<scf::ForOp>(function);
  auto exp = onlyOp<math::ExpOp>(function);
  auto sub = onlyOp<arith::SubFOp>(function);
  if (!loop || !exp || !sub || loop->getParentOp() != function ||
      exp->getParentOp() != loop || sub->getParentOp() != loop ||
      exp.getOperand() != sub.getResult() ||
      integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != keys || integer(loop.getStep()) != 1)
    return std::nullopt;
  auto read = sub.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto row = sub.getRhs().getDefiningOp<vector::TransferReadOp>();
  auto slice = read ? read.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                    : tensor::ExtractSliceOp{};
  auto input =
      slice ? dyn_cast<BlockArgument>(slice.getSource()) : BlockArgument{};
  auto maximum =
      row ? dyn_cast<BlockArgument>(row.getSource()) : BlockArgument{};
  Builder b(function.getContext());
  if (!read || !row || !input || !maximum ||
      input.getOwner() != &function.getBody().front() || !input.hasOneUse() ||
      maximum.getOwner() != &function.getBody().front() ||
      !shape(input.getType(), {keys, 64}, b.getF32Type()) ||
      !keySlice(slice, input, loop.getInductionVar()) ||
      !slice.getResult().hasOneUse() || read->getParentOp() != loop ||
      !vectorShape(read.getType(), {1, 64}, b.getF32Type()) || read.getMask() ||
      !zeroIndices(read.getIndices()) || !inBounds(read, 2) ||
      !read.getPermutationMap().isIdentity() || !read.getResult().hasOneUse() ||
      !sub.getResult().hasOneUse() || row->getParentOp() != function ||
      row.getType() != read.getType() || row.getMask() ||
      !zeroIndices(row.getIndices()) || !inBounds(row, 2) ||
      !shape(row.getSource().getType(), {64}, b.getF32Type()))
    return std::nullopt;
  auto broadcast =
      AffineMap::get(1, 0, {b.getAffineConstantExpr(0), b.getAffineDimExpr(0)},
                     b.getContext());
  if (row.getPermutationMap() != broadcast)
    return std::nullopt;
  return ProbabilityExpression{function, loop, read, sub, input.getArgNumber()};
}

std::optional<ProbabilityConsumer>
matchConsumer(func::FuncOp function, int64_t keys, std::string &reason) {
  reason = "consumer function interface or metadata";
  if (!plainTensorVF(function) || function.getNumArguments() != 3 ||
      function.getFunctionType().getNumResults() != 2)
    return std::nullopt;
  Builder b(function.getContext());
  auto loop = onlyOp<scf::ForOp>(function);
  auto read = onlyOp<vector::TransferReadOp>(function);
  auto sum = onlyOp<arith::AddFOp>(function);
  auto cast = onlyOp<arith::TruncFOp>(function);
  auto insert = onlyOp<tensor::InsertSliceOp>(function);
  auto ret =
      dyn_cast<func::ReturnOp>(function.getBody().front().getTerminator());
  reason = "consumer loop or return coverage";
  if (!keyLoop(loop, keys, 2) || !read || !sum || !cast || !insert || !ret ||
      ret.getNumOperands() != 2 || loop->getParentOp() != function)
    return std::nullopt;
  auto source = read.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto input =
      source ? dyn_cast<BlockArgument>(source.getSource()) : BlockArgument{};
  reason = "consumer input, positive-zero initializer, slice or conversion";
  if (!input || input.getOwner() != &function.getBody().front() ||
      !shape(input.getType(), {keys, 64}, b.getF32Type()) ||
      !keySlice(source, input, loop.getInductionVar()) ||
      !shape(loop.getInitArgs()[0].getType(), {4, keys, 16}, b.getF16Type()) ||
      !vectorShape(loop.getInitArgs()[1].getType(), {64}, b.getF32Type()) ||
      !matchPattern(loop.getInitArgs()[1], m_PosZeroFloat()) ||
      !vectorShape(read.getType(), {1, 64}, b.getF32Type()) || read.getMask() ||
      !zeroIndices(read.getIndices()) || !inBounds(read, 2) ||
      !read.getPermutationMap().isIdentity() || read->getParentOp() != loop ||
      cast.getIn() != read.getResult() ||
      !vectorShape(cast.getType(), {1, 64}, b.getF16Type()))
    return std::nullopt;
  Value sumValue = sum.getLhs() == loop.getRegionIterArg(1)   ? sum.getRhs()
                   : sum.getRhs() == loop.getRegionIterArg(1) ? sum.getLhs()
                                                              : Value{};
  auto sumInput = sumValue ? sumValue.getDefiningOp<vector::ShapeCastOp>()
                           : vector::ShapeCastOp{};
  reason = "consumer FP32 sum dataflow";
  if (!sumInput || sumInput.getSource() != read.getResult() ||
      !vectorShape(sumInput.getType(), {64}, b.getF32Type()))
    return std::nullopt;
  auto pStore = insert.getSource().getDefiningOp<vector::TransferWriteOp>();
  auto pShape = pStore ? pStore.getVector().getDefiningOp<vector::ShapeCastOp>()
                       : vector::ShapeCastOp{};
  auto pView = pStore
                   ? pStore.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                   : tensor::ExtractSliceOp{};
  auto rowStore = ret.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
  reason = "consumer P or sum output mapping";
  if (!pStore || !pShape || !pView || !rowStore ||
      pShape.getSource() != cast.getResult() ||
      !vectorShape(pShape.getType(), {4, 1, 16}, b.getF16Type()) ||
      pView.getSource() != loop.getRegionIterArg(0) ||
      pView.getMixedOffsets().size() != 3 || pStore.getMask() ||
      !zeroIndices(pStore.getIndices()) || !inBounds(pStore, 3) ||
      !pStore.getPermutationMap().isIdentity() ||
      !sameSlice(insert, pView, pStore->getResult(0)) ||
      rowStore.getVector() != loop.getResult(1) || rowStore.getMask() ||
      !shape(rowStore.getSource().getType(), {64}, b.getF32Type()) ||
      !zeroIndices(rowStore.getIndices()) || !inBounds(rowStore, 1) ||
      !rowStore.getPermutationMap().isIdentity() ||
      ret.getOperand(1) != loop.getResult(0))
    return std::nullopt;
  auto offsets = pView.getMixedOffsets();
  auto sizes = pView.getMixedSizes();
  auto strides = pView.getMixedStrides();
  reason = "consumer normalized P coordinates";
  if (!constantFold(offsets[0], 0) ||
      dyn_cast<Value>(offsets[1]) != loop.getInductionVar() ||
      !constantFold(offsets[2], 0) || !constantFold(sizes[0], 4) ||
      !constantFold(sizes[1], 1) || !constantFold(sizes[2], 16) ||
      !llvm::all_of(
          strides, [](OpFoldResult stride) { return constantFold(stride, 1); }))
    return std::nullopt;
  auto yield = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  reason = "consumer closed operation set";
  if (!yield || yield.getNumOperands() != 2 ||
      yield.getOperand(0) != insert.getResult() ||
      yield.getOperand(1) != sum.getResult() ||
      !onlyOperations(function,
                      {loop, read, source, sumInput, sum, cast, pShape, pView,
                       pStore, insert, rowStore, yield, ret}))
    return std::nullopt;
  reason.clear();
  return ProbabilityConsumer{function, loop, read, input.getArgNumber()};
}

std::optional<int64_t> keyMajorKeys(func::FuncOp caller) {
  if (caller->hasAttr("hivm.backup_function") ||
      !caller->hasAttr("hivm.part_of_mix") ||
      caller->getAttr("hivm.func_core_type") !=
          TFuncCoreTypeAttr::get(caller.getContext(), TFuncCoreType::AIV))
    return std::nullopt;
  auto proofs = caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
  if (!proofs || proofs.size() != 1)
    return std::nullopt;
  auto plan = dyn_cast<DictionaryAttr>(proofs[0]);
  auto layout = plan ? plan.getAs<StringAttr>("score_layout") : StringAttr{};
  auto m = plan ? plan.getAs<IntegerAttr>("block_m") : IntegerAttr{};
  auto n = plan ? plan.getAs<IntegerAttr>("block_n") : IntegerAttr{};
  auto d = plan ? plan.getAs<IntegerAttr>("head_dim") : IntegerAttr{};
  if (!layout || layout.getValue() != "key-major" || !m || m.getInt() != 128 ||
      !n || (n.getInt() != 128 && n.getInt() != 256) || !d ||
      (d.getInt() != 64 && d.getInt() != 128))
    return std::nullopt;
  return n.getInt();
}

bool protectedPScope(func::CallOp producer, func::CallOp consumer,
                     bool allowTensorCalls = false,
                     bool allowIdentical = false) {
  auto scopeOp = dyn_cast<scope::ScopeOp>(producer->getParentOp());
  if (!scopeOp || consumer->getParentOp() != scopeOp ||
      (producer == consumer
           ? !allowIdentical
           : !producer->isBeforeInBlock(consumer.getOperation())))
    return false;
  auto loop = dyn_cast<scf::ForOp>(scopeOp->getParentOp());
  if (!loop)
    return false;
  SmallVector<scope::ScopeOp> scopes(loop.getBody()->getOps<scope::ScopeOp>());
  if (scopes.size() != 4 || scopes[1] != scopeOp)
    return false;
  for (unsigned i = 0; i < 4; ++i) {
    auto lag = scopes[i]->getAttrOfType<IntegerAttr>(PreloadNumAttr::name);
    auto core = TCoreTypeAttr::get(loop.getContext(),
                                   i % 2 ? TCoreType::VECTOR : TCoreType::CUBE);
    if (!lag || lag.getInt() != 3 - i ||
        scopes[i]->getAttr(kPipelinedLoopCoreTypeAttrName) != core)
      return false;
  }
  // All score/P ownership handoffs must remain outside this scheduled scope.
  bool hasHandoff = false;
  scopeOp.walk([&](Operation *op) {
    if (isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      hasHandoff = true;
  });
  if (hasHandoff)
    return false;
  if (producer == consumer)
    return true;
  for (Operation *op = producer->getNextNode(); op != consumer;
       op = op->getNextNode()) {
    if (allowTensorCalls)
      if (auto call = dyn_cast<func::CallOp>(op)) {
        auto callee =
            producer->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
                call.getCallee());
        if (plainCallMetadata(call) &&
            tensorOnlyVF(callee, /*allowFillMarker=*/true))
          continue;
      }
    if (!isa<AnchorOp, tensor::EmptyOp, tensor::ExpandShapeOp,
             tensor::CollapseShapeOp, arith::ConstantOp>(op))
      return false;
  }
  return true;
}

struct Fusion {
  ProbabilityProducer producer;
  ProbabilityConsumer consumer;
  func::CallOp first, second;
};

void applyProbabilityFusion(Fusion &fusion) {
  auto &p = fusion.producer;
  auto &c = fusion.consumer;
  OpBuilder builder(c.function.getContext());
  unsigned rowIndex = c.function.getNumArguments();
  c.function.insertArgument(
      rowIndex, p.function.getArgument(p.maximumIndex).getType(),
      DictionaryAttr::get(c.function.getContext()), c.function.getLoc());
  IRMapping mapping;
  mapping.map(p.function.getArgument(p.inputIndex),
              c.function.getArgument(c.inputIndex));
  mapping.map(p.function.getArgument(p.maximumIndex),
              c.function.getArgument(rowIndex));
  builder.setInsertionPoint(c.loop);
  for (arith::ConstantOp constant :
       p.function.getBody().front().getOps<arith::ConstantOp>())
    builder.clone(*constant, mapping);
  auto row = cast<vector::TransferReadOp>(builder.clone(*p.maximum, mapping));
  mapping.map(p.maximum.getResult(), row.getResult());
  mapping.map(p.read.getResult(), c.read.getResult());
  SmallVector<OpOperand *> originalUses;
  for (OpOperand &use : c.read.getResult().getUses())
    originalUses.push_back(&use);
  builder.setInsertionPointAfter(c.read);
  auto subtract = cast<arith::SubFOp>(builder.clone(*p.subtract, mapping));
  mapping.map(p.subtract.getResult(), subtract.getResult());
  auto exponential = cast<math::ExpOp>(builder.clone(*p.exponential, mapping));
  for (OpOperand *use : originalUses)
    use->set(exponential.getResult());
  fusion.second->setOperand(c.inputIndex,
                            fusion.first.getOperand(p.inputIndex));
  fusion.second.getOperandsMutable().append(
      ValueRange{fusion.first.getOperand(p.maximumIndex)});
  fusion.first.erase();
  p.function.erase();
}

struct ScoreRematerialization {
  ScoreProducer producer;
  ProbabilityExpression consumer;
  func::CallOp first, second;
};

void applyScoreRematerialization(ScoreRematerialization &rewrite,
                                 bool scaleAfterMax) {
  auto &p = rewrite.producer;
  auto &c = rewrite.consumer;
  OpBuilder builder(c.function.getContext());
  IRMapping mapping;
  builder.setInsertionPoint(c.loop);
  builder.clone(*p.factor, mapping);
  mapping.map(p.rawRead.getResult(), c.read.getResult());
  builder.setInsertionPointAfter(c.read);
  auto scaled = cast<arith::MulFOp>(builder.clone(*p.multiply, mapping));
  c.subtract->setOperand(0, scaled.getResult());

  // Both reads stay within the same P scope and before its score-release
  // handoff. The raw tensor remains live until its final SSA reader.
  p.maxSlice->setOperand(0, p.function.getArgument(p.inputIndex));
  IRMapping maxMapping;
  if (scaleAfterMax) {
    builder.setInsertionPoint(p.rowStore);
    auto type = cast<VectorType>(p.maxLoop.getResult(0).getType());
    auto factor = builder.create<arith::ConstantOp>(
        p.factor.getLoc(),
        cast<DenseElementsAttr>(p.factor.getValue()).reshape(type));
    maxMapping.map(p.factor.getResult(), factor.getResult());
    maxMapping.map(p.rawRead.getResult(), p.maxLoop.getResult(0));
    auto multiply = cast<arith::MulFOp>(builder.clone(*p.multiply, maxMapping));
    multiply.getResult().setType(type);
    p.rowStore->setOperand(0, multiply.getResult());
  } else {
    builder.setInsertionPointAfter(p.maxRead);
    maxMapping.map(p.rawRead.getResult(), p.maxRead.getResult());
    auto multiply = cast<arith::MulFOp>(builder.clone(*p.multiply, maxMapping));
    p.maxShape->setOperand(0, multiply.getResult());
  }

  Value raw = rewrite.first.getOperand(p.inputIndex);
  Value destination = rewrite.first.getOperand(p.destinationIndex);
  SmallVector<Value> operands(rewrite.first.getOperands());
  operands.erase(operands.begin() + p.destinationIndex);
  auto ret = cast<func::ReturnOp>(p.function.getBody().front().getTerminator());
  ret->eraseOperand(0);
  p.scaleLoop.erase();
  p.function.eraseResult(0);
  p.function.eraseArgument(p.destinationIndex);
  builder.setInsertionPoint(rewrite.first);
  auto replacement = builder.create<func::CallOp>(rewrite.first.getLoc(),
                                                  p.function, operands);
  replacement->setAttrs(rewrite.first->getAttrs());
  rewrite.first.getResult(0).replaceAllUsesWith(raw);
  rewrite.first.getResult(1).replaceAllUsesWith(replacement.getResult(0));
  rewrite.first.erase();
  if (auto empty = destination.getDefiningOp<tensor::EmptyOp>();
      empty && empty.getResult().use_empty())
    empty.erase();
}

void addCount(ModuleOp module, StringRef name, size_t count) {
  if (!count)
    return;
  auto old = module->getAttrOfType<IntegerAttr>(name);
  module->setAttr(name,
                  IntegerAttr::get(IntegerType::get(module.getContext(), 64),
                                   (old ? old.getInt() : 0) + count));
}

struct SumProducer {
  func::FuncOp function;
  scf::ForOp loop;
  vector::TransferWriteOp sumStore;
  unsigned destinationIndex;
};

std::optional<SumProducer> matchSumProducer(func::FuncOp function,
                                            int64_t keys) {
  if (!tensorOnlyVF(function) ||
      function.getFunctionType().getNumResults() != 2)
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (auto attrs = function.getResultAttrDict(i); attrs && !attrs.empty())
      return std::nullopt;
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto store = ret.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
  auto loop = ret.getOperand(1).getDefiningOp<scf::ForOp>();
  auto castOp = onlyOp<arith::TruncFOp>(function);
  Builder b(function.getContext());
  if (!store || !keyLoop(loop, keys, 2) || !castOp ||
      loop->getParentOp() != function || store->getParentOp() != function ||
      ret.getOperand(1) != loop.getResult(0) ||
      store.getVector() != loop.getResult(1) || store.getMask() ||
      !zeroIndices(store.getIndices()) || !inBounds(store, 1) ||
      !store.getPermutationMap().isIdentity() ||
      !store.getResult().hasOneUse() ||
      !shape(loop.getResult(0).getType(), {4, keys, 16}, b.getF16Type()) ||
      !vectorShape(loop.getResult(1).getType(), {64}, b.getF32Type()) ||
      !matchPattern(loop.getInitArgs()[1], m_PosZeroFloat()))
    return std::nullopt;
  auto destination = dyn_cast<BlockArgument>(store.getSource());
  if (!destination || destination.getOwner() != &function.getBody().front() ||
      !destination.hasOneUse() ||
      !shape(destination.getType(), {64}, b.getF32Type()))
    return std::nullopt;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto sum = yield.getOperand(1).getDefiningOp<arith::AddFOp>();
  if (!sum || sum->getParentOp() != loop)
    return std::nullopt;
  Value term = sum.getLhs() == loop.getRegionIterArg(1)   ? sum.getRhs()
               : sum.getRhs() == loop.getRegionIterArg(1) ? sum.getLhs()
                                                          : Value{};
  auto reshape =
      term ? term.getDefiningOp<vector::ShapeCastOp>() : vector::ShapeCastOp{};
  if (!reshape || reshape.getSource() != castOp.getIn() ||
      !vectorShape(castOp.getIn().getType(), {1, 64}, b.getF32Type()) ||
      !vectorShape(castOp.getType(), {1, 64}, b.getF16Type()))
    return std::nullopt;
  return SumProducer{function, loop, store, destination.getArgNumber()};
}

struct StateConsumer {
  func::FuncOp function;
  vector::TransferReadOp maximumRead, sumRead;
  unsigned maximumIndex, sumIndex;
};

std::optional<StateConsumer> matchStateConsumer(func::FuncOp function) {
  if (!tensorOnlyVF(function) || function.getNumArguments() != 6 ||
      function.getFunctionType().getNumResults() != 2)
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (auto attrs = function.getResultAttrDict(i); attrs && !attrs.empty())
      return std::nullopt;
  auto sub = onlyOp<arith::SubFOp>(function);
  auto exp = onlyOp<math::ExpOp>(function);
  auto mul = onlyOp<arith::MulFOp>(function);
  auto add = onlyOp<arith::AddFOp>(function);
  if (!sub || !exp || !mul || !add || exp.getOperand() != sub.getResult())
    return std::nullopt;
  auto oldM = sub.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto newM = sub.getRhs().getDefiningOp<vector::TransferReadOp>();
  Value oldLValue = mul.getLhs() == exp.getResult()   ? mul.getRhs()
                    : mul.getRhs() == exp.getResult() ? mul.getLhs()
                                                      : Value{};
  Value sumValue = add.getLhs() == mul.getResult()   ? add.getRhs()
                   : add.getRhs() == mul.getResult() ? add.getLhs()
                                                     : Value{};
  auto oldL = oldLValue ? oldLValue.getDefiningOp<vector::TransferReadOp>()
                        : vector::TransferReadOp{};
  auto sum = sumValue ? sumValue.getDefiningOp<vector::TransferReadOp>()
                      : vector::TransferReadOp{};
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto lStore = ret.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
  auto alphaStore = ret.getOperand(1).getDefiningOp<vector::TransferWriteOp>();
  if (!oldM || !newM || !oldL || !sum || !lStore || !alphaStore ||
      lStore.getVector() != add.getResult() ||
      alphaStore.getVector() != exp.getResult())
    return std::nullopt;
  Builder b(function.getContext());
  SmallVector<unsigned> indices;
  for (auto read : {oldM, newM, oldL, sum}) {
    auto arg = dyn_cast<BlockArgument>(read.getSource());
    if (!arg || arg.getOwner() != &function.getBody().front() ||
        !arg.hasOneUse() || !shape(arg.getType(), {64}, b.getF32Type()) ||
        !vectorShape(read.getType(), {64}, b.getF32Type()) || read.getMask() ||
        !zeroIndices(read.getIndices()) || !inBounds(read, 1) ||
        !read.getPermutationMap().isIdentity())
      return std::nullopt;
    indices.push_back(arg.getArgNumber());
  }
  for (auto store : {lStore, alphaStore}) {
    auto arg = dyn_cast<BlockArgument>(store.getSource());
    if (!arg || arg.getOwner() != &function.getBody().front() ||
        !arg.hasOneUse() || !shape(arg.getType(), {64}, b.getF32Type()) ||
        store.getMask() || !zeroIndices(store.getIndices()) ||
        !inBounds(store, 1) || !store.getPermutationMap().isIdentity())
      return std::nullopt;
    indices.push_back(arg.getArgNumber());
  }
  llvm::sort(indices);
  for (unsigned i = 0; i < 6; ++i)
    if (indices[i] != i)
      return std::nullopt;
  if (!onlyOperations(function, {oldM, newM, oldL, sum, sub, exp, mul, add,
                                 lStore, alphaStore, ret}))
    return std::nullopt;
  return StateConsumer{function, newM, sum,
                       cast<BlockArgument>(newM.getSource()).getArgNumber(),
                       cast<BlockArgument>(sum.getSource()).getArgNumber()};
}

struct StateFusion {
  SumProducer producer;
  StateConsumer consumer;
  func::CallOp first, second;
};

void applyStateFusion(StateFusion &fusion) {
  auto &p = fusion.producer;
  auto &c = fusion.consumer;
  OpBuilder builder(p.function.getContext());
  auto ret = cast<func::ReturnOp>(p.function.getBody().front().getTerminator());
  auto stateRet =
      cast<func::ReturnOp>(c.function.getBody().front().getTerminator());
  builder.setInsertionPoint(ret);
  IRMapping mapping;
  mapping.map(c.sumRead.getResult(), p.loop.getResult(1));
  // The P expression already broadcasts new_m across its query vector. Reuse
  // those bits when the caller proves that state reads the same tensor value.
  Value maximum;
  for (auto read :
       p.function.getBody().front().getOps<vector::TransferReadOp>()) {
    auto arg = dyn_cast<BlockArgument>(read.getSource());
    if (!arg || arg.getOwner() != &p.function.getBody().front() ||
        fusion.first.getOperand(arg.getArgNumber()) !=
            fusion.second.getOperand(c.maximumIndex) ||
        read.getMask() || !zeroIndices(read.getIndices()) ||
        !inBounds(read, 2) ||
        !vectorShape(read.getType(), {1, 64}, builder.getF32Type()))
      continue;
    auto broadcast = AffineMap::get(
        1, 0, {builder.getAffineConstantExpr(0), builder.getAffineDimExpr(0)},
        builder.getContext());
    if (read.getPermutationMap() == broadcast) {
      maximum = builder.create<vector::ShapeCastOp>(
          read.getLoc(), c.maximumRead.getType(), read.getResult());
      break;
    }
  }
  if (maximum)
    mapping.map(c.maximumRead.getResult(), maximum);
  SmallVector<Value> operands(fusion.first.getOperands());
  for (unsigned i = 0; i < c.function.getNumArguments(); ++i) {
    if (i == c.sumIndex || (maximum && i == c.maximumIndex))
      continue;
    unsigned index = p.function.getNumArguments();
    Value input = fusion.second.getOperand(i);
    p.function.insertArgument(index, input.getType(),
                              builder.getDictionaryAttr({}), input.getLoc());
    mapping.map(c.function.getArgument(i), p.function.getArgument(index));
    operands.push_back(input);
  }
  for (Operation &op : c.function.getBody().front().without_terminator()) {
    if (&op == c.sumRead.getOperation() ||
        (maximum && &op == c.maximumRead.getOperation()))
      continue;
    builder.clone(op, mapping);
  }
  Value probability = ret.getOperand(1);
  ret->setOperands({probability, mapping.lookup(stateRet.getOperand(0)),
                    mapping.lookup(stateRet.getOperand(1))});
  p.sumStore.erase();
  p.function.eraseResult(0);
  p.function.insertResult(1, stateRet.getOperand(0).getType(),
                          builder.getDictionaryAttr({}));
  p.function.insertResult(2, stateRet.getOperand(1).getType(),
                          builder.getDictionaryAttr({}));
  p.function.eraseArgument(p.destinationIndex);
  operands.erase(operands.begin() + p.destinationIndex);
  builder.setInsertionPoint(fusion.second);
  auto call = builder.create<func::CallOp>(fusion.second.getLoc(), p.function,
                                           operands);
  call->setAttrs(fusion.first->getAttrs());
  fusion.first.getResult(1).replaceAllUsesWith(call.getResult(0));
  fusion.second.getResult(0).replaceAllUsesWith(call.getResult(1));
  fusion.second.getResult(1).replaceAllUsesWith(call.getResult(2));
  fusion.second.erase();
  fusion.first.erase();
  c.function.erase();
}

struct MaximumProducer {
  func::FuncOp function;
  vector::TransferWriteOp store;
  unsigned inputIndex, destinationIndex;
};

std::optional<MaximumProducer> matchMaximumProducer(func::FuncOp function,
                                                    int64_t keys) {
  if (!tensorOnlyVF(function) || function.getNumArguments() != 2 ||
      function.getFunctionType().getNumResults() != 1)
    return std::nullopt;
  if (auto attrs = function.getResultAttrDict(0); attrs && !attrs.empty())
    return std::nullopt;
  auto loop = onlyOp<scf::ForOp>(function);
  auto read = onlyOp<vector::TransferReadOp>(function);
  auto store = onlyOp<vector::TransferWriteOp>(function);
  auto multiply = onlyOp<arith::MulFOp>(function);
  auto maximum = onlyOp<arith::MaximumFOp>(function);
  auto reshape = onlyOp<vector::ShapeCastOp>(function);
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  Builder b(function.getContext());
  if (!keyLoop(loop, keys, 1) || !read || !store || !multiply || !maximum ||
      !reshape || loop->getParentOp() != function ||
      read->getParentOp() != loop || maximum->getParentOp() != loop ||
      store->getParentOp() != function ||
      ret.getOperand(0) != store.getResult() ||
      !vectorShape(loop.getResult(0).getType(), {64}, b.getF32Type()) ||
      !matchPattern(loop.getInitArgs()[0], m_NegInfFloat()))
    return std::nullopt;
  auto slice = read.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto input =
      slice ? dyn_cast<BlockArgument>(slice.getSource()) : BlockArgument{};
  auto destination = dyn_cast<BlockArgument>(store.getSource());
  if (!input || !destination ||
      input.getOwner() != &function.getBody().front() ||
      destination.getOwner() != &function.getBody().front() ||
      !input.hasOneUse() || !destination.hasOneUse() ||
      !shape(input.getType(), {keys, 64}, b.getF32Type()) ||
      !shape(destination.getType(), {64}, b.getF32Type()) ||
      !keySlice(slice, input, loop.getInductionVar()) ||
      !vectorShape(read.getType(), {1, 64}, b.getF32Type()) || read.getMask() ||
      !zeroIndices(read.getIndices()) || !inBounds(read, 2) ||
      !read.getPermutationMap().isIdentity() || store.getMask() ||
      !zeroIndices(store.getIndices()) || !inBounds(store, 1) ||
      !store.getPermutationMap().isIdentity())
    return std::nullopt;
  Value maximumInput =
      maximum.getLhs() == loop.getRegionIterArg(0)   ? maximum.getRhs()
      : maximum.getRhs() == loop.getRegionIterArg(0) ? maximum.getLhs()
                                                     : Value{};
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (maximumInput != reshape.getResult() ||
      yield.getOperand(0) != maximum.getResult())
    return std::nullopt;
  Value unscaled;
  if (multiply->getParentOp() == loop &&
      reshape.getSource() == multiply.getResult() &&
      store.getVector() == loop.getResult(0))
    unscaled = read.getResult();
  else if (multiply->getParentOp() == function &&
           reshape.getSource() == read.getResult() &&
           store.getVector() == multiply.getResult())
    unscaled = loop.getResult(0);
  else
    return std::nullopt;
  Value factor = multiply.getLhs() == unscaled   ? multiply.getRhs()
                 : multiply.getRhs() == unscaled ? multiply.getLhs()
                                                 : Value{};
  DenseElementsAttr constant;
  if (!factor || !matchPattern(factor, m_Constant(&constant)) ||
      !constant.isSplat() || !constant.getElementType().isF32() ||
      !constant.getSplatValue<APFloat>().bitwiseIsEqual(APFloat(0.5f)) ||
      !onlyOperations(function, {loop, slice, read, reshape, multiply, maximum,
                                 yield, store, ret}))
    return std::nullopt;
  return MaximumProducer{function, store, input.getArgNumber(),
                         destination.getArgNumber()};
}

struct RowMaximum {
  func::FuncOp function;
  vector::TransferReadOp oldRead, tileRead;
  vector::TransferWriteOp fillStore, maximumStore;
  arith::MaximumFOp maximum;
  unsigned oldIndex, tileIndex, destinationIndex;
};

std::optional<RowMaximum> matchRowMaximum(func::FuncOp function) {
  if (!tensorOnlyVF(function, /*allowFillMarker=*/true) ||
      function.getNumArguments() != 4 ||
      function.getFunctionType().getNumResults() != 2)
    return std::nullopt;
  for (unsigned i = 0; i < 2; ++i)
    if (auto attrs = function.getResultAttrDict(i); attrs && !attrs.empty())
      return std::nullopt;
  auto maximum = onlyOp<arith::MaximumFOp>(function);
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  auto fill = ret.getOperand(0).getDefiningOp<vector::TransferWriteOp>();
  auto store = ret.getOperand(1).getDefiningOp<vector::TransferWriteOp>();
  if (!maximum || !fill || !store ||
      !matchPattern(fill.getVector(), m_PosZeroFloat()) ||
      store.getVector() != maximum.getResult())
    return std::nullopt;
  auto old = maximum.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto tile = maximum.getRhs().getDefiningOp<vector::TransferReadOp>();
  if (!old || !tile)
    return std::nullopt;
  Builder b(function.getContext());
  SmallVector<unsigned> indices;
  for (auto read : {old, tile}) {
    auto arg = dyn_cast<BlockArgument>(read.getSource());
    if (!arg || arg.getOwner() != &function.getBody().front() ||
        !arg.hasOneUse() || !shape(arg.getType(), {64}, b.getF32Type()) ||
        !vectorShape(read.getType(), {64}, b.getF32Type()) || read.getMask() ||
        !zeroIndices(read.getIndices()) || !inBounds(read, 1) ||
        !read.getPermutationMap().isIdentity())
      return std::nullopt;
    indices.push_back(arg.getArgNumber());
  }
  for (auto write : {fill, store}) {
    auto arg = dyn_cast<BlockArgument>(write.getSource());
    if (!arg || arg.getOwner() != &function.getBody().front() ||
        !arg.hasOneUse() || !shape(arg.getType(), {64}, b.getF32Type()) ||
        write.getMask() || !zeroIndices(write.getIndices()) ||
        !inBounds(write, 1) || !write.getPermutationMap().isIdentity())
      return std::nullopt;
    indices.push_back(arg.getArgNumber());
  }
  llvm::sort(indices);
  for (unsigned i = 0; i < 4; ++i)
    if (indices[i] != i)
      return std::nullopt;
  if (!onlyOperations(function, {old, tile, maximum, fill, store, ret}))
    return std::nullopt;
  return RowMaximum{function,
                    old,
                    tile,
                    fill,
                    store,
                    maximum,
                    cast<BlockArgument>(old.getSource()).getArgNumber(),
                    cast<BlockArgument>(tile.getSource()).getArgNumber(),
                    cast<BlockArgument>(store.getSource()).getArgNumber()};
}

struct FusedProbabilityState {
  func::FuncOp function;
  vector::TransferReadOp maximumRead, oldRead;
  unsigned rawIndex, maximumIndex, oldIndex;
};

std::optional<FusedProbabilityState>
matchFusedProbabilityState(func::FuncOp function, int64_t keys) {
  if (!tensorOnlyVF(function) ||
      function.getFunctionType().getNumResults() != 3)
    return std::nullopt;
  for (unsigned i = 0; i < 3; ++i)
    if (auto attrs = function.getResultAttrDict(i); attrs && !attrs.empty())
      return std::nullopt;
  auto loop = onlyOp<scf::ForOp>(function);
  if (!keyLoop(loop, keys, 2) || loop->getParentOp() != function)
    return std::nullopt;
  SmallVector<math::ExpOp> probabilities(loop.getBody()->getOps<math::ExpOp>());
  SmallVector<math::ExpOp> alphas(
      function.getBody().front().getOps<math::ExpOp>());
  if (probabilities.size() != 1 || alphas.size() != 1)
    return std::nullopt;
  auto sub = probabilities[0].getOperand().getDefiningOp<arith::SubFOp>();
  auto stateSub = alphas[0].getOperand().getDefiningOp<arith::SubFOp>();
  if (!sub || !stateSub)
    return std::nullopt;
  auto maximum = sub.getRhs().getDefiningOp<vector::TransferReadOp>();
  auto old = stateSub.getLhs().getDefiningOp<vector::TransferReadOp>();
  auto rowShape = stateSub.getRhs().getDefiningOp<vector::ShapeCastOp>();
  auto scale = sub.getLhs().getDefiningOp<arith::MulFOp>();
  if (!maximum || !old || !rowShape ||
      rowShape.getSource() != maximum.getResult() || !scale)
    return std::nullopt;
  auto raw = scale.getLhs().getDefiningOp<vector::TransferReadOp>();
  if (!raw)
    raw = scale.getRhs().getDefiningOp<vector::TransferReadOp>();
  auto slice = raw ? raw.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                   : tensor::ExtractSliceOp{};
  auto rawArg =
      slice ? dyn_cast<BlockArgument>(slice.getSource()) : BlockArgument{};
  auto maximumArg = dyn_cast<BlockArgument>(maximum.getSource());
  auto oldArg = dyn_cast<BlockArgument>(old.getSource());
  Builder b(function.getContext());
  if (!rawArg || !maximumArg || !oldArg ||
      rawArg.getOwner() != &function.getBody().front() ||
      maximumArg.getOwner() != &function.getBody().front() ||
      oldArg.getOwner() != &function.getBody().front() || !rawArg.hasOneUse() ||
      !maximumArg.hasOneUse() || !oldArg.hasOneUse() ||
      !shape(rawArg.getType(), {keys, 64}, b.getF32Type()) ||
      !shape(maximumArg.getType(), {64}, b.getF32Type()) ||
      !shape(oldArg.getType(), {64}, b.getF32Type()) ||
      !keySlice(slice, rawArg, loop.getInductionVar()) ||
      maximum->getParentOp() != function || old->getParentOp() != function ||
      !vectorShape(maximum.getType(), {1, 64}, b.getF32Type()) ||
      !vectorShape(old.getType(), {64}, b.getF32Type()) || maximum.getMask() ||
      old.getMask() || !zeroIndices(maximum.getIndices()) ||
      !zeroIndices(old.getIndices()) || !inBounds(maximum, 2) ||
      !inBounds(old, 1) || !old.getPermutationMap().isIdentity())
    return std::nullopt;
  auto broadcast =
      AffineMap::get(1, 0, {b.getAffineConstantExpr(0), b.getAffineDimExpr(0)},
                     b.getContext());
  if (maximum.getPermutationMap() != broadcast)
    return std::nullopt;
  return FusedProbabilityState{function,
                               maximum,
                               old,
                               rawArg.getArgNumber(),
                               maximumArg.getArgNumber(),
                               oldArg.getArgNumber()};
}

struct MaximumFusion {
  MaximumProducer producer;
  RowMaximum row;
  FusedProbabilityState consumer;
  func::CallOp first, middle, last;
};

void applyMaximumFusion(MaximumFusion &fusion) {
  auto &p = fusion.producer;
  auto &r = fusion.row;
  auto &c = fusion.consumer;
  OpBuilder builder(c.function.getContext());
  builder.setInsertionPointToStart(&c.function.getBody().front());
  IRMapping maximumMap;
  maximumMap.map(p.function.getArgument(p.inputIndex),
                 c.function.getArgument(c.rawIndex));
  for (Operation &op : p.function.getBody().front().without_terminator())
    if (&op != p.store.getOperation())
      builder.clone(op, maximumMap);
  Value tileMaximum = maximumMap.lookup(p.store.getVector());
  IRMapping rowMap;
  rowMap.map(r.tileRead.getResult(), tileMaximum);
  rowMap.map(r.function.getArgument(r.oldIndex),
             c.function.getArgument(c.oldIndex));
  unsigned destination = c.function.getNumArguments();
  Value destinationOperand = fusion.middle.getOperand(r.destinationIndex);
  c.function.insertArgument(destination, destinationOperand.getType(),
                            builder.getDictionaryAttr({}),
                            destinationOperand.getLoc());
  rowMap.map(r.function.getArgument(r.destinationIndex),
             c.function.getArgument(destination));
  for (Operation &op : r.function.getBody().front().without_terminator())
    if (&op != r.tileRead.getOperation() && &op != r.fillStore.getOperation())
      builder.clone(op, rowMap);
  Value newMaximum = rowMap.lookup(r.maximum.getResult());
  auto broadcast = builder.create<vector::ShapeCastOp>(
      c.maximumRead.getLoc(), c.maximumRead.getType(), newMaximum);
  c.maximumRead.getResult().replaceAllUsesWith(broadcast.getResult());
  c.oldRead.getResult().replaceAllUsesWith(
      rowMap.lookup(r.oldRead.getResult()));
  c.maximumRead.erase();
  c.oldRead.erase();
  auto ret = cast<func::ReturnOp>(c.function.getBody().front().getTerminator());
  ret.getOperandsMutable().append(
      ValueRange{rowMap.lookup(r.maximumStore.getResult())});
  c.function.insertResult(3, r.maximumStore.getResult().getType(),
                          builder.getDictionaryAttr({}));
  c.function.eraseArgument(c.maximumIndex);
  SmallVector<Value> operands(fusion.last.getOperands());
  operands.push_back(destinationOperand);
  operands.erase(operands.begin() + c.maximumIndex);
  builder.setInsertionPoint(fusion.last);
  auto call =
      builder.create<func::CallOp>(fusion.last.getLoc(), c.function, operands);
  call->setAttrs(fusion.last->getAttrs());
  for (unsigned i = 0; i < 3; ++i)
    fusion.last.getResult(i).replaceAllUsesWith(call.getResult(i));
  fusion.middle.getResult(1).replaceAllUsesWith(call.getResult(3));
  fusion.last.erase();
  fusion.middle.erase();
  fusion.first.erase();
  r.function.erase();
  p.function.erase();
}

struct KeyReduction {
  scf::ForOp loop;
  Operation *reduction;
  bool maximum;
};

std::optional<KeyReduction> matchKeyReduction(scf::ForOp loop, int64_t keys) {
  if (!isa<IndexType>(loop.getInductionVar().getType()) ||
      integer(loop.getLowerBound()) != 0 ||
      integer(loop.getUpperBound()) != keys || integer(loop.getStep()) != 1 ||
      keys % 4 != 0 || !isa<func::FuncOp>(loop->getParentOp()))
    return std::nullopt;
  Builder b(loop.getContext());
  for (Operation &op : loop.getBody()->without_terminator())
    if (op.getNumRegions())
      return std::nullopt;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (loop.getInitArgs().size() == 1 &&
      vectorShape(loop.getInitArgs()[0].getType(), {64}, b.getF32Type()) &&
      matchPattern(loop.getInitArgs()[0], m_NegInfFloat())) {
    auto maximum = yield.getOperand(0).getDefiningOp<arith::MaximumFOp>();
    Value carry = loop.getRegionIterArg(0);
    if (!maximum || !carry.hasOneUse() || maximum->getParentOp() != loop)
      return std::nullopt;
    Value term = maximum.getLhs() == carry   ? maximum.getRhs()
                 : maximum.getRhs() == carry ? maximum.getLhs()
                                             : Value{};
    auto reshape = term ? term.getDefiningOp<vector::ShapeCastOp>()
                        : vector::ShapeCastOp{};
    if (!reshape ||
        !vectorShape(reshape.getSource().getType(), {1, 64}, b.getF32Type()))
      return std::nullopt;
    auto read = reshape.getSource().getDefiningOp<vector::TransferReadOp>();
    auto scale = reshape.getSource().getDefiningOp<arith::MulFOp>();
    if (!read && scale) {
      read = scale.getLhs().getDefiningOp<vector::TransferReadOp>();
      if (!read)
        read = scale.getRhs().getDefiningOp<vector::TransferReadOp>();
    }
    auto slice = read ? read.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                      : tensor::ExtractSliceOp{};
    if (!read || !slice || read->getParentOp() != loop || read.getMask() ||
        !shape(slice.getSource().getType(), {keys, 64}, b.getF32Type()) ||
        !keySlice(slice, slice.getSource(), loop.getInductionVar()) ||
        !zeroIndices(read.getIndices()) || !inBounds(read, 2) ||
        !read.getPermutationMap().isIdentity())
      return std::nullopt;
    // A maximum body has no output tensor or other carried state to reorder.
    for (Operation &op : loop.getBody()->without_terminator())
      if (!isa<arith::ConstantOp>(&op) && &op != slice.getOperation() &&
          &op != read.getOperation() && &op != reshape.getOperation() &&
          &op != maximum.getOperation() &&
          (!scale || &op != scale.getOperation()))
        return std::nullopt;
    return KeyReduction{loop, maximum, true};
  }
  if (loop.getInitArgs().size() != 2 ||
      !shape(loop.getInitArgs()[0].getType(), {4, keys, 16}, b.getF16Type()) ||
      !vectorShape(loop.getInitArgs()[1].getType(), {64}, b.getF32Type()) ||
      !matchPattern(loop.getInitArgs()[1], m_PosZeroFloat()))
    return std::nullopt;
  unsigned reads = 0;
  for (auto read : loop.getBody()->getOps<vector::TransferReadOp>()) {
    auto slice = read.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!slice ||
        !shape(slice.getSource().getType(), {keys, 64}, b.getF32Type()) ||
        !keySlice(slice, slice.getSource(), loop.getInductionVar()) ||
        !vectorShape(read.getType(), {1, 64}, b.getF32Type()) ||
        read.getMask() || !zeroIndices(read.getIndices()) ||
        !inBounds(read, 2) || !read.getPermutationMap().isIdentity())
      return std::nullopt;
    ++reads;
  }
  if (reads != 1)
    return std::nullopt;
  auto sum = yield.getOperand(1).getDefiningOp<arith::AddFOp>();
  Value carry = loop.getRegionIterArg(1);
  if (!sum || !carry.hasOneUse() || sum->getParentOp() != loop)
    return std::nullopt;
  Value term = sum.getLhs() == carry   ? sum.getRhs()
               : sum.getRhs() == carry ? sum.getLhs()
                                       : Value{};
  auto sumShape =
      term ? term.getDefiningOp<vector::ShapeCastOp>() : vector::ShapeCastOp{};
  auto insert = yield.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  auto store = insert
                   ? insert.getSource().getDefiningOp<vector::TransferWriteOp>()
                   : vector::TransferWriteOp{};
  auto pShape = store ? store.getVector().getDefiningOp<vector::ShapeCastOp>()
                      : vector::ShapeCastOp{};
  auto truncate = pShape ? pShape.getSource().getDefiningOp<arith::TruncFOp>()
                         : arith::TruncFOp{};
  auto view = store ? store.getSource().getDefiningOp<tensor::ExtractSliceOp>()
                    : tensor::ExtractSliceOp{};
  if (!sumShape || !insert || !store || !pShape || !truncate || !view ||
      sumShape.getSource() != truncate.getIn() ||
      !vectorShape(truncate.getIn().getType(), {1, 64}, b.getF32Type()) ||
      !vectorShape(truncate.getType(), {1, 64}, b.getF16Type()) ||
      !vectorShape(pShape.getType(), {4, 1, 16}, b.getF16Type()) ||
      view.getSource() != loop.getRegionIterArg(0) ||
      view.getMixedOffsets().size() != 3 || store.getMask() ||
      !zeroIndices(store.getIndices()) || !inBounds(store, 3) ||
      !store.getPermutationMap().isIdentity() ||
      !sameSlice(insert, view, store.getResult()) ||
      !view.getResult().hasOneUse() || !store.getResult().hasOneUse() ||
      !insert.getResult().hasOneUse())
    return std::nullopt;
  auto offsets = view.getMixedOffsets();
  auto sizes = view.getMixedSizes();
  if (!constantFold(offsets[0], 0) ||
      dyn_cast<Value>(offsets[1]) != loop.getInductionVar() ||
      !constantFold(offsets[2], 0) || !constantFold(sizes[0], 4) ||
      !constantFold(sizes[1], 1) || !constantFold(sizes[2], 16) ||
      !llvm::all_of(view.getMixedStrides(),
                    [](OpFoldResult s) { return constantFold(s, 1); }))
    return std::nullopt;
  // P must be a destination-only carry. Otherwise separating the numerical
  // carries could change a hidden dependence on the preceding key's P values.
  if (!llvm::hasNItems(loop.getRegionIterArg(0).getUses(), 2))
    return std::nullopt;
  for (Operation *user : loop.getRegionIterArg(0).getUsers())
    if (user != view && user != insert)
      return std::nullopt;
  return KeyReduction{loop, sum, false};
}

void applyFourChains(KeyReduction &rewrite) {
  auto old = rewrite.loop;
  OpBuilder builder(old.getContext());
  builder.setInsertionPoint(old);
  Value step = builder.create<arith::ConstantIndexOp>(old.getLoc(), 4);
  unsigned carryIndex = rewrite.maximum ? 0 : 1;
  SmallVector<Value> initial;
  if (!rewrite.maximum)
    initial.push_back(old.getInitArgs()[0]);
  initial.append(4, old.getInitArgs()[carryIndex]);
  auto replacement = builder.create<scf::ForOp>(
      old.getLoc(), old.getLowerBound(), old.getUpperBound(), step, initial,
      [&](OpBuilder &body, Location loc, Value induction, ValueRange carried) {
        Value probability = rewrite.maximum ? Value{} : carried[0];
        SmallVector<Value> sums;
        auto oldYield = cast<scf::YieldOp>(old.getBody()->getTerminator());
        for (unsigned chain = 0; chain < 4; ++chain) {
          Value key = induction;
          if (chain) {
            Value offset = body.create<arith::ConstantIndexOp>(loc, chain);
            key = body.create<arith::AddIOp>(loc, induction, offset);
          }
          IRMapping mapping;
          mapping.map(old.getInductionVar(), key);
          if (!rewrite.maximum)
            mapping.map(old.getRegionIterArg(0), probability);
          mapping.map(old.getRegionIterArg(carryIndex),
                      carried[chain + carryIndex]);
          for (Operation &op : old.getBody()->without_terminator())
            body.clone(op, mapping);
          if (!rewrite.maximum)
            probability = mapping.lookup(oldYield.getOperand(0));
          sums.push_back(mapping.lookup(oldYield.getOperand(carryIndex)));
        }
        SmallVector<Value> results;
        if (!rewrite.maximum)
          results.push_back(probability);
        llvm::append_range(results, sums);
        body.create<scf::YieldOp>(loc, results);
      });
  replacement->setAttrs(old->getAttrs());
  replacement->setAttr("hivm.cvp_attention.reduction_chains",
                       builder.getI64IntegerAttr(4));
  builder.setInsertionPointAfter(replacement);
  auto merge = [&](Value lhs, Value rhs) {
    IRMapping mapping;
    mapping.map(rewrite.reduction->getOperand(0), lhs);
    mapping.map(rewrite.reduction->getOperand(1), rhs);
    return builder.clone(*rewrite.reduction, mapping)->getResult(0);
  };
  Value first = merge(replacement.getResult(carryIndex),
                      replacement.getResult(carryIndex + 1));
  Value second = merge(replacement.getResult(carryIndex + 2),
                       replacement.getResult(carryIndex + 3));
  Value reduced = merge(first, second);
  if (!rewrite.maximum)
    old.getResult(0).replaceAllUsesWith(replacement.getResult(0));
  old.getResult(carryIndex).replaceAllUsesWith(reduced);
  old.erase();
}

// Called only for a matched full-key maximum or the packed P loop just built
// from a closed DenseProbability. The P tensor remains serially threaded
// through every cloned write; only the explicitly permitted numerical carry is
// split.
void applyTwoChains(KeyReduction &rewrite, unsigned keyGroups) {
  auto old = rewrite.loop;
  auto upper = integer(old.getUpperBound());
  assert(upper && *upper > 0 && *upper % keyGroups == 0 &&
         integer(old.getLowerBound()) == 0 && integer(old.getStep()) == 1 &&
         ((rewrite.maximum && keyGroups == 4) ||
          (!rewrite.maximum && keyGroups == 2)) &&
         "expected a closed quarter-key dense-P reduction");
  int64_t quarter = *upper / keyGroups;
  unsigned carryIndex = rewrite.maximum ? 0 : 1;
  OpBuilder builder(old.getContext());
  builder.setInsertionPoint(old);
  Value limit = builder.create<arith::ConstantIndexOp>(old.getLoc(), quarter);
  SmallVector<Value> initial;
  if (!rewrite.maximum)
    initial.push_back(old.getInitArgs()[0]);
  initial.append(2, old.getInitArgs()[carryIndex]);
  auto replacement = builder.create<scf::ForOp>(
      old.getLoc(), old.getLowerBound(), limit, old.getStep(), initial,
      [&](OpBuilder &body, Location loc, Value induction, ValueRange carried) {
        Value probability = rewrite.maximum ? Value{} : carried[0];
        SmallVector<Value> values{carried[carryIndex], carried[carryIndex + 1]};
        auto oldYield = cast<scf::YieldOp>(old.getBody()->getTerminator());
        for (unsigned group = 0; group < keyGroups; ++group) {
          Value key = induction;
          if (group) {
            Value offset = body.create<arith::ConstantIndexOp>(
                loc, quarter * static_cast<int64_t>(group));
            key = body.create<arith::AddIOp>(loc, induction, offset);
          }
          unsigned chain = group % 2;
          IRMapping mapping;
          mapping.map(old.getInductionVar(), key);
          if (!rewrite.maximum)
            mapping.map(old.getRegionIterArg(0), probability);
          mapping.map(old.getRegionIterArg(carryIndex), values[chain]);
          for (Operation &op : old.getBody()->without_terminator())
            body.clone(op, mapping);
          if (!rewrite.maximum)
            probability = mapping.lookup(oldYield.getOperand(0));
          values[chain] = mapping.lookup(oldYield.getOperand(carryIndex));
        }
        SmallVector<Value> results;
        if (!rewrite.maximum)
          results.push_back(probability);
        llvm::append_range(results, values);
        body.create<scf::YieldOp>(loc, results);
      });
  replacement->setAttrs(old->getAttrs());
  replacement->setAttr("hivm.cvp_attention.reduction_chains",
                       builder.getI64IntegerAttr(2));
  replacement->setAttr("hivm.cvp_attention.reduction_key_groups",
                       builder.getI64IntegerAttr(keyGroups));
  builder.setInsertionPointAfter(replacement);
  IRMapping mapping;
  mapping.map(rewrite.reduction->getOperand(0),
              replacement.getResult(carryIndex));
  mapping.map(rewrite.reduction->getOperand(1),
              replacement.getResult(carryIndex + 1));
  Value reduced = builder.clone(*rewrite.reduction, mapping)->getResult(0);
  if (!rewrite.maximum)
    old.getResult(0).replaceAllUsesWith(replacement.getResult(0));
  old.getResult(carryIndex).replaceAllUsesWith(reduced);
  old.erase();
}

struct DenseProbability {
  func::FuncOp function;
  func::CallOp call;
  scf::ForOp loop;
  arith::AddFOp sum;
  arith::TruncFOp truncate;
  vector::ShapeCastOp sumShape, pShape;
  vector::TransferWriteOp store;
  tensor::ExtractSliceOp view;
  tensor::InsertSliceOp insert;
  tensor::EmptyOp destination;
  tensor::ExpandShapeOp publicationView;
  CopyOp copy;
  unsigned destinationIndex, resultIndex;
  int64_t keys;
};

bool onlyAttributes(Operation *op, ArrayRef<StringRef> names) {
  return llvm::all_of(op->getAttrs(), [&](NamedAttribute attribute) {
    return llvm::is_contained(names, attribute.getName().strref());
  });
}

std::optional<DenseProbability>
matchDenseProbability(func::FuncOp function, func::CallOp call, int64_t keys) {
  if (!tensorOnlyVF(function) || !plainCallMetadata(call) ||
      !protectedPScope(call, call, /*allowTensorCalls=*/false,
                       /*allowIdentical=*/true))
    return std::nullopt;
  for (unsigned i = 0; i < function.getFunctionType().getNumResults(); ++i)
    if (auto attributes = function.getResultAttrDict(i);
        attributes && !attributes.empty())
      return std::nullopt;
  auto ret = cast<func::ReturnOp>(function.getBody().front().getTerminator());
  Builder b(function.getContext());
  SmallVector<unsigned> pResults;
  for (unsigned i = 0; i < ret.getNumOperands(); ++i)
    if (shape(ret.getOperand(i).getType(), {4, keys, 16}, b.getF16Type()))
      pResults.push_back(i);
  if (pResults.size() != 1)
    return std::nullopt;
  unsigned resultIndex = pResults.front();
  auto loop = ret.getOperand(resultIndex).getDefiningOp<scf::ForOp>();
  if (!loop || ret.getOperand(resultIndex) != loop.getResult(0) ||
      !loop.getResult(0).hasOneUse())
    return std::nullopt;
  auto reduction = matchKeyReduction(loop, keys);
  if (!reduction || reduction->maximum)
    return std::nullopt;
  auto sum = cast<arith::AddFOp>(reduction->reduction);
  if (!sum.getResult().hasOneUse())
    return std::nullopt;
  Value sumTerm =
      sum.getLhs() == loop.getRegionIterArg(1) ? sum.getRhs() : sum.getLhs();
  auto sumShape = sumTerm.getDefiningOp<vector::ShapeCastOp>();
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  auto insert = yield.getOperand(0).getDefiningOp<tensor::InsertSliceOp>();
  auto store = insert.getSource().getDefiningOp<vector::TransferWriteOp>();
  auto pShape = store.getVector().getDefiningOp<vector::ShapeCastOp>();
  auto truncate = pShape.getSource().getDefiningOp<arith::TruncFOp>();
  auto view = store.getSource().getDefiningOp<tensor::ExtractSliceOp>();
  auto argument = dyn_cast<BlockArgument>(loop.getInitArgs()[0]);
  if (!argument || argument.getOwner() != &function.getBody().front() ||
      !argument.hasOneUse() || !pShape.getResult().hasOneUse())
    return std::nullopt;
  auto destination =
      call.getOperand(argument.getArgNumber()).getDefiningOp<tensor::EmptyOp>();
  if (!destination || !destination.getResult().hasOneUse() ||
      !onlyAttributes(destination, {"static_sizes"}) ||
      !call.getResult(resultIndex).hasOneUse())
    return std::nullopt;
  auto publicationView = dyn_cast<tensor::ExpandShapeOp>(
      *call.getResult(resultIndex).getUsers().begin());
  if (!publicationView || !publicationView.getResult().hasOneUse() ||
      publicationView->getParentOp() != call->getParentOp() ||
      !onlyAttributes(publicationView,
                      {"reassociation", "static_output_shape"}) ||
      publicationView.getReassociationIndices() !=
          SmallVector<ReassociationIndices>{{0}, {1, 2}, {3}} ||
      !shape(publicationView.getType(), {4, keys / 16, 16, 16}, b.getF16Type()))
    return std::nullopt;
  auto copy = dyn_cast<CopyOp>(*publicationView.getResult().getUsers().begin());
  if (!copy || copy.getSrc() != publicationView.getResult() ||
      copy->getNumOperands() != 2 || copy->getNumResults() != 0 ||
      copy->getParentOp() != call->getParentOp() ||
      !call->isBeforeInBlock(copy) ||
      !onlyAttributes(
          copy, {"operandSegmentSizes", "hivm.inserted-copy", "tiled_op"}))
    return std::nullopt;
  auto target = dyn_cast<MemRefType>(copy.getDst().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!target || target.getShape() != ArrayRef<int64_t>{4, keys / 16, 16, 16} ||
      !target.getElementType().isF16() ||
      target.getMemorySpace() !=
          AddressSpaceAttr::get(function.getContext(), AddressSpace::L1) ||
      failed(getStridesAndOffset(target, strides, offset)) ||
      !llvm::all_of(strides, [](int64_t stride) { return stride > 0; }))
    return std::nullopt;
  return DenseProbability{function,    call,
                          loop,        sum,
                          truncate,    sumShape,
                          pShape,      store,
                          view,        insert,
                          destination, publicationView,
                          copy,        argument.getArgNumber(),
                          resultIndex, keys};
}

scf::ForOp applyDenseProbability(DenseProbability &rewrite) {
  // A packed vector holds keys k and k + K/2 for all 64 query rows. Pair the
  // original FP32 values before rounding; the caller explicitly permits this
  // sum reassociation. The two L1 copies reconstruct the original logical P.
  // Physical H+1 padding is a separate post-bufferization transformation.
  auto old = rewrite.loop;
  auto function = rewrite.function;
  OpBuilder builder(function.getContext());
  int64_t half = rewrite.keys / 2;
  auto denseType = RankedTensorType::get({8, half, 16}, builder.getF16Type());
  SmallVector<Type> arguments(function.getArgumentTypes());
  SmallVector<Type> results(function.getResultTypes());
  arguments[rewrite.destinationIndex] = denseType;
  results[rewrite.resultIndex] = denseType;
  function.setType(
      FunctionType::get(function.getContext(), arguments, results));
  function.getArgument(rewrite.destinationIndex).setType(denseType);
  builder.setInsertionPoint(old);
  Value limit = builder.create<arith::ConstantIndexOp>(old.getLoc(), half);
  auto replacement = builder.create<scf::ForOp>(
      old.getLoc(), old.getLowerBound(), limit, old.getStep(),
      ValueRange{function.getArgument(rewrite.destinationIndex),
                 old.getInitArgs()[1]},
      [&](OpBuilder &body, Location loc, Value key, ValueRange carried) {
        SmallVector<Value> probabilities, sums;
        for (unsigned plane = 0; plane < 2; ++plane) {
          Value index = key;
          if (plane)
            index = body.create<arith::AddIOp>(loc, key, limit);
          IRMapping mapping;
          mapping.map(old.getInductionVar(), index);
          for (Operation &op : old.getBody()->without_terminator()) {
            if (&op == rewrite.sum.getOperation() ||
                &op == rewrite.pShape.getOperation() ||
                &op == rewrite.store.getOperation() ||
                &op == rewrite.view.getOperation() ||
                &op == rewrite.insert.getOperation())
              continue;
            body.clone(op, mapping);
          }
          sums.push_back(mapping.lookup(rewrite.sumShape.getResult()));
          probabilities.push_back(body.create<vector::ShapeCastOp>(
              loc, VectorType::get({64}, body.getF16Type()),
              mapping.lookup(rewrite.truncate.getResult())));
        }
        auto add = [&](Value lhs, Value rhs) {
          IRMapping mapping;
          mapping.map(rewrite.sum.getLhs(), lhs);
          mapping.map(rewrite.sum.getRhs(), rhs);
          return body.clone(*rewrite.sum, mapping)->getResult(0);
        };
        Value pair = add(sums[0], sums[1]);
        Value nextSum = rewrite.sum.getLhs() == old.getRegionIterArg(1)
                            ? add(carried[1], pair)
                            : add(pair, carried[1]);
        auto packedType = VectorType::get({128}, body.getF16Type());
        auto packed = body.create<hivmave::VFDeInterleaveOp>(
            loc, packedType, packedType, probabilities[0], probabilities[1],
            hivmave::Layout_Change::DENSE);
        Value vector = body.create<vector::ShapeCastOp>(
            loc, VectorType::get({8, 1, 16}, body.getF16Type()),
            packed.getRes1());
        SmallVector<OpFoldResult> offsets{body.getIndexAttr(0), key,
                                          body.getIndexAttr(0)};
        SmallVector<OpFoldResult> sizes{
            body.getIndexAttr(8), body.getIndexAttr(1), body.getIndexAttr(16)};
        SmallVector<OpFoldResult> strides(3, body.getIndexAttr(1));
        Value view = body.create<tensor::ExtractSliceOp>(
            loc, carried[0], offsets, sizes, strides);
        Value zero = body.create<arith::ConstantIndexOp>(loc, 0);
        auto store = body.create<vector::TransferWriteOp>(
            loc, vector, view, ValueRange{zero, zero, zero});
        store->setAttrs(rewrite.store->getAttrs());
        Value inserted = body.create<tensor::InsertSliceOp>(
            loc, store.getResult(), carried[0], offsets, sizes, strides);
        body.create<scf::YieldOp>(loc, ValueRange{inserted, nextSum});
      });
  replacement->setAttrs(old->getAttrs());
  old.getResult(0).replaceAllUsesWith(replacement.getResult(0));
  old.getResult(1).replaceAllUsesWith(replacement.getResult(1));
  old.erase();

  builder.setInsertionPoint(rewrite.destination);
  Value destination = builder.create<tensor::EmptyOp>(
      rewrite.destination.getLoc(), denseType.getShape(),
      denseType.getElementType());
  SmallVector<Value> operands(rewrite.call.getOperands());
  operands[rewrite.destinationIndex] = destination;
  builder.setInsertionPoint(rewrite.call);
  auto call =
      builder.create<func::CallOp>(rewrite.call.getLoc(), function, operands);
  call->setAttrs(rewrite.call->getAttrs());
  // Keep the tensor/function ABI logical. The existing annotation bufferizer
  // forwards this request to P's buffer, and EnableStrideAlign then pads that
  // allocation and propagates its physical strides through all VF/copy views.
  // Regbase alignment units use element-byte units for this dimension: H+1,
  // rather than the complete row stride (H+1)*16, is the requested extent.
  auto padding = builder.create<annotation::MarkOp>(
      call.getLoc(), call.getResult(rewrite.resultIndex));
  padding->setAttr(StrideAlignDimsAttr::name,
                   builder.getDenseI32ArrayAttr({1}));
  padding->setAttr(
      StrideAlignValueInByteAttr::name,
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(2 * (half + 1))}));
  for (unsigned i = 0; i < call.getNumResults(); ++i)
    if (i != rewrite.resultIndex)
      rewrite.call.getResult(i).replaceAllUsesWith(call.getResult(i));
  builder.setInsertionPoint(rewrite.copy);
  for (int64_t plane = 0; plane < 2; ++plane) {
    SmallVector<OpFoldResult> offsets{builder.getIndexAttr(plane * 4),
                                      builder.getIndexAttr(0),
                                      builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{builder.getIndexAttr(4),
                                    builder.getIndexAttr(half),
                                    builder.getIndexAttr(16)};
    SmallVector<OpFoldResult> strides(3, builder.getIndexAttr(1));
    Value slice = builder.create<tensor::ExtractSliceOp>(
        rewrite.copy.getLoc(), call.getResult(rewrite.resultIndex), offsets,
        sizes, strides);
    Value expanded = builder.create<tensor::ExpandShapeOp>(
        rewrite.publicationView.getLoc(),
        RankedTensorType::get({4, half / 16, 16, 16}, builder.getF16Type()),
        slice, ArrayRef<ReassociationIndices>{{0}, {1, 2}, {3}});
    SmallVector<OpFoldResult> dstOffsets{
        builder.getIndexAttr(0), builder.getIndexAttr(plane * half / 16),
        builder.getIndexAttr(0), builder.getIndexAttr(0)};
    SmallVector<OpFoldResult> dstSizes{
        builder.getIndexAttr(4), builder.getIndexAttr(half / 16),
        builder.getIndexAttr(16), builder.getIndexAttr(16)};
    SmallVector<OpFoldResult> dstStrides(4, builder.getIndexAttr(1));
    Value target = builder.create<memref::SubViewOp>(
        rewrite.copy.getLoc(), rewrite.copy.getDst(), dstOffsets, dstSizes,
        dstStrides);
    IRMapping mapping;
    mapping.map(rewrite.copy.getSrc(), expanded);
    mapping.map(rewrite.copy.getDst(), target);
    builder.clone(*rewrite.copy, mapping);
  }
  rewrite.copy.erase();
  rewrite.publicationView.erase();
  rewrite.call.erase();
  rewrite.destination.erase();
  function->setAttr(
      "hivm.cvp_attention.dense_p",
      builder.getDictionaryAttr(
          {builder.getNamedAttr("keys",
                                builder.getI64IntegerAttr(rewrite.keys)),
           builder.getNamedAttr("query_rows", builder.getI64IntegerAttr(64)),
           builder.getNamedAttr("padding_required",
                                builder.getBoolAttr(true))}));
  return replacement;
}

struct CVPAttentionQueryVectorPass
    : impl::CVPAttentionQueryVectorBase<CVPAttentionQueryVectorPass> {
  using Base::Base;
  void runOnOperation() override {
    if (enableDenseP && ((reductionChains != 1 && reductionChains != 2) ||
                         !allowSumReassociation)) {
      getOperation().emitError(
          "CVP dense P requires one- or two-chain policy and "
          "explicit FP32 sum reassociation permission");
      signalPassFailure();
      return;
    }
    if (reductionChains != 1 && reductionChains != 2 && reductionChains != 4) {
      getOperation().emitError("CVP reduction chains must be 1, 2 or 4");
      signalPassFailure();
      return;
    }
    if (reductionChains == 2 &&
        (!enableDenseP || !fuseMaximum || !fuseProbability || !fuseState ||
         !enableScoreRematerialization || !allowSumReassociation)) {
      getOperation().emitError(
          "CVP two-chain reduction requires dense P, fused maximum/P/state, "
          "score rematerialization and explicit FP32 sum reassociation");
      signalPassFailure();
      return;
    }
    if (reductionChains == 4 && !allowSumReassociation) {
      getOperation().emitError("CVP four-chain reduction requires explicit "
                               "FP32 sum reassociation permission");
      signalPassFailure();
      return;
    }
    if (fuseMaximum &&
        (!fuseProbability || !fuseState || !enableScoreRematerialization)) {
      getOperation().emitError(
          "CVP maximum fusion requires score rematerialization, "
          "probability fusion and state fusion");
      signalPassFailure();
      return;
    }
    if (enableMaxScaling && !enableScoreRematerialization) {
      getOperation().emitError(
          "CVP query max scaling requires score rematerialization");
      signalPassFailure();
      return;
    }
    if (!fuseProbability && !enableScoreRematerialization && !fuseState &&
        reductionChains == 1 && !enableDenseP)
      return;
    ModuleOp original = getOperation();
    auto module = cast<ModuleOp>(original->clone());
    auto cleanup = llvm::make_scope_exit([&] { module.erase(); });
    SmallVector<Fusion> rewrites;
    llvm::SmallPtrSet<Operation *, 8> touched;
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (!fuseProbability)
        break;
      auto keys = keyMajorKeys(caller);
      if (!keys)
        continue;
      caller.walk([&](func::CallOp first) {
        if (first.getNumResults() != 1 || !first.getResult(0).hasOneUse())
          return;
        auto second =
            dyn_cast<func::CallOp>(*first.getResult(0).getUsers().begin());
        auto producerFunction =
            module.lookupSymbol<func::FuncOp>(first.getCallee());
        auto consumerFunction =
            second ? module.lookupSymbol<func::FuncOp>(second.getCallee())
                   : func::FuncOp{};
        if (!second || !producerFunction || !consumerFunction ||
            producerFunction == consumerFunction ||
            touched.contains(producerFunction) ||
            touched.contains(consumerFunction))
          return;
        if (!plainCallMetadata(first) || !plainCallMetadata(second) ||
            !uniqueCall(producerFunction, first, module) ||
            !uniqueCall(consumerFunction, second, module)) {
          if (requireMatch)
            first.emitRemark("CVP probability pair rejected: call metadata or "
                             "shared callee");
          return;
        }
        if (!protectedPScope(first, second)) {
          if (requireMatch)
            first.emitRemark(
                "CVP probability pair rejected: protected Skew P scope");
          return;
        }
        std::string producerReason, consumerReason;
        auto producer = matchProducer(producerFunction, *keys, producerReason);
        auto consumer = matchConsumer(consumerFunction, *keys, consumerReason);
        if (requireMatch && (!producer || !consumer))
          first.emitRemark() << "CVP probability pair rejected: "
                             << (producer ? consumerReason : producerReason);
        if (!producer || !consumer ||
            second.getOperand(consumer->inputIndex) != first.getResult(0))
          return;
        rewrites.push_back({*producer, *consumer, first, second});
        touched.insert(producerFunction);
        touched.insert(consumerFunction);
      });
    }
    if (fuseProbability && rewrites.empty() && requireMatch) {
      original.emitError("CVP probability fusion requires a closed key-major "
                         "tensor VF pair in one protected Skew P scope");
      signalPassFailure();
      return;
    }
    for (Fusion &fusion : rewrites)
      applyProbabilityFusion(fusion);
    addCount(module, kProbabilityFusions, rewrites.size());

    // Match after probability fusion so the same typed reader can be either
    // the standalone exp VF or its fused sum/cast successor.
    SmallVector<ScoreRematerialization> scoreRewrites;
    touched.clear();
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (!enableScoreRematerialization)
        break;
      auto keys = keyMajorKeys(caller);
      if (!keys)
        continue;
      caller.walk([&](func::CallOp first) {
        if (first.getNumResults() != 2 || !first.getResult(0).hasOneUse())
          return;
        auto second =
            dyn_cast<func::CallOp>(*first.getResult(0).getUsers().begin());
        auto producerFunction =
            module.lookupSymbol<func::FuncOp>(first.getCallee());
        auto consumerFunction =
            second ? module.lookupSymbol<func::FuncOp>(second.getCallee())
                   : func::FuncOp{};
        if (!second || !producerFunction || !consumerFunction ||
            producerFunction == consumerFunction ||
            touched.contains(producerFunction) ||
            touched.contains(consumerFunction) || !plainCallMetadata(first) ||
            !plainCallMetadata(second) ||
            !uniqueCall(producerFunction, first, module) ||
            !uniqueCall(consumerFunction, second, module) ||
            !protectedPScope(first, second, /*allowTensorCalls=*/true))
          return;
        auto producer = matchScoreProducer(producerFunction, *keys);
        auto consumer = matchProbabilityExpression(consumerFunction, *keys);
        if (!producer || !consumer ||
            second.getOperand(consumer->inputIndex) != first.getResult(0))
          return;
        scoreRewrites.push_back({*producer, *consumer, first, second});
        touched.insert(producerFunction);
        touched.insert(consumerFunction);
      });
    }
    if (enableScoreRematerialization && scoreRewrites.empty() && requireMatch) {
      original.emitError(
          "CVP score rematerialization requires a closed key-major "
          "scale/max and probability pair in one protected Skew P scope");
      signalPassFailure();
      return;
    }
    for (ScoreRematerialization &rewrite : scoreRewrites)
      applyScoreRematerialization(rewrite, enableMaxScaling);
    addCount(module, kScoreRematerializations, scoreRewrites.size());
    if (enableMaxScaling)
      addCount(module, kQueryMaxScalings, scoreRewrites.size());
    SmallVector<StateFusion> stateRewrites;
    touched.clear();
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (!fuseState)
        break;
      auto keys = keyMajorKeys(caller);
      if (!keys)
        continue;
      caller.walk([&](func::CallOp first) {
        if (first.getNumResults() != 2 || !first.getResult(0).hasOneUse())
          return;
        auto second =
            dyn_cast<func::CallOp>(*first.getResult(0).getUsers().begin());
        auto producerFunction =
            module.lookupSymbol<func::FuncOp>(first.getCallee());
        auto consumerFunction =
            second ? module.lookupSymbol<func::FuncOp>(second.getCallee())
                   : func::FuncOp{};
        if (!second || !producerFunction || !consumerFunction ||
            producerFunction == consumerFunction ||
            touched.contains(producerFunction) ||
            touched.contains(consumerFunction) || !plainCallMetadata(first) ||
            !plainCallMetadata(second) ||
            !uniqueCall(producerFunction, first, module) ||
            !uniqueCall(consumerFunction, second, module) ||
            !protectedPScope(first, second))
          return;
        auto producer = matchSumProducer(producerFunction, *keys);
        auto consumer = matchStateConsumer(consumerFunction);
        if (!producer || !consumer ||
            second.getOperand(consumer->sumIndex) != first.getResult(0))
          return;
        // Placing the fused call at the state update must dominate every P use.
        for (Operation *user : first.getResult(1).getUsers())
          if (user->getBlock() != second->getBlock() ||
              !second->isBeforeInBlock(user))
            return;
        stateRewrites.push_back({*producer, *consumer, first, second});
        touched.insert(producerFunction);
        touched.insert(consumerFunction);
      });
    }
    if (fuseState && stateRewrites.empty() && requireMatch) {
      original.emitError(
          "CVP state fusion requires a closed key-major P-sum and "
          "alpha/l pair in one protected Skew P scope");
      signalPassFailure();
      return;
    }
    for (StateFusion &fusion : stateRewrites)
      applyStateFusion(fusion);
    addCount(module, kStateFusions, stateRewrites.size());
    SmallVector<MaximumFusion> maximumRewrites;
    touched.clear();
    for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
      if (!fuseMaximum)
        break;
      auto keys = keyMajorKeys(caller);
      if (!keys)
        continue;
      caller.walk([&](func::CallOp first) {
        if (first.getNumResults() != 1 || !first.getResult(0).hasOneUse())
          return;
        auto middle =
            dyn_cast<func::CallOp>(*first.getResult(0).getUsers().begin());
        if (!middle || middle.getNumResults() != 2 ||
            !middle.getResult(0).use_empty())
          return;
        SmallVector<func::CallOp> consumers;
        for (Operation *user : middle.getResult(1).getUsers())
          if (auto call = dyn_cast<func::CallOp>(user))
            consumers.push_back(call);
        if (consumers.size() != 1)
          return;
        auto last = consumers.front();
        auto producerFunction =
            module.lookupSymbol<func::FuncOp>(first.getCallee());
        auto rowFunction =
            module.lookupSymbol<func::FuncOp>(middle.getCallee());
        auto consumerFunction =
            module.lookupSymbol<func::FuncOp>(last.getCallee());
        if (!producerFunction || !rowFunction || !consumerFunction ||
            producerFunction == rowFunction ||
            producerFunction == consumerFunction ||
            rowFunction == consumerFunction ||
            touched.contains(producerFunction) ||
            touched.contains(rowFunction) ||
            touched.contains(consumerFunction) || !plainCallMetadata(first) ||
            !plainCallMetadata(middle) || !plainCallMetadata(last) ||
            !uniqueCall(producerFunction, first, module) ||
            !uniqueCall(rowFunction, middle, module) ||
            !uniqueCall(consumerFunction, last, module) ||
            !protectedPScope(first, middle) ||
            !protectedPScope(first, last, /*allowTensorCalls=*/true))
          return;
        auto producer = matchMaximumProducer(producerFunction, *keys);
        auto row = matchRowMaximum(rowFunction);
        auto consumer = matchFusedProbabilityState(consumerFunction, *keys);
        if (!producer || !row || !consumer ||
            middle.getOperand(row->tileIndex) != first.getResult(0) ||
            last.getOperand(consumer->rawIndex) !=
                first.getOperand(producer->inputIndex) ||
            last.getOperand(consumer->maximumIndex) != middle.getResult(1) ||
            last.getOperand(consumer->oldIndex) !=
                middle.getOperand(row->oldIndex) ||
            llvm::count(last.getOperands(), middle.getResult(1)) != 1)
          return;
        for (Operation *user : middle.getResult(1).getUsers())
          if (user != last && (user->getBlock() != last->getBlock() ||
                               !last->isBeforeInBlock(user)))
            return;
        maximumRewrites.push_back(
            {*producer, *row, *consumer, first, middle, last});
        touched.insert(producerFunction);
        touched.insert(rowFunction);
        touched.insert(consumerFunction);
      });
    }
    if (fuseMaximum && maximumRewrites.empty() && requireMatch) {
      original.emitError(
          "CVP maximum fusion requires a closed key-major tile-max, "
          "row-combine and fused P/state chain in one protected Skew P scope");
      signalPassFailure();
      return;
    }
    for (MaximumFusion &fusion : maximumRewrites)
      applyMaximumFusion(fusion);
    addCount(module, kMaximumFusions, maximumRewrites.size());
    SmallVector<KeyReduction> chainRewrites;
    if (reductionChains == 4) {
      for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
        auto keys = keyMajorKeys(caller);
        if (!keys)
          continue;
        auto proofs =
            caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
        auto policy = cast<DictionaryAttr>(proofs[0]).getAs<StringAttr>(
            "numerical_policy");
        if (!policy || policy.getValue() != "reassociate-fp32-sum")
          continue;
        caller.walk([&](scope::ScopeOp scopeOp) {
          SmallVector<KeyReduction> candidates;
          for (auto call : scopeOp.getBody()->getOps<func::CallOp>()) {
            auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
            if (!plainCallMetadata(call) || !tensorOnlyVF(function) ||
                !uniqueCall(function, call, module) ||
                !protectedPScope(call, call, /*allowTensorCalls=*/false,
                                 /*allowIdentical=*/true))
              continue;
            function.walk([&](scf::ForOp loop) {
              if (auto reduction = matchKeyReduction(loop, *keys))
                candidates.push_back(*reduction);
            });
          }
          if (candidates.size() == 2 &&
              llvm::count_if(candidates, [](const KeyReduction &r) {
                return r.maximum;
              }) == 1)
            llvm::append_range(chainRewrites, candidates);
        });
      }
      if (chainRewrites.empty() && requireMatch) {
        original.emitError(
            "CVP four-chain reduction requires closed full-key maximum and "
            "FP32 P-sum loops in a protected Skew P scope");
        signalPassFailure();
        return;
      }
    }
    for (KeyReduction &rewrite : chainRewrites)
      applyFourChains(rewrite);
    addCount(module, kReductionChainLoops, chainRewrites.size());
    SmallVector<DenseProbability> denseRewrites;
    if (enableDenseP) {
      for (func::FuncOp caller : module.getOps<func::FuncOp>()) {
        auto keys = keyMajorKeys(caller);
        if (!keys)
          continue;
        auto proofs =
            caller->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
        auto policy = cast<DictionaryAttr>(proofs[0]).getAs<StringAttr>(
            "numerical_policy");
        if (!policy || policy.getValue() != "reassociate-fp32-sum")
          continue;
        caller.walk([&](func::CallOp call) {
          auto function = module.lookupSymbol<func::FuncOp>(call.getCallee());
          if (!function || !uniqueCall(function, call, module))
            return;
          if (auto rewrite = matchDenseProbability(function, call, *keys))
            denseRewrites.push_back(*rewrite);
        });
      }
      if (denseRewrites.empty() && (requireMatch || reductionChains == 2)) {
        original.emitError("CVP dense P requires a closed full-key P producer "
                           "and canonical L1 publication");
        signalPassFailure();
        return;
      }
    }
    SmallVector<KeyReduction> denseMaximumRewrites;
    if (reductionChains == 2) {
      // Validate every producer before mutating any of its loops. In
      // particular, do not silently fall back to one chain when maximum fusion
      // was absent.
      for (DenseProbability &rewrite : denseRewrites) {
        SmallVector<KeyReduction> maxima;
        rewrite.function.walk([&](scf::ForOp loop) {
          if (auto reduction = matchKeyReduction(loop, rewrite.keys);
              reduction && reduction->maximum)
            maxima.push_back(*reduction);
        });
        if (maxima.size() != 1) {
          original.emitError(
              "CVP two-chain dense P requires one closed full-key maximum "
              "inside each fused probability/state producer");
          signalPassFailure();
          return;
        }
        denseMaximumRewrites.push_back(maxima.front());
      }
    }
    for (auto [index, rewrite] : llvm::enumerate(denseRewrites)) {
      auto packedLoop = applyDenseProbability(rewrite);
      if (reductionChains != 2)
        continue;
      applyTwoChains(denseMaximumRewrites[index], /*keyGroups=*/4);
      auto yield = cast<scf::YieldOp>(packedLoop.getBody()->getTerminator());
      auto sum = cast<arith::AddFOp>(yield.getOperand(1).getDefiningOp());
      KeyReduction packedSum{packedLoop, sum.getOperation(), false};
      applyTwoChains(packedSum, /*keyGroups=*/2);
    }
    if (reductionChains == 2)
      addCount(module, kDenseTwoChainLoops, 2 * denseRewrites.size());
    addCount(module, kDensePLayouts, denseRewrites.size());
    if (rewrites.empty() && scoreRewrites.empty() && stateRewrites.empty() &&
        maximumRewrites.empty() && chainRewrites.empty() &&
        denseRewrites.empty())
      return;
    if (failed(verify(module))) {
      original.emitError(
          "CVP query-vector transformation failed IR verification");
      signalPassFailure();
      return;
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};
} // namespace

std::unique_ptr<Pass> createCVPAttentionQueryVectorPass(
    const CVPAttentionQueryVectorOptions &options) {
  return std::make_unique<CVPAttentionQueryVectorPass>(options);
}
} // namespace hivm
} // namespace mlir

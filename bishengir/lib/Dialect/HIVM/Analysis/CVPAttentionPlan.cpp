//===- CVPAttentionPlan.cpp - Typed attention recognition
//-------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::hivm;

StringRef mlir::hivm::stringifyCVPAttentionRole(CVPAttentionRole role) {
  switch (role) {
  case CVPAttentionRole::QK:
    return "qk";
  case CVPAttentionRole::QKTransfer:
    return "qk.transfer";
  case CVPAttentionRole::Scale:
    return "scale";
  case CVPAttentionRole::TileMax:
    return "tile_max";
  case CVPAttentionRole::Max:
    return "max";
  case CVPAttentionRole::P32:
    return "p32";
  case CVPAttentionRole::P16:
    return "p16";
  case CVPAttentionRole::PV:
    return "pv";
  case CVPAttentionRole::PVTransfer:
    return "pv.transfer";
  case CVPAttentionRole::Alpha:
    return "alpha";
  case CVPAttentionRole::TileSum:
    return "tile_sum";
  case CVPAttentionRole::Sum:
    return "sum";
  case CVPAttentionRole::Output:
    return "output";
  }
  llvm_unreachable("unknown attention role");
}

namespace {
Value stripShape(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}
Value stripMemoryCast(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::MemorySpaceCastOp, memref::CastOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}
FloatAttr scalarConstant(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  return constant ? dyn_cast<FloatAttr>(constant.getValue()) : FloatAttr();
}
FloatAttr initialConstant(Value value) {
  value = stripShape(value);
  if (auto brc = value.getDefiningOp<VBrcOp>())
    return scalarConstant(brc->getOperand(0));
  return scalarConstant(value);
}
std::optional<int64_t> integerConstant(Value value) {
  APInt integer;
  if (!matchPattern(value, m_ConstantInt(&integer)))
    return std::nullopt;
  return integer.getBitWidth() == 1 ? integer.getZExtValue()
                                    : integer.getSExtValue();
}
std::optional<unsigned> stateIndex(Value value, scf::ForOp loop) {
  auto arg = dyn_cast<BlockArgument>(stripShape(value));
  if (!arg || arg.getOwner() != loop.getBody() || arg.getArgNumber() == 0)
    return std::nullopt;
  return arg.getArgNumber() - 1;
}
bool sameShape(Value value, ArrayRef<int64_t> shape, unsigned bits) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.hasStaticShape() && type.getShape() == shape &&
         type.getElementType().isF32() == (bits == 32) &&
         (bits == 32 || type.getElementType().isF16());
}
bool isLocalAllocation(Value value, scf::ForOp loop) {
  auto alloc = stripMemoryCast(value).getDefiningOp<memref::AllocOp>();
  return alloc && alloc->getBlock() == loop.getBody();
}
/// Only whole-buffer casts and tensor views are allowed around a private
/// allocation. A subview, escape or additional writer invalidates this proof.
template <typename Writer> Writer uniqueWriter(Value tensor, scf::ForOp loop) {
  auto view = stripShape(tensor).getDefiningOp<bufferization::ToTensorOp>();
  if (!view || !view->hasAttr("restrict"))
    return {};
  Value root = stripMemoryCast(view->getOperand(0));
  if (!isLocalAllocation(root, loop))
    return {};
  SmallVector<Value> pending{root};
  llvm::SmallPtrSet<Operation *, 8> seen;
  Writer writer;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (!seen.insert(user).second)
        continue;
      if (user->getBlock() != loop.getBody())
        return {};
      if (isa<memref::MemorySpaceCastOp, memref::CastOp>(user)) {
        pending.push_back(user->getResult(0));
        continue;
      }
      if (isa<bufferization::ToTensorOp, annotation::MarkOp>(user))
        continue;
      auto candidate = dyn_cast<Writer>(user);
      if (!candidate || writer || stripMemoryCast(candidate.getDst()) != root)
        return {};
      writer = candidate;
    }
  }
  return writer;
}
MmadL1Op matrixProducer(Value tensor, scf::ForOp loop, FixpipeOp &transfer) {
  transfer = uniqueWriter<FixpipeOp>(tensor, loop);
  return transfer ? transfer.getSrc().getDefiningOp<MmadL1Op>() : MmadL1Op();
}
bool reduction(VReduceOp op, ReduceOperation kind, int64_t keyAxis) {
  return op && op->getNumResults() == 1 &&
         op.getArith().getReduceOp() == kind &&
         op.getReduceDims() == ArrayRef<int64_t>{keyAxis};
}
template <typename Op> Op defining(Value value) {
  return stripShape(value).getDefiningOp<Op>();
}
bool multiplication(Value value, Value factor, Value &other) {
  auto op = defining<VMulOp>(value);
  if (!op)
    return false;
  for (unsigned index : {0u, 1u}) {
    if (stripShape(op->getOperand(index)) == factor) {
      other = stripShape(op->getOperand(1 - index));
      return true;
    }
  }
  return false;
}
bool supportedEffects(scf::ForOp loop) {
  for (Operation &op : *loop.getBody()) {
    if (isa<func::CallOp>(&op) || op.getNumRegions())
      return false;
    if (auto transfer = dyn_cast<FixpipeOp>(&op)) {
      if (!isLocalAllocation(transfer.getDst(), loop))
        return false;
    } else if (auto copy = dyn_cast<CopyOp>(&op)) {
      if (!isLocalAllocation(copy.getDst(), loop))
        return false;
    } else if (auto load = dyn_cast<ND2NZOp>(&op)) {
      if (!isLocalAllocation(load.getDst(), loop))
        return false;
    } else if (auto view = dyn_cast<bufferization::ToTensorOp>(&op)) {
      if (!view->hasAttr("restrict") ||
          !isLocalAllocation(view->getOperand(0), loop))
        return false;
    } else if (!isa<memref::AllocOp, annotation::MarkOp, scf::YieldOp>(&op) &&
               !isMemoryEffectFree(&op)) {
      return false;
    }
  }
  return true;
}
} // namespace

FailureOr<CVPAttentionPlan>
mlir::hivm::analyzeCVPAttention(scf::ForOp loop, std::string &reason,
                                CVPAttentionScoreLayout layout) {
  const bool keyMajor = layout == CVPAttentionScoreLayout::KeyMajor;
  const int64_t keyAxis = keyMajor ? 0 : 1;
  auto reject = [&](StringRef message) -> FailureOr<CVPAttentionPlan> {
    reason = message.str();
    return failure();
  };
  SmallVector<MmadL1Op> matrices(loop.getBody()->getOps<MmadL1Op>());
  if (matrices.size() != 2)
    return reject("expected two direct tensor matmuls");
  for (MmadL1Op matrix : matrices) {
    for (Value input : {matrix.getA(), matrix.getB()}) {
      auto type = dyn_cast<RankedTensorType>(input.getType());
      if (!type || !type.getElementType().isF16())
        return reject("attention matrix inputs must be FP16");
    }
  }
  CVPAttentionPlan plan;
  plan.loop = loop;
  plan.scoreLayout = layout;
  auto yielded =
      cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperands();
  VMaxOp maximum;
  VReduceOp tileMax;
  for (Value value : yielded) {
    auto candidate = defining<VMaxOp>(value);
    if (!candidate)
      continue;
    for (unsigned index : {0u, 1u}) {
      auto state = stateIndex(candidate->getOperand(index), loop);
      auto reduce = defining<VReduceOp>(candidate->getOperand(1 - index));
      if (state && reduction(reduce, ReduceOperation::max, keyAxis)) {
        if (maximum)
          return reject("ambiguous maximum recurrence");
        maximum = candidate;
        tileMax = reduce;
        plan.maxState = *state;
      }
    }
  }
  if (!maximum)
    return reject("missing online maximum recurrence");
  auto scale = defining<VMulOp>(tileMax.getSrc());
  if (!scale)
    return reject("missing scaled QK input");
  FixpipeOp qkTransfer;
  for (unsigned index : {0u, 1u}) {
    auto constant = scalarConstant(scale->getOperand(index));
    if (!constant || !constant.getType().isF32() ||
        constant.getValueAsDouble() != 0.5)
      continue;
    plan.rawScore = scale->getOperand(1 - index);
    plan.qk = matrixProducer(plan.rawScore, loop, qkTransfer);
    plan.scale = constant;
  }
  if (!plan.qk)
    return reject("QK must have a private unique fixpipe and FP32 scale 0.5");
  auto scoreType = dyn_cast<RankedTensorType>(plan.rawScore.getType());
  if (!scoreType || scoreType.getRank() != 2 || !scoreType.hasStaticShape() ||
      !scoreType.getElementType().isF32())
    return reject("unsupported raw score type");
  plan.blockM = scoreType.getDimSize(1 - keyAxis);
  plan.blockN = scoreType.getDimSize(keyAxis);
  plan.scaledScore = scale->getResult(0);
  VExpOp probability, alpha;
  for (auto exp : loop.getBody()->getOps<VExpOp>()) {
    auto sub = defining<VSubOp>(exp->getOperand(0));
    if (!sub || stripShape(sub->getOperand(1)) != maximum->getResult(0))
      continue;
    Value lhs = stripShape(sub->getOperand(0));
    if (lhs == plan.scaledScore) {
      if (probability)
        return reject("ambiguous P producer");
      probability = exp;
    } else if (lhs == loop.getRegionIterArg(plan.maxState)) {
      if (alpha)
        return reject("ambiguous alpha producer");
      alpha = exp;
    }
  }
  if (!probability || !alpha)
    return reject("missing P or alpha exponential");
  plan.p32 = probability->getResult(0);
  plan.alpha = alpha->getResult(0);
  VCastOp probabilityCast;
  VReduceOp tileSum;
  for (Operation *user : plan.p32.getUsers()) {
    if (auto cast = dyn_cast<VCastOp>(user)) {
      if (probabilityCast)
        return reject("ambiguous FP16 P cast");
      probabilityCast = cast;
    } else if (auto reduce = dyn_cast<VReduceOp>(user)) {
      if (tileSum || !reduction(reduce, ReduceOperation::sum, keyAxis))
        return reject("unsupported P reduction");
      tileSum = reduce;
    } else
      return reject("P has an unsupported consumer");
  }
  if (!probabilityCast || !tileSum)
    return reject("missing FP16 P or row sum");
  plan.p16 = probabilityCast->getResult(0);
  if (!sameShape(plan.p16, scoreType.getShape(), 16))
    return reject("unsupported probability conversion type");
  VAddOp sumUpdate, outputUpdate;
  FixpipeOp pvTransfer;
  for (auto [index, value] : llvm::enumerate(yielded)) {
    auto add = defining<VAddOp>(value);
    if (!add)
      continue;
    for (unsigned operand : {0u, 1u}) {
      Value previous;
      if (!multiplication(add->getOperand(operand), plan.alpha, previous))
        continue;
      auto state = stateIndex(previous, loop);
      if (!state || *state != index)
        continue;
      Value contribution = add->getOperand(1 - operand);
      if (stripShape(contribution) == tileSum->getResult(0)) {
        if (sumUpdate)
          return reject("ambiguous sum recurrence");
        sumUpdate = add;
        plan.sumState = *state;
      } else {
        FixpipeOp transfer;
        auto matrix = matrixProducer(contribution, loop, transfer);
        if (!matrix || matrix == plan.qk)
          continue;
        if (outputUpdate)
          return reject("ambiguous output recurrence");
        outputUpdate = add;
        plan.pv = matrix;
        pvTransfer = transfer;
        plan.outputState = *state;
      }
    }
  }
  if (!sumUpdate || !outputUpdate)
    return reject("missing l or O recurrence");
  if (plan.maxState == plan.sumState || plan.maxState == plan.outputState ||
      plan.sumState == plan.outputState)
    return reject("overlapping state roles");
  auto outType =
      dyn_cast<RankedTensorType>(outputUpdate->getResult(0).getType());
  if (!outType || outType.getRank() != 2 || !outType.hasStaticShape())
    return reject("unsupported output tile type");
  plan.headDim = outType.getDimSize(1);
  if (plan.blockM != 128 || (plan.blockN != 128 && plan.blockN != 256) ||
      (plan.headDim != 64 && plan.headDim != 128) ||
      !sameShape(outputUpdate->getResult(0), {plan.blockM, plan.headDim}, 32) ||
      !sameShape(maximum->getResult(0), {plan.blockM}, 32) ||
      !sameShape(sumUpdate->getResult(0), {plan.blockM}, 32))
    return reject("unsupported attention geometry");
  auto lower = integerConstant(loop.getLowerBound()),
       upper = integerConstant(loop.getUpperBound()),
       step = integerConstant(loop.getStep());
  if (!lower || !upper || !step || *lower != 0 || *step != plan.blockN ||
      *upper < *step || *upper % *step)
    return reject("unsupported key iteration domain");
  plan.keyCount = *upper;
  auto matchesMatrix = [&](MmadL1Op matrix, int64_t m, int64_t k, int64_t n) {
    return integerConstant(matrix.getRealM()) == m &&
           integerConstant(matrix.getRealK()) == k &&
           integerConstant(matrix.getRealN()) == n &&
           integerConstant(matrix.getInitCondition()) == 1 &&
           matrix->getNumResults() == 1 && matrix->getNumOperands() == 7;
  };
  if (!matchesMatrix(plan.qk, keyMajor ? plan.blockN : plan.blockM,
                     plan.headDim, keyMajor ? plan.blockM : plan.blockN) ||
      !matchesMatrix(plan.pv, plan.blockM, plan.blockN, plan.headDim) ||
      plan.qk->hasAttr("a_transpose") || !plan.qk->hasAttr("b_transpose") ||
      plan.pv->hasAttr("a_transpose") != keyMajor ||
      plan.pv->hasAttr("b_transpose"))
    return reject("unsupported matrix orientation or accumulation");
  auto pCopy = uniqueWriter<CopyOp>(plan.pv.getA(), loop);
  if (!pCopy)
    return reject("P operand lacks a private unique copy");
  auto transpose = defining<VTransposeOp>(pCopy.getSrc());
  if (!transpose || transpose.getPermutation() != ArrayRef<int64_t>{1, 0, 2} ||
      stripShape(transpose->getOperand(0)) != plan.p16)
    return reject("PV operand is not the matched FP16 probability layout");
  plan.initialMax = initialConstant(loop.getInitArgs()[plan.maxState]);
  plan.initialSum = initialConstant(loop.getInitArgs()[plan.sumState]);
  plan.initialOutput = initialConstant(loop.getInitArgs()[plan.outputState]);
  auto maxInit = initialConstant(tileMax->getOperand(1));
  auto sumInit = initialConstant(tileSum->getOperand(1));
  if (!plan.initialMax || !plan.initialMax.getValue().isInfinity() ||
      !plan.initialMax.getValue().isNegative() || !plan.initialSum ||
      plan.initialSum.getValueAsDouble() != 1.0 || !plan.initialOutput ||
      !plan.initialOutput.getValue().isPosZero() ||
      maxInit != plan.initialMax || !sumInit || !sumInit.getValue().isPosZero())
    return reject("unsupported state or reduction initialization");
  if (!supportedEffects(loop))
    return reject("unknown effect, nested region or buffer escape");
  plan.roles = {{plan.qk, CVPAttentionRole::QK},
                {qkTransfer, CVPAttentionRole::QKTransfer},
                {scale, CVPAttentionRole::Scale},
                {tileMax, CVPAttentionRole::TileMax},
                {maximum, CVPAttentionRole::Max},
                {probability, CVPAttentionRole::P32},
                {probabilityCast, CVPAttentionRole::P16},
                {plan.pv, CVPAttentionRole::PV},
                {pvTransfer, CVPAttentionRole::PVTransfer},
                {alpha, CVPAttentionRole::Alpha},
                {tileSum, CVPAttentionRole::TileSum},
                {sumUpdate, CVPAttentionRole::Sum},
                {outputUpdate, CVPAttentionRole::Output}};
  reason.clear();
  return plan;
}

DictionaryAttr CVPAttentionPlan::getSummary(MLIRContext *context) const {
  Builder b(context);
  return b.getDictionaryAttr(
      {b.getNamedAttr("version", b.getI64IntegerAttr(1)),
       b.getNamedAttr("block_m", b.getI64IntegerAttr(blockM)),
       b.getNamedAttr("block_n", b.getI64IntegerAttr(blockN)),
       b.getNamedAttr("head_dim", b.getI64IntegerAttr(headDim)),
       b.getNamedAttr("key_count", b.getI64IntegerAttr(keyCount)),
       b.getNamedAttr("key_tiles", b.getI64IntegerAttr(keyCount / blockN)),
       b.getNamedAttr("output_state", b.getI64IntegerAttr(outputState)),
       b.getNamedAttr("sum_state", b.getI64IntegerAttr(sumState)),
       b.getNamedAttr("max_state", b.getI64IntegerAttr(maxState)),
       b.getNamedAttr("scale", scale),
       b.getNamedAttr("initial_max", initialMax),
       b.getNamedAttr("initial_sum", initialSum),
       b.getNamedAttr("initial_output", initialOutput),
       b.getNamedAttr("numerical_policy", b.getStringAttr("unchanged")),
       b.getNamedAttr("effective_features", b.getArrayAttr({})),
       b.getNamedAttr("physical_lifetimes_verified", b.getBoolAttr(false))});
}

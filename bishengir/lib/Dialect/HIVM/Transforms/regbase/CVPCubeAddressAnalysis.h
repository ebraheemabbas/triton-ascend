//===- CVPCubeAddressAnalysis.h - Local Q-cache spatial checks -*- C++ -*-===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef BISHENGIR_HIVM_CVP_CUBE_ADDRESS_ANALYSIS_H
#define BISHENGIR_HIVM_CVP_CUBE_ADDRESS_ANALYSIS_H
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/MathExtras.h"
#include <optional>
#include <utility>
namespace mlir::hivm::cvp_detail {
// Spatial bounds for this pass's existing closed set of Cube L1 pointers.
// No lifetime inference, allocation, GSS modification or completion proof.
struct CubeStorageSpan {
  int64_t offsetInBits = 0;
  int64_t extentInBits = 0;
  std::optional<int64_t> endInBits() const {
    if (offsetInBits < 0 || extentInBits <= 0)
      return std::nullopt;
    return llvm::checkedAdd(offsetInBits, extentInBits);
  }
};
inline std::optional<int64_t> scalarWidth(MemRefType type) {
  if (!type || !type.getElementType().isIntOrFloat())
    return std::nullopt;
  int64_t width = type.getElementType().getIntOrFloatBitWidth();
  if (width < 8 || width > 64 || !llvm::isPowerOf2_64(width))
    return std::nullopt;
  return width;
}

inline bool checkedStrides(MemRefType type, SmallVectorImpl<int64_t> &strides,
                           int64_t &offset) {
  if (!type.hasStaticShape())
    return false;
  if (auto layout = dyn_cast<StridedLayoutAttr>(type.getLayout())) {
    strides.assign(layout.getStrides().begin(), layout.getStrides().end());
    offset = layout.getOffset();
    return strides.size() == static_cast<size_t>(type.getRank());
  }
  if (!type.getLayout().isIdentity())
    return false;
  // Do not delegate identity-layout products to affine-expression building:
  // an inner dimension product can overflow before our final span check.
  strides.resize(type.getRank());
  offset = 0;
  int64_t stride = 1;
  for (int64_t i = type.getRank(); i-- > 0;) {
    if (type.getDimSize(i) <= 0)
      return false;
    strides[i] = stride;
    if (i != 0) {
      auto next = llvm::checkedMul(stride, type.getDimSize(i));
      if (!next)
        return false;
      stride = *next;
    }
  }
  return true;
}

inline std::optional<CubeStorageSpan>
getStaticCubeStorageSpan(MemRefType type) {
  auto width = scalarWidth(type);
  if (!width || !type.hasStaticShape())
    return std::nullopt;
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!checkedStrides(type, strides, offset) || offset < 0)
    return std::nullopt;
  int64_t last = 0;
  for (auto [dimension, stride] : llvm::zip(type.getShape(), strides)) {
    if (dimension <= 0 || stride <= 0)
      return std::nullopt;
    auto distance = llvm::checkedMul(dimension - 1, stride);
    auto next = distance ? llvm::checkedAdd(last, *distance) : std::nullopt;
    if (!next)
      return std::nullopt;
    last = *next;
  }
  auto count = llvm::checkedAdd(last, int64_t{1});
  auto extent = count ? llvm::checkedMul(*count, *width) : std::nullopt;
  auto offsetBits = llvm::checkedMul(offset, *width);
  if (!extent || !offsetBits)
    return std::nullopt;
  CubeStorageSpan result{*offsetBits, *extent};
  return result.endInBits() ? std::optional<CubeStorageSpan>(result)
                            : std::nullopt;
}

class AddressAlternativeResolver {
public:
  std::optional<SmallVector<int64_t>> resolve(Value value) {
    if (!value || !value.getType().isSignlessInteger(64))
      return std::nullopt;
    if (auto it = cache.find(value); it != cache.end())
      return it->second;
    if (remainingNodes == 0)
      return std::nullopt;
    --remainingNodes;
    if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
      auto attr = dyn_cast<IntegerAttr>(constant.getValue());
      if (!attr || attr.getInt() < 0)
        return std::nullopt;
      SmallVector<int64_t> result{attr.getInt()};
      cache.try_emplace(value, result);
      return result;
    }

    auto *op = value.getDefiningOp();
    if (!op)
      return std::nullopt;
    Value left, right;
    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      left = select.getTrueValue();
      right = select.getFalseValue();
    } else if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op)) {
      left = op->getOperand(0);
      right = op->getOperand(1);
    } else {
      return std::nullopt;
    }
    auto lhs = resolve(left), rhs = resolve(right);
    if (!lhs || !rhs)
      return std::nullopt;
    // Integer DenseSet reserves sentinel values (including INT64_MAX for
    // int64_t on this target). Every nonnegative i64 is a legitimate value at
    // this layer, including values a later byte-to-bit conversion must reject.
    // A bounded linear set cannot silently discard one of those alternatives.
    SmallVector<int64_t> values;
    auto insert = [&](int64_t candidate) {
      if (!llvm::is_contained(values, candidate))
        values.push_back(candidate);
      return values.size() <= maxAlternatives;
    };
    if (isa<arith::SelectOp>(op)) {
      for (int64_t candidate : *lhs)
        if (!insert(candidate))
          return std::nullopt;
      for (int64_t candidate : *rhs)
        if (!insert(candidate))
          return std::nullopt;
    } else {
      // Ignore predicate correlations. The Cartesian product is conservative;
      // an unproved correlation cannot establish disjoint storage.
      for (int64_t a : *lhs) {
        for (int64_t b : *rhs) {
          std::optional<int64_t> result;
          if (isa<arith::AddIOp>(op))
            result = llvm::checkedAdd(a, b);
          else if (isa<arith::SubIOp>(op))
            result = llvm::checkedSub(a, b);
          else
            result = llvm::checkedMul(a, b);
          if (!result || *result < 0)
            return std::nullopt;
          if (!insert(*result))
            return std::nullopt;
        }
      }
    }
    if (values.empty() || values.size() > maxAlternatives)
      return std::nullopt;
    SmallVector<int64_t> result = std::move(values);
    llvm::sort(result);
    cache.try_emplace(value, result);
    return result;
  }

private:
  static constexpr unsigned maxAlternatives = 32;
  unsigned remainingNodes = 64;
  llvm::DenseMap<Value, SmallVector<int64_t>> cache;
};

inline std::optional<SmallVector<int64_t>>
getCubeAddressAlternatives(Value value) {
  return AddressAlternativeResolver().resolve(value);
}
} // namespace mlir::hivm::cvp_detail
#endif

//===- CVPQCache.cpp - Query-owned normal Cube template cache
//--------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "CVPCubeAddressAnalysis.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/Analysis/CVPAttentionPlan.h"
#include "bishengir/Dialect/HIVM/IR/CVPQCacheContract.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>
#include <optional>

namespace mlir {
#define GEN_PASS_DEF_CVPQCACHE
#define GEN_PASS_DEF_VERIFYCVPQCACHE
#define GEN_PASS_DEF_CVPCUBEREADINESS
#define GEN_PASS_DEF_CVPCUBEOVERLAP
#define GEN_PASS_DEF_CVPCUBEDERIVEDDRAIN
#include "bishengir/Dialect/HIVM/Transforms/Passes.h.inc"
namespace hivm {
namespace {

std::optional<int64_t> integer(Value value) {
  return getConstantIntValue(value);
}

bool isCubeEntry(func::FuncOp function) {
  return function->hasAttr("hacc.entry") &&
         !function->hasAttr("hivm.backup_function") &&
         function->getAttr("hivm.func_core_type") ==
             TFuncCoreTypeAttr::get(function.getContext(), TFuncCoreType::AIC);
}

bool isShape(MmadL1Op op, int64_t m, int64_t k, int64_t n) {
  return integer(op.getRealM()) == m && integer(op.getRealK()) == k &&
         integer(op.getRealN()) == n;
}

// LowerMultiBufferCounter materializes a private scalar ordinal at function
// entry, with one unconditional loop-body load and increment/store. Verify its
// complete storage-use graph before admitting these memory-effecting scalar
// operations alongside the template-managed L0A/L0B reservation.
bool privateScalarCounter(memref::AllocaOp allocation, func::FuncOp function) {
  if (!allocation || allocation->getParentOp() != function.getOperation() ||
      allocation.getResult().getType() !=
          MemRefType::get({1}, IntegerType::get(function.getContext(), 64)))
    return false;
  Block *entry = &function.getBody().front();
  if (allocation->getBlock() != entry)
    return false;

  memref::LoadOp load;
  SmallVector<memref::StoreOp> stores;
  Value storage = allocation.getResult();
  for (Operation *user : storage.getUsers()) {
    if (auto read = dyn_cast<memref::LoadOp>(user)) {
      if (load || read.getMemRef() != storage ||
          read.getIndices().size() != 1 || integer(read.getIndices()[0]) != 0)
        return false;
      load = read;
    } else if (auto write = dyn_cast<memref::StoreOp>(user)) {
      if (write.getMemRef() != storage || write.getIndices().size() != 1 ||
          integer(write.getIndices()[0]) != 0)
        return false;
      stores.push_back(write);
    } else {
      // Aliases, casts, calls and escaping descriptors need their own proof.
      return false;
    }
  }
  if (!load || stores.size() != 2 ||
      load->getParentOfType<func::FuncOp>() != function)
    return false;
  auto loop = dyn_cast<scf::ForOp>(load->getParentOp());
  if (!loop || load->getBlock() != loop.getBody())
    return false;

  memref::StoreOp initialize, update;
  for (memref::StoreOp store : stores) {
    if (store->getBlock() == entry) {
      if (initialize || integer(store.getValueToStore()) != 0)
        return false;
      initialize = store;
    } else if (store->getBlock() == load->getBlock()) {
      if (update)
        return false;
      update = store;
    } else {
      return false;
    }
  }
  if (!initialize || !update || !allocation->isBeforeInBlock(initialize) ||
      !load->isBeforeInBlock(update))
    return false;
  auto increment = update.getValueToStore().getDefiningOp<arith::AddIOp>();
  if (!increment || increment->getBlock() != load->getBlock() ||
      !((increment.getLhs() == load.getResult() &&
         integer(increment.getRhs()) == 1) ||
        (increment.getRhs() == load.getResult() &&
         integer(increment.getLhs()) == 1)))
    return false;

  Operation *entryChild = loop;
  while (entryChild && entryChild->getBlock() != entry)
    entryChild = entryChild->getParentOp();
  return entryChild && initialize->isBeforeInBlock(entryChild);
}

bool noOtherTemplateConsumers(func::FuncOp function) {
  auto directL0 = [](Type type) {
    auto memory = dyn_cast<MemRefType>(type);
    auto space =
        memory ? dyn_cast_or_null<AddressSpaceAttr>(memory.getMemorySpace())
               : AddressSpaceAttr();
    return space && (space.getAddressSpace() == AddressSpace::L0A ||
                     space.getAddressSpace() == AddressSpace::L0B);
  };
  bool valid = !llvm::any_of(function.getArgumentTypes(), directL0);
  function.walk([&](Operation *op) {
    if (!valid || op == function.getOperation())
      return;
    if (llvm::any_of(op->getOperandTypes(), directL0) ||
        llvm::any_of(op->getResultTypes(), directL0) || isa<func::CallOp>(op)) {
      valid = false;
      return;
    }
    if (auto allocation = dyn_cast<memref::AllocaOp>(op)) {
      valid = privateScalarCounter(allocation, function);
      return;
    }
    if (auto read = dyn_cast<memref::LoadOp>(op)) {
      valid = privateScalarCounter(
          read.getMemRef().getDefiningOp<memref::AllocaOp>(), function);
      return;
    }
    if (auto write = dyn_cast<memref::StoreOp>(op)) {
      valid = privateScalarCounter(
          write.getMemRef().getDefiningOp<memref::AllocaOp>(), function);
      return;
    }
    if (isa<MmadL1Op, ND2NZOp, FixpipeOp, CopyOp, LoadOp, StoreOp,
            PointerCastOp, GetBlockIdxOp, GetSubBlockIdxOp, SetCtrlOp,
            SetFlagOp, WaitFlagOp, PipeBarrierOp, SyncBlockOp, SyncBlockSetOp,
            SyncBlockWaitOp, AnchorOp, annotation::MarkOp, scope::ScopeOp,
            scope::ReturnOp, scf::ForOp, scf::IfOp, scf::WhileOp, scf::YieldOp,
            scf::ConditionOp, func::ReturnOp, memref::AllocOp>(op))
      return;
    if (op->getNumRegions() || !isMemoryEffectFree(op))
      valid = false;
  });
  return valid;
}

// Return the numerator of (numerator % divisor == result), independent of
// operand spelling and the order of the equality operands.
Value remainderEquality(Value condition, int64_t divisor, int64_t result) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::eq)
    return {};
  for (unsigned side : {0u, 1u}) {
    auto remainder = compare->getOperand(side).getDefiningOp<arith::RemSIOp>();
    if (remainder && integer(remainder.getRhs()) == divisor &&
        integer(compare->getOperand(1 - side)) == result)
      return remainder.getLhs();
  }
  return {};
}

bool queryAllocation(Value value, unsigned bank) {
  auto allocation = value.getDefiningOp<memref::AllocOp>();
  auto storage =
      allocation
          ? allocation->getAttrOfType<DictionaryAttr>("hivm.cvp_query.storage")
          : DictionaryAttr();
  auto role = storage ? storage.getAs<StringAttr>("role") : StringAttr();
  auto clock = storage ? storage.getAs<StringAttr>("clock") : StringAttr();
  auto index = storage ? storage.getAs<IntegerAttr>("bank") : IntegerAttr();
  auto slots =
      storage ? storage.getAs<IntegerAttr>("required_slots") : IntegerAttr();
  if (!allocation || !allocation.getDynamicSizes().empty() || !role ||
      role.getValue() != "Q" || !clock || clock.getValue() != "query" ||
      !index || index.getInt() != bank || !slots || slots.getInt() != 2)
    return false;
  for (Operation *user : value.getUsers())
    if (auto mark = dyn_cast<annotation::MarkOp>(user))
      if (mark->hasAttr("hivm.multi_buffer"))
        return false;
  return true;
}

bool onlyQUses(Value root, arith::SelectOp select) {
  return llvm::all_of(root.getUsers(), [&](Operation *op) {
    return op == select || isa<annotation::MarkOp>(op);
  });
}

struct Pair {
  func::FuncOp function;
  scf::ForOp loop;
  MmadL1Op qk, pv;
  arith::SelectOp q;
  ND2NZOp load;
  Value first, generation, odd, key;
  int64_t keys = 0;
};

FailureOr<Pair> identify(func::FuncOp function, bool lowered,
                         std::string &reason) {
  auto reject = [&](StringRef message) -> FailureOr<Pair> {
    reason = message.str();
    return failure();
  };
  if (!isCubeEntry(function) || !function->hasAttr("hivm.cvp_query.stream") ||
      !noOtherTemplateConsumers(function))
    return reject(
        "requires a closed AIC query stream with no other L0A/L0B consumer");
  Pair pair;
  pair.function = function;
  SmallVector<MmadL1Op> matrices;
  function.walk([&](MmadL1Op op) { matrices.push_back(op); });
  if (matrices.size() != 2)
    return reject("requires exactly one QK/PV pair");
  for (MmadL1Op op : matrices) {
    if (isShape(op, 256, 64, 128))
      pair.qk = op;
    if (isShape(op, 128, 256, 64))
      pair.pv = op;
  }
  if (!pair.qk || !pair.pv)
    return reject("unsupported QK/PV matrix geometry");
  pair.loop = pair.qk->getParentOfType<scf::ForOp>();
  if (!pair.loop || pair.pv->getParentOfType<scf::ForOp>() != pair.loop ||
      !pair.loop.getInductionVar().getType().isInteger(64) ||
      integer(pair.loop.getLowerBound()) != 0 ||
      integer(pair.loop.getStep()) != 256)
    return reject("requires the shared continuous i64 key clock");
  pair.q = pair.qk.getB().getDefiningOp<arith::SelectOp>();
  if (!pair.q)
    return reject("Q must select the two query-owned L1 banks");
  pair.odd = pair.q.getCondition();
  pair.generation = remainderEquality(pair.odd, 2, 1);
  auto ordinal = pair.generation
                     ? pair.generation.getDefiningOp<arith::DivSIOp>()
                     : arith::DivSIOp();
  auto keys = ordinal ? integer(ordinal.getRhs()) : std::optional<int64_t>();
  if (!ordinal || !keys || *keys < 512 || *keys % 256 ||
      ordinal.getLhs() != pair.loop.getInductionVar())
    return reject("unproved query generation or bank clock");
  pair.key = ordinal.getLhs();
  pair.keys = *keys;
  for (OpOperand &use : pair.q.getResult().getUses()) {
    Operation *user = use.getOwner();
    if (user == pair.qk && use.get() == pair.qk.getB())
      continue;
    auto load = dyn_cast<ND2NZOp>(user);
    if (!load || load.getDst() != pair.q.getResult() || pair.load)
      return reject("Q has an extra writer or consumer");
    pair.load = load;
  }
  auto initialize =
      pair.load ? dyn_cast<scf::IfOp>(pair.load->getParentOp()) : scf::IfOp();
  if (!initialize || pair.load->getBlock() != initialize.thenBlock() ||
      initialize.getNumResults())
    return reject("Q initialization must be guarded by its first local key");
  pair.first = initialize.getCondition();
  if (remainderEquality(pair.first, pair.keys, 0) != pair.key ||
      initialize->getBlock() != pair.qk->getBlock() ||
      !initialize->isBeforeInBlock(pair.qk))
    return reject("Q initialization does not match the query-local first key");
  if (lowered && (!onlyQUses(pair.q.getFalseValue(), pair.q) ||
                  !onlyQUses(pair.q.getTrueValue(), pair.q)))
    return reject("final Q bank root has an unowned reader, writer or alias");
  if (!lowered) {
    auto verified =
        function->getAttrOfType<ArrayAttr>(kCVPAttentionVerifiedAttr);
    auto geometry = verified && verified.size() == 1
                        ? dyn_cast<DictionaryAttr>(verified[0])
                        : DictionaryAttr();
    auto geometryValue = [&](StringRef name, int64_t expected) {
      auto value = geometry ? geometry.getAs<IntegerAttr>(name) : IntegerAttr();
      return value && value.getInt() == expected;
    };
    auto layout =
        geometry ? geometry.getAs<StringAttr>("score_layout") : StringAttr();
    if (!geometryValue("block_m", 128) || !geometryValue("block_n", 256) ||
        !geometryValue("head_dim", 64) ||
        !geometryValue("key_count", pair.keys) || !layout ||
        layout.getValue() != "key-major")
      return reject(
          "requires the verified HD64/BN256 CVPipeline attention plan");
    auto qScope = dyn_cast<scope::ScopeOp>(pair.qk->getParentOp());
    auto pScope = dyn_cast<scope::ScopeOp>(pair.pv->getParentOp());
    auto qPhase = qScope
                      ? qScope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                      : IntegerAttr();
    auto pPhase = pScope
                      ? pScope->getAttrOfType<IntegerAttr>(PreloadNumAttr::name)
                      : IntegerAttr();
    auto qDepth =
        qScope ? qScope->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
               : IntegerAttr();
    auto pDepth =
        pScope ? pScope->getAttrOfType<IntegerAttr>(MaxPreloadNumAttr::name)
               : IntegerAttr();
    if (!qPhase || qPhase.getInt() != 3 || !pPhase || pPhase.getInt() != 1 ||
        !qDepth || qDepth.getInt() != 4 || !pDepth || pDepth.getInt() != 4 ||
        qScope->getParentOp() != pair.loop ||
        pScope->getParentOp() != pair.loop ||
        !qScope->isBeforeInBlock(pScope) ||
        !queryAllocation(pair.q.getFalseValue(), 0) ||
        !queryAllocation(pair.q.getTrueValue(), 1) ||
        !onlyQUses(pair.q.getFalseValue(), pair.q) ||
        !onlyQUses(pair.q.getTrueValue(), pair.q))
      return reject("unproved Skew scope or persistent Q allocation");
    auto stream =
        pair.loop->getAttrOfType<DictionaryAttr>("hivm.cvp_query.stream");
    auto count =
        stream ? stream.getAs<IntegerAttr>("key_count") : IntegerAttr();
    auto step = stream ? stream.getAs<IntegerAttr>("key_step") : IntegerAttr();
    if (!count || count.getInt() != pair.keys || !step || step.getInt() != 256)
      return reject("query stream domain disagrees with cache generation");
  }
  return pair;
}

bool matchesUpper(Value condition, Value key, Value &end) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
      compare.getLhs() != key)
    return false;
  end = compare.getRhs();
  return true;
}

bool matchesLower(Value condition, Value key) {
  auto compare = condition.getDefiningOp<arith::CmpIOp>();
  return compare && compare.getPredicate() == arith::CmpIPredicate::sge &&
         compare.getLhs() == key && integer(compare.getRhs()) == 0;
}

bool validityCondition(Value condition, Value key, Value &end,
                       bool nonnegative) {
  if (nonnegative && matchesUpper(condition, key, end))
    return true;
  auto both = condition.getDefiningOp<arith::AndIOp>();
  return both && ((matchesLower(both.getLhs(), key) &&
                   matchesUpper(both.getRhs(), key, end)) ||
                  (matchesLower(both.getRhs(), key) &&
                   matchesUpper(both.getLhs(), key, end)));
}

bool validity(scf::IfOp branch, Value key, Value &end, bool nonnegative) {
  return branch &&
         validityCondition(branch.getCondition(), key, end, nonnegative);
}

bool verifyDerivedCubeClock(Pair &pair, Value work);

bool verifyLoweredClock(Pair &pair) {
  auto qActive = dyn_cast<scf::IfOp>(pair.qk->getParentOp());
  auto pActive = dyn_cast<scf::IfOp>(pair.pv->getParentOp());
  if (!qActive || !pActive || qActive->getParentOp() != pair.loop ||
      pActive->getParentOp() != pair.loop ||
      pair.qk->getBlock() != qActive.thenBlock() ||
      pair.pv->getBlock() != pActive.thenBlock() ||
      !qActive->isBeforeInBlock(pActive))
    return false;
  Value work;
  if (!validity(qActive, pair.key, work, true))
    return false;
  auto wholeQueries = work.getDefiningOp<arith::MulIOp>();
  if (!wholeQueries || (integer(wholeQueries.getLhs()) != pair.keys &&
                        integer(wholeQueries.getRhs()) != pair.keys))
    return false;
  Value pvKey;
  for (auto subtract : pair.loop.getBody()->getOps<arith::SubIOp>())
    if (subtract.getLhs() == pair.key && integer(subtract.getRhs()) == 512)
      pvKey = subtract;
  if (!pvKey)
    return false;
  Value pvEnd;
  if (!validity(pActive, pvKey, pvEnd, false) || pvEnd != work)
    return false;
  if (pair.function->hasAttr(kCVPCubeDerivedDrain))
    return verifyDerivedCubeClock(pair, work);
  auto drain = pair.loop.getUpperBound().getDefiningOp<arith::AddIOp>();
  if (!drain)
    return false;
  for (unsigned side : {0u, 1u}) {
    auto extra = integer(drain->getOperand(1 - side));
    if (drain->getOperand(side) == work && extra && *extra >= 512 &&
        *extra % 256 == 0)
      return true;
  }
  return false;
}

bool physicalQBanks(Pair &pair) {
  std::optional<int64_t> bases[2];
  unsigned slot = 0;
  for (Value value : {pair.q.getFalseValue(), pair.q.getTrueValue()}) {
    auto pointer = value.getDefiningOp<PointerCastOp>();
    if (!pointer || pointer->getNumOperands() != 1)
      return false;
    bases[slot++] = integer(pointer->getOperand(0));
  }
  if (!bases[0] || !bases[1] || *bases[0] < 0 || *bases[1] < 0 ||
      *bases[0] % 32 || *bases[1] % 32)
    return false;
  const uint64_t bytes = 128 * 64 * 2;
  bishengir::cvp::ByteRange a{uint64_t(*bases[0]), uint64_t(*bases[0]) + bytes};
  bishengir::cvp::ByteRange b{uint64_t(*bases[1]), uint64_t(*bases[1]) + bytes};
  auto target =
      hacc::utils::getNPUTargetSpec(utils::getTopLevelModuleOp(pair.function));
  if (!target)
    return false;
  auto entry = target->getSpecForIdentifierEnum(hacc::DeviceSpec::L1_SIZE);
  auto bits = entry ? dyn_cast<IntegerAttr>(entry.getValue()) : IntegerAttr();
  if (!bits || bits.getInt() <= 0 || bits.getInt() % 8 || a.overlaps(b))
    return false;
  const uint64_t capacity = bits.getInt() / 8;
  if (a.end > capacity || b.end > capacity)
    return false;

  // Placement removes allocation identity. A different SSA root may still
  // name either persistent Q reservation, so checking only a versus b is not
  // enough. Admit the actual final AIC form: bounded physical pointer roots
  // and selections between them. An unproved view, block argument or local
  // allocation must not silently establish disjointness.
  auto isL1 = [](Type type) {
    auto memory = dyn_cast<BaseMemRefType>(type);
    auto space =
        memory ? dyn_cast_or_null<AddressSpaceAttr>(memory.getMemorySpace())
               : AddressSpaceAttr();
    return space && space.getAddressSpace() == AddressSpace::L1;
  };
  auto unknownL1 = [&](Value value) {
    if (!isL1(value.getType()))
      return false;
    Operation *definition = value.getDefiningOp();
    return !definition || !isa<PointerCastOp, arith::SelectOp>(definition);
  };
  bool valid = !llvm::any_of(pair.function.getArgumentTypes(), isL1);
  pair.function.walk([&](Operation *operation) {
    if (!valid)
      return;
    if (llvm::any_of(operation->getOperands(), unknownL1) ||
        llvm::any_of(operation->getResults(), unknownL1)) {
      valid = false;
      return;
    }
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        if (llvm::any_of(block.getArgumentTypes(), isL1)) {
          valid = false;
          return;
        }
    auto pointer = dyn_cast<PointerCastOp>(operation);
    if (!pointer || !isL1(pointer->getResult(0).getType()))
      return;
    Value root = pointer->getResult(0);
    if (root == pair.q.getFalseValue() || root == pair.q.getTrueValue())
      return;
    auto span = cvp_detail::getStaticCubeStorageSpan(
        dyn_cast<MemRefType>(root.getType()));
    auto end = span ? span->endInBits() : std::nullopt;
    if (!end || span->offsetInBits % 8 || *end % 8 ||
        static_cast<uint64_t>(*end / 8) > capacity ||
        !pointer->getNumOperands()) {
      valid = false;
      return;
    }
    const uint64_t offsetBytes = span->offsetInBits / 8;
    const uint64_t endBytes = *end / 8;
    for (Value address : pointer->getOperands()) {
      auto alternatives = cvp_detail::getCubeAddressAlternatives(address);
      if (!alternatives || alternatives->empty()) {
        valid = false;
        return;
      }
      for (int64_t base : *alternatives) {
        if (base < 0 || static_cast<uint64_t>(base) > capacity - endBytes) {
          valid = false;
          return;
        }
        bishengir::cvp::ByteRange other{static_cast<uint64_t>(base) +
                                            offsetBytes,
                                        static_cast<uint64_t>(base) + endBytes};
        if (other.overlaps(a) || other.overlaps(b)) {
          valid = false;
          return;
        }
      }
    }
  });
  // This closes the Cube spatial root contract only. The query schedule and
  // existing synchronization must still preserve producer/consumer ordering.
  // A spatial check alone does not authorize removing a completion wait.
  return valid;
}

struct CubeReadyMatch {
  SetFlagOp pvReady, pvResult;
  WaitFlagOp pvResultWait;
  Value pvCredit;
};

LogicalResult matchCubeReadiness(Pair &pair, bool banked, CubeReadyMatch &match,
                                 std::string &reason);

struct CubeOverlapMatch {
  SetFlagOp intraSet;
  WaitFlagOp intraWait;
  scf::IfOp qActive, pActive;
};

LogicalResult matchCubeOverlap(Pair &pair, bool assigned,
                               CubeOverlapMatch &match, std::string &reason);

LogicalResult verifyPair(func::FuncOp function, StringRef mode, bool lowered,
                         std::string &reason) {
  auto plan = identify(function, lowered, reason);
  if (failed(plan))
    return failure();
  Pair &pair = *plan;
  auto resources = readCVPQCacheResources(function);
  bool valid = succeeded(resources);
  std::string diagnostics;
  {
    ScopedDiagnosticHandler capture(
        function.getContext(), [&](Diagnostic &diagnostic) {
          llvm::raw_string_ostream output(diagnostics);
          diagnostic.print(output);
          return success();
        });
    valid &= succeeded(verifyCVPQCacheMmad(pair.qk));
    valid &= succeeded(verifyCVPQCacheMmad(pair.pv));
  }
  if (!valid) {
    reason = "invalid target/template resource contract: " + diagnostics;
    return failure();
  }
  auto q = pair.qk.getCvpQCacheAttr(), p = pair.pv.getCvpQCacheAttr();
  if (q.getAs<StringAttr>("role").getValue() != "qk" ||
      p.getAs<StringAttr>("role").getValue() != "pv" ||
      q.getAs<StringAttr>("mode").getValue() != mode ||
      p.getAs<StringAttr>("mode").getValue() != mode ||
      pair.qk.getQCacheInit() != pair.first ||
      pair.qk.getQCacheGeneration() != pair.generation) {
    reason = "cache context changed query identity";
    return failure();
  }
  auto offset = pair.qk.getQCacheByteOffset().getDefiningOp<arith::SelectOp>();
  if (!offset || offset.getCondition() != pair.odd ||
      integer(offset.getTrueValue()) !=
          static_cast<int64_t>(resources->q[1].begin) ||
      integer(offset.getFalseValue()) !=
          static_cast<int64_t>(resources->q[0].begin)) {
    reason = "cache range changed query-bank identity";
    return failure();
  }
  if (lowered && (!verifyLoweredClock(pair) || !physicalQBanks(pair))) {
    reason = "unproved final QK/PV clock, drain or physical Q banks";
    return failure();
  }
  if (lowered && (function->hasAttr(kCVPCubeReadiness) ||
                  pair.qk->hasAttr(kCVPCubeBankedReady) ||
                  pair.pv->hasAttr(kCVPCubeBankedReady))) {
    CubeReadyMatch ready;
    if (failed(matchCubeReadiness(pair, true, ready, reason)))
      return failure();
  }
  if (lowered && function->hasAttr(kCVPCubeOverlap)) {
    CubeOverlapMatch overlap;
    if (failed(matchCubeOverlap(pair, true, overlap, reason)))
      return failure();
  }
  return success();
}

// Readiness is selected only after the final scheduled pair is known. The
// helper owns MTE1->M events 0/1 and shares one alternating M->MTE1 counter.
// Caller events are derived from the existing complete L1 credit operand;
// no L0C fence, active predicate, load, matrix operation or loop bound changes.
LogicalResult matchCubeReadiness(Pair &pair, bool banked, CubeReadyMatch &match,
                                 std::string &reason) {
  auto reject = [&](StringRef message) -> LogicalResult {
    reason = message.str();
    return failure();
  };
  auto channel = [](Operation *op, PIPE source, PIPE target) {
    if (!op || !isa<SetFlagOp, WaitFlagOp>(op))
      return false;
    auto from = op->getAttrOfType<PipeAttr>("set_pipe");
    auto to = op->getAttrOfType<PipeAttr>("wait_pipe");
    return from && to && from.getPipe() == source && to.getPipe() == target;
  };
  auto staticID = [](Operation *op) -> std::optional<int64_t> {
    if (!op || op->getNumOperands())
      return std::nullopt;
    auto value = op->getAttrOfType<EventAttr>("static_event_id");
    return value
               ? std::optional<int64_t>(static_cast<int64_t>(value.getEvent()))
               : std::nullopt;
  };
  auto sameID = [&](Operation *op, Value value) {
    if (auto constant = integer(value))
      if (staticID(op) == constant)
        return true;
    return op && !op->hasAttr("static_event_id") && op->getNumOperands() == 1 &&
           op->getOperand(0) == value;
  };
  auto preservedCrossCoreHandoff = [](Operation *op) {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op) || op->getNumResults() ||
        op->getNumRegions())
      return false;
    auto core = op->getAttrOfType<TCoreTypeAttr>("tcore_type");
    auto from = op->getAttrOfType<PipeAttr>("tpipe");
    auto to = op->getAttrOfType<PipeAttr>("pipe");
    if (!core || core.getTcoretype() != TCoreType::CUBE || !from || !to)
      return false;
    if (isa<SyncBlockWaitOp>(op))
      return (from.getPipe() == PIPE::PIPE_MTE3 &&
              to.getPipe() == PIPE::PIPE_MTE1) ||
             (from.getPipe() == PIPE::PIPE_V && to.getPipe() == PIPE::PIPE_FIX);
    return (from.getPipe() == PIPE::PIPE_MTE1 &&
            to.getPipe() == PIPE::PIPE_MTE3) ||
           (from.getPipe() == PIPE::PIPE_FIX && to.getPipe() == PIPE::PIPE_V);
  };
  auto effectNeighbor = [&](Operation *op, bool previous) {
    Operation *next = previous ? op->getPrevNode() : op->getNextNode();
    // Stock GSS can place a cross-core handoff between a local ready event
    // and its reader, or between the matrix-result set and FIX wait. Keep
    // those handoffs at their exact positions while locating the local pair.
    // The complete event ownership, bank and entry/return checks below still
    // apply; arbitrary effects and unknown handoff directions remain barriers.
    while (
        next && !next->getNumRegions() &&
        (preservedCrossCoreHandoff(next) ||
         (!isa<SetFlagOp, WaitFlagOp, SyncBlockSetOp, SyncBlockWaitOp>(next) &&
          isMemoryEffectFree(next))))
      next = previous ? next->getPrevNode() : next->getNextNode();
    return next;
  };
  auto policy = pair.function->getAttrOfType<StringAttr>(kCVPCubeReadiness);
  if (banked) {
    if (!policy || policy.getValue() != "banked-v1" ||
        !pair.qk->getAttrOfType<UnitAttr>(kCVPCubeBankedReady) ||
        !pair.pv->getAttrOfType<UnitAttr>(kCVPCubeBankedReady))
      return reject("incomplete banked readiness pair or caller policy");
  } else if (pair.function->hasAttr(kCVPCubeReadiness) ||
             pair.qk->hasAttr(kCVPCubeBankedReady) ||
             pair.pv->hasAttr(kCVPCubeBankedReady)) {
    return reject("partially assigned Cube readiness policy");
  }

  auto q = pair.qk.getSyncRelatedArgs(), p = pair.pv.getSyncRelatedArgs();
  if (q.size() != 7 || p.size() != 7 || integer(q[0]) != 0 ||
      integer(q[1]) != -1 || integer(q[2]) != -1 || integer(q[3]) != 0 ||
      integer(p[0]) != -1 || integer(p[2]) != -1)
    return reject("unsupported complete QK/PV L1 credit signature");
  for (unsigned index = 4; index < 7; ++index)
    if (integer(q[index]) != -1 || integer(p[index]) != -1)
      return reject("helper backward events or K-loop protocol not reserved");
  auto bankCredit = p[3].getDefiningOp<arith::SelectOp>();
  if (!bankCredit || integer(bankCredit.getTrueValue()) != 1 ||
      integer(bankCredit.getFalseValue()) != 2)
    return reject("PV readiness requires the existing two-bank L1 credit");
  if ((!banked && integer(p[1]) != 0) || (banked && p[1] != p[3]))
    return reject("PV producer and helper readiness do not share its credit");

  auto qReady = dyn_cast_or_null<SetFlagOp>(effectNeighbor(pair.qk, true));
  auto pReady = dyn_cast_or_null<SetFlagOp>(effectNeighbor(pair.pv, true));
  if (!qReady || !pReady ||
      !channel(qReady, PIPE::PIPE_MTE2, PIPE::PIPE_MTE1) ||
      !channel(pReady, PIPE::PIPE_MTE2, PIPE::PIPE_MTE1) ||
      staticID(qReady) != 0 ||
      (banked ? !sameID(pReady, p[3]) : staticID(pReady) != 0))
    return reject("unpaired QK/PV MTE2 readiness producer");
  auto vCopy = dyn_cast_or_null<ND2NZOp>(effectNeighbor(pReady, true));
  if (!vCopy || vCopy.getDpsInits().size() != 1 ||
      vCopy.getDpsInits()[0] != pair.pv.getB())
    return reject("PV ready producer is not its actual V copy");
  auto vFree = dyn_cast_or_null<WaitFlagOp>(effectNeighbor(vCopy, true));
  if (!vFree || !channel(vFree, PIPE::PIPE_MTE1, PIPE::PIPE_MTE2) ||
      !sameID(vFree, p[3]))
    return reject("PV copy does not acquire the same L1 bank credit");

  auto qResult = dyn_cast_or_null<SetFlagOp>(effectNeighbor(pair.qk, false));
  auto pResult = dyn_cast_or_null<SetFlagOp>(effectNeighbor(pair.pv, false));
  auto qWait =
      qResult ? dyn_cast_or_null<WaitFlagOp>(effectNeighbor(qResult, false))
              : WaitFlagOp();
  auto pWait =
      pResult ? dyn_cast_or_null<WaitFlagOp>(effectNeighbor(pResult, false))
              : WaitFlagOp();
  if (!qResult || !pResult || !qWait || !pWait ||
      !channel(qResult, PIPE::PIPE_M, PIPE::PIPE_FIX) ||
      !channel(pResult, PIPE::PIPE_M, PIPE::PIPE_FIX) ||
      !channel(qWait, PIPE::PIPE_M, PIPE::PIPE_FIX) ||
      !channel(pWait, PIPE::PIPE_M, PIPE::PIPE_FIX) || staticID(qResult) != 0 ||
      staticID(qWait) != 0 || staticID(pResult) != (banked ? 1 : 0) ||
      staticID(pWait) != (banked ? 1 : 0))
    return reject("unpaired matrix-to-FIX readiness");
  auto qFix = dyn_cast_or_null<FixpipeOp>(effectNeighbor(qWait, false));
  auto pFix = dyn_cast_or_null<FixpipeOp>(effectNeighbor(pWait, false));
  if (!qFix || !pFix || qFix.getDpsInputs()[0] != pair.qk.getC() ||
      pFix.getDpsInputs()[0] != pair.pv.getC())
    return reject("matrix readiness is detached from its FIX reader");

  Operation *entryLoop = pair.loop;
  Block *entry = &pair.function.getBody().front();
  while (entryLoop && entryLoop->getBlock() != entry)
    entryLoop = entryLoop->getParentOp();
  if (!entryLoop)
    return reject("no function-entry owner for template event credits");
  unsigned initialize[2] = {0, 0}, drain[2] = {0, 0};
  bool reserved = true;
  pair.function.walk([&](Operation *op) {
    if (!reserved || !isa<SetFlagOp, WaitFlagOp>(op))
      return;
    if (channel(op, PIPE::PIPE_MTE1, PIPE::PIPE_M)) {
      // All ready events in this direction belong inside the paired helper.
      reserved = false;
    } else if (channel(op, PIPE::PIPE_M, PIPE::PIPE_MTE1)) {
      auto id = staticID(op);
      if (!id || *id < 0 || *id >= 2 || op->getBlock() != entry) {
        reserved = false;
      } else if (isa<SetFlagOp>(op) && op->isBeforeInBlock(entryLoop)) {
        ++initialize[*id];
      } else if (isa<WaitFlagOp>(op) && entryLoop->isBeforeInBlock(op)) {
        ++drain[*id];
      } else {
        reserved = false;
      }
    } else if (channel(op, PIPE::PIPE_MTE2, PIPE::PIPE_MTE1)) {
      reserved = op == qReady.getOperation() || op == pReady.getOperation();
    } else if (channel(op, PIPE::PIPE_M, PIPE::PIPE_FIX)) {
      reserved = op == qResult.getOperation() || op == pResult.getOperation() ||
                 op == qWait.getOperation() || op == pWait.getOperation();
    }
  });
  if (!reserved || initialize[0] != 1 || initialize[1] != 1 || drain[0] != 1 ||
      drain[1] != 1)
    return reject("Cube readiness channels conflict or lack complete "
                  "entry/return credits");
  match = {pReady, pResult, pWait, p[3]};
  return success();
}

// The current QK FIX and PV matrix may run together only when both physical
// L0C selections are disjoint. This proof is intentionally closed: casts,
// subviews, escaping descriptors and additional L0C effects are not inferred
// safe from metadata or names. The next-write/return F->M fence remains intact.
LogicalResult matchCubeOverlap(Pair &pair, bool assigned,
                               CubeOverlapMatch &match, std::string &reason) {
  auto reject = [&](StringRef message) -> LogicalResult {
    reason = message.str();
    return failure();
  };
  auto ready = pair.function->getAttrOfType<StringAttr>(kCVPCubeReadiness);
  auto policy = pair.function->getAttrOfType<StringAttr>(kCVPCubeOverlap);
  if (!ready || ready.getValue() != "banked-v1" ||
      !pair.qk->hasAttr(kCVPCubeBankedReady) ||
      !pair.pv->hasAttr(kCVPCubeBankedReady))
    return reject("Cube overlap requires complete banked readiness");
  if (assigned ? (!policy || policy.getValue() != "guarded-v1")
               : pair.function->hasAttr(kCVPCubeOverlap))
    return reject("unsupported or partially assigned Cube overlap policy");

  auto qActive = dyn_cast<scf::IfOp>(pair.qk->getParentOp());
  auto pActive = dyn_cast<scf::IfOp>(pair.pv->getParentOp());
  if (!qActive || !pActive || qActive.getNumResults() ||
      pActive.getNumResults() || !qActive.getElseRegion().empty() ||
      !pActive.getElseRegion().empty() || qActive->getParentOp() != pair.loop ||
      pActive->getParentOp() != pair.loop || !qActive->isBeforeInBlock(pActive))
    return reject("overlap requires the closed ordered QK/PV active regions");
  Value work;
  if (!validity(qActive, pair.key, work, true))
    return reject("overlap QK domain is not the verified work domain");
  Value pvKey;
  for (auto subtract : pair.loop.getBody()->getOps<arith::SubIOp>())
    if (subtract.getLhs() == pair.key && integer(subtract.getRhs()) == 512)
      pvKey = subtract;
  Value pvEnd;
  if (!pvKey || !validity(pActive, pvKey, pvEnd, false) || pvEnd != work)
    return reject("overlap PV domain is not the verified delayed work domain");

  auto qc = pair.qk.getC().getDefiningOp<arith::SelectOp>();
  auto pc = pair.pv.getC().getDefiningOp<arith::SelectOp>();
  if (!qc || !pc || qc.getCondition() != pc.getCondition())
    return reject("QK/PV L0C selections must share the actual bank condition");
  auto target =
      hacc::utils::getNPUTargetSpec(utils::getTopLevelModuleOp(pair.function));
  if (!target)
    return reject("overlap requires the actual target L0C capacity");
  auto spec = target->getSpecForIdentifierEnum(hacc::DeviceSpec::L0C_SIZE);
  auto bits = spec ? dyn_cast<IntegerAttr>(spec.getValue()) : IntegerAttr();
  if (!bits || bits.getInt() <= 0 || bits.getInt() % 8)
    return reject("overlap requires the actual target L0C capacity");
  uint64_t capacity = bits.getInt() / 8;
  llvm::SmallPtrSet<Operation *, 16> allowed;
  auto isL0C = [](Type type) {
    auto memory = dyn_cast<MemRefType>(type);
    auto space =
        memory ? dyn_cast_or_null<AddressSpaceAttr>(memory.getMemorySpace())
               : AddressSpaceAttr();
    return space && space.getAddressSpace() == AddressSpace::L0C;
  };
  auto range = [&](Value value, bishengir::cvp::ByteRange &result) {
    auto pointer = value.getDefiningOp<PointerCastOp>();
    auto type = dyn_cast<MemRefType>(value.getType());
    if (!pointer || pointer->getNumOperands() != 1 || !type || !isL0C(type) ||
        !type.hasStaticShape() || !type.getLayout().isIdentity() ||
        !type.getElementType().isF32() || type.getNumElements() <= 0 ||
        uint64_t(type.getNumElements()) > capacity / 4)
      return false;
    auto offset = integer(pointer->getOperand(0));
    uint64_t bytes = uint64_t(type.getNumElements()) * 4;
    if (!offset || *offset < 0 || *offset % 512 ||
        uint64_t(*offset) > capacity - bytes)
      return false;
    result = {uint64_t(*offset), uint64_t(*offset) + bytes};
    allowed.insert(pointer);
    return true;
  };
  for (unsigned branch : {0u, 1u}) {
    Value q = branch ? qc.getTrueValue() : qc.getFalseValue();
    Value p = branch ? pc.getTrueValue() : pc.getFalseValue();
    bishengir::cvp::ByteRange qRange, pRange;
    if (!range(q, qRange) || !range(p, pRange) || qRange.overlaps(pRange))
      return reject(
          "overlap requires disjoint physical QK/PV L0C ranges in both banks");
    if (!llvm::all_of(q.getUsers(), [&](Operation *op) { return op == qc; }) ||
        !llvm::all_of(p.getUsers(), [&](Operation *op) { return op == pc; }))
      return reject("physical L0C bank has another alias or consumer");
  }
  auto closedResult = [&](arith::SelectOp select, MmadL1Op matrix,
                          scf::IfOp active) {
    FixpipeOp fix;
    unsigned matrixUses = 0;
    for (OpOperand &use : select.getResult().getUses()) {
      Operation *op = use.getOwner();
      if (op == matrix) {
        if (use.get() != matrix.getC())
          return false;
        ++matrixUses;
        continue;
      }
      auto read = dyn_cast<FixpipeOp>(op);
      if (!read || fix || read.getDpsInputs().size() != 1 ||
          read.getDpsInputs()[0] != select.getResult() ||
          read->getParentOp() != active || !matrix->isBeforeInBlock(read))
        return false;
      fix = read;
    }
    if (!fix || matrixUses != 1)
      return false;
    allowed.insert(select);
    allowed.insert(matrix);
    allowed.insert(fix);
    return true;
  };
  if (!closedResult(qc, pair.qk, qActive) ||
      !closedResult(pc, pair.pv, pActive))
    return reject(
        "overlap requires exactly the matrix writer and its last FIX reader");
  bool closed = !llvm::any_of(pair.function.getArgumentTypes(), isL0C);
  pair.function.walk([&](Operation *op) {
    if ((llvm::any_of(op->getOperandTypes(), isL0C) ||
         llvm::any_of(op->getResultTypes(), isL0C)) &&
        !allowed.count(op))
      closed = false;
  });
  if (!closed)
    return reject("unknown physical L0C reader, writer or alias");

  auto nextEffect = [](Operation *op, bool previous) {
    Operation *next = previous ? op->getPrevNode() : op->getNextNode();
    while (next && !isa<SetFlagOp, WaitFlagOp>(next) &&
           !next->getNumRegions() && isMemoryEffectFree(next))
      next = previous ? next->getPrevNode() : next->getNextNode();
    return next;
  };
  auto inactivePV = [&](Value condition) {
    auto invert = condition.getDefiningOp<arith::XOrIOp>();
    if (!invert)
      return false;
    for (unsigned side : {0u, 1u}) {
      Value end;
      // A signed extension of i1 true is -1. Match the Boolean bit pattern
      // explicitly; the signed helper remains appropriate for loop indices.
      auto constant =
          invert->getOperand(side).getDefiningOp<arith::ConstantOp>();
      auto truth =
          constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr();
      if (truth && truth.getType().isInteger(1) && truth.getValue().isOne() &&
          validityCondition(invert->getOperand(1 - side), pvKey, end, false) &&
          end == work)
        return true;
    }
    return false;
  };
  Value guardCondition;
  auto position = [&](Operation *flag) -> Operation * {
    if (!assigned)
      return flag->getBlock() == pair.loop.getBody() ? flag : nullptr;
    auto guard = dyn_cast<scf::IfOp>(flag->getParentOp());
    if (!guard || guard->getBlock() != pair.loop.getBody() ||
        guard.getNumResults() || !guard.getElseRegion().empty() ||
        guard.thenBlock()->getOperations().size() != 2 ||
        &guard.thenBlock()->front() != flag ||
        !inactivePV(guard.getCondition()) ||
        (guardCondition && guardCondition != guard.getCondition()))
      return nullptr;
    guardCondition = guard.getCondition();
    return guard;
  };
  Operation *entryLoop = pair.loop;
  Block *entry = &pair.function.getBody().front();
  while (entryLoop && entryLoop->getBlock() != entry)
    entryLoop = entryLoop->getParentOp();
  if (!entryLoop)
    return reject("missing function entry owner for L0C reuse credits");
  SetFlagOp intraSet, reuseSet, initialSet;
  WaitFlagOp intraWait, reuseWait, returnWait;
  bool fences = true;
  pair.function.walk([&](Operation *op) {
    if (!fences || !isa<SetFlagOp, WaitFlagOp>(op))
      return;
    auto from = op->getAttrOfType<PipeAttr>("set_pipe");
    auto to = op->getAttrOfType<PipeAttr>("wait_pipe");
    if (!from || !to || from.getPipe() != PIPE::PIPE_FIX ||
        to.getPipe() != PIPE::PIPE_M)
      return;
    auto event = op->getAttrOfType<EventAttr>("static_event_id");
    if (op->getNumOperands() || !event) {
      fences = false;
      return;
    }
    if (event.getEvent() == EVENT::EVENT_ID0) {
      Operation *owner = position(op);
      if (!owner) {
        fences = false;
      } else if (auto set = dyn_cast<SetFlagOp>(op)) {
        if (intraSet || nextEffect(qActive, false) != owner)
          fences = false;
        intraSet = set;
      } else {
        if (intraWait || nextEffect(pActive, true) != owner)
          fences = false;
        intraWait = cast<WaitFlagOp>(op);
      }
    } else if (event.getEvent() == EVENT::EVENT_ID1) {
      if (auto set = dyn_cast<SetFlagOp>(op)) {
        if (op->getBlock() == entry && op->isBeforeInBlock(entryLoop)) {
          if (initialSet)
            fences = false;
          initialSet = set;
        } else if (op->getBlock() == pair.loop.getBody() &&
                   nextEffect(pActive, false) == op) {
          if (reuseSet)
            fences = false;
          reuseSet = set;
        } else {
          fences = false;
        }
      } else if (op->getBlock() == entry && entryLoop->isBeforeInBlock(op)) {
        if (returnWait)
          fences = false;
        returnWait = cast<WaitFlagOp>(op);
      } else if (op->getBlock() == pair.loop.getBody() &&
                 nextEffect(qActive, true) == op) {
        if (reuseWait)
          fences = false;
        reuseWait = cast<WaitFlagOp>(op);
      } else {
        fences = false;
      }
    } else {
      fences = false;
    }
  });
  if (!fences || !intraSet || !intraWait || !reuseSet || !reuseWait ||
      !initialSet || !returnWait)
    return reject("L0C overlap lacks the complete intra-iteration, next-write "
                  "and return fences");
  if (!assigned) {
    DominanceInfo dominance(pair.function);
    auto predicate = pActive.getCondition().getDefiningOp<arith::AndIOp>();
    for (Value value : predicate->getOperands())
      for (Value input : value.getDefiningOp()->getOperands())
        if (!dominance.dominates(input, intraSet.getOperation()))
          return reject(
              "PV predicate inputs do not dominate the guarded fence");
  }
  match = {intraSet, intraWait, qActive, pActive};
  return success();
}

struct CubeDrainMatch {
  Value work, oldEnd;
  int64_t step = 0, delay = 0, extra = 0;
  SmallVector<Operation *> cross;
  scf::ForOp suffix;
  bool guarded = false;
};

std::optional<int64_t> extraAfterWork(Value upper, Value work) {
  auto add = upper.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;
  for (unsigned side : {0u, 1u})
    if (add->getOperand(side) == work)
      if (auto value = integer(add->getOperand(1 - side)))
        return value;
  return std::nullopt;
}

// Prove the actual work expression, including signed arithmetic bounds. Tags
// such as query_count/work_end are descriptive and are not used as evidence.
bool boundedWholeQueryWork(Value work, int64_t keys, int64_t extra) {
  if (!work.getType().isInteger(64) || keys <= 0 || extra < 0)
    return false;
  auto multiply = work.getDefiningOp<arith::MulIOp>();
  if (!multiply)
    return false;
  Value queries;
  for (unsigned side : {0u, 1u})
    if (integer(multiply->getOperand(side)) == keys)
      queries = multiply->getOperand(1 - side);
  auto ceiling = queries ? queries.getDefiningOp<arith::CeilDivSIOp>()
                         : arith::CeilDivSIOp();
  auto grid = ceiling ? integer(ceiling.getRhs()) : std::optional<int64_t>();
  auto maximum = ceiling ? ceiling.getLhs().getDefiningOp<arith::MaxSIOp>()
                         : arith::MaxSIOp();
  if (!grid || *grid <= 0 || !maximum)
    return false;
  Value difference;
  for (unsigned side : {0u, 1u})
    if (integer(maximum->getOperand(side)) == 0)
      difference = maximum->getOperand(1 - side);
  auto subtract =
      difference ? difference.getDefiningOp<arith::SubIOp>() : arith::SubIOp();
  auto total = subtract ? integer(subtract.getLhs()) : std::optional<int64_t>();
  auto start = subtract ? subtract.getRhs().getDefiningOp<arith::ExtSIOp>()
                        : arith::ExtSIOp();
  if (!total || *total < 0 || !start || !start.getIn().getType().isInteger(32))
    return false;
  // i32 query_start is in [-2^31, 2^31-1]. Check the widened bound before
  // converting back to i64, including any ceildiv lowering that adds grid-1.
  llvm::APInt limit(128, uint64_t(std::numeric_limits<int64_t>::max()));
  llvm::APInt numerator = llvm::APInt(128, uint64_t(*total)) +
                          llvm::APInt(128, uint64_t(1) << 31) +
                          llvm::APInt(128, uint64_t(*grid - 1));
  llvm::APInt largestWork = numerator.udiv(llvm::APInt(128, uint64_t(*grid))) *
                            llvm::APInt(128, uint64_t(keys));
  return numerator.ule(limit) &&
         largestWork.ule(limit - llvm::APInt(128, uint64_t(extra)));
}

// Compare the actual mixed Cube/Vector scalar domains. Only integer arithmetic
// and the shared mixed-kernel block index are admitted. Function parameters
// match by argument position and outer query IVs match only after their own
// bounds/step have been compared. Unknown reads/calls are not equivalent.
bool sameQueryExpression(Value lhs, Value rhs, func::FuncOp leftFunction,
                         func::FuncOp rightFunction, Value leftIV = {},
                         Value rightIV = {}, unsigned depth = 0) {
  if (!lhs || !rhs || lhs.getType() != rhs.getType() || depth > 64 ||
      !isa<IntegerType>(lhs.getType()))
    return false;
  if (lhs == leftIV)
    return rhs == rightIV;
  auto a = dyn_cast<BlockArgument>(lhs), b = dyn_cast<BlockArgument>(rhs);
  if (a || b)
    return a && b && a.getOwner() == &leftFunction.getBody().front() &&
           b.getOwner() == &rightFunction.getBody().front() &&
           a.getArgNumber() == b.getArgNumber();
  Operation *left = lhs.getDefiningOp(), *right = rhs.getDefiningOp();
  if (!left || !right || left->getName() != right->getName() ||
      left->getNumResults() != 1 || right->getNumResults() != 1 ||
      left->getNumOperands() != right->getNumOperands() ||
      !isa<arith::ConstantOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
           arith::RemSIOp, arith::RemUIOp, arith::MaxSIOp, arith::CeilDivSIOp,
           arith::ExtSIOp, arith::TruncIOp, GetBlockIdxOp>(left))
    return false;
  NamedAttrList leftAttributes(left->getAttrs()),
      rightAttributes(right->getAttrs());
  leftAttributes.erase("hivm.cvp_query.coordinate");
  rightAttributes.erase("hivm.cvp_query.coordinate");
  if (leftAttributes.getAttrs() != rightAttributes.getAttrs())
    return false;
  for (unsigned index = 0; index < left->getNumOperands(); ++index)
    if (!sameQueryExpression(left->getOperand(index), right->getOperand(index),
                             leftFunction, rightFunction, leftIV, rightIV,
                             depth + 1))
      return false;
  return true;
}

bool pairedVectorDrain(Pair &pair, Value work, Value oldEnd, int64_t step,
                       int64_t &extra, scf::ForOp &vectorLoop) {
  ModuleOp module = utils::getTopLevelModuleOp(pair.function);
  func::FuncOp vector;
  for (auto function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr("hacc.entry") ||
        function->hasAttr("hivm.backup_function") ||
        function->getAttr("hivm.func_core_type") !=
            TFuncCoreTypeAttr::get(function.getContext(), TFuncCoreType::AIV))
      continue;
    if (vector)
      return false;
    vector = function;
  }
  auto outer = dyn_cast<scf::ForOp>(pair.loop->getParentOp());
  if (!vector || !pair.function->hasAttr("hivm.part_of_mix") ||
      !vector->hasAttr("hivm.part_of_mix") ||
      pair.function.getFunctionType() != vector.getFunctionType() || !outer ||
      outer->getParentOp() != pair.function || !outer.getInitArgs().empty())
    return false;
  // After guarded shortening there is no signal-only suffix. Recover the
  // original drain from the unchanged paired Vector loop, never from a tag.
  auto cubeExtra =
      oldEnd ? extraAfterWork(oldEnd, work) : std::optional<int64_t>();
  if (oldEnd && !cubeExtra)
    return false;
  unsigned matches = 0;
  vector.walk([&](scf::ForOp loop) {
    if (!loop.getInductionVar().getType().isInteger(64) ||
        integer(loop.getLowerBound()) != 0 || integer(loop.getStep()) != step)
      return;
    auto vectorOuter = dyn_cast<scf::ForOp>(loop->getParentOp());
    if (!vectorOuter || vectorOuter->getParentOp() != vector ||
        !vectorOuter.getInitArgs().empty())
      return;
    for (auto bounds :
         {std::make_pair(outer.getLowerBound(), vectorOuter.getLowerBound()),
          std::make_pair(outer.getUpperBound(), vectorOuter.getUpperBound()),
          std::make_pair(outer.getStep(), vectorOuter.getStep())})
      if (!sameQueryExpression(bounds.first, bounds.second, pair.function,
                               vector))
        return;
    if (oldEnd) {
      if (!sameQueryExpression(oldEnd, loop.getUpperBound(), pair.function,
                               vector, outer.getInductionVar(),
                               vectorOuter.getInductionVar()))
        return;
    } else {
      auto end = loop.getUpperBound().getDefiningOp<arith::AddIOp>();
      std::optional<int64_t> vectorExtra;
      if (end)
        for (unsigned side : {0u, 1u})
          if (auto amount = integer(end->getOperand(1 - side)))
            if (sameQueryExpression(work, end->getOperand(side), pair.function,
                                    vector, outer.getInductionVar(),
                                    vectorOuter.getInductionVar()))
              vectorExtra = amount;
      if (!vectorExtra)
        return;
      cubeExtra = vectorExtra;
    }
    ++matches;
    vectorLoop = loop;
  });
  if (matches != 1 || !cubeExtra)
    return false;
  extra = *cubeExtra;
  return true;
}

bool pairedCrossSignals(ArrayRef<Operation *> left, ArrayRef<Operation *> right,
                        TCoreType leftCore, TCoreType rightCore) {
  if (left.size() != right.size())
    return false;
  SmallVector<bool> used(right.size(), false);
  auto valid = [](Operation *op, TCoreType core) {
    auto kind = op->getAttrOfType<TCoreTypeAttr>("tcore_type");
    auto flag = op->getAttrOfType<IntegerAttr>("static_flag_id");
    return isa<SyncBlockSetOp, SyncBlockWaitOp>(op) && !op->getNumOperands() &&
           !op->getNumResults() && kind && kind.getTcoretype() == core &&
           flag && flag.getInt() >= 0 && op->getAttrOfType<PipeAttr>("tpipe") &&
           op->getAttrOfType<PipeAttr>("pipe");
  };
  for (Operation *producer : left) {
    if (!valid(producer, leftCore))
      return false;
    bool found = false;
    for (unsigned i = 0; i < right.size(); ++i) {
      Operation *consumer = right[i];
      if (used[i] || !valid(consumer, rightCore) ||
          isa<SyncBlockSetOp>(producer) == isa<SyncBlockSetOp>(consumer))
        continue;
      bool equal = true;
      for (StringRef name :
           {"tpipe", "pipe", "static_flag_id", "tsync_instr_mode"})
        equal &= producer->getAttr(name) == consumer->getAttr(name);
      if (!equal)
        continue;
      used[i] = true;
      found = true;
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

bool crossBoundaryCredits(scf::ForOp main, scf::ForOp suffix,
                          SmallVectorImpl<Operation *> &initial,
                          SmallVectorImpl<Operation *> &returns,
                          bool preserveGuardedBody = false) {
  auto outer = dyn_cast<scf::ForOp>(main->getParentOp());
  auto function = main->getParentOfType<func::FuncOp>();
  if (!outer || !function || outer->getParentOp() != function)
    return false;
  Block *entry = &function.getBody().front();
  bool closed = true;
  function.walk([&](Operation *op) {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return;
    if (op->getBlock() == main.getBody() ||
        (suffix && op->getBlock() == suffix.getBody()) ||
        (preserveGuardedBody && main->isProperAncestor(op)))
      return;
    if (op->getBlock() != entry) {
      closed = false;
    } else if (isa<SyncBlockSetOp>(op) && op->isBeforeInBlock(outer)) {
      initial.push_back(op);
    } else if (isa<SyncBlockWaitOp>(op) && outer->isBeforeInBlock(op)) {
      returns.push_back(op);
    } else {
      closed = false;
    }
  });
  return closed;
}

bool schedulerCounterOnlyModuloTwo(Pair &pair, memref::LoadOp &load,
                                   memref::StoreOp &update) {
  for (auto candidate : pair.loop.getBody()->getOps<memref::LoadOp>()) {
    if (load)
      return false;
    load = candidate;
  }
  auto allocation = load ? load.getMemRef().getDefiningOp<memref::AllocaOp>()
                         : memref::AllocaOp();
  if (!privateScalarCounter(allocation, pair.function))
    return false;
  for (auto candidate : pair.loop.getBody()->getOps<memref::StoreOp>()) {
    if (update || candidate.getMemRef() != load.getMemRef())
      return false;
    update = candidate;
  }
  auto increment = update
                       ? update.getValueToStore().getDefiningOp<arith::AddIOp>()
                       : arith::AddIOp();
  if (!increment || !increment.getResult().hasOneUse() ||
      increment.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return false;
  unsigned moduloUses = 0;
  for (Operation *user : load.getResult().getUsers()) {
    if (user == increment)
      continue;
    auto remainder = dyn_cast<arith::RemUIOp>(user);
    if (!remainder || remainder.getLhs() != load.getResult() ||
        integer(remainder.getRhs()) != 2 ||
        remainder->getBlock() != pair.loop.getBody())
      return false;
    ++moduloUses;
  }
  return moduloUses != 0;
}

LogicalResult matchCubeDrain(Pair &pair, bool assigned, CubeDrainMatch &match,
                             std::string &reason) {
  auto reject = [&](StringRef message) -> LogicalResult {
    reason = message.str();
    return failure();
  };
  CubeReadyMatch ready;
  CubeOverlapMatch overlap;
  if (failed(matchCubeReadiness(pair, true, ready, reason)) ||
      failed(matchCubeOverlap(pair, pair.function->hasAttr(kCVPCubeOverlap),
                              overlap, reason)))
    return failure();
  auto policy = pair.function->getAttrOfType<StringAttr>(kCVPCubeDerivedDrain);
  if (assigned ? (!policy || (policy.getValue() != "cross-only-v1" &&
                              policy.getValue() != "guarded-cross-v1"))
               : pair.function->hasAttr(kCVPCubeDerivedDrain))
    return reject("invalid or partially assigned derived Cube drain");
  bool guarded = false, allGuarded = true;
  pair.loop.walk([&](Operation *op) {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return;
    bool inside = overlap.qActive->isProperAncestor(op) ||
                  overlap.pActive->isProperAncestor(op);
    guarded |= inside;
    allGuarded &= inside;
  });
  if ((guarded && !allGuarded) ||
      (assigned && guarded != (policy.getValue() == "guarded-cross-v1")))
    return reject("guarded Cube drain requires every loop handoff inside the "
                  "unchanged QK/PV active regions");
  if (!pair.loop.getInitArgs().empty() || pair.loop.getNumResults())
    return reject("derived drain cannot discard loop-carried values");
  Value work;
  if (!validity(overlap.qActive, pair.key, work, true))
    return reject("derived drain requires the actual QK work predicate");
  auto both = overlap.pActive.getCondition().getDefiningOp<arith::AndIOp>();
  int64_t delay = 0;
  if (both)
    for (Value term : both->getOperands()) {
      auto compare = term.getDefiningOp<arith::CmpIOp>();
      auto subtract = compare ? compare.getLhs().getDefiningOp<arith::SubIOp>()
                              : arith::SubIOp();
      auto offset =
          subtract ? integer(subtract.getRhs()) : std::optional<int64_t>();
      Value end;
      if (compare && compare.getPredicate() == arith::CmpIPredicate::sge &&
          integer(compare.getRhs()) == 0 && subtract &&
          subtract.getLhs() == pair.key && offset && *offset > 0 &&
          validity(overlap.pActive, subtract.getResult(), end, false) &&
          end == work)
        delay = *offset;
    }
  auto step = integer(pair.loop.getStep());
  if (!step || *step <= 0 || delay <= 0 || delay % *step || pair.keys < delay ||
      pair.keys % *step)
    return reject("unproved useful Cube interval or stage delay");

  Value oldEnd = pair.loop.getUpperBound();
  scf::ForOp suffix;
  if (assigned) {
    if (guarded) {
      bool unexpectedSuffix = false;
      pair.function.walk([&](Operation *op) {
        unexpectedSuffix |= op->hasAttr(kCVPCubeCreditDrainLoop);
      });
      if (unexpectedSuffix)
        return reject("guarded Cube drain must not add a signal-only suffix");
      oldEnd = {};
    } else {
      suffix = dyn_cast_or_null<scf::ForOp>(pair.loop->getNextNode());
      if (!suffix || !suffix->hasAttr(kCVPCubeCreditDrainLoop) ||
          !suffix.getInitArgs().empty() || suffix.getNumResults() ||
          suffix.getLowerBound() != pair.loop.getUpperBound() ||
          integer(suffix.getStep()) != *step)
        return reject("missing or malformed actual cross-core drain suffix");
      oldEnd = suffix.getUpperBound();
    }
    auto end = pair.loop.getUpperBound().getDefiningOp<arith::SelectOp>();
    auto positive = end ? end.getCondition().getDefiningOp<arith::CmpIOp>()
                        : arith::CmpIOp();
    if (!end || !positive ||
        positive.getPredicate() != arith::CmpIPredicate::sgt ||
        positive.getLhs() != work || integer(positive.getRhs()) != 0 ||
        integer(end.getFalseValue()) != 0 ||
        extraAfterWork(end.getTrueValue(), work) != delay)
      return reject("derived main bound does not match the useful work and "
                    "zero-work case");
  }
  int64_t extra = 0;
  scf::ForOp vectorLoop;
  if (!pairedVectorDrain(pair, work, oldEnd, *step, extra, vectorLoop))
    return reject("Cube drain must preserve the paired Vector query domain and "
                  "original end");
  if (extra < delay || extra % *step || ((extra - delay) / *step) % 2 ||
      (extra / *step) % 2 || !boundedWholeQueryWork(work, pair.keys, extra))
    return reject(
        "unproved bounded work or even scheduler-counter drain difference");
  memref::LoadOp counter;
  memref::StoreOp update;
  if (!schedulerCounterOnlyModuloTwo(pair, counter, update))
    return reject(
        "omitted scheduler-counter increments have an observable use");

  auto fenceOwner = [&](Operation *flag) {
    return flag->getBlock() == pair.loop.getBody() ? flag : flag->getParentOp();
  };
  SmallVector<Operation *> cross;
  for (Operation &op : pair.loop.getBody()->without_terminator()) {
    if (&op == overlap.qActive || &op == overlap.pActive ||
        &op == fenceOwner(overlap.intraSet) ||
        &op == fenceOwner(overlap.intraWait) || &op == counter || &op == update)
      continue;
    if (isa<SyncBlockSetOp, SyncBlockWaitOp>(op)) {
      if (op.getNumOperands() || op.getNumResults())
        return reject("dynamic cross-core operands cannot be moved to a "
                      "signal-only suffix");
      cross.push_back(&op);
      continue;
    }
    if (isa<SetFlagOp, WaitFlagOp>(op)) {
      auto from = op.getAttrOfType<PipeAttr>("set_pipe");
      auto to = op.getAttrOfType<PipeAttr>("wait_pipe");
      auto id = op.getAttrOfType<EventAttr>("static_event_id");
      if (from && to && id && !op.getNumOperands() &&
          from.getPipe() == PIPE::PIPE_FIX && to.getPipe() == PIPE::PIPE_M &&
          id.getEvent() == EVENT::EVENT_ID1)
        continue; // Complete next-write/return pairing was proved above.
    }
    if (op.getNumRegions() ||
        (op.getName().getDialectNamespace() != "arith" &&
         !(guarded && isa<affine::AffineApplyOp>(op))) ||
        !isMemoryEffectFree(&op))
      return reject("unknown effect in the proposed empty Cube suffix");
  }
  // A guarded stream has no handoff in an omitted tick: both work predicates
  // are false there. Preserve their complete regions and shorten only the
  // bound. The static path below instead preserves unconditional handoffs in
  // its existing signal-only suffix, with its original stricter grammar.
  if (guarded) {
    SmallVector<Operation *> cubeInitial, cubeReturns, vectorInitial,
        vectorReturns;
    if (!cross.empty() ||
        !crossBoundaryCredits(pair.loop, {}, cubeInitial, cubeReturns, true) ||
        !crossBoundaryCredits(vectorLoop, {}, vectorInitial, vectorReturns,
                              true) ||
        cubeInitial.empty() || cubeReturns.empty() || vectorInitial.empty() ||
        vectorReturns.empty() ||
        !pairedCrossSignals(cubeInitial, vectorReturns, TCoreType::CUBE,
                            TCoreType::VECTOR) ||
        !pairedCrossSignals(vectorInitial, cubeReturns, TCoreType::VECTOR,
                            TCoreType::CUBE))
      return reject("Cube/Vector initial and return cross-core credits are "
                    "incomplete for guarded drain");
    match = {work, oldEnd, *step, delay, extra, {}, {}, true};
    return success();
  }
  if (cross.empty())
    return reject(
        "derived drain requires the existing mixed-core signal protocol");
  SmallVector<Operation *> vectorCross;
  bool unconditional = true;
  vectorLoop.walk([&](Operation *op) {
    if (!isa<SyncBlockSetOp, SyncBlockWaitOp>(op))
      return;
    unconditional &= op->getBlock() == vectorLoop.getBody();
    vectorCross.push_back(op);
  });
  if (!unconditional || !pairedCrossSignals(cross, vectorCross, TCoreType::CUBE,
                                            TCoreType::VECTOR))
    return reject("Cube and Vector loop signals have unpaired static channels "
                  "or generations");
  SmallVector<Operation *> cubeInitial, cubeReturns, vectorInitial,
      vectorReturns;
  if (!crossBoundaryCredits(pair.loop, suffix, cubeInitial, cubeReturns) ||
      !crossBoundaryCredits(vectorLoop, {}, vectorInitial, vectorReturns) ||
      !pairedCrossSignals(cubeInitial, vectorReturns, TCoreType::CUBE,
                          TCoreType::VECTOR) ||
      !pairedCrossSignals(vectorInitial, cubeReturns, TCoreType::VECTOR,
                          TCoreType::CUBE))
    return reject(
        "Cube/Vector initial and return cross-core credits are incomplete");
  if (assigned) {
    unsigned index = 0;
    for (Operation &op : suffix.getBody()->without_terminator()) {
      if (index == cross.size() || op.getNumOperands() || op.getNumResults() ||
          op.getName() != cross[index]->getName() ||
          op.getAttrs() != cross[index]->getAttrs())
        return reject("cross-core suffix dropped, reordered or changed an "
                      "original signal");
      ++index;
    }
    if (index != cross.size())
      return reject(
          "cross-core suffix has incomplete original signal coverage");
  }
  match = {work, oldEnd, *step, delay, extra, cross, suffix};
  return success();
}

bool verifyDerivedCubeClock(Pair &pair, Value work) {
  CubeDrainMatch match;
  std::string reason;
  return succeeded(matchCubeDrain(pair, true, match, reason)) &&
         match.work == work;
}

struct CVPCubeDerivedDrainPass
    : impl::CVPCubeDerivedDrainBase<CVPCubeDerivedDrainPass> {
  using Base::Base;
  void runOnOperation() override {
    if (!enable)
      return;
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    ModuleOp module = *copy;
    unsigned count = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!isCubeEntry(function))
        continue;
      auto resource =
          function->getAttrOfType<DictionaryAttr>(kCVPQCacheResources);
      auto mode = resource ? resource.getAs<StringAttr>("mode") : StringAttr();
      std::string reason;
      if (!mode ||
          failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError(
            "CVP Cube derived drain requires its verified scheduled pair: ")
            << reason;
        return signalPassFailure();
      }
      auto pair = identify(function, true, reason);
      CubeDrainMatch match;
      bool assigned = function->hasAttr(kCVPCubeDerivedDrain);
      if (failed(pair) ||
          failed(matchCubeDrain(*pair, assigned, match, reason))) {
        original.emitError("CVP Cube derived drain: ") << reason;
        return signalPassFailure();
      }
      if (!assigned) {
        OpBuilder builder(pair->loop);
        Location loc = pair->loop.getLoc();
        Value zero = pair->loop.getLowerBound();
        Value positive = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::sgt, match.work, zero);
        Value delay =
            builder.create<arith::ConstantIntOp>(loc, match.delay, 64);
        Value usefulEnd = builder.create<arith::AddIOp>(loc, match.work, delay);
        Value mainEnd =
            builder.create<arith::SelectOp>(loc, positive, usefulEnd, zero);
        pair->loop.setUpperBound(mainEnd);
        if (!match.guarded) {
          builder.setInsertionPointAfter(pair->loop);
          auto suffix = builder.create<scf::ForOp>(loc, mainEnd, match.oldEnd,
                                                   pair->loop.getStep());
          suffix->setAttr(kCVPCubeCreditDrainLoop,
                          UnitAttr::get(&getContext()));
          builder.setInsertionPointToStart(suffix.getBody());
          for (Operation *signal : match.cross)
            builder.clone(*signal);
        }
        function->setAttr(
            kCVPCubeDerivedDrain,
            StringAttr::get(&getContext(), match.guarded ? "guarded-cross-v1"
                                                         : "cross-only-v1"));
      }
      if (failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError("CVP Cube derived drain final contract: ") << reason;
        return signalPassFailure();
      }
      ++count;
    }
    if (count != 1 || failed(verify(module))) {
      original.emitError(
          "CVP Cube derived drain requires exactly one complete mixed pair");
      return signalPassFailure();
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

struct CVPCubeOverlapPass : impl::CVPCubeOverlapBase<CVPCubeOverlapPass> {
  using Base::Base;
  void runOnOperation() override {
    if (!enable)
      return;
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    ModuleOp module = *copy;
    unsigned count = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!isCubeEntry(function))
        continue;
      auto resource =
          function->getAttrOfType<DictionaryAttr>(kCVPQCacheResources);
      auto mode = resource ? resource.getAs<StringAttr>("mode") : StringAttr();
      std::string reason;
      if (!mode ||
          failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError(
            "CVP Cube overlap requires its verified scheduled pair: ")
            << reason;
        return signalPassFailure();
      }
      auto pair = identify(function, true, reason);
      CubeOverlapMatch match;
      bool assigned = function->hasAttr(kCVPCubeOverlap);
      if (failed(pair) ||
          failed(matchCubeOverlap(*pair, assigned, match, reason))) {
        original.emitError("CVP Cube overlap: ") << reason;
        return signalPassFailure();
      }
      if (!assigned) {
        // Clone the proved two-comparison PV predicate before the first fence.
        // Its iv/constant/work operands already dominate this loop position.
        auto condition =
            match.pActive.getCondition().getDefiningOp<arith::AndIOp>();
        OpBuilder builder(match.intraSet);
        IRMapping mapping;
        for (Value operand : condition->getOperands())
          builder.clone(*operand.getDefiningOp(), mapping);
        auto active = cast<arith::AndIOp>(builder.clone(*condition, mapping));
        Value one =
            builder.create<arith::ConstantIntOp>(condition.getLoc(), 1, 1);
        Value inactive =
            builder.create<arith::XOrIOp>(condition.getLoc(), active, one);
        for (Operation *flag :
             {match.intraSet.getOperation(), match.intraWait.getOperation()}) {
          builder.setInsertionPoint(flag);
          auto guard =
              builder.create<scf::IfOp>(flag->getLoc(), inactive, false);
          flag->moveBefore(guard.thenBlock()->getTerminator());
        }
        function->setAttr(kCVPCubeOverlap,
                          StringAttr::get(&getContext(), "guarded-v1"));
      }
      if (failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError("CVP Cube overlap final contract: ") << reason;
        return signalPassFailure();
      }
      ++count;
    }
    if (count != 1 || failed(verify(module))) {
      original.emitError(
          "CVP Cube overlap requires exactly one complete eligible pair");
      return signalPassFailure();
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

struct CVPCubeReadinessPass : impl::CVPCubeReadinessBase<CVPCubeReadinessPass> {
  using Base::Base;
  void runOnOperation() override {
    if (!enable)
      return;
    ModuleOp original = getOperation();
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    ModuleOp module = *copy;
    unsigned count = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!isCubeEntry(function))
        continue;
      auto resource =
          function->getAttrOfType<DictionaryAttr>(kCVPQCacheResources);
      auto mode = resource ? resource.getAs<StringAttr>("mode") : StringAttr();
      std::string reason;
      if (!mode ||
          failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError(
            "CVP Cube readiness requires its scheduled normal-template pair: ")
            << reason;
        return signalPassFailure();
      }
      auto pair = identify(function, true, reason);
      CubeReadyMatch match;
      bool assigned = function->hasAttr(kCVPCubeReadiness);
      if (failed(pair) ||
          failed(matchCubeReadiness(*pair, assigned, match, reason))) {
        original.emitError("CVP Cube readiness: ") << reason;
        return signalPassFailure();
      }
      if (!assigned) {
        // Match the producer and consumer to the unchanged free-credit SSA.
        match.pvReady->removeAttr("static_event_id");
        match.pvReady.getDynamicEventIdMutable().assign(
            ValueRange{match.pvCredit});
        SmallVector<Value> sync(pair->pv.getSyncRelatedArgs());
        sync[1] = match.pvCredit;
        pair->pv.getSyncRelatedArgsMutable().assign(sync);
        auto resultEvent = EventAttr::get(&getContext(), EVENT::EVENT_ID1);
        match.pvResult.setStaticEventIdAttr(resultEvent);
        match.pvResultWait.setStaticEventIdAttr(resultEvent);
        function->setAttr(kCVPCubeReadiness,
                          StringAttr::get(&getContext(), "banked-v1"));
        pair->qk->setAttr(kCVPCubeBankedReady, UnitAttr::get(&getContext()));
        pair->pv->setAttr(kCVPCubeBankedReady, UnitAttr::get(&getContext()));
      }
      if (failed(verifyPair(function, mode.getValue(), true, reason))) {
        original.emitError("CVP Cube readiness final contract: ") << reason;
        return signalPassFailure();
      }
      ++count;
    }
    if (count != 1 || failed(verify(module))) {
      original.emitError(
          "CVP Cube readiness requires exactly one complete eligible pair");
      return signalPassFailure();
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

struct CVPQCachePass : impl::CVPQCacheBase<CVPQCachePass> {
  using Base::Base;
  void runOnOperation() override {
    StringRef selectedMode = mode.getValue();
    if (selectedMode == "none")
      return;
    ModuleOp original = getOperation();
    if (!crossCoreSync || unitFlagSync) {
      original.emitError("CVP Q-cache requires cross-core synchronization and "
                         "unit-flag mode0");
      return signalPassFailure();
    }
    if (selectedMode != "neutral" && selectedMode != "cached") {
      original.emitError("CVP Q-cache mode must be none, neutral or cached");
      return signalPassFailure();
    }
    OwningOpRef<ModuleOp> copy(cast<ModuleOp>(original->clone()));
    ModuleOp module = *copy;
    unsigned count = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!isCubeEntry(function))
        continue;
      std::string reason;
      auto plan = identify(function, false, reason);
      auto capacity = readCVPQCacheTargetCapacity(function);
      if (failed(plan) || failed(capacity)) {
        if (succeeded(plan) && failed(capacity))
          reason = "missing or unsupported A5 L0A/L0B capacities";
        if (requireMatch) {
          original.emitError("CVP Q-cache: ") << reason;
          signalPassFailure();
        }
        return;
      }
      if (function->hasAttr(kCVPQCacheResources)) {
        if (failed(verifyPair(function, selectedMode, false, reason))) {
          original.emitError("CVP Q-cache: ") << reason;
          return signalPassFailure();
        }
        ++count;
        continue;
      }
      Pair &pair = *plan;
      OpBuilder builder(pair.qk);
      auto resources = bishengir::cvp::deriveQCacheResources(
          capacity->first, capacity->second, 128, 256, 64, 2);
      function->setAttr(kCVPQCacheResources,
                        buildCVPQCacheResources(&getContext(), selectedMode,
                                                capacity->first,
                                                capacity->second));
      Value even = builder.create<arith::ConstantIntOp>(
          pair.qk.getLoc(), resources.q[0].begin, 64);
      Value odd = builder.create<arith::ConstantIntOp>(
          pair.qk.getLoc(), resources.q[1].begin, 64);
      Value offset = builder.create<arith::SelectOp>(pair.qk.getLoc(), pair.odd,
                                                     odd, even);
      auto descriptor = [&](StringRef role) {
        return builder.getDictionaryAttr(
            {builder.getNamedAttr("version", builder.getI64IntegerAttr(1)),
             builder.getNamedAttr("mode", builder.getStringAttr(selectedMode)),
             builder.getNamedAttr("role", builder.getStringAttr(role))});
      };
      pair.qk.getQCacheInitMutable().assign(ValueRange{pair.first});
      pair.qk.getQCacheGenerationMutable().assign(ValueRange{pair.generation});
      pair.qk.getQCacheByteOffsetMutable().assign(ValueRange{offset});
      pair.qk.setCvpQCacheAttr(descriptor("qk"));
      pair.pv.setCvpQCacheAttr(descriptor("pv"));
      if (failed(verifyPair(function, selectedMode, false, reason))) {
        original.emitError("CVP Q-cache: ") << reason;
        return signalPassFailure();
      }
      ++count;
    }
    if (!count || failed(verify(module))) {
      if (requireMatch) {
        original.emitError(
            "CVP Q-cache requires a verified eligible AIC query pair");
        signalPassFailure();
      }
      return;
    }
    original.getBodyRegion().takeBody(module.getBodyRegion());
    original->setAttrs(module->getAttrs());
  }
};

struct VerifyCVPQCachePass : impl::VerifyCVPQCacheBase<VerifyCVPQCachePass> {
  using Base::Base;
  void runOnOperation() override {
    StringRef selectedMode = mode.getValue();
    unsigned count = 0;
    ModuleOp module = getOperation();
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function->hasAttr(kCVPQCacheResources))
        continue;
      std::string reason;
      if (failed(verifyPair(function, selectedMode, true, reason))) {
        function.emitError("CVP Q-cache final verification: ") << reason;
        return signalPassFailure();
      }
      ++count;
    }
    bool unowned = false;
    module.walk([&](MmadL1Op op) {
      if (op.getCvpQCacheAttr()) {
        auto function = op->getParentOfType<func::FuncOp>();
        unowned |= !function || !function->hasAttr(kCVPQCacheResources);
      }
    });
    if (unowned || (requireMatch && count != 1)) {
      module.emitError(
          "CVP Q-cache requires one complete verified template pair");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> createCVPQCachePass(const CVPQCacheOptions &options) {
  return std::make_unique<CVPQCachePass>(options);
}
std::unique_ptr<Pass>
createVerifyCVPQCachePass(const VerifyCVPQCacheOptions &options) {
  return std::make_unique<VerifyCVPQCachePass>(options);
}
std::unique_ptr<Pass>
createCVPCubeReadinessPass(const CVPCubeReadinessOptions &options) {
  return std::make_unique<CVPCubeReadinessPass>(options);
}

std::unique_ptr<Pass>
createCVPCubeOverlapPass(const CVPCubeOverlapOptions &options) {
  return std::make_unique<CVPCubeOverlapPass>(options);
}

std::unique_ptr<Pass>
createCVPCubeDerivedDrainPass(const CVPCubeDerivedDrainOptions &options) {
  return std::make_unique<CVPCubeDerivedDrainPass>(options);
}
} // namespace hivm
} // namespace mlir

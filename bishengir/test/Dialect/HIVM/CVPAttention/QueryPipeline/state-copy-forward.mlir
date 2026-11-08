// RUN: bishengir-opt %s --hivm-cvp-query-state-copy-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s source-overwrite | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s snapshot-write | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s snapshot-escape | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s bank-escape | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s missing-unique | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s partial-copy | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s stronger-alignment | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s hidden-effect | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s bank-output-alias | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s exclusive-branches | bishengir-opt --hivm-cvp-query-state-copy-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-state-copy-forward.py %s same-branch-overwrite | bishengir-opt --hivm-cvp-query-state-copy-forward | FileCheck %s --check-prefix=REJECT

// FORWARD: hivm.cvp_query.snapshot_copies_forwarded = 1 : i64
// FORWARD-LABEL: func.func @case(
// FORWARD: [[BANK:%.*]] = memref.alloc() {hivm.cvp_query.state_bank
// FORWARD: annotation.mark [[BANK]] {mem_unique}
// FORWARD: scf.for
// FORWARD-NOT: memref.alloc()
// FORWARD: [[INPUT:%.*]] = arith.select %arg2, %arg1, [[BANK]]
// FORWARD: call @update(%arg0, [[INPUT]], %arg0)
// FORWARD: hivm.hir.copy ins(%arg0
// FORWARD-SAME: outs([[BANK]]

// REJECT: hivm.cvp_query.snapshot_copies_forwarded = 0 : i64
// REJECT-LABEL: func.func @case(
// REJECT: scf.for
// REJECT: memref.alloc()
// REJECT: hivm.hir.copy

#ub = #hivm.address_space<ub>
module {
  func.func private @opaque(memref<2x4xf32, #ub>)
  func.func private @global_effect()
  func.func @update(%pv: memref<2x4xf32, #ub>,
                    %old: memref<2x4xf32, #ub>,
                    %out: memref<2x4xf32, #ub>)
      attributes {hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %a = vector.transfer_read %old[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x4xf32, #ub>, vector<2x4xf32>
    %b = vector.transfer_read %pv[%c0, %c0], %zero {in_bounds = [true, true]} : memref<2x4xf32, #ub>, vector<2x4xf32>
    %sum = arith.addf %a, %b : vector<2x4xf32>
    vector.transfer_write %sum, %out[%c0, %c0] {in_bounds = [true, true]} : vector<2x4xf32>, memref<2x4xf32, #ub>
    return
  }
  func.func @case(%pv: memref<2x4xf32, #ub>,
                  %zero: memref<2x4xf32, #ub>, %first: i1)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %bank = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "O", required_slots = 1 : i32, role = "accumulator"}} : memref<2x4xf32, #ub>
    annotation.mark %bank {mem_unique} : memref<2x4xf32, #ub>
    hivm.hir.copy ins(%zero : memref<2x4xf32, #ub>) outs(%bank : memref<2x4xf32, #ub>)
    scf.for %key = %c0 to %c2 step %c1 {
      scope.scope : () -> () {
        %snapshot = memref.alloc() : memref<2x4xf32, #ub>
        hivm.hir.copy ins(%bank : memref<2x4xf32, #ub>) outs(%snapshot : memref<2x4xf32, #ub>)
        %selected = arith.select %first, %zero, %snapshot : memref<2x4xf32, #ub>
        func.call @update(%pv, %selected, %pv) {hivm.vector_function} : (memref<2x4xf32, #ub>, memref<2x4xf32, #ub>, memref<2x4xf32, #ub>) -> ()
        hivm.hir.copy ins(%pv : memref<2x4xf32, #ub>) outs(%bank : memref<2x4xf32, #ub>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 4 : i32}
    } {hivm.cvp_query.stream = {key_count = 2 : i64, key_step = 1 : i64, query_step = 1 : i64}, hivm.cvp_query.state_storage}
    return
  }
}

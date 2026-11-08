// RUN: bishengir-opt %s --hivm-cvp-query-final-output-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s rename | bishengir-opt --hivm-cvp-query-final-output-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s stronger-alignment | bishengir-opt --hivm-cvp-query-final-output-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-input | bishengir-opt --hivm-cvp-query-final-output-forward="require-match" | FileCheck %s --check-prefix=FORWARD
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-extra-fresh | bishengir-opt --hivm-cvp-query-final-output-forward="require-match" | FileCheck %s --check-prefix=SCOPED
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-input-unknown | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-extra-alias | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-extra-select | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s scoped-extra-view | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s read-output | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s partial-rows | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s partial-columns | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s temporary-escape | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s unknown-between | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s fence-between | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s wrong-clock | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s same-bank | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s missing-storage | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s bank-escape | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate-final-output-forward.py %s later-condition | bishengir-opt --hivm-cvp-query-final-output-forward | FileCheck %s --check-prefix=REJECT

// FORWARD: hivm.cvp_query.final_output_copies_forwarded = 1 : i64
// FORWARD-LABEL: func.func @case(
// FORWARD: scf.for
// FORWARD: scope.scope
// FORWARD-NOT: memref.alloc
// FORWARD: [[DEST:%.*]] = arith.select
// FORWARD: call @{{convert|renamed_convert}}({{.*}}, [[DEST]])
// FORWARD-NOT: hivm.hir.copy
// FORWARD: hivm.hir.store
// SCOPED: hivm.cvp_query.final_output_copies_forwarded = 1 : i64
// SCOPED-LABEL: func.func @case(
// SCOPED: scf.for
// SCOPED: scope.scope
// SCOPED-NOT: memref.alloc
// SCOPED: [[DEST:%.*]] = arith.select
// SCOPED: call @convert({{.*}}, [[DEST]], {{.*}})
// SCOPED-NOT: hivm.hir.copy
// SCOPED: hivm.hir.store
// REJECT: hivm.cvp_query.final_output_copies_forwarded = 0 : i64
// REJECT-LABEL: func.func @case(
// REJECT: scf.for
// REJECT: memref.alloc
// REJECT: hivm.hir.copy

#ub = #hivm.address_space<ub>
#gm = #hivm.address_space<gm>
module {
  func.func private @opaque(memref<4x64xf16, #ub>)
  func.func private @global_effect()
  func.func @convert(%input: memref<4x64xf32, #ub>, %output: memref<4x64xf16, #ub>)
      attributes {hivm.vector_function, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %c4 step %c1 {
      %inrow = memref.subview %input[%row, 0] [1, 64] [1, 1] : memref<4x64xf32, #ub> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
      %outrow = memref.subview %output[%row, 0] [1, 64] [1, 1] : memref<4x64xf16, #ub> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ub>
      %values = vector.transfer_read %inrow[%c0, %c0], %zero {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf32>
      %halves = arith.truncf %values : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %halves, %outrow[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ub>
    }
    return
  }
  func.func @case(%destination: memref<4x64xf16, #gm>, %slot: i1)
      attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %true = arith.constant true
    %input = memref.alloc() : memref<4x64xf32, #ub>
    %bank0 = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<4x64xf16, #ub>
    %bank1 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<4x64xf16, #ub>
    annotation.mark %bank0 {mem_unique} : memref<4x64xf16, #ub>
    annotation.mark %bank1 {mem_unique} : memref<4x64xf16, #ub>
    scf.for %key = %c0 to %c2 step %c1 {
      scope.scope : () -> () {
        %temporary = memref.alloc() {alignment = 64 : i64} : memref<4x64xf16, #ub>
        func.call @convert(%input, %temporary) {hivm.vector_function} : (memref<4x64xf32, #ub>, memref<4x64xf16, #ub>) -> ()
        %selected = arith.select %slot, %bank1, %bank0 : memref<4x64xf16, #ub>
        hivm.hir.copy ins(%temporary : memref<4x64xf16, #ub>) outs(%selected : memref<4x64xf16, #ub>)
        hivm.hir.store ins(%selected : memref<4x64xf16, #ub>) outs(%destination : memref<4x64xf16, #gm>)
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 4 : i32}
    } {hivm.cvp_query.stream = {key_count = 2 : i64, key_step = 1 : i64, query_step = 1 : i64}, hivm.cvp_query.output_storage}
    return
  }
}

// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: bishengir-opt %t --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.second
// RUN: diff %t %t.second
// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-overlap="enable=true" --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.combined
// RUN: FileCheck %s --input-file=%t.combined
// The suffix keeps the six original signals in order and has no matrix,
// local compute-fence or scalar bank-counter operation.
// CHECK-LABEL: func.func @_attn_fwd_mix_aic
// CHECK-SAME: hivm.cvp_cube.derived_drain = "cross-only-v1"
// CHECK: %[[WORK:[^ ]+]] = arith.muli {{.*}}hivm.cvp_query.coordinate = "work_end"
// CHECK: %[[OLD:[^ ]+]] = arith.addi %[[WORK]], {{.*}} : i64
// CHECK: %[[POS:[^ ]+]] = arith.cmpi sgt, %[[WORK]], %[[ZERO:[^ ]+]] : i64
// CHECK: %[[USEFUL:[^ ]+]] = arith.addi %[[WORK]], {{.*}} : i64
// CHECK: %[[END:[^ ]+]] = arith.select %[[POS]], %[[USEFUL]], %[[ZERO]] : i64
// CHECK: scf.for {{.*}} = %[[ZERO]] to %[[END]] step %[[STEP:[^ ]+]]
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.mmadL1
// CHECK: memref.store
// CHECK: scf.for {{.*}} = %[[END]] to %[[OLD]] step %[[STEP]]
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
// CHECK-NEXT: } {hivm.cvp_cube.credit_drain_loop}
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
// CHECK: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1

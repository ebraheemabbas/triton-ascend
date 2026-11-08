// RUN: bishengir-opt %S/Inputs/gss-handoffs-captured.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: python3 %S/Inputs/mutate-gss-handoffs.py check %S/Inputs/gss-handoffs-captured.mlir %t
// RUN: bishengir-opt %t --hivm-cvp-cube-readiness="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.second
// RUN: diff %t %t.second
// Actual initial BeforeReadiness capture from the normal source compiler with
// live filtered-backup restoration. Stock GSS handoffs remain in their places.
// Fixture SHA256: efb936e7fa7765d832a07940e616eb1b040aa234ff62e835f0691636f51053d6.
// This fixture is used for CPU pass tests, never injected into a hardware run.

// CHECK-LABEL: func.func @_attn_fwd_mix_aic
// CHECK-SAME: hivm.cvp_cube.readiness = "banked-v1"
// CHECK: hivm.hir.mmadL1 {{.*}}hivm.cvp_cube.banked_ready
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = {{.*}}
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
// CHECK-NEXT: hivm.hir.fixpipe
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %[[BANK:[a-zA-Z0-9_.$]+]]]
// CHECK-NEXT: hivm.hir.nd2nz
// CHECK-NEXT: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %[[BANK]]]
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
// CHECK-NEXT: hivm.hir.mmadL1 {{.*}}hivm.cvp_cube.banked_ready
// CHECK-NEXT: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
// CHECK-NEXT: hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = {{.*}}
// CHECK-NEXT: hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = {{.*}}
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
// CHECK-NEXT: hivm.hir.fixpipe

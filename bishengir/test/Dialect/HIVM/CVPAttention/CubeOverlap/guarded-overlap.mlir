// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-overlap="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: bishengir-opt %t --hivm-cvp-cube-overlap="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.second
// RUN: diff %t %t.second
// This captured schedule uses the actual scalar bank, physical L0C ranges,
// active QK/PV predicates, normal helper readiness and all return credits.
// CHECK-LABEL: func.func @_attn_fwd_mix_aic
// CHECK-SAME: hivm.cvp_cube.overlap = "guarded-v1"
// CHECK-SAME: hivm.cvp_cube.readiness = "banked-v1"
// CHECK: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK: %[[INACTIVE:[^ ]+]] = arith.xori
// CHECK: scf.if %[[INACTIVE]] {
// CHECK-NEXT: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
// CHECK-NEXT: }
// CHECK: scf.if %[[INACTIVE]] {
// CHECK-NEXT: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
// CHECK-NEXT: }
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.fixpipe
// CHECK: hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]

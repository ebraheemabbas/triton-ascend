// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: bishengir-opt %t --hivm-cvp-cube-readiness="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.second
// RUN: diff %t %t.second
// The captured scheduled input exercises the real scalar bank counter, both
// template calls, existing L1 free credits, L0C reuse fences and return drains.
// No compiler output is replaced by this fixture in the native candidate path.
// CHECK-LABEL: func.func @_attn_fwd_mix_aic
// CHECK-SAME: hivm.cvp_cube.readiness = "banked-v1"
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
// CHECK: %[[BANK:[^ ]+]] = arith.select {{.*}}, %c1_i64, %c2_i64
// CHECK: hivm.hir.mmadL1 {{.*}}hivm.cvp_cube.banked_ready
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %[[BANK]]]
// CHECK: hivm.hir.nd2nz
// CHECK: hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, %[[BANK]]]
// CHECK: hivm.hir.mmadL1 {{.*}}hivm.cvp_cube.banked_ready{{.*}}sync_related_args({{.*}}%[[BANK]]{{.*}}%[[BANK]]
// CHECK: hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID1>]
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
// CHECK: hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]

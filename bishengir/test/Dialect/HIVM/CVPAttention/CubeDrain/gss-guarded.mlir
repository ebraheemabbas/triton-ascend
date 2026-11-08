// RUN: bishengir-opt %S/Inputs/gss-guarded-captured.mlir --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: %python %S/Inputs/guarded-drain.py preserve %S/Inputs/gss-guarded-captured.mlir %t
// RUN: FileCheck %s --input-file=%t
// RUN: bishengir-opt %t --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.second
// RUN: diff %t %t.second
// Actual normal compiler input after overlap; structured GSS handoffs retain
// every active QK/PV region and flag expression. No signal-only suffix is added.
// CHECK-LABEL: func.func @_attn_fwd_mix_aic(
// CHECK-SAME: hivm.cvp_cube.derived_drain = "guarded-cross-v1"
// CHECK: %[[WORK:[^ ]+]] = arith.muli {{.*}}hivm.cvp_query.coordinate = "work_end"
// CHECK: %[[POS:[^ ]+]] = arith.cmpi sgt, %[[WORK]], %[[ZERO:[^ ]+]] : i64
// CHECK: %[[USEFUL:[^ ]+]] = arith.addi %[[WORK]], {{.*}} : i64
// CHECK: %[[END:[^ ]+]] = arith.select %[[POS]], %[[USEFUL]], %[[ZERO]] : i64
// CHECK: scf.for {{.*}} = %[[ZERO]] to %[[END]] step
// CHECK: affine.apply
// CHECK: hivm.hir.mmadL1
// CHECK: hivm.hir.mmadL1
// CHECK: memref.store

// Actual original HD128/BN128/VF-off triplet. The companion input retains the
// tensor backup and split kernels; proof metadata is restored for this late pass.
// RUN: bishengir-opt %S/Inputs/actual-hd128.mlir --verify-each --hivm-cvp-attention-vector="require-match=true" | FileCheck %s --check-prefixes=CHECK,REMAT
// RUN: bishengir-opt %S/Inputs/actual-hd128.mlir --verify-each --hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true" | FileCheck %s --check-prefixes=CHECK,MAX
// RUN: bishengir-opt %S/Inputs/actual-hd128.mlir --verify-each --hivm-cvp-attention-vector="enable-max-scaling=true enable-triton-kernel-compile=true" -o %t.one
// RUN: bishengir-opt %t.one --verify-each --hivm-cvp-attention-vector="enable-max-scaling=true enable-triton-kernel-compile=true" -o %t.two
// RUN: diff %t.one %t.two
// RUN: %python %S/Inputs/check_delta.py %S/Inputs/actual-hd128.mlir %t.one
// CHECK: module attributes {{.*}}hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// CHECK-SAME: hivm.cvp_attention.score_store_removals = 1 : i64
// CHECK-LABEL: func.func @_attn_fwd_mix_aiv_outlined_vf_0(
// CHECK-SAME: %arg0: memref<64x128xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>) attributes
// CHECK: %[[R:[^ ]+]] = vector.transfer_read
// REMAT-NEXT: %[[S:[^ ]+]] = arith.mulf %[[R]],
// REMAT-NEXT: %{{[^ ]+}} = arith.maximumf %[[S]],
// MAX-NEXT: %{{[^ ]+}} = arith.maximumf %[[R]],
// CHECK-NOT: vector.transfer_write {{.*}}vector<1x64xf32>
// CHECK: %[[M:[^ ]+]] = vector.multi_reduction <maximumf>
// MAX-NEXT: %[[F:[^ ]+]] = arith.constant dense<5.000000e-01> : vector<1xf32>
// MAX-NEXT: %[[MS:[^ ]+]] = arith.mulf %[[M]], %[[F]] : vector<1xf32>
// MAX-NEXT: vector.transfer_write %[[MS]],
// CHECK-LABEL: func.func @_attn_fwd_mix_aiv_outlined_vf_2(
// CHECK: %[[VIEW:[^ ]+]] = memref.subview %arg1[
// CHECK-NEXT: %[[PR:[^ ]+]] = vector.transfer_read %[[VIEW]][
// CHECK-NEXT: %[[PF:[^ ]+]] = arith.constant dense<5.000000e-01> : vector<1x64xf32>
// CHECK-NEXT: %[[PS:[^ ]+]] = arith.mulf %[[PR]], %[[PF]] : vector<1x64xf32>
// CHECK-NEXT: %{{[^ ]+}} = arith.subf %[[PS]],
// CHECK-LABEL: func.func @_attn_fwd_mix_aiv(
// CHECK: annotation.mark %[[RAW:[^ ]+]] {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>
// CHECK: func.call @_attn_fwd_mix_aiv_outlined_vf_0(%[[RAW]], %[[TMAX:[^ )]+]]) {{.*}} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
// CHECK: func.call @_attn_fwd_mix_aiv_outlined_vf_2(%{{[^,]+}}, %[[RAW]], %{{[^,]+}}, %{{[^)]+}})
// CHECK: hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1

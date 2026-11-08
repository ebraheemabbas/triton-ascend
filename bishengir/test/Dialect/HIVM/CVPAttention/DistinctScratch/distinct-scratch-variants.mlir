// RUN: %python %S/Inputs/mutate.py renamed %S/Inputs/actual-hd128.mlir > %t.renamed
// RUN: bishengir-opt %t.renamed --verify-each --hivm-cvp-attention-vector="require-match=true" | FileCheck %s --check-prefix=RENAMED
// RUN: %python %S/Inputs/mutate.py renumbered %S/Inputs/actual-hd128.mlir > %t.renumbered
// RUN: bishengir-opt %t.renumbered --verify-each --hivm-cvp-attention-vector="require-match=true" | FileCheck %s --check-prefix=NUMBERED
// RUN: %python %S/Inputs/mutate.py multiply-attributes %S/Inputs/actual-hd128.mlir > %t.multiply
// RUN: bishengir-opt %t.multiply --verify-each --hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true" | FileCheck %s --check-prefix=ATTRS
// RUN: %python %S/Inputs/mutate.py argument-attributes %S/Inputs/actual-hd128.mlir > %t.args
// RUN: bishengir-opt %t.args --verify-each --hivm-cvp-attention-vector="require-match=true" | FileCheck %s --check-prefix=ARGS
// RENAMED: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// RENAMED: func.func @renamed_kernel_mix_aiv_outlined_vf_0(
// NUMBERED: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// NUMBERED: hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<17>
// ATTRS: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// ATTRS-LABEL: func.func @_attn_fwd_mix_aiv_outlined_vf_0(
// ATTRS: %[[M:[^ ]+]] = vector.multi_reduction <maximumf>
// ATTRS-NEXT: %[[F:[^ ]+]] = arith.constant dense<5.000000e-01> : vector<1xf32>
// ATTRS-NEXT: %{{[^ ]+}} = arith.mulf %[[F]], %[[M]] fastmath<contract> {cvp_test_marker = "original-scale"} : vector<1xf32>
// ATTRS-LABEL: func.func @_attn_fwd_mix_aiv_outlined_vf_2(
// ATTRS: %[[V:[^ ]+]] = memref.subview %arg1[
// ATTRS-NEXT: %[[R:[^ ]+]] = vector.transfer_read %[[V]][
// ATTRS-NEXT: %[[PF:[^ ]+]] = arith.constant dense<5.000000e-01> : vector<1x64xf32>
// ATTRS-NEXT: %{{[^ ]+}} = arith.mulf %[[PF]], %[[R]] fastmath<contract> {cvp_test_marker = "original-scale"} : vector<1x64xf32>
// ARGS: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// ARGS-LABEL: func.func @_attn_fwd_mix_aiv_outlined_vf_0(
// ARGS-SAME: %arg0: memref<64x128xf32, #hivm.address_space<ub>> {llvm.readonly}, %arg1: memref<64xf32, #hivm.address_space<ub>> {llvm.align = 64 : i64}) attributes
// ARGS-SAME: hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline
// ARGS: func.call @_attn_fwd_mix_aiv_outlined_vf_0(%{{[^,]+}}, %{{[^)]+}}) {hivm.vector_function, no_inline} : (memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()

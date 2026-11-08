// RUN: bishengir-opt %s --verify-each --hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true" | FileCheck %s
// RUN: bishengir-opt %s --verify-each --hivm-cvp-attention-vector="enable-max-scaling=true enable-triton-kernel-compile=true" -o %t.one
// RUN: bishengir-opt %t.one --verify-each --hivm-cvp-attention-vector="enable-max-scaling=true enable-triton-kernel-compile=true" -o %t.two
// RUN: diff %t.one %t.two
// CHECK: module attributes {{.*}}hivm.cvp_attention.max_scalings = 1 : i64{{.*}}hivm.cvp_attention.score_store_removals = 1 : i64
// CHECK-LABEL: func.func @unrelated_a(
// CHECK: %[[RAW:[^ ]+]] = vector.transfer_read
// CHECK-NEXT: %{{[^ ]+}} = arith.maximumf %[[RAW]],
// CHECK: %[[MAX:[^ ]+]] = vector.multi_reduction <maximumf>
// CHECK-NEXT: %[[FACTOR:[^ ]+]] = arith.constant dense<5.000000e-01> : vector<1xf32>
// CHECK-NEXT: %[[SCALED:[^ ]+]] = arith.mulf %[[MAX]], %[[FACTOR]] : vector<1xf32>
// CHECK-NEXT: vector.transfer_write %[[SCALED]],
// CHECK-LABEL: func.func @unrelated_b(
// CHECK: %[[VIEW:[^ ]+]] = memref.subview %arg1[
// CHECK: %[[PRAW:[^ ]+]] = vector.transfer_read %[[VIEW]][
// CHECK: %[[PSCALED:[^ ]+]] = arith.mulf %[[PRAW]],
// CHECK-NEXT: %{{[^ ]+}} = arith.subf %[[PSCALED]],

#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @unrelated_a(%renamed_arg0: memref<64x256xf32, #hivm.address_space<ub>>, %renamed_arg1: memref<64xf32, #hivm.address_space<ub>>, %renamed_arg2: memref<64x256xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %renamed_cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %renamed_cst_0 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %renamed_cst_1 = arith.constant 0.000000e+00 : f32
    %renamed_c1 = arith.constant 1 : index
    %renamed_c64 = arith.constant 64 : index
    %renamed_c256 = arith.constant 256 : index
    %renamed_c0 = arith.constant 0 : index
    scf.for %renamed_arg3 = %renamed_c0 to %renamed_c64 step %renamed_c1 {
      %renamed_subview = memref.subview %renamed_arg1[%renamed_arg3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = scf.for %renamed_arg4 = %renamed_c0 to %renamed_c256 step %renamed_c64 iter_args(%renamed_arg5 = %renamed_cst_0) -> (vector<1x64xf32>) {
        %renamed_subview_2 = memref.subview %renamed_arg0[%renamed_arg3, %renamed_arg4] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %renamed_subview_3 = memref.subview %renamed_arg2[%renamed_arg3, %renamed_arg4] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %3 = vector.transfer_read %renamed_subview_2[%renamed_c0, %renamed_c0], %renamed_cst_1 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %4 = arith.mulf %3, %renamed_cst : vector<1x64xf32>
        vector.transfer_write %4, %renamed_subview_3[%renamed_c0, %renamed_c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = arith.maximumf %4, %renamed_arg5 {reductionOp} : vector<1x64xf32>
        scf.yield %5 : vector<1x64xf32>
      } {reductionLoop}
      %1 = vector.transfer_read %renamed_subview[%renamed_c0], %renamed_cst_1 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %2 = vector.multi_reduction <maximumf>, %0, %1 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %2, %renamed_subview[%renamed_c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @unrelated_b(%renamed_arg0: memref<64xf32, #hivm.address_space<ub>>, %renamed_arg1: memref<64x256xf32, #hivm.address_space<ub>>, %renamed_arg2: memref<64xf32, #hivm.address_space<ub>>, %renamed_arg3: memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %renamed_cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %renamed_cst_0 = arith.constant 0.000000e+00 : f32
    %renamed_c1 = arith.constant 1 : index
    %renamed_c64 = arith.constant 64 : index
    %renamed_c4 = arith.constant 4 : index
    %renamed_c16 = arith.constant 16 : index
    %renamed_c0 = arith.constant 0 : index
    scf.for %renamed_arg4 = %renamed_c0 to %renamed_c64 step %renamed_c1 {
      %renamed_subview = memref.subview %renamed_arg2[%renamed_arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %renamed_subview_1 = memref.subview %renamed_arg0[%renamed_arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %renamed_subview_1[%renamed_c0], %renamed_cst_0 {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = scf.for %renamed_arg5 = %renamed_c0 to %renamed_c16 step %renamed_c4 iter_args(%renamed_arg6 = %renamed_cst) -> (vector<1x64xf32>) {
        %4 = affine.apply #map3(%renamed_arg5)
        %renamed_subview_2 = memref.subview %renamed_arg1[%renamed_arg4, %4] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %renamed_subview_2[%renamed_c0, %renamed_c0], %renamed_cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %6 = arith.subf %5, %0 : vector<1x64xf32>
        %7 = math.exp %6 : vector<1x64xf32>
        %8 = arith.addf %7, %renamed_arg6 {reductionOp} : vector<1x64xf32>
        %9 = arith.truncf %7 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
        %renamed_subview_3 = memref.subview %renamed_arg3[%renamed_arg5, %renamed_arg4, 0] [4, 1, 16] [1, 1, 1] : memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %10 = vector.shape_cast %9 : vector<1x64xf16> to vector<4x1x16xf16>
        vector.transfer_write %10, %renamed_subview_3[%renamed_c0, %renamed_c0, %renamed_c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        scf.yield %8 : vector<1x64xf32>
      } {reductionLoop, unroll_for_vsstb}
      %2 = vector.transfer_read %renamed_subview[%renamed_c0], %renamed_cst_0 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %3 = vector.multi_reduction <add>, %1, %2 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %3, %renamed_subview[%renamed_c0] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @unrelated_c(%renamed_maximum: memref<64xf32, #hivm.address_space<ub>>, %renamed_sum: memref<64xf32, #hivm.address_space<ub>>, %renamed_p: memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64}], hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix} {
    %renamed_raw = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
    scope.scope : () -> () {
      func.call @unrelated_a(%renamed_raw, %renamed_maximum, %renamed_raw) : (memref<64x256xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x256xf32, #hivm.address_space<ub>>) -> ()
      func.call @unrelated_b(%renamed_maximum, %renamed_raw, %renamed_sum, %renamed_p) : (memref<64xf32, #hivm.address_space<ub>>, memref<64x256xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) -> ()
      scope.return
    } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

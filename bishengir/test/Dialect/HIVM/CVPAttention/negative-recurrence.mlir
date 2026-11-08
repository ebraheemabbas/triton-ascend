// RUN: bishengir-opt --verify-each %s --hivm-cvp-attention-prepare -o %t 2>&1 | FileCheck %s --check-prefix=DIAG
// RUN: FileCheck %s --check-prefix=NO --implicit-check-not=hivm.cvp_attention.plan --implicit-check-not=hivm.cvp_attention.role --implicit-check-not=hivm.cvp_attention.applied --implicit-check-not=hivm.cvp_attention.verified --input-file=%t
// RUN: bishengir-opt %t --hivm-verify-cvp-attention="emit-remarks=false drop-annotations=true retain-provenance=true" | FileCheck %s --check-prefix=CLEAN
// DIAG: CVP attention skipped: missing l or O recurrence
// NO: func.func
// NO: hivm.cvp_attention.expected_count = 0 : i64
// CLEAN: module
// CLEAN-NOT: hivm.cvp_attention

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @_attn_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %cst = arith.constant 0.000000e+00 : f32
  %c8192_i32 = arith.constant 8192 : i32
  %c128_i32 = arith.constant 128 : i32
  %c65536_i64 = arith.constant 65536 : i64
  %c524288_i64 = arith.constant 524288 : i64
  %c8_i32 = arith.constant 8 : i32
  %c0_i32 = arith.constant 0 : i32
  %c1024_i32 = arith.constant 1024 : i32
  %c64 = arith.constant 64 : index
  %cst_0 = arith.constant 5.000000e-01 : f32
  %c256_i32 = arith.constant 256 : i32
  %cst_1 = arith.constant 0xFF800000 : f32
  %cst_2 = arith.constant 1.000000e+00 : f32
  %c128 = arith.constant 128 : index
  %c256 = arith.constant 256 : index
  %true = arith.constant true
  %c28_i32 = arith.constant 28 : i32
  %0 = arith.muli %arg7, %arg8 : i32
  %1 = arith.muli %0, %arg9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %4 = arith.remsi %arg10, %arg7 : i32
    %5 = tensor.empty() : tensor<128x64xf32>
    %6 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %7 = tensor.empty() : tensor<128xf32>
    %8 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_1 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %9 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_2 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    scf.for %arg11 = %4 to %c8192_i32 step %c28_i32  : i32 {
      %10 = arith.divsi %arg11, %c8_i32 : i32
      %11 = arith.remsi %arg11, %c8_i32 : i32
      %12 = arith.divsi %10, %c8_i32 : i32
      %13 = arith.remsi %10, %c8_i32 : i32
      %14 = arith.extsi %12 : i32 to i64
      %15 = arith.muli %14, %c524288_i64 : i64
      %16 = arith.extsi %13 : i32 to i64
      %17 = arith.muli %16, %c65536_i64 : i64
      %18 = arith.addi %15, %17 : i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.muli %11, %c128_i32 : i32
      %21 = arith.maxsi %20, %c0_i32 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %22]
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %alloc = memref.alloc() : memref<4x8x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<4x8x16x16xf16>)
      %24 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16x16xf16>
      %25:5 = scf.for %arg12 = %c0_i32 to %c1024_i32 step %c256_i32 iter_args(%arg13 = %6, %arg14 = %9, %arg15 = %8, %arg16 = %c0_i32, %arg17 = %c0_i32) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %35 = arith.maxsi %arg16, %c0_i32 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %36]
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%37], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %38 = arith.maxsi %arg17, %c0_i32 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %39]
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<4x16x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_5 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_7 : memref<4x16x16x16xf16>)
        %41 = bufferization.to_tensor %alloc_7 restrict writable : memref<4x16x16x16xf16>
        %42 = tensor.empty() : tensor<16x8x16x16xf32>
        %43 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%24, %41, %true, %c128, %c64, %c256 : tensor<4x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%42 : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
        %alloc_8 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x256xf32, #hivm.address_space<ub>>
        %memspacecast = memref.memory_space_cast %alloc_8 : memref<128x256xf32, #hivm.address_space<ub>> to memref<128x256xf32>
        %44 = bufferization.to_tensor %memspacecast restrict writable : memref<128x256xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%43 : tensor<16x8x16x16xf32>) outs(%alloc_8 : memref<128x256xf32, #hivm.address_space<ub>>)
        %45 = tensor.empty() : tensor<128x256xf32>
        %46 = hivm.hir.vmul ins(%44, %cst_0 : tensor<128x256xf32>, f32) outs(%45 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %expanded_9 = tensor.expand_shape %8 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %47 = hivm.hir.vreduce <max> ins(%46 : tensor<128x256xf32>) outs(%expanded_9 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed = tensor.collapse_shape %47 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %48 = hivm.hir.vmax ins(%arg15, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_10 = tensor.expand_shape %48 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %49 = hivm.hir.vsub ins(%46, %expanded_10 : tensor<128x256xf32>, tensor<128x1xf32>) outs(%45 : tensor<128x256xf32>) broadcast = [1] -> tensor<128x256xf32>
        %50 = hivm.hir.vexp ins(%49 : tensor<128x256xf32>) outs(%45 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %51 = tensor.empty() : tensor<128x256xf16>
        %52 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%50 : tensor<128x256xf32>) outs(%51 : tensor<128x256xf16>) -> tensor<128x256xf16>
        %alloc_11 = memref.alloc() : memref<4x16x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_6 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_11 : memref<4x16x16x16xf16>)
        %53 = bufferization.to_tensor %alloc_11 restrict writable : memref<4x16x16x16xf16>
        %54 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_12 = tensor.expand_shape %54 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %55 = hivm.hir.vreduce <sum> ins(%50 : tensor<128x256xf32>) outs(%expanded_12 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %collapsed_13 = tensor.collapse_shape %55 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %56 = hivm.hir.vsub ins(%arg15, %48 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %57 = hivm.hir.vexp ins(%56 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %58 = hivm.hir.vmul ins(%arg14, %57 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %59 = hivm.hir.vadd ins(%58, %collapsed_13 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded_14 = tensor.expand_shape %57 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %60 = hivm.hir.vmul ins(%arg13, %expanded_14 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
        %expanded_15 = tensor.expand_shape %52 [[0], [1, 2]] output_shape [128, 16, 16] : tensor<128x256xf16> into tensor<128x16x16xf16>
        %61 = tensor.empty() : tensor<16x128x16xf16>
        %62 = hivm.hir.vtranspose ins(%expanded_15 : tensor<128x16x16xf16>) outs(%61 : tensor<16x128x16xf16>) permutation = [1, 0, 2] -> tensor<16x128x16xf16>
        %expanded_16 = tensor.expand_shape %62 [[0], [1, 2], [3]] output_shape [16, 8, 16, 16] : tensor<16x128x16xf16> into tensor<16x8x16x16xf16>
        %alloc_17 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_17 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        %63 = bufferization.to_tensor %alloc_17 restrict writable : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%expanded_16 : tensor<16x8x16x16xf16>) outs(%alloc_17 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %64 = tensor.empty() : tensor<4x8x16x16xf32>
        %65 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%63, %53, %true, %c128, %c256, %c64 : tensor<16x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%64 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
        %alloc_18 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_18 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x64xf32, #hivm.address_space<ub>>
        %memspacecast_19 = memref.memory_space_cast %alloc_18 : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %66 = bufferization.to_tensor %memspacecast_19 restrict writable : memref<128x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%65 : tensor<4x8x16x16xf32>) outs(%alloc_18 : memref<128x64xf32, #hivm.address_space<ub>>)
        %67 = hivm.hir.vadd ins(%66, %60 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %68 = arith.addi %arg17, %c256_i32 : i32
        %69 = arith.addi %arg16, %c256_i32 : i32
        scf.yield %67, %arg14, %48, %69, %68 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
      } {fixpipe_for_mmad_result_already_inserted = true}
      %26 = hivm.hir.vln ins(%25#1 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %27 = hivm.hir.vadd ins(%25#2, %26 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %expanded = tensor.expand_shape %25#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %28 = hivm.hir.vdiv ins(%25#0, %expanded : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
      %29 = arith.muli %10, %c1024_i32 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %20 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %31]
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%32], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%27 : tensor<128xf32>) outs(%reinterpret_cast_4 : memref<128xf32, strided<[1], offset: ?>>)
      %33 = tensor.empty() : tensor<128x64xf16>
      %34 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%28 : tensor<128x64xf32>) outs(%33 : tensor<128x64xf16>) -> tensor<128x64xf16>
      hivm.hir.store ins(%34 : tensor<128x64xf16>) outs(%reinterpret_cast_3 : memref<128x64xf16, strided<[64, 1], offset: ?>>)
    }
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  return
}


}

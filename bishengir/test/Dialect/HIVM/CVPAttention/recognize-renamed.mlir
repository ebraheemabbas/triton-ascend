// RUN: bishengir-opt --verify-each %s --hivm-cvp-attention-prepare=emit-remarks=false | FileCheck %s --check-prefix=PREPARE
// RUN: bishengir-opt --verify-each %s --hivm-cvp-attention-prepare=emit-remarks=false --cv-pipelining="pipeline-mode=skew set-depth-in-unroll-mode=2 annotate-attention=true enable-lazy-loading=true" --hivm-verify-cvp-attention=emit-remarks=false | FileCheck %s --check-prefix=PIPE
// RUN: bishengir-opt --verify-each %s --hivm-cvp-attention-prepare=emit-remarks=false --cv-pipelining="pipeline-mode=skew set-depth-in-unroll-mode=2 annotate-attention=true enable-lazy-loading=true" --hivm-verify-cvp-attention="emit-remarks=false drop-annotations=true" | FileCheck %s --check-prefix=CLEAN
// RUN: bishengir-opt --verify-each %s --hivm-cvp-attention-prepare=emit-remarks=false -o %t.one
// RUN: bishengir-opt --verify-each %t.one --hivm-cvp-attention-prepare=emit-remarks=false -o %t.two
// RUN: diff %t.one %t.two
// PREPARE: hivm.cvp_attention.role = "qk"
// PREPARE: hivm.cvp_attention.role = "p32"
// PREPARE: hivm.cvp_attention.role = "pv"
// PREPARE: hivm.cvp_attention.role = "output"
// PREPARE: hivm.cvp_attention.plan = {{.*}}block_m = 128 : i64, block_n = 256 : i64
// PREPARE-SAME: head_dim = 64 : i64
// PREPARE-SAME: initial_sum = 1.000000e+00 : f32
// PIPE-COUNT-4: scope.scope
// PIPE: hivm.cvp_attention.applied = "skew"
// CLEAN: func.func
// CLEAN-NOT: hivm.cvp_attention

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
func.func @different_symbol(%renamed0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %renamed1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %renamed2: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %renamed3: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %renamed4: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %renamed5: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %renamed6: memref<?xf16> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %renamed7: i32, %renamed8: i32, %renamed9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
  %renamed10 = arith.constant 0.000000e+00 : f32
  %renamed11 = arith.constant 8192 : i32
  %renamed12 = arith.constant 128 : i32
  %renamed13 = arith.constant 65536 : i64
  %renamed14 = arith.constant 524288 : i64
  %renamed15 = arith.constant 8 : i32
  %renamed16 = arith.constant 0 : i32
  %renamed17 = arith.constant 1024 : i32
  %renamed18 = arith.constant 64 : index
  %renamed19 = arith.constant 5.000000e-01 : f32
  %renamed20 = arith.constant 256 : i32
  %renamed21 = arith.constant 0xFF800000 : f32
  %renamed22 = arith.constant 1.000000e+00 : f32
  %renamed23 = arith.constant 128 : index
  %renamed24 = arith.constant 256 : index
  %renamed25 = arith.constant true
  %renamed26 = arith.constant 28 : i32
  %0 = arith.muli %renamed7, %renamed8 : i32
  %1 = arith.muli %0, %renamed9 : i32
  annotation.mark %1 {logical_block_num} : i32
  %2 = hivm.hir.get_block_idx -> i64
  %3 = arith.trunci %2 : i64 to i32
  scf.for %renamed27 = %3 to %1 step %renamed26  : i32 {
    hivm.hir.set_ctrl false at ctrl[60]
    hivm.hir.set_ctrl true at ctrl[48]
    %4 = arith.remsi %renamed27, %renamed7 : i32
    %5 = tensor.empty() : tensor<128x64xf32>
    %6 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%renamed10 : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
    %7 = tensor.empty() : tensor<128xf32>
    %8 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%renamed21 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    %9 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%renamed22 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
    scf.for %renamed28 = %4 to %renamed11 step %renamed26  : i32 {
      %10 = arith.divsi %renamed28, %renamed15 : i32
      %11 = arith.remsi %renamed28, %renamed15 : i32
      %12 = arith.divsi %10, %renamed15 : i32
      %13 = arith.remsi %10, %renamed15 : i32
      %14 = arith.extsi %12 : i32 to i64
      %15 = arith.muli %14, %renamed14 : i64
      %16 = arith.extsi %13 : i32 to i64
      %17 = arith.muli %16, %renamed13 : i64
      %18 = arith.addi %15, %17 : i64
      %19 = arith.index_cast %18 : i64 to index
      %20 = arith.muli %11, %renamed12 : i32
      %21 = arith.maxsi %20, %renamed16 : i32
      %22 = arith.index_cast %21 : i32 to index
      %23 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %22]
      %renamed29 = memref.reinterpret_cast %renamed2 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %renamed30 = memref.reinterpret_cast %renamed6 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
      %renamed31 = memref.alloc() : memref<4x8x16x16xf16>
      hivm.hir.nd2nz {dst_continuous} ins(%renamed29 : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%renamed31 : memref<4x8x16x16xf16>)
      %24 = bufferization.to_tensor %renamed31 restrict writable : memref<4x8x16x16xf16>
      %25:5 = scf.for %renamed32 = %renamed16 to %renamed17 step %renamed20 iter_args(%renamed33 = %6, %renamed34 = %9, %renamed35 = %8, %renamed36 = %renamed16, %renamed37 = %renamed16) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %35 = arith.maxsi %renamed36, %renamed16 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %36]
        %renamed38 = memref.reinterpret_cast %renamed3 to offset: [%37], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %38 = arith.maxsi %renamed37, %renamed16 : i32
        %39 = arith.index_cast %38 : i32 to index
        %40 = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%19, %39]
        %renamed39 = memref.reinterpret_cast %renamed4 to offset: [%40], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
        %renamed40 = memref.alloc() : memref<4x16x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%renamed38 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%renamed40 : memref<4x16x16x16xf16>)
        %41 = bufferization.to_tensor %renamed40 restrict writable : memref<4x16x16x16xf16>
        %42 = tensor.empty() : tensor<16x8x16x16xf32>
        %43 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%24, %41, %renamed25, %renamed23, %renamed18, %renamed24 : tensor<4x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%42 : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
        %renamed41 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
        annotation.mark %renamed41 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x256xf32, #hivm.address_space<ub>>
        %renamed42 = memref.memory_space_cast %renamed41 : memref<128x256xf32, #hivm.address_space<ub>> to memref<128x256xf32>
        %44 = bufferization.to_tensor %renamed42 restrict writable : memref<128x256xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%43 : tensor<16x8x16x16xf32>) outs(%renamed41 : memref<128x256xf32, #hivm.address_space<ub>>)
        %45 = tensor.empty() : tensor<128x256xf32>
        %46 = hivm.hir.vmul ins(%44, %renamed19 : tensor<128x256xf32>, f32) outs(%45 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %renamed43 = tensor.expand_shape %8 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %47 = hivm.hir.vreduce <max> ins(%46 : tensor<128x256xf32>) outs(%renamed43 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %renamed44 = tensor.collapse_shape %47 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %48 = hivm.hir.vmax ins(%renamed35, %renamed44 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %renamed45 = tensor.expand_shape %48 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %49 = hivm.hir.vsub ins(%46, %renamed45 : tensor<128x256xf32>, tensor<128x1xf32>) outs(%45 : tensor<128x256xf32>) broadcast = [1] -> tensor<128x256xf32>
        %50 = hivm.hir.vexp ins(%49 : tensor<128x256xf32>) outs(%45 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %51 = tensor.empty() : tensor<128x256xf16>
        %52 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%50 : tensor<128x256xf32>) outs(%51 : tensor<128x256xf16>) -> tensor<128x256xf16>
        %renamed46 = memref.alloc() : memref<4x16x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%renamed39 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%renamed46 : memref<4x16x16x16xf16>)
        %53 = bufferization.to_tensor %renamed46 restrict writable : memref<4x16x16x16xf16>
        %54 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%renamed10 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %renamed47 = tensor.expand_shape %54 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %55 = hivm.hir.vreduce <sum> ins(%50 : tensor<128x256xf32>) outs(%renamed47 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
        %renamed48 = tensor.collapse_shape %55 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
        %56 = hivm.hir.vsub ins(%renamed35, %48 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %57 = hivm.hir.vexp ins(%56 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %58 = hivm.hir.vmul ins(%renamed34, %57 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %59 = hivm.hir.vadd ins(%58, %renamed48 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %renamed49 = tensor.expand_shape %57 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %60 = hivm.hir.vmul ins(%renamed33, %renamed49 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
        %renamed50 = tensor.expand_shape %52 [[0], [1, 2]] output_shape [128, 16, 16] : tensor<128x256xf16> into tensor<128x16x16xf16>
        %61 = tensor.empty() : tensor<16x128x16xf16>
        %62 = hivm.hir.vtranspose ins(%renamed50 : tensor<128x16x16xf16>) outs(%61 : tensor<16x128x16xf16>) permutation = [1, 0, 2] -> tensor<16x128x16xf16>
        %renamed51 = tensor.expand_shape %62 [[0], [1, 2], [3]] output_shape [16, 8, 16, 16] : tensor<16x128x16xf16> into tensor<16x8x16x16xf16>
        %renamed52 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %renamed52 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        %63 = bufferization.to_tensor %renamed52 restrict writable : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.copy ins(%renamed51 : tensor<16x8x16x16xf16>) outs(%renamed52 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
        %64 = tensor.empty() : tensor<4x8x16x16xf32>
        %65 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%63, %53, %renamed25, %renamed23, %renamed24, %renamed18 : tensor<16x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%64 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
        %renamed53 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %renamed53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x64xf32, #hivm.address_space<ub>>
        %renamed54 = memref.memory_space_cast %renamed53 : memref<128x64xf32, #hivm.address_space<ub>> to memref<128x64xf32>
        %66 = bufferization.to_tensor %renamed54 restrict writable : memref<128x64xf32>
        hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%65 : tensor<4x8x16x16xf32>) outs(%renamed53 : memref<128x64xf32, #hivm.address_space<ub>>)
        %67 = hivm.hir.vadd ins(%66, %60 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
        %68 = arith.addi %renamed37, %renamed20 : i32
        %69 = arith.addi %renamed36, %renamed20 : i32
        scf.yield %67, %59, %48, %69, %68 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
      } {fixpipe_for_mmad_result_already_inserted = true}
      %26 = hivm.hir.vln ins(%25#1 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %27 = hivm.hir.vadd ins(%25#2, %26 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      %renamed55 = tensor.expand_shape %25#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
      %28 = hivm.hir.vdiv ins(%25#0, %renamed55 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
      %29 = arith.muli %10, %renamed17 : i32
      %30 = arith.index_cast %29 : i32 to index
      %31 = arith.index_cast %20 : i32 to index
      %32 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%30, %31]
      %renamed56 = memref.reinterpret_cast %renamed5 to offset: [%32], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      hivm.hir.store ins(%27 : tensor<128xf32>) outs(%renamed56 : memref<128xf32, strided<[1], offset: ?>>)
      %33 = tensor.empty() : tensor<128x64xf16>
      %34 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%28 : tensor<128x64xf32>) outs(%33 : tensor<128x64xf16>) -> tensor<128x64xf16>
      hivm.hir.store ins(%34 : tensor<128x64xf16>) outs(%renamed30 : memref<128x64xf16, strided<[64, 1], offset: ?>>)
    }
    hivm.hir.set_ctrl true at ctrl[60]
  } {autoblockify.subloop}
  return
} loc("renamed-attention.mlir":7:3)

}

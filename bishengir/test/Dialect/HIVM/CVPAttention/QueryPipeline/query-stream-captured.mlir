// RUN: bishengir-opt %s --hivm-cvp-query-pipeline --verify-each | FileCheck %s
// RUN: bishengir-opt %s --hivm-cvp-query-pipeline --hivm-cvp-query-pipeline --verify-each | FileCheck %s
// Captured original BS128/H8/N16384/HD64, BM128/BN256 after real CVPipeline
// and key-major layout; no rewritten math or fixed address injection.
// CHECK-LABEL: func.func @_attn_fwd
// CHECK-SAME: hivm.cvp_query.stream
// CHECK: arith.ceildivsi
// CHECK-SAME: hivm.cvp_query.coordinate = "query_count"
// CHECK: hivm.cvp_query.storage = {{.*}}clock = "query"
// CHECK: %[[Q0:.*]] = memref.alloc() {{.*}}hivm.cvp_query.storage = {{.*}}bank = 0 : i32{{.*}}role = "Q"{{.*}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: %[[Q1:.*]] = memref.alloc() {{.*}}hivm.cvp_query.storage = {{.*}}bank = 1 : i32{{.*}}role = "Q"{{.*}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: scf.for
// CHECK-NOT: tensor<4x8x16x16xf16>
// CHECK: arith.divsi
// CHECK-SAME: hivm.cvp_query.coordinate = "query_ordinal"
// CHECK: arith.trunci
// CHECK-SAME: hivm.cvp_query.coordinate = "local_key"
// CHECK: scope.scope
// CHECK: %[[QSLOT:.*]] = arith.select {{.*}}, %[[Q1]], %[[Q0]] : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: scf.if
// CHECK: hivm.hir.nd2nz
// CHECK-SAME: ins({{.*}} : memref<128x64xf16
// CHECK-SAME: outs(%[[QSLOT]] : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
// CHECK: bufferization.to_tensor %[[QSLOT]] restrict : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
// CHECK: hivm.hir.mmadL1
// CHECK: scope.scope
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: hivm.hir.vreduce <max>
// CHECK: scope.scope
// CHECK: hivm.hir.mmadL1
// CHECK: scope.scope
// CHECK: scf.if
// CHECK: hivm.hir.vbrc
// CHECK: %[[RAW_O:.*]] = hivm.hir.vadd
// CHECK: scf.if {{.*}} -> (tensor<128x64xf16>, tensor<128xf32>) {
// CHECK: hivm.hir.vln
// CHECK: hivm.hir.vdiv
// CHECK: hivm.hir.store
// CHECK: hivm.hir.store
// CHECK: scope.return %[[RAW_O]],
// CHECK: physical_lifetimes_verified = false
#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.func_core_type = #hivm.func_core_type<MIX>, mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c131072_i32 = arith.constant 131072 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
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
      scf.for %arg11 = %4 to %c131072_i32 step %c28_i32  : i32 {
        %10 = arith.divsi %arg11, %c128_i32 : i32
        %11 = arith.remsi %arg11, %c128_i32 : i32
        %12 = arith.divsi %10, %c8_i32 : i32
        %13 = arith.remsi %10, %c8_i32 : i32
        %14 = arith.extsi %12 : i32 to i64
        %15 = arith.muli %14, %c8388608_i64 : i64
        %16 = arith.extsi %13 : i32 to i64
        %17 = arith.muli %16, %c1048576_i64 : i64
        %18 = arith.addi %15, %17 : i64
        %19 = arith.index_cast %18 : i64 to index
        %20 = arith.muli %11, %c128_i32 : i32
        %21 = arith.maxsi %20, %c0_i32 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = affine.apply #map()[%19, %22]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc = memref.alloc() : memref<4x8x16x16xf16>
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<4x8x16x16xf16>)
        %24 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16x16xf16>
        %25:5 = scf.for %arg12 = %c0_i32 to %c16384_i32 step %c256_i32 iter_args(%arg13 = %6, %arg14 = %9, %arg15 = %8, %arg16 = %c0_i32, %arg17 = %c0_i32) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
          %alloc_5 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
          %alloc_6 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_7 = memref.alloc() : memref<256x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_7 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x128xf32, #hivm.address_space<ub>>
          %35 = scope.scope : () -> i32 {
            %42 = arith.maxsi %arg16, %c0_i32 : i32
            %43 = arith.index_cast %42 : i32 to index
            %44 = affine.apply #map()[%19, %43]
            %reinterpret_cast_8 = memref.reinterpret_cast %arg3 to offset: [%44], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_9 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_8 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_9 : memref<4x16x16x16xf16>)
            %45 = bufferization.to_tensor %alloc_9 restrict writable : memref<4x16x16x16xf16>
            %46 = tensor.empty() : tensor<8x16x16x16xf32>
            %47 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%45, %24, %true, %c256, %c64, %c128 : tensor<4x16x16x16xf16>, tensor<4x8x16x16xf16>, i1, index, index, index) outs(%46 : tensor<8x16x16x16xf32>) -> tensor<8x16x16x16xf32>
            annotation.mark %alloc_7 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<256x128xf32, #hivm.address_space<ub>>
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%47 : tensor<8x16x16x16xf32>) outs(%alloc_7 : memref<256x128xf32, #hivm.address_space<ub>>)
            %48 = arith.addi %arg16, %c256_i32 : i32
            scope.return %48 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          %36 = bufferization.to_tensor %alloc_7 restrict writable : memref<256x128xf32, #hivm.address_space<ub>>
          %37:3 = scope.scope : () -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
            %42 = tensor.empty() : tensor<256x128xf32>
            %43 = hivm.hir.vmul ins(%36, %cst_0 : tensor<256x128xf32>, f32) outs(%42 : tensor<256x128xf32>) -> tensor<256x128xf32>
            %expanded_8 = tensor.expand_shape %8 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
            %44 = hivm.hir.vreduce <max> ins(%43 : tensor<256x128xf32>) outs(%expanded_8 : tensor<1x128xf32>) reduce_dims = [0] -> tensor<1x128xf32>
            %collapsed = tensor.collapse_shape %44 [[0, 1]] : tensor<1x128xf32> into tensor<128xf32>
            %45 = hivm.hir.vmax ins(%arg15, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %expanded_9 = tensor.expand_shape %45 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
            %46 = hivm.hir.vsub ins(%43, %expanded_9 : tensor<256x128xf32>, tensor<1x128xf32>) outs(%42 : tensor<256x128xf32>) broadcast = [0] -> tensor<256x128xf32>
            %47 = hivm.hir.vexp ins(%46 : tensor<256x128xf32>) outs(%42 : tensor<256x128xf32>) -> tensor<256x128xf32>
            %48 = tensor.empty() : tensor<256x128xf16>
            %49 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%47 : tensor<256x128xf32>) outs(%48 : tensor<256x128xf16>) -> tensor<256x128xf16>
            %50 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %expanded_10 = tensor.expand_shape %50 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
            %51 = hivm.hir.vreduce <sum> ins(%47 : tensor<256x128xf32>) outs(%expanded_10 : tensor<1x128xf32>) reduce_dims = [0] -> tensor<1x128xf32>
            %collapsed_11 = tensor.collapse_shape %51 [[0, 1]] : tensor<1x128xf32> into tensor<128xf32>
            %52 = hivm.hir.vsub ins(%arg15, %45 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %53 = hivm.hir.vexp ins(%52 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %54 = hivm.hir.vmul ins(%arg14, %53 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %55 = hivm.hir.vadd ins(%54, %collapsed_11 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
            %expanded_12 = tensor.expand_shape %49 [[0], [1, 2]] output_shape [256, 8, 16] : tensor<256x128xf16> into tensor<256x8x16xf16>
            %56 = tensor.empty() : tensor<8x256x16xf16>
            %57 = hivm.hir.vtranspose ins(%expanded_12 : tensor<256x8x16xf16>) outs(%56 : tensor<8x256x16xf16>) permutation = [1, 0, 2] -> tensor<8x256x16xf16>
            %expanded_13 = tensor.expand_shape %57 [[0], [1, 2], [3]] output_shape [8, 16, 16, 16] : tensor<8x256x16xf16> into tensor<8x16x16x16xf16>
            annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%expanded_13 : tensor<8x16x16x16xf16>) outs(%alloc_6 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy"}
            scope.return %53, %45, %55 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          %38 = bufferization.to_tensor %alloc_6 restrict writable : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %39 = scope.scope : () -> i32 {
            %42 = arith.maxsi %arg17, %c0_i32 : i32
            %43 = arith.index_cast %42 : i32 to index
            %44 = affine.apply #map()[%19, %43]
            %reinterpret_cast_8 = memref.reinterpret_cast %arg4 to offset: [%44], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_9 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_8 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_9 : memref<4x16x16x16xf16>)
            %45 = bufferization.to_tensor %alloc_9 restrict writable : memref<4x16x16x16xf16>
            %46 = tensor.empty() : tensor<4x8x16x16xf32>
            %47 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%38, %45, %true, %c128, %c256, %c64 : tensor<8x16x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%46 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
            annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x64xf32, #hivm.address_space<ub>>
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%47 : tensor<4x8x16x16xf32>) outs(%alloc_5 : memref<128x64xf32, #hivm.address_space<ub>>)
            %48 = arith.addi %arg17, %c256_i32 : i32
            scope.return %48 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          %40 = bufferization.to_tensor %alloc_5 restrict writable : memref<128x64xf32, #hivm.address_space<ub>>
          %41 = scope.scope : () -> tensor<128x64xf32> {
            %expanded_8 = tensor.expand_shape %37#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            %42 = hivm.hir.vmul ins(%arg13, %expanded_8 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
            %43 = hivm.hir.vadd ins(%40, %42 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
            scope.return %43 : tensor<128x64xf32>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          scf.yield %41, %37#2, %37#1, %35, %39 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        %26 = hivm.hir.vln ins(%25#1 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %27 = hivm.hir.vadd ins(%25#2, %26 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
        %expanded = tensor.expand_shape %25#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %28 = hivm.hir.vdiv ins(%25#0, %expanded : tensor<128x64xf32>, tensor<128x1xf32>) outs(%5 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
        %29 = arith.muli %10, %c16384_i32 : i32
        %30 = arith.index_cast %29 : i32 to index
        %31 = arith.index_cast %20 : i32 to index
        %32 = affine.apply #map1()[%30, %31]
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



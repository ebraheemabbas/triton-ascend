#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<(d0) -> (d0 * 16)>
#map3 = affine_map<()[s0] -> (s0 * 64)>
#map4 = affine_map<()[s0] -> (s0 * 4)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_backup_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-cvp-attention-vector">, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 32768 : i64, key_tiles = 128 : i64, max_state = 2 : i64, numerical_policy = "unchanged", output_state = 0 : i64, physical_lifetimes_verified = false, scale = 5.000000e-01 : f32, sum_state = 1 : i64, version = 1 : i64}], hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16777216_i64 = arith.constant 16777216 : i64
    %c2097152_i64 = arith.constant 2097152 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c262144_i32 = arith.constant 262144 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg10, %arg7 : i32
      hivm.hir.anchor {id = 3 : i64}
      hivm.hir.anchor {id = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.anchor {id = 6 : i64}
      scf.for %arg11 = %4 to %c262144_i32 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %5 = arith.divsi %arg11, %c256_i32 : i32
        %6 = arith.remsi %arg11, %c256_i32 : i32
        %7 = arith.divsi %5, %c8_i32 : i32
        %8 = arith.remsi %5, %c8_i32 : i32
        %9 = arith.extsi %7 : i32 to i64
        %10 = arith.muli %9, %c16777216_i64 : i64
        %11 = arith.extsi %8 : i32 to i64
        %12 = arith.muli %11, %c2097152_i64 : i64
        %13 = arith.addi %10, %12 : i64
        %14 = arith.index_cast %13 : i64 to index
        %15 = arith.muli %6, %c128_i32 : i32
        %16 = arith.maxsi %15, %c0_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = affine.apply #map()[%14, %17]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc = memref.alloc() : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<4x8x16x16xf16>)
        hivm.hir.anchor {id = 9 : i64}
        %19 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 10 : i64}
        %20:2 = scf.for %arg12 = %c0_i32 to %c32768_i32 step %c256_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %alloc_0 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_0 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
          %alloc_1 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_2 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.anchor {id = 12 : i64}
          %21 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 13 : i64}
            %24 = arith.maxsi %arg13, %c0_i32 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.apply #map()[%14, %25]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_4 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 15 : i64}
            %27 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 16 : i64}
            %28 = tensor.empty() : tensor<16x8x16x16xf32>
            %29 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%19, %27, %true, %c128, %c64, %c256 : tensor<4x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%28 : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%29 : tensor<16x8x16x16xf32>) outs(%alloc_2 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 19 : i64}
            %30 = arith.addi %arg13, %c256_i32 : i32
            scope.return %30 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 20 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 21 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 22 : i64}
            hivm.hir.anchor {id = 23 : i64}
            hivm.hir.anchor {id = 24 : i64}
            hivm.hir.anchor {id = 25 : i64}
            hivm.hir.anchor {id = 26 : i64}
            hivm.hir.anchor {id = 27 : i64}
            hivm.hir.anchor {id = 28 : i64}
            hivm.hir.anchor {id = 29 : i64}
            hivm.hir.anchor {id = 30 : i64}
            hivm.hir.anchor {id = 31 : i64}
            hivm.hir.anchor {id = 32 : i64}
            hivm.hir.anchor {id = 33 : i64}
            hivm.hir.anchor {id = 34 : i64}
            hivm.hir.anchor {id = 35 : i64}
            hivm.hir.anchor {id = 36 : i64}
            annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            hivm.hir.anchor {id = 38 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 39 : i64}
          %22 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.anchor {id = 40 : i64}
          %23 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 41 : i64}
            %24 = arith.maxsi %arg14, %c0_i32 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.apply #map()[%14, %25]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_4 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 43 : i64}
            %27 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 44 : i64}
            %28 = tensor.empty() : tensor<4x8x16x16xf32>
            %29 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%22, %27, %true, %c128, %c256, %c64 : tensor<16x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%28 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
            hivm.hir.anchor {id = 45 : i64}
            annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%29 : tensor<4x8x16x16xf32>) outs(%alloc_0 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 47 : i64}
            %30 = arith.addi %arg14, %c256_i32 : i32
            scope.return %30 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          hivm.hir.anchor {id = 49 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 50 : i64}
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            hivm.hir.anchor {id = 53 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 54 : i64}
          scf.yield %21, %23 : i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 55 : i64}
        hivm.hir.anchor {id = 56 : i64}
        hivm.hir.anchor {id = 57 : i64}
        hivm.hir.anchor {id = 58 : i64}
        hivm.hir.anchor {id = 59 : i64}
        hivm.hir.anchor {id = 60 : i64}
        hivm.hir.anchor {id = 61 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 62 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 63 : i64}
    return
  }
  func.func @_attn_fwd_backup_mix_aiv_fused_3_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = arith.subf %0, %1 : vector<64xf32>
    %3 = math.exp %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %5 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %6 = vector.transfer_read %arg3[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %7 = arith.mulf %5, %3 : vector<64xf32>
    %8 = arith.addf %7, %6 : vector<64xf32>
    %9 = vector.transfer_write %8, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %9, %4 : tensor<64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_fused_4_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %1 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %2 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %3 = vector.transfer_read %extracted_slice[%c0], %cst {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %4 = arith.mulf %2, %3 : vector<1x64xf32>
      %5 = arith.addf %1, %4 : vector<1x64xf32>
      %6 = vector.transfer_write %5, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_fused_5_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_fused_6_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x64xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [1, 64] [1, 1] : tensor<64x64xf16> to tensor<1x64xf16>
      %1 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %2 = vector.transfer_read %extracted_slice[%c0], %cst {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %3 = arith.divf %1, %2 : vector<1x64xf32>
      %4 = arith.truncf %3 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %5 = vector.transfer_write %4, %extracted_slice_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
      %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0] [1, 64] [1, 1] : tensor<1x64xf16> into tensor<64x64xf16>
      scf.yield %inserted_slice : tensor<64x64xf16>
    }
    return %0 : tensor<64x64xf16>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_0(%arg0: tensor<1x64xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x256xf32>) -> (tensor<64xf32>, tensor<64x256xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<64xf32>, tensor<64x256xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %1:2 = scf.for %arg7 = %c0 to %c256 step %c64 iter_args(%arg8 = %arg6, %arg9 = %cst_0) -> (tensor<64x256xf32>, vector<1x64xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %extracted_slice_3 = tensor.extract_slice %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %5 = vector.transfer_read %extracted_slice_2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
        %6 = arith.mulf %5, %cst : vector<1x64xf32>
        %7 = vector.transfer_write %6, %extracted_slice_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
        %8 = arith.maximumf %6, %arg9 {reductionOp} : vector<1x64xf32>
        %inserted_slice_4 = tensor.insert_slice %7 into %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x256xf32>
        scf.yield %inserted_slice_4, %8 : tensor<64x256xf32>, vector<1x64xf32>
      } {reductionLoop}
      %2 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %3 = vector.multi_reduction <maximumf>, %1#1, %2 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %4 = vector.transfer_write %3, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %1#0 : tensor<64xf32>, tensor<64x256xf32>
    }
    return %0#0, %0#1 : tensor<64xf32>, tensor<64x256xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = arith.maximumf %0, %1 : vector<64xf32>
    %3 = vector.transfer_write %2, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_2(%arg0: tensor<1x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64xf32>, %arg4: tensor<16x64x16xf16>) -> (tensor<64xf32>, tensor<16x64x16xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0:2 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<64xf32>, tensor<16x64x16xf16>) {
      %extracted_slice = tensor.extract_slice %arg6[%arg5] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg5] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %1 = vector.transfer_read %extracted_slice_1[%c0], %cst_0 {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %2:2 = scf.for %arg8 = %c0 to %c16 step %c4 iter_args(%arg9 = %arg7, %arg10 = %cst) -> (tensor<16x64x16xf16>, vector<1x64xf32>) {
        %6 = affine.apply #map2(%arg8)
        %extracted_slice_2 = tensor.extract_slice %arg2[%arg5, %6] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %7 = vector.transfer_read %extracted_slice_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
        %8 = arith.subf %7, %1 : vector<1x64xf32>
        %9 = math.exp %8 : vector<1x64xf32>
        %10 = arith.addf %9, %arg10 {reductionOp} : vector<1x64xf32>
        %11 = arith.truncf %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
        %extracted_slice_3 = tensor.extract_slice %arg9[%arg8, %arg5, 0] [4, 1, 16] [1, 1, 1] : tensor<16x64x16xf16> to tensor<4x1x16xf16>
        %12 = vector.shape_cast %11 : vector<1x64xf16> to vector<4x1x16xf16>
        %13 = vector.transfer_write %12, %extracted_slice_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
        %inserted_slice_4 = tensor.insert_slice %13 into %arg9[%arg8, %arg5, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<16x64x16xf16>
        scf.yield %inserted_slice_4, %10 : tensor<16x64x16xf16>, vector<1x64xf32>
      } {reductionLoop, unroll_for_vsstb}
      %3 = vector.transfer_read %extracted_slice[%c0], %cst_0 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %4 = vector.multi_reduction <add>, %2#1, %3 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %5 = vector.transfer_write %4, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg5] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %2#0 : tensor<64xf32>, tensor<16x64x16xf16>
    }
    return %0#0, %0#1 : tensor<64xf32>, tensor<16x64x16xf16>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_3(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_4(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %arg0) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg2[%arg1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %1 = vector.transfer_write %cst, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %1 into %arg2[%arg1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv_outlined_vf_5(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_backup_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-cvp-attention-vector">, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 32768 : i64, key_tiles = 128 : i64, max_state = 2 : i64, numerical_policy = "unchanged", output_state = 0 : i64, physical_lifetimes_verified = false, scale = 5.000000e-01 : f32, sum_state = 1 : i64, version = 1 : i64}], hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c262144_i32 = arith.constant 262144 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2097152_i64 = arith.constant 2097152 : i64
    %c16777216_i64 = arith.constant 16777216 : i64
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = tensor.empty() : tensor<64xf32>
    %6 = call @_attn_fwd_backup_mix_aiv_outlined_vf_5(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
    scf.for %arg10 = %c0 to %c2 step %c1 {
      %7 = affine.apply #map3()[%arg10]
      %8 = affine.apply #map4()[%arg10]
      hivm.hir.anchor {id = 0 : i64}
      annotation.mark %1 {logical_block_num} : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.anchor {id = 1 : i64}
      scf.for %arg11 = %3 to %1 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 2 : i64}
        hivm.hir.set_ctrl false at ctrl[60]
        hivm.hir.set_ctrl true at ctrl[48]
        %9 = arith.remsi %arg11, %arg7 : i32
        hivm.hir.anchor {id = 3 : i64}
        hivm.hir.anchor {id = 4 : i64}
        hivm.hir.anchor {id = 5 : i64}
        hivm.hir.anchor {id = 6 : i64}
        scf.for %arg12 = %9 to %c262144_i32 step %c28_i32  : i32 {
          hivm.hir.anchor {id = 7 : i64}
          %10 = arith.divsi %arg12, %c256_i32 : i32
          %11 = arith.remsi %arg12, %c256_i32 : i32
          %12 = arith.divsi %10, %c8_i32 : i32
          %13 = arith.remsi %10, %c8_i32 : i32
          %14 = arith.extsi %12 : i32 to i64
          %15 = arith.muli %14, %c16777216_i64 : i64
          %16 = arith.extsi %13 : i32 to i64
          %17 = arith.muli %16, %c2097152_i64 : i64
          %18 = arith.addi %15, %17 : i64
          %19 = arith.index_cast %18 : i64 to index
          %20 = arith.muli %11, %c128_i32 : i32
          %21 = arith.maxsi %20, %c0_i32 : i32
          %22 = arith.index_cast %21 : i32 to index
          %23 = affine.apply #map()[%19, %22]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %subview = memref.subview %reinterpret_cast[%7, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          hivm.hir.anchor {id = 8 : i64}
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
          %24 = func.call @_attn_fwd_backup_mix_aiv_outlined_vf_3(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
          %25 = func.call @_attn_fwd_backup_mix_aiv_outlined_vf_4(%4) {hivm.vector_function, no_inline} : (tensor<64x64xf32>) -> tensor<64x64xf32>
          %26:3 = scf.for %arg13 = %c0_i32 to %c32768_i32 step %c256_i32 iter_args(%arg14 = %25, %arg15 = %24, %arg16 = %6) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
            hivm.hir.anchor {id = 11 : i64}
            %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
            annotation.mark %alloc {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
            %alloc_2 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            %subview_3 = memref.subview %alloc_2[0, %8, 0, 0] [16, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            %alloc_4 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 12 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 13 : i64}
              hivm.hir.anchor {id = 14 : i64}
              hivm.hir.anchor {id = 15 : i64}
              hivm.hir.anchor {id = 16 : i64}
              hivm.hir.anchor {id = 17 : i64}
              annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 18 : i64}
              hivm.hir.anchor {id = 19 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
            hivm.hir.anchor {id = 20 : i64}
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 21 : i64}
            %35:3 = scope.scope : () -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
              hivm.hir.anchor {id = 22 : i64}
              hivm.hir.anchor {id = 23 : i64}
              hivm.hir.anchor {id = 24 : i64}
              %38 = tensor.empty() : tensor<64x256xf32>
              %39 = tensor.empty() : tensor<1x64xf32>
              %40:2 = func.call @_attn_fwd_backup_mix_aiv_outlined_vf_0(%39, %34, %5, %38) {hivm.vector_function, no_inline} : (tensor<1x64xf32>, tensor<64x256xf32>, tensor<64xf32>, tensor<64x256xf32>) -> (tensor<64xf32>, tensor<64x256xf32>)
              hivm.hir.anchor {id = 25 : i64}
              hivm.hir.anchor {id = 26 : i64}
              hivm.hir.anchor {id = 27 : i64}
              hivm.hir.anchor {id = 28 : i64}
              hivm.hir.anchor {id = 29 : i64}
              hivm.hir.anchor {id = 30 : i64}
              %41 = func.call @_attn_fwd_backup_mix_aiv_outlined_vf_1(%arg16, %40#0, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
              hivm.hir.anchor {id = 31 : i64}
              hivm.hir.anchor {id = 32 : i64}
              hivm.hir.anchor {id = 33 : i64}
              hivm.hir.anchor {id = 34 : i64}
              hivm.hir.anchor {id = 35 : i64}
              hivm.hir.anchor {id = 36 : i64}
              %42 = tensor.empty() : tensor<16x64x16xf16>
              %43:2 = func.call @_attn_fwd_backup_mix_aiv_outlined_vf_2(%39, %41, %40#1, %5, %42) {hivm.vector_function, no_inline} : (tensor<1x64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64xf32>, tensor<16x64x16xf16>) -> (tensor<64xf32>, tensor<16x64x16xf16>)
              %44:2 = func.call @_attn_fwd_backup_mix_aiv_fused_3_outlined_vf_0(%arg15, %arg16, %41, %43#0, %5, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
              %expanded = tensor.expand_shape %43#1 [[0], [1, 2], [3]] output_shape [16, 4, 16, 16] : tensor<16x64x16xf16> into tensor<16x4x16x16xf16>
              annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
              hivm.hir.anchor {id = 37 : i64}
              hivm.hir.copy ins(%expanded : tensor<16x4x16x16xf16>) outs(%subview_3 : memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
              hivm.hir.anchor {id = 38 : i64}
              scope.return %44#1, %41, %44#0 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
            hivm.hir.anchor {id = 39 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
            hivm.hir.anchor {id = 40 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 41 : i64}
              hivm.hir.anchor {id = 42 : i64}
              hivm.hir.anchor {id = 43 : i64}
              hivm.hir.anchor {id = 44 : i64}
              hivm.hir.anchor {id = 45 : i64}
              annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 46 : i64}
              hivm.hir.anchor {id = 47 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
            hivm.hir.anchor {id = 48 : i64}
            %36 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 49 : i64}
            %37 = scope.scope : () -> tensor<64x64xf32> {
              hivm.hir.anchor {id = 50 : i64}
              hivm.hir.anchor {id = 51 : i64}
              hivm.hir.anchor {id = 52 : i64}
              %38 = func.call @_attn_fwd_backup_mix_aiv_fused_4_outlined_vf_0(%35#0, %36, %arg14, %4) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              hivm.hir.anchor {id = 53 : i64}
              scope.return %38 : tensor<64x64xf32>
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
            hivm.hir.anchor {id = 54 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
            scf.yield %37, %35#2, %35#1 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
          } {fixpipe_for_mmad_result_already_inserted = true}
          hivm.hir.anchor {id = 55 : i64}
          hivm.hir.anchor {id = 56 : i64}
          %27 = func.call @_attn_fwd_backup_mix_aiv_fused_5_outlined_vf_0(%26#2, %26#1, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
          hivm.hir.anchor {id = 57 : i64}
          hivm.hir.anchor {id = 58 : i64}
          %28 = arith.muli %10, %c32768_i32 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.index_cast %20 : i32 to index
          %31 = affine.apply #map5()[%29, %30]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %subview_1 = memref.subview %reinterpret_cast_0[%7] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
          hivm.hir.store ins(%27 : tensor<64xf32>) outs(%subview_1 : memref<64xf32, strided<[1], offset: ?>>) {tiled_op}
          hivm.hir.anchor {id = 59 : i64}
          %32 = tensor.empty() : tensor<64x64xf16>
          %33 = func.call @_attn_fwd_backup_mix_aiv_fused_6_outlined_vf_0(%26#1, %26#0, %32) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf16>) -> tensor<64x64xf16>
          hivm.hir.anchor {id = 60 : i64}
          hivm.hir.store ins(%33 : tensor<64x64xf16>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>>) {tiled_op}
          hivm.hir.anchor {id = 61 : i64}
        }
        hivm.hir.set_ctrl true at ctrl[60]
        hivm.hir.anchor {id = 62 : i64}
      } {autoblockify.subloop}
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.anchor {id = 63 : i64}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 32768 : i64, key_tiles = 128 : i64, max_state = 2 : i64, numerical_policy = "unchanged", output_state = 0 : i64, physical_lifetimes_verified = false, scale = 5.000000e-01 : f32, sum_state = 1 : i64, version = 1 : i64}], hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16777216_i64 = arith.constant 16777216 : i64
    %c2097152_i64 = arith.constant 2097152 : i64
    %c128_i32 = arith.constant 128 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c262144_i32 = arith.constant 262144 : i32
    hivm.hir.anchor {id = 0 : i64}
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg10, %arg7 : i32
      hivm.hir.anchor {id = 3 : i64}
      hivm.hir.anchor {id = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.anchor {id = 6 : i64}
      scf.for %arg11 = %4 to %c262144_i32 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %5 = arith.divsi %arg11, %c256_i32 : i32
        %6 = arith.remsi %arg11, %c256_i32 : i32
        %7 = arith.divsi %5, %c8_i32 : i32
        %8 = arith.remsi %5, %c8_i32 : i32
        %9 = arith.extsi %7 : i32 to i64
        %10 = arith.muli %9, %c16777216_i64 : i64
        %11 = arith.extsi %8 : i32 to i64
        %12 = arith.muli %11, %c2097152_i64 : i64
        %13 = arith.addi %10, %12 : i64
        %14 = arith.index_cast %13 : i64 to index
        %15 = arith.muli %6, %c128_i32 : i32
        %16 = arith.maxsi %15, %c0_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = affine.apply #map()[%14, %17]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %alloc = memref.alloc() : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc : memref<4x8x16x16xf16>)
        hivm.hir.anchor {id = 9 : i64}
        %19 = bufferization.to_tensor %alloc restrict writable : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 10 : i64}
        %20:2 = scf.for %arg12 = %c0_i32 to %c32768_i32 step %c256_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %alloc_0 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_0 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
          %alloc_1 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_2 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.anchor {id = 12 : i64}
          %21 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 13 : i64}
            %24 = arith.maxsi %arg13, %c0_i32 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.apply #map()[%14, %25]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_4 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 15 : i64}
            %27 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 16 : i64}
            %28 = tensor.empty() : tensor<16x8x16x16xf32>
            %29 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%19, %27, %true, %c128, %c64, %c256 : tensor<4x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%28 : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%29 : tensor<16x8x16x16xf32>) outs(%alloc_2 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 19 : i64}
            %30 = arith.addi %arg13, %c256_i32 : i32
            scope.return %30 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 20 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 21 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 22 : i64}
            hivm.hir.anchor {id = 23 : i64}
            hivm.hir.anchor {id = 24 : i64}
            hivm.hir.anchor {id = 25 : i64}
            hivm.hir.anchor {id = 26 : i64}
            hivm.hir.anchor {id = 27 : i64}
            hivm.hir.anchor {id = 28 : i64}
            hivm.hir.anchor {id = 29 : i64}
            hivm.hir.anchor {id = 30 : i64}
            hivm.hir.anchor {id = 31 : i64}
            hivm.hir.anchor {id = 32 : i64}
            hivm.hir.anchor {id = 33 : i64}
            hivm.hir.anchor {id = 34 : i64}
            hivm.hir.anchor {id = 35 : i64}
            hivm.hir.anchor {id = 36 : i64}
            annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            hivm.hir.anchor {id = 38 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 39 : i64}
          %22 = bufferization.to_tensor %alloc_1 restrict writable : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.anchor {id = 40 : i64}
          %23 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 41 : i64}
            %24 = arith.maxsi %arg14, %c0_i32 : i32
            %25 = arith.index_cast %24 : i32 to index
            %26 = affine.apply #map()[%14, %25]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%26], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %alloc_4 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_4 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 43 : i64}
            %27 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 44 : i64}
            %28 = tensor.empty() : tensor<4x8x16x16xf32>
            %29 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%22, %27, %true, %c128, %c256, %c64 : tensor<16x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%28 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
            hivm.hir.anchor {id = 45 : i64}
            annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%29 : tensor<4x8x16x16xf32>) outs(%alloc_0 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 47 : i64}
            %30 = arith.addi %arg14, %c256_i32 : i32
            scope.return %30 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          hivm.hir.anchor {id = 49 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 50 : i64}
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            hivm.hir.anchor {id = 53 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 54 : i64}
          scf.yield %21, %23 : i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 55 : i64}
        hivm.hir.anchor {id = 56 : i64}
        hivm.hir.anchor {id = 57 : i64}
        hivm.hir.anchor {id = 58 : i64}
        hivm.hir.anchor {id = 59 : i64}
        hivm.hir.anchor {id = 60 : i64}
        hivm.hir.anchor {id = 61 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 62 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 63 : i64}
    return
  }
  func.func @_attn_fwd_mix_aiv_fused_3_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = arith.subf %0, %1 : vector<64xf32>
    %3 = math.exp %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %5 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %6 = vector.transfer_read %arg3[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %7 = arith.mulf %5, %3 : vector<64xf32>
    %8 = arith.addf %7, %6 : vector<64xf32>
    %9 = vector.transfer_write %8, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %9, %4 : tensor<64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_fused_4_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg3) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg2[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_2 = tensor.extract_slice %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %1 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %2 = vector.transfer_read %extracted_slice_1[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %3 = vector.transfer_read %extracted_slice[%c0], %cst {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %4 = arith.mulf %2, %3 : vector<1x64xf32>
      %5 = arith.addf %1, %4 : vector<1x64xf32>
      %6 = vector.transfer_write %5, %extracted_slice_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %6 into %arg5[%arg4, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_mix_aiv_fused_5_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg3 = %c0 to %c64 step %c1 iter_args(%arg4 = %arg2) -> (tensor<64x64xf16>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg3] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_0 = tensor.extract_slice %arg1[%arg3, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %extracted_slice_1 = tensor.extract_slice %arg4[%arg3, 0] [1, 64] [1, 1] : tensor<64x64xf16> to tensor<1x64xf16>
      %1 = vector.transfer_read %extracted_slice_0[%c0, %c0], %cst {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %2 = vector.transfer_read %extracted_slice[%c0], %cst {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %3 = arith.divf %1, %2 : vector<1x64xf32>
      %4 = arith.truncf %3 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %5 = vector.transfer_write %4, %extracted_slice_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, tensor<1x64xf16>
      %inserted_slice = tensor.insert_slice %5 into %arg4[%arg3, 0] [1, 64] [1, 1] : tensor<1x64xf16> into tensor<64x64xf16>
      scf.yield %inserted_slice : tensor<64x64xf16>
    }
    return %0 : tensor<64x64xf16>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_0(%arg0: tensor<1x64xf32>, %arg1: tensor<64x256xf32>, %arg2: tensor<64xf32>, %arg3: tensor<64x256xf32>) -> (tensor<64xf32>, tensor<64x256xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_0 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0:2 = scf.for %arg4 = %c0 to %c64 step %c1 iter_args(%arg5 = %arg2, %arg6 = %arg3) -> (tensor<64xf32>, tensor<64x256xf32>) {
      %extracted_slice = tensor.extract_slice %arg5[%arg4] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %1:2 = scf.for %arg7 = %c0 to %c256 step %c64 iter_args(%arg8 = %arg6, %arg9 = %cst_0) -> (tensor<64x256xf32>, vector<1x64xf32>) {
        %extracted_slice_2 = tensor.extract_slice %arg1[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %extracted_slice_3 = tensor.extract_slice %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %5 = vector.transfer_read %extracted_slice_2[%c0, %c0], %cst_1 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
        %6 = arith.mulf %5, %cst : vector<1x64xf32>
        %7 = vector.transfer_write %6, %extracted_slice_3[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
        %8 = arith.maximumf %6, %arg9 {reductionOp} : vector<1x64xf32>
        %inserted_slice_4 = tensor.insert_slice %7 into %arg8[%arg4, %arg7] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x256xf32>
        scf.yield %inserted_slice_4, %8 : tensor<64x256xf32>, vector<1x64xf32>
      } {reductionLoop}
      %2 = vector.transfer_read %extracted_slice[%c0], %cst_1 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %3 = vector.multi_reduction <maximumf>, %1#1, %2 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %4 = vector.transfer_write %3, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %4 into %arg5[%arg4] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %1#0 : tensor<64xf32>, tensor<64x256xf32>
    }
    return %0#0, %0#1 : tensor<64xf32>, tensor<64x256xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_1(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = arith.maximumf %0, %1 : vector<64xf32>
    %3 = vector.transfer_write %2, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %3 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_2(%arg0: tensor<1x64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64x256xf32>, %arg3: tensor<64xf32>, %arg4: tensor<16x64x16xf16>) -> (tensor<64xf32>, tensor<16x64x16xf16>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %0:2 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<64xf32>, tensor<16x64x16xf16>) {
      %extracted_slice = tensor.extract_slice %arg6[%arg5] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %extracted_slice_1 = tensor.extract_slice %arg1[%arg5] [1] [1] : tensor<64xf32> to tensor<1xf32>
      %1 = vector.transfer_read %extracted_slice_1[%c0], %cst_0 {in_bounds = [true, true], permutation_map = #map1} : tensor<1xf32>, vector<1x64xf32>
      %2:2 = scf.for %arg8 = %c0 to %c16 step %c4 iter_args(%arg9 = %arg7, %arg10 = %cst) -> (tensor<16x64x16xf16>, vector<1x64xf32>) {
        %6 = affine.apply #map2(%arg8)
        %extracted_slice_2 = tensor.extract_slice %arg2[%arg5, %6] [1, 64] [1, 1] : tensor<64x256xf32> to tensor<1x64xf32>
        %7 = vector.transfer_read %extracted_slice_2[%c0, %c0], %cst_0 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
        %8 = arith.subf %7, %1 : vector<1x64xf32>
        %9 = math.exp %8 : vector<1x64xf32>
        %10 = arith.addf %9, %arg10 {reductionOp} : vector<1x64xf32>
        %11 = arith.truncf %9 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
        %extracted_slice_3 = tensor.extract_slice %arg9[%arg8, %arg5, 0] [4, 1, 16] [1, 1, 1] : tensor<16x64x16xf16> to tensor<4x1x16xf16>
        %12 = vector.shape_cast %11 : vector<1x64xf16> to vector<4x1x16xf16>
        %13 = vector.transfer_write %12, %extracted_slice_3[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, tensor<4x1x16xf16>
        %inserted_slice_4 = tensor.insert_slice %13 into %arg9[%arg8, %arg5, 0] [4, 1, 16] [1, 1, 1] : tensor<4x1x16xf16> into tensor<16x64x16xf16>
        scf.yield %inserted_slice_4, %10 : tensor<16x64x16xf16>, vector<1x64xf32>
      } {reductionLoop, unroll_for_vsstb}
      %3 = vector.transfer_read %extracted_slice[%c0], %cst_0 {in_bounds = [true]} : tensor<1xf32>, vector<1xf32>
      %4 = vector.multi_reduction <add>, %2#1, %3 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      %5 = vector.transfer_write %4, %extracted_slice[%c0] {in_bounds = [true]} : vector<1xf32>, tensor<1xf32>
      %inserted_slice = tensor.insert_slice %5 into %arg6[%arg5] [1] [1] : tensor<1xf32> into tensor<64xf32>
      scf.yield %inserted_slice, %2#0 : tensor<64xf32>, tensor<16x64x16xf16>
    }
    return %0#0, %0#1 : tensor<64xf32>, tensor<16x64x16xf16>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_4(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg1 = %c0 to %c64 step %c1 iter_args(%arg2 = %arg0) -> (tensor<64x64xf32>) {
      %extracted_slice = tensor.extract_slice %arg2[%arg1, 0] [1, 64] [1, 1] : tensor<64x64xf32> to tensor<1x64xf32>
      %1 = vector.transfer_write %cst, %extracted_slice[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, tensor<1x64xf32>
      %inserted_slice = tensor.insert_slice %1 into %arg2[%arg1, 0] [1, 64] [1, 1] : tensor<1x64xf32> into tensor<64x64xf32>
      scf.yield %inserted_slice : tensor<64x64xf32>
    }
    return %0 : tensor<64x64xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_5(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 32768 : i64, key_tiles = 128 : i64, max_state = 2 : i64, numerical_policy = "unchanged", output_state = 0 : i64, physical_lifetimes_verified = false, scale = 5.000000e-01 : f32, sum_state = 1 : i64, version = 1 : i64}], hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c262144_i32 = arith.constant 262144 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c128_i32 = arith.constant 128 : i32
    %c2097152_i64 = arith.constant 2097152 : i64
    %c16777216_i64 = arith.constant 16777216 : i64
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = tensor.empty() : tensor<64xf32>
    %6 = call @_attn_fwd_mix_aiv_outlined_vf_5(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
    scf.for %arg10 = %c0 to %c2 step %c1 {
      %7 = affine.apply #map3()[%arg10]
      %8 = affine.apply #map4()[%arg10]
      hivm.hir.anchor {id = 0 : i64}
      annotation.mark %1 {logical_block_num} : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.anchor {id = 1 : i64}
      scf.for %arg11 = %3 to %1 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 2 : i64}
        hivm.hir.set_ctrl false at ctrl[60]
        hivm.hir.set_ctrl true at ctrl[48]
        %9 = arith.remsi %arg11, %arg7 : i32
        hivm.hir.anchor {id = 3 : i64}
        hivm.hir.anchor {id = 4 : i64}
        hivm.hir.anchor {id = 5 : i64}
        hivm.hir.anchor {id = 6 : i64}
        scf.for %arg12 = %9 to %c262144_i32 step %c28_i32  : i32 {
          hivm.hir.anchor {id = 7 : i64}
          %10 = arith.divsi %arg12, %c256_i32 : i32
          %11 = arith.remsi %arg12, %c256_i32 : i32
          %12 = arith.divsi %10, %c8_i32 : i32
          %13 = arith.remsi %10, %c8_i32 : i32
          %14 = arith.extsi %12 : i32 to i64
          %15 = arith.muli %14, %c16777216_i64 : i64
          %16 = arith.extsi %13 : i32 to i64
          %17 = arith.muli %16, %c2097152_i64 : i64
          %18 = arith.addi %15, %17 : i64
          %19 = arith.index_cast %18 : i64 to index
          %20 = arith.muli %11, %c128_i32 : i32
          %21 = arith.maxsi %20, %c0_i32 : i32
          %22 = arith.index_cast %21 : i32 to index
          %23 = affine.apply #map()[%19, %22]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%23], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %subview = memref.subview %reinterpret_cast[%7, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
          hivm.hir.anchor {id = 8 : i64}
          hivm.hir.anchor {id = 9 : i64}
          hivm.hir.anchor {id = 10 : i64}
          %24 = func.call @_attn_fwd_mix_aiv_outlined_vf_3(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
          %25 = func.call @_attn_fwd_mix_aiv_outlined_vf_4(%4) {hivm.vector_function, no_inline} : (tensor<64x64xf32>) -> tensor<64x64xf32>
          %26:3 = scf.for %arg13 = %c0_i32 to %c32768_i32 step %c256_i32 iter_args(%arg14 = %25, %arg15 = %24, %arg16 = %6) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>)  : i32 {
            hivm.hir.anchor {id = 11 : i64}
            %alloc = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
            annotation.mark %alloc {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
            %alloc_2 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            %subview_3 = memref.subview %alloc_2[0, %8, 0, 0] [16, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            %alloc_4 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 12 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 13 : i64}
              hivm.hir.anchor {id = 14 : i64}
              hivm.hir.anchor {id = 15 : i64}
              hivm.hir.anchor {id = 16 : i64}
              hivm.hir.anchor {id = 17 : i64}
              annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 18 : i64}
              hivm.hir.anchor {id = 19 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
            hivm.hir.anchor {id = 20 : i64}
            %34 = bufferization.to_tensor %alloc_4 restrict writable : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 21 : i64}
            %35:3 = scope.scope : () -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
              hivm.hir.anchor {id = 22 : i64}
              hivm.hir.anchor {id = 23 : i64}
              hivm.hir.anchor {id = 24 : i64}
              %38 = tensor.empty() : tensor<64x256xf32>
              %39 = tensor.empty() : tensor<1x64xf32>
              %40:2 = func.call @_attn_fwd_mix_aiv_outlined_vf_0(%39, %34, %5, %38) {hivm.vector_function, no_inline} : (tensor<1x64xf32>, tensor<64x256xf32>, tensor<64xf32>, tensor<64x256xf32>) -> (tensor<64xf32>, tensor<64x256xf32>)
              hivm.hir.anchor {id = 25 : i64}
              hivm.hir.anchor {id = 26 : i64}
              hivm.hir.anchor {id = 27 : i64}
              hivm.hir.anchor {id = 28 : i64}
              hivm.hir.anchor {id = 29 : i64}
              hivm.hir.anchor {id = 30 : i64}
              %41 = func.call @_attn_fwd_mix_aiv_outlined_vf_1(%arg16, %40#0, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
              hivm.hir.anchor {id = 31 : i64}
              hivm.hir.anchor {id = 32 : i64}
              hivm.hir.anchor {id = 33 : i64}
              hivm.hir.anchor {id = 34 : i64}
              hivm.hir.anchor {id = 35 : i64}
              hivm.hir.anchor {id = 36 : i64}
              %42 = tensor.empty() : tensor<16x64x16xf16>
              %43:2 = func.call @_attn_fwd_mix_aiv_outlined_vf_2(%39, %41, %40#1, %5, %42) {hivm.vector_function, no_inline} : (tensor<1x64xf32>, tensor<64xf32>, tensor<64x256xf32>, tensor<64xf32>, tensor<16x64x16xf16>) -> (tensor<64xf32>, tensor<16x64x16xf16>)
              %44:2 = func.call @_attn_fwd_mix_aiv_fused_3_outlined_vf_0(%arg15, %arg16, %41, %43#0, %5, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>)
              %expanded = tensor.expand_shape %43#1 [[0], [1, 2], [3]] output_shape [16, 4, 16, 16] : tensor<16x64x16xf16> into tensor<16x4x16x16xf16>
              annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
              hivm.hir.anchor {id = 37 : i64}
              hivm.hir.copy ins(%expanded : tensor<16x4x16x16xf16>) outs(%subview_3 : memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
              hivm.hir.anchor {id = 38 : i64}
              scope.return %44#1, %41, %44#0 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
            hivm.hir.anchor {id = 39 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
            hivm.hir.anchor {id = 40 : i64}
            scope.scope : () -> () {
              hivm.hir.anchor {id = 41 : i64}
              hivm.hir.anchor {id = 42 : i64}
              hivm.hir.anchor {id = 43 : i64}
              hivm.hir.anchor {id = 44 : i64}
              hivm.hir.anchor {id = 45 : i64}
              annotation.mark %alloc {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
              hivm.hir.anchor {id = 46 : i64}
              hivm.hir.anchor {id = 47 : i64}
              scope.return
            } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
            hivm.hir.anchor {id = 48 : i64}
            %36 = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
            hivm.hir.anchor {id = 49 : i64}
            %37 = scope.scope : () -> tensor<64x64xf32> {
              hivm.hir.anchor {id = 50 : i64}
              hivm.hir.anchor {id = 51 : i64}
              hivm.hir.anchor {id = 52 : i64}
              %38 = func.call @_attn_fwd_mix_aiv_fused_4_outlined_vf_0(%35#0, %36, %arg14, %4) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
              hivm.hir.anchor {id = 53 : i64}
              scope.return %38 : tensor<64x64xf32>
            } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
            hivm.hir.anchor {id = 54 : i64}
            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
            scf.yield %37, %35#2, %35#1 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>
          } {fixpipe_for_mmad_result_already_inserted = true}
          hivm.hir.anchor {id = 55 : i64}
          hivm.hir.anchor {id = 56 : i64}
          %27 = func.call @_attn_fwd_mix_aiv_fused_5_outlined_vf_0(%26#2, %26#1, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
          hivm.hir.anchor {id = 57 : i64}
          hivm.hir.anchor {id = 58 : i64}
          %28 = arith.muli %10, %c32768_i32 : i32
          %29 = arith.index_cast %28 : i32 to index
          %30 = arith.index_cast %20 : i32 to index
          %31 = affine.apply #map5()[%29, %30]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
          %subview_1 = memref.subview %reinterpret_cast_0[%7] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
          hivm.hir.store ins(%27 : tensor<64xf32>) outs(%subview_1 : memref<64xf32, strided<[1], offset: ?>>) {tiled_op}
          hivm.hir.anchor {id = 59 : i64}
          %32 = tensor.empty() : tensor<64x64xf16>
          %33 = func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%26#1, %26#0, %32) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf16>) -> tensor<64x64xf16>
          hivm.hir.anchor {id = 60 : i64}
          hivm.hir.store ins(%33 : tensor<64x64xf16>) outs(%subview : memref<64x64xf16, strided<[64, 1], offset: ?>>) {tiled_op}
          hivm.hir.anchor {id = 61 : i64}
        }
        hivm.hir.set_ctrl true at ctrl[60]
        hivm.hir.anchor {id = 62 : i64}
      } {autoblockify.subloop}
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.anchor {id = 63 : i64}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}


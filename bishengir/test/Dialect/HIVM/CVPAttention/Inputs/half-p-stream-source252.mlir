#map = affine_map<()[s0, s1] -> (s0 + s1 * 128)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<()[s0] -> (s0 + 64)>
#map4 = affine_map<()[s0] -> (s0 + 128)>
#map5 = affine_map<()[s0] -> (s0 + 192)>
#map6 = affine_map<()[s0] -> (s0 * 64)>
#map7 = affine_map<()[s0] -> (s0 * 4)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.dense_p_layouts = 1 : i64, hivm.cvp_attention.dense_two_chain_loops = 2 : i64, hivm.cvp_attention.first_output_specializations = 1 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_mix_aiv_fused_5_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x128xf32, #hivm.address_space<ub>>, %arg5: memref<64x128xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    scf.for %arg6 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg6] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      scf.for %arg7 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg4[%arg6, %arg7] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg5[%arg6, %arg7] [1, 64] [1, 1] : memref<64x128xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %5 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %6 = arith.divf %5, %4 : vector<1x64xf32>
        %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
        vector.transfer_write %7, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c256_i32 = arith.constant 256 : i32
    %c128 = arith.constant 128 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
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
      scf.for %arg11 = %4 to %c8192_i32 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %5 = arith.divsi %arg11, %c8_i32 : i32
        %6 = arith.remsi %arg11, %c8_i32 : i32
        %7 = arith.divsi %5, %c8_i32 : i32
        %8 = arith.remsi %5, %c8_i32 : i32
        %9 = arith.extsi %7 : i32 to i64
        %10 = arith.muli %9, %c1048576_i64 : i64
        %11 = arith.extsi %8 : i32 to i64
        %12 = arith.muli %11, %c131072_i64 : i64
        %13 = arith.addi %10, %12 : i64
        %14 = arith.index_cast %13 : i64 to index
        %15 = arith.muli %6, %c128_i32 : i32
        %16 = arith.maxsi %15, %c0_i32 : i32
        %17 = arith.index_cast %16 : i32 to index
        %18 = affine.apply #map()[%14, %17]
        %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%18], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %alloc = memref.alloc() : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc : memref<8x8x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.anchor {id = 9 : i64}
        hivm.hir.anchor {id = 10 : i64}
        %19:2 = scf.for %arg12 = %c0_i32 to %c1024_i32 step %c256_i32 iter_args(%arg13 = %c0_i32, %arg14 = %c0_i32) -> (i32, i32)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %alloc_0 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_0 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
          %alloc_1 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_2 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.anchor {id = 12 : i64}
          %20 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 13 : i64}
            %22 = arith.maxsi %arg13, %c0_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = affine.apply #map()[%14, %23]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [256, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            %alloc_4 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
            %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_4, %alloc, %true, %c256, %c128, %c128 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<8x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_5 : memref<8x16x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_5 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_2 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
            hivm.hir.anchor {id = 19 : i64}
            %25 = arith.addi %arg13, %c256_i32 : i32
            scope.return %25 : i32
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
            annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            hivm.hir.anchor {id = 38 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 39 : i64}
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.anchor {id = 40 : i64}
          %21 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 41 : i64}
            %22 = arith.maxsi %arg14, %c0_i32 : i32
            %23 = arith.index_cast %22 : i32 to index
            %24 = affine.apply #map()[%14, %23]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg4 to offset: [%24], sizes: [256, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
            %alloc_4 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_3 : memref<256x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 43 : i64}
            hivm.hir.anchor {id = 44 : i64}
            %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<8x8x16x16xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_1, %alloc_4, %true, %c128, %c256, %c128 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_5 : memref<8x8x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.anchor {id = 45 : i64}
            annotation.mark %alloc_0 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_5 : memref<8x8x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_0 : memref<64x128xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 47 : i64}
            %25 = arith.addi %arg14, %c256_i32 : i32
            scope.return %25 : i32
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
          scf.yield %20, %21 : i32, i32
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
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.dense_p = {keys = 256 : i64, padding_required = true, query_rows = 64 : i64}, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : vector<64xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<64xf32>
    %cst_2 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %0:2 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %cst_1, %arg9 = %cst_1) -> (vector<64xf32>, vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %13 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %14 = vector.shape_cast %13 : vector<1x64xf32> to vector<64xf32>
      %15 = arith.maximumf %arg8, %14 : vector<64xf32>
      %16 = affine.apply #map3()[%arg7]
      %subview_4 = memref.subview %arg0[%16, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %17 = vector.transfer_read %subview_4[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %18 = vector.shape_cast %17 : vector<1x64xf32> to vector<64xf32>
      %19 = arith.maximumf %arg9, %18 : vector<64xf32>
      %20 = affine.apply #map4()[%arg7]
      %subview_5 = memref.subview %arg0[%20, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %21 = vector.transfer_read %subview_5[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %22 = vector.shape_cast %21 : vector<1x64xf32> to vector<64xf32>
      %23 = arith.maximumf %15, %22 : vector<64xf32>
      %24 = affine.apply #map5()[%arg7]
      %subview_6 = memref.subview %arg0[%24, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %25 = vector.transfer_read %subview_6[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %26 = vector.shape_cast %25 : vector<1x64xf32> to vector<64xf32>
      %27 = arith.maximumf %19, %26 : vector<64xf32>
      scf.yield %23, %27 : vector<64xf32>, vector<64xf32>
    } {hivm.cvp_attention.reduction_chains = 2 : i64, hivm.cvp_attention.reduction_key_groups = 4 : i64}
    %1 = arith.maximumf %0#0, %0#1 : vector<64xf32>
    %2 = arith.mulf %1, %cst_0 : vector<64xf32>
    %3 = vector.transfer_read %arg3[%c0], %cst_3 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %4 = arith.maximumf %3, %2 : vector<64xf32>
    vector.transfer_write %4, %arg6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %5 = vector.shape_cast %4 : vector<64xf32> to vector<1x64xf32>
    %6:2 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %cst, %arg9 = %cst) -> (vector<64xf32>, vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %13 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %14 = arith.mulf %13, %cst_2 : vector<1x64xf32>
      %15 = arith.subf %14, %5 : vector<1x64xf32>
      %16 = math.exp %15 : vector<1x64xf32>
      %17 = vector.shape_cast %16 : vector<1x64xf32> to vector<64xf32>
      %18 = arith.truncf %16 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %19 = vector.shape_cast %18 : vector<1x64xf16> to vector<64xf16>
      %20 = affine.apply #map4()[%arg7]
      %subview_4 = memref.subview %arg0[%20, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %21 = vector.transfer_read %subview_4[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %22 = arith.mulf %21, %cst_2 : vector<1x64xf32>
      %23 = arith.subf %22, %5 : vector<1x64xf32>
      %24 = math.exp %23 : vector<1x64xf32>
      %25 = vector.shape_cast %24 : vector<1x64xf32> to vector<64xf32>
      %26 = arith.truncf %24 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %27 = vector.shape_cast %26 : vector<1x64xf16> to vector<64xf16>
      %28 = arith.addf %17, %25 : vector<64xf32>
      %29 = arith.addf %arg8, %28 : vector<64xf32>
      %res1, %res2 = ave.hir.vdintlv %19, %27 {layout_change = #ave<layout_change DENSE>} : vector<64xf16>, vector<128xf16>
      %30 = vector.shape_cast %res1 : vector<128xf16> to vector<8x1x16xf16>
      %subview_5 = memref.subview %arg1[0, %arg7, 0] [8, 1, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %30, %subview_5[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<8x1x16xf16>, memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %31 = affine.apply #map3()[%arg7]
      %subview_6 = memref.subview %arg0[%31, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %32 = vector.transfer_read %subview_6[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %33 = arith.mulf %32, %cst_2 : vector<1x64xf32>
      %34 = arith.subf %33, %5 : vector<1x64xf32>
      %35 = math.exp %34 : vector<1x64xf32>
      %36 = vector.shape_cast %35 : vector<1x64xf32> to vector<64xf32>
      %37 = arith.truncf %35 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %38 = vector.shape_cast %37 : vector<1x64xf16> to vector<64xf16>
      %39 = affine.apply #map5()[%arg7]
      %subview_7 = memref.subview %arg0[%39, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %40 = vector.transfer_read %subview_7[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %41 = arith.mulf %40, %cst_2 : vector<1x64xf32>
      %42 = arith.subf %41, %5 : vector<1x64xf32>
      %43 = math.exp %42 : vector<1x64xf32>
      %44 = vector.shape_cast %43 : vector<1x64xf32> to vector<64xf32>
      %45 = arith.truncf %43 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %46 = vector.shape_cast %45 : vector<1x64xf16> to vector<64xf16>
      %47 = arith.addf %36, %44 : vector<64xf32>
      %48 = arith.addf %arg9, %47 : vector<64xf32>
      %res1_8, %res2_9 = ave.hir.vdintlv %38, %46 {layout_change = #ave<layout_change DENSE>} : vector<64xf16>, vector<128xf16>
      %49 = vector.shape_cast %res1_8 : vector<128xf16> to vector<8x1x16xf16>
      %subview_10 = memref.subview %arg1[0, %31, 0] [8, 1, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %49, %subview_10[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<8x1x16xf16>, memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.yield %29, %48 : vector<64xf32>, vector<64xf32>
    } {hivm.cvp_attention.reduction_chains = 2 : i64, hivm.cvp_attention.reduction_key_groups = 2 : i64}
    %7 = arith.addf %6#0, %6#1 : vector<64xf32>
    %8 = arith.subf %3, %4 : vector<64xf32>
    %9 = math.exp %8 : vector<64xf32>
    vector.transfer_write %9, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %10 = vector.transfer_read %arg2[%c0], %cst_3 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %11 = arith.mulf %10, %9 : vector<64xf32>
    %12 = arith.addf %11, %7 : vector<64xf32>
    vector.transfer_write %12, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_4(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>, %arg3: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      scf.for %arg5 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg1[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_2 = memref.subview %arg3[%arg4, %arg5] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %2 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %3 = arith.mulf %2, %0 : vector<1x64xf32>
        %4 = arith.addf %1, %3 : vector<1x64xf32>
        vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_5(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_6(%arg0: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg1 = %c0 to %c64 step %c1 {
      scf.for %arg2 = %c0 to %c128 step %c64 {
        %subview = memref.subview %arg0[%arg1, %arg2] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        vector.transfer_write %cst, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_8(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c8192_i32 = arith.constant 8192 : i32
    %c128_i32 = arith.constant 128 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c256_i32 = arith.constant 256 : i32
    %c28_i32 = arith.constant 28 : i32
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
    call @_attn_fwd_mix_aiv_outlined_vf_8(%alloc) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %4 = hivm.hir.get_sub_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map6()[%5]
    %7 = affine.apply #map7()[%5]
    hivm.hir.anchor {id = 0 : i64}
    annotation.mark %1 {logical_block_num} : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %8 = arith.remsi %arg10, %arg7 : i32
      hivm.hir.anchor {id = 3 : i64}
      hivm.hir.anchor {id = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.anchor {id = 6 : i64}
      scf.for %arg11 = %8 to %c8192_i32 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %9 = arith.divsi %arg11, %c8_i32 : i32
        %10 = arith.remsi %arg11, %c8_i32 : i32
        %11 = arith.divsi %9, %c8_i32 : i32
        %12 = arith.remsi %9, %c8_i32 : i32
        %13 = arith.extsi %11 : i32 to i64
        %14 = arith.muli %13, %c1048576_i64 : i64
        %15 = arith.extsi %12 : i32 to i64
        %16 = arith.muli %15, %c131072_i64 : i64
        %17 = arith.addi %14, %16 : i64
        %18 = arith.index_cast %17 : i64 to index
        %19 = arith.muli %10, %c128_i32 : i32
        %20 = arith.maxsi %19, %c0_i32 : i32
        %21 = arith.index_cast %20 : i32 to index
        %22 = affine.apply #map()[%18, %21]
        %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [128, 128], strides: [128, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        %subview = memref.subview %reinterpret_cast[%6, 0] [64, 128] [1, 1] {to_be_bubbled_slice} : memref<128x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.anchor {id = 9 : i64}
        hivm.hir.anchor {id = 10 : i64}
        %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        func.call @_attn_fwd_mix_aiv_outlined_vf_5(%alloc_0) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
        %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%alloc : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_2 : memref<64xf32, #hivm.address_space<ub>>)
        %23:3 = scf.for %arg12 = %c0_i32 to %c1024_i32 step %c256_i32 iter_args(%arg13 = %alloc_1, %arg14 = %alloc_0, %arg15 = %alloc_2) -> (memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %alloc_8 = memref.alloc() : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_8 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x128xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_8 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x128xf32, #hivm.address_space<ub>>
          %alloc_9 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %subview_10 = memref.subview %alloc_9[%7, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          annotation.mark %alloc_9 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_11 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_11 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_11 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 12 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 13 : i64}
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
            hivm.hir.anchor {id = 17 : i64}
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.anchor {id = 19 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 20 : i64}
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 21 : i64}
          %28:3 = scope.scope : () -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
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
            %alloc_12 = memref.alloc() : memref<8x129x16x1xf16, #hivm.address_space<ub>>
            %subview_13 = memref.subview %alloc_12[0, 0, 0, 0] [8, 128, 16, 1] [1, 1, 1, 1] : memref<8x129x16x1xf16, #hivm.address_space<ub>> to memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
            %alloc_14 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            annotation.mark %alloc_15 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
            %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            func.call @_attn_fwd_mix_aiv_outlined_vf_3(%alloc_11, %subview_13, %arg14, %arg15, %alloc_14, %alloc_15, %alloc_16) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
            annotation.mark %alloc_9 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            %subview_17 = memref.subview %subview_13[0, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
            %expand_shape = memref.expand_shape %subview_17 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>>
            %subview_18 = memref.subview %subview_10[0, 0, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            %collapse_shape_19 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>
            %collapse_shape_20 = memref.collapse_shape %subview_18 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%collapse_shape_19 : memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_20 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
            %subview_21 = memref.subview %subview_13[4, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>>
            %expand_shape_22 = memref.expand_shape %subview_21 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>>
            %subview_23 = memref.subview %subview_10[0, 8, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            %collapse_shape_24 = memref.collapse_shape %expand_shape_22 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>
            %collapse_shape_25 = memref.collapse_shape %subview_23 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%collapse_shape_24 : memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
            hivm.hir.anchor {id = 38 : i64}
            scope.return %alloc_15, %alloc_16, %alloc_14 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
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
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.anchor {id = 47 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 49 : i64}
          %29 = scope.scope : () -> memref<64x128xf32, #hivm.address_space<ub>> {
            hivm.hir.anchor {id = 50 : i64}
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf32, #hivm.address_space<ub>>
            %30 = arith.cmpi eq, %arg12, %c0_i32 : i32
            scf.if %30 {
              func.call @__cvp_attention_first_output(%28#0, %alloc_8, %alloc_12) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
            } else {
              func.call @_attn_fwd_mix_aiv_outlined_vf_4(%28#0, %alloc_8, %arg13, %alloc_12) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>) -> ()
            }
            hivm.hir.anchor {id = 53 : i64}
            scope.return %alloc_12 : memref<64x128xf32, #hivm.address_space<ub>>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 54 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
          scf.yield %29, %28#2, %28#1 : memref<64x128xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 55 : i64}
        hivm.hir.anchor {id = 56 : i64}
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %24 = arith.muli %9, %c1024_i32 : i32
        %25 = arith.index_cast %24 : i32 to index
        %26 = arith.index_cast %19 : i32 to index
        %27 = affine.apply #map1()[%25, %26]
        %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%27], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %subview_5 = memref.subview %reinterpret_cast_4[%6] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64x128xf16, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32} : memref<64x128xf16, #hivm.address_space<ub>>
        func.call @_attn_fwd_mix_aiv_fused_5_outlined_merged_vf_0(%23#2, %23#1, %alloc_3, %23#1, %23#0, %alloc_6) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x128xf32, #hivm.address_space<ub>>, memref<64x128xf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.anchor {id = 57 : i64}
        hivm.hir.anchor {id = 58 : i64}
        hivm.hir.anchor {id = 59 : i64}
        hivm.hir.store ins(%alloc_3 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        hivm.hir.anchor {id = 60 : i64}
        %collapse_shape = memref.collapse_shape %alloc_6 [[0, 1]] : memref<64x128xf16, #hivm.address_space<ub>> into memref<8192xf16, #hivm.address_space<ub>>
        %collapse_shape_7 = memref.collapse_shape %subview [[0, 1]] : memref<64x128xf16, strided<[128, 1], offset: ?>, #hivm.address_space<gm>> into memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%collapse_shape : memref<8192xf16, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<8192xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        hivm.hir.anchor {id = 61 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 62 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 63 : i64}
    return
  }
  func.func @__cvp_attention_first_output(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x128xf32, #hivm.address_space<ub>>, %arg2: memref<64x128xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "first_output", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      scf.for %arg4 = %c0 to %c128 step %c64 {
        %subview_0 = memref.subview %arg1[%arg3, %arg4] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %subview_1 = memref.subview %arg2[%arg3, %arg4] [1, 64] [1, 1] : memref<64x128xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
        %1 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %cst_2 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
        %2 = arith.mulf %cst_2, %0 : vector<1x64xf32>
        %3 = arith.addf %1, %2 : vector<1x64xf32>
        vector.transfer_write %3, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[128, 1], offset: ?>, #hivm.address_space<ub>>
      }
    }
    return
  }
}

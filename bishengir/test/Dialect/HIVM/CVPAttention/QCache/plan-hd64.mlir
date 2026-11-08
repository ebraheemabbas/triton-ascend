// RUN: bishengir-opt %s --hivm-cvp-q-cache="mode=none" --verify-each | FileCheck %s --check-prefix=OFF
// RUN: bishengir-opt %s --hivm-cvp-q-cache="mode=neutral" --verify-each | FileCheck %s --check-prefix=NEUTRAL
// RUN: bishengir-opt %s --hivm-cvp-q-cache="mode=cached" --verify-each | FileCheck %s --check-prefix=CACHED
// RUN: bishengir-opt %s --hivm-cvp-q-cache="mode=cached" --hivm-cvp-q-cache="mode=cached" --verify-each | FileCheck %s --check-prefix=CACHED
// OFF: module attributes
// OFF-NOT: cvp_q_cache
// OFF-NOT: q_cache_context
// NEUTRAL-LABEL: func.func @_attn_fwd_mix_aic(
// NEUTRAL-SAME: hivm.cvp_q_cache.resources
// NEUTRAL-SAME: mode = "neutral"
// NEUTRAL: hivm.hir.mmadL1 {{.*}}cvp_q_cache = {mode = "neutral", role = "qk", version = 1 : i64}
// NEUTRAL-SAME: q_cache_context(
// NEUTRAL: hivm.hir.mmadL1 {{.*}}cvp_q_cache = {mode = "neutral", role = "pv", version = 1 : i64}
// CACHED-LABEL: func.func @_attn_fwd_mix_aic(
// CACHED-SAME: hivm.cvp_q_cache.resources
// CACHED-SAME: mode = "cached"
// CACHED-SAME: pv_b_ranges = array<i64: 0, 16384, 32768, 49152>
// CACHED-SAME: q_ranges = array<i64: 16384, 32768, 49152, 65536>
// CACHED: hivm.hir.mmadL1 {{.*}}cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}
// CACHED-SAME: q_cache_context(
// CACHED: hivm.hir.mmadL1 {{.*}}cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}

#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 4)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x64xf32, #hivm.address_space<ub>>, %arg5: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
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
      %subview_0 = memref.subview %arg4[%arg6, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg5[%arg6, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %6 = arith.divf %4, %5 : vector<1x64xf32>
      %7 = arith.truncf %6 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %7, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst_0, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    scf.for %arg2 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg1[%arg2, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %cst, %subview[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c8388608_i64 = arith.constant 8388608 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c256_i64 = arith.constant 256 : i64
    %c28_i64 = arith.constant 28 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c16128_i64 = arith.constant 16128 : i64
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
      %5 = arith.extsi %4 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %6 = arith.subi %c131072_i64, %5 : i64
      %7 = arith.maxsi %6, %c0_i64 : i64
      %8 = arith.ceildivsi %7, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %9 = arith.muli %8, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %alloc = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_0 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 7 : i64}
      scf.for %arg11 = %c0_i64 to %9 step %c256_i64  : i64 {
        hivm.hir.anchor {id = 8 : i64}
        %10 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %11 = arith.remsi %arg11, %c16384_i64 : i64
        %12 = arith.trunci %11 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %13 = arith.muli %10, %c28_i64 : i64
        %14 = arith.addi %5, %13 : i64
        %15 = arith.trunci %14 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %16 = arith.remsi %10, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %17 = arith.cmpi eq, %16, %c1_i64 : i64
        %18 = arith.cmpi eq, %11, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %19 = arith.cmpi eq, %11, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        %alloc_1 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_1 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        %alloc_2 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_2 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_3 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.anchor {id = 9 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 10 : i64}
          %20 = arith.divsi %15, %c128_i32 : i32
          %21 = arith.remsi %15, %c128_i32 : i32
          %22 = arith.divsi %20, %c8_i32 : i32
          %23 = arith.remsi %20, %c8_i32 : i32
          %24 = arith.extsi %22 : i32 to i64
          %25 = arith.muli %24, %c8388608_i64 : i64
          %26 = arith.extsi %23 : i32 to i64
          %27 = arith.muli %26, %c1048576_i64 : i64
          %28 = arith.addi %25, %27 : i64
          %29 = arith.index_cast %28 : i64 to index
          %30 = arith.muli %21, %c128_i32 : i32
          %31 = arith.maxsi %30, %c0_i32 : i32
          %32 = arith.index_cast %31 : i32 to index
          %33 = affine.apply #map1()[%29, %32]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%33], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %34 = arith.select %17, %alloc_0, %alloc : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 11 : i64}
          scf.if %18 {
            hivm.hir.anchor {id = 12 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%34 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 13 : i64}
          }
          %35 = arith.maxsi %12, %c0_i32 : i32
          %36 = arith.index_cast %35 : i32 to index
          %37 = affine.apply #map1()[%29, %36]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%37], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_5 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_5 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
          annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_5, %34, %true, %c256, %c64, %c128 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_6 : memref<8x16x16x16xf32, #hivm.address_space<cc>>)
          hivm.hir.anchor {id = 17 : i64}
          annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 18 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_6 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_3 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
          hivm.hir.anchor {id = 19 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
        hivm.hir.anchor {id = 20 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.anchor {id = 21 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 22 : i64}
          hivm.hir.anchor {id = 23 : i64}
          scf.if %18 {
            hivm.hir.anchor {id = 24 : i64}
            hivm.hir.anchor {id = 25 : i64}
            hivm.hir.anchor {id = 26 : i64}
          } else {
            hivm.hir.anchor {id = 27 : i64}
          }
          hivm.hir.anchor {id = 28 : i64}
          hivm.hir.anchor {id = 29 : i64}
          hivm.hir.anchor {id = 30 : i64}
          hivm.hir.anchor {id = 31 : i64}
          hivm.hir.anchor {id = 32 : i64}
          hivm.hir.anchor {id = 33 : i64}
          hivm.hir.anchor {id = 34 : i64}
          hivm.hir.anchor {id = 35 : i64}
          hivm.hir.anchor {id = 36 : i64}
          hivm.hir.anchor {id = 37 : i64}
          hivm.hir.anchor {id = 38 : i64}
          hivm.hir.anchor {id = 39 : i64}
          hivm.hir.anchor {id = 40 : i64}
          hivm.hir.anchor {id = 41 : i64}
          annotation.mark %alloc_2 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 42 : i64}
          hivm.hir.anchor {id = 43 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 44 : i64}
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.anchor {id = 45 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 46 : i64}
          %20 = arith.divsi %15, %c128_i32 : i32
          %21 = arith.divsi %20, %c8_i32 : i32
          %22 = arith.remsi %20, %c8_i32 : i32
          %23 = arith.extsi %21 : i32 to i64
          %24 = arith.muli %23, %c8388608_i64 : i64
          %25 = arith.extsi %22 : i32 to i64
          %26 = arith.muli %25, %c1048576_i64 : i64
          %27 = arith.addi %24, %26 : i64
          %28 = arith.index_cast %27 : i64 to index
          %29 = arith.maxsi %12, %c0_i32 : i32
          %30 = arith.index_cast %29 : i32 to index
          %31 = affine.apply #map1()[%28, %30]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_4 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 47 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.anchor {id = 49 : i64}
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_2, %alloc_4, %true, %c128, %c256, %c64 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_5 : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
          hivm.hir.anchor {id = 50 : i64}
          annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 51 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%alloc_5 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) outs(%alloc_1 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.anchor {id = 52 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 53 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        hivm.hir.anchor {id = 54 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 55 : i64}
          hivm.hir.anchor {id = 56 : i64}
          scf.if %18 {
            hivm.hir.anchor {id = 57 : i64}
            hivm.hir.anchor {id = 58 : i64}
          } else {
            hivm.hir.anchor {id = 59 : i64}
          }
          hivm.hir.anchor {id = 60 : i64}
          hivm.hir.anchor {id = 61 : i64}
          hivm.hir.anchor {id = 62 : i64}
          scf.if %19 {
            hivm.hir.anchor {id = 63 : i64}
            hivm.hir.anchor {id = 64 : i64}
            hivm.hir.anchor {id = 65 : i64}
            hivm.hir.anchor {id = 66 : i64}
            hivm.hir.anchor {id = 67 : i64}
            hivm.hir.anchor {id = 68 : i64}
            hivm.hir.anchor {id = 69 : i64}
            hivm.hir.anchor {id = 70 : i64}
          } else {
            hivm.hir.anchor {id = 71 : i64}
          }
          hivm.hir.anchor {id = 72 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 73 : i64}
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.stream = {key_count = 16384 : i64, key_step = 256 : i64, physical_lifetimes_verified = false, query_step = 28 : i64}}
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 74 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 75 : i64}
    return
  }
  func.func @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>, %arg2: memref<64x64xf32, #hivm.address_space<ub>>, %arg3: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg3[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = vector.transfer_read %subview_1[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %2 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = arith.mulf %1, %2 : vector<1x64xf32>
      %4 = arith.addf %0, %3 : vector<1x64xf32>
      vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : vector<64xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<64xf32>
    %cst_2 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %cst_1) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %11 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %12 = vector.shape_cast %11 : vector<1x64xf32> to vector<64xf32>
      %13 = arith.maximumf %arg8, %12 : vector<64xf32>
      scf.yield %13 : vector<64xf32>
    }
    %1 = arith.mulf %0, %cst_0 : vector<64xf32>
    %2 = vector.transfer_read %arg3[%c0], %cst_3 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %3 = arith.maximumf %2, %1 : vector<64xf32>
    vector.transfer_write %3, %arg6[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %4 = vector.shape_cast %3 : vector<64xf32> to vector<1x64xf32>
    %5 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %cst) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %11 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %12 = arith.mulf %11, %cst_2 : vector<1x64xf32>
      %13 = arith.subf %12, %4 : vector<1x64xf32>
      %14 = math.exp %13 : vector<1x64xf32>
      %15 = vector.shape_cast %14 : vector<1x64xf32> to vector<64xf32>
      %16 = arith.addf %arg8, %15 : vector<64xf32>
      %17 = arith.truncf %14 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %subview_4 = memref.subview %arg1[0, %arg7, 0] [4, 1, 16] [1, 1, 1] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %18 = vector.shape_cast %17 : vector<1x64xf16> to vector<4x1x16xf16>
      vector.transfer_write %18, %subview_4[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.yield %16 : vector<64xf32>
    }
    %6 = arith.subf %2, %3 : vector<64xf32>
    %7 = math.exp %6 : vector<64xf32>
    vector.transfer_write %7, %arg5[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %8 = vector.transfer_read %arg2[%c0], %cst_3 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %9 = arith.mulf %8, %7 : vector<64xf32>
    %10 = arith.addf %9, %5 : vector<64xf32>
    vector.transfer_write %10, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_6(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0 = arith.constant 0 : index
    vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c16128_i64 = arith.constant 16128 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c28_i64 = arith.constant 28 : i64
    %c256_i64 = arith.constant 256 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
    call @_attn_fwd_mix_aiv_outlined_vf_6(%alloc) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %4 = hivm.hir.get_sub_block_idx -> i64
    %5 = arith.index_cast %4 : i64 to index
    %6 = affine.apply #map2()[%5]
    %7 = affine.apply #map3()[%5]
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
      %9 = arith.extsi %8 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %10 = arith.subi %c131072_i64, %9 : i64
      %11 = arith.maxsi %10, %c0_i64 : i64
      %12 = arith.ceildivsi %11, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %13 = arith.muli %12, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      hivm.hir.anchor {id = 7 : i64}
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
      func.call @_attn_fwd_mix_aiv_outlined_merged_vf_0(%alloc_0, %alloc_1) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %alloc_2 = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_9 = memref.collapse_shape %alloc_8 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_9 : memref<4096xf32, #hivm.address_space<ub>>)
      %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%alloc_0 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64xf32, #hivm.address_space<ub>>)
      %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%alloc : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_11 : memref<64xf32, #hivm.address_space<ub>>)
      %14:5 = scf.for %arg11 = %c0_i64 to %13 step %c256_i64 iter_args(%arg12 = %alloc_8, %arg13 = %alloc_10, %arg14 = %alloc_11, %arg15 = %alloc_6, %arg16 = %alloc_7) -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i64 {
        hivm.hir.anchor {id = 8 : i64}
        %15 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %16 = arith.remsi %arg11, %c16384_i64 : i64
        %17 = arith.muli %15, %c28_i64 : i64
        %18 = arith.addi %9, %17 : i64
        %19 = arith.trunci %18 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %20 = arith.remsi %15, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %21 = arith.cmpi eq, %20, %c1_i64 : i64
        %22 = arith.cmpi eq, %16, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %23 = arith.cmpi eq, %16, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        %alloc_12 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_12 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        %alloc_13 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %subview = memref.subview %alloc_13[%7, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        annotation.mark %alloc_13 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_14 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_14 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_14 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        hivm.hir.anchor {id = 9 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 10 : i64}
          hivm.hir.anchor {id = 11 : i64}
          scf.if %22 {
            hivm.hir.anchor {id = 12 : i64}
            hivm.hir.anchor {id = 13 : i64}
          }
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
        %24:5 = scope.scope : () -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
          hivm.hir.anchor {id = 22 : i64}
          %26 = arith.select %21, %arg15, %arg13 : memref<64xf32, #hivm.address_space<ub>>
          %27 = arith.select %21, %arg16, %arg14 : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 23 : i64}
          %28 = arith.select %22, %alloc_0, %26 : memref<64xf32, #hivm.address_space<ub>>
          %29 = arith.select %22, %alloc, %27 : memref<64xf32, #hivm.address_space<ub>>
          scf.if %22 {
            hivm.hir.anchor {id = 24 : i64}
            hivm.hir.anchor {id = 25 : i64}
            hivm.hir.anchor {id = 26 : i64}
          } else {
            hivm.hir.anchor {id = 27 : i64}
          }
          hivm.hir.anchor {id = 28 : i64}
          hivm.hir.anchor {id = 29 : i64}
          hivm.hir.anchor {id = 30 : i64}
          hivm.hir.anchor {id = 31 : i64}
          hivm.hir.anchor {id = 32 : i64}
          hivm.hir.anchor {id = 33 : i64}
          hivm.hir.anchor {id = 34 : i64}
          hivm.hir.anchor {id = 35 : i64}
          hivm.hir.anchor {id = 36 : i64}
          hivm.hir.anchor {id = 37 : i64}
          hivm.hir.anchor {id = 38 : i64}
          hivm.hir.anchor {id = 39 : i64}
          hivm.hir.anchor {id = 40 : i64}
          hivm.hir.anchor {id = 41 : i64}
          %alloc_19 = memref.alloc() : memref<4x257x16x1xf16, #hivm.address_space<ub>>
          %subview_20 = memref.subview %alloc_19[0, 0, 0, 0] [4, 256, 16, 1] [1, 1, 1, 1] : memref<4x257x16x1xf16, #hivm.address_space<ub>> to memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>
          %alloc_21 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_22 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
          %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          func.call @_attn_fwd_mix_aiv_outlined_vf_3(%alloc_14, %subview_20, %28, %29, %alloc_21, %alloc_22, %alloc_23) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          %expand_shape = memref.expand_shape %subview_20 [[0], [1, 2], [3]] output_shape [4, 16, 16, 16] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> into memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>>
          annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 42 : i64}
          %collapse_shape_24 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>
          %collapse_shape_25 = memref.collapse_shape %subview [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%collapse_shape_24 : memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_25 : memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          hivm.hir.anchor {id = 43 : i64}
          %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_21 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_26 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_26 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_27 : memref<64xf32, #hivm.address_space<ub>>)
          %30 = arith.select %21, %arg13, %alloc_27 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_23 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_28 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_29 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_28 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_29 : memref<64xf32, #hivm.address_space<ub>>)
          %31 = arith.select %21, %arg14, %alloc_29 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_30 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg15 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_30 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_30 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_31 : memref<64xf32, #hivm.address_space<ub>>)
          %32 = arith.select %21, %alloc_21, %alloc_31 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_32 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg16 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_32 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_33 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_32 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_33 : memref<64xf32, #hivm.address_space<ub>>)
          %33 = arith.select %21, %alloc_23, %alloc_33 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_34 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%32 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_34 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_35 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%33 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_35 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_36 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_36 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_34 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_36 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_37 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_37 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_35 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_37 : memref<64xf32, #hivm.address_space<ub>>)
          scope.return %alloc_22, %31, %30, %alloc_36, %alloc_37 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 44 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.anchor {id = 45 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 46 : i64}
          hivm.hir.anchor {id = 47 : i64}
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.anchor {id = 49 : i64}
          hivm.hir.anchor {id = 50 : i64}
          hivm.hir.anchor {id = 51 : i64}
          hivm.hir.anchor {id = 52 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 53 : i64}
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.anchor {id = 54 : i64}
        %25 = scope.scope : () -> memref<64x64xf32, #hivm.address_space<ub>> {
          hivm.hir.anchor {id = 55 : i64}
          %26 = arith.divsi %19, %c128_i32 : i32
          %27 = arith.remsi %19, %c128_i32 : i32
          %28 = arith.divsi %26, %c8_i32 : i32
          %29 = arith.remsi %26, %c8_i32 : i32
          %30 = arith.extsi %28 : i32 to i64
          %31 = arith.muli %30, %c8388608_i64 : i64
          %32 = arith.extsi %29 : i32 to i64
          %33 = arith.muli %32, %c1048576_i64 : i64
          %34 = arith.addi %31, %33 : i64
          %35 = arith.index_cast %34 : i64 to index
          %36 = arith.muli %27, %c128_i32 : i32
          %37 = arith.maxsi %36, %c0_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %39 = affine.apply #map1()[%35, %38]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%39], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_19 = memref.subview %reinterpret_cast[%6, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.anchor {id = 56 : i64}
          %40 = arith.select %22, %alloc_1, %arg12 : memref<64x64xf32, #hivm.address_space<ub>>
          scf.if %22 {
            hivm.hir.anchor {id = 57 : i64}
            hivm.hir.anchor {id = 58 : i64}
          } else {
            hivm.hir.anchor {id = 59 : i64}
          }
          hivm.hir.anchor {id = 60 : i64}
          hivm.hir.anchor {id = 61 : i64}
          func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%24#0, %alloc_12, %40, %alloc_12) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.anchor {id = 62 : i64}
          scf.if %23 {
            hivm.hir.anchor {id = 63 : i64}
            %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%24#2 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_24 : memref<64xf32, #hivm.address_space<ub>>)
            %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%alloc_24 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_25 : memref<64xf32, #hivm.address_space<ub>>)
            %41 = arith.select %21, %24#3, %alloc_25 : memref<64xf32, #hivm.address_space<ub>>
            %alloc_26 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%24#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_26 : memref<64xf32, #hivm.address_space<ub>>)
            %alloc_27 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%alloc_26 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_27 : memref<64xf32, #hivm.address_space<ub>>)
            %42 = arith.select %21, %24#4, %alloc_27 : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 64 : i64}
            hivm.hir.anchor {id = 65 : i64}
            %alloc_28 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            %43 = arith.muli %26, %c16384_i32 : i32
            %44 = arith.index_cast %43 : i32 to index
            %45 = arith.index_cast %36 : i32 to index
            %46 = affine.apply #map4()[%44, %45]
            %reinterpret_cast_29 = memref.reinterpret_cast %arg5 to offset: [%46], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_30 = memref.subview %reinterpret_cast_29[%6] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %47 = arith.select %21, %alloc_5, %alloc_4 : memref<64xf32, #hivm.address_space<ub>>
            %alloc_31 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, #hivm.address_space<ub>>
            func.call @_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0(%42, %41, %alloc_28, %41, %alloc_12, %alloc_31) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
            hivm.hir.anchor {id = 66 : i64}
            hivm.hir.anchor {id = 67 : i64}
            hivm.hir.anchor {id = 68 : i64}
            hivm.hir.copy ins(%alloc_28 : memref<64xf32, #hivm.address_space<ub>>) outs(%47 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.store ins(%47 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_30 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.anchor {id = 69 : i64}
            %48 = arith.select %21, %alloc_3, %alloc_2 : memref<64x64xf16, #hivm.address_space<ub>>
            %collapse_shape_32 = memref.collapse_shape %alloc_31 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            %collapse_shape_33 = memref.collapse_shape %48 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            hivm.hir.copy ins(%collapse_shape_32 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_33 : memref<4096xf16, #hivm.address_space<ub>>)
            %collapse_shape_34 = memref.collapse_shape %subview_19 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_33 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.anchor {id = 70 : i64}
          } else {
            hivm.hir.anchor {id = 71 : i64}
          }
          hivm.hir.anchor {id = 72 : i64}
          %collapse_shape_20 = memref.collapse_shape %alloc_12 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          %alloc_21 = memref.alloc() : memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%collapse_shape_20 : memref<4096xf32, #hivm.address_space<ub>>) outs(%alloc_21 : memref<4096xf32, #hivm.address_space<ub>>)
          %alloc_22 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
          %collapse_shape_23 = memref.collapse_shape %alloc_22 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_21 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<4096xf32, #hivm.address_space<ub>>)
          scope.return %alloc_22 : memref<64x64xf32, #hivm.address_space<ub>>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 73 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        %alloc_15 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%24#2 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64xf32, #hivm.address_space<ub>>)
        %alloc_16 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%24#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64xf32, #hivm.address_space<ub>>)
        %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%alloc_15 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64xf32, #hivm.address_space<ub>>)
        %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%alloc_16 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_18 : memref<64xf32, #hivm.address_space<ub>>)
        scf.yield %25, %alloc_17, %alloc_18, %24#3, %24#4 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.output_storage, hivm.cvp_query.stream = {key_count = 16384 : i64, key_step = 256 : i64, physical_lifetimes_verified = false, query_step = 28 : i64}}
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 74 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 75 : i64}
    return
  }
}



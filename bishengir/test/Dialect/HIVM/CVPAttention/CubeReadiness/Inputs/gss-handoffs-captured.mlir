#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<()[s0] -> (s0 mod 2 + 2)>
#map2 = affine_map<()[s0] -> (s0 mod 2 + 1)>
#map3 = affine_map<()[s0] -> (s0 + 3)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map5 = affine_map<()[s0] -> (s0 + 128)>
#map6 = affine_map<(d0, d1)[s0] -> (d0 * 2064 + d1 + s0)>
#map7 = affine_map<()[s0] -> (s0 * 64)>
#map8 = affine_map<()[s0] -> (s0 * 4)>
#map9 = affine_map<()[s0] -> (s0 - ((s0 + 1) floordiv 2) * 2 + 2)>
#map10 = affine_map<()[s0] -> (s0 - ((s0 + 1) floordiv 2) * 2 + 4)>
#map11 = affine_map<()[s0] -> (s0 - ((s0 + 1) floordiv 2) * 2 + 3)>
#map12 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.dense_p_layouts = 1 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x64xf32, #hivm.address_space<ub>>, %arg5: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %res = ave.hir.vload <NORM> %arg0[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %res_0 = ave.hir.vload <NORM> %arg1[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.vln %res_0, %0 : vector<64xf32>, vector<64xi1>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.vadd %res, %1, %2 : vector<64xf32>, vector<64xi1>
    %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg2[%c0], %4, %3 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.for %arg6 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg6] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg4[%arg6, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg5[%arg6, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %res_5 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %6 = ave.hir.vdiv %res_4, %res_5, %5 : vector<64xf32>, vector<64xi1>
      %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %8 = ave.hir.vtruncf %6, <rint>, false, <part_even>, %7 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_6 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, #map, #hivm.address_space<ub>>
      %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %10 = builtin.unrealized_conversion_cast %8 : vector<64xf16> to vector<128xf16>
      %11 = llvm.bitcast %10 : vector<128xf16> to vector<64xf32>
      ave.hir.masked_store <PK_B32> %subview_6[%c0], %9, %11 {functionType = #ave.func_dist_type<pk>, hivm.is_continuous} : memref<64xf16, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_merged_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 1.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst_0, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
    %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg0[%c0], %4, %3 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    scf.for %arg2 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg1[%arg2, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_1[%c0], %5, %1 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c131072_i64 = arith.constant 131072 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c50176_i64 = arith.constant 50176 : i64
    %c115712_i64 = arith.constant 115712 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c17408_i64 = arith.constant 17408 : i64
    %c33792_i64 = arith.constant 33792 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c49152_i64 = arith.constant 49152 : i64
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
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c256_i64 = arith.constant 256 : i64
    %c28_i64 = arith.constant 28 : i64
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %4 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %6 = hivm.hir.pointer_cast(%c98304_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %7 = hivm.hir.pointer_cast(%c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %8 = hivm.hir.pointer_cast(%c50176_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c115712_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %11 = hivm.hir.pointer_cast(%c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %12 = hivm.hir.pointer_cast(%c17408_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %14 = arith.muli %arg7, %arg8 : i32
    %15 = arith.muli %14, %arg9 : i32
    %16 = hivm.hir.get_block_idx -> i64
    %17 = arith.trunci %16 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    scf.for %arg10 = %17 to %15 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %18 = arith.remsi %arg10, %arg7 : i32
      %19 = arith.extsi %18 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %20 = arith.subi %c131072_i64, %19 : i64
      %21 = arith.maxsi %20, %c0_i64 : i64
      %22 = arith.ceildivsi %21, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %23 = arith.muli %22, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %24 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %25 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %26 = arith.addi %23, %c1024_i64 : i64
      scf.for %arg11 = %c0_i64 to %26 step %c256_i64  : i64 {
        %27 = arith.subi %arg11, %c512_i64 : i64
        %28 = memref.load %alloca[%c0] : memref<1xi64>
        %29 = arith.remui %28, %c2_i64 : i64
        %30 = arith.index_cast %29 : i64 to index
        %31 = arith.cmpi eq, %29, %c1_i64 : i64
        %32 = arith.select %31, %1, %0 : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %33 = arith.select %31, %3, %2 : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %34 = arith.select %31, %5, %4 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %35 = arith.select %31, %7, %6 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %36 = arith.select %31, %9, %8 : memref<256x64xf32, #hivm.address_space<ub>>
        %37 = arith.select %31, %11, %10 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %38 = arith.select %31, %13, %12 : memref<64x64xf32, #hivm.address_space<ub>>
        %39 = arith.index_cast %30 : index to i1
        %40 = arith.select %39, %c1_i64, %c2_i64 : i64
        %41 = affine.apply #map1()[%30]
        %42 = arith.index_cast %41 : index to i64
        %43 = affine.apply #map2()[%30]
        %44 = arith.index_cast %43 : index to i64
        %45 = affine.apply #map3()[%30]
        %46 = arith.index_cast %45 : index to i64
        %47 = arith.divsi %27, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %48 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %49 = arith.remsi %27, %c16384_i64 : i64
        %50 = arith.remsi %arg11, %c16384_i64 : i64
        %51 = arith.trunci %49 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %52 = arith.trunci %50 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %53 = arith.muli %47, %c28_i64 : i64
        %54 = arith.muli %48, %c28_i64 : i64
        %55 = arith.addi %19, %53 : i64
        %56 = arith.addi %19, %54 : i64
        %57 = arith.trunci %55 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %58 = arith.trunci %56 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %59 = arith.remsi %48, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %60 = arith.cmpi eq, %59, %c1_i64 : i64
        %61 = arith.cmpi eq, %50, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %62 = arith.cmpi sge, %arg11, %c0_i64 : i64
        %63 = arith.cmpi slt, %arg11, %23 : i64
        %64 = arith.andi %62, %63 : i1
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        scf.if %64 {
          %69 = arith.divsi %58, %c128_i32 : i32
          %70 = arith.remsi %58, %c128_i32 : i32
          %71 = arith.divsi %69, %c8_i32 : i32
          %72 = arith.remsi %69, %c8_i32 : i32
          %73 = arith.extsi %71 : i32 to i64
          %74 = arith.muli %73, %c8388608_i64 : i64
          %75 = arith.extsi %72 : i32 to i64
          %76 = arith.muli %75, %c1048576_i64 : i64
          %77 = arith.addi %74, %76 : i64
          %78 = arith.index_cast %77 : i64 to index
          %79 = arith.muli %70, %c128_i32 : i32
          %80 = arith.maxsi %79, %c0_i32 : i32
          %81 = arith.index_cast %80 : i32 to index
          %82 = affine.apply #map4()[%78, %81]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%82], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %83 = arith.select %60, %25, %24 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          scf.if %61 {
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%83 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
          }
          %84 = arith.maxsi %52, %c0_i32 : i32
          %85 = arith.index_cast %84 : i32 to index
          %86 = affine.apply #map4()[%78, %85]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%86], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%35 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          %87 = arith.select %60, %c49152_i64, %c16384_i64 : i64
          hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%35, %83, %true, %c256, %c64, %c128 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%32 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64, %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64) q_cache_context(%61, %48, %87)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = %46
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%32 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) outs(%36 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        }
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        %65 = arith.cmpi sge, %27, %c0_i64 : i64
        %66 = arith.cmpi slt, %27, %23 : i64
        %67 = arith.andi %65, %66 : i1
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        scf.if %67 {
          %69 = arith.divsi %57, %c128_i32 : i32
          %70 = arith.divsi %69, %c8_i32 : i32
          %71 = arith.remsi %69, %c8_i32 : i32
          %72 = arith.extsi %70 : i32 to i64
          %73 = arith.muli %72, %c8388608_i64 : i64
          %74 = arith.extsi %71 : i32 to i64
          %75 = arith.muli %74, %c1048576_i64 : i64
          %76 = arith.addi %73, %75 : i64
          %77 = arith.index_cast %76 : i64 to index
          %78 = arith.maxsi %51, %c0_i32 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = affine.apply #map4()[%77, %79]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%80], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %40]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%34 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
          hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%37, %34, %true, %c128, %c256, %c64 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%33 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64, %c0_i64, %c-1_i64, %40, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %42
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = %44
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%33 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) outs(%38 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 0
        }
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        %68 = arith.addi %28, %c1_i64 : i64
        memref.store %68, %alloca[%c0] : memref<1xi64>
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>, %arg2: memref<64x64xf32, #hivm.address_space<ub>>, %arg3: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %arg4 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg3[%arg4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %subview_4 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_5 = ave.hir.vload <NORM> %subview_4[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %res_6 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %1 = ave.hir.vmula %res, %res_5, %res_6, %0 : vector<64xf32>, vector<64xi1>
      %subview_7 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_7[%c0], %2, %1 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.dense_p = {keys = 256 : i64, padding_required = true, query_rows = 64 : i64}, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c2064 = arith.constant 2064 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %c128 = arith.constant 128 : index
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst_1, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
    %4 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %3) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %17 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %18 = ave.hir.vmax %arg8, %res_4, %17 : vector<64xf32>, vector<64xi1>
      scf.yield %18 : vector<64xf32>
    }
    %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %6 = ave.hir.vmuls %4, %cst_0, %5 : vector<64xf32>, f32, vector<64xi1>
    %res = ave.hir.vload <NORM> %arg3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %8 = ave.hir.vmax %res, %6, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg6[%c0], %9, %8 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %10 = scf.for %arg7 = %c0 to %c128 step %c1 iter_args(%arg8 = %1) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %17 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %18 = ave.hir.vmuls %res_4, %cst_0, %17 : vector<64xf32>, f32, vector<64xi1>
      %19 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %20 = ave.hir.vexpdif %18, %8, %19 : vector<64xf32>, vector<64xi1>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = ave.hir.vtruncf %20, <rint>, false, <part_even>, %21 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %23 = affine.apply #map5()[%arg7]
      %subview_5 = memref.subview %arg0[%23, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %subview_5[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_7 = ave.hir.vload <NORM> %subview_6[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %24 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %25 = ave.hir.vmuls %res_7, %cst_0, %24 : vector<64xf32>, f32, vector<64xi1>
      %26 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %27 = ave.hir.vexpdif %25, %8, %26 : vector<64xf32>, vector<64xi1>
      %28 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %29 = ave.hir.vtruncf %27, <rint>, false, <part_even>, %28 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %30 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %31 = ave.hir.vadd %20, %27, %30 : vector<64xf32>, vector<64xi1>
      %32 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %33 = ave.hir.vadd %arg8, %31, %32 : vector<64xf32>, vector<64xi1>
      %res1, %res2 = ave.hir.vdintlv %22, %29 {layout_change = #ave<layout_change DENSE>} : vector<64xf16>, vector<128xf16>
      %subview_8 = memref.subview %arg1[0, %arg7, 0] [8, 1, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_9 = memref.subview %subview_8[0, 0, 0] [8, 1, 16] [1, 1, 1] : memref<8x1x16xf16, strided<[2064, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<8x16xf16, #map6, #hivm.address_space<ub>>
      %34 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      ave.hir.store_with_stride %subview_9[%c0, %c0], %c2064, %34, %res1 {functionType = #ave.func_dist_type<norm>} : memref<8x16xf16, #map6, #hivm.address_space<ub>>, vector<128xi1>, vector<128xf16>
      scf.yield %33 : vector<64xf32>
    }
    %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %12 = ave.hir.vexpdif %res, %8, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg5[%c0], %13, %12 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %res_2 = ave.hir.vload <NORM> %arg2[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %15 = ave.hir.vmula %10, %res_2, %12, %14 : vector<64xf32>, vector<64xi1>
    %16 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg4[%c0], %16, %15 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_6(%arg0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFF800000 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg0[%c0], %2, %1 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c248064_i64 = arith.constant 248064 : i64
    %c248576_i64 = arith.constant 248576 : i64
    %c248320_i64 = arith.constant 248320 : i64
    %c50176_i64 = arith.constant 50176 : i64
    %c115712_i64 = arith.constant 115712 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c17408_i64 = arith.constant 17408 : i64
    %c33792_i64 = arith.constant 33792 : i64
    %c768_i64 = arith.constant 768 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c249600_i64 = arith.constant 249600 : i64
    %c181248_i64 = arith.constant 181248 : i64
    %c249344_i64 = arith.constant 249344 : i64
    %c249088_i64 = arith.constant 249088 : i64
    %c248832_i64 = arith.constant 248832 : i64
    %c17152_i64 = arith.constant 17152 : i64
    %c16896_i64 = arith.constant 16896 : i64
    %c16640_i64 = arith.constant 16640 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c239360_i64 = arith.constant 239360 : i64
    %c239616_i64 = arith.constant 239616 : i64
    %c231168_i64 = arith.constant 231168 : i64
    %c239872_i64 = arith.constant 239872 : i64
    %c214784_i64 = arith.constant 214784 : i64
    %c214528_i64 = arith.constant 214528 : i64
    %c214272_i64 = arith.constant 214272 : i64
    %c16128_i64 = arith.constant 16128 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c28_i64 = arith.constant 28 : i64
    %c256_i64 = arith.constant 256 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16384_i32 = arith.constant 16384 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c28_i32 = arith.constant 28 : i32
    %c3_i64 = arith.constant 3 : i64
    %c0 = arith.constant 0 : index
    %c0_i64 = arith.constant 0 : i64
    %alloca = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c248064_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c248320_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c248576_i64) : memref<64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c248320_i64) : memref<64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c248576_i64) : memref<64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c248064_i64) : memref<64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c115712_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c50176_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %9 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %10 = hivm.hir.pointer_cast(%c33792_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c17408_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %12 = arith.muli %arg7, %arg8 : i32
    %13 = arith.muli %12, %arg9 : i32
    %14 = hivm.hir.get_block_idx -> i64
    %15 = arith.trunci %14 : i64 to i32
    %16 = hivm.hir.pointer_cast(%c214272_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @_attn_fwd_mix_aiv_outlined_vf_6(%16) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %17 = hivm.hir.get_sub_block_idx -> i64
    %18 = arith.index_cast %17 : i64 to index
    %19 = affine.apply #map7()[%18]
    %20 = affine.apply #map8()[%18]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 2
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 3
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 4
    scf.for %arg10 = %15 to %13 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %21 = arith.remsi %arg10, %arg7 : i32
      %22 = arith.extsi %21 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %23 = arith.subi %c131072_i64, %22 : i64
      %24 = arith.maxsi %23, %c0_i64 : i64
      %25 = arith.ceildivsi %24, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %26 = arith.muli %25, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %27 = hivm.hir.pointer_cast(%c214528_i64) : memref<64xf32, #hivm.address_space<ub>>
      %28 = hivm.hir.pointer_cast(%c214784_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      func.call @_attn_fwd_mix_aiv_outlined_merged_vf_0(%27, %28) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %29 = hivm.hir.pointer_cast(%c239872_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %30 = hivm.hir.pointer_cast(%c231168_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %31 = hivm.hir.pointer_cast(%c239616_i64) : memref<64xf32, #hivm.address_space<ub>>
      %32 = hivm.hir.pointer_cast(%c239360_i64) : memref<64xf32, #hivm.address_space<ub>>
      %33 = hivm.hir.pointer_cast(%c0_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      %34 = hivm.hir.pointer_cast(%c16384_i64) : memref<64xf32, #hivm.address_space<ub>>
      %35 = hivm.hir.pointer_cast(%c16640_i64) : memref<64xf32, #hivm.address_space<ub>>
      %36 = hivm.hir.pointer_cast(%c16896_i64) : memref<64xf32, #hivm.address_space<ub>>
      %37 = hivm.hir.pointer_cast(%c17152_i64) : memref<64xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %28 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_0 = memref.collapse_shape %33 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_0 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%27 : memref<64xf32, #hivm.address_space<ub>>) outs(%34 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%16 : memref<64xf32, #hivm.address_space<ub>>) outs(%35 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%27 : memref<64xf32, #hivm.address_space<ub>>) outs(%36 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%16 : memref<64xf32, #hivm.address_space<ub>>) outs(%37 : memref<64xf32, #hivm.address_space<ub>>)
      %38 = arith.addi %26, %c1024_i64 : i64
      scf.for %arg11 = %c0_i64 to %38 step %c256_i64  : i64 {
        %39 = arith.subi %arg11, %c256_i64 : i64
        %40 = arith.subi %arg11, %c768_i64 : i64
        %41 = memref.load %alloca[%c0] : memref<1xi64>
        %42 = arith.remui %41, %c3_i64 : i64
        %43 = arith.cmpi eq, %42, %c1_i64 : i64
        %44 = arith.select %43, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
        %45 = arith.cmpi eq, %42, %c2_i64 : i64
        %46 = arith.select %45, %2, %44 : memref<64xf32, #hivm.address_space<ub>>
        %47 = arith.select %43, %4, %3 : memref<64xf32, #hivm.address_space<ub>>
        %48 = arith.select %45, %5, %47 : memref<64xf32, #hivm.address_space<ub>>
        %49 = arith.remui %41, %c2_i64 : i64
        %50 = arith.index_cast %49 : i64 to index
        %51 = arith.cmpi eq, %49, %c1_i64 : i64
        %52 = arith.select %51, %7, %6 : memref<256x64xf32, #hivm.address_space<ub>>
        %53 = arith.select %51, %9, %8 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %54 = arith.select %51, %11, %10 : memref<64x64xf32, #hivm.address_space<ub>>
        %55 = affine.apply #map9()[%50]
        %56 = arith.index_cast %55 : index to i64
        %57 = affine.apply #map10()[%50]
        %58 = arith.index_cast %57 : index to i64
        %59 = affine.apply #map11()[%50]
        %60 = arith.index_cast %59 : index to i64
        %61 = arith.divsi %40, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %62 = arith.divsi %39, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %63 = arith.remsi %40, %c16384_i64 : i64
        %64 = arith.remsi %39, %c16384_i64 : i64
        %65 = arith.muli %61, %c28_i64 : i64
        %66 = arith.addi %22, %65 : i64
        %67 = arith.trunci %66 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %68 = arith.remsi %61, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %69 = arith.remsi %62, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %70 = arith.cmpi eq, %68, %c1_i64 : i64
        %71 = arith.cmpi eq, %69, %c1_i64 : i64
        %72 = arith.cmpi eq, %63, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %73 = arith.cmpi eq, %64, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %74 = arith.cmpi eq, %63, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        %subview = memref.subview %53[%20, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        %75 = arith.cmpi sge, %39, %c0_i64 : i64
        %76 = arith.cmpi slt, %39, %26 : i64
        %77 = arith.andi %75, %76 : i1
        scf.if %77 {
          %82 = arith.select %71, %36, %34 : memref<64xf32, #hivm.address_space<ub>>
          %83 = arith.select %71, %37, %35 : memref<64xf32, #hivm.address_space<ub>>
          %84 = hivm.hir.pointer_cast(%c248832_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%82 : memref<64xf32, #hivm.address_space<ub>>) outs(%84 : memref<64xf32, #hivm.address_space<ub>>)
          %85 = hivm.hir.pointer_cast(%c249088_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%83 : memref<64xf32, #hivm.address_space<ub>>) outs(%85 : memref<64xf32, #hivm.address_space<ub>>)
          %86 = hivm.hir.pointer_cast(%c249344_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%84 : memref<64xf32, #hivm.address_space<ub>>) outs(%86 : memref<64xf32, #hivm.address_space<ub>>)
          %87 = hivm.hir.pointer_cast(%c248832_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%86 : memref<64xf32, #hivm.address_space<ub>>) outs(%87 : memref<64xf32, #hivm.address_space<ub>>)
          %88 = arith.select %73, %27, %87 : memref<64xf32, #hivm.address_space<ub>>
          %89 = hivm.hir.pointer_cast(%c249344_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%85 : memref<64xf32, #hivm.address_space<ub>>) outs(%89 : memref<64xf32, #hivm.address_space<ub>>)
          %90 = hivm.hir.pointer_cast(%c249088_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%89 : memref<64xf32, #hivm.address_space<ub>>) outs(%90 : memref<64xf32, #hivm.address_space<ub>>)
          %91 = arith.select %73, %16, %90 : memref<64xf32, #hivm.address_space<ub>>
          %92 = hivm.hir.pointer_cast(%c181248_i64) : memref<8x129x16x1xf16, #hivm.address_space<ub>>
          %subview_1 = memref.subview %92[0, 0, 0, 0] [8, 128, 16, 1] [1, 1, 1, 1] : memref<8x129x16x1xf16, #hivm.address_space<ub>> to memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
          %93 = hivm.hir.pointer_cast(%c249344_i64) : memref<64xf32, #hivm.address_space<ub>>
          %94 = hivm.hir.pointer_cast(%c249600_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          func.call @_attn_fwd_mix_aiv_outlined_vf_3(%52, %subview_1, %88, %91, %93, %46, %94) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %58
          %subview_2 = memref.subview %subview_1[0, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
          %expand_shape = memref.expand_shape %subview_2 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>>
          %subview_3 = memref.subview %subview[0, 0, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          %collapse_shape_4 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>
          %collapse_shape_5 = memref.collapse_shape %subview_3 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = %60
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.copy ins(%collapse_shape_4 : memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          %subview_6 = memref.subview %subview_1[4, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>>
          %expand_shape_7 = memref.expand_shape %subview_6 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>>
          %subview_8 = memref.subview %subview[0, 8, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          %collapse_shape_9 = memref.collapse_shape %expand_shape_7 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>
          %collapse_shape_10 = memref.collapse_shape %subview_8 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.pipe_barrier[<PIPE_MTE3>]
          hivm.hir.copy ins(%collapse_shape_9 : memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>) outs(%collapse_shape_10 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0
          hivm.hir.copy ins(%93 : memref<64xf32, #hivm.address_space<ub>>) outs(%82 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%94 : memref<64xf32, #hivm.address_space<ub>>) outs(%83 : memref<64xf32, #hivm.address_space<ub>>)
        }
        hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        %78 = arith.cmpi sge, %40, %c0_i64 : i64
        %79 = arith.cmpi slt, %40, %26 : i64
        %80 = arith.andi %78, %79 : i1
        hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
        scf.if %80 {
          %82 = hivm.hir.pointer_cast(%c181248_i64) : memref<64x64xf32, #hivm.address_space<ub>>
          %collapse_shape_1 = memref.collapse_shape %82 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%collapse_shape_0 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<4096xf32, #hivm.address_space<ub>>)
          %83 = arith.divsi %67, %c128_i32 : i32
          %84 = arith.remsi %67, %c128_i32 : i32
          %85 = arith.divsi %83, %c8_i32 : i32
          %86 = arith.remsi %83, %c8_i32 : i32
          %87 = arith.extsi %85 : i32 to i64
          %88 = arith.muli %87, %c8388608_i64 : i64
          %89 = arith.extsi %86 : i32 to i64
          %90 = arith.muli %89, %c1048576_i64 : i64
          %91 = arith.addi %88, %90 : i64
          %92 = arith.index_cast %91 : i64 to index
          %93 = arith.muli %84, %c128_i32 : i32
          %94 = arith.maxsi %93, %c0_i32 : i32
          %95 = arith.index_cast %94 : i32 to index
          %96 = affine.apply #map4()[%92, %95]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%96], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %reinterpret_cast[%19, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %97 = arith.select %72, %28, %82 : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 0
          func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%48, %54, %97, %54) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
          %collapse_shape_3 = memref.collapse_shape %54 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%collapse_shape_3 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_0 : memref<4096xf32, #hivm.address_space<ub>>)
          scf.if %74 {
            %98 = arith.select %70, %36, %34 : memref<64xf32, #hivm.address_space<ub>>
            %99 = hivm.hir.pointer_cast(%c248832_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%98 : memref<64xf32, #hivm.address_space<ub>>) outs(%99 : memref<64xf32, #hivm.address_space<ub>>)
            %100 = arith.select %70, %37, %35 : memref<64xf32, #hivm.address_space<ub>>
            %101 = hivm.hir.pointer_cast(%c249088_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%100 : memref<64xf32, #hivm.address_space<ub>>) outs(%101 : memref<64xf32, #hivm.address_space<ub>>)
            %102 = hivm.hir.pointer_cast(%c249344_i64) : memref<64xf32, #hivm.address_space<ub>>
            %103 = arith.muli %83, %c16384_i32 : i32
            %104 = arith.index_cast %103 : i32 to index
            %105 = arith.index_cast %93 : i32 to index
            %106 = affine.apply #map12()[%104, %105]
            %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%106], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_5 = memref.subview %reinterpret_cast_4[%19] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %107 = arith.select %70, %32, %31 : memref<64xf32, #hivm.address_space<ub>>
            %108 = hivm.hir.pointer_cast(%c181248_i64) : memref<64x64xf16, #hivm.address_space<ub>>
            func.call @_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0(%101, %99, %102, %99, %54, %108) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
            hivm.hir.copy ins(%102 : memref<64xf32, #hivm.address_space<ub>>) outs(%107 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%107 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_5 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            %109 = arith.select %70, %30, %29 : memref<64x64xf16, #hivm.address_space<ub>>
            %collapse_shape_6 = memref.collapse_shape %108 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            %collapse_shape_7 = memref.collapse_shape %109 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            hivm.hir.copy ins(%collapse_shape_6 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<4096xf16, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            %collapse_shape_8 = memref.collapse_shape %subview_2 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%collapse_shape_7 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_8 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
          }
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = %56
        }
        %81 = arith.addi %41, %c1_i64 : i64
        memref.store %81, %alloca[%c0] : memref<1xi64>
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 2
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 3
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
}

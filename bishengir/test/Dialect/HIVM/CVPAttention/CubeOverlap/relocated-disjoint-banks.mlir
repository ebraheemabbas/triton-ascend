// RUN: bishengir-opt %s --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-overlap="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: FileCheck %S/guarded-overlap.mlir --input-file=%t
// Both physical bank pairs are relocated; the proof must use ranges.
#map = affine_map<(d0)[s0] -> (d0 + s0)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<(d0, d1)[s0] -> (d0 * 4112 + d1 + s0)>
#map3 = affine_map<()[s0] -> (s0 * 64)>
#map4 = affine_map<()[s0] -> (s0 * 4)>
#map5 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.first_output_specializations = 1 : i64, hivm.cvp_attention.first_state_specializations = 1 : i64, hivm.cvp_attention.last_output_specializations = 1 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
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
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %qcache_even = arith.constant 16384 : i64
    %qcache_odd = arith.constant 49152 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c102272_i64 = arith.constant 102272 : i64
    %c167808_i64 = arith.constant 167808 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c69504_i64 = arith.constant 69504 : i64
    %c85888_i64 = arith.constant 85888 : i64
    %c-1_i64 = arith.constant -1 : i64
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c16384_i64 = arith.constant 16384 : i64
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
    %0 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %4 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %5 = hivm.hir.pointer_cast(%c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %6 = hivm.hir.pointer_cast(%c98304_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %7 = hivm.hir.pointer_cast(%c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
    %8 = hivm.hir.pointer_cast(%c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %11 = hivm.hir.pointer_cast(%c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %12 = hivm.hir.pointer_cast(%c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %14 = arith.muli %arg7, %arg8 : i32
    %15 = arith.muli %14, %arg9 : i32
    %16 = hivm.hir.get_block_idx -> i64
    %17 = arith.trunci %16 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
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
      %24 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %25 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %26 = arith.addi %23, %c1024_i64 : i64
      scf.for %arg11 = %c0_i64 to %26 step %c256_i64  : i64 {
        %27 = memref.load %alloca[%c0] : memref<1xi64>
        %28 = arith.remui %27, %c2_i64 : i64
        %29 = arith.index_cast %28 : i64 to index
        %30 = arith.cmpi eq, %28, %c1_i64 : i64
        %31 = arith.select %30, %1, %0 : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %32 = arith.select %30, %3, %2 : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %33 = arith.select %30, %5, %4 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %34 = arith.select %30, %7, %6 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %35 = arith.select %30, %9, %8 : memref<256x64xf32, #hivm.address_space<ub>>
        %36 = arith.select %30, %11, %10 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %37 = arith.select %30, %13, %12 : memref<64x64xf32, #hivm.address_space<ub>>
        %38 = arith.index_cast %29 : index to i1
        %39 = arith.select %38, %c1_i64, %c2_i64 : i64
        %40 = arith.subi %arg11, %c512_i64 : i64
        %41 = arith.divsi %40, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %42 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %43 = arith.remsi %40, %c16384_i64 : i64
        %44 = arith.remsi %arg11, %c16384_i64 : i64
        %45 = arith.trunci %43 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %46 = arith.trunci %44 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %47 = arith.muli %41, %c28_i64 : i64
        %48 = arith.muli %42, %c28_i64 : i64
        %49 = arith.addi %19, %47 : i64
        %50 = arith.addi %19, %48 : i64
        %51 = arith.trunci %49 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %52 = arith.trunci %50 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %53 = arith.remsi %42, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %54 = arith.cmpi eq, %53, %c1_i64 : i64
        %55 = arith.cmpi eq, %44, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        %56 = arith.cmpi sge, %arg11, %c0_i64 : i64
        %57 = arith.cmpi slt, %arg11, %23 : i64
        %58 = arith.andi %56, %57 : i1
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        scf.if %58 {
          %63 = arith.divsi %52, %c128_i32 : i32
          %64 = arith.remsi %52, %c128_i32 : i32
          %65 = arith.divsi %63, %c8_i32 : i32
          %66 = arith.remsi %63, %c8_i32 : i32
          %67 = arith.extsi %65 : i32 to i64
          %68 = arith.muli %67, %c8388608_i64 : i64
          %69 = arith.extsi %66 : i32 to i64
          %70 = arith.muli %69, %c1048576_i64 : i64
          %71 = arith.addi %68, %70 : i64
          %72 = arith.index_cast %71 : i64 to index
          %73 = arith.muli %64, %c128_i32 : i32
          %74 = arith.maxsi %73, %c0_i32 : i32
          %75 = arith.index_cast %74 : i32 to index
          %76 = affine.apply #map1()[%72, %75]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%76], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %77 = arith.select %54, %25, %24 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
          scf.if %55 {
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%77 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
          }
          %78 = arith.maxsi %46, %c0_i32 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = affine.apply #map1()[%72, %79]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%80], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%34 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          %qcache_offset = arith.select %54, %qcache_odd, %qcache_even : i64
          hivm.hir.mmadL1 {cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}, already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%34, %77, %true, %c256, %c64, %c128 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%31 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c0_i64, %c-1_i64, %c-1_i64, %c0_i64, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64) q_cache_context(%55, %42, %qcache_offset)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%31 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) outs(%35 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
        }
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %59 = arith.cmpi sge, %40, %c0_i64 : i64
        %60 = arith.cmpi slt, %40, %23 : i64
        %61 = arith.andi %59, %60 : i1
        hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>]
        scf.if %61 {
          %63 = arith.divsi %51, %c128_i32 : i32
          %64 = arith.divsi %63, %c8_i32 : i32
          %65 = arith.remsi %63, %c8_i32 : i32
          %66 = arith.extsi %64 : i32 to i64
          %67 = arith.muli %66, %c8388608_i64 : i64
          %68 = arith.extsi %65 : i32 to i64
          %69 = arith.muli %68, %c1048576_i64 : i64
          %70 = arith.addi %67, %69 : i64
          %71 = arith.index_cast %70 : i64 to index
          %72 = arith.maxsi %45, %c0_i32 : i32
          %73 = arith.index_cast %72 : i32 to index
          %74 = affine.apply #map1()[%71, %73]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%74], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %39]
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%33 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
          hivm.hir.mmadL1 {cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}, a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%36, %33, %true, %c128, %c256, %c64 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%32 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) sync_related_args(%c-1_i64, %c0_i64, %c-1_i64, %39, %c-1_i64, %c-1_i64, %c-1_i64 : i64, i64, i64, i64, i64, i64, i64)
          hivm.hir.set_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.wait_flag[<PIPE_M>, <PIPE_FIX>, <EVENT_ID0>]
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%32 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) outs(%37 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        }
        hivm.hir.set_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        %62 = arith.addi %27, %c1_i64 : i64
        memref.store %62, %alloca[%c0] : memref<1xi64>
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_M>, <PIPE_MTE1>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_FIX>, <PIPE_M>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, <EVENT_ID2>]
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
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
      %1 = ave.hir.vmul %res_5, %res_6, %0 : vector<64xf32>, vector<64xi1>
      %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %3 = ave.hir.vadd %res, %1, %2 : vector<64xf32>, vector<64xi1>
      %subview_7 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_7[%c0], %4, %3 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>, %arg5: memref<64xf32, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c4112 = arith.constant 4112 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst_1, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
    %4 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %3) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %19 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %20 = ave.hir.vmax %arg8, %res_4, %19 : vector<64xf32>, vector<64xi1>
      scf.yield %20 : vector<64xf32>
    }
    %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %6 = ave.hir.vmuls %4, %cst_0, %5 : vector<64xf32>, f32, vector<64xi1>
    %res = ave.hir.vload <NORM> %arg3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %8 = ave.hir.vmax %res, %6, %7 : vector<64xf32>, vector<64xi1>
    %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg6[%c0], %9, %8 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %10 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %1) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg7, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %19 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %20 = ave.hir.vmuls %res_4, %cst_0, %19 : vector<64xf32>, f32, vector<64xi1>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = ave.hir.vexpdif %20, %8, %21 : vector<64xf32>, vector<64xi1>
      %23 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %24 = ave.hir.vadd %arg8, %22, %23 : vector<64xf32>, vector<64xi1>
      %25 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %26 = ave.hir.vtruncf %22, <rint>, false, <part_even>, %25 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_5 = memref.subview %arg1[0, %arg7, 0] [4, 1, 16] [1, 1, 1] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_6 = memref.subview %subview_5[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, #map2, #hivm.address_space<ub>>
      %27 = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      %res1, %res2 = ave.hir.vdintlv %26, %26 : vector<64xf16>, vector<64xf16>
      ave.hir.store_with_stride %subview_6[%c0, %c0], %c4112, %27, %res1 : memref<4x16xf16, #map2, #hivm.address_space<ub>>, vector<128xi1>, vector<64xf16>
      scf.yield %24 : vector<64xf32>
    }
    %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %12 = ave.hir.vexpdif %res, %8, %11 : vector<64xf32>, vector<64xi1>
    %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg5[%c0], %13, %12 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %res_2 = ave.hir.vload <NORM> %arg2[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>> into vector<64xf32>
    %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %15 = ave.hir.vmul %res_2, %12, %14 : vector<64xf32>, vector<64xi1>
    %16 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %17 = ave.hir.vadd %15, %10, %16 : vector<64xf32>, vector<64xi1>
    %18 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg4[%c0], %18, %17 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
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
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c68224_i64 = arith.constant 68224 : i64
    %c67968_i64 = arith.constant 67968 : i64
    %c67712_i64 = arith.constant 67712 : i64
    %c68736_i64 = arith.constant 68736 : i64
    %c68992_i64 = arith.constant 68992 : i64
    %c69248_i64 = arith.constant 69248 : i64
    %c66944_i64 = arith.constant 66944 : i64
    %c67200_i64 = arith.constant 67200 : i64
    %c66688_i64 = arith.constant 66688 : i64
    %c102272_i64 = arith.constant 102272 : i64
    %c167808_i64 = arith.constant 167808 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c69504_i64 = arith.constant 69504 : i64
    %c85888_i64 = arith.constant 85888 : i64
    %c768_i64 = arith.constant 768 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c69760_i64 = arith.constant 69760 : i64
    %c103040_i64 = arith.constant 103040 : i64
    %c102784_i64 = arith.constant 102784 : i64
    %c102528_i64 = arith.constant 102528 : i64
    %c234880_i64 = arith.constant 234880 : i64
    %c234624_i64 = arith.constant 234624 : i64
    %c234368_i64 = arith.constant 234368 : i64
    %c234112_i64 = arith.constant 234112 : i64
    %c233856_i64 = arith.constant 233856 : i64
    %c233600_i64 = arith.constant 233600 : i64
    %c233344_i64 = arith.constant 233344 : i64
    %c68480_i64 = arith.constant 68480 : i64
    %c67456_i64 = arith.constant 67456 : i64
    %c235136_i64 = arith.constant 235136 : i64
    %c49792_i64 = arith.constant 49792 : i64
    %c66432_i64 = arith.constant 66432 : i64
    %c58240_i64 = arith.constant 58240 : i64
    %c50048_i64 = arith.constant 50048 : i64
    %c33408_i64 = arith.constant 33408 : i64
    %c33152_i64 = arith.constant 33152 : i64
    %c32896_i64 = arith.constant 32896 : i64
    %c16128_i64 = arith.constant 16128 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c28_i64 = arith.constant 28 : i64
    %c256_i64 = arith.constant 256 : i64
    %c16384_i64 = arith.constant 16384 : i64
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
    %alloca_0 = memref.alloca() : memref<1xi64>
    memref.store %c0_i64, %alloca_0[%c0] : memref<1xi64>
    %0 = hivm.hir.pointer_cast(%c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
    %1 = hivm.hir.pointer_cast(%c67968_i64) : memref<64xf32, #hivm.address_space<ub>>
    %2 = hivm.hir.pointer_cast(%c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
    %3 = hivm.hir.pointer_cast(%c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
    %4 = hivm.hir.pointer_cast(%c68992_i64) : memref<64xf32, #hivm.address_space<ub>>
    %5 = hivm.hir.pointer_cast(%c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
    %6 = hivm.hir.pointer_cast(%c68992_i64) : memref<64xf32, #hivm.address_space<ub>>
    %7 = hivm.hir.pointer_cast(%c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
    %8 = hivm.hir.pointer_cast(%c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
    %9 = hivm.hir.pointer_cast(%c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
    %10 = hivm.hir.pointer_cast(%c68992_i64) : memref<64xf32, #hivm.address_space<ub>>
    %11 = hivm.hir.pointer_cast(%c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
    %12 = hivm.hir.pointer_cast(%c68992_i64) : memref<64xf32, #hivm.address_space<ub>>
    %13 = hivm.hir.pointer_cast(%c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
    %14 = hivm.hir.pointer_cast(%c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
    %15 = hivm.hir.pointer_cast(%c67968_i64) : memref<64xf32, #hivm.address_space<ub>>
    %16 = hivm.hir.pointer_cast(%c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
    %17 = hivm.hir.pointer_cast(%c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
    %18 = hivm.hir.pointer_cast(%c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
    %19 = hivm.hir.pointer_cast(%c67968_i64) : memref<64xf32, #hivm.address_space<ub>>
    %20 = hivm.hir.pointer_cast(%c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
    %21 = hivm.hir.pointer_cast(%c67968_i64) : memref<64xf32, #hivm.address_space<ub>>
    %22 = hivm.hir.pointer_cast(%c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
    %23 = hivm.hir.pointer_cast(%c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
    %24 = hivm.hir.pointer_cast(%c66688_i64) : memref<64xf32, #hivm.address_space<ub>>
    %25 = hivm.hir.pointer_cast(%c66944_i64) : memref<64xf32, #hivm.address_space<ub>>
    %26 = hivm.hir.pointer_cast(%c67200_i64) : memref<64xf32, #hivm.address_space<ub>>
    %27 = hivm.hir.pointer_cast(%c66944_i64) : memref<64xf32, #hivm.address_space<ub>>
    %28 = hivm.hir.pointer_cast(%c67200_i64) : memref<64xf32, #hivm.address_space<ub>>
    %29 = hivm.hir.pointer_cast(%c66688_i64) : memref<64xf32, #hivm.address_space<ub>>
    %30 = hivm.hir.pointer_cast(%c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %31 = hivm.hir.pointer_cast(%c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
    %32 = hivm.hir.pointer_cast(%c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %33 = hivm.hir.pointer_cast(%c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
    %34 = hivm.hir.pointer_cast(%c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %35 = hivm.hir.pointer_cast(%c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
    %36 = arith.muli %arg7, %arg8 : i32
    %37 = arith.muli %36, %arg9 : i32
    %38 = hivm.hir.get_block_idx -> i64
    %39 = arith.trunci %38 : i64 to i32
    %40 = hivm.hir.pointer_cast(%c32896_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @_attn_fwd_mix_aiv_outlined_vf_6(%40) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %41 = hivm.hir.get_sub_block_idx -> i64
    %42 = arith.index_cast %41 : i64 to index
    %43 = affine.apply #map3()[%42]
    %44 = affine.apply #map4()[%42]
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    scf.for %arg10 = %39 to %37 step %c28_i32  : i32 {
      %45 = memref.load %alloca_0[%c0] : memref<1xi64>
      %46 = arith.remui %45, %c3_i64 : i64
      %47 = arith.cmpi eq, %46, %c1_i64 : i64
      %48 = arith.select %47, %1, %0 : memref<64xf32, #hivm.address_space<ub>>
      %49 = arith.cmpi eq, %46, %c2_i64 : i64
      %50 = arith.select %49, %2, %48 : memref<64xf32, #hivm.address_space<ub>>
      %51 = arith.select %47, %4, %3 : memref<64xf32, #hivm.address_space<ub>>
      %52 = arith.select %49, %5, %51 : memref<64xf32, #hivm.address_space<ub>>
      %53 = arith.select %47, %7, %6 : memref<64xf32, #hivm.address_space<ub>>
      %54 = arith.select %49, %8, %53 : memref<64xf32, #hivm.address_space<ub>>
      %55 = arith.select %47, %16, %15 : memref<64xf32, #hivm.address_space<ub>>
      %56 = arith.select %49, %17, %55 : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %57 = arith.remsi %arg10, %arg7 : i32
      %58 = arith.extsi %57 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %59 = arith.subi %c131072_i64, %58 : i64
      %60 = arith.maxsi %59, %c0_i64 : i64
      %61 = arith.ceildivsi %60, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %62 = arith.muli %61, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %63 = hivm.hir.pointer_cast(%c33152_i64) : memref<64xf32, #hivm.address_space<ub>>
      %64 = hivm.hir.pointer_cast(%c33408_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      func.call @_attn_fwd_mix_aiv_outlined_merged_vf_0(%63, %64) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %65 = hivm.hir.pointer_cast(%c50048_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %66 = hivm.hir.pointer_cast(%c58240_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %67 = hivm.hir.pointer_cast(%c66432_i64) : memref<64xf32, #hivm.address_space<ub>>
      %68 = hivm.hir.pointer_cast(%c49792_i64) : memref<64xf32, #hivm.address_space<ub>>
      %69 = hivm.hir.pointer_cast(%c235136_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %64 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_1 = memref.collapse_shape %69 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_1 : memref<4096xf32, #hivm.address_space<ub>>)
      %70 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%63 : memref<64xf32, #hivm.address_space<ub>>) outs(%70 : memref<64xf32, #hivm.address_space<ub>>)
      %71 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%40 : memref<64xf32, #hivm.address_space<ub>>) outs(%71 : memref<64xf32, #hivm.address_space<ub>>)
      %72 = arith.addi %62, %c1024_i64 : i64
      %73:5 = scf.for %arg11 = %c0_i64 to %72 step %c256_i64 iter_args(%arg12 = %69, %arg13 = %70, %arg14 = %71, %arg15 = %50, %arg16 = %52) -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i64 {
        %75 = memref.load %alloca[%c0] : memref<1xi64>
        %76 = arith.remui %75, %c3_i64 : i64
        %77 = arith.cmpi eq, %76, %c1_i64 : i64
        %78 = arith.select %77, %10, %9 : memref<64xf32, #hivm.address_space<ub>>
        %79 = arith.cmpi eq, %76, %c2_i64 : i64
        %80 = arith.select %79, %11, %78 : memref<64xf32, #hivm.address_space<ub>>
        %81 = arith.select %77, %13, %12 : memref<64xf32, #hivm.address_space<ub>>
        %82 = arith.select %79, %14, %81 : memref<64xf32, #hivm.address_space<ub>>
        %83 = arith.select %77, %19, %18 : memref<64xf32, #hivm.address_space<ub>>
        %84 = arith.select %79, %20, %83 : memref<64xf32, #hivm.address_space<ub>>
        %85 = arith.select %77, %22, %21 : memref<64xf32, #hivm.address_space<ub>>
        %86 = arith.select %79, %23, %85 : memref<64xf32, #hivm.address_space<ub>>
        %87 = arith.select %77, %25, %24 : memref<64xf32, #hivm.address_space<ub>>
        %88 = arith.select %79, %26, %87 : memref<64xf32, #hivm.address_space<ub>>
        %89 = arith.select %77, %28, %27 : memref<64xf32, #hivm.address_space<ub>>
        %90 = arith.select %79, %29, %89 : memref<64xf32, #hivm.address_space<ub>>
        %91 = arith.remui %75, %c2_i64 : i64
        %92 = arith.cmpi eq, %91, %c1_i64 : i64
        %93 = arith.select %92, %31, %30 : memref<256x64xf32, #hivm.address_space<ub>>
        %94 = arith.select %92, %33, %32 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %95 = arith.select %92, %35, %34 : memref<64x64xf32, #hivm.address_space<ub>>
        %96 = arith.subi %arg11, %c256_i64 : i64
        %97 = arith.subi %arg11, %c768_i64 : i64
        %98 = arith.divsi %97, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %99 = arith.divsi %96, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %100 = arith.remsi %97, %c16384_i64 : i64
        %101 = arith.remsi %96, %c16384_i64 : i64
        %102 = arith.muli %98, %c28_i64 : i64
        %103 = arith.addi %58, %102 : i64
        %104 = arith.trunci %103 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %105 = arith.remsi %98, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %106 = arith.remsi %99, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %107 = arith.cmpi eq, %105, %c1_i64 : i64
        %108 = arith.cmpi eq, %106, %c1_i64 : i64
        %109 = arith.cmpi eq, %100, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %110 = arith.cmpi eq, %101, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %111 = arith.cmpi eq, %100, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        %subview = memref.subview %94[%44, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %112 = arith.cmpi sge, %96, %c0_i64 : i64
        %113 = arith.cmpi slt, %96, %62 : i64
        %114 = arith.andi %112, %113 : i1
        %115:2 = scf.if %114 -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
          %137 = arith.select %108, %arg15, %arg13 : memref<64xf32, #hivm.address_space<ub>>
          %138 = arith.select %108, %arg16, %arg14 : memref<64xf32, #hivm.address_space<ub>>
          %139 = hivm.hir.pointer_cast(%c0_i64) : memref<4x257x16x1xf16, #hivm.address_space<ub>>
          %subview_2 = memref.subview %139[0, 0, 0, 0] [4, 256, 16, 1] [1, 1, 1, 1] : memref<4x257x16x1xf16, #hivm.address_space<ub>> to memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>
          %140 = hivm.hir.pointer_cast(%c233344_i64) : memref<64xf32, #hivm.address_space<ub>>
          %141 = hivm.hir.pointer_cast(%c233600_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          scf.if %110 {
            func.call @__cvp_attention_query_first_state(%93, %subview_2, %140, %88, %141) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          } else {
            func.call @_attn_fwd_mix_aiv_outlined_vf_3(%93, %subview_2, %137, %138, %140, %88, %141) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          }
          hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          %expand_shape = memref.expand_shape %subview_2 [[0], [1, 2], [3]] output_shape [4, 16, 16, 16] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> into memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>>
          %collapse_shape_3 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>
          %collapse_shape_4 = memref.collapse_shape %subview [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
          hivm.hir.copy ins(%collapse_shape_3 : memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
          %142 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%140 : memref<64xf32, #hivm.address_space<ub>>) outs(%142 : memref<64xf32, #hivm.address_space<ub>>)
          %143 = hivm.hir.pointer_cast(%c234112_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%142 : memref<64xf32, #hivm.address_space<ub>>) outs(%143 : memref<64xf32, #hivm.address_space<ub>>)
          %144 = arith.select %108, %arg13, %143 : memref<64xf32, #hivm.address_space<ub>>
          %145 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%141 : memref<64xf32, #hivm.address_space<ub>>) outs(%145 : memref<64xf32, #hivm.address_space<ub>>)
          %146 = hivm.hir.pointer_cast(%c234368_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%145 : memref<64xf32, #hivm.address_space<ub>>) outs(%146 : memref<64xf32, #hivm.address_space<ub>>)
          %147 = arith.select %108, %arg14, %146 : memref<64xf32, #hivm.address_space<ub>>
          %148 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg15 : memref<64xf32, #hivm.address_space<ub>>) outs(%148 : memref<64xf32, #hivm.address_space<ub>>)
          %149 = hivm.hir.pointer_cast(%c234624_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%148 : memref<64xf32, #hivm.address_space<ub>>) outs(%149 : memref<64xf32, #hivm.address_space<ub>>)
          %150 = arith.select %108, %140, %149 : memref<64xf32, #hivm.address_space<ub>>
          %151 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg16 : memref<64xf32, #hivm.address_space<ub>>) outs(%151 : memref<64xf32, #hivm.address_space<ub>>)
          %152 = hivm.hir.pointer_cast(%c234880_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%151 : memref<64xf32, #hivm.address_space<ub>>) outs(%152 : memref<64xf32, #hivm.address_space<ub>>)
          %153 = arith.select %108, %141, %152 : memref<64xf32, #hivm.address_space<ub>>
          %154 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%150 : memref<64xf32, #hivm.address_space<ub>>) outs(%154 : memref<64xf32, #hivm.address_space<ub>>)
          %155 = hivm.hir.pointer_cast(%c233344_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%153 : memref<64xf32, #hivm.address_space<ub>>) outs(%155 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%154 : memref<64xf32, #hivm.address_space<ub>>) outs(%84 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%155 : memref<64xf32, #hivm.address_space<ub>>) outs(%80 : memref<64xf32, #hivm.address_space<ub>>)
          scf.yield %147, %144 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg14, %arg13 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %116 = arith.cmpi sge, %97, %c0_i64 : i64
        %117 = arith.cmpi slt, %97, %62 : i64
        %118 = arith.andi %116, %117 : i1
        %119 = scf.if %118 -> (memref<64x64xf32, #hivm.address_space<ub>>) {
          %137 = arith.divsi %104, %c128_i32 : i32
          %138 = arith.remsi %104, %c128_i32 : i32
          %139 = arith.divsi %137, %c8_i32 : i32
          %140 = arith.remsi %137, %c8_i32 : i32
          %141 = arith.extsi %139 : i32 to i64
          %142 = arith.muli %141, %c8388608_i64 : i64
          %143 = arith.extsi %140 : i32 to i64
          %144 = arith.muli %143, %c1048576_i64 : i64
          %145 = arith.addi %142, %144 : i64
          %146 = arith.index_cast %145 : i64 to index
          %147 = arith.muli %138, %c128_i32 : i32
          %148 = arith.maxsi %147, %c0_i32 : i32
          %149 = arith.index_cast %148 : i32 to index
          %150 = affine.apply #map1()[%146, %149]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%150], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_2 = memref.subview %reinterpret_cast[%43, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %151 = scf.if %111 -> (memref<64x64xf32, #hivm.address_space<ub>>) {
            %152 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%115#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%152 : memref<64xf32, #hivm.address_space<ub>>)
            %153 = hivm.hir.pointer_cast(%c102528_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%152 : memref<64xf32, #hivm.address_space<ub>>) outs(%153 : memref<64xf32, #hivm.address_space<ub>>)
            %154 = arith.select %107, %86, %153 : memref<64xf32, #hivm.address_space<ub>>
            %155 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%115#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%155 : memref<64xf32, #hivm.address_space<ub>>)
            %156 = hivm.hir.pointer_cast(%c102784_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%155 : memref<64xf32, #hivm.address_space<ub>>) outs(%156 : memref<64xf32, #hivm.address_space<ub>>)
            %157 = arith.select %107, %82, %156 : memref<64xf32, #hivm.address_space<ub>>
            %158 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            %159 = arith.muli %137, %c16384_i32 : i32
            %160 = arith.index_cast %159 : i32 to index
            %161 = arith.index_cast %147 : i32 to index
            %162 = affine.apply #map5()[%160, %161]
            %reinterpret_cast_3 = memref.reinterpret_cast %arg5 to offset: [%162], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_4 = memref.subview %reinterpret_cast_3[%43] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %163 = arith.select %107, %68, %67 : memref<64xf32, #hivm.address_space<ub>>
            %164 = hivm.hir.pointer_cast(%c103040_i64) : memref<64x64xf16, #hivm.address_space<ub>>
            func.call @__cvp_attention_last_output(%157, %154, %158, %154, %arg12, %164, %90, %95) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
            hivm.hir.copy ins(%158 : memref<64xf32, #hivm.address_space<ub>>) outs(%163 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%163 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_4 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
            %165 = arith.select %107, %66, %65 : memref<64x64xf16, #hivm.address_space<ub>>
            %collapse_shape_5 = memref.collapse_shape %164 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            %collapse_shape_6 = memref.collapse_shape %165 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
            hivm.hir.copy ins(%collapse_shape_5 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<4096xf16, #hivm.address_space<ub>>)
            hivm.hir.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            %collapse_shape_7 = memref.collapse_shape %subview_2 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
            hivm.hir.store ins(%collapse_shape_6 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_7 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.set_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
            scf.yield %arg12 : memref<64x64xf32, #hivm.address_space<ub>>
          } else {
            scf.if %109 {
              func.call @__cvp_attention_query_first_output(%90, %95, %95) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            } else {
              func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%90, %95, %arg12, %95) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            }
            %collapse_shape_3 = memref.collapse_shape %95 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %152 = hivm.hir.pointer_cast(%c103040_i64) : memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%collapse_shape_3 : memref<4096xf32, #hivm.address_space<ub>>) outs(%152 : memref<4096xf32, #hivm.address_space<ub>>)
            %153 = hivm.hir.pointer_cast(%c235136_i64) : memref<64x64xf32, #hivm.address_space<ub>>
            %collapse_shape_4 = memref.collapse_shape %153 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%152 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_4 : memref<4096xf32, #hivm.address_space<ub>>)
            scf.yield %153 : memref<64x64xf32, #hivm.address_space<ub>>
          }
          scf.yield %151 : memref<64x64xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg12 : memref<64x64xf32, #hivm.address_space<ub>>
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        %120 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %121 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %122 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %123 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%115#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%120 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%121 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%122 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%123 : memref<64xf32, #hivm.address_space<ub>>)
        %124 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %125 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %126 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %127 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%115#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%124 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%125 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%126 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%115#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%127 : memref<64xf32, #hivm.address_space<ub>>)
        %128 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %129 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %130 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %131 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%120 : memref<64xf32, #hivm.address_space<ub>>) outs(%128 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%121 : memref<64xf32, #hivm.address_space<ub>>) outs(%129 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%122 : memref<64xf32, #hivm.address_space<ub>>) outs(%130 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%123 : memref<64xf32, #hivm.address_space<ub>>) outs(%131 : memref<64xf32, #hivm.address_space<ub>>)
        %132 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %133 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %134 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %135 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%124 : memref<64xf32, #hivm.address_space<ub>>) outs(%132 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%125 : memref<64xf32, #hivm.address_space<ub>>) outs(%133 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%126 : memref<64xf32, #hivm.address_space<ub>>) outs(%134 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%127 : memref<64xf32, #hivm.address_space<ub>>) outs(%135 : memref<64xf32, #hivm.address_space<ub>>)
        %136 = arith.addi %75, %c1_i64 : i64
        memref.store %136, %alloca[%c0] : memref<1xi64>
        scf.yield %119, %131, %135, %56, %54 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      }
      hivm.hir.set_ctrl true at ctrl[60]
      %74 = arith.addi %45, %c1_i64 : i64
      memref.store %74, %alloca_0[%c0] : memref<1xi64>
    } {autoblockify.subloop}
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID0>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID1>]
    hivm.hir.wait_flag[<PIPE_MTE3>, <PIPE_V>, <EVENT_ID2>]
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.pipe_barrier[<PIPE_ALL>]
    return
  }
  func.func @__cvp_attention_query_first_output(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>, %arg2: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_first_output", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview_0[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_2[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %res_3 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %1 = ave.hir.vmuls %res_3, %cst, %0 : vector<64xf32>, f32, vector<64xi1>
      %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %3 = ave.hir.vadd %res, %1, %2 : vector<64xf32>, vector<64xi1>
      %subview_4 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %4 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      ave.hir.masked_store <NORM_B32> %subview_4[%c0], %4, %3 {functionType = #ave.func_dist_type<norm>, hivm.is_continuous} : memref<64xf32, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @__cvp_attention_last_output(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x64xf32, #hivm.address_space<ub>>, %arg5: memref<64x64xf16, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>, %arg7: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_last_output", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
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
    scf.for %arg8 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg4[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg5[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_3 = memref.subview %subview_1[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_4 = ave.hir.vload <NORM> %subview_3[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %subview_5 = memref.subview %arg6[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %res_6 = ave.hir.vload <BRC_B32> %subview_5[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %res_7 = ave.hir.vload <BRC_B32> %subview[%c0] {ave.unaligned_ub_access = #ave.unaligned_ub_access, functionType = #ave.func_dist_type<norm>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>> into vector<64xf32>
      %subview_8 = memref.subview %arg7[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_9 = memref.subview %subview_8[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res_10 = ave.hir.vload <NORM> %subview_9[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %6 = ave.hir.vmul %res_4, %res_6, %5 : vector<64xf32>, vector<64xi1>
      %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %8 = ave.hir.vadd %res_10, %6, %7 : vector<64xf32>, vector<64xi1>
      %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %10 = ave.hir.vdiv %8, %res_7, %9 : vector<64xf32>, vector<64xi1>
      %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %12 = ave.hir.vtruncf %10, <rint>, false, <part_even>, %11 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_11 = memref.subview %subview_2[0, 0] [1, 64] [1, 1] : memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf16, #map, #hivm.address_space<ub>>
      %13 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %14 = builtin.unrealized_conversion_cast %12 : vector<64xf16> to vector<128xf16>
      %15 = llvm.bitcast %14 : vector<128xf16> to vector<64xf32>
      ave.hir.masked_store <PK_B32> %subview_11[%c0], %13, %15 {functionType = #ave.func_dist_type<pk>, hivm.is_continuous} : memref<64xf16, #map, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    }
    return
  }
  func.func @__cvp_attention_query_first_state(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_first_state", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c4112 = arith.constant 4112 : index
    %c0 = arith.constant 0 : index
    %c256 = arith.constant 256 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0xFF800000 : f32
    %cst_0 = arith.constant 5.000000e-01 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %0 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %1 = ave.hir.broadcast %cst_1, %0 : f32, vector<64xi1> -> vector<64xf32>
    %2 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %3 = ave.hir.broadcast %cst, %2 : f32, vector<64xi1> -> vector<64xf32>
    %4 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %3) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg5, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_2[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %15 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %16 = ave.hir.vmax %arg6, %res, %15 : vector<64xf32>, vector<64xi1>
      scf.yield %16 : vector<64xf32>
    }
    %5 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %6 = ave.hir.vmuls %4, %cst_0, %5 : vector<64xf32>, f32, vector<64xi1>
    %7 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg4[%c0], %7, %6 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %8 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %1) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg5, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %subview[0, 0] [1, 64] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>> to memref<64xf32, #map, #hivm.address_space<ub>>
      %res = ave.hir.vload <NORM> %subview_2[%c0] {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #map, #hivm.address_space<ub>> into vector<64xf32>
      %15 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %16 = ave.hir.vmuls %res, %cst_0, %15 : vector<64xf32>, f32, vector<64xi1>
      %17 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %18 = ave.hir.vexpdif %16, %6, %17 : vector<64xf32>, vector<64xi1>
      %19 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %20 = ave.hir.vadd %arg6, %18, %19 : vector<64xf32>, vector<64xi1>
      %21 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
      %22 = ave.hir.vtruncf %18, <rint>, false, <part_even>, %21 {functionType = #ave.func_dist_type<even>} : vector<64xf32>, vector<64xf16>, vector<64xi1>
      %subview_3 = memref.subview %arg1[0, %arg5, 0] [4, 1, 16] [1, 1, 1] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_4 = memref.subview %subview_3[0, 0, 0] [4, 1, 16] [1, 1, 1] : memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>> to memref<4x16xf16, #map2, #hivm.address_space<ub>>
      %23 = ave.hir.pge <H> {functionType = #ave.func_dist_type<pb16>} : vector<128xi1>
      %res1, %res2 = ave.hir.vdintlv %22, %22 : vector<64xf16>, vector<64xf16>
      ave.hir.store_with_stride %subview_4[%c0, %c0], %c4112, %23, %res1 : memref<4x16xf16, #map2, #hivm.address_space<ub>>, vector<128xi1>, vector<64xf16>
      scf.yield %20 : vector<64xf32>
    }
    %9 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %10 = ave.hir.vexpdif %3, %6, %9 : vector<64xf32>, vector<64xi1>
    %11 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg3[%c0], %11, %10 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    %12 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    %13 = ave.hir.vadd %10, %8, %12 : vector<64xf32>, vector<64xi1>
    %14 = ave.hir.pge <ALL> {functionType = #ave.func_dist_type<pb32>} : vector<64xi1>
    ave.hir.masked_store <NORM_B32> %arg2[%c0], %14, %13 {functionType = #ave.func_dist_type<norm>} : memref<64xf32, #hivm.address_space<ub>>, vector<64xi1>, vector<64xf32>
    return
  }
}



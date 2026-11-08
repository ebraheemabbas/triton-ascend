// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: bishengir-opt %s --hivm-verify-cvp-q-cache="mode=cached require-match=true" --split-input-file --verify-each -o %t
// Full captured modules retain their target specs, AIC/AIV functions and VF helpers.
// Donor scheduled-scalar-counter.mlir SHA256 cd177ffd0b52f7af8cae11dc7eab0643f5cd68ec572f3da1e9c68a1528722aee
// Donor scheduled-clock.mlir SHA256 842f845588e1cea011100b823f224ae800fa7ec41a769edf2c0d539bb11217ce
// These positives check only the Stage8 spatial contract, not completion or whole-kernel safety.

// CASE: original complete scheduled module before scalar counter lowering
#map = affine_map<(d0) -> (d0, 0)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 4)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.first_output_specializations = 1 : i64, hivm.cvp_attention.first_state_specializations = 1 : i64, hivm.cvp_attention.last_output_specializations = 1 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
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
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}, SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c512_i64 = arith.constant 512 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c229376_i64 = arith.constant 229376 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c167808_i64 = arith.constant 167808 : i64
    %c102272_i64 = arith.constant 102272 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c85888_i64 = arith.constant 85888 : i64
    %c69504_i64 = arith.constant 69504 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c0_i64 = arith.constant 0 : i64
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
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    annotation.mark %1 {logical_block_num} : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg10, %arg7 : i32
      %5 = arith.extsi %4 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %6 = arith.subi %c131072_i64, %5 : i64
      %7 = arith.maxsi %6, %c0_i64 : i64
      %8 = arith.ceildivsi %7, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %9 = arith.muli %8, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %10 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %11 = hivm.hir.pointer_cast(%c16384_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %12 = arith.addi %9, %c1024_i64 : i64
      scf.for %arg11 = %c0_i64 to %12 step %c256_i64  : i64 {
        %13 = arith.subi %arg11, %c256_i64 : i64
        %14 = arith.subi %arg11, %c512_i64 : i64
        %15 = hivm.hir.pointer_cast(%c131072_i64, %c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %16 = hivm.hir.pointer_cast(%c131072_i64, %c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %17 = hivm.hir.pointer_cast(%c131072_i64, %c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %18 = hivm.hir.pointer_cast(%c131072_i64, %c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %15 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %16 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %17 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %18 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
        %19 = hivm.hir.pointer_cast(%c131072_i64, %c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %20 = hivm.hir.pointer_cast(%c131072_i64, %c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %21 = hivm.hir.pointer_cast(%c131072_i64, %c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %22 = hivm.hir.pointer_cast(%c131072_i64, %c262144_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %19 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %20 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %21 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %22 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %23 = hivm.hir.pointer_cast(%c0_i64, %c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %24 = hivm.hir.pointer_cast(%c0_i64, %c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %25 = hivm.hir.pointer_cast(%c0_i64, %c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %26 = hivm.hir.pointer_cast(%c0_i64, %c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %23 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %24 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %25 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        annotation.mark %26 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
        %27 = hivm.hir.pointer_cast(%c98304_i64, %c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %28 = hivm.hir.pointer_cast(%c98304_i64, %c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %29 = hivm.hir.pointer_cast(%c98304_i64, %c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %30 = hivm.hir.pointer_cast(%c98304_i64, %c229376_i64) : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %27 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %28 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %29 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %30 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
        %31 = hivm.hir.pointer_cast(%c102272_i64, %c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %32 = hivm.hir.pointer_cast(%c167808_i64, %c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %33 = hivm.hir.pointer_cast(%c102272_i64, %c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %34 = hivm.hir.pointer_cast(%c167808_i64, %c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %34 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %33 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %32 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %31 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        %35 = hivm.hir.pointer_cast(%c32768_i64, %c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %36 = hivm.hir.pointer_cast(%c163840_i64, %c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %37 = hivm.hir.pointer_cast(%c32768_i64, %c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %38 = hivm.hir.pointer_cast(%c163840_i64, %c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %38 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %37 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %36 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %35 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %39 = hivm.hir.pointer_cast(%c69504_i64, %c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %40 = hivm.hir.pointer_cast(%c85888_i64, %c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %41 = hivm.hir.pointer_cast(%c69504_i64, %c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %42 = hivm.hir.pointer_cast(%c85888_i64, %c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %42 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %41 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %40 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %39 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        %43 = arith.divsi %14, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %44 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %45 = arith.remsi %14, %c16384_i64 : i64
        %46 = arith.remsi %arg11, %c16384_i64 : i64
        %47 = arith.trunci %45 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %48 = arith.trunci %46 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %49 = arith.muli %43, %c28_i64 : i64
        %50 = arith.muli %44, %c28_i64 : i64
        %51 = arith.addi %5, %49 : i64
        %52 = arith.addi %5, %50 : i64
        %53 = arith.trunci %51 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %54 = arith.trunci %52 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %55 = arith.remsi %44, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %56 = arith.cmpi eq, %55, %c1_i64 : i64
        %57 = arith.cmpi eq, %46, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        %58 = arith.cmpi sge, %arg11, %c0_i64 : i64
        %59 = arith.cmpi slt, %arg11, %9 : i64
        %60 = arith.andi %58, %59 : i1
        scf.if %60 {
          %67 = arith.divsi %54, %c128_i32 : i32
          %68 = arith.remsi %54, %c128_i32 : i32
          %69 = arith.divsi %67, %c8_i32 : i32
          %70 = arith.remsi %67, %c8_i32 : i32
          %71 = arith.extsi %69 : i32 to i64
          %72 = arith.muli %71, %c8388608_i64 : i64
          %73 = arith.extsi %70 : i32 to i64
          %74 = arith.muli %73, %c1048576_i64 : i64
          %75 = arith.addi %72, %74 : i64
          %76 = arith.index_cast %75 : i64 to index
          %77 = arith.muli %68, %c128_i32 : i32
          %78 = arith.maxsi %77, %c0_i32 : i32
          %79 = arith.index_cast %78 : i32 to index
          %80 = affine.apply #map1()[%76, %79]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%80], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %81 = arith.select %56, %11, %10 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          scf.if %57 {
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%81 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
          }
          %82 = arith.maxsi %48, %c0_i32 : i32
          %83 = arith.index_cast %82 : i32 to index
          %84 = affine.apply #map1()[%76, %83]
          %reinterpret_cast_0 = memref.reinterpret_cast %arg3 to offset: [%84], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_0 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%30 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          %cache_even = arith.constant 16384 : i64
          %cache_odd = arith.constant 49152 : i64
          %cache_offset = arith.select %56, %cache_odd, %cache_even : i64
          hivm.hir.mmadL1 {cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64},already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%30, %81, %true, %c256, %c64, %c128 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%26 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%57, %44, %cache_offset)
          annotation.mark %31 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%26 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) outs(%31 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
        }
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %61 = arith.cmpi sge, %13, %c0_i64 : i64
        %62 = arith.cmpi slt, %13, %9 : i64
        %63 = arith.andi %61, %62 : i1
        scf.if %63 {
          annotation.mark %36 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        }
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        %64 = arith.cmpi sge, %14, %c0_i64 : i64
        %65 = arith.cmpi slt, %14, %9 : i64
        %66 = arith.andi %64, %65 : i1
        scf.if %66 {
          %67 = arith.divsi %53, %c128_i32 : i32
          %68 = arith.divsi %67, %c8_i32 : i32
          %69 = arith.remsi %67, %c8_i32 : i32
          %70 = arith.extsi %68 : i32 to i64
          %71 = arith.muli %70, %c8388608_i64 : i64
          %72 = arith.extsi %69 : i32 to i64
          %73 = arith.muli %72, %c1048576_i64 : i64
          %74 = arith.addi %71, %73 : i64
          %75 = arith.index_cast %74 : i64 to index
          %76 = arith.maxsi %47, %c0_i32 : i32
          %77 = arith.index_cast %76 : i32 to index
          %78 = affine.apply #map1()[%75, %77]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%78], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%20 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.mmadL1 {cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64},a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%37, %20, %true, %c128, %c256, %c64 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%16 : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
          annotation.mark %41 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%16 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) outs(%41 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
        }
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
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
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
    %c67200_i64 = arith.constant 67200 : i64
    %c66944_i64 = arith.constant 66944 : i64
    %c66688_i64 = arith.constant 66688 : i64
    %c233344_i64 = arith.constant 233344 : i64
    %c0_i64 = arith.constant 0 : i64
    %c167808_i64 = arith.constant 167808 : i64
    %c102272_i64 = arith.constant 102272 : i64
    %c163840_i64 = arith.constant 163840 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c85888_i64 = arith.constant 85888 : i64
    %c69504_i64 = arith.constant 69504 : i64
    %c68480_i64 = arith.constant 68480 : i64
    %c67456_i64 = arith.constant 67456 : i64
    %c235136_i64 = arith.constant 235136 : i64
    %c69248_i64 = arith.constant 69248 : i64
    %c68992_i64 = arith.constant 68992 : i64
    %c68736_i64 = arith.constant 68736 : i64
    %c68224_i64 = arith.constant 68224 : i64
    %c67968_i64 = arith.constant 67968 : i64
    %c67712_i64 = arith.constant 67712 : i64
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
    %0 = arith.muli %arg7, %arg8 : i32
    %1 = arith.muli %0, %arg9 : i32
    %2 = hivm.hir.get_block_idx -> i64
    %3 = arith.trunci %2 : i64 to i32
    %4 = hivm.hir.pointer_cast(%c32896_i64) : memref<64xf32, #hivm.address_space<ub>>
    call @_attn_fwd_mix_aiv_outlined_vf_6(%4) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %5 = hivm.hir.get_sub_block_idx -> i64
    %6 = arith.index_cast %5 : i64 to index
    %7 = affine.apply #map2()[%6]
    %8 = affine.apply #map3()[%6]
    annotation.mark %1 {logical_block_num} : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %9 = arith.remsi %arg10, %arg7 : i32
      %10 = arith.extsi %9 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %11 = arith.subi %c131072_i64, %10 : i64
      %12 = arith.maxsi %11, %c0_i64 : i64
      %13 = arith.ceildivsi %12, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %14 = arith.muli %13, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %15 = hivm.hir.pointer_cast(%c33152_i64) : memref<64xf32, #hivm.address_space<ub>>
      %16 = hivm.hir.pointer_cast(%c33408_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      func.call @_attn_fwd_mix_aiv_outlined_merged_vf_0(%15, %16) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %17 = hivm.hir.pointer_cast(%c50048_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %18 = hivm.hir.pointer_cast(%c58240_i64) : memref<64x64xf16, #hivm.address_space<ub>>
      %19 = hivm.hir.pointer_cast(%c66432_i64) : memref<64xf32, #hivm.address_space<ub>>
      %20 = hivm.hir.pointer_cast(%c49792_i64) : memref<64xf32, #hivm.address_space<ub>>
      %21 = hivm.hir.pointer_cast(%c67712_i64, %c67968_i64, %c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
      %22 = hivm.hir.pointer_cast(%c68736_i64, %c68992_i64, %c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
      %23 = hivm.hir.pointer_cast(%c235136_i64) : memref<64x64xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %16 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_0 = memref.collapse_shape %23 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_0 : memref<4096xf32, #hivm.address_space<ub>>)
      %24 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%15 : memref<64xf32, #hivm.address_space<ub>>) outs(%24 : memref<64xf32, #hivm.address_space<ub>>)
      %25 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%4 : memref<64xf32, #hivm.address_space<ub>>) outs(%25 : memref<64xf32, #hivm.address_space<ub>>)
      %26 = arith.addi %14, %c1024_i64 : i64
      %27:5 = scf.for %arg11 = %c0_i64 to %26 step %c256_i64 iter_args(%arg12 = %23, %arg13 = %24, %arg14 = %25, %arg15 = %21, %arg16 = %22) -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i64 {
        %28 = arith.subi %arg11, %c256_i64 : i64
        %29 = arith.subi %arg11, %c768_i64 : i64
        %30 = hivm.hir.pointer_cast(%c68992_i64, %c69248_i64, %c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
        %31 = hivm.hir.pointer_cast(%c68736_i64, %c68992_i64, %c69248_i64) : memref<64xf32, #hivm.address_space<ub>>
        %32 = hivm.hir.pointer_cast(%c69248_i64, %c68736_i64, %c68992_i64) : memref<64xf32, #hivm.address_space<ub>>
        %33 = hivm.hir.pointer_cast(%c68992_i64, %c69248_i64, %c68736_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %33 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %32 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %31 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %30 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %34 = hivm.hir.pointer_cast(%c67968_i64, %c68224_i64, %c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
        %35 = hivm.hir.pointer_cast(%c67712_i64, %c67968_i64, %c68224_i64) : memref<64xf32, #hivm.address_space<ub>>
        %36 = hivm.hir.pointer_cast(%c68224_i64, %c67712_i64, %c67968_i64) : memref<64xf32, #hivm.address_space<ub>>
        %37 = hivm.hir.pointer_cast(%c67968_i64, %c68224_i64, %c67712_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %37 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %36 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %35 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %34 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %38 = hivm.hir.pointer_cast(%c66944_i64, %c67200_i64, %c66688_i64) : memref<64xf32, #hivm.address_space<ub>>
        %39 = hivm.hir.pointer_cast(%c66688_i64, %c66944_i64, %c67200_i64) : memref<64xf32, #hivm.address_space<ub>>
        %40 = hivm.hir.pointer_cast(%c67200_i64, %c66688_i64, %c66944_i64) : memref<64xf32, #hivm.address_space<ub>>
        %41 = hivm.hir.pointer_cast(%c66944_i64, %c67200_i64, %c66688_i64) : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %41 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %40 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %39 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %38 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %42 = hivm.hir.pointer_cast(%c102272_i64, %c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %43 = hivm.hir.pointer_cast(%c167808_i64, %c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %44 = hivm.hir.pointer_cast(%c102272_i64, %c167808_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        %45 = hivm.hir.pointer_cast(%c167808_i64, %c102272_i64) : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %45 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %44 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %43 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %42 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
        %46 = hivm.hir.pointer_cast(%c32768_i64, %c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %47 = hivm.hir.pointer_cast(%c163840_i64, %c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %48 = hivm.hir.pointer_cast(%c32768_i64, %c163840_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %49 = hivm.hir.pointer_cast(%c163840_i64, %c32768_i64) : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %49 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %48 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %47 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %46 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %50 = hivm.hir.pointer_cast(%c69504_i64, %c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %51 = hivm.hir.pointer_cast(%c85888_i64, %c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %52 = hivm.hir.pointer_cast(%c69504_i64, %c85888_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        %53 = hivm.hir.pointer_cast(%c85888_i64, %c69504_i64) : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %53 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %52 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %51 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %50 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
        %54 = arith.divsi %29, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %55 = arith.divsi %28, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %56 = arith.remsi %29, %c16384_i64 : i64
        %57 = arith.remsi %28, %c16384_i64 : i64
        %58 = arith.muli %54, %c28_i64 : i64
        %59 = arith.addi %10, %58 : i64
        %60 = arith.trunci %59 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %61 = arith.remsi %54, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %62 = arith.remsi %55, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %63 = arith.cmpi eq, %61, %c1_i64 : i64
        %64 = arith.cmpi eq, %62, %c1_i64 : i64
        %65 = arith.cmpi eq, %56, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %66 = arith.cmpi eq, %57, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %67 = arith.cmpi eq, %56, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        annotation.mark %53 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %52 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %51 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
        annotation.mark %50 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
        %subview = memref.subview %47[%8, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
        annotation.mark %45 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %44 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %43 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
        annotation.mark %42 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %68 = arith.cmpi sge, %28, %c0_i64 : i64
        %69 = arith.cmpi slt, %28, %14 : i64
        %70 = arith.andi %68, %69 : i1
        %71:2 = scf.if %70 -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
          %92 = arith.select %64, %arg15, %arg13 : memref<64xf32, #hivm.address_space<ub>>
          %93 = arith.select %64, %arg16, %arg14 : memref<64xf32, #hivm.address_space<ub>>
          %94 = hivm.hir.pointer_cast(%c0_i64) : memref<4x257x16x1xf16, #hivm.address_space<ub>>
          %subview_1 = memref.subview %94[0, 0, 0, 0] [4, 256, 16, 1] [1, 1, 1, 1] : memref<4x257x16x1xf16, #hivm.address_space<ub>> to memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>
          %95 = hivm.hir.pointer_cast(%c233344_i64) : memref<64xf32, #hivm.address_space<ub>>
          %96 = hivm.hir.pointer_cast(%c233600_i64) : memref<64xf32, #hivm.address_space<ub>>
          scf.if %66 {
            func.call @__cvp_attention_query_first_state(%43, %subview_1, %95, %39, %96) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          } else {
            func.call @_attn_fwd_mix_aiv_outlined_vf_3(%43, %subview_1, %92, %93, %95, %39, %96) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          }
          %expand_shape = memref.expand_shape %subview_1 [[0], [1, 2], [3]] output_shape [4, 16, 16, 16] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> into memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>>
          annotation.mark %47 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %collapse_shape_2 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4112, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>
          %collapse_shape_3 = memref.collapse_shape %subview [[0], [1, 2, 3]] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%collapse_shape_2 : memref<4x4096xf16, strided<[4112, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<4x4096xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          %97 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%95 : memref<64xf32, #hivm.address_space<ub>>) outs(%97 : memref<64xf32, #hivm.address_space<ub>>)
          %98 = hivm.hir.pointer_cast(%c234112_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%97 : memref<64xf32, #hivm.address_space<ub>>) outs(%98 : memref<64xf32, #hivm.address_space<ub>>)
          %99 = arith.select %64, %arg13, %98 : memref<64xf32, #hivm.address_space<ub>>
          %100 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%96 : memref<64xf32, #hivm.address_space<ub>>) outs(%100 : memref<64xf32, #hivm.address_space<ub>>)
          %101 = hivm.hir.pointer_cast(%c234368_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%100 : memref<64xf32, #hivm.address_space<ub>>) outs(%101 : memref<64xf32, #hivm.address_space<ub>>)
          %102 = arith.select %64, %arg14, %101 : memref<64xf32, #hivm.address_space<ub>>
          %103 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg15 : memref<64xf32, #hivm.address_space<ub>>) outs(%103 : memref<64xf32, #hivm.address_space<ub>>)
          %104 = hivm.hir.pointer_cast(%c234624_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%103 : memref<64xf32, #hivm.address_space<ub>>) outs(%104 : memref<64xf32, #hivm.address_space<ub>>)
          %105 = arith.select %64, %95, %104 : memref<64xf32, #hivm.address_space<ub>>
          %106 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%arg16 : memref<64xf32, #hivm.address_space<ub>>) outs(%106 : memref<64xf32, #hivm.address_space<ub>>)
          %107 = hivm.hir.pointer_cast(%c234880_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%106 : memref<64xf32, #hivm.address_space<ub>>) outs(%107 : memref<64xf32, #hivm.address_space<ub>>)
          %108 = arith.select %64, %96, %107 : memref<64xf32, #hivm.address_space<ub>>
          %109 = hivm.hir.pointer_cast(%c233856_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%105 : memref<64xf32, #hivm.address_space<ub>>) outs(%109 : memref<64xf32, #hivm.address_space<ub>>)
          %110 = hivm.hir.pointer_cast(%c233344_i64) : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%108 : memref<64xf32, #hivm.address_space<ub>>) outs(%110 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%109 : memref<64xf32, #hivm.address_space<ub>>) outs(%35 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%110 : memref<64xf32, #hivm.address_space<ub>>) outs(%31 : memref<64xf32, #hivm.address_space<ub>>)
          scf.yield %102, %99 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg14, %arg13 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %72 = arith.cmpi sge, %29, %c0_i64 : i64
        %73 = arith.cmpi slt, %29, %14 : i64
        %74 = arith.andi %72, %73 : i1
        %75 = scf.if %74 -> (memref<64x64xf32, #hivm.address_space<ub>>) {
          %92 = arith.divsi %60, %c128_i32 : i32
          %93 = arith.remsi %60, %c128_i32 : i32
          %94 = arith.divsi %92, %c8_i32 : i32
          %95 = arith.remsi %92, %c8_i32 : i32
          %96 = arith.extsi %94 : i32 to i64
          %97 = arith.muli %96, %c8388608_i64 : i64
          %98 = arith.extsi %95 : i32 to i64
          %99 = arith.muli %98, %c1048576_i64 : i64
          %100 = arith.addi %97, %99 : i64
          %101 = arith.index_cast %100 : i64 to index
          %102 = arith.muli %93, %c128_i32 : i32
          %103 = arith.maxsi %102, %c0_i32 : i32
          %104 = arith.index_cast %103 : i32 to index
          %105 = affine.apply #map1()[%101, %104]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%105], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_1 = memref.subview %reinterpret_cast[%7, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %106 = scf.if %67 -> (memref<64x64xf32, #hivm.address_space<ub>>) {
            %107 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%71#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%107 : memref<64xf32, #hivm.address_space<ub>>)
            %108 = hivm.hir.pointer_cast(%c102528_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%107 : memref<64xf32, #hivm.address_space<ub>>) outs(%108 : memref<64xf32, #hivm.address_space<ub>>)
            %109 = arith.select %63, %37, %108 : memref<64xf32, #hivm.address_space<ub>>
            %110 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%71#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%110 : memref<64xf32, #hivm.address_space<ub>>)
            %111 = hivm.hir.pointer_cast(%c102784_i64) : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%110 : memref<64xf32, #hivm.address_space<ub>>) outs(%111 : memref<64xf32, #hivm.address_space<ub>>)
            %112 = arith.select %63, %33, %111 : memref<64xf32, #hivm.address_space<ub>>
            %113 = hivm.hir.pointer_cast(%c102272_i64) : memref<64xf32, #hivm.address_space<ub>>
            %114 = arith.muli %92, %c16384_i32 : i32
            %115 = arith.index_cast %114 : i32 to index
            %116 = arith.index_cast %102 : i32 to index
            %117 = affine.apply #map4()[%115, %116]
            %reinterpret_cast_2 = memref.reinterpret_cast %arg5 to offset: [%117], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_3 = memref.subview %reinterpret_cast_2[%7] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %118 = arith.select %63, %20, %19 : memref<64xf32, #hivm.address_space<ub>>
            %119 = hivm.hir.pointer_cast(%c103040_i64) : memref<64x64xf16, #hivm.address_space<ub>>
            func.call @__cvp_attention_last_output(%112, %109, %113, %109, %arg12, %119, %41, %53) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.copy ins(%113 : memref<64xf32, #hivm.address_space<ub>>) outs(%118 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.store ins(%118 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_3 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            %120 = arith.select %63, %18, %17 : memref<64x64xf16, #hivm.address_space<ub>>
            %collapse_shape_4 = memref.collapse_shape %119 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            %collapse_shape_5 = memref.collapse_shape %120 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            hivm.hir.copy ins(%collapse_shape_4 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_5 : memref<4096xf16, #hivm.address_space<ub>>)
            %collapse_shape_6 = memref.collapse_shape %subview_1 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_5 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_6 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            scf.yield %arg12 : memref<64x64xf32, #hivm.address_space<ub>>
          } else {
            scf.if %65 {
              func.call @__cvp_attention_query_first_output(%41, %53, %53) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            } else {
              func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%41, %53, %arg12, %53) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            }
            %collapse_shape_2 = memref.collapse_shape %53 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            %107 = hivm.hir.pointer_cast(%c103040_i64) : memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%collapse_shape_2 : memref<4096xf32, #hivm.address_space<ub>>) outs(%107 : memref<4096xf32, #hivm.address_space<ub>>)
            %108 = hivm.hir.pointer_cast(%c235136_i64) : memref<64x64xf32, #hivm.address_space<ub>>
            %collapse_shape_3 = memref.collapse_shape %108 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%107 : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_3 : memref<4096xf32, #hivm.address_space<ub>>)
            scf.yield %108 : memref<64x64xf32, #hivm.address_space<ub>>
          }
          scf.yield %106 : memref<64x64xf32, #hivm.address_space<ub>>
        } else {
          scf.yield %arg12 : memref<64x64xf32, #hivm.address_space<ub>>
        }
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        %76 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %77 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %78 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        %79 = hivm.hir.pointer_cast(%c69504_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%71#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%76 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%77 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%78 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#1 : memref<64xf32, #hivm.address_space<ub>>) outs(%79 : memref<64xf32, #hivm.address_space<ub>>)
        %80 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %81 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %82 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        %83 = hivm.hir.pointer_cast(%c69760_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%71#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%80 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%81 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%82 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%71#0 : memref<64xf32, #hivm.address_space<ub>>) outs(%83 : memref<64xf32, #hivm.address_space<ub>>)
        %84 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %85 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %86 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        %87 = hivm.hir.pointer_cast(%c67456_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%76 : memref<64xf32, #hivm.address_space<ub>>) outs(%84 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%77 : memref<64xf32, #hivm.address_space<ub>>) outs(%85 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%78 : memref<64xf32, #hivm.address_space<ub>>) outs(%86 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%79 : memref<64xf32, #hivm.address_space<ub>>) outs(%87 : memref<64xf32, #hivm.address_space<ub>>)
        %88 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %89 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %90 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        %91 = hivm.hir.pointer_cast(%c68480_i64) : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%80 : memref<64xf32, #hivm.address_space<ub>>) outs(%88 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%81 : memref<64xf32, #hivm.address_space<ub>>) outs(%89 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%82 : memref<64xf32, #hivm.address_space<ub>>) outs(%90 : memref<64xf32, #hivm.address_space<ub>>)
        hivm.hir.copy ins(%83 : memref<64xf32, #hivm.address_space<ub>>) outs(%91 : memref<64xf32, #hivm.address_space<ub>>)
        scf.yield %75, %87, %91, %34, %30 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
      }
      hivm.hir.set_ctrl true at ctrl[60]
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    return
  }
  func.func @__cvp_attention_query_first_output(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64x64xf32, #hivm.address_space<ub>>, %arg2: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_first_output", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.for %arg3 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg0[%arg3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg1[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_2 = memref.subview %arg2[%arg3, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %0 = vector.transfer_read %subview_1[%c0, %c0], %cst_0 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %1 = vector.transfer_read %subview[%c0], %cst_0 {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %2 = arith.mulf %1, %cst : vector<1x64xf32>
      %3 = arith.addf %0, %2 : vector<1x64xf32>
      vector.transfer_write %3, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @__cvp_attention_last_output(%arg0: memref<64xf32, #hivm.address_space<ub>>, %arg1: memref<64xf32, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64x64xf32, #hivm.address_space<ub>>, %arg5: memref<64x64xf16, #hivm.address_space<ub>>, %arg6: memref<64xf32, #hivm.address_space<ub>>, %arg7: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_last_output", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    scf.for %arg8 = %c0 to %c64 step %c1 {
      %subview = memref.subview %arg3[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %subview_0 = memref.subview %arg4[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %subview_1 = memref.subview %arg5[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %4 = vector.transfer_read %subview_0[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %subview_2 = memref.subview %arg6[%arg8] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %5 = vector.transfer_read %subview_2[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %6 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %subview_3 = memref.subview %arg7[%arg8, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview_3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %8 = arith.mulf %4, %5 : vector<1x64xf32>
      %9 = arith.addf %7, %8 : vector<1x64xf32>
      %10 = arith.divf %9, %6 : vector<1x64xf32>
      %11 = arith.truncf %10 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %11, %subview_1[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @__cvp_attention_query_first_state(%arg0: memref<256x64xf32, #hivm.address_space<ub>>, %arg1: memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>>, %arg2: memref<64xf32, #hivm.address_space<ub>>, %arg3: memref<64xf32, #hivm.address_space<ub>>, %arg4: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.cvp_attention.boundary = "query_first_state", hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : vector<64xf32>
    %cst_1 = arith.constant dense<0xFF800000> : vector<64xf32>
    %cst_2 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %cst_1) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg5, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %8 = vector.shape_cast %7 : vector<1x64xf32> to vector<64xf32>
      %9 = arith.maximumf %arg6, %8 : vector<64xf32>
      scf.yield %9 : vector<64xf32>
    }
    %1 = arith.mulf %0, %cst_0 : vector<64xf32>
    vector.transfer_write %1, %arg4[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %2 = vector.shape_cast %1 : vector<64xf32> to vector<1x64xf32>
    %3 = scf.for %arg5 = %c0 to %c256 step %c1 iter_args(%arg6 = %cst) -> (vector<64xf32>) {
      %subview = memref.subview %arg0[%arg5, 0] [1, 64] [1, 1] : memref<256x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %7 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %8 = arith.mulf %7, %cst_2 : vector<1x64xf32>
      %9 = arith.subf %8, %2 : vector<1x64xf32>
      %10 = math.exp %9 : vector<1x64xf32>
      %11 = vector.shape_cast %10 : vector<1x64xf32> to vector<64xf32>
      %12 = arith.addf %arg6, %11 : vector<64xf32>
      %13 = arith.truncf %10 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %subview_4 = memref.subview %arg1[0, %arg5, 0] [4, 1, 16] [1, 1, 1] : memref<4x256x16xf16, strided<[4112, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      %14 = vector.shape_cast %13 : vector<1x64xf16> to vector<4x1x16xf16>
      vector.transfer_write %14, %subview_4[%c0, %c0, %c0] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[4112, 16, 1], offset: ?>, #hivm.address_space<ub>>
      scf.yield %12 : vector<64xf32>
    }
    %4 = arith.subf %cst_1, %1 : vector<64xf32>
    %5 = math.exp %4 : vector<64xf32>
    vector.transfer_write %5, %arg3[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %6 = arith.addf %5, %3 : vector<64xf32>
    vector.transfer_write %6, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
}



// -----

// CASE: original complete scheduled module after scalar counter lowering
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
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
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



// -----

// CASE: independent transfer starts exactly after Q bank one
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
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
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
            %stage8_alias = hivm.hir.pointer_cast(%c32768_i64) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%stage8_alias : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
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



// -----

// CASE: all bounded address alternatives are outside Q
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
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
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
            %stage8_address = arith.select %54, %c32768_i64, %c98304_i64 : i64
            %stage8_alias = hivm.hir.pointer_cast(%stage8_address) : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%stage8_alias : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
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



// -----

// CASE: descriptor offset places the view outside Q
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
    %0 = hivm.hir.pointer_cast(%c0_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %1 = hivm.hir.pointer_cast(%c131072_i64) : memref<8x16x16x16xf32, #hivm.address_space<cc>>
    %2 = hivm.hir.pointer_cast(%c131072_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
    %3 = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf32, #hivm.address_space<cc>>
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
            %stage8_alias = hivm.hir.pointer_cast(%c0_i64) : memref<4x8x16x16xf16, strided<[2048, 256, 16, 1], offset: 16384>, #hivm.address_space<cbuf>>
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%stage8_alias : memref<4x8x16x16xf16, strided<[2048, 256, 16, 1], offset: 16384>, #hivm.address_space<cbuf>>)
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



// RUN: not bishengir-opt %s --hivm-cvp-query-state-storage 2>&1 | FileCheck %s
// An out-of-stage state read must not be hidden by explicit bank lowering.
// CHECK: CVP state storage: loop state escapes its owning stage
// -----// IR Dump After CVPQueryOutputStorage (hivm-cvp-query-output-storage) ('builtin.module' operation) //----- //
#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<(d0) -> (d0, 0)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0] -> (s0 * 4)>
#map4 = affine_map<()[s0, s1] -> (s0 + s1)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.dense_p_layouts = 1 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 512 : i64, key_tiles = 2 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c28_i32 = arith.constant 28 : i32
    %true = arith.constant true
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
    %c8_i32 = arith.constant 8 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c128_i32 = arith.constant 128 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c512_i64 = arith.constant 512 : i64
    %c256_i64 = arith.constant 256 : i64
    %c28_i64 = arith.constant 28 : i64
    %c64_i64 = arith.constant 64 : i64
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
      %6 = arith.subi %c64_i64, %5 : i64
      %7 = arith.maxsi %6, %c0_i64 : i64
      %8 = arith.ceildivsi %7, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %9 = arith.muli %8, %c512_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %alloc = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_0 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      hivm.hir.anchor {id = 7 : i64}
      scf.for %arg11 = %c0_i64 to %9 step %c256_i64  : i64 {
        hivm.hir.anchor {id = 8 : i64}
        %10 = arith.divsi %arg11, %c512_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %11 = arith.remsi %arg11, %c512_i64 : i64
        %12 = arith.trunci %11 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %13 = arith.muli %10, %c28_i64 : i64
        %14 = arith.addi %5, %13 : i64
        %15 = arith.trunci %14 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %16 = arith.remsi %10, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %17 = arith.cmpi eq, %16, %c1_i64 : i64
        %18 = arith.cmpi eq, %11, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %19 = arith.cmpi eq, %11, %c256_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
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
          %21 = arith.divsi %15, %c4_i32 : i32
          %22 = arith.remsi %15, %c4_i32 : i32
          %23 = arith.divsi %21, %c8_i32 : i32
          %24 = arith.remsi %21, %c8_i32 : i32
          %25 = arith.extsi %23 : i32 to i64
          %26 = arith.muli %25, %c262144_i64 : i64
          %27 = arith.extsi %24 : i32 to i64
          %28 = arith.muli %27, %c32768_i64 : i64
          %29 = arith.addi %26, %28 : i64
          %30 = arith.index_cast %29 : i64 to index
          %31 = arith.muli %22, %c128_i32 : i32
          %32 = arith.maxsi %31, %c0_i32 : i32
          %33 = arith.index_cast %32 : i32 to index
          %34 = affine.apply #map()[%30, %33]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%34], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %35 = arith.select %17, %alloc_0, %alloc : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 11 : i64}
          scf.if %18 {
            hivm.hir.anchor {id = 12 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%35 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 13 : i64}
          }
          %36 = bufferization.to_tensor %35 restrict : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          %37 = arith.maxsi %12, %c0_i32 : i32
          %38 = arith.index_cast %37 : i32 to index
          %39 = affine.apply #map()[%30, %38]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%39], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %alloc_5 = memref.alloc() : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_5 : memref<4x16x16x16xf16>)
          hivm.hir.anchor {id = 15 : i64}
          %40 = bufferization.to_tensor %alloc_5 restrict writable : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 16 : i64}
          %41 = tensor.empty() : tensor<8x16x16x16xf32>
          %42 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%40, %36, %true, %c256, %c64, %c128 : tensor<4x16x16x16xf16>, tensor<4x8x16x16xf16>, i1, index, index, index) outs(%41 : tensor<8x16x16x16xf32>) -> tensor<8x16x16x16xf32>
          hivm.hir.anchor {id = 17 : i64}
          annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 18 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%42 : tensor<8x16x16x16xf32>) outs(%alloc_3 : memref<256x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <COLUMN_SPLIT>
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
        %20 = bufferization.to_tensor %alloc_2 restrict writable : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.anchor {id = 45 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 46 : i64}
          %21 = arith.divsi %15, %c4_i32 : i32
          %22 = arith.divsi %21, %c8_i32 : i32
          %23 = arith.remsi %21, %c8_i32 : i32
          %24 = arith.extsi %22 : i32 to i64
          %25 = arith.muli %24, %c262144_i64 : i64
          %26 = arith.extsi %23 : i32 to i64
          %27 = arith.muli %26, %c32768_i64 : i64
          %28 = arith.addi %25, %27 : i64
          %29 = arith.index_cast %28 : i64 to index
          %30 = arith.maxsi %12, %c0_i32 : i32
          %31 = arith.index_cast %30 : i32 to index
          %32 = affine.apply #map()[%29, %31]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%32], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %alloc_4 = memref.alloc() : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 47 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_4 : memref<4x16x16x16xf16>)
          hivm.hir.anchor {id = 48 : i64}
          %33 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 49 : i64}
          %34 = tensor.empty() : tensor<4x8x16x16xf32>
          %35 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%20, %33, %true, %c128, %c256, %c64 : tensor<8x16x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%34 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
          hivm.hir.anchor {id = 50 : i64}
          annotation.mark %alloc_1 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 51 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%35 : tensor<4x8x16x16xf32>) outs(%alloc_1 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
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
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.stream = {key_count = 512 : i64, key_step = 256 : i64, key_unroll = 1 : i32, physical_lifetimes_verified = false, pipeline_mode = "skew", query_slots = 2 : i32, query_step = 28 : i64}}
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 74 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 75 : i64}
    return
  }
  func.func @_attn_fwd_mix_aiv_fused_4_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>, %arg2: tensor<64xf32>) -> tensor<64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %1 = vector.transfer_read %arg1[%c0], %cst {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %2 = math.log %1 : vector<64xf32>
    %3 = arith.addf %0, %2 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg2[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %4 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_fused_5_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%arg0: tensor<64xf32>, %arg1: tensor<64x64xf32>, %arg2: tensor<64x64xf32>, %arg3: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @_attn_fwd_mix_aiv_outlined_vf_3(%arg0: tensor<256x64xf32>, %arg1: tensor<8x128x16xf16>, %arg2: tensor<64xf32>, %arg3: tensor<64xf32>, %arg4: tensor<64xf32>, %arg5: tensor<64xf32>, %arg6: tensor<64xf32>) -> (tensor<8x128x16xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) attributes {hivm.cvp_attention.dense_p = {keys = 256 : i64, padding_required = true, query_rows = 64 : i64}, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %cst_0 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c0 = arith.constant 0 : index
    %0 = scf.for %arg7 = %c0 to %c256 step %c1 iter_args(%arg8 = %cst) -> (vector<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [1, 64] [1, 1] : tensor<256x64xf32> to tensor<1x64xf32>
      %15 = vector.transfer_read %extracted_slice[%c0, %c0], %cst_1 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %16 = vector.shape_cast %15 : vector<1x64xf32> to vector<64xf32>
      %17 = arith.maximumf %arg8, %16 : vector<64xf32>
      scf.yield %17 : vector<64xf32>
    }
    %cst_2 = arith.constant dense<5.000000e-01> : vector<64xf32>
    %1 = arith.mulf %0, %cst_2 : vector<64xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c0_5 = arith.constant 0 : index
    %2 = vector.transfer_read %arg3[%c0_5], %cst_4 {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %3 = arith.maximumf %2, %1 : vector<64xf32>
    %4 = vector.transfer_write %3, %arg6[%c0_5] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %5 = vector.shape_cast %3 : vector<64xf32> to vector<1x64xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %c1_7 = arith.constant 1 : index
    %c256_8 = arith.constant 256 : index
    %c0_9 = arith.constant 0 : index
    %cst_10 = arith.constant dense<0.000000e+00> : vector<64xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %c1_12 = arith.constant 1 : index
    %c256_13 = arith.constant 256 : index
    %c0_14 = arith.constant 0 : index
    %cst_15 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %c128 = arith.constant 128 : index
    %6:2 = scf.for %arg7 = %c0_9 to %c128 step %c1_7 iter_args(%arg8 = %arg1, %arg9 = %cst_10) -> (tensor<8x128x16xf16>, vector<64xf32>) {
      %extracted_slice = tensor.extract_slice %arg0[%arg7, 0] [1, 64] [1, 1] : tensor<256x64xf32> to tensor<1x64xf32>
      %15 = vector.transfer_read %extracted_slice[%c0_9, %c0_9], %cst_6 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %16 = arith.mulf %15, %cst_15 : vector<1x64xf32>
      %17 = arith.subf %16, %5 : vector<1x64xf32>
      %18 = math.exp %17 : vector<1x64xf32>
      %19 = vector.shape_cast %18 : vector<1x64xf32> to vector<64xf32>
      %20 = arith.truncf %18 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %21 = vector.shape_cast %20 : vector<1x64xf16> to vector<64xf16>
      %22 = arith.addi %arg7, %c128 : index
      %extracted_slice_18 = tensor.extract_slice %arg0[%22, 0] [1, 64] [1, 1] : tensor<256x64xf32> to tensor<1x64xf32>
      %23 = vector.transfer_read %extracted_slice_18[%c0_9, %c0_9], %cst_6 {in_bounds = [true, true]} : tensor<1x64xf32>, vector<1x64xf32>
      %24 = arith.mulf %23, %cst_15 : vector<1x64xf32>
      %25 = arith.subf %24, %5 : vector<1x64xf32>
      %26 = math.exp %25 : vector<1x64xf32>
      %27 = vector.shape_cast %26 : vector<1x64xf32> to vector<64xf32>
      %28 = arith.truncf %26 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      %29 = vector.shape_cast %28 : vector<1x64xf16> to vector<64xf16>
      %30 = arith.addf %19, %27 : vector<64xf32>
      %31 = arith.addf %arg9, %30 : vector<64xf32>
      %res1, %res2 = ave.hir.vdintlv %21, %29 {layout_change = #ave<layout_change DENSE>} : vector<64xf16>, vector<128xf16>
      %32 = vector.shape_cast %res1 : vector<128xf16> to vector<8x1x16xf16>
      %extracted_slice_19 = tensor.extract_slice %arg8[0, %arg7, 0] [8, 1, 16] [1, 1, 1] : tensor<8x128x16xf16> to tensor<8x1x16xf16>
      %c0_20 = arith.constant 0 : index
      %33 = vector.transfer_write %32, %extracted_slice_19[%c0_20, %c0_20, %c0_20] {in_bounds = [true, true, true]} : vector<8x1x16xf16>, tensor<8x1x16xf16>
      %inserted_slice = tensor.insert_slice %33 into %arg8[0, %arg7, 0] [8, 1, 16] [1, 1, 1] : tensor<8x1x16xf16> into tensor<8x128x16xf16>
      scf.yield %inserted_slice, %31 : tensor<8x128x16xf16>, vector<64xf32>
    }
    %7 = vector.shape_cast %5 : vector<1x64xf32> to vector<64xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %c0_17 = arith.constant 0 : index
    %8 = arith.subf %2, %7 : vector<64xf32>
    %9 = math.exp %8 : vector<64xf32>
    %10 = vector.transfer_write %9, %arg5[%c0_17] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    %11 = vector.transfer_read %arg2[%c0_17], %cst_16 {in_bounds = [true]} : tensor<64xf32>, vector<64xf32>
    %12 = arith.mulf %11, %9 : vector<64xf32>
    %13 = arith.addf %12, %6#1 : vector<64xf32>
    %14 = vector.transfer_write %13, %arg4[%c0_17] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %6#0, %14, %10, %4 : tensor<8x128x16xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_4(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv_outlined_vf_5(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
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
  func.func @_attn_fwd_mix_aiv_outlined_vf_6(%arg0: tensor<64xf32>) -> tensor<64xf32> attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0xFF800000> : vector<64xf32>
    %c0 = arith.constant 0 : index
    %0 = vector.transfer_write %cst, %arg0[%c0] {in_bounds = [true]} : vector<64xf32>, tensor<64xf32>
    return %0 : tensor<64xf32>
  }
  func.func @_attn_fwd_mix_aiv(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 512 : i64, key_tiles = 2 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %c64_i64 = arith.constant 64 : i64
    %c28_i64 = arith.constant 28 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c512_i32 = arith.constant 512 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32768_i64 = arith.constant 32768 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c8_i32 = arith.constant 8 : i32
    %c4_i32 = arith.constant 4 : i32
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
    %6 = call @_attn_fwd_mix_aiv_outlined_vf_6(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
    %7 = tensor.empty() : tensor<64x64xf16>
    scf.for %arg10 = %c0 to %c2 step %c1 {
      %8 = affine.apply #map2()[%arg10]
      %9 = affine.apply #map3()[%arg10]
      hivm.hir.anchor {id = 0 : i64}
      annotation.mark %1 {logical_block_num} : i32
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
      hivm.hir.anchor {id = 1 : i64}
      scf.for %arg11 = %3 to %1 step %c28_i32  : i32 {
        hivm.hir.anchor {id = 2 : i64}
        hivm.hir.set_ctrl false at ctrl[60]
        hivm.hir.set_ctrl true at ctrl[48]
        %10 = arith.remsi %arg11, %arg7 : i32
        hivm.hir.anchor {id = 3 : i64}
        hivm.hir.anchor {id = 4 : i64}
        hivm.hir.anchor {id = 5 : i64}
        hivm.hir.anchor {id = 6 : i64}
        %11 = arith.extsi %10 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
        %12 = arith.subi %c64_i64, %11 : i64
        %13 = arith.maxsi %12, %c0_i64 : i64
        %14 = arith.ceildivsi %13, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
        %15 = arith.muli %14, %c512_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
        hivm.hir.anchor {id = 7 : i64}
        %16 = func.call @_attn_fwd_mix_aiv_outlined_vf_4(%5) {hivm.vector_function, no_inline} : (tensor<64xf32>) -> tensor<64xf32>
        %17 = func.call @_attn_fwd_mix_aiv_outlined_vf_5(%4) {hivm.vector_function, no_inline} : (tensor<64x64xf32>) -> tensor<64x64xf32>
        %alloc = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
        %alloc_0 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
        %alloc_1 = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
        %alloc_2 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
        %18:5 = scf.for %arg12 = %c0_i64 to %15 step %c256_i64 iter_args(%arg13 = %17, %arg14 = %16, %arg15 = %6, %arg16 = %5, %arg17 = %5) -> (tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)  : i64 {
          hivm.hir.anchor {id = 8 : i64}
          %19 = arith.divsi %arg12, %c512_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
          %20 = arith.remsi %arg12, %c512_i64 : i64
          %21 = arith.muli %19, %c28_i64 : i64
          %22 = arith.addi %11, %21 : i64
          %23 = arith.trunci %22 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
          %24 = arith.remsi %19, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
          %25 = arith.cmpi eq, %24, %c1_i64 : i64
          %26 = arith.cmpi eq, %20, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
          %27 = arith.cmpi eq, %20, %c256_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
          %escaped = arith.select %25, %arg16, %arg14 : tensor<64xf32>
          %alloc_3 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_3 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
          %alloc_4 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %subview = memref.subview %alloc_4[%9, 0, 0, 0] [4, 16, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>> to memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          %alloc_5 = memref.alloc() : memref<256x64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 9 : i64}
          scope.scope : () -> () {
            hivm.hir.anchor {id = 10 : i64}
            hivm.hir.anchor {id = 11 : i64}
            scf.if %26 {
              hivm.hir.anchor {id = 12 : i64}
              hivm.hir.anchor {id = 13 : i64}
            }
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 1 : index, tiledAlloc} : memref<256x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.anchor {id = 19 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 20 : i64}
          %28 = bufferization.to_tensor %alloc_5 restrict writable : memref<256x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 21 : i64}
          %29:5 = scope.scope : () -> (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) {
            hivm.hir.anchor {id = 22 : i64}
            %32 = arith.select %25, %arg16, %arg14 : tensor<64xf32>
            %33 = arith.select %25, %arg17, %arg15 : tensor<64xf32>
            hivm.hir.anchor {id = 23 : i64}
            %34 = arith.select %26, %16, %32 : tensor<64xf32>
            %35 = arith.select %26, %6, %33 : tensor<64xf32>
            scf.if %26 {
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
            %36 = tensor.empty() : tensor<8x128x16xf16>
            %37:4 = func.call @_attn_fwd_mix_aiv_outlined_vf_3(%28, %36, %34, %35, %5, %5, %5) {hivm.vector_function, no_inline} : (tensor<256x64xf32>, tensor<8x128x16xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> (tensor<8x128x16xf16>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
            annotation.mark %37#0 {hivm.stride_align_dims = array<i32: 1>, hivm.stride_align_value_in_byte = array<i32: 258>} : tensor<8x128x16xf16>
            annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 42 : i64}
            %extracted_slice = tensor.extract_slice %37#0[0, 0, 0] [4, 128, 16] [1, 1, 1] : tensor<8x128x16xf16> to tensor<4x128x16xf16>
            %expanded = tensor.expand_shape %extracted_slice [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : tensor<4x128x16xf16> into tensor<4x8x16x16xf16>
            %subview_6 = memref.subview %subview[0, 0, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%expanded : tensor<4x8x16x16xf16>) outs(%subview_6 : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
            %extracted_slice_7 = tensor.extract_slice %37#0[4, 0, 0] [4, 128, 16] [1, 1, 1] : tensor<8x128x16xf16> to tensor<4x128x16xf16>
            %expanded_8 = tensor.expand_shape %extracted_slice_7 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : tensor<4x128x16xf16> into tensor<4x8x16x16xf16>
            %subview_9 = memref.subview %subview[0, 8, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%expanded_8 : tensor<4x8x16x16xf16>) outs(%subview_9 : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
            hivm.hir.anchor {id = 43 : i64}
            %38 = arith.select %25, %arg14, %37#1 : tensor<64xf32>
            %39 = arith.select %25, %arg15, %37#3 : tensor<64xf32>
            %40 = arith.select %25, %37#1, %arg16 : tensor<64xf32>
            %41 = arith.select %25, %37#3, %arg17 : tensor<64xf32>
            scope.return %37#2, %39, %38, %40, %41 : tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
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
            annotation.mark %alloc_3 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 53 : i64}
          %30 = bufferization.to_tensor %alloc_3 restrict writable : memref<64x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 54 : i64}
          %31 = scope.scope : () -> tensor<64x64xf32> {
            hivm.hir.anchor {id = 55 : i64}
            %32 = arith.divsi %23, %c4_i32 : i32
            %33 = arith.remsi %23, %c4_i32 : i32
            %34 = arith.divsi %32, %c8_i32 : i32
            %35 = arith.remsi %32, %c8_i32 : i32
            %36 = arith.extsi %34 : i32 to i64
            %37 = arith.muli %36, %c262144_i64 : i64
            %38 = arith.extsi %35 : i32 to i64
            %39 = arith.muli %38, %c32768_i64 : i64
            %40 = arith.addi %37, %39 : i64
            %41 = arith.index_cast %40 : i64 to index
            %42 = arith.muli %33, %c128_i32 : i32
            %43 = arith.maxsi %42, %c0_i32 : i32
            %44 = arith.index_cast %43 : i32 to index
            %45 = affine.apply #map()[%41, %44]
            %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%45], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
            %subview_6 = memref.subview %reinterpret_cast[%8, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>> to memref<64x64xf16, strided<[64, 1], offset: ?>>
            hivm.hir.anchor {id = 56 : i64}
            %46 = arith.select %26, %17, %arg13 : tensor<64x64xf32>
            scf.if %26 {
              hivm.hir.anchor {id = 57 : i64}
              hivm.hir.anchor {id = 58 : i64}
            } else {
              hivm.hir.anchor {id = 59 : i64}
            }
            hivm.hir.anchor {id = 60 : i64}
            hivm.hir.anchor {id = 61 : i64}
            %47 = func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%29#0, %30, %46, %4) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
            hivm.hir.anchor {id = 62 : i64}
            scf.if %27 {
              hivm.hir.anchor {id = 63 : i64}
              %48 = arith.select %25, %29#3, %29#2 : tensor<64xf32>
              %49 = arith.select %25, %29#4, %29#1 : tensor<64xf32>
              hivm.hir.anchor {id = 64 : i64}
              hivm.hir.anchor {id = 65 : i64}
              %50 = func.call @_attn_fwd_mix_aiv_fused_4_outlined_vf_0(%49, %48, %5) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
              hivm.hir.anchor {id = 66 : i64}
              hivm.hir.anchor {id = 67 : i64}
              %51 = arith.muli %32, %c512_i32 : i32
              %52 = arith.index_cast %51 : i32 to index
              %53 = arith.index_cast %42 : i32 to index
              %54 = affine.apply #map4()[%52, %53]
              %reinterpret_cast_7 = memref.reinterpret_cast %arg5 to offset: [%54], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
              %subview_8 = memref.subview %reinterpret_cast_7[%8] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>> to memref<64xf32, strided<[1], offset: ?>>
              %55 = arith.select %25, %alloc_2, %alloc_1 : memref<64xf32, #hivm.address_space<ub>>
              bufferization.materialize_in_destination %50 in writable %55 : (tensor<64xf32>, memref<64xf32, #hivm.address_space<ub>>) -> ()
              hivm.hir.store ins(%55 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_8 : memref<64xf32, strided<[1], offset: ?>>) {tiled_op}
              hivm.hir.anchor {id = 68 : i64}
              %56 = func.call @_attn_fwd_mix_aiv_fused_5_outlined_vf_0(%48, %47, %7) {hivm.vector_function, no_inline} : (tensor<64xf32>, tensor<64x64xf32>, tensor<64x64xf16>) -> tensor<64x64xf16>
              hivm.hir.anchor {id = 69 : i64}
              %57 = arith.select %25, %alloc_0, %alloc : memref<64x64xf16, #hivm.address_space<ub>>
              bufferization.materialize_in_destination %56 in writable %57 : (tensor<64x64xf16>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
              hivm.hir.store ins(%57 : memref<64x64xf16, #hivm.address_space<ub>>) outs(%subview_6 : memref<64x64xf16, strided<[64, 1], offset: ?>>) {tiled_op}
              hivm.hir.anchor {id = 70 : i64}
            } else {
              hivm.hir.anchor {id = 71 : i64}
            }
            hivm.hir.anchor {id = 72 : i64}
            scope.return %47 : tensor<64x64xf32>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 73 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
          scf.yield %31, %29#2, %29#1, %29#3, %29#4 : tensor<64x64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>
        } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.output_storage, hivm.cvp_query.stream = {key_count = 512 : i64, key_step = 256 : i64, key_unroll = 1 : i32, physical_lifetimes_verified = false, pipeline_mode = "skew", query_slots = 2 : i32, query_step = 28 : i64}}
        hivm.hir.set_ctrl true at ctrl[60]
        hivm.hir.anchor {id = 74 : i64}
      } {autoblockify.subloop}
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
      hivm.hir.anchor {id = 75 : i64}
    } {map_for_to_forall, mapping = [#hivm.sub_block<x>]}
    return
  }
}



#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<()[s0] -> (s0 + 64)>
#map4 = affine_map<()[s0] -> (s0 + 128)>
#map5 = affine_map<()[s0] -> (s0 + 192)>
#map6 = affine_map<()[s0] -> (s0 * 64)>
#map7 = affine_map<()[s0] -> (s0 * 4)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.dense_p_layouts = 1 : i64, hivm.cvp_attention.dense_two_chain_loops = 2 : i64, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.cvp_query.final_output_copies_forwarded = 1 : i64, hivm.cvp_query.output_copies_forwarded = 1 : i64, hivm.cvp_query.snapshot_copies_forwarded = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @_attn_fwd_backup(%arg0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-cvp-attention-vector">, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.anchor {id = 0 : i64}
    %c16128_i64 = arith.constant 16128 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c28_i64 = arith.constant 28 : i64
    %c256_i64 = arith.constant 256 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %c16384_i32 = arith.constant 16384 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c8388608_i64 = arith.constant 8388608 : i64
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64 = arith.constant 64 : index
    %cst_0 = arith.constant 5.000000e-01 : f32
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
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %arg10 = %3 to %1 step %c28_i32  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %4 = arith.remsi %arg10, %arg7 : i32
      %5 = tensor.empty() : tensor<128x64xf32>
      hivm.hir.anchor {id = 3 : i64}
      %6 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%5 : tensor<128x64xf32>) -> tensor<128x64xf32>
      hivm.hir.anchor {id = 4 : i64}
      %7 = tensor.empty() : tensor<128xf32>
      %8 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_1 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      hivm.hir.anchor {id = 5 : i64}
      %9 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_2 : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
      hivm.hir.anchor {id = 6 : i64}
      %10 = arith.extsi %4 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %11 = arith.subi %c131072_i64, %10 : i64
      %12 = arith.maxsi %11, %c0_i64 : i64
      %13 = arith.ceildivsi %12, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %14 = arith.muli %13, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      %15 = tensor.empty() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "sum"}} : tensor<128xf32>
      %16 = tensor.empty() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "maximum"}} : tensor<128xf32>
      %alloc = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %alloc_3 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "Q"}} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
      %17 = tensor.empty() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : tensor<128x64xf16>
      %18 = tensor.empty() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : tensor<128x64xf16>
      %19 = tensor.empty() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : tensor<128xf32>
      %20 = tensor.empty() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : tensor<128xf32>
      hivm.hir.anchor {id = 7 : i64}
      %21:9 = scf.for %arg11 = %c0_i64 to %14 step %c256_i64 iter_args(%arg12 = %6, %arg13 = %9, %arg14 = %8, %arg15 = %15, %arg16 = %16, %arg17 = %17, %arg18 = %18, %arg19 = %19, %arg20 = %20) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64xf16>, tensor<128x64xf16>, tensor<128xf32>, tensor<128xf32>)  : i64 {
        hivm.hir.anchor {id = 8 : i64}
        %22 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %23 = arith.remsi %arg11, %c16384_i64 : i64
        %24 = arith.trunci %23 {hivm.cvp_query.coordinate = "local_key"} : i64 to i32
        %25 = arith.muli %22, %c28_i64 : i64
        %26 = arith.addi %10, %25 : i64
        %27 = arith.trunci %26 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %28 = arith.remsi %22, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %29 = arith.cmpi eq, %28, %c1_i64 : i64
        %30 = arith.cmpi eq, %23, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %31 = arith.cmpi eq, %23, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
        %alloc_4 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
        %alloc_5 = memref.alloc() : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        %alloc_6 = memref.alloc() : memref<256x128xf32, #hivm.address_space<ub>>
        annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<256x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
        hivm.hir.anchor {id = 9 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 10 : i64}
          %37 = arith.divsi %27, %c128_i32 : i32
          %38 = arith.remsi %27, %c128_i32 : i32
          %39 = arith.divsi %37, %c8_i32 : i32
          %40 = arith.remsi %37, %c8_i32 : i32
          %41 = arith.extsi %39 : i32 to i64
          %42 = arith.muli %41, %c8388608_i64 : i64
          %43 = arith.extsi %40 : i32 to i64
          %44 = arith.muli %43, %c1048576_i64 : i64
          %45 = arith.addi %42, %44 : i64
          %46 = arith.index_cast %45 : i64 to index
          %47 = arith.muli %38, %c128_i32 : i32
          %48 = arith.maxsi %47, %c0_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = affine.apply #map()[%46, %49]
          %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%50], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          %51 = arith.select %29, %alloc_3, %alloc : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 11 : i64}
          scf.if %30 {
            hivm.hir.anchor {id = 12 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%51 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 13 : i64}
          }
          %52 = bufferization.to_tensor %51 restrict : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
          %53 = arith.maxsi %24, %c0_i32 : i32
          %54 = arith.index_cast %53 : i32 to index
          %55 = affine.apply #map()[%46, %54]
          %reinterpret_cast_7 = memref.reinterpret_cast %arg3 to offset: [%55], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %alloc_8 = memref.alloc() : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_7 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_8 : memref<4x16x16x16xf16>)
          hivm.hir.anchor {id = 15 : i64}
          %56 = bufferization.to_tensor %alloc_8 restrict writable : memref<4x16x16x16xf16>
          %57 = tensor.empty() : tensor<8x16x16x16xf32>
          hivm.hir.anchor {id = 16 : i64}
          %58 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%56, %52, %true, %c256, %c64, %c128 : tensor<4x16x16x16xf16>, tensor<4x8x16x16xf16>, i1, index, index, index) outs(%57 : tensor<8x16x16x16xf32>) -> tensor<8x16x16x16xf32>
          hivm.hir.anchor {id = 17 : i64}
          annotation.mark %alloc_6 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<256x128xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 18 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%58 : tensor<8x16x16x16xf32>) outs(%alloc_6 : memref<256x128xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 19 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
        hivm.hir.anchor {id = 20 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        %32 = bufferization.to_tensor %alloc_6 restrict writable : memref<256x128xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.anchor {id = 21 : i64}
        %33:5 = scope.scope : () -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
          hivm.hir.anchor {id = 22 : i64}
          %37 = arith.select %29, %arg15, %arg13 : tensor<128xf32>
          %38 = arith.select %29, %arg16, %arg14 : tensor<128xf32>
          hivm.hir.anchor {id = 23 : i64}
          %39:2 = scf.if %30 -> (tensor<128xf32>, tensor<128xf32>) {
            hivm.hir.anchor {id = 24 : i64}
            %60 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_2 : f32) outs(%37 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 25 : i64}
            %61 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst_1 : f32) outs(%38 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 26 : i64}
            scf.yield %60, %61 : tensor<128xf32>, tensor<128xf32>
          } else {
            hivm.hir.anchor {id = 27 : i64}
            scf.yield %37, %38 : tensor<128xf32>, tensor<128xf32>
          }
          %40 = tensor.empty() : tensor<256x128xf32>
          hivm.hir.anchor {id = 28 : i64}
          %41 = hivm.hir.vmul ins(%32, %cst_0 : tensor<256x128xf32>, f32) outs(%40 : tensor<256x128xf32>) -> tensor<256x128xf32>
          hivm.hir.anchor {id = 29 : i64}
          %expanded = tensor.expand_shape %8 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
          %42 = hivm.hir.vreduce <max> ins(%41 : tensor<256x128xf32>) outs(%expanded : tensor<1x128xf32>) reduce_dims = [0] -> tensor<1x128xf32>
          hivm.hir.anchor {id = 30 : i64}
          %collapsed = tensor.collapse_shape %42 [[0, 1]] : tensor<1x128xf32> into tensor<128xf32>
          %43 = hivm.hir.vmax ins(%39#1, %collapsed : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 31 : i64}
          %expanded_7 = tensor.expand_shape %43 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
          %44 = hivm.hir.vsub ins(%41, %expanded_7 : tensor<256x128xf32>, tensor<1x128xf32>) outs(%40 : tensor<256x128xf32>) broadcast = [0] -> tensor<256x128xf32>
          hivm.hir.anchor {id = 32 : i64}
          %45 = hivm.hir.vexp ins(%44 : tensor<256x128xf32>) outs(%40 : tensor<256x128xf32>) -> tensor<256x128xf32>
          hivm.hir.anchor {id = 33 : i64}
          %46 = tensor.empty() : tensor<256x128xf16>
          %47 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%45 : tensor<256x128xf32>) outs(%46 : tensor<256x128xf16>) -> tensor<256x128xf16>
          hivm.hir.anchor {id = 34 : i64}
          %48 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 35 : i64}
          %expanded_8 = tensor.expand_shape %48 [[0, 1]] output_shape [1, 128] : tensor<128xf32> into tensor<1x128xf32>
          %49 = hivm.hir.vreduce <sum> ins(%45 : tensor<256x128xf32>) outs(%expanded_8 : tensor<1x128xf32>) reduce_dims = [0] -> tensor<1x128xf32>
          hivm.hir.anchor {id = 36 : i64}
          %collapsed_9 = tensor.collapse_shape %49 [[0, 1]] : tensor<1x128xf32> into tensor<128xf32>
          %50 = hivm.hir.vsub ins(%39#1, %43 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 37 : i64}
          %51 = hivm.hir.vexp ins(%50 : tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 38 : i64}
          %52 = hivm.hir.vmul ins(%39#0, %51 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 39 : i64}
          %53 = hivm.hir.vadd ins(%52, %collapsed_9 : tensor<128xf32>, tensor<128xf32>) outs(%7 : tensor<128xf32>) -> tensor<128xf32>
          hivm.hir.anchor {id = 40 : i64}
          %expanded_10 = tensor.expand_shape %47 [[0], [1, 2]] output_shape [256, 8, 16] : tensor<256x128xf16> into tensor<256x8x16xf16>
          %54 = tensor.empty() : tensor<8x256x16xf16>
          %55 = hivm.hir.vtranspose ins(%expanded_10 : tensor<256x8x16xf16>) outs(%54 : tensor<8x256x16xf16>) permutation = [1, 0, 2] -> tensor<8x256x16xf16>
          hivm.hir.anchor {id = 41 : i64}
          %expanded_11 = tensor.expand_shape %55 [[0], [1, 2], [3]] output_shape [8, 16, 16, 16] : tensor<8x256x16xf16> into tensor<8x16x16x16xf16>
          annotation.mark %alloc_5 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 42 : i64}
          hivm.hir.copy ins(%expanded_11 : tensor<8x16x16x16xf16>) outs(%alloc_5 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", hivm.tcore_type = #hivm.tcore_type<VECTOR>}
          hivm.hir.anchor {id = 43 : i64}
          %56 = arith.select %29, %arg13, %53 : tensor<128xf32>
          %57 = arith.select %29, %arg14, %43 : tensor<128xf32>
          %58 = arith.select %29, %53, %arg15 : tensor<128xf32>
          %59 = arith.select %29, %43, %arg16 : tensor<128xf32>
          scope.return %51, %57, %56, %58, %59 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
        hivm.hir.anchor {id = 44 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
        %34 = bufferization.to_tensor %alloc_5 restrict writable : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
        hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
        hivm.hir.anchor {id = 45 : i64}
        scope.scope : () -> () {
          hivm.hir.anchor {id = 46 : i64}
          %37 = arith.divsi %27, %c128_i32 : i32
          %38 = arith.divsi %37, %c8_i32 : i32
          %39 = arith.remsi %37, %c8_i32 : i32
          %40 = arith.extsi %38 : i32 to i64
          %41 = arith.muli %40, %c8388608_i64 : i64
          %42 = arith.extsi %39 : i32 to i64
          %43 = arith.muli %42, %c1048576_i64 : i64
          %44 = arith.addi %41, %43 : i64
          %45 = arith.index_cast %44 : i64 to index
          %46 = arith.maxsi %24, %c0_i32 : i32
          %47 = arith.index_cast %46 : i32 to index
          %48 = affine.apply #map()[%45, %47]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%48], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
          %alloc_7 = memref.alloc() : memref<4x16x16x16xf16>
          hivm.hir.anchor {id = 47 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%alloc_7 : memref<4x16x16x16xf16>)
          hivm.hir.anchor {id = 48 : i64}
          %49 = bufferization.to_tensor %alloc_7 restrict writable : memref<4x16x16x16xf16>
          %50 = tensor.empty() : tensor<4x8x16x16xf32>
          hivm.hir.anchor {id = 49 : i64}
          %51 = hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%34, %49, %true, %c128, %c256, %c64 : tensor<8x16x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%50 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
          hivm.hir.anchor {id = 50 : i64}
          annotation.mark %alloc_4 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x64xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 51 : i64}
          hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%51 : tensor<4x8x16x16xf32>) outs(%alloc_4 : memref<128x64xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 52 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
        hivm.hir.anchor {id = 53 : i64}
        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
        %35 = bufferization.to_tensor %alloc_4 restrict writable : memref<128x64xf32, #hivm.address_space<ub>>
        hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
        hivm.hir.anchor {id = 54 : i64}
        %36:5 = scope.scope : () -> (tensor<128x64xf32>, tensor<128x64xf16>, tensor<128x64xf16>, tensor<128xf32>, tensor<128xf32>) {
          hivm.hir.anchor {id = 55 : i64}
          %37 = arith.divsi %27, %c128_i32 : i32
          %38 = arith.remsi %27, %c128_i32 : i32
          %39 = arith.divsi %37, %c8_i32 : i32
          %40 = arith.remsi %37, %c8_i32 : i32
          %41 = arith.extsi %39 : i32 to i64
          %42 = arith.muli %41, %c8388608_i64 : i64
          %43 = arith.extsi %40 : i32 to i64
          %44 = arith.muli %43, %c1048576_i64 : i64
          %45 = arith.addi %42, %44 : i64
          %46 = arith.index_cast %45 : i64 to index
          %47 = arith.muli %38, %c128_i32 : i32
          %48 = arith.maxsi %47, %c0_i32 : i32
          %49 = arith.index_cast %48 : i32 to index
          %50 = affine.apply #map()[%46, %49]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%50], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
          hivm.hir.anchor {id = 56 : i64}
          %51 = scf.if %30 -> (tensor<128x64xf32>) {
            hivm.hir.anchor {id = 57 : i64}
            %61 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%cst : f32) outs(%arg12 : tensor<128x64xf32>) -> tensor<128x64xf32>
            hivm.hir.anchor {id = 58 : i64}
            scf.yield %61 : tensor<128x64xf32>
          } else {
            hivm.hir.anchor {id = 59 : i64}
            scf.yield %arg12 : tensor<128x64xf32>
          }
          %expanded = tensor.expand_shape %33#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
          hivm.hir.anchor {id = 60 : i64}
          %52 = hivm.hir.vmul ins(%51, %expanded : tensor<128x64xf32>, tensor<128x1xf32>) outs(%51 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
          hivm.hir.anchor {id = 61 : i64}
          %53 = hivm.hir.vadd ins(%35, %52 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%51 : tensor<128x64xf32>) -> tensor<128x64xf32>
          hivm.hir.anchor {id = 62 : i64}
          %54 = arith.select %29, %arg18, %arg17 : tensor<128x64xf16>
          %55 = arith.select %29, %arg20, %arg19 : tensor<128xf32>
          %56:2 = scf.if %31 -> (tensor<128x64xf16>, tensor<128xf32>) {
            hivm.hir.anchor {id = 63 : i64}
            %61 = arith.select %29, %33#3, %33#2 : tensor<128xf32>
            %62 = arith.select %29, %33#4, %33#1 : tensor<128xf32>
            hivm.hir.anchor {id = 64 : i64}
            %63 = hivm.hir.vln ins(%61 : tensor<128xf32>) outs(%55 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 65 : i64}
            %64 = hivm.hir.vadd ins(%62, %63 : tensor<128xf32>, tensor<128xf32>) outs(%55 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 66 : i64}
            %expanded_7 = tensor.expand_shape %61 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            %65 = hivm.hir.vdiv ins(%53, %expanded_7 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%53 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
            hivm.hir.anchor {id = 67 : i64}
            %66 = arith.muli %37, %c16384_i32 : i32
            %67 = arith.index_cast %66 : i32 to index
            %68 = arith.index_cast %47 : i32 to index
            %69 = affine.apply #map1()[%67, %68]
            %reinterpret_cast_8 = memref.reinterpret_cast %arg5 to offset: [%69], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
            hivm.hir.store ins(%64 : tensor<128xf32>) outs(%reinterpret_cast_8 : memref<128xf32, strided<[1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
            hivm.hir.anchor {id = 68 : i64}
            %70 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%65 : tensor<128x64xf32>) outs(%54 : tensor<128x64xf16>) -> tensor<128x64xf16>
            hivm.hir.anchor {id = 69 : i64}
            hivm.hir.store ins(%70 : tensor<128x64xf16>) outs(%reinterpret_cast : memref<128x64xf16, strided<[64, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
            hivm.hir.anchor {id = 70 : i64}
            scf.yield %70, %64 : tensor<128x64xf16>, tensor<128xf32>
          } else {
            hivm.hir.anchor {id = 71 : i64}
            scf.yield %54, %55 : tensor<128x64xf16>, tensor<128xf32>
          }
          %57 = arith.select %29, %arg17, %56#0 : tensor<128x64xf16>
          %58 = arith.select %29, %56#0, %arg18 : tensor<128x64xf16>
          %59 = arith.select %29, %arg19, %56#1 : tensor<128xf32>
          %60 = arith.select %29, %56#1, %arg20 : tensor<128xf32>
          hivm.hir.anchor {id = 72 : i64}
          scope.return %53, %57, %58, %59, %60 : tensor<128x64xf32>, tensor<128x64xf16>, tensor<128x64xf16>, tensor<128xf32>, tensor<128xf32>
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 73 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
        scf.yield %36#0, %33#2, %33#1, %33#3, %33#4, %36#1, %36#2, %36#3, %36#4 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64xf16>, tensor<128x64xf16>, tensor<128xf32>, tensor<128xf32>
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.stream = {key_count = 16384 : i64, key_step = 256 : i64, key_unroll = 1 : i32, physical_lifetimes_verified = false, pipeline_mode = "skew", query_slots = 2 : i32, query_step = 28 : i64}}
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 74 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 75 : i64}
    return
  }
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
      %5 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
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
  func.func @_attn_fwd_mix_aic(%arg0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %arg1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %arg2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.cvp_attention.layout_rewrites = 1 : i64, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64, effective_features = [], head_dim = 64 : i64, initial_max = 0xFF800000 : f32, initial_output = 0.000000e+00 : f32, initial_sum = 1.000000e+00 : f32, key_count = 16384 : i64, key_tiles = 64 : i64, max_state = 2 : i64, numerical_policy = "reassociate-fp32-sum", output_state = 0 : i64, physical_lifetimes_verified = false, physical_score_shape = array<i64: 256, 128>, scale = 5.000000e-01 : f32, score_layout = "key-major", sum_state = 1 : i64, version = 1 : i64}], hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}, hivm.cvp_query.stream, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
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
          %33 = affine.apply #map()[%29, %32]
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
          %37 = affine.apply #map()[%29, %36]
          %reinterpret_cast_4 = memref.reinterpret_cast %arg3 to offset: [%37], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_5 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 14 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast_4 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_5 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.anchor {id = 15 : i64}
          hivm.hir.anchor {id = 16 : i64}
          %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
          annotation.mark %alloc_6 {hivm.multi_buffer = 2 : i32} : memref<8x16x16x16xf32, #hivm.address_space<cc>>
          %38 = arith.select %17, %c49152_i64, %c16384_i64 : i64
          hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_5, %34, %true, %c256, %c64, %c128 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_6 : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%18, %10, %38)
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
          %31 = affine.apply #map()[%28, %30]
          %reinterpret_cast = memref.reinterpret_cast %arg4 to offset: [%31], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %alloc_4 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %alloc_4 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 47 : i64}
          hivm.hir.nd2nz {dst_continuous} ins(%reinterpret_cast : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%alloc_4 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.anchor {id = 49 : i64}
          %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
          annotation.mark %alloc_5 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
          hivm.hir.mmadL1 {a_transpose, already_set_real_mkn, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%alloc_2, %alloc_4, %true, %c128, %c256, %c64 : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%alloc_5 : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
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
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.stream = {key_count = 16384 : i64, key_step = 256 : i64, key_unroll = 1 : i32, physical_lifetimes_verified = false, pipeline_mode = "skew", query_slots = 2 : i32, query_step = 28 : i64}}
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
      %2 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %3 = arith.mulf %1, %2 : vector<1x64xf32>
      %4 = arith.addf %0, %3 : vector<1x64xf32>
      vector.transfer_write %4, %subview_2[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
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
      %9 = arith.extsi %8 {hivm.cvp_query.coordinate = "query_start"} : i32 to i64
      %10 = arith.subi %c131072_i64, %9 : i64
      %11 = arith.maxsi %10, %c0_i64 : i64
      %12 = arith.ceildivsi %11, %c28_i64 {hivm.cvp_query.coordinate = "query_count"} : i64
      %13 = arith.muli %12, %c16384_i64 {hivm.cvp_query.coordinate = "work_end"} : i64
      hivm.hir.anchor {id = 7 : i64}
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
      func.call @_attn_fwd_mix_aiv_outlined_merged_vf_0(%alloc_0, %alloc_1) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
      %alloc_2 = memref.alloc() {alignment = 64 : i64, hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
      %alloc_3 = memref.alloc() {alignment = 64 : i64, hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "output"}} : memref<64x64xf16, #hivm.address_space<ub>>
      %alloc_4 = memref.alloc() {hivm.cvp_query.storage = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_5 = memref.alloc() {hivm.cvp_query.storage = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "M"}} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_6 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "O", required_slots = 1 : i32, role = "accumulator"}} : memref<64x64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_6 {mem_unique} : memref<64x64xf32, #hivm.address_space<ub>>
      %alloc_7 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "sum"}} : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_7 {mem_unique} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_8 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "maximum"}} : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_8 {mem_unique} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_9 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "sum"}} : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_9 {mem_unique} : memref<64xf32, #hivm.address_space<ub>>
      %alloc_10 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "maximum"}} : memref<64xf32, #hivm.address_space<ub>>
      annotation.mark %alloc_10 {mem_unique} : memref<64xf32, #hivm.address_space<ub>>
      %collapse_shape = memref.collapse_shape %alloc_1 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      %collapse_shape_11 = memref.collapse_shape %alloc_6 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
      hivm.hir.copy ins(%collapse_shape : memref<4096xf32, #hivm.address_space<ub>>) outs(%collapse_shape_11 : memref<4096xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%alloc_0 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_7 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%alloc : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_8 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%alloc_0 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_9 : memref<64xf32, #hivm.address_space<ub>>)
      hivm.hir.copy ins(%alloc : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_10 : memref<64xf32, #hivm.address_space<ub>>)
      scf.for %arg11 = %c0_i64 to %13 step %c256_i64  : i64 {
        hivm.hir.anchor {id = 8 : i64}
        %14 = arith.divsi %arg11, %c16384_i64 {hivm.cvp_query.coordinate = "query_ordinal"} : i64
        %15 = arith.remsi %arg11, %c16384_i64 : i64
        %16 = arith.muli %14, %c28_i64 : i64
        %17 = arith.addi %9, %16 : i64
        %18 = arith.trunci %17 {hivm.cvp_query.coordinate = "query_task"} : i64 to i32
        %19 = arith.remsi %14, %c2_i64 {hivm.cvp_query.coordinate = "query_slot"} : i64
        %20 = arith.cmpi eq, %19, %c1_i64 : i64
        %21 = arith.cmpi eq, %15, %c0_i64 {hivm.cvp_query.coordinate = "first_key"} : i64
        %22 = arith.cmpi eq, %15, %c16128_i64 {hivm.cvp_query.coordinate = "last_key"} : i64
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
          scf.if %21 {
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
        %23 = scope.scope : () -> memref<64xf32, #hivm.address_space<ub>> {
          %24 = arith.select %20, %alloc_9, %alloc_7 : memref<64xf32, #hivm.address_space<ub>>
          %25 = arith.select %20, %alloc_10, %alloc_8 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_15 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%24 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_15 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_16 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%25 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_16 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.anchor {id = 22 : i64}
          hivm.hir.anchor {id = 23 : i64}
          %alloc_17 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_15 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_18 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_17 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_18 : memref<64xf32, #hivm.address_space<ub>>)
          %26 = arith.select %21, %alloc_0, %alloc_18 : memref<64xf32, #hivm.address_space<ub>>
          %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_16 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_19 : memref<64xf32, #hivm.address_space<ub>>)
          %alloc_20 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          hivm.hir.copy ins(%alloc_19 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_20 : memref<64xf32, #hivm.address_space<ub>>)
          %27 = arith.select %21, %alloc, %alloc_20 : memref<64xf32, #hivm.address_space<ub>>
          scf.if %21 {
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
          %alloc_21 = memref.alloc() : memref<8x129x16x1xf16, #hivm.address_space<ub>>
          %subview_22 = memref.subview %alloc_21[0, 0, 0, 0] [8, 128, 16, 1] [1, 1, 1, 1] : memref<8x129x16x1xf16, #hivm.address_space<ub>> to memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
          %alloc_23 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          %alloc_24 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          annotation.mark %alloc_24 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
          %alloc_25 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
          func.call @_attn_fwd_mix_aiv_outlined_vf_3(%alloc_14, %subview_22, %26, %27, %alloc_23, %alloc_24, %alloc_25) {hivm.vector_function, no_inline} : (memref<256x64xf32, #hivm.address_space<ub>>, memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
          annotation.mark %alloc_13 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 0 : index} : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.anchor {id = 42 : i64}
          %subview_26 = memref.subview %subview_22[0, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>
          %expand_shape = memref.expand_shape %subview_26 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>>
          %subview_27 = memref.subview %subview[0, 0, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          %collapse_shape_28 = memref.collapse_shape %expand_shape [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1]>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>
          %collapse_shape_29 = memref.collapse_shape %subview_27 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%collapse_shape_28 : memref<4x2048xf16, strided<[2064, 1]>, #hivm.address_space<ub>>) outs(%collapse_shape_29 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          %subview_30 = memref.subview %subview_22[4, 0, 0] [4, 128, 16] [1, 1, 1] : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>>
          %expand_shape_31 = memref.expand_shape %subview_30 [[0], [1, 2], [3]] output_shape [4, 8, 16, 16] : memref<4x128x16xf16, strided<[2064, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>>
          %subview_32 = memref.subview %subview[0, 8, 0, 0] [4, 8, 16, 16] [1, 1, 1, 1] : memref<4x16x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> to memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          %collapse_shape_33 = memref.collapse_shape %expand_shape_31 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[2064, 256, 16, 1], offset: 8256>, #hivm.address_space<ub>> into memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>
          %collapse_shape_34 = memref.collapse_shape %subview_32 [[0], [1, 2, 3]] : memref<4x8x16x16xf16, strided<[4096, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>
          hivm.hir.copy ins(%collapse_shape_33 : memref<4x2048xf16, strided<[2064, 1], offset: 8256>, #hivm.address_space<ub>>) outs(%collapse_shape_34 : memref<4x2048xf16, strided<[4096, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
          hivm.hir.anchor {id = 43 : i64}
          hivm.hir.copy ins(%alloc_23 : memref<64xf32, #hivm.address_space<ub>>) outs(%24 : memref<64xf32, #hivm.address_space<ub>>)
          hivm.hir.copy ins(%alloc_25 : memref<64xf32, #hivm.address_space<ub>>) outs(%25 : memref<64xf32, #hivm.address_space<ub>>)
          scope.return %alloc_24 : memref<64xf32, #hivm.address_space<ub>>
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
        scope.scope : () -> () {
          hivm.hir.anchor {id = 55 : i64}
          %24 = arith.divsi %18, %c128_i32 : i32
          %25 = arith.remsi %18, %c128_i32 : i32
          %26 = arith.divsi %24, %c8_i32 : i32
          %27 = arith.remsi %24, %c8_i32 : i32
          %28 = arith.extsi %26 : i32 to i64
          %29 = arith.muli %28, %c8388608_i64 : i64
          %30 = arith.extsi %27 : i32 to i64
          %31 = arith.muli %30, %c1048576_i64 : i64
          %32 = arith.addi %29, %31 : i64
          %33 = arith.index_cast %32 : i64 to index
          %34 = arith.muli %25, %c128_i32 : i32
          %35 = arith.maxsi %34, %c0_i32 : i32
          %36 = arith.index_cast %35 : i32 to index
          %37 = affine.apply #map()[%33, %36]
          %reinterpret_cast = memref.reinterpret_cast %arg6 to offset: [%37], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          %subview_15 = memref.subview %reinterpret_cast[%6, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
          hivm.hir.anchor {id = 56 : i64}
          %38 = arith.select %21, %alloc_1, %alloc_6 : memref<64x64xf32, #hivm.address_space<ub>>
          scf.if %21 {
            hivm.hir.anchor {id = 57 : i64}
            hivm.hir.anchor {id = 58 : i64}
          } else {
            hivm.hir.anchor {id = 59 : i64}
          }
          hivm.hir.anchor {id = 60 : i64}
          hivm.hir.anchor {id = 61 : i64}
          func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0(%23, %alloc_12, %38, %alloc_6) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
          %collapse_shape_16 = memref.collapse_shape %alloc_12 [[0, 1]] : memref<64x64xf32, #hivm.address_space<ub>> into memref<4096xf32, #hivm.address_space<ub>>
          hivm.hir.anchor {id = 62 : i64}
          scf.if %22 {
            %39 = arith.select %20, %alloc_9, %alloc_7 : memref<64xf32, #hivm.address_space<ub>>
            %alloc_17 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%39 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_17 : memref<64xf32, #hivm.address_space<ub>>)
            %40 = arith.select %20, %alloc_10, %alloc_8 : memref<64xf32, #hivm.address_space<ub>>
            %alloc_18 = memref.alloc() : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%40 : memref<64xf32, #hivm.address_space<ub>>) outs(%alloc_18 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.anchor {id = 63 : i64}
            hivm.hir.anchor {id = 64 : i64}
            hivm.hir.anchor {id = 65 : i64}
            %alloc_19 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            %41 = arith.muli %24, %c16384_i32 : i32
            %42 = arith.index_cast %41 : i32 to index
            %43 = arith.index_cast %34 : i32 to index
            %44 = affine.apply #map1()[%42, %43]
            %reinterpret_cast_20 = memref.reinterpret_cast %arg5 to offset: [%44], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %subview_21 = memref.subview %reinterpret_cast_20[%6] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
            %45 = arith.select %20, %alloc_5, %alloc_4 : memref<64xf32, #hivm.address_space<ub>>
            %46 = arith.select %20, %alloc_3, %alloc_2 : memref<64x64xf16, #hivm.address_space<ub>>
            func.call @_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0(%alloc_18, %alloc_17, %alloc_19, %alloc_17, %alloc_6, %46) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
            hivm.hir.anchor {id = 66 : i64}
            hivm.hir.anchor {id = 67 : i64}
            hivm.hir.anchor {id = 68 : i64}
            hivm.hir.copy ins(%alloc_19 : memref<64xf32, #hivm.address_space<ub>>) outs(%45 : memref<64xf32, #hivm.address_space<ub>>)
            hivm.hir.store ins(%45 : memref<64xf32, #hivm.address_space<ub>>) outs(%subview_21 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.anchor {id = 69 : i64}
            %47 = arith.select %20, %alloc_3, %alloc_2 : memref<64x64xf16, #hivm.address_space<ub>>
            %collapse_shape_22 = memref.collapse_shape %47 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
            %collapse_shape_23 = memref.collapse_shape %subview_15 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
            hivm.hir.store ins(%collapse_shape_22 : memref<4096xf16, #hivm.address_space<ub>>) outs(%collapse_shape_23 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
            hivm.hir.anchor {id = 70 : i64}
          } else {
            hivm.hir.anchor {id = 71 : i64}
          }
          hivm.hir.anchor {id = 72 : i64}
          scope.return
        } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
        hivm.hir.anchor {id = 73 : i64}
        hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
      } {fixpipe_for_mmad_result_already_inserted = true, hivm.cvp_query.output_storage, hivm.cvp_query.state_storage, hivm.cvp_query.stream = {key_count = 16384 : i64, key_step = 256 : i64, key_unroll = 1 : i32, physical_lifetimes_verified = false, pipeline_mode = "skew", query_slots = 2 : i32, query_step = 28 : i64}}
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 74 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 75 : i64}
    return
  }
}

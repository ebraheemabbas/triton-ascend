// RUN: bishengir-opt %s --verify-each --hivm-cvp-attention-vector="require-match=true" | FileCheck %s
// CHECK: hivm.cvp_attention.score_store_removals = 1 : i64

#map = affine_map<()[s0, s1] -> (s0 + s1 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<(d0) -> (d0, 0)>
#map3 = affine_map<(d0) -> (d0 * 16)>
#map4 = affine_map<()[s0] -> (s0 * 64)>
#map5 = affine_map<()[s0] -> (s0 * 4)>
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @renamed_symbol_0(%named_value_0: memref<?xi8> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %named_value_1: memref<?xi8> {hacc.arg_type = #hacc.arg_type<workspace>}, %named_value_2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %named_value_3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %named_value_4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %named_value_5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %named_value_6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %named_value_7: i32, %named_value_8: i32, %named_value_9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module">, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.backup_function, hivm.func_core_type = #hivm.func_core_type<MIX>, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    hivm.hir.anchor {id = 0 : i64}
    %named_value_10 = arith.constant 0.000000e+00 : f32
    %named_value_11 = arith.constant 262144 : i32
    %named_value_12 = arith.constant 32768 : i32
    %named_value_13 = arith.constant 128 : i32
    %named_value_14 = arith.constant 2097152 : i64
    %named_value_15 = arith.constant 16777216 : i64
    %named_value_16 = arith.constant 8 : i32
    %named_value_17 = arith.constant 256 : i32
    %named_value_18 = arith.constant 0 : i32
    %named_value_19 = arith.constant 64 : index
    %named_value_20 = arith.constant 5.000000e-01 : f32
    %named_value_21 = arith.constant 0xFF800000 : f32
    %named_value_22 = arith.constant 1.000000e+00 : f32
    %named_value_23 = arith.constant 128 : index
    %named_value_24 = arith.constant 256 : index
    %named_value_25 = arith.constant true
    %named_value_26 = arith.constant 28 : i32
    %named_value_27 = arith.muli %named_value_7, %named_value_8 : i32
    %named_value_28 = arith.muli %named_value_27, %named_value_9 : i32
    annotation.mark %named_value_28 {logical_block_num} : i32
    %named_value_29 = hivm.hir.get_block_idx -> i64
    %named_value_30 = arith.trunci %named_value_29 : i64 to i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %named_value_31 = %named_value_30 to %named_value_28 step %named_value_26  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %named_value_32 = arith.remsi %named_value_31, %named_value_7 : i32
      %named_value_33 = tensor.empty() : tensor<128x64xf32>
      hivm.hir.anchor {id = 3 : i64}
      %named_value_34 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%named_value_10 : f32) outs(%named_value_33 : tensor<128x64xf32>) -> tensor<128x64xf32>
      hivm.hir.anchor {id = 4 : i64}
      %named_value_35 = tensor.empty() : tensor<128xf32>
      %named_value_36 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%named_value_21 : f32) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
      hivm.hir.anchor {id = 5 : i64}
      %named_value_37 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%named_value_22 : f32) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
      hivm.hir.anchor {id = 6 : i64}
      scf.for %named_value_38 = %named_value_32 to %named_value_11 step %named_value_26  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %named_value_39 = arith.divsi %named_value_38, %named_value_17 : i32
        %named_value_40 = arith.remsi %named_value_38, %named_value_17 : i32
        %named_value_41 = arith.divsi %named_value_39, %named_value_16 : i32
        %named_value_42 = arith.remsi %named_value_39, %named_value_16 : i32
        %named_value_43 = arith.extsi %named_value_41 : i32 to i64
        %named_value_44 = arith.muli %named_value_43, %named_value_15 : i64
        %named_value_45 = arith.extsi %named_value_42 : i32 to i64
        %named_value_46 = arith.muli %named_value_45, %named_value_14 : i64
        %named_value_47 = arith.addi %named_value_44, %named_value_46 : i64
        %named_value_48 = arith.index_cast %named_value_47 : i64 to index
        %named_value_49 = arith.muli %named_value_40, %named_value_13 : i32
        %named_value_50 = arith.maxsi %named_value_49, %named_value_18 : i32
        %named_value_51 = arith.index_cast %named_value_50 : i32 to index
        %named_value_52 = affine.apply #map()[%named_value_48, %named_value_51]
        %named_value_53 = memref.reinterpret_cast %named_value_2 to offset: [%named_value_52], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %named_value_54 = memref.reinterpret_cast %named_value_6 to offset: [%named_value_52], sizes: [128, 64], strides: [64, 1] : memref<?xf16> to memref<128x64xf16, strided<[64, 1], offset: ?>>
        %named_value_55 = memref.alloc() : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%named_value_53 : memref<128x64xf16, strided<[64, 1], offset: ?>>) outs(%named_value_55 : memref<4x8x16x16xf16>)
        hivm.hir.anchor {id = 9 : i64}
        %named_value_56 = bufferization.to_tensor %named_value_55 restrict writable : memref<4x8x16x16xf16>
        hivm.hir.anchor {id = 10 : i64}
        %named_value_57:5 = scf.for %named_value_58 = %named_value_18 to %named_value_12 step %named_value_17 iter_args(%named_value_59 = %named_value_34, %named_value_60 = %named_value_37, %named_value_61 = %named_value_36, %named_value_62 = %named_value_18, %named_value_63 = %named_value_18) -> (tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %named_value_64 = memref.alloc() : memref<128x64xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_64 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128x64xf32, #hivm.address_space<ub>>
          %named_value_65 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %named_value_65 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %named_value_66 = memref.alloc() : memref<128x256xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_66 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<128x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.anchor {id = 12 : i64}
          %named_value_67 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 13 : i64}
            %named_value_68 = arith.maxsi %named_value_62, %named_value_18 : i32
            %named_value_69 = arith.index_cast %named_value_68 : i32 to index
            %named_value_70 = affine.apply #map()[%named_value_48, %named_value_69]
            %named_value_71 = memref.reinterpret_cast %named_value_3 to offset: [%named_value_70], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %named_value_72 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%named_value_71 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%named_value_72 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 15 : i64}
            %named_value_73 = bufferization.to_tensor %named_value_72 restrict writable : memref<4x16x16x16xf16>
            %named_value_74 = tensor.empty() : tensor<16x8x16x16xf32>
            hivm.hir.anchor {id = 16 : i64}
            %named_value_75 = hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%named_value_56, %named_value_73, %named_value_25, %named_value_23, %named_value_19, %named_value_24 : tensor<4x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%named_value_74 : tensor<16x8x16x16xf32>) -> tensor<16x8x16x16xf32>
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %named_value_66 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>} : memref<128x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%named_value_75 : tensor<16x8x16x16xf32>) outs(%named_value_66 : memref<128x256xf32, #hivm.address_space<ub>>)
            hivm.hir.anchor {id = 19 : i64}
            %named_value_76 = arith.addi %named_value_62, %named_value_17 : i32
            scope.return %named_value_76 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 3 : i32, no_inline}
          hivm.hir.anchor {id = 20 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          %named_value_77 = bufferization.to_tensor %named_value_66 restrict writable : memref<128x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 21 : i64}
          %named_value_78:3 = scope.scope : () -> (tensor<128xf32>, tensor<128xf32>, tensor<128xf32>) {
            hivm.hir.anchor {id = 22 : i64}
            %named_value_68 = tensor.empty() : tensor<128x256xf32>
            hivm.hir.anchor {id = 23 : i64}
            %named_value_69 = hivm.hir.vmul ins(%named_value_77, %named_value_20 : tensor<128x256xf32>, f32) outs(%named_value_68 : tensor<128x256xf32>) -> tensor<128x256xf32>
            hivm.hir.anchor {id = 24 : i64}
            %named_value_79 = tensor.expand_shape %named_value_36 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            %named_value_70 = hivm.hir.vreduce <max> ins(%named_value_69 : tensor<128x256xf32>) outs(%named_value_79 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
            hivm.hir.anchor {id = 25 : i64}
            %named_value_80 = tensor.collapse_shape %named_value_70 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
            %named_value_73 = hivm.hir.vmax ins(%named_value_61, %named_value_80 : tensor<128xf32>, tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 26 : i64}
            %named_value_81 = tensor.expand_shape %named_value_73 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            %named_value_74 = hivm.hir.vsub ins(%named_value_69, %named_value_81 : tensor<128x256xf32>, tensor<128x1xf32>) outs(%named_value_68 : tensor<128x256xf32>) broadcast = [1] -> tensor<128x256xf32>
            hivm.hir.anchor {id = 27 : i64}
            %named_value_75 = hivm.hir.vexp ins(%named_value_74 : tensor<128x256xf32>) outs(%named_value_68 : tensor<128x256xf32>) -> tensor<128x256xf32>
            hivm.hir.anchor {id = 28 : i64}
            %named_value_76 = tensor.empty() : tensor<128x256xf16>
            %named_value_82 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%named_value_75 : tensor<128x256xf32>) outs(%named_value_76 : tensor<128x256xf16>) -> tensor<128x256xf16>
            hivm.hir.anchor {id = 29 : i64}
            %named_value_83 = hivm.hir.vbrc {hivm.tcore_type = #hivm.tcore_type<VECTOR>} ins(%named_value_10 : f32) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 30 : i64}
            %named_value_84 = tensor.expand_shape %named_value_83 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            %named_value_85 = hivm.hir.vreduce <sum> ins(%named_value_75 : tensor<128x256xf32>) outs(%named_value_84 : tensor<128x1xf32>) reduce_dims = [1] -> tensor<128x1xf32>
            hivm.hir.anchor {id = 31 : i64}
            %named_value_86 = tensor.collapse_shape %named_value_85 [[0, 1]] : tensor<128x1xf32> into tensor<128xf32>
            %named_value_87 = hivm.hir.vsub ins(%named_value_61, %named_value_73 : tensor<128xf32>, tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 32 : i64}
            %named_value_88 = hivm.hir.vexp ins(%named_value_87 : tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 33 : i64}
            %named_value_89 = hivm.hir.vmul ins(%named_value_60, %named_value_88 : tensor<128xf32>, tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 34 : i64}
            %named_value_90 = hivm.hir.vadd ins(%named_value_89, %named_value_86 : tensor<128xf32>, tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
            hivm.hir.anchor {id = 35 : i64}
            %named_value_91 = tensor.expand_shape %named_value_82 [[0], [1, 2]] output_shape [128, 16, 16] : tensor<128x256xf16> into tensor<128x16x16xf16>
            %named_value_92 = tensor.empty() : tensor<16x128x16xf16>
            %named_value_93 = hivm.hir.vtranspose ins(%named_value_91 : tensor<128x16x16xf16>) outs(%named_value_92 : tensor<16x128x16xf16>) permutation = [1, 0, 2] -> tensor<16x128x16xf16>
            hivm.hir.anchor {id = 36 : i64}
            %named_value_94 = tensor.expand_shape %named_value_93 [[0], [1, 2], [3]] output_shape [16, 8, 16, 16] : tensor<16x128x16xf16> into tensor<16x8x16x16xf16>
            annotation.mark %named_value_65 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            hivm.hir.copy ins(%named_value_94 : tensor<16x8x16x16xf16>) outs(%named_value_65 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", hivm.tcore_type = #hivm.tcore_type<VECTOR>}
            hivm.hir.anchor {id = 38 : i64}
            scope.return %named_value_88, %named_value_73, %named_value_90 : tensor<128xf32>, tensor<128xf32>, tensor<128xf32>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 39 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
          %named_value_95 = bufferization.to_tensor %named_value_65 restrict writable : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.anchor {id = 40 : i64}
          %named_value_96 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 41 : i64}
            %named_value_68 = arith.maxsi %named_value_63, %named_value_18 : i32
            %named_value_69 = arith.index_cast %named_value_68 : i32 to index
            %named_value_70 = affine.apply #map()[%named_value_48, %named_value_69]
            %named_value_71 = memref.reinterpret_cast %named_value_4 to offset: [%named_value_70], sizes: [256, 64], strides: [64, 1] : memref<?xf16> to memref<256x64xf16, strided<[64, 1], offset: ?>>
            %named_value_72 = memref.alloc() : memref<4x16x16x16xf16>
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%named_value_71 : memref<256x64xf16, strided<[64, 1], offset: ?>>) outs(%named_value_72 : memref<4x16x16x16xf16>)
            hivm.hir.anchor {id = 43 : i64}
            %named_value_73 = bufferization.to_tensor %named_value_72 restrict writable : memref<4x16x16x16xf16>
            %named_value_74 = tensor.empty() : tensor<4x8x16x16xf32>
            hivm.hir.anchor {id = 44 : i64}
            %named_value_75 = hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%named_value_95, %named_value_73, %named_value_25, %named_value_23, %named_value_24, %named_value_19 : tensor<16x8x16x16xf16>, tensor<4x16x16x16xf16>, i1, index, index, index) outs(%named_value_74 : tensor<4x8x16x16xf32>) -> tensor<4x8x16x16xf32>
            hivm.hir.anchor {id = 45 : i64}
            annotation.mark %named_value_64 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>} : memref<128x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%named_value_75 : tensor<4x8x16x16xf32>) outs(%named_value_64 : memref<128x64xf32, #hivm.address_space<ub>>)
            hivm.hir.anchor {id = 47 : i64}
            %named_value_76 = arith.addi %named_value_63, %named_value_17 : i32
            scope.return %named_value_76 : i32
          } {hivm.loop_core_type = #hivm.tcore_type<CUBE>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 1 : i32, no_inline}
          hivm.hir.anchor {id = 48 : i64}
          hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
          %named_value_97 = bufferization.to_tensor %named_value_64 restrict writable : memref<128x64xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
          hivm.hir.anchor {id = 49 : i64}
          %named_value_98 = scope.scope : () -> tensor<128x64xf32> {
            hivm.hir.anchor {id = 50 : i64}
            %named_value_79 = tensor.expand_shape %named_value_78#0 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
            hivm.hir.anchor {id = 51 : i64}
            %named_value_68 = hivm.hir.vmul ins(%named_value_59, %named_value_79 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%named_value_33 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
            hivm.hir.anchor {id = 52 : i64}
            %named_value_69 = hivm.hir.vadd ins(%named_value_97, %named_value_68 : tensor<128x64xf32>, tensor<128x64xf32>) outs(%named_value_33 : tensor<128x64xf32>) -> tensor<128x64xf32>
            hivm.hir.anchor {id = 53 : i64}
            scope.return %named_value_69 : tensor<128x64xf32>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 54 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
          scf.yield %named_value_98, %named_value_78#2, %named_value_78#1, %named_value_67, %named_value_96 : tensor<128x64xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 55 : i64}
        %named_value_99 = hivm.hir.vln ins(%named_value_57#1 : tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
        hivm.hir.anchor {id = 56 : i64}
        %named_value_100 = hivm.hir.vadd ins(%named_value_57#2, %named_value_99 : tensor<128xf32>, tensor<128xf32>) outs(%named_value_35 : tensor<128xf32>) -> tensor<128xf32>
        hivm.hir.anchor {id = 57 : i64}
        %named_value_101 = tensor.expand_shape %named_value_57#1 [[0, 1]] output_shape [128, 1] : tensor<128xf32> into tensor<128x1xf32>
        %named_value_102 = hivm.hir.vdiv ins(%named_value_57#0, %named_value_101 : tensor<128x64xf32>, tensor<128x1xf32>) outs(%named_value_33 : tensor<128x64xf32>) broadcast = [1] -> tensor<128x64xf32>
        hivm.hir.anchor {id = 58 : i64}
        %named_value_103 = arith.muli %named_value_39, %named_value_12 : i32
        %named_value_104 = arith.index_cast %named_value_103 : i32 to index
        %named_value_105 = arith.index_cast %named_value_49 : i32 to index
        %named_value_106 = affine.apply #map1()[%named_value_104, %named_value_105]
        %named_value_107 = memref.reinterpret_cast %named_value_5 to offset: [%named_value_106], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
        hivm.hir.store ins(%named_value_100 : tensor<128xf32>) outs(%named_value_107 : memref<128xf32, strided<[1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        hivm.hir.anchor {id = 59 : i64}
        %named_value_108 = tensor.empty() : tensor<128x64xf16>
        %named_value_109 = hivm.hir.vcast {enable_overflow = true, enable_saturate = false, hivm.unsigned_mode = #hivm.unsigned_mode<si2si>} ins(%named_value_102 : tensor<128x64xf32>) outs(%named_value_108 : tensor<128x64xf16>) -> tensor<128x64xf16>
        hivm.hir.anchor {id = 60 : i64}
        hivm.hir.store ins(%named_value_109 : tensor<128x64xf16>) outs(%named_value_54 : memref<128x64xf16, strided<[64, 1], offset: ?>>) {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
        hivm.hir.anchor {id = 61 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 62 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 63 : i64}
    return
  }
  func.func @renamed_symbol_1(%named_value_0: memref<64xf32, #hivm.address_space<ub>>, %named_value_1: memref<64xf32, #hivm.address_space<ub>>, %named_value_2: memref<64xf32, #hivm.address_space<ub>>, %named_value_3: memref<64xf32, #hivm.address_space<ub>>, %named_value_4: memref<64x64xf32, #hivm.address_space<ub>>, %named_value_5: memref<64x64xf16, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function} {
    %named_value_19 = arith.constant 64 : index
    %named_value_110 = arith.constant 1 : index
    %named_value_10 = arith.constant 0.000000e+00 : f32
    %named_value_111 = arith.constant 0 : index
    %named_value_27 = vector.transfer_read %named_value_0[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_28 = vector.transfer_read %named_value_1[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_29 = math.log %named_value_28 : vector<64xf32>
    %named_value_30 = arith.addf %named_value_27, %named_value_29 : vector<64xf32>
    vector.transfer_write %named_value_30, %named_value_2[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    scf.for %named_value_6 = %named_value_111 to %named_value_19 step %named_value_110 {
      %named_value_112 = memref.subview %named_value_3[%named_value_6] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %named_value_113 = memref.subview %named_value_4[%named_value_6, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %named_value_114 = memref.subview %named_value_5[%named_value_6, 0] [1, 64] [1, 1] : memref<64x64xf16, #hivm.address_space<ub>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %named_value_32 = vector.transfer_read %named_value_113[%named_value_111, %named_value_111], %named_value_10 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_33 = vector.transfer_read %named_value_112[%named_value_111], %named_value_10 {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_34 = arith.divf %named_value_32, %named_value_33 : vector<1x64xf32>
      %named_value_35 = arith.truncf %named_value_34 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %named_value_35, %named_value_114[%named_value_111, %named_value_111] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @renamed_symbol_2(%named_value_0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %named_value_1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %named_value_2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %named_value_6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %named_value_7: i32, %named_value_8: i32, %named_value_9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %named_value_26 = arith.constant 28 : i32
    %named_value_25 = arith.constant true
    %named_value_24 = arith.constant 256 : index
    %named_value_23 = arith.constant 128 : index
    %named_value_19 = arith.constant 64 : index
    %named_value_18 = arith.constant 0 : i32
    %named_value_17 = arith.constant 256 : i32
    %named_value_16 = arith.constant 8 : i32
    %named_value_15 = arith.constant 16777216 : i64
    %named_value_14 = arith.constant 2097152 : i64
    %named_value_13 = arith.constant 128 : i32
    %named_value_12 = arith.constant 32768 : i32
    %named_value_11 = arith.constant 262144 : i32
    hivm.hir.anchor {id = 0 : i64}
    %named_value_27 = arith.muli %named_value_7, %named_value_8 : i32
    %named_value_28 = arith.muli %named_value_27, %named_value_9 : i32
    annotation.mark %named_value_28 {logical_block_num} : i32
    %named_value_29 = hivm.hir.get_block_idx -> i64
    %named_value_30 = arith.trunci %named_value_29 : i64 to i32
    hivm.hir.sync_block_set[<CUBE>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 1 : i64}
    scf.for %named_value_31 = %named_value_30 to %named_value_28 step %named_value_26  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %named_value_32 = arith.remsi %named_value_31, %named_value_7 : i32
      hivm.hir.anchor {id = 3 : i64}
      hivm.hir.anchor {id = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.anchor {id = 6 : i64}
      scf.for %named_value_38 = %named_value_32 to %named_value_11 step %named_value_26  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %named_value_33 = arith.divsi %named_value_38, %named_value_17 : i32
        %named_value_34 = arith.remsi %named_value_38, %named_value_17 : i32
        %named_value_35 = arith.divsi %named_value_33, %named_value_16 : i32
        %named_value_36 = arith.remsi %named_value_33, %named_value_16 : i32
        %named_value_37 = arith.extsi %named_value_35 : i32 to i64
        %named_value_39 = arith.muli %named_value_37, %named_value_15 : i64
        %named_value_40 = arith.extsi %named_value_36 : i32 to i64
        %named_value_41 = arith.muli %named_value_40, %named_value_14 : i64
        %named_value_42 = arith.addi %named_value_39, %named_value_41 : i64
        %named_value_43 = arith.index_cast %named_value_42 : i64 to index
        %named_value_44 = arith.muli %named_value_34, %named_value_13 : i32
        %named_value_45 = arith.maxsi %named_value_44, %named_value_18 : i32
        %named_value_46 = arith.index_cast %named_value_45 : i32 to index
        %named_value_47 = affine.apply #map()[%named_value_43, %named_value_46]
        %named_value_53 = memref.reinterpret_cast %named_value_2 to offset: [%named_value_47], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %named_value_55 = memref.alloc() : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        annotation.mark %named_value_55 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.nd2nz {dst_continuous} ins(%named_value_53 : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%named_value_55 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>)
        hivm.hir.anchor {id = 9 : i64}
        hivm.hir.anchor {id = 10 : i64}
        %named_value_48:2 = scf.for %named_value_58 = %named_value_18 to %named_value_12 step %named_value_17 iter_args(%named_value_59 = %named_value_18, %named_value_60 = %named_value_18) -> (i32, i32)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %named_value_115 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_115 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
          %named_value_116 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          annotation.mark %named_value_116 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %named_value_117 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_117 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1
          hivm.hir.anchor {id = 12 : i64}
          %named_value_49 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 13 : i64}
            %named_value_51 = arith.maxsi %named_value_59, %named_value_18 : i32
            %named_value_52 = arith.index_cast %named_value_51 : i32 to index
            %named_value_56 = affine.apply #map()[%named_value_43, %named_value_52]
            %named_value_54 = memref.reinterpret_cast %named_value_3 to offset: [%named_value_56], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
            %named_value_118 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
            annotation.mark %named_value_118 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 14 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%named_value_54 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%named_value_118 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 15 : i64}
            hivm.hir.anchor {id = 16 : i64}
            %named_value_64 = memref.alloc() {alignment = 64 : i64} : memref<16x8x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %named_value_64 {hivm.multi_buffer = 2 : i32} : memref<16x8x16x16xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {already_set_real_mkn, b_transpose, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%named_value_55, %named_value_118, %named_value_25, %named_value_23, %named_value_19, %named_value_24 : memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%named_value_64 : memref<16x8x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.anchor {id = 17 : i64}
            annotation.mark %named_value_117 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 18 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%named_value_64 : memref<16x8x16x16xf32, #hivm.address_space<cc>>) outs(%named_value_117 : memref<64x256xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 19 : i64}
            %named_value_57 = arith.addi %named_value_59, %named_value_17 : i32
            scope.return %named_value_57 : i32
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
            annotation.mark %named_value_116 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = -1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            hivm.hir.anchor {id = 38 : i64}
            scope.return
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 2 : i32, no_inline}
          hivm.hir.anchor {id = 39 : i64}
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0
          hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
          hivm.hir.anchor {id = 40 : i64}
          %named_value_50 = scope.scope : () -> i32 {
            hivm.hir.anchor {id = 41 : i64}
            %named_value_51 = arith.maxsi %named_value_60, %named_value_18 : i32
            %named_value_52 = arith.index_cast %named_value_51 : i32 to index
            %named_value_56 = affine.apply #map()[%named_value_43, %named_value_52]
            %named_value_54 = memref.reinterpret_cast %named_value_4 to offset: [%named_value_56], sizes: [256, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
            %named_value_118 = memref.alloc() : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
            annotation.mark %named_value_118 {hivm.multi_buffer = 2 : i32} : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 42 : i64}
            hivm.hir.nd2nz {dst_continuous} ins(%named_value_54 : memref<256x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>) outs(%named_value_118 : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>)
            hivm.hir.anchor {id = 43 : i64}
            hivm.hir.anchor {id = 44 : i64}
            %named_value_64 = memref.alloc() {alignment = 64 : i64} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
            annotation.mark %named_value_64 {hivm.multi_buffer = 2 : i32} : memref<4x8x16x16xf32, #hivm.address_space<cc>>
            hivm.hir.mmadL1 {already_set_real_mkn, fixpipe_for_result_already_inserted = true, normalized_in_L0C} ins(%named_value_116, %named_value_118, %named_value_25, %named_value_23, %named_value_24, %named_value_19 : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%named_value_64 : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
            hivm.hir.anchor {id = 45 : i64}
            annotation.mark %named_value_115 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
            hivm.hir.anchor {id = 46 : i64}
            hivm.hir.fixpipe {dma_mode = #hivm.dma_mode<nz2nd>} ins(%named_value_64 : memref<4x8x16x16xf32, #hivm.address_space<cc>>) outs(%named_value_115 : memref<64x64xf32, #hivm.address_space<ub>>) dual_dst_mode = <ROW_SPLIT>
            hivm.hir.anchor {id = 47 : i64}
            %named_value_57 = arith.addi %named_value_60, %named_value_17 : i32
            scope.return %named_value_57 : i32
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
          scf.yield %named_value_49, %named_value_50 : i32, i32
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
  func.func @renamed_symbol_3(%named_value_0: memref<64xf32, #hivm.address_space<ub>>, %named_value_1: memref<64xf32, #hivm.address_space<ub>>, %named_value_2: memref<64xf32, #hivm.address_space<ub>>, %named_value_3: memref<64xf32, #hivm.address_space<ub>>, %named_value_4: memref<64xf32, #hivm.address_space<ub>>, %named_value_5: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant 0.000000e+00 : f32
    %named_value_111 = arith.constant 0 : index
    %named_value_27 = vector.transfer_read %named_value_1[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_28 = vector.transfer_read %named_value_2[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_29 = arith.subf %named_value_27, %named_value_28 : vector<64xf32>
    %named_value_30 = math.exp %named_value_29 : vector<64xf32>
    vector.transfer_write %named_value_30, %named_value_5[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    %named_value_32 = vector.transfer_read %named_value_0[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_33 = vector.transfer_read %named_value_3[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_34 = arith.mulf %named_value_32, %named_value_30 : vector<64xf32>
    %named_value_35 = arith.addf %named_value_34, %named_value_33 : vector<64xf32>
    vector.transfer_write %named_value_35, %named_value_4[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @renamed_symbol_4(%named_value_0: memref<64xf32, #hivm.address_space<ub>>, %named_value_1: memref<64x64xf32, #hivm.address_space<ub>>, %named_value_2: memref<64x64xf32, #hivm.address_space<ub>>, %named_value_3: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant 0.000000e+00 : f32
    %named_value_110 = arith.constant 1 : index
    %named_value_19 = arith.constant 64 : index
    %named_value_111 = arith.constant 0 : index
    scf.for %named_value_4 = %named_value_111 to %named_value_19 step %named_value_110 {
      %named_value_112 = memref.subview %named_value_0[%named_value_4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %named_value_113 = memref.subview %named_value_1[%named_value_4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %named_value_114 = memref.subview %named_value_2[%named_value_4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %named_value_119 = memref.subview %named_value_3[%named_value_4, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %named_value_27 = vector.transfer_read %named_value_113[%named_value_111, %named_value_111], %named_value_10 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_28 = vector.transfer_read %named_value_114[%named_value_111, %named_value_111], %named_value_10 {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_29 = vector.transfer_read %named_value_112[%named_value_111], %named_value_10 {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_30 = arith.mulf %named_value_28, %named_value_29 : vector<1x64xf32>
      %named_value_32 = arith.addf %named_value_27, %named_value_30 : vector<1x64xf32>
      vector.transfer_write %named_value_32, %named_value_119[%named_value_111, %named_value_111] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @renamed_symbol_5(%named_value_0: memref<64x256xf32, #hivm.address_space<ub>>, %named_value_1: memref<64xf32, #hivm.address_space<ub>>, %named_value_2: memref<64x256xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant dense<5.000000e-01> : vector<1x64xf32>
    %named_value_20 = arith.constant dense<0xFF800000> : vector<1x64xf32>
    %named_value_21 = arith.constant 0.000000e+00 : f32
    %named_value_110 = arith.constant 1 : index
    %named_value_19 = arith.constant 64 : index
    %named_value_24 = arith.constant 256 : index
    %named_value_111 = arith.constant 0 : index
    scf.for %named_value_3 = %named_value_111 to %named_value_19 step %named_value_110 {
      %named_value_112 = memref.subview %named_value_1[%named_value_3] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %named_value_27 = scf.for %named_value_4 = %named_value_111 to %named_value_24 step %named_value_19 iter_args(%named_value_5 = %named_value_20) -> (vector<1x64xf32>) {
        %named_value_119 = memref.subview %named_value_0[%named_value_3, %named_value_4] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %named_value_120 = memref.subview %named_value_2[%named_value_3, %named_value_4] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %named_value_30 = vector.transfer_read %named_value_119[%named_value_111, %named_value_111], %named_value_21 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %named_value_32 = arith.mulf %named_value_30, %named_value_10 : vector<1x64xf32>
        vector.transfer_write %named_value_32, %named_value_120[%named_value_111, %named_value_111] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %named_value_33 = arith.maximumf %named_value_32, %named_value_5 {reductionOp} : vector<1x64xf32>
        scf.yield %named_value_33 : vector<1x64xf32>
      } {reductionLoop}
      %named_value_28 = vector.transfer_read %named_value_112[%named_value_111], %named_value_21 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %named_value_29 = vector.multi_reduction <maximumf>, %named_value_27, %named_value_28 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %named_value_29, %named_value_112[%named_value_111] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @renamed_symbol_6(%named_value_0: memref<64xf32, #hivm.address_space<ub>>, %named_value_1: memref<64xf32, #hivm.address_space<ub>>, %named_value_2: memref<64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant 0.000000e+00 : f32
    %named_value_111 = arith.constant 0 : index
    %named_value_27 = vector.transfer_read %named_value_0[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_28 = vector.transfer_read %named_value_1[%named_value_111], %named_value_10 {in_bounds = [true]} : memref<64xf32, #hivm.address_space<ub>>, vector<64xf32>
    %named_value_29 = arith.maximumf %named_value_27, %named_value_28 : vector<64xf32>
    vector.transfer_write %named_value_29, %named_value_2[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @renamed_symbol_7(%named_value_0: memref<64xf32, #hivm.address_space<ub>>, %named_value_1: memref<64x256xf32, #hivm.address_space<ub>>, %named_value_2: memref<64xf32, #hivm.address_space<ub>>, %named_value_3: memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %named_value_20 = arith.constant 0.000000e+00 : f32
    %named_value_110 = arith.constant 1 : index
    %named_value_19 = arith.constant 64 : index
    %named_value_121 = arith.constant 4 : index
    %named_value_122 = arith.constant 16 : index
    %named_value_111 = arith.constant 0 : index
    scf.for %named_value_4 = %named_value_111 to %named_value_19 step %named_value_110 {
      %named_value_112 = memref.subview %named_value_2[%named_value_4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %named_value_114 = memref.subview %named_value_0[%named_value_4] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %named_value_27 = vector.transfer_read %named_value_114[%named_value_111], %named_value_20 {in_bounds = [true, true], permutation_map = #map2} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %named_value_28 = scf.for %named_value_5 = %named_value_111 to %named_value_122 step %named_value_121 iter_args(%named_value_6 = %named_value_10) -> (vector<1x64xf32>) {
        %named_value_32 = affine.apply #map3(%named_value_5)
        %named_value_119 = memref.subview %named_value_1[%named_value_4, %named_value_32] [1, 64] [1, 1] : memref<64x256xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>
        %named_value_33 = vector.transfer_read %named_value_119[%named_value_111, %named_value_111], %named_value_20 {in_bounds = [true, true]} : memref<1x64xf32, strided<[256, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
        %named_value_34 = arith.subf %named_value_33, %named_value_27 : vector<1x64xf32>
        %named_value_35 = math.exp %named_value_34 : vector<1x64xf32>
        %named_value_36 = arith.addf %named_value_35, %named_value_6 {reductionOp} : vector<1x64xf32>
        %named_value_37 = arith.truncf %named_value_35 {round_mode = #hfusion.round_mode<rint>} : vector<1x64xf32> to vector<1x64xf16>
        %named_value_120 = memref.subview %named_value_3[%named_value_5, %named_value_4, 0] [4, 1, 16] [1, 1, 1] : memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> to memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        %named_value_39 = vector.shape_cast %named_value_37 : vector<1x64xf16> to vector<4x1x16xf16>
        vector.transfer_write %named_value_39, %named_value_120[%named_value_111, %named_value_111, %named_value_111] {in_bounds = [true, true, true]} : vector<4x1x16xf16>, memref<4x1x16xf16, strided<[1040, 16, 1], offset: ?>, #hivm.address_space<ub>>
        scf.yield %named_value_36 : vector<1x64xf32>
      } {reductionLoop, unroll_for_vsstb}
      %named_value_29 = vector.transfer_read %named_value_112[%named_value_111], %named_value_20 {in_bounds = [true]} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1xf32>
      %named_value_30 = vector.multi_reduction <add>, %named_value_28, %named_value_29 {withoutInitMergeOp} [1] : vector<1x64xf32> to vector<1xf32>
      vector.transfer_write %named_value_30, %named_value_112[%named_value_111] {in_bounds = [true]} : vector<1xf32>, memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @renamed_symbol_8(%named_value_0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant dense<1.000000e+00> : vector<64xf32>
    %named_value_111 = arith.constant 0 : index
    vector.transfer_write %named_value_10, %named_value_0[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @renamed_symbol_9(%named_value_0: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant dense<0.000000e+00> : vector<1x64xf32>
    %named_value_110 = arith.constant 1 : index
    %named_value_19 = arith.constant 64 : index
    %named_value_111 = arith.constant 0 : index
    scf.for %named_value_1 = %named_value_111 to %named_value_19 step %named_value_110 {
      %named_value_112 = memref.subview %named_value_0[%named_value_1, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      vector.transfer_write %named_value_10, %named_value_112[%named_value_111, %named_value_111] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
  func.func @renamed_symbol_10(%named_value_0: memref<64xf32, #hivm.address_space<ub>>) attributes {hfusion.has_fill, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %named_value_10 = arith.constant dense<0xFF800000> : vector<64xf32>
    %named_value_111 = arith.constant 0 : index
    vector.transfer_write %named_value_10, %named_value_0[%named_value_111] {in_bounds = [true]} : vector<64xf32>, memref<64xf32, #hivm.address_space<ub>>
    return
  }
  func.func @renamed_symbol_11(%named_value_0: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<sync_block_lock>}, %named_value_1: memref<?xi8, #hivm.address_space<gm>> {hacc.arg_type = #hacc.arg_type<workspace>}, %named_value_2: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_3: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_4: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 0 : i32}, %named_value_5: memref<?xf32, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %named_value_6: memref<?xf16, #hivm.address_space<gm>> {tt.tensor_kind = 1 : i32}, %named_value_7: i32, %named_value_8: i32, %named_value_9: i32) attributes {hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 256 : i64}], SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, func_dyn_memref_args = dense<[true, true, true, true, true, true, true, false, false, false]> : vector<10xi1>, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.vf_mode = #hivm.vf_mode<SIMD>, mix_mode = "mix", parallel_mode = "simd"} {
    %named_value_11 = arith.constant 262144 : i32
    %named_value_12 = arith.constant 32768 : i32
    %named_value_13 = arith.constant 128 : i32
    %named_value_14 = arith.constant 2097152 : i64
    %named_value_15 = arith.constant 16777216 : i64
    %named_value_16 = arith.constant 8 : i32
    %named_value_17 = arith.constant 256 : i32
    %named_value_18 = arith.constant 0 : i32
    %named_value_26 = arith.constant 28 : i32
    %named_value_27 = arith.muli %named_value_7, %named_value_8 : i32
    %named_value_28 = arith.muli %named_value_27, %named_value_9 : i32
    %named_value_29 = hivm.hir.get_block_idx -> i64
    %named_value_30 = arith.trunci %named_value_29 : i64 to i32
    %named_value_55 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
    call @renamed_symbol_10(%named_value_55) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
    %named_value_32 = hivm.hir.get_sub_block_idx -> i64
    %named_value_33 = arith.index_cast %named_value_32 : i64 to index
    %named_value_34 = affine.apply #map4()[%named_value_33]
    %named_value_35 = affine.apply #map5()[%named_value_33]
    hivm.hir.anchor {id = 0 : i64}
    annotation.mark %named_value_28 {logical_block_num} : i32
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    hivm.hir.anchor {id = 1 : i64}
    scf.for %named_value_31 = %named_value_30 to %named_value_28 step %named_value_26  : i32 {
      hivm.hir.anchor {id = 2 : i64}
      hivm.hir.set_ctrl false at ctrl[60]
      hivm.hir.set_ctrl true at ctrl[48]
      %named_value_36 = arith.remsi %named_value_31, %named_value_7 : i32
      hivm.hir.anchor {id = 3 : i64}
      hivm.hir.anchor {id = 4 : i64}
      hivm.hir.anchor {id = 5 : i64}
      hivm.hir.anchor {id = 6 : i64}
      scf.for %named_value_38 = %named_value_36 to %named_value_11 step %named_value_26  : i32 {
        hivm.hir.anchor {id = 7 : i64}
        %named_value_37 = arith.divsi %named_value_38, %named_value_17 : i32
        %named_value_39 = arith.remsi %named_value_38, %named_value_17 : i32
        %named_value_40 = arith.divsi %named_value_37, %named_value_16 : i32
        %named_value_41 = arith.remsi %named_value_37, %named_value_16 : i32
        %named_value_42 = arith.extsi %named_value_40 : i32 to i64
        %named_value_43 = arith.muli %named_value_42, %named_value_15 : i64
        %named_value_44 = arith.extsi %named_value_41 : i32 to i64
        %named_value_45 = arith.muli %named_value_44, %named_value_14 : i64
        %named_value_46 = arith.addi %named_value_43, %named_value_45 : i64
        %named_value_47 = arith.index_cast %named_value_46 : i64 to index
        %named_value_48 = arith.muli %named_value_39, %named_value_13 : i32
        %named_value_49 = arith.maxsi %named_value_48, %named_value_18 : i32
        %named_value_50 = arith.index_cast %named_value_49 : i32 to index
        %named_value_51 = affine.apply #map()[%named_value_47, %named_value_50]
        %named_value_53 = memref.reinterpret_cast %named_value_6 to offset: [%named_value_51], sizes: [128, 64], strides: [64, 1] : memref<?xf16, #hivm.address_space<gm>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        %named_value_112 = memref.subview %named_value_53[%named_value_34, 0] [64, 64] [1, 1] {to_be_bubbled_slice} : memref<128x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> to memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.anchor {id = 8 : i64}
        hivm.hir.anchor {id = 9 : i64}
        hivm.hir.anchor {id = 10 : i64}
        %named_value_115 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        func.call @renamed_symbol_8(%named_value_115) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>) -> ()
        %named_value_116 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
        func.call @renamed_symbol_9(%named_value_116) {hivm.vector_function, no_inline} : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()
        %named_value_117 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        hivm.hir.copy ins(%named_value_55 : memref<64xf32, #hivm.address_space<ub>>) outs(%named_value_117 : memref<64xf32, #hivm.address_space<ub>>)
        %named_value_52:3 = scf.for %named_value_58 = %named_value_18 to %named_value_12 step %named_value_17 iter_args(%named_value_59 = %named_value_116, %named_value_60 = %named_value_115, %named_value_61 = %named_value_117) -> (memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>)  : i32 {
          hivm.hir.anchor {id = 11 : i64}
          %named_value_123 = memref.alloc() : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_123 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x64xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_123 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x64xf32, #hivm.address_space<ub>>
          %named_value_72 = memref.alloc() : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %named_value_124 = memref.subview %named_value_72[0, %named_value_35, 0, 0] [16, 4, 16, 16] [1, 1, 1, 1] {to_be_bubbled_slice} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>> to memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>>
          annotation.mark %named_value_72 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
          %named_value_125 = memref.alloc() : memref<64x256xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_125 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<0>, hivm.tiling_dim = 0 : index, tiledAlloc} : memref<64x256xf32, #hivm.address_space<ub>>
          annotation.mark %named_value_125 {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64x256xf32, #hivm.address_space<ub>>
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
          %named_value_102:3 = scope.scope : () -> (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) {
            hivm.hir.anchor {id = 22 : i64}
            hivm.hir.anchor {id = 23 : i64}
            hivm.hir.anchor {id = 24 : i64}
            %named_value_126 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            func.call @renamed_symbol_5(%named_value_125, %named_value_126, %named_value_125) {hivm.vector_function, no_inline} : (memref<64x256xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x256xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.anchor {id = 25 : i64}
            hivm.hir.anchor {id = 26 : i64}
            hivm.hir.anchor {id = 27 : i64}
            hivm.hir.anchor {id = 28 : i64}
            hivm.hir.anchor {id = 29 : i64}
            hivm.hir.anchor {id = 30 : i64}
            %named_value_127 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            func.call @renamed_symbol_6(%named_value_61, %named_value_126, %named_value_127) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.anchor {id = 31 : i64}
            hivm.hir.anchor {id = 32 : i64}
            hivm.hir.anchor {id = 33 : i64}
            hivm.hir.anchor {id = 34 : i64}
            hivm.hir.anchor {id = 35 : i64}
            hivm.hir.anchor {id = 36 : i64}
            %named_value_128 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            %named_value_129 = memref.alloc() : memref<16x65x16x1xf16, #hivm.address_space<ub>>
            %named_value_130 = memref.subview %named_value_129[0, 0, 0, 0] [16, 64, 16, 1] [1, 1, 1, 1] : memref<16x65x16x1xf16, #hivm.address_space<ub>> to memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>
            func.call @renamed_symbol_7(%named_value_127, %named_value_125, %named_value_128, %named_value_130) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x256xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>>) -> ()
            %named_value_131 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            %named_value_132 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            annotation.mark %named_value_132 {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<64xf32, #hivm.address_space<ub>>
            func.call @renamed_symbol_3(%named_value_60, %named_value_61, %named_value_127, %named_value_128, %named_value_131, %named_value_132) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>) -> ()
            %named_value_133 = memref.expand_shape %named_value_130 [[0], [1, 2], [3]] output_shape [16, 4, 16, 16] : memref<16x64x16xf16, strided<[1040, 16, 1]>, #hivm.address_space<ub>> into memref<16x4x16x16xf16, strided<[1040, 256, 16, 1]>, #hivm.address_space<ub>>
            annotation.mark %named_value_72 {effects = ["write", "read"], hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<2>, hivm.tiling_dim = 1 : index} : memref<16x8x16x16xf16, #hivm.address_space<cbuf>>
            hivm.hir.anchor {id = 37 : i64}
            %named_value_134 = memref.collapse_shape %named_value_133 [[0], [1, 2, 3]] : memref<16x4x16x16xf16, strided<[1040, 256, 16, 1]>, #hivm.address_space<ub>> into memref<16x1024xf16, strided<[1040, 1]>, #hivm.address_space<ub>>
            %named_value_135 = memref.collapse_shape %named_value_124 [[0], [1, 2, 3]] : memref<16x4x16x16xf16, strided<[2048, 256, 16, 1], offset: ?>, #hivm.address_space<cbuf>> into memref<16x1024xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>
            hivm.hir.copy ins(%named_value_134 : memref<16x1024xf16, strided<[1040, 1]>, #hivm.address_space<ub>>) outs(%named_value_135 : memref<16x1024xf16, strided<[2048, 1], offset: ?>, #hivm.address_space<cbuf>>) {"hivm.inserted-copy", tiled_op}
            hivm.hir.anchor {id = 38 : i64}
            %named_value_136 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
            hivm.hir.copy ins(%named_value_127 : memref<64xf32, #hivm.address_space<ub>>) outs(%named_value_136 : memref<64xf32, #hivm.address_space<ub>>)
            scope.return %named_value_132, %named_value_136, %named_value_131 : memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
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
          %named_value_103 = scope.scope : () -> memref<64x64xf32, #hivm.address_space<ub>> {
            hivm.hir.anchor {id = 50 : i64}
            hivm.hir.anchor {id = 51 : i64}
            hivm.hir.anchor {id = 52 : i64}
            %named_value_126 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf32, #hivm.address_space<ub>>
            func.call @renamed_symbol_4(%named_value_102#0, %named_value_123, %named_value_59, %named_value_126) {hivm.vector_function, no_inline} : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
            hivm.hir.anchor {id = 53 : i64}
            scope.return %named_value_126 : memref<64x64xf32, #hivm.address_space<ub>>
          } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.max_preload_num = 4 : i32, hivm.preload_num = 0 : i32, no_inline}
          hivm.hir.anchor {id = 54 : i64}
          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 0
          scf.yield %named_value_103, %named_value_102#2, %named_value_102#1 : memref<64x64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>
        } {fixpipe_for_mmad_result_already_inserted = true}
        hivm.hir.anchor {id = 55 : i64}
        hivm.hir.anchor {id = 56 : i64}
        %named_value_137 = memref.alloc() {alignment = 64 : i64} : memref<64xf32, #hivm.address_space<ub>>
        annotation.mark %named_value_137 {hivm.multi_buffer = 2 : i32} : memref<64xf32, #hivm.address_space<ub>>
        %named_value_56 = arith.muli %named_value_37, %named_value_12 : i32
        %named_value_57 = arith.index_cast %named_value_56 : i32 to index
        %named_value_99 = arith.index_cast %named_value_48 : i32 to index
        %named_value_100 = affine.apply #map1()[%named_value_57, %named_value_99]
        %named_value_107 = memref.reinterpret_cast %named_value_5 to offset: [%named_value_100], sizes: [128], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %named_value_138 = memref.subview %named_value_107[%named_value_34] [64] [1] {to_be_bubbled_slice} : memref<128xf32, strided<[1], offset: ?>, #hivm.address_space<gm>> to memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
        %named_value_65 = memref.alloc() {alignment = 64 : i64} : memref<64x64xf16, #hivm.address_space<ub>>
        annotation.mark %named_value_65 {hivm.multi_buffer = 2 : i32} : memref<64x64xf16, #hivm.address_space<ub>>
        func.call @renamed_symbol_1(%named_value_52#2, %named_value_52#1, %named_value_137, %named_value_52#1, %named_value_52#0, %named_value_65) {hivm.vector_function, no_inline, ptc_simdvf} : (memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf16, #hivm.address_space<ub>>) -> ()
        hivm.hir.anchor {id = 57 : i64}
        hivm.hir.anchor {id = 58 : i64}
        hivm.hir.anchor {id = 59 : i64}
        hivm.hir.store ins(%named_value_137 : memref<64xf32, #hivm.address_space<ub>>) outs(%named_value_138 : memref<64xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        hivm.hir.anchor {id = 60 : i64}
        %named_value_139 = memref.collapse_shape %named_value_65 [[0, 1]] : memref<64x64xf16, #hivm.address_space<ub>> into memref<4096xf16, #hivm.address_space<ub>>
        %named_value_140 = memref.collapse_shape %named_value_112 [[0, 1]] : memref<64x64xf16, strided<[64, 1], offset: ?>, #hivm.address_space<gm>> into memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>
        hivm.hir.store ins(%named_value_139 : memref<4096xf16, #hivm.address_space<ub>>) outs(%named_value_140 : memref<4096xf16, strided<[1], offset: ?>, #hivm.address_space<gm>>) {tiled_op}
        hivm.hir.anchor {id = 61 : i64}
      }
      hivm.hir.set_ctrl true at ctrl[60]
      hivm.hir.anchor {id = 62 : i64}
    } {autoblockify.subloop}
    hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0
    hivm.hir.anchor {id = 63 : i64}
    return
  }
}



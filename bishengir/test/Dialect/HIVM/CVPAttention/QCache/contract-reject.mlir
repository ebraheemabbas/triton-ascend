// RUN: bishengir-opt %s --split-input-file --verify-diagnostics --mlir-print-op-on-diagnostic=false
// missing_q_context
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{unsupported QK layout, geometry or cache context}}
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// context_without_descriptor
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{expected version, mode and role}}
    hivm.hir.mmadL1 {b_transpose} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// unsupported_role
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{unsupported version, mode or role}}
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "other", version = 1 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// unsupported_version
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{unsupported version, mode or role}}
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 2 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// wrong_transpose
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{unsupported QK layout, geometry or cache context}}
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// wrong_matrix_size
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    // expected-error@+1 {{unsupported QK layout, geometry or cache context}}
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}} ins(%a, %b, %true, %n, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// out_of_reservation
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %outside = arith.constant 65536 : i64
    %offset = arith.select %is_odd, %outside, %even : i64
    // expected-error@+1 {{cache byte offset is outside the assigned query slots}}
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>)
    return
  }
}

// -----
// pv_cannot_own_q_context
module attributes {dlti.target_system_spec = #dlti.target_system_spec<"NPU" : #hacc.target_device_spec<#dlti.dl_entry<"AI_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"CUBE_CORE_COUNT", 28 : i32>, #dlti.dl_entry<"VECTOR_CORE_COUNT", 56 : i32>, #dlti.dl_entry<"UB_SIZE", 2031616 : i32>, #dlti.dl_entry<"L1_SIZE", 4194304 : i32>, #dlti.dl_entry<"L0A_SIZE", 524288 : i32>, #dlti.dl_entry<"L0B_SIZE", 524288 : i32>, #dlti.dl_entry<"L0C_SIZE", 2097152 : i32>, #dlti.dl_entry<"UB_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L1_ALIGN_SIZE", 256 : i32>, #dlti.dl_entry<"L0C_ALIGN_SIZE", 4096 : i32>, #dlti.dl_entry<"MINIMAL_D_CACHE_SIZE", 262144 : i32>, #dlti.dl_entry<"MAXIMUM_D_CACHE_SIZE", 983040 : i32>, #dlti.dl_entry<"ARCH", "dav-c310">>>, hacc.target = #hacc.target<"Ascend950PR_9579">, hivm.cvp_attention.p32_materializations_removed = 1 : i64, hivm.cvp_attention.query_max_scalings = 1 : i64, hivm.cvp_attention.query_score_materializations_removed = 1 : i64, hivm.cvp_attention.sum_materializations_removed = 1 : i64, hivm.cvp_attention.tile_max_materializations_removed = 1 : i64, hivm.module_core_type = #hivm.module_core_type<MIX>} {
  func.func @cached_pair(%a: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %b: memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, %c: memref<8x16x16x16xf32, #hivm.address_space<cc>>, %p: memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, %v: memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, %pv: memref<4x8x16x16xf32, #hivm.address_space<cc>>, %first: i1, %generation: i64) attributes {hivm.func_core_type = #hivm.func_core_type<AIC>, hivm.cvp_q_cache.resources = {a_ranges = array<i64: 0, 32768, 32768, 65536>, bm = 128 : i64, bn = 256 : i64, counter_banks = 2 : i64, element_bytes = 2 : i64, hd = 64 : i64, l0a_bytes = 65536 : i64, l0b_bytes = 65536 : i64, mode = "cached", pv_b_ranges = array<i64: 0, 16384, 32768, 49152>, pv_chunk_k = 128 : i64, pv_chunks = 2 : i64, q_clock = "query", q_ranges = array<i64: 16384, 32768, 49152, 65536>, q_slots = 2 : i64, qk_chunk_k = 64 : i64, target = "dav-c310", unit_flag = 0 : i64, version = 1 : i64}} {
    %true = arith.constant true
    %m = arith.constant 256 : index
    %k = arith.constant 64 : index
    %n = arith.constant 128 : index
    %two = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %even = arith.constant 16384 : i64
    %odd = arith.constant 49152 : i64
    %slot = arith.remsi %generation, %two : i64
    %is_odd = arith.cmpi eq, %slot, %one : i64
    %offset = arith.select %is_odd, %odd, %even : i64
    hivm.hir.mmadL1 {b_transpose, cvp_q_cache = {mode = "cached", role = "qk", version = 1 : i64}} ins(%a, %b, %true, %m, %k, %n : memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x8x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%c : memref<8x16x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    // expected-error@+1 {{unsupported PV layout, geometry or cache context}}
    hivm.hir.mmadL1 {a_transpose, cvp_q_cache = {mode = "cached", role = "pv", version = 1 : i64}} ins(%p, %v, %true, %n, %m, %k : memref<8x16x16x16xf16, #hivm.address_space<cbuf>>, memref<4x16x16x16xf16, #hivm.address_space<cbuf>>, i1, index, index, index) outs(%pv : memref<4x8x16x16xf32, #hivm.address_space<cc>>) q_cache_context(%first, %generation, %offset)
    return
  }
}

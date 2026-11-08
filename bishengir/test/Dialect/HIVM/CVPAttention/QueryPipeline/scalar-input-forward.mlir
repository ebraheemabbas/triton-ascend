// RUN: %python %S/Inputs/check-scalar-input-forward.py %s bishengir-opt

#ub = #hivm.address_space<ub>
module {
  func.func private @opaque(memref<4xf32, #ub>)
  func.func private @effect()
  func.func @produce(%score: memref<4xf32, #ub>, %p: memref<4xf32, #ub>, %old_l: memref<4xf32, #ub>, %old_m: memref<4xf32, #ub>, %new_l: memref<4xf32, #ub>, %alpha: memref<4xf32, #ub>, %new_m: memref<4xf32, #ub>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %zero = arith.constant 0.0 : f32
    %l = vector.transfer_read %old_l[%c0], %zero {in_bounds = [true]} : memref<4xf32, #ub>, vector<4xf32>
    %m = vector.transfer_read %old_m[%c0], %zero {in_bounds = [true]} : memref<4xf32, #ub>, vector<4xf32>
    %s = vector.transfer_read %score[%c0], %zero {in_bounds = [true]} : memref<4xf32, #ub>, vector<4xf32>
    %sum = arith.addf %l, %s : vector<4xf32>
    %maximum = arith.maximumf %m, %s : vector<4xf32>
    vector.transfer_write %sum, %new_l[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #ub>
    vector.transfer_write %maximum, %new_m[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #ub>
    vector.transfer_write %s, %alpha[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #ub>
    vector.transfer_write %s, %p[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #ub>
    return
  }
  func.func @case(%first: i1, %slot: i1) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %ones = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
    %minusinf = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
    %l0 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "sum"}} : memref<4xf32, #ub>
    annotation.mark %l0 {mem_unique} : memref<4xf32, #ub>
    %l1 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "sum"}} : memref<4xf32, #ub>
    annotation.mark %l1 {mem_unique} : memref<4xf32, #ub>
    %m0 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "query", required_slots = 2 : i32, role = "maximum"}} : memref<4xf32, #ub>
    annotation.mark %m0 {mem_unique} : memref<4xf32, #ub>
    %m1 = memref.alloc() {hivm.cvp_query.state_bank = {bank = 1 : i32, clock = "query", required_slots = 2 : i32, role = "maximum"}} : memref<4xf32, #ub>
    annotation.mark %m1 {mem_unique} : memref<4xf32, #ub>
    hivm.hir.copy ins(%ones : memref<4xf32, #ub>) outs(%l0 : memref<4xf32, #ub>)
    hivm.hir.copy ins(%ones : memref<4xf32, #ub>) outs(%l1 : memref<4xf32, #ub>)
    hivm.hir.copy ins(%minusinf : memref<4xf32, #ub>) outs(%m0 : memref<4xf32, #ub>)
    hivm.hir.copy ins(%minusinf : memref<4xf32, #ub>) outs(%m1 : memref<4xf32, #ub>)
    scf.for %key = %c0 to %c2 step %c1 {
      %score = memref.alloc() : memref<4xf32, #ub>
      hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1
      %returned_alpha = scope.scope : () -> memref<4xf32, #ub> {
        %selected_l = arith.select %slot, %l1, %l0 : memref<4xf32, #ub>
        %selected_m = arith.select %slot, %m1, %m0 : memref<4xf32, #ub>
        %la = memref.alloc() : memref<4xf32, #ub>
        hivm.hir.copy ins(%selected_l : memref<4xf32, #ub>) outs(%la : memref<4xf32, #ub>)
        %ma = memref.alloc() : memref<4xf32, #ub>
        hivm.hir.copy ins(%selected_m : memref<4xf32, #ub>) outs(%ma : memref<4xf32, #ub>)
        hivm.hir.anchor {id = 22 : i64}
        %lb = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        hivm.hir.copy ins(%la : memref<4xf32, #ub>) outs(%lb : memref<4xf32, #ub>)
        %lc = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        hivm.hir.copy ins(%lb : memref<4xf32, #ub>) outs(%lc : memref<4xf32, #ub>)
        %old_l = arith.select %first, %ones, %lc : memref<4xf32, #ub>
        %mb = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        hivm.hir.copy ins(%ma : memref<4xf32, #ub>) outs(%mb : memref<4xf32, #ub>)
        %mc = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        hivm.hir.copy ins(%mb : memref<4xf32, #ub>) outs(%mc : memref<4xf32, #ub>)
        %old_m = arith.select %first, %minusinf, %mc : memref<4xf32, #ub>
        scf.if %first {
          hivm.hir.anchor {id = 24 : i64}
        } else {
          hivm.hir.anchor {id = 25 : i64}
        }
        %packed = memref.alloc() : memref<4xf32, #ub>
        %new_l = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        %alpha = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        annotation.mark %alpha {hivm.multi_buffer = 3 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4xf32, #ub>
        %new_m = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>
        func.call @produce(%score, %packed, %old_l, %old_m, %new_l, %alpha, %new_m) {hivm.vector_function} : (memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>) -> ()
        hivm.hir.copy ins(%new_l : memref<4xf32, #ub>) outs(%selected_l : memref<4xf32, #ub>)
        hivm.hir.copy ins(%new_m : memref<4xf32, #ub>) outs(%selected_m : memref<4xf32, #ub>)
        scope.return %alpha : memref<4xf32, #ub>
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32, hivm.max_preload_num = 4 : i32}
      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1
    } {hivm.cvp_query.stream = {pipeline_mode = "skew", key_unroll = 1 : i32, query_slots = 2 : i32}, hivm.cvp_query.state_storage}
    return
  }
}

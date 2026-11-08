// RUN: %python %S/Inputs/check-output-copy-forward.py %s bishengir-opt

#ub = #hivm.address_space<ub>
#broadcast = affine_map<(d0) -> (d0, 0)>
module {
  func.func private @opaque(memref<4x64xf32, #ub>)
  func.func private @effect()
  func.func @update(%alpha: memref<4xf32, #ub>, %pv: memref<4x64xf32, #ub>, %old: memref<4x64xf32, #ub>, %out: memref<4x64xf32, #ub>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %c4 step %c1 {
      %arow = memref.subview %alpha[%row] [1] [1] : memref<4xf32, #ub> to memref<1xf32, strided<[1], offset: ?>, #ub>
      %prow = memref.subview %pv[%row, 0] [1, 64] [1, 1] : memref<4x64xf32, #ub> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
      %orow = memref.subview %old[%row, 0] [1, 64] [1, 1] : memref<4x64xf32, #ub> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
      %drow = memref.subview %out[%row, 0] [1, 64] [1, 1] : memref<4x64xf32, #ub> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
      %p = vector.transfer_read %prow[%c0, %c0], %zero {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf32>
      %o = vector.transfer_read %orow[%c0, %c0], %zero {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf32>
      %a = vector.transfer_read %arow[%c0], %zero {in_bounds = [true, true], permutation_map = #broadcast} : memref<1xf32, strided<[1], offset: ?>, #ub>, vector<1x64xf32>
      %scaled = arith.mulf %o, %a : vector<1x64xf32>
      %result = arith.addf %p, %scaled : vector<1x64xf32>
      vector.transfer_write %result, %drow[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
    }
    return
  }
  func.func @finish(%m: memref<4xf32, #ub>, %l: memref<4xf32, #ub>, %stats: memref<4xf32, #ub>, %denominator: memref<4xf32, #ub>, %updated: memref<4x64xf32, #ub>, %out: memref<4x64xf16, #ub>) attributes {hivm.vector_function} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant 0.0 : f32
    scf.for %row = %c0 to %c4 step %c1 {
      %inrow = memref.subview %updated[%row, 0] [1, 64] [1, 1] : memref<4x64xf32, #ub> to memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>
      %outrow = memref.subview %out[%row, 0] [1, 64] [1, 1] : memref<4x64xf16, #ub> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ub>
      %value = vector.transfer_read %inrow[%c0, %c0], %zero {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf32>
      %half = arith.truncf %value : vector<1x64xf32> to vector<1x64xf16>
      vector.transfer_write %half, %outrow[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ub>
    }
    return
  }
  func.func @case(%first: i1, %last: i1) attributes {hacc.entry, hivm.func_core_type = #hivm.func_core_type<AIV>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alpha = memref.alloc() : memref<4xf32, #ub>
    %zero = memref.alloc() : memref<4x64xf32, #ub>
    %bank = memref.alloc() {hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "O", required_slots = 1 : i32, role = "accumulator"}} : memref<4x64xf32, #ub>
    annotation.mark %bank {mem_unique} : memref<4x64xf32, #ub>
    %scalar = memref.alloc() : memref<4xf32, #ub>
    %out = memref.alloc() : memref<4x64xf16, #ub>
    scf.for %key = %c0 to %c2 step %c1 {
      %pv = memref.alloc() : memref<4x64xf32, #ub>
      annotation.mark %pv {hivm.multi_buffer = 2 : i32, hivm.preload_local_buffer = 1 : i32} : memref<4x64xf32, #ub>
      scope.scope : () -> () {
        %old = arith.select %first, %zero, %bank : memref<4x64xf32, #ub>
        func.call @update(%alpha, %pv, %old, %pv) {hivm.vector_function} : (memref<4xf32, #ub>, memref<4x64xf32, #ub>, memref<4x64xf32, #ub>, memref<4x64xf32, #ub>) -> ()
        hivm.hir.copy ins(%pv : memref<4x64xf32, #ub>) outs(%bank : memref<4x64xf32, #ub>)
        hivm.hir.anchor {id = 62 : i64}
        scf.if %last {
          func.call @finish(%scalar, %scalar, %scalar, %scalar, %pv, %out) {hivm.vector_function} : (memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4xf32, #ub>, memref<4x64xf32, #ub>, memref<4x64xf16, #ub>) -> ()
        }
        scope.return
      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 4 : i32}
    } {hivm.cvp_query.stream = {key_count = 2 : i64, key_step = 1 : i64, query_step = 1 : i64, pipeline_mode = "skew", key_unroll = 1 : i32}, hivm.cvp_query.state_storage, hivm.cvp_query.output_storage}
    return
  }
}

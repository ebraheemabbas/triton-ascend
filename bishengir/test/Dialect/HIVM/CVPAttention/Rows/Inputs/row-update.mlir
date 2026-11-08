module {
  func.func @caller(%alpha: memref<64xf32, #hivm.address_space<ub>>, %pv: memref<64x64xf32, #hivm.address_space<ub>>, %old: memref<64x64xf32, #hivm.address_space<ub>>, %out: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.part_of_mix, hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 128 : i64, head_dim = 64 : i64}]} {
    func.call @update(%alpha, %pv, %old, %out) : (memref<64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>, memref<64x64xf32, #hivm.address_space<ub>>) -> ()
    return
  }
  func.func @update(%alpha: memref<64xf32, #hivm.address_space<ub>>, %pv: memref<64x64xf32, #hivm.address_space<ub>>, %old: memref<64x64xf32, #hivm.address_space<ub>>, %out: memref<64x64xf32, #hivm.address_space<ub>>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %pad = arith.constant 0.0 : f32
    scf.for %row = %c0 to %c64 step %c1 {
      %a = memref.subview %alpha[%row] [1] [1] : memref<64xf32, #hivm.address_space<ub>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>
      %p = memref.subview %pv[%row, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %o = memref.subview %old[%row, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %d = memref.subview %out[%row, 0] [1, 64] [1, 1] : memref<64x64xf32, #hivm.address_space<ub>> to memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
      %av = vector.transfer_read %a[%c0], %pad {in_bounds = [true, true], permutation_map = affine_map<(d0) -> (d0, 0)>} : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %pv_value = vector.transfer_read %p[%c0, %c0], %pad {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %old_value = vector.transfer_read %o[%c0, %c0], %pad {in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>, vector<1x64xf32>
      %scaled = arith.mulf %old_value, %av : vector<1x64xf32>
      %updated = arith.addf %pv_value, %scaled : vector<1x64xf32>
      vector.transfer_write %updated, %d[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #hivm.address_space<ub>>
    }
    return
  }
}

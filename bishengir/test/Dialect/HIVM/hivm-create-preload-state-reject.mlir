// RUN: bishengir-opt %s --create-preload --split-input-file --verify-diagnostics

func.func @overwritten_subview(%initial: memref<64xf32>, %produced: memref<64xf32>,
                               %target: memref<64xf32>, %end: i32, %choose: i1) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %idx = arith.constant 0 : index
  %value = arith.constant 0.0 : f32
  %part = memref.subview %target[0] [8] [1] : memref<64xf32> to memref<8xf32, strided<[1]>>
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<64xf32>) : i32 {
    // expected-error@+1 {{cannot preserve a preload scope result in the inactive stage}}
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    memref.copy %selected, %target : memref<64xf32> to memref<64xf32>
    memref.store %value, %part[%idx] : memref<8xf32, strided<[1]>>
    scf.yield %target : memref<64xf32>
  }
  return
}

// -----

func.func private @unknown_write(memref<64xf32>)
func.func @unknown_mutation(%initial: memref<64xf32>, %produced: memref<64xf32>,
                            %target: memref<64xf32>, %end: i32, %choose: i1) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<64xf32>) : i32 {
    // expected-error@+1 {{cannot preserve a preload scope result in the inactive stage}}
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    memref.copy %selected, %target : memref<64xf32> to memref<64xf32>
    func.call @unknown_write(%target) : (memref<64xf32>) -> ()
    scf.yield %target : memref<64xf32>
  }
  return
}

// -----

func.func @wrong_carried_slot(%initial: memref<64xf32>, %produced: memref<64xf32>,
                              %target: memref<64xf32>, %end: i32, %choose: i1) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result:2 = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial, %other = %produced)
      -> (memref<64xf32>, memref<64xf32>) : i32 {
    // expected-error@+1 {{cannot preserve a preload scope result in the inactive stage}}
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    memref.copy %selected, %target : memref<64xf32> to memref<64xf32>
    scf.yield %previous, %target : memref<64xf32>, memref<64xf32>
  }
  return
}

// -----

func.func @partial_copy(%initial: memref<64xf32>, %produced: memref<64xf32>,
                        %target: memref<64xf32>, %end: i32, %choose: i1) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %target_part = memref.subview %target[0] [8] [1] : memref<64xf32> to memref<8xf32, strided<[1]>>
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<64xf32>) : i32 {
    // expected-error@+1 {{cannot preserve a preload scope result in the inactive stage}}
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    %part = memref.subview %selected[0] [8] [1] : memref<64xf32> to memref<8xf32, strided<[1]>>
    memref.copy %part, %target_part : memref<8xf32, strided<[1]>> to memref<8xf32, strided<[1]>>
    scf.yield %target : memref<64xf32>
  }
  return
}

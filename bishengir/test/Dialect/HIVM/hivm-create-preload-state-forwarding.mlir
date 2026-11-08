// RUN: bishengir-opt %s --create-preload --split-input-file --verify-each | FileCheck %s

// A selected state returned through a copy still needs its previous loop value
// in the inactive stage. Selecting the new producer buffer would read data from
// a stage that did not execute during fill/drain.
// CHECK-LABEL: func.func @copied_state
// CHECK: scf.for {{.*}} iter_args(%[[PREV:.*]] = {{.*}}) -> (memref<64xf32>)
// CHECK: %[[STATE:.*]] = scf.if {{.*}} -> (memref<64xf32>) {
// CHECK: arith.select
// CHECK: scf.yield
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PREV]] : memref<64xf32>
// CHECK: memref.copy %[[STATE]]
func.func @copied_state(%initial: memref<64xf32>, %produced: memref<64xf32>,
                        %target: memref<64xf32>, %end: i32, %choose: i1)
    -> memref<64xf32> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<64xf32>) : i32 {
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 1 : i32, hivm.max_preload_num = 2 : i32}
    memref.copy %selected, %target : memref<64xf32> to memref<64xf32>
    scf.yield %target : memref<64xf32>
  }
  return %result : memref<64xf32>
}

// -----

// Follow two full copies and rank-only views, including a copy into a view
// whose base is the actual loop yield. The preserved arm may be the false arm.
// CHECK-LABEL: func.func @reshaped_state
// CHECK: scf.for {{.*}} iter_args(%[[PREV:.*]] = {{.*}}) -> (memref<8x8xf32>)
// CHECK: %[[STATE:.*]] = scf.if {{.*}} -> (memref<8x8xf32>) {
// CHECK: arith.select
// CHECK: scf.yield
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PREV]] : memref<8x8xf32>
// CHECK: memref.collapse_shape %[[STATE]]
// CHECK: memref.copy
func.func @reshaped_state(%initial: memref<8x8xf32>, %produced: memref<8x8xf32>,
                          %scratch: memref<64xf32>, %target: memref<8x8xf32>,
                          %end: i64, %choose: i1) -> memref<8x8xf32> {
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %flat_target = memref.collapse_shape %target [[0, 1]] : memref<8x8xf32> into memref<64xf32>
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<8x8xf32>) : i64 {
    %selected = scope.scope : () -> memref<8x8xf32> {
      %state = arith.select %choose, %produced, %previous : memref<8x8xf32>
      scope.return %state : memref<8x8xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    %flat = memref.collapse_shape %selected [[0, 1]] : memref<8x8xf32> into memref<64xf32>
    memref.copy %flat, %scratch : memref<64xf32> to memref<64xf32>
    memref.copy %scratch, %flat_target : memref<64xf32> to memref<64xf32>
    scf.yield %target : memref<8x8xf32>
  }
  return %result : memref<8x8xf32>
}

// -----

// Directly carried state can also have another read use. It still preserves
// the matching loop argument in the inactive branch.
// CHECK-LABEL: func.func @direct_state_with_reader
// CHECK: scf.for {{.*}} iter_args(%[[PREV:.*]] = {{.*}}) -> (memref<64xf32>)
// CHECK: scf.if {{.*}} -> (memref<64xf32>) {
// CHECK: } else {
// CHECK-NEXT: scf.yield %[[PREV]] : memref<64xf32>
func.func @direct_state_with_reader(%initial: memref<64xf32>,
                                    %produced: memref<64xf32>,
                                    %sink: memref<64xf32>, %end: i32,
                                    %choose: i1) -> memref<64xf32> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %result = scf.for %i = %zero to %end step %one
      iter_args(%previous = %initial) -> (memref<64xf32>) : i32 {
    %selected = scope.scope : () -> memref<64xf32> {
      %state = arith.select %choose, %previous, %produced : memref<64xf32>
      scope.return %state : memref<64xf32>
    } {no_inline, hivm.preload_num = 0 : i32, hivm.max_preload_num = 2 : i32}
    memref.copy %selected, %sink : memref<64xf32> to memref<64xf32>
    scf.yield %selected : memref<64xf32>
  }
  return %result : memref<64xf32>
}

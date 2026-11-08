// RUN: bishengir-opt %s --one-shot-bufferize --verify-each | FileCheck %s
// Q is read through a restricted tensor boundary; ND2NZ has pure tensor
// operands before bufferization and the original memref source afterward.
// CHECK-LABEL: func.func @q_source_hd64(
// CHECK-SAME: %[[SOURCE64:[^:]+]]: memref<128x64xf16>
// CHECK: hivm.hir.nd2nz
// CHECK-SAME: ins(%[[SOURCE64]] : memref<128x64xf16>)
// CHECK: memref.copy
func.func @q_source_hd64(%source: memref<128x64xf16>, %output: memref<4x8x16x16xf16>) {
  %view = bufferization.to_tensor %source restrict : memref<128x64xf16>
  %slot = tensor.empty() : tensor<4x8x16x16xf16>
  %loaded = hivm.hir.nd2nz ins(%view : tensor<128x64xf16>) outs(%slot : tensor<4x8x16x16xf16>) -> tensor<4x8x16x16xf16>
  %buffer = bufferization.to_memref %loaded : memref<4x8x16x16xf16>
  memref.copy %buffer, %output : memref<4x8x16x16xf16> to memref<4x8x16x16xf16>
  return
}

// CHECK-LABEL: func.func @q_source_hd128(
// CHECK-SAME: %[[SOURCE128:[^:]+]]: memref<128x128xf16>
// CHECK: hivm.hir.nd2nz
// CHECK-SAME: ins(%[[SOURCE128]] : memref<128x128xf16>)
// CHECK: memref.copy
func.func @q_source_hd128(%source: memref<128x128xf16>, %output: memref<8x8x16x16xf16>) {
  %view = bufferization.to_tensor %source restrict : memref<128x128xf16>
  %slot = tensor.empty() : tensor<8x8x16x16xf16>
  %loaded = hivm.hir.nd2nz ins(%view : tensor<128x128xf16>) outs(%slot : tensor<8x8x16x16xf16>) -> tensor<8x8x16x16xf16>
  %buffer = bufferization.to_memref %loaded : memref<8x8x16x16xf16>
  memref.copy %buffer, %output : memref<8x8x16x16xf16> to memref<8x8x16x16xf16>
  return
}


// The original BN256/HD128 CVPipeline input fails its first allocation attempt.
// A one-attempt diagnostic must preserve that capacity error rather than clear
// it while preparing a fallback that will never run. No NPU execution is needed.
// RUN: not bishengir-compile %s --target=Ascend950PR_9579 --enable-auto-multi-buffer=true --enable-preload=true --enable-auto-bind-sub-block=true --disable-ffts --enable-hivm-graph-sync-solver=true --limit-auto-multi-buffer-of-local-buffer=no-limit --enable-mixed-cv=true --enable-auto-blockify-loop --enable-vf-operand-substitution=true --enable-fused-multiply-add=false --enable-cvp-attention-analysis=true --enable-hfusion-compile=true --enable-triton-kernel-compile=true --enable-vf-merge-level=1 --enable-tuning-mode=true -o %t 2>&1 | FileCheck %s
// CHECK: {{(ub|cc|cbuf)}} overflow, requires {{[0-9]+}} bits while {{[0-9]+}} bits available!

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @_attn_fwd(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg3: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg4: memref<?xf16> {tt.tensor_kind = 0 : i32}, %arg5: memref<?xf32> {tt.tensor_kind = 1 : i32}, %arg6: memref<?xf16> {tt.tensor_kind = 1 : i32}, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant 5.000000e-01 : f32
    %c128 = arith.constant 128 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1048576_i64 = arith.constant 1048576 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8192_i32 = arith.constant 8192 : i32
    %c28_i32 = arith.constant 28 : i32
    %cst_2 = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<128x128xf32>
    %1 = linalg.fill ins(%cst_2 : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    %2 = tensor.empty() : tensor<128x256xf32>
    %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %4 = linalg.fill ins(%cst_2 : f32) outs(%2 : tensor<128x256xf32>) -> tensor<128x256xf32>
    %5 = tensor.empty() : tensor<128xf32>
    %6 = linalg.fill ins(%cst_0 : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
    scf.for %arg13 = %arg10 to %c8192_i32 step %c28_i32  : i32 {
      %8 = arith.divsi %arg13, %c8_i32 : i32
      %9 = arith.remsi %arg13, %c8_i32 : i32
      %10 = arith.divsi %8, %c8_i32 : i32
      %11 = arith.remsi %8, %c8_i32 : i32
      %12 = arith.extsi %10 : i32 to i64
      %13 = arith.muli %12, %c1048576_i64 : i64
      %14 = arith.extsi %11 : i32 to i64
      %15 = arith.muli %14, %c131072_i64 : i64
      %16 = arith.addi %13, %15 : i64
      %17 = arith.index_cast %16 : i64 to index
      %18 = arith.muli %9, %c128_i32 : i32
      %19 = arith.maxsi %18, %c0_i32 : i32
      %20 = arith.index_cast %19 : i32 to index
      %21 = arith.muli %20, %c128 : index
      %22 = arith.addi %21, %17 : index
      %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%22], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %reinterpret_cast_3 = memref.reinterpret_cast %arg6 to offset: [%22], sizes: [128, 128], strides: [128, 1] : memref<?xf16> to memref<128x128xf16, strided<[128, 1], offset: ?>>
      %alloc = memref.alloc() : memref<128x128xf16>
      memref.copy %reinterpret_cast, %alloc : memref<128x128xf16, strided<[128, 1], offset: ?>> to memref<128x128xf16>
      %23 = bufferization.to_tensor %alloc restrict writable : memref<128x128xf16>
      %24:5 = scf.for %arg14 = %c0_i32 to %c1024_i32 step %c256_i32 iter_args(%arg15 = %1, %arg16 = %7, %arg17 = %6, %arg18 = %c0_i32, %arg19 = %c0_i32) -> (tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32)  : i32 {
        %33 = arith.maxsi %arg18, %c0_i32 : i32
        %34 = arith.index_cast %33 : i32 to index
        %35 = arith.muli %34, %c128 : index
        %36 = arith.addi %35, %17 : index
        %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%36], sizes: [256, 128], strides: [128, 1] : memref<?xf16> to memref<256x128xf16, strided<[128, 1], offset: ?>>
        %37 = arith.maxsi %arg19, %c0_i32 : i32
        %38 = arith.index_cast %37 : i32 to index
        %39 = arith.muli %38, %c128 : index
        %40 = arith.addi %39, %17 : index
        %reinterpret_cast_6 = memref.reinterpret_cast %arg4 to offset: [%40], sizes: [256, 128], strides: [128, 1] : memref<?xf16> to memref<256x128xf16, strided<[128, 1], offset: ?>>
        %alloc_7 = memref.alloc() : memref<256x128xf16>
        memref.copy %reinterpret_cast_5, %alloc_7 : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<256x128xf16>
        %41 = bufferization.to_tensor %alloc_7 restrict writable : memref<256x128xf16>
        %42 = tensor.empty() : tensor<128x256xf16>
        %transposed = linalg.transpose ins(%41 : tensor<256x128xf16>) outs(%42 : tensor<128x256xf16>) permutation = [1, 0] 
        %43 = linalg.matmul {input_precision = "ieee"} ins(%23, %transposed : tensor<128x128xf16>, tensor<128x256xf16>) outs(%4 : tensor<128x256xf32>) -> tensor<128x256xf32>
        %44 = arith.mulf %43, %3 : tensor<128x256xf32>
        %reduced = linalg.reduce ins(%44 : tensor<128x256xf32>) outs(%6 : tensor<128xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %59 = arith.maximumf %in, %init : f32
            linalg.yield %59 : f32
          }
        %45 = arith.maximumf %arg17, %reduced : tensor<128xf32>
        %broadcasted_8 = linalg.broadcast ins(%45 : tensor<128xf32>) outs(%2 : tensor<128x256xf32>) dimensions = [1] 
        %46 = arith.subf %44, %broadcasted_8 : tensor<128x256xf32>
        %47 = math.exp %46 : tensor<128x256xf32>
        %48 = arith.truncf %47 : tensor<128x256xf32> to tensor<128x256xf16>
        %alloc_9 = memref.alloc() : memref<256x128xf16>
        memref.copy %reinterpret_cast_6, %alloc_9 : memref<256x128xf16, strided<[128, 1], offset: ?>> to memref<256x128xf16>
        %49 = bufferization.to_tensor %alloc_9 restrict writable : memref<256x128xf16>
        %50 = linalg.fill ins(%cst_2 : f32) outs(%5 : tensor<128xf32>) -> tensor<128xf32>
        %reduced_10 = linalg.reduce ins(%47 : tensor<128x256xf32>) outs(%50 : tensor<128xf32>) dimensions = [1] 
          (%in: f32, %init: f32) {
            %59 = arith.addf %in, %init : f32
            linalg.yield %59 : f32
          }
        %51 = arith.subf %arg17, %45 : tensor<128xf32>
        %52 = math.exp %51 : tensor<128xf32>
        %53 = arith.mulf %arg16, %52 : tensor<128xf32>
        %54 = arith.addf %53, %reduced_10 : tensor<128xf32>
        %broadcasted_11 = linalg.broadcast ins(%52 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1] 
        %55 = arith.mulf %arg15, %broadcasted_11 : tensor<128x128xf32>
        %56 = linalg.matmul {input_precision = "ieee"} ins(%48, %49 : tensor<128x256xf16>, tensor<256x128xf16>) outs(%55 : tensor<128x128xf32>) -> tensor<128x128xf32>
        %57 = arith.addi %arg19, %c256_i32 : i32
        %58 = arith.addi %arg18, %c256_i32 : i32
        scf.yield %56, %54, %45, %58, %57 : tensor<128x128xf32>, tensor<128xf32>, tensor<128xf32>, i32, i32
      }
      %25 = math.log %24#1 : tensor<128xf32>
      %26 = arith.addf %24#2, %25 : tensor<128xf32>
      %broadcasted = linalg.broadcast ins(%24#1 : tensor<128xf32>) outs(%0 : tensor<128x128xf32>) dimensions = [1] 
      %27 = arith.divf %24#0, %broadcasted : tensor<128x128xf32>
      %28 = arith.muli %8, %c1024_i32 : i32
      %29 = arith.index_cast %28 : i32 to index
      %30 = arith.index_cast %18 : i32 to index
      %31 = arith.addi %29, %30 : index
      %reinterpret_cast_4 = memref.reinterpret_cast %arg5 to offset: [%31], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
      bufferization.materialize_in_destination %26 in writable %reinterpret_cast_4 : (tensor<128xf32>, memref<128xf32, strided<[1], offset: ?>>) -> ()
      %32 = arith.truncf %27 : tensor<128x128xf32> to tensor<128x128xf16>
      bufferization.materialize_in_destination %32 in writable %reinterpret_cast_3 : (tensor<128x128xf16>, memref<128x128xf16, strided<[128, 1], offset: ?>>) -> ()
    }
    return
  }
}


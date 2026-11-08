// RUN: bishengir-opt %S/../DistinctScratch/Inputs/actual-hd128.mlir -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true retain-provenance=true" -verify-each -o %t.once
// RUN: FileCheck %s < %t.once
// RUN: bishengir-opt %t.once -hivm-cvp-attention-boundary="enable-first-output=true enable-last-output=true enable-first-state=true retain-provenance=true" -verify-each -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: %python %S/Inputs/mutate.py renamed %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.renamed
// RUN: bishengir-opt %t.renamed -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s
// RUN: bishengir-opt %S/../DistinctScratch/Inputs/actual-hd128.mlir -hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true retain-provenance=true" -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true retain-provenance=true" -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each | FileCheck %s --check-prefix=COMBINED

// CHECK: hivm.cvp_attention.first_output_specializations = 1 : i64
// CHECK-SAME: hivm.cvp_attention.first_state_specializations = 1 : i64
// CHECK-SAME: hivm.cvp_attention.last_output_specializations = 1 : i64
// CHECK-LABEL: func.func @__cvp_attention_first_output(
// CHECK: arith.constant dense<0.000000e+00> : vector<1x64xf32>
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @__cvp_attention_last_output(
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: arith.divf
// CHECK: arith.truncf
// CHECK: vector.transfer_write
// CHECK-LABEL: func.func @__cvp_attention_first_maximum(
// CHECK: arith.constant dense<0xFF800000> : vector<64xf32>
// CHECK: arith.maximumf
// CHECK-LABEL: func.func @__cvp_attention_first_state(
// CHECK: arith.constant dense<0xFF800000> : vector<64xf32>
// CHECK: math.exp
// CHECK: arith.constant dense<1.000000e+00> : vector<64xf32>
// CHECK: arith.mulf
// CHECK: arith.addf

// COMBINED: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.first_output_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.first_state_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.last_output_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.max_scalings = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.row_unrolls =

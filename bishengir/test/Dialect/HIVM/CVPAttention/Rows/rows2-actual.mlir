// RUN: bishengir-opt %S/../DistinctScratch/Inputs/actual-hd128.mlir -hivm-cvp-attention-vector-schedule="require-match=true retain-provenance=true" -verify-each -o %t.once
// RUN: FileCheck %s --check-prefix=ROWS < %t.once
// RUN: bishengir-opt %t.once -hivm-cvp-attention-vector-schedule="require-match=true retain-provenance=true" -verify-each -o %t.twice
// RUN: diff %t.once %t.twice
// RUN: bishengir-opt %S/../DistinctScratch/Inputs/actual-hd128.mlir -hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true retain-provenance=true" -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each | FileCheck %s --check-prefix=COMBINED

// Three per-tile functions change: score maximum, P/sum, and O update.
// The normalization and zero-fill functions are outside the rows2 selection.
// ROWS: hivm.cvp_attention.row_unrolls = 3 : i64
// ROWS-COUNT-3: hivm.cvp_attention.rows_per_iteration = 2 : i64
// COMBINED: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.max_scalings = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.row_unrolls = 3 : i64
// COMBINED-SAME: hivm.cvp_attention.score_store_removals = 1 : i64
// COMBINED-COUNT-3: hivm.cvp_attention.rows_per_iteration = 2 : i64

// RUN: bishengir-opt %S/Inputs/actual-native-n16384-hd128.mlir -o %t.before
// RUN: bishengir-opt %t.before -hivm-cvp-attention-vector="require-match=true enable-triton-kernel-compile=true" -verify-each -o %t.score
// RUN: FileCheck %s --check-prefix=SCORE < %t.score
// RUN: %python %S/Inputs/check_backup_preserved.py %t.before %t.score
// RUN: bishengir-opt %t.before -hivm-cvp-attention-vector="require-match=true enable-max-scaling=true enable-triton-kernel-compile=true retain-provenance=true" -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true retain-provenance=true" -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each -o %t.combined
// RUN: FileCheck %s --check-prefix=COMBINED < %t.combined
// RUN: %python %S/Inputs/check_backup_preserved.py %t.before %t.combined
// RUN: bishengir-opt %t.before -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each -o %t.boundary
// RUN: %python %S/Inputs/check_backup_preserved.py %t.before %t.boundary
// RUN: bishengir-opt %t.before -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each -o %t.rows
// RUN: %python %S/Inputs/check_backup_preserved.py %t.before %t.rows

// This is the full compiler's captured N16384/HD128 input, including the real
// provenance on its backup. A standalone pass must preserve that backup too.
// The full native comparison additionally exercises BiShengIRPassManager's
// hide/restore action handler, which a standalone bishengir-opt pipeline lacks.

// SCORE: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// SCORE-SAME: hivm.cvp_attention.score_store_removals = 1 : i64
// COMBINED: hivm.cvp_attention.distinct_scratch_removals = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.first_output_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.first_state_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.last_output_specializations = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.max_scalings = 1 : i64
// COMBINED-SAME: hivm.cvp_attention.row_unrolls = 5 : i64
// COMBINED-SAME: hivm.cvp_attention.score_store_removals = 1 : i64

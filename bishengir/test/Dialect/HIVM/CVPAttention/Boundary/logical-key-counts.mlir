// These are logical-boundary component inputs, not replacement runtime shapes.
// RUN: %python %S/Inputs/mutate.py keys-1 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k1
// RUN: bishengir-opt %t.k1 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=SINGLE
// RUN: %python %S/Inputs/mutate.py keys-2 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k2
// RUN: bishengir-opt %t.k2 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI
// RUN: %python %S/Inputs/mutate.py keys-3 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k3
// RUN: bishengir-opt %t.k3 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI
// RUN: %python %S/Inputs/mutate.py keys-4 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k4
// RUN: bishengir-opt %t.k4 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI
// RUN: %python %S/Inputs/mutate.py keys-5 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k5
// RUN: bishengir-opt %t.k5 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI
// RUN: %python %S/Inputs/mutate.py keys-6 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k6
// RUN: bishengir-opt %t.k6 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI
// RUN: %python %S/Inputs/mutate.py keys-8 %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.k8
// RUN: bishengir-opt %t.k8 -hivm-cvp-attention-boundary="require-match=true enable-first-output=true enable-last-output=true enable-first-state=true" -verify-each | FileCheck %s --check-prefix=MULTI

// SINGLE: hivm.cvp_attention.first_output_specializations = 1 : i64
// SINGLE-SAME: hivm.cvp_attention.last_output_specializations = 1 : i64
// SINGLE: hivm.cvp_attention.boundary = "first_last_output"
// MULTI: hivm.cvp_attention.first_output_specializations = 1 : i64
// MULTI-SAME: hivm.cvp_attention.last_output_specializations = 1 : i64
// MULTI: arith.subi
// MULTI: arith.cmpi eq

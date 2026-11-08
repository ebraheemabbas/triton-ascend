// RUN: bishengir-opt %S/Inputs/row-update.mlir -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each | FileCheck %s --check-prefix=PAIR
// RUN: %python %S/Inputs/mutate.py renamed %S/Inputs/row-update.mlir > %t.renamed
// RUN: bishengir-opt %t.renamed -hivm-cvp-attention-vector-schedule="require-match=true" -verify-each | FileCheck %s --check-prefix=PAIR
// RUN: %python %S/Inputs/mutate.py odd %S/Inputs/row-update.mlir > %t.odd
// RUN: bishengir-opt %t.odd -verify-each -o /dev/null
// RUN: not bishengir-opt %t.odd -hivm-cvp-attention-vector-schedule="require-match=true" 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate.py effect %S/Inputs/row-update.mlir > %t.effect
// RUN: bishengir-opt %t.effect -verify-each -o /dev/null
// RUN: not bishengir-opt %t.effect -hivm-cvp-attention-vector-schedule="require-match=true" 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate.py overlap %S/Inputs/row-update.mlir > %t.overlap
// RUN: bishengir-opt %t.overlap -verify-each -o /dev/null
// RUN: not bishengir-opt %t.overlap -hivm-cvp-attention-vector-schedule="require-match=true" 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate.py carry %S/Inputs/row-update.mlir > %t.carry
// RUN: bishengir-opt %t.carry -verify-each -o /dev/null
// RUN: not bishengir-opt %t.carry -hivm-cvp-attention-vector-schedule="require-match=true" 2>&1 | FileCheck %s --check-prefix=REJECT
// RUN: %python %S/Inputs/mutate.py no-route %S/Inputs/row-update.mlir > %t.no-route
// RUN: bishengir-opt %t.no-route -verify-each -o /dev/null
// RUN: not bishengir-opt %t.no-route -hivm-cvp-attention-vector-schedule="require-match=true" 2>&1 | FileCheck %s --check-prefix=REJECT

// PAIR: hivm.cvp_attention.row_unrolls = 1 : i64
// PAIR: arith.constant 2 : index
// PAIR: scf.for
// PAIR: arith.mulf
// PAIR: arith.addf
// PAIR: vector.transfer_write
// PAIR: arith.addi
// PAIR: arith.mulf
// PAIR: arith.addf
// PAIR: vector.transfer_write
// PAIR: hivm.cvp_attention.rows_per_iteration = 2 : i64
// REJECT: CVP rows2 requires a verified route and an independent even row loop without external effects

// RUN: %python %S/Inputs/guarded-drain.py mutate tail-barrier %S/Inputs/gss-guarded-captured.mlir %t.tail-barrier
// RUN: bishengir-opt %t.tail-barrier --verify-each -o /dev/null
// RUN: not bishengir-opt %t.tail-barrier --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate unguarded-handoff %S/Inputs/gss-guarded-captured.mlir %t.unguarded-handoff
// RUN: bishengir-opt %t.unguarded-handoff --verify-each -o /dev/null
// RUN: not bishengir-opt %t.unguarded-handoff --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate missing-return %S/Inputs/gss-guarded-captured.mlir %t.missing-return
// RUN: bishengir-opt %t.missing-return --verify-each -o /dev/null
// RUN: not bishengir-opt %t.missing-return --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate foreign-sibling-handoff %S/Inputs/gss-guarded-captured.mlir %t.foreign-sibling-handoff
// RUN: bishengir-opt %t.foreign-sibling-handoff --verify-each -o /dev/null
// RUN: not bishengir-opt %t.foreign-sibling-handoff --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate observable-counter %S/Inputs/gss-guarded-captured.mlir %t.observable-counter
// RUN: bishengir-opt %t.observable-counter --verify-each -o /dev/null
// RUN: not bishengir-opt %t.observable-counter --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate odd-extra %S/Inputs/gss-guarded-captured.mlir %t.odd-extra
// RUN: bishengir-opt %t.odd-extra --verify-each -o /dev/null
// RUN: not bishengir-opt %t.odd-extra --hivm-cvp-cube-derived-drain="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: CVP Cube derived drain

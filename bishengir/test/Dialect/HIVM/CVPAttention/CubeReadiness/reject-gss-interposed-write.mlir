// RUN: python3 %S/Inputs/mutate-gss-handoffs.py mutate interposed-write %S/Inputs/gss-handoffs-captured.mlir > %t
// RUN: bishengir-opt %t --verify-each -o %t.valid
// RUN: not bishengir-opt %t --hivm-cvp-cube-readiness="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: CVP Cube readiness
// The inserted counter write is rejected by the unchanged private-counter
// ownership guard before readiness navigation. Require rejection at the public
// pass boundary without assuming which prerequisite emits the diagnostic.

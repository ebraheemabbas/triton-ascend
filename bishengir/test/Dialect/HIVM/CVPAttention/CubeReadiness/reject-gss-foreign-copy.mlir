// RUN: python3 %S/Inputs/mutate-gss-handoffs.py mutate foreign-copy %S/Inputs/gss-handoffs-captured.mlir > %t
// RUN: bishengir-opt %t --verify-each -o %t.valid
// RUN: not bishengir-opt %t --hivm-cvp-cube-readiness="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: CVP Cube readiness: PV ready producer is not its actual V copy
// The V copy must target the actual PV descriptor, even with GSS handoffs present.

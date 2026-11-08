// RUN: python3 %S/Inputs/mutate-gss-handoffs.py mutate unknown-direction %S/Inputs/gss-handoffs-captured.mlir > %t
// RUN: bishengir-opt %t --verify-each -o %t.valid
// RUN: not bishengir-opt %t --hivm-cvp-cube-readiness="enable=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: CVP Cube readiness: unpaired QK/PV MTE2 readiness producer
// A different cross-core pipe direction is outside the preserved-handoff grammar.

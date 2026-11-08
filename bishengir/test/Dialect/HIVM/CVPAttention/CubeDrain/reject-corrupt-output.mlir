// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-overlap="enable=true" --hivm-cvp-cube-derived-drain="enable=true" --verify-each -o %t.good
// RUN: %python %S/mutate-output.py drop-suffix-signal %t.good %t.drop-suffix-signal
// RUN: bishengir-opt %t.drop-suffix-signal --verify-each -o %t.valid
// RUN: not bishengir-opt %t.drop-suffix-signal --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %python %S/mutate-output.py reorder-suffix-signals %t.good %t.reorder-suffix-signals
// RUN: bishengir-opt %t.reorder-suffix-signals --verify-each -o %t.valid
// RUN: not bishengir-opt %t.reorder-suffix-signals --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %python %S/mutate-output.py truncate-suffix %t.good %t.truncate-suffix
// RUN: bishengir-opt %t.truncate-suffix --verify-each -o %t.valid
// RUN: not bishengir-opt %t.truncate-suffix --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %python %S/mutate-output.py drop-both-cube-signals %t.good %t.drop-both-cube-signals
// RUN: bishengir-opt %t.drop-both-cube-signals --verify-each -o %t.valid
// RUN: not bishengir-opt %t.drop-both-cube-signals --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s --check-prefix=ERROR
// RUN: %python %S/mutate-output.py drop-return-credit %t.good %t.drop-return-credit
// RUN: bishengir-opt %t.drop-return-credit --verify-each -o %t.valid
// RUN: not bishengir-opt %t.drop-return-credit --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s --check-prefix=ERROR
// ERROR: unproved final QK/PV clock, drain or physical Q banks

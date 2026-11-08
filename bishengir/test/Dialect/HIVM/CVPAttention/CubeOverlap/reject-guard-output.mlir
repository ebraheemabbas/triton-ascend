// RUN: bishengir-opt %S/../QCache/scheduled-scalar-counter.mlir --hivm-cvp-cube-readiness="enable=true" --hivm-cvp-cube-overlap="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t
// RUN: %python %S/mutate-guard-output.py %t commuted %t.commuted
// RUN: bishengir-opt %t.commuted --hivm-cvp-cube-overlap="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o /dev/null
// RUN: %python %S/mutate-guard-output.py %t false %t.false
// RUN: not bishengir-opt %t.false --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/mutate-guard-output.py %t active-wait %t.active
// RUN: not bishengir-opt %t.active --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/mutate-guard-output.py %t missing-wait %t.missing
// RUN: not bishengir-opt %t.missing --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// The typed i1 true is accepted in either XOR operand position. A false bit,
// a wait guarded by active PV, or a missing wait must still be rejected.
// CHECK: L0C overlap lacks the complete intra-iteration, next-write and return fences

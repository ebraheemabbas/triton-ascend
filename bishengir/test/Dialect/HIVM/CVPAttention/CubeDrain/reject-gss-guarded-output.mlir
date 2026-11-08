// RUN: bishengir-opt %S/Inputs/gss-guarded-captured.mlir --hivm-cvp-cube-derived-drain="enable=true" --verify-each -o %t.good
// RUN: %python %S/Inputs/guarded-drain.py mutate wrong-main-bound %t.good %t.wrong-main-bound
// RUN: bishengir-opt %t.wrong-main-bound --verify-each -o /dev/null
// RUN: not bishengir-opt %t.wrong-main-bound --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate wrong-policy %t.good %t.wrong-policy
// RUN: bishengir-opt %t.wrong-policy --verify-each -o /dev/null
// RUN: not bishengir-opt %t.wrong-policy --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate vector-work-mismatch %t.good %t.vector-work-mismatch
// RUN: bishengir-opt %t.vector-work-mismatch --verify-each -o /dev/null
// RUN: not bishengir-opt %t.vector-work-mismatch --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate missing-return %t.good %t.missing-return
// RUN: bishengir-opt %t.missing-return --verify-each -o /dev/null
// RUN: not bishengir-opt %t.missing-return --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate unguarded-handoff %t.good %t.unguarded-handoff
// RUN: bishengir-opt %t.unguarded-handoff --verify-each -o /dev/null
// RUN: not bishengir-opt %t.unguarded-handoff --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/guarded-drain.py mutate odd-extra %t.good %t.odd-extra
// RUN: bishengir-opt %t.odd-extra --verify-each -o /dev/null
// RUN: not bishengir-opt %t.odd-extra --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each --mlir-print-op-on-diagnostic=false 2>&1 | FileCheck %s
// CHECK: unproved final QK/PV clock, drain or physical Q banks

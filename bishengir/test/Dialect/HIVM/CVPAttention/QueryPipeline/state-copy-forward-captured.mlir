// RUN: bishengir-opt %S/Inputs/state-copy-forward-captured.mlir --hivm-cvp-query-state-copy-forward="require-match" | FileCheck %s
// CHECK: hivm.cvp_query.snapshot_copies_forwarded = 1 : i64
// CHECK: hivm.cvp_query.state_bank = {bank = 0 : i32, clock = "O"
// CHECK: annotation.mark
// CHECK: scf.for
// CHECK: hivm.cvp_query.state_storage

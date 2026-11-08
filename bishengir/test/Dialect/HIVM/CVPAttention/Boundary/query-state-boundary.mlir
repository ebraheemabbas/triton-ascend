// RUN: %python %S/Inputs/check_query_state_boundary.py --opt bishengir-opt --input %S/Inputs/QueryState/hd64-bn256.before-boundary.mlir | FileCheck %s
// Captured normal original-kernel B2/H8/N512/HD64/BM128/BN256 query/cache input.
// Tests all first/final-O/first-state combinations and valid hostile use graphs.
// CHECK: Explicit query-state boundary cases: 7; valid adversarial graphs rejected transactionally: 16
module {}

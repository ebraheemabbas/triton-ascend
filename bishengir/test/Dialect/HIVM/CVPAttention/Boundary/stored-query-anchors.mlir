// RUN: %python %S/Inputs/check_stored_query_anchors.py --opt bishengir-opt --input %S/Inputs/stored-query-anchors-captured.mlir | FileCheck %s
// Real B060 original-kernel B2/H8/N512/HD128/BN128 stored-query input.
// Preserve frozen MIX marker ownership while specializing first/final updates.
// CHECK: Stored query anchor cases: 7; valid adversarial graphs rejected transactionally: 4
module {}

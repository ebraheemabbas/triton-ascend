// RUN: %python %S/Inputs/check_query_unroll.py --opt bishengir-opt --inputs %S/.. | FileCheck %s
// The input is recognized before genuine ordinary-Unroll emission. All U1/U2/U4
// lane and boundary checks consume the emitted scalar SSA operations. This
// verifies per-query emitter provenance/tails, not continuous query support.
// CHECK: ordinary-Unroll emitter cases: 36; provenance rollback cases: 20; continuous-Unroll support: rejected

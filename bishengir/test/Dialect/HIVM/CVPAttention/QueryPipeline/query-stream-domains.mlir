// RUN: %python %S/Inputs/check_query_domains.py --opt bishengir-opt --inputs %S | FileCheck %s
// Actual-pass checks on captured HD64/HD128 modules: zero, one, two and unequal
// query assignments, 2/3/4/5/6/8 key tiles, local first/last predicates, and
// byte-identical optional rejection. One-key and ordinary Unroll remain explicit
// unsupported cases. U2/U4 lane arithmetic is tested separately in the C++ unit
// tests; it must not be reported as an implemented ordinary Unroll emitter.
// CHECK: query-domain cases: 84; rejection/rollback cases: 12; module transactions: 2

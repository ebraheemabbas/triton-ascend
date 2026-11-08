// Synthetic CPU work-domain boundaries of the actual structured GSS fixture.
// No hardware correctness or timing claim.
// RUN: %python %S/Inputs/guarded-drain.py mutate zero-work %S/Inputs/gss-guarded-captured.mlir %t.zero-work
// RUN: bishengir-opt %t.zero-work --verify-each -o /dev/null
// RUN: bishengir-opt %t.zero-work --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.zero-work.good
// RUN: %python %S/Inputs/guarded-drain.py preserve %t.zero-work %t.zero-work.good
// RUN: bishengir-opt %t.zero-work.good --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.zero-work.second
// RUN: diff %t.zero-work.good %t.zero-work.second
// RUN: %python %S/Inputs/guarded-drain.py mutate one-query %S/Inputs/gss-guarded-captured.mlir %t.one-query
// RUN: bishengir-opt %t.one-query --verify-each -o /dev/null
// RUN: bishengir-opt %t.one-query --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.one-query.good
// RUN: %python %S/Inputs/guarded-drain.py preserve %t.one-query %t.one-query.good
// RUN: bishengir-opt %t.one-query.good --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.one-query.second
// RUN: diff %t.one-query.good %t.one-query.second
// RUN: %python %S/Inputs/guarded-drain.py mutate three-queries %S/Inputs/gss-guarded-captured.mlir %t.three-queries
// RUN: bishengir-opt %t.three-queries --verify-each -o /dev/null
// RUN: bishengir-opt %t.three-queries --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.three-queries.good
// RUN: %python %S/Inputs/guarded-drain.py preserve %t.three-queries %t.three-queries.good
// RUN: bishengir-opt %t.three-queries.good --hivm-cvp-cube-derived-drain="enable=true" --hivm-verify-cvp-q-cache="mode=cached require-match=true" --verify-each -o %t.three-queries.second
// RUN: diff %t.three-queries.good %t.three-queries.second

// RUN: %python %S/Inputs/mutate.py release-inside-output %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.release
// RUN: bishengir-opt %t.release -verify-each -o /dev/null
// RUN: not bishengir-opt %t.release -hivm-cvp-attention-boundary="require-match=true enable-first-output=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py initial-output-escape %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.escape
// RUN: bishengir-opt %t.escape -verify-each -o /dev/null
// RUN: not bishengir-opt %t.escape -hivm-cvp-attention-boundary="require-match=true enable-first-output=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py final-output-extra-reader %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.reader
// RUN: bishengir-opt %t.reader -verify-each -o /dev/null
// RUN: not bishengir-opt %t.reader -hivm-cvp-attention-boundary="require-match=true enable-last-output=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py state-initial-two %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.initial
// RUN: bishengir-opt %t.initial -verify-each -o /dev/null
// RUN: not bishengir-opt %t.initial -hivm-cvp-attention-boundary="require-match=true enable-first-state=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py output-index-metadata %S/../DistinctScratch/Inputs/actual-hd128.mlir > %t.metadata
// RUN: bishengir-opt %t.metadata -verify-each -o /dev/null
// RUN: not bishengir-opt %t.metadata -hivm-cvp-attention-boundary="require-match=true enable-first-output=true" 2>&1 | FileCheck %s

// CHECK: CVP boundary specialization requires a proved Skew logical boundary, private initialization, and complete output/state use graph

// Every mutant is independently verified before requesting the transformation.
// Rejection must be the pass's legality diagnostic, not a parser/verifier error.
// RUN: %python %S/Inputs/mutate.py scratch-reader %S/Inputs/actual-hd128.mlir > %t.scratch-reader
// RUN: bishengir-opt %t.scratch-reader --verify-each -o /dev/null
// RUN: not bishengir-opt %t.scratch-reader --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py scratch-alias %S/Inputs/actual-hd128.mlir > %t.scratch-alias
// RUN: bishengir-opt %t.scratch-alias --verify-each -o /dev/null
// RUN: not bishengir-opt %t.scratch-alias --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py raw-alias %S/Inputs/actual-hd128.mlir > %t.raw-alias
// RUN: bishengir-opt %t.raw-alias --verify-each -o /dev/null
// RUN: not bishengir-opt %t.raw-alias --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py raw-writer %S/Inputs/actual-hd128.mlir > %t.raw-writer
// RUN: bishengir-opt %t.raw-writer --verify-each -o /dev/null
// RUN: not bishengir-opt %t.raw-writer --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py scratch-mark %S/Inputs/actual-hd128.mlir > %t.scratch-mark
// RUN: bishengir-opt %t.scratch-mark --verify-each -o /dev/null
// RUN: not bishengir-opt %t.scratch-mark --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py scratch-escape %S/Inputs/actual-hd128.mlir > %t.scratch-escape
// RUN: bishengir-opt %t.scratch-escape --verify-each -o /dev/null
// RUN: not bishengir-opt %t.scratch-escape --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py producer-extra-call %S/Inputs/actual-hd128.mlir > %t.producer-extra-call
// RUN: bishengir-opt %t.producer-extra-call --verify-each -o /dev/null
// RUN: not bishengir-opt %t.producer-extra-call --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py consumer-extra-call %S/Inputs/actual-hd128.mlir > %t.consumer-extra-call
// RUN: bishengir-opt %t.consumer-extra-call --verify-each -o /dev/null
// RUN: not bishengir-opt %t.consumer-extra-call --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py function-index-metadata %S/Inputs/actual-hd128.mlir > %t.function-index-metadata
// RUN: bishengir-opt %t.function-index-metadata --verify-each -o /dev/null
// RUN: not bishengir-opt %t.function-index-metadata --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py extra-symbol-reference %S/Inputs/actual-hd128.mlir > %t.extra-symbol-reference
// RUN: bishengir-opt %t.extra-symbol-reference --verify-each -o /dev/null
// RUN: not bishengir-opt %t.extra-symbol-reference --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py call-index-metadata %S/Inputs/actual-hd128.mlir > %t.call-index-metadata
// RUN: bishengir-opt %t.call-index-metadata --verify-each -o /dev/null
// RUN: not bishengir-opt %t.call-index-metadata --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py wrong-channel %S/Inputs/actual-hd128.mlir > %t.wrong-channel
// RUN: bishengir-opt %t.wrong-channel --verify-each -o /dev/null
// RUN: not bishengir-opt %t.wrong-channel --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py foreign-mte-credit %S/Inputs/actual-hd128.mlir > %t.foreign-mte-credit
// RUN: bishengir-opt %t.foreign-mte-credit --verify-each -o /dev/null
// RUN: not bishengir-opt %t.foreign-mte-credit --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py foreign-mte-ready %S/Inputs/actual-hd128.mlir > %t.foreign-mte-ready
// RUN: bishengir-opt %t.foreign-mte-ready --verify-each -o /dev/null
// RUN: not bishengir-opt %t.foreign-mte-ready --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py foreign-dynamic-event %S/Inputs/actual-hd128.mlir > %t.foreign-dynamic-event
// RUN: bishengir-opt %t.foreign-dynamic-event --verify-each -o /dev/null
// RUN: not bishengir-opt %t.foreign-dynamic-event --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py generic-block-sync %S/Inputs/actual-hd128.mlir > %t.generic-block-sync
// RUN: bishengir-opt %t.generic-block-sync --verify-each -o /dev/null
// RUN: not bishengir-opt %t.generic-block-sync --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py foreign-sync-mode %S/Inputs/actual-hd128.mlir > %t.foreign-sync-mode
// RUN: bishengir-opt %t.foreign-sync-mode --verify-each -o /dev/null
// RUN: not bishengir-opt %t.foreign-sync-mode --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py release-before-p %S/Inputs/actual-hd128.mlir > %t.release-before-p
// RUN: bishengir-opt %t.release-before-p --verify-each -o /dev/null
// RUN: not bishengir-opt %t.release-before-p --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py conditional-release %S/Inputs/actual-hd128.mlir > %t.conditional-release
// RUN: bishengir-opt %t.conditional-release --verify-each -o /dev/null
// RUN: not bishengir-opt %t.conditional-release --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py duplicate-initial-credit %S/Inputs/actual-hd128.mlir > %t.duplicate-initial-credit
// RUN: bishengir-opt %t.duplicate-initial-credit --verify-each -o /dev/null
// RUN: not bishengir-opt %t.duplicate-initial-credit --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py outer-generation %S/Inputs/actual-hd128.mlir > %t.outer-generation
// RUN: bishengir-opt %t.outer-generation --verify-each -o /dev/null
// RUN: not bishengir-opt %t.outer-generation --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py key-generation %S/Inputs/actual-hd128.mlir > %t.key-generation
// RUN: bishengir-opt %t.key-generation --verify-each -o /dev/null
// RUN: not bishengir-opt %t.key-generation --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py phase %S/Inputs/actual-hd128.mlir > %t.phase
// RUN: bishengir-opt %t.phase --verify-each -o /dev/null
// RUN: not bishengir-opt %t.phase --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py wrong-slot %S/Inputs/actual-hd128.mlir > %t.wrong-slot
// RUN: bishengir-opt %t.wrong-slot --verify-each -o /dev/null
// RUN: not bishengir-opt %t.wrong-slot --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py pv-role %S/Inputs/actual-hd128.mlir > %t.pv-role
// RUN: bishengir-opt %t.pv-role --verify-each -o /dev/null
// RUN: not bishengir-opt %t.pv-role --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py qk-input-lineage %S/Inputs/actual-hd128.mlir > %t.qk-input-lineage
// RUN: bishengir-opt %t.qk-input-lineage --verify-each -o /dev/null
// RUN: not bishengir-opt %t.qk-input-lineage --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py destination-only-exp-edge %S/Inputs/actual-hd128.mlir > %t.destination-only-exp-edge
// RUN: bishengir-opt %t.destination-only-exp-edge --verify-each -o /dev/null
// RUN: not bishengir-opt %t.destination-only-exp-edge --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py unknown-call %S/Inputs/actual-hd128.mlir > %t.unknown-call
// RUN: bishengir-opt %t.unknown-call --verify-each -o /dev/null
// RUN: not bishengir-opt %t.unknown-call --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// RUN: %python %S/Inputs/mutate.py missing-drain %S/Inputs/actual-hd128.mlir > %t.missing-drain
// RUN: bishengir-opt %t.missing-drain --verify-each -o /dev/null
// RUN: not bishengir-opt %t.missing-drain --verify-each --hivm-cvp-attention-vector="require-match=true" 2>&1 | FileCheck %s
// CHECK: error: CVP score rematerialization requires a verified private in-place score/P call chain or a proved distinct-scratch lifetime

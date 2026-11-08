// RUN: split-file %s %t
// RUN: not bishengir-opt %t/lost.mlir --hivm-verify-cvp-attention 2>&1 | FileCheck %s --check-prefix=COUNT
// RUN: not bishengir-opt %t/missing.mlir --hivm-verify-cvp-attention="retain-provenance=true" 2>&1 | FileCheck %s --check-prefix=COUNT
// RUN: not bishengir-opt %t/malformed.mlir --hivm-verify-cvp-attention 2>&1 | FileCheck %s --check-prefix=MALFORMED
// COUNT: CVP attention route invalid: recognized plan count changed or was not recorded
// MALFORMED: CVP attention route invalid: malformed expected count

//--- lost.mlir
module {
  func.func @lost_plan() attributes {hivm.cvp_attention.expected_count = 1 : i64} {
    return
  }
}
//--- missing.mlir
module {
  func.func @missing_prepare() {
    return
  }
}
//--- malformed.mlir
module {
  func.func @invalid_count() attributes {hivm.cvp_attention.expected_count = -1 : i64} {
    return
  }
}

// RUN: bishengir-opt %s -hivm-insert-anchors-and-backup -verify-each | FileCheck %s

// Only a verified attention backup is visible to the vector pass. Other
// transforms still see the frozen backup through the existing filter policy.
module {
  func.func @attention() attributes {
    hivm.func_core_type = #hivm.func_core_type<MIX>,
    hivm.cvp_attention.verified = [{block_m = 128 : i64, block_n = 128 : i64,
      head_dim = 128 : i64, key_count = 16384 : i64}]
  } {
    return
  }
  func.func @ordinary() attributes {
    hivm.func_core_type = #hivm.func_core_type<MIX>
  } {
    return
  }
}

// CHECK-LABEL: func.func @attention_backup()
// CHECK-SAME: annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module,hivm-cvp-attention-vector">
// CHECK-SAME: hivm.backup_function
// CHECK-SAME: hivm.cvp_attention.verified
// CHECK-LABEL: func.func @ordinary_backup()
// CHECK-SAME: annotation.filter_passes = #annotation.filter_passes<"hivm-insert-anchors-and-backup,hivm-delayed-cross-core-gss,split-simt-module">
// CHECK-SAME: hivm.backup_function
// CHECK-NOT: hivm.cvp_attention.verified

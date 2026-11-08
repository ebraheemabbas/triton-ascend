// RUN: not bishengir-opt %s --hivm-cvp-attention-vector="enable-max-scaling=true enable-score-rematerialization=false" 2>&1 | FileCheck %s
// CHECK: CVP attention max scaling requires score rematerialization
module {}

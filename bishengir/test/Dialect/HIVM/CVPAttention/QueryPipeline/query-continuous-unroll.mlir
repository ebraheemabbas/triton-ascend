// RUN: %python %S/Inputs/check_continuous_unroll.py --opt bishengir-opt --inputs %S/.. | FileCheck %s
// Actual direct-emitter IR is decoded independently for all QK/P/PV/O work,
// query resets/finalization, empty and unequal query assignments, partial U4
// groups, and MIX/AIC/AIV paths. Separate original attention inputs exercise
// real ordinary-mode preparation/layout/query transformation. Explicitly
// disabling continuous queries must preserve the original emitter bytes.
// No native asynchronous, numerical-attention or hardware claim follows.
// CHECK: decoded continuous cases: 378; transactional rejections: 4; actual query preparations and disabled controls: 36

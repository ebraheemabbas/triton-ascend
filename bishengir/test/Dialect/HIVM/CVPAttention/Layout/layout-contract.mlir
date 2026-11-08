// RUN: %python %S/Inputs/check_layout.py --opt bishengir-opt --fixtures %S/Inputs

// The helper uses actual post-CVP fixtures, checks whole-chain mappings and
// preserves the schedule. Negative cases must reject before producing a module.
module {}

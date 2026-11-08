// RUN: %python %S/Inputs/check-scalar-input-forward.py %S/Inputs/scalar-input-forward-captured.mlir bishengir-opt captured

// Actual source240v2 full-kernel IR after direct-O forwarding, before ordinary
// memory planning. The checked fixture is retained unchanged under Inputs.

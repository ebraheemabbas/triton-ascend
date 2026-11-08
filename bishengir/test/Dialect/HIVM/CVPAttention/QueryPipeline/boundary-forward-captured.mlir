// RUN: %python %S/Inputs/check-boundary-forward-captured.py %S/Inputs/boundary-forwarding-source244.mlir bishengir-opt
// Real source244 BS128/H8/N16384/HD64/BM128/BN256 pre-forward capture.
// Input SHA256:f06fc4d20e204e7177ecf8ba0772d92580a3093f759548423d64e90dcbbe4047.
// This structural regression does not claim small-shape hardware timing.

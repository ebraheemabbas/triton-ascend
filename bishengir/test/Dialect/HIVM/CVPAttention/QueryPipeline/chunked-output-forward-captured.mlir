// The fixture is the actual BS128/H8/N1024/HD128/BN128 native pre-forwarding
// module, with only the unreferenced pass-filtered backup function excluded.
// It is structural compiler evidence, never an injected benchmark input.
// RUN: %python %S/Inputs/check-chunked-output-forward.py %S/Inputs/chunked-output-source247-hd128.mlir bishengir-opt
module {}

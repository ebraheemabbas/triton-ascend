"""Make defined alias/effect hazards from the positive snapshot fixture."""
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
mode = sys.argv[2]
copy = "hivm.hir.copy ins(%bank : memref<2x4xf32, #ub>) outs(%snapshot : memref<2x4xf32, #ub>)"
call = "func.call @update(%pv, %selected, %pv)"


def replace(before, after):
    global text
    assert text.count(before) == 1, (mode, before)
    text = text.replace(before, after)


if mode == "source-overwrite":
    replace(copy, copy + "\n        hivm.hir.copy ins(%zero : memref<2x4xf32, #ub>) outs(%bank : memref<2x4xf32, #ub>)")
elif mode == "snapshot-write":
    replace(call, "func.call @update(%pv, %selected, %snapshot)")
elif mode == "snapshot-escape":
    replace(copy, copy + "\n        func.call @opaque(%snapshot) : (memref<2x4xf32, #ub>) -> ()")
elif mode == "bank-escape":
    replace(copy, "func.call @opaque(%bank) : (memref<2x4xf32, #ub>) -> ()\n        " + copy)
elif mode == "missing-unique":
    replace("    annotation.mark %bank {mem_unique} : memref<2x4xf32, #ub>\n", "")
elif mode == "partial-copy":
    replace(copy, """%source_slice = memref.subview %bank[0, 0] [1, 4] [1, 1] : memref<2x4xf32, #ub> to memref<1x4xf32, strided<[4, 1]>, #ub>
        %snapshot_slice = memref.subview %snapshot[0, 0] [1, 4] [1, 1] : memref<2x4xf32, #ub> to memref<1x4xf32, strided<[4, 1]>, #ub>
        hivm.hir.copy ins(%source_slice : memref<1x4xf32, strided<[4, 1]>, #ub>) outs(%snapshot_slice : memref<1x4xf32, strided<[4, 1]>, #ub>)""")
elif mode == "stronger-alignment":
    replace("%snapshot = memref.alloc() :", "%snapshot = memref.alloc() {alignment = 64 : i64} :")
elif mode == "hidden-effect":
    replace("    %c0 = arith.constant 0 : index\n    %zero",
            "    func.call @global_effect() : () -> ()\n    %c0 = arith.constant 0 : index\n    %zero")
elif mode == "bank-output-alias":
    replace(call, "func.call @update(%pv, %selected, %bank)")
elif mode in ("exclusive-branches", "same-branch-overwrite"):
    # Keep the same typed call while moving an actual bank write onto either
    # the mutually exclusive path or the path leading to the snapshot read.
    typed = ('func.call @update(%pv, %selected, %pv) {hivm.vector_function} : '
             '(memref<2x4xf32, #ub>, memref<2x4xf32, #ub>, memref<2x4xf32, #ub>) -> ()')
    write = "hivm.hir.copy ins(%zero : memref<2x4xf32, #ub>) outs(%bank : memref<2x4xf32, #ub>)"
    then = write if mode == "exclusive-branches" else ""
    otherwise = typed if mode == "exclusive-branches" else write + "\n          " + typed
    replace(typed, "scf.if %first {\n          " + then +
            "\n        } else {\n          " + otherwise + "\n        }")
else:
    raise SystemExit("unknown mutation: " + mode)
print(text)

"""Check the bounded structural delta after MLIR verification."""
import pathlib
import re
import sys

before, after = (pathlib.Path(p).read_text() for p in sys.argv[1:])
for operation in ("memref.alloc(", "memref.subview ", "vector.transfer_write "):
    assert before.count(operation) - after.count(operation) == 1, operation

producer = "_attn_fwd_mix_aiv_outlined_vf_0"
consumer = "_attn_fwd_mix_aiv_outlined_vf_2"

def arguments(text, function):
    line = next(line for line in text.splitlines() if "func.func @" + function + "(" in line)
    return re.findall(r"%[A-Za-z_0-9]+:", line.split(") attributes")[0])

def operands(text, function):
    line = next(line for line in text.splitlines() if "func.call @" + function + "(" in line)
    return line.split("(", 1)[1].split(")", 1)[0].split(", ")

assert len(arguments(before, producer)) == 3
assert len(arguments(after, producer)) == 2
assert len(operands(before, producer)) == 3
assert len(operands(after, producer)) == 2
assert operands(before, producer)[2] == operands(before, consumer)[1]
assert operands(after, producer)[0] == operands(after, consumer)[1]
assert len(arguments(before, consumer)) == len(arguments(after, consumer)) == 4

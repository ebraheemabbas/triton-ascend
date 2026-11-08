"""One bounded mutation per actual late-HD128 fixture; never a compiler oracle."""
import argparse
import pathlib
import re

AIV = "_attn_fwd_mix_aiv"
AIC = "_attn_fwd_mix_aic"
BACKUP = "_attn_fwd_backup"
MAX = "_attn_fwd_mix_aiv_outlined_vf_0"
P = "_attn_fwd_mix_aiv_outlined_vf_2"
TILE = "memref<64x128xf32, #hivm.address_space<ub>>"
ROW = "memref<64xf32, #hivm.address_space<ub>>"

def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)

def function(text, name, change):
    start = text.index("  func.func @" + name + "(")
    end = text.index("\n  }", start) + len("\n  }")
    return text[:start] + change(text[start:end]) + text[end:]

def call_line(text, name):
    lines = [line for line in text.splitlines() if "func.call @" + name + "(" in line]
    assert len(lines) == 1, (name, len(lines))
    return lines[0]

def after_max(text, extra):
    def change(body):
        line = call_line(body, MAX)
        return once(body, line, line + "\n" + extra)
    return function(text, AIV, change)

def replace_aiv(text, old, new):
    return function(text, AIV, lambda body: once(body, old, new))

def mutate(text, case):
    if case == "base":
        return text
    if case == "renamed":
        text = text.replace("_attn_fwd", "renamed_kernel")
        return re.sub(r"%([A-Za-z_0-9]+)", lambda m: "%renamed_" + m[1], text)
    if case == "renumbered":
        text = re.sub(r"flag = ([012])\b", lambda m: "flag = " + {"0":"5", "1":"7", "2":"9"}[m[1]], text)
        return re.sub(r"tightly_coupled_buffer<([012])>", lambda m: "tightly_coupled_buffer<" + {"0":"17", "1":"23", "2":"19"}[m[1]] + ">", text)
    if case == "multiply-attributes":
        return function(text, MAX, lambda b: once(b, "arith.mulf %3, %cst :", 'arith.mulf %cst, %3 fastmath<contract> {cvp_test_marker = "original-scale"} :'))
    if case == "argument-attributes":
        def move_argument(body):
            body = once(body, "%arg0: " + TILE, "%arg0: " + TILE + " {llvm.readonly}")
            body = once(body, "%arg1: " + ROW, "%arg1: " + ROW + " {llvm.align = 64 : i64}")
            body = once(body, "%arg2: " + TILE, "%arg2: " + TILE + " {llvm.writeonly}")
            body = re.sub(r"%arg([012])\b", lambda m: "%arg" + {"0":"1", "1":"2", "2":"0"}[m[1]], body)
            a = "%arg1: " + TILE + " {llvm.readonly}"
            b = "%arg2: " + ROW + " {llvm.align = 64 : i64}"
            c = "%arg0: " + TILE + " {llvm.writeonly}"
            return once(body, a + ", " + b + ", " + c, c + ", " + a + ", " + b)
        text = function(text, MAX, move_argument)
        def move_call(body):
            line = call_line(body, MAX)
            moved = once(line, "(%alloc_11, %alloc_12, %alloc_13)", "(%alloc_13, %alloc_11, %alloc_12)")
            moved = once(moved, "(" + TILE + ", " + ROW + ", " + TILE + ")", "(" + TILE + ", " + TILE + ", " + ROW + ")")
            return once(body, line, moved)
        return function(text, AIV, move_call)
    if case == "scratch-reader":
        return after_max(text, "            %test_zero = arith.constant 0 : index\n            %test_read = memref.load %alloc_13[%test_zero, %test_zero] : " + TILE)
    if case in ("scratch-alias", "raw-alias"):
        raw = "%alloc_13" if case == "scratch-alias" else "%alloc_11"
        return after_max(text, "            %test_alias = memref.cast " + raw + " : " + TILE + " to memref<?x?xf32, #hivm.address_space<ub>>")
    if case == "raw-writer":
        return after_max(text, "            %test_zero = arith.constant 0 : index\n            %test_value = arith.constant 0.0 : f32\n             memref.store %test_value, %alloc_11[%test_zero, %test_zero] : " + TILE)
    if case == "scratch-mark":
        return after_max(text, "             annotation.mark %alloc_13 {test_external_role} : " + TILE)
    if case == "scratch-escape":
        text = after_max(text, "             func.call @escape(%alloc_13) : (" + TILE + ") -> ()")
        at = text.index("  func.func")
        return text[:at] + "  func.func private @escape(" + TILE + ")\n" + text[at:]
    if case in ("producer-extra-call", "consumer-extra-call"):
        target = MAX if case == "producer-extra-call" else P
        return function(text, AIV, lambda b: once(b, call_line(b, target), call_line(b, target) + "\n" + call_line(b, target)))
    if case in ("function-index-metadata", "extra-symbol-reference"):
        attr = "test.argument_index = 2 : i64" if case == "function-index-metadata" else "test.symbol_reference = @" + MAX
        target = MAX if case == "function-index-metadata" else P
        return function(text, target, lambda b: once(b, " attributes {", " attributes {" + attr + ", "))
    if case == "call-index-metadata":
        return function(text, AIV, lambda b: once(b, call_line(b, MAX), call_line(b, MAX).replace("{hivm.vector_function", "{test.argument_index = 2 : i64, hivm.vector_function")))
    if case == "wrong-channel":
        return replace_aiv(text, "sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 1\n          hivm.hir.anchor {id = 21", "sync_block_wait[<VECTOR>, <PIPE_FIX>, <PIPE_V>] flag = 3\n          hivm.hir.anchor {id = 21")
    if case == "foreign-mte-credit":
        return text.replace("sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2", "sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1").replace("sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2", "sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1")
    if case == "foreign-mte-ready":
        return text.replace("<PIPE_MTE1>, <PIPE_MTE3>] flag = 0", "<PIPE_MTE1>, <PIPE_MTE3>] flag = 1")
    if case == "foreign-dynamic-event":
        return replace_aiv(text, "          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2", "          %test_dynamic = arith.constant 2 : i64\n          hivm.hir.sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = %test_dynamic")
    if case == "generic-block-sync":
        return replace_aiv(text, "    return", "    hivm.hir.sync_block[<BARRIER_VECTOR>]\n    return")
    if case == "foreign-sync-mode":
        return replace_aiv(text, "sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2", "sync_block_set[<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 2 sync_instr_mode = <INTER_BLOCK_SYNCHRONIZATION>")
    if case in ("release-before-p", "conditional-release"):
        release = "          hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1"
        def move_release(body):
            body = once(body, release + "\n", "")
            if case == "release-before-p":
                line = call_line(body, P)
                return once(body, line, "  " + release + "\n" + line)
            needle = "          hivm.hir.anchor {id = 39 : i64}"
            extra = "          %test_condition = arith.constant true\n          scf.if %test_condition {\n  " + release + "\n           }\n"
            return once(body, needle, extra + needle)
        return function(text, AIV, move_release)
    if case == "duplicate-initial-credit":
        line = "\n    hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1"
        return replace_aiv(text, line, line + "\n" + line)
    if case == "outer-generation":
        return replace_aiv(text, "%c28_i32 = arith.constant 28 : i32", "%c28_i32 = arith.constant 29 : i32")
    if case == "key-generation":
        return replace_aiv(text, "%c4096_i32 = arith.constant 4096 : i32", "%c4096_i32 = arith.constant 2048 : i32")
    if case == "phase":
        return replace_aiv(text, "hivm.preload_num = 2 : i32", "hivm.preload_num = 1 : i32")
    if case == "wrong-slot":
        return replace_aiv(text, "tightly_coupled_buffer<0>", "tightly_coupled_buffer<17>")
    if case == "pv-role":
        return function(text, BACKUP, lambda b: re.sub(r"tightly_coupled_buffer<([01])>", lambda m: "tightly_coupled_buffer<" + str(1-int(m[1])) + ">", b))
    if case == "qk-input-lineage":
        def replace_key(body):
            lines = [line for line in body.splitlines() if "memref.reinterpret_cast %arg3 " in line]
            assert len(lines) == 1
            return once(body, lines[0], lines[0].replace("memref.reinterpret_cast %arg3 ", "memref.reinterpret_cast %arg4 "))
        return function(text, AIC, replace_key)
    if case == "destination-only-exp-edge":
        return function(text, BACKUP, lambda b: once(b, "hivm.hir.vexp ins(%45 : tensor<128x128xf32>) outs(%5 : tensor<128x128xf32>)", "hivm.hir.vexp ins(%arg13 : tensor<128x128xf32>) outs(%45 : tensor<128x128xf32>)"))
    if case == "unknown-call":
        text = after_max(text, "             func.call @unknown_effect() : () -> ()")
        at = text.index("  func.func")
        return text[:at] + "  func.func private @unknown_effect()\n" + text[at:]
    if case == "missing-drain":
        return function(text, AIC, lambda b: once(b, "\n    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1\n", "\n"))
    raise ValueError(case)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("input", type=pathlib.Path)
    args = parser.parse_args()
    print(mutate(args.input.read_text(), args.case), end="")

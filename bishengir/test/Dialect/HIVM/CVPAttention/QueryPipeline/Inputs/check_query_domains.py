"""Exercise real query-pass output against the original nested-loop domain.

Only scalar coordinate operations are evaluated. This does not model vector
arithmetic, DMA completion, cross-core synchronization or an Unroll emitter.
"""
import argparse
from pathlib import Path
import re
import subprocess

# MLIR printer names negative constants with a hyphen (for example %c-1_i64).
SSA = r"%[\w.$-]+(?:#\d+)?"


def replace_once(pattern, replacement, text):
    text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    assert count == 1, (pattern, count)
    return text


def make_input(source, tiles, start, upper, stride, negative=None):
    source = re.sub(r"(?m)^//.*\n", "", source)
    bn = int(re.search(r"block_n = (\d+) : i64", source).group(1))
    keys = tiles * bn - (negative == "fractional-tile")
    source = replace_once(r"key_count = \d+ : i64", f"key_count = {keys} : i64", source)
    source = replace_once(r"key_tiles = \d+ : i64", f"key_tiles = {tiles + (negative == 'wrong-tiles')} : i64", source)
    declarations = (
        f"    %cvp_start = arith.constant {start} : i32\n"
        f"    %cvp_upper = arith.constant {upper} : i32\n"
        f"    %cvp_stride = arith.constant {stride} : i32\n"
        f"    %cvp_keys = arith.constant {keys} : i32\n"
        f"    %cvp_twobn = arith.constant {2 * bn} : i32\n")
    source = replace_once(r"^(  func.func .* \{)\n", lambda m: m[1] + "\n" + declarations, source)
    source = replace_once(
        r"^      scf.for (%\w+) = %\w+ to %\w+ step %\w+  : i32 \{",
        lambda m: f"      scf.for {m[1]} = %cvp_start to %cvp_upper step %cvp_stride  : i32 {{", source)
    source = replace_once(
        r"^(        %\w+:5 = scf.for %\w+ = %\w+ to )%\w+( step )(%\w+)( iter_args)",
        lambda m: m[1] + "%cvp_keys" + m[2] +
        ("%cvp_twobn" if negative == "unrolled-step" else m[3]) + m[4], source)
    if negative == "missing-scope":
        source = source.replace(", hivm.preload_num = 3 : i32", "", 1)
    elif negative == "no-skew-scopes":
        source, count = re.subn(
            r", hivm.max_preload_num = 4 : i32, hivm.preload_num = [0-3] : i32", "", source)
        assert count == 4
    return source, bn


def run(opt, source, *options, succeeds=True):
    result = subprocess.run([opt, "-", "--verify-each", *options], input=source,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert (result.returncode == 0) == succeeds, (options, result.returncode, result.stderr)
    return result


def signed_div(lhs, rhs):
    assert rhs != 0
    value = abs(lhs) // abs(rhs)
    return -value if (lhs < 0) != (rhs < 0) else value


class Coordinates:
    """Read the actual pre-scope scalar SSA expressions, independent of names."""
    def __init__(self, text):
        prefix = text[:text.index("scope.scope")]
        self.defs = {}
        self.names = {}
        for line in prefix.splitlines():
            match = re.match(rf"\s*({SSA}) = arith\.(\w+) (.*)", line)
            if not match:
                continue
            result, op, rhs = match.groups()
            self.defs[result] = (op, rhs)
            tag = re.search(r'hivm.cvp_query.coordinate = "(\w+)"', rhs)
            if tag:
                assert tag[1] not in self.names
                self.names[tag[1]] = result
        loops = re.findall(rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA}) iter_args", prefix)
        assert len(loops) == 1
        self.iv, self.lower, self.upper, self.step = loops[0]
        assert set(self.names) == {"query_start", "query_count", "work_end", "query_ordinal",
                                   "local_key", "query_task", "query_slot", "first_key", "last_key"}

    def evaluate(self, name, tick=None):
        if name == self.iv:
            assert tick is not None
            return tick
        op, rhs = self.defs[name]
        operands = re.findall(SSA, rhs)
        values = [self.evaluate(operand, tick) for operand in operands]
        if op == "constant":
            return int(re.match(r"(-?\d+)", rhs)[1])
        if op in ("extsi", "index_cast"):
            return values[0]
        if op == "trunci":
            width = int(re.search(r"to i(\d+)", rhs)[1])
            value = values[0] % (1 << width)
            return value - (1 << width) if value >= (1 << (width - 1)) else value
        if op == "cmpi":
            assert rhs.startswith("eq,")
            return values[0] == values[1]
        lhs, right = values
        if op == "addi":
            return lhs + right
        if op == "subi":
            return lhs - right
        if op == "muli":
            return lhs * right
        if op == "divsi":
            return signed_div(lhs, right)
        if op == "remsi":
            return lhs - signed_div(lhs, right) * right
        if op == "maxsi":
            return max(lhs, right)
        if op == "ceildivsi":
            assert right > 0
            return -((-lhs) // right)
        raise AssertionError((op, rhs))

    def coordinate(self, name, tick=None):
        return self.evaluate(self.names[name], tick)


def verify_coordinates(text, tiles, bn, start, upper, stride):
    assert 'pipeline_mode = "skew"' in text
    assert "key_unroll = 1 : i32" in text
    assert "query_slots = 2 : i32" in text
    assert "physical_lifetimes_verified = false" in text
    coordinate = Coordinates(text)
    tasks = list(range(start, upper, stride))
    expected = [(ordinal, task, key) for ordinal, task in enumerate(tasks)
                for key in range(0, tiles * bn, bn)]
    assert coordinate.coordinate("query_start") == start
    assert coordinate.coordinate("query_count") == len(tasks)
    assert coordinate.coordinate("work_end") == len(expected) * bn
    assert coordinate.evaluate(coordinate.lower) == 0
    assert coordinate.evaluate(coordinate.upper) == len(expected) * bn
    assert coordinate.evaluate(coordinate.step) == bn
    for index, (ordinal, task, key) in enumerate(expected):
        tick = index * bn
        for name, value in (("query_ordinal", ordinal), ("query_task", task),
                            ("local_key", key), ("query_slot", ordinal % 2),
                            ("first_key", key == 0), ("last_key", key == (tiles - 1) * bn)):
            assert coordinate.coordinate(name, tick) == value, (name, tick, value)
    # The first predicate guards Q load and both state resets; last guards the
    # original complete M/O finalization. This is before preload materialization.
    first = re.escape(coordinate.names["first_key"])
    last = re.escape(coordinate.names["last_key"])
    assert len(re.findall(r"scf.if " + first + r"(?:\s|$)", text)) == 3
    assert len(re.findall(r"scf.if " + last + r"(?:\s|$)", text)) == 1


def verify_transaction(opt, source):
    marker = source.rfind("}")
    source = source[:marker] + (
        "  func.func @unsupported() attributes {hivm.cvp_attention.verified = []} {\n"
        "    return\n  }\n") + source[marker:]
    rejected = run(opt, source, "--hivm-cvp-query-pipeline", "--mlir-disable-threading",
                   "--mlir-print-ir-after-failure", succeeds=False)
    assert "expected one verified attention recurrence" in rejected.stderr
    assert "IR Dump After" in rejected.stderr
    assert "func.func @_attn_fwd" in rejected.stderr and "func.func @unsupported" in rejected.stderr
    assert "hivm.cvp_query.stream" not in rejected.stderr
    optional = run(opt, source, "--hivm-cvp-query-pipeline=require-match=false")
    assert "hivm.cvp_query.stream" in optional.stdout
    assert "func.func @unsupported" in optional.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    args = parser.parse_args()
    positives = negatives = 0
    for filename in ("query-stream-captured.mlir", "query-stream-hd128-captured.mlir"):
        captured = (args.inputs / filename).read_text()
        for tiles in (2, 3, 4, 5, 6, 8):
            for start, upper, stride in ((9, 9, 3), (12, 9, 3), (7, 8, 3), (2, 6, 3),
                                         (1, 12, 3), (0, 11, 4), (3, 11, 4)):
                source, bn = make_input(captured, tiles, start, upper, stride)
                output = run(args.opt, source, "--hivm-cvp-query-pipeline").stdout
                verify_coordinates(output, tiles, bn, start, upper, stride)
                positives += 1
        for negative, diagnostic in (
                ("one-key", "two query-state slots require at least two key tiles"),
                ("fractional-tile", "unproved static key domain"),
                ("wrong-tiles", "verified key-tile count disagrees"),
                ("missing-scope", "expected four uniquely indexed"),
                ("no-skew-scopes", "ordinary Unroll query emission is not implemented"),
                ("unrolled-step", "unproved static key domain")):
            source, _ = make_input(captured, 1 if negative == "one-key" else 6,
                                   0, 11, 4, negative)
            baseline = run(args.opt, source).stdout
            rejected = run(args.opt, source, "--hivm-cvp-query-pipeline", succeeds=False)
            assert diagnostic in rejected.stderr, rejected.stderr
            fallback = run(args.opt, source, "--hivm-cvp-query-pipeline=require-match=false")
            assert fallback.stdout == baseline, negative
            negatives += 1
        source, _ = make_input(captured, 6, 0, 11, 4)
        once = run(args.opt, source, "--hivm-cvp-query-pipeline").stdout
        twice = run(args.opt, once, "--hivm-cvp-query-pipeline").stdout
        assert once == twice
        verify_transaction(args.opt, source)
    print(f"query-domain cases: {positives}; rejection/rollback cases: {negatives}; module transactions: 2")


if __name__ == "__main__":
    main()

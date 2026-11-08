"""Check real ordinary-Unroll output and reject unsupported continuous queries.

The emitter is cv-pipelining in mode=unroll, followed by ordinary lower-affine.
No Skew pass, saved output or query-rewrite injection is used. These are compiler
component checks; numerical, native ownership and cross-query qualification are
separate. The continuous-query request must still reject this per-query output.
"""
import argparse
from pathlib import Path
import re
from check_query_domains import Coordinates, SSA, replace_once, run


def key_domain(source, tiles):
    source = re.sub(r"(?m)^//.*\n", "", source)
    match = re.search(r"(%\w+:5 = scf.for %\w+ = %\w+ to )(%\w+)( step )(%\w+)( iter_args)", source)
    assert match
    step_name = match[4]
    bn = int(re.search(re.escape(step_name) + r" = arith.constant (\d+) : i32", source)[1])
    source = source[:match.start()] + match[1] + "%query_test_keys" + match[3] + match[4] + match[5] + source[match.end():]
    source = replace_once(r"^(\s*func.func .* \{)\n",
                          lambda m: m[1] + f"\n  %query_test_keys = arith.constant {tiles * bn} : i32\n", source)
    return source, bn


def loops(text):
    result = []
    for match in re.finditer(rf"scf.for ({SSA}) = ({SSA}) to ({SSA}) step ({SSA})" + r"[^\n]*\{\n", text):
        start = match.end() - 2
        depth = 1
        end = start + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        line_end = text.find("\n", end)
        result.append(dict(iv=match[1], lower=match[2], upper=match[3], step=match[4],
                           start=match.start(), body=text[start + 1:end - 1],
                           suffix=text[end:line_end]))
    return result


class Arithmetic(Coordinates):
    def __init__(self, text, bindings):
        self.bindings = bindings
        self.iv = None
        self.defs = {}
        self.names = {}
        for line in text.splitlines():
            match = re.match(rf"\s*({SSA}) = arith\.(\w+) (.*)", line)
            if not match:
                continue
            result, op, rhs = match.groups()
            self.defs[result] = (op, rhs)
            tag = re.search(r'hivm.cvp_query.unroll_coordinate = "(\w+)"', rhs)
            if tag:
                self.names[tag[1]] = result

    def evaluate(self, name, tick=None):
        if name in self.bindings:
            return self.bindings[name]
        op, rhs = self.defs[name]
        if op == "select":
            condition, yes, no = [self.evaluate(v) for v in re.findall(SSA, rhs)]
            return yes if condition else no
        if op == "cmpi" and not rhs.startswith("eq,"):
            lhs, right = [self.evaluate(v) for v in re.findall(SSA, rhs)]
            predicate = rhs.split(",", 1)[0]
            return {"ne": lhs != right, "slt": lhs < right, "sle": lhs <= right,
                    "sgt": lhs > right, "sge": lhs >= right}[predicate]
        if op in ("minui", "minsi", "floordivsi"):
            lhs, right = [self.evaluate(v) for v in re.findall(SSA, rhs)]
            if op == "minui":
                assert lhs >= 0 and right >= 0
            return lhs // right if op == "floordivsi" else min(lhs, right)
        return super().evaluate(name, tick)


def check_emitted(text, tiles, bn, factor):
    assert "scope.scope" not in text
    assert 'hivm.cvp_attention.applied = "unroll"' in text
    assert 'hivm.cvp_attention.applied = "skew"' not in text
    assert "hivm.cvp_query.stream" not in text
    assert "continuous = false" in text
    assert f"emitted_lanes = {factor} : i32" in text
    assert f"workspace_depth = {factor} : i32" in text
    for role in ("qk", "p32", "pv", "output"):
        assert f'hivm.cvp_attention.role = "{role}"' in text
    all_loops = loops(text)
    outer = [loop for loop in all_loops if "hivm.cvp_query.unroll_domain" in loop["suffix"]]
    stages = [loop for loop in all_loops if "hivm.cvp_query.unroll_stage" in loop["suffix"]]
    assert len(outer) == 1 and len(stages) == 4, (len(outer), len(stages))
    outer = outer[0]
    for stage in stages:
        visited = []
        prefix = text[:stage["start"]]
        for base in range(0, tiles * bn, factor * bn):
            scalar = Arithmetic(prefix + stage["body"], {outer["iv"]: base})
            assert scalar.evaluate(outer["lower"]) == 0
            assert scalar.evaluate(outer["upper"]) == tiles * bn
            assert scalar.evaluate(outer["step"]) == factor * bn
            low = scalar.evaluate(stage["lower"])
            high = scalar.evaluate(stage["upper"])
            step = scalar.evaluate(stage["step"])
            assert low == 0 and step == 1 and 0 < high <= factor
            assert scalar.coordinate("valid_lanes") == high
            for lane in range(low, high, step):
                scalar.bindings[stage["iv"]] = lane
                key = scalar.coordinate("logical_key")
                assert key == base + lane * bn
                assert scalar.coordinate("first_key") == (key == 0)
                assert scalar.coordinate("last_key") == (key == (tiles - 1) * bn)
                visited.append(key)
        assert visited == list(range(0, tiles * bn, bn)), visited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    positives = negatives = 0
    for filename in ("recognize-hd64.mlir", "recognize-hd128.mlir"):
        original = (args.inputs / filename).read_text()
        for tiles in (2, 3, 4, 5, 6, 8):
            source, bn = key_domain(original, tiles)
            prepared = run(args.opt, source, "--hivm-cvp-attention-prepare=emit-remarks=false").stdout
            # This is the existing default U1 no-op, independently of opt-in
            # annotated U1 below. It must remain unchanged for other callers.
            ordinary = run(args.opt, prepared, "--cv-pipelining=pipeline-mode=unroll set-depth-in-unroll-mode=1").stdout
            assert ordinary == run(args.opt, prepared).stdout
            for factor in (1, 2, 4):
                options = f"--cv-pipelining=pipeline-mode=unroll set-depth-in-unroll-mode={factor} annotate-attention=true enable-lazy-loading=true"
                output = run(args.opt, prepared, options, "--lower-affine").stdout
                check_emitted(output, tiles, bn, factor)
                baseline = run(args.opt, output).stdout
                rejected = run(args.opt, output, "--hivm-cvp-query-pipeline", succeeds=False)
                assert "continuous query state and cross-stage forwarding are not implemented" in rejected.stderr
                optional = run(args.opt, output, "--hivm-cvp-query-pipeline=require-match=false").stdout
                assert optional == baseline
                positives += 1
        for kind in ("one-key", "bad-summary", "bad-role", "unsupported-factor"):
            source, bn = key_domain(original, 1 if kind == "one-key" else 6)
            prepared = run(args.opt, source, "--hivm-cvp-attention-prepare=emit-remarks=false").stdout
            if kind == "bad-summary":
                prepared = replace_once(r"key_count = \d+ : i64", f"key_count = {7 * bn} : i64", prepared)
            elif kind == "bad-role":
                prepared = replace_once(r'hivm.cvp_attention.role = "pv"',
                                        'hivm.cvp_attention.role = "pv-invalid"', prepared)
            baseline = run(args.opt, prepared).stdout
            for factor in ((3,) if kind == "unsupported-factor" else (1, 2, 4)):
                output = run(args.opt, prepared,
                             f"--cv-pipelining=pipeline-mode=unroll set-depth-in-unroll-mode={factor} annotate-attention=true enable-lazy-loading=true").stdout
                assert output == baseline, (filename, kind, factor)
                negatives += 1
    print(f"ordinary-Unroll emitter cases: {positives}; provenance rollback cases: {negatives}; continuous-Unroll support: rejected")


if __name__ == "__main__":
    main()

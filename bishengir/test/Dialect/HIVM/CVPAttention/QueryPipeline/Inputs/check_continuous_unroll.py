"""Decode actual direct-Unroll scalar IR against an independent query oracle.

Calls model Q load, QK/P/PV/O and final writes; no hardware, DMA or numerical
attention is executed. The oracle owns state by logical query, independently of
the physical even/odd state carried by the compiler fixture.
"""
import argparse
from pathlib import Path
import re
from check_query_domains import run
from check_query_unroll import key_domain

SSA = r"%[\w.$-]+(?:#\d+)?"


def fixture(keys, start, upper, stride, factor, core="MIX", name="test", bad=None):
    queries = len(range(start, upper, stride))
    def scope(body, index, results=""):
        prefix = f"%{results.split(':')[0]}:{results.split(':')[1]} = " if results else ""
        types = " : () -> (" + ", ".join(["i64"] * int(results.split(':')[1])) + ")" if results else " : () -> ()"
        core_type = "CUBE" if index % 2 else "VECTOR"
        marked = factor if bad != "factor" or index != 0 else 3
        return (f"    {prefix}scope.scope{types} {{\n{body}\n    }} "
                f"{{hivm.loop_core_type = #hivm.tcore_type<{core_type}>, "
                f"hivm.max_preload_num = 4 : i32, hivm.preload_num = {index} : i32, "
                f"hivm.cvp_query.unroll_factor = {marked} : i32, no_inline}}\n")
    qk = scope("""      scf.if %first {
        func.call @qload(%task, %slot) : (i64, i64) -> ()
      }
      func.call @qk(%task, %key) : (i64, i64) -> ()
      scope.return""", 3)
    p = scope("""      %sum = arith.select %odd, %s1, %s0 : i64
      %max = arith.select %odd, %m1, %m0 : i64
      %oldsum = arith.select %first, %zero, %sum : i64
      %oldmax = arith.select %first, %minusone, %max : i64
      %weight = arith.addi %key, %one : i64
      %newsum = arith.addi %oldsum, %weight : i64
      %newmax = arith.maxsi %oldmax, %weight : i64
      func.call @p(%task, %key, %oldsum, %oldmax, %newsum, %newmax) : (i64, i64, i64, i64, i64, i64) -> ()
      %sumeven = arith.select %odd, %s0, %newsum : i64
      %sumodd = arith.select %odd, %newsum, %s1 : i64
      %maxeven = arith.select %odd, %m0, %newmax : i64
      %maxodd = arith.select %odd, %newmax, %m1 : i64
      scope.return %sumeven, %sumodd, %maxeven, %maxodd : i64, i64, i64, i64""", 2, "pstate:4")
    pv = scope("""      func.call @pv(%task, %key) : (i64, i64) -> ()
      scope.return""", 1)
    o = scope("""      %oldout = arith.select %first, %zero, %out : i64
      %weight = arith.addi %key, %one : i64
      %newout = arith.addi %oldout, %weight : i64
      func.call @o(%task, %key, %oldout, %newout) : (i64, i64, i64, i64) -> ()
      %finalcount = scf.if %last -> (i64) {
        %finalsum = arith.select %odd, %pstate#1, %pstate#0 : i64
        %finalmax = arith.select %odd, %pstate#3, %pstate#2 : i64
        func.call @output(%task, %newout, %finalsum, %finalmax) : (i64, i64, i64, i64) -> ()
        %added = arith.addi %outputs, %one : i64
        scf.yield %added : i64
      } else {
        scf.yield %outputs : i64
      }
      scope.return %newout, %finalcount : i64, i64""", 0, "ostate:2")
    if core == "AIC":
        stages = qk + pv
        yields = "%out, %s0, %s1, %m0, %m1, %outputs"
    else:
        stages = (qk if core == "MIX" else "") + p + (pv if core == "MIX" else "") + o
        yields = "%ostate#0, %pstate#0, %pstate#1, %pstate#2, %pstate#3, %ostate#1"
    if bad == "missing-stage":
        stages = stages.replace(pv, "")
    if bad == "outside-effect":
        stages += "    func.call @unknown() : () -> ()\n"
    certified = "true" if bad == "certificate" else "false"
    return f"""func.func @{name}() attributes {{hivm.func_core_type = #hivm.func_core_type<{core}>}} {{
  %zero = arith.constant 0 : i64
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %minusone = arith.constant -1 : i64
  %keys = arith.constant {keys} : i64
  %lastkey = arith.constant {keys - 1} : i64
  %start = arith.constant {start} : i64
  %stride = arith.constant {stride} : i64
  %end = arith.constant {queries * keys} : i64
  %result:6 = scf.for %tick = %zero to %end step %one iter_args(%out = %zero, %s0 = %zero, %s1 = %zero, %m0 = %minusone, %m1 = %minusone, %outputs = %zero) -> (i64, i64, i64, i64, i64, i64) : i64 {{
    %ordinal = arith.divsi %tick, %keys : i64
    %key = arith.remsi %tick, %keys : i64
    %offset = arith.muli %ordinal, %stride : i64
    %task = arith.addi %start, %offset : i64
    %slot = arith.remsi %ordinal, %two : i64
    %odd = arith.cmpi eq, %slot, %one : i64
    %first = arith.cmpi eq, %key, %zero : i64
    %last = arith.cmpi eq, %key, %lastkey : i64
{stages}    scf.yield {yields} : i64, i64, i64, i64, i64, i64
  }} {{hivm.cvp_query.stream = {{pipeline_mode = "unroll", key_unroll = {factor} : i32, key_count = {keys} : i64, key_step = 1 : i64, query_upper = {upper} : i64, query_step = {stride} : i64, physical_lifetimes_verified = {certified}}}}}
  func.call @finish(%result#5) : (i64) -> ()
  return
}}
"""


DECLARATIONS = """func.func private @qload(i64, i64)
func.func private @qk(i64, i64)
func.func private @p(i64, i64, i64, i64, i64, i64)
func.func private @pv(i64, i64)
func.func private @o(i64, i64, i64, i64)
func.func private @output(i64, i64, i64, i64)
func.func private @finish(i64)
func.func private @unknown()
"""


def module(*functions):
    return "module {\n" + DECLARATIONS + "".join(functions) + "}\n"


class Decoder:
    def __init__(self, text):
        lines = text.splitlines()
        begin = next(i for i, line in enumerate(lines) if "func.func @test(" in line)
        self.nodes, _ = self.parse(lines, begin + 1)
        self.calls = []

    def parse(self, lines, index):
        nodes = []
        while index < len(lines):
            line = lines[index].strip()
            if line.startswith("}"):
                return nodes, index
            if not line or line.startswith("//"):
                index += 1
                continue
            if "scf.for " in line or "scf.if " in line or "scope.scope" in line:
                body, end = self.parse(lines, index + 1)
                other = []
                if " else {" in lines[end]:
                    other, end = self.parse(lines, end + 1)
                nodes.append((line, body, other))
                index = end + 1
            else:
                nodes.append((line, None, None))
                index += 1
        raise AssertionError("unterminated emitted region")

    @staticmethod
    def bind(line, values, env):
        match = re.match(r"(%[\w.$-]+)(?::(\d+))? = ", line)
        if not match:
            assert not values
            return
        count = int(match[2] or 1)
        assert len(values) == count, (line, values)
        for index, value in enumerate(values):
            env[match[1] + (f"#{index}" if count > 1 else "")] = value

    def execute(self, nodes, env):
        for line, body, other in nodes:
            if "scf.for " in line:
                match = re.search(r"scf.for (" + SSA + r") = (" + SSA + r") to (" + SSA + r") step (" + SSA + r")", line)
                assert match, line
                iv, lower, upper, step = match.groups()
                carried = re.search(r"iter_args\((.*?)\)", line)
                bindings = [re.findall(SSA, item) for item in carried[1].split(",")] if carried else []
                state = [env[value] for _, value in bindings]
                for tick in range(env[lower], env[upper], env[step]):
                    nested = dict(env)
                    nested[iv] = tick
                    nested.update({argument: value for (argument, _), value in zip(bindings, state)})
                    state = self.execute(body, nested)
                self.bind(line, state, env)
            elif "scf.if " in line:
                condition = re.search(r"scf.if (" + SSA + ")", line)[1]
                values = self.execute(body if env[condition] else other, dict(env))
                self.bind(line, values, env)
            elif "scope.scope" in line:
                self.bind(line, self.execute(body, dict(env)), env)
            elif line.startswith(("scf.yield", "scope.return", "return", "func.return")):
                return [env[value] for value in re.findall(SSA, line.split(" : ", 1)[0])]
            elif re.search(r"(?:^|\s)(?:func\.)?call ", line):
                match = re.search(r"(?:func\.)?call @(\w+)\((.*?)\)", line)
                self.calls.append((match[1], tuple(env[v] for v in re.findall(SSA, match[2]))))
            elif "arith." in line:
                match = re.search(r"arith\.(\w+) (.*)", line)
                op, rhs = match.groups()
                values = [env[value] for value in re.findall(SSA, rhs)]
                if op == "constant":
                    literal = rhs.split()[0]
                    value = {"true": True, "false": False}.get(literal)
                    if value is None:
                        value = int(literal)
                elif op == "select":
                    value = values[1] if values[0] else values[2]
                elif op == "cmpi":
                    a, b = values
                    value = {"eq": a == b, "ne": a != b, "slt": a < b,
                             "sle": a <= b, "sgt": a > b, "sge": a >= b}[rhs.split(",")[0]]
                elif op in ("index_cast", "extsi", "trunci"):
                    value = values[0]
                else:
                    a, b = values
                    if op == "addi": value = a + b
                    elif op == "subi": value = a - b
                    elif op == "muli": value = a * b
                    elif op == "andi": value = a & b
                    elif op == "ori": value = a | b
                    elif op == "maxsi": value = max(a, b)
                    elif op == "minsi": value = min(a, b)
                    elif op in ("divsi", "remsi"):
                        quotient = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
                        value = quotient if op == "divsi" else a - quotient * b
                    else: raise AssertionError((op, line))
                self.bind(line, [value], env)
            else:
                raise AssertionError("unsupported emitted operation: " + line)
        return []

    def run(self):
        self.execute(self.nodes, {})
        return self.calls


def expected(keys, start, upper, stride, core):
    tasks = list(range(start, upper, stride))
    sums = {task: 0 for task in tasks}
    maxima = {task: -1 for task in tasks}
    outputs = {task: 0 for task in tasks}
    calls = []
    for issue in range(len(tasks) * keys + 4):
        for stage, lag in (("qk", 0), ("p", 1), ("pv", 2), ("o", 3)):
            tile = issue - lag
            if not 0 <= tile < len(tasks) * keys:
                continue
            ordinal, key = divmod(tile, keys)
            task = tasks[ordinal]
            active = core == "MIX" or (core == "AIC") == (stage in ("qk", "pv"))
            if stage == "qk":
                if active and key == 0: calls.append(("qload", (task, ordinal % 2)))
                record = ("qk", (task, key))
            elif stage == "p":
                before, oldmax = sums[task], maxima[task]
                sums[task] += key + 1
                maxima[task] = max(maxima[task], key + 1)
                record = ("p", (task, key, before, oldmax, sums[task], maxima[task]))
            elif stage == "pv":
                record = ("pv", (task, key))
            else:
                before = outputs[task]
                outputs[task] += key + 1
                record = ("o", (task, key, before, outputs[task]))
            if active: calls.append(record)
            if stage == "o" and key == keys - 1 and active:
                calls.append(("output", (task, outputs[task], sums[task], maxima[task])))
    calls.append(("finish", (0 if core == "AIC" else len(tasks),)))
    return calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    args = parser.parse_args()
    decoded = 0
    domains = ((9, 9, 3), (12, 9, 3), (7, 8, 3), (2, 6, 3), (1, 12, 3), (0, 11, 4), (3, 11, 4))
    for keys in (2, 3, 4, 5, 6, 8):
        for start, upper, stride in domains:
            for factor in (1, 2, 4):
                for core in ("MIX", "AIC", "AIV"):
                    source = module(fixture(keys, start, upper, stride, factor, core))
                    output = run(args.opt, source, "--create-preload").stdout
                    assert "scope.scope" not in output
                    assert 'hivm.cvp_attention.applied = "skew"' not in output
                    assert Decoder(output).run() == expected(keys, start, upper, stride, core), (keys, start, upper, factor, core)
                    decoded += 1
    for bad in ("factor", "missing-stage", "outside-effect", "certificate"):
        source = module(fixture(6, 0, 11, 4, 4, name="good"), fixture(6, 0, 11, 4, 4, name="bad", bad=bad))
        rejected = run(args.opt, source, "--create-preload", "--mlir-disable-threading",
                       "--mlir-print-ir-after-failure", succeeds=False)
        assert "IR Dump After" in rejected.stderr
        assert "hivm.cvp_query.unroll_emitted" not in rejected.stderr
    prepared_paths = ("recognize-hd64.mlir", "recognize-hd128.mlir")
    prepared_cases = 0
    for filename in prepared_paths:
        captured = (args.inputs / filename).read_text()
        for keys in (2, 3, 4, 5, 6, 8):
            source, _ = key_domain(captured, keys)
            prepared = run(args.opt, source, "--hivm-cvp-attention-prepare=emit-remarks=false").stdout
            common = "--cv-pipelining=pipeline-mode=unroll set-depth-in-unroll-mode=2 annotate-attention=true enable-lazy-loading=true"
            disabled = run(args.opt, prepared, common).stdout
            for factor in (1, 2, 4):
                explicit_off = run(args.opt, prepared, common + f" continuous-query=false query-key-unroll={factor}").stdout
                assert explicit_off == disabled
                output = run(args.opt, prepared, common + f" continuous-query=true query-key-unroll={factor}",
                             "--hivm-verify-cvp-attention=emit-remarks=false retain-provenance=true",
                             "--hivm-cvp-attention-layout=require-match=true allow-fp32-sum-reassociation=true",
                             "--hivm-cvp-query-pipeline").stdout
                assert 'pipeline_mode = "unroll"' in output
                assert f"key_unroll = {factor} : i32" in output
                assert "hivm.cvp_query.ownership_pending" in output
                prepared_cases += 1
    print(f"decoded continuous cases: {decoded}; transactional rejections: 4; actual query preparations and disabled controls: {prepared_cases}")


if __name__ == "__main__":
    main()

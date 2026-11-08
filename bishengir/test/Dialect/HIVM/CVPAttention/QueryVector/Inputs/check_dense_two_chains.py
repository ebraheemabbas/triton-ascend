"""Audit the actual two-chain key reads, numerical carries and packed writes."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--opt', required=True)
parser.add_argument('--fixtures', type=Path, required=True)
parser.add_argument('--artifacts', type=Path)
args = parser.parse_args()
if args.artifacts:
    args.artifacts.mkdir(parents=True, exist_ok=False)
counter = 0
unexpected_negative_cases = []
prefix = '--hivm-cvp-attention-query-vector='
base = ('enable-score-rematerialization=true fuse-probability=true '
        'fuse-state=true fuse-maximum=true allow-sum-reassociation=true')
two = base + ' enable-dense-p=true reduction-chains=2'
token = r'%[\w.$]+(?:#\d+)?'


def run(source, options, expected=None):
    global counter
    command = [args.opt, *options, '-verify-each']
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if args.artifacts:
        stem = f'{counter:03d}'
        for suffix, value in [('input.mlir', source), ('output.mlir', result.stdout),
                              ('stderr.txt', result.stderr)]:
            (args.artifacts / (stem + '.' + suffix)).write_text(value)
        (args.artifacts / (stem + '.json')).write_text(json.dumps(
            dict(argv=command, exit=result.returncode, expected_error=expected), indent=2) + '\n')
    counter += 1
    if expected is not None:
        if not (result.returncode != 0 and expected in result.stderr):
            unexpected_negative_cases.append(dict(case=counter-1, exit=result.returncode,
                expected_error=expected, input_sha256=hashlib.sha256(source.encode()).hexdigest(), stderr=result.stderr))
        return None
    assert result.returncode == 0, result.stderr
    return result.stdout


def functions(text):
    found = list(re.finditer(r'^  func.func [^\n]+', text, re.M))
    return {re.search(r'@([^ (]+)', match[0])[1]:
            text[match.start():found[i + 1].start() if i + 1 < len(found) else len(text)]
            for i, match in enumerate(found)}


def scope(body):
    found = []
    for match in re.finditer(r'^[ \t]*(?:%\w+(?::\d+)? = )?scope.scope [^\n]*\{\n', body, re.M):
        end, depth = match.end(), 1
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        tail = body[end:body.find('\n', end)]
        if '#hivm.tcore_type<VECTOR>' in tail and 'hivm.preload_num = 2 : i32' in tail:
            found.append(body[match.end():end - 1])
    assert len(found) == 1
    return found[0]


def roles(text):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    producer = [name for name in re.findall(r'func.call @([^ (]+)', scope(fs[callers[0]]))
                if 'arith.truncf ' in fs[name] and 'scf.for ' in fs[name]]
    assert len(producer) == 1
    return fs, callers[0], producer[0]


def loops(body):
    found = []
    for match in re.finditer(r'(%[\w.$]+)(?::(\d+))? = scf.for ([^\n]+)\{\n', body):
        end, depth = match.end(), 1
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        found.append(dict(value=match[1], results=int(match[2] or 1), header=match[3],
                          body=body[match.end():end - 1], tail=body[end:body.find('\n', end)],
                          start=match.start(), end=end))
    return found


def definitions(text):
    return dict(re.findall(r'(' + token + r') = ([^\n]+)', text))


def validate(control, candidate, keys):
    old, caller, producer = roles(control)
    new, new_caller, new_producer = roles(candidate)
    assert (caller, producer) == (new_caller, new_producer)
    assert 'hivm.cvp_attention.dense_two_chain_loops = 2 : i64' in candidate
    assert scope(old[caller]) == scope(new[caller]), 'P publication or caller changed'
    protected = lambda text: [line.strip() for line in text.splitlines()
                              if 'hivm.hir.anchor' in line or 'hivm.hir.sync_block' in line]
    assert protected(control) == protected(candidate)
    for name in old:
        if name != producer:
            assert old[name] == new[name], name
    body = new[producer]
    assert old[producer].splitlines()[0] == body.splitlines()[0]
    assert f'tensor<8x{keys // 2}x16xf16>' in body.splitlines()[0]
    assert 'padding_required = true' in body.splitlines()[0]
    regions = loops(body)
    assert len(regions) == 2
    top = body
    for region in reversed(regions):
        top = top[:region['start']] + '\n' + top[region['end']:]
    outer = definitions(top)
    for region in regions:
        maximum = region['results'] == 2
        assert region['results'] == (2 if maximum else 3)
        assert 'hivm.cvp_attention.reduction_chains = 2 : i64' in region['tail']
        groups = 4 if maximum else 2
        assert f'hivm.cvp_attention.reduction_key_groups = {groups} : i64' in region['tail']
        bound = re.match(r'(' + token + r') = (' + token + r') to (' + token + r') step (' + token + r') iter_args', region['header'])
        assert bound, region['header']
        induction = bound[1]
        local = definitions(region['body'])
        defs = {**outer, **local}

        def index(value, iteration, visiting=None):
            if value == induction:
                return iteration
            visiting = set() if visiting is None else visiting
            assert value not in visiting, value
            expression = defs[value]
            constant = re.fullmatch(r'arith.constant (-?\d+) : index', expression)
            if constant:
                return int(constant[1])
            add = re.match(r'arith.addi (' + token + r'), (' + token + r') : index', expression)
            assert add, expression
            return sum(index(x, iteration, visiting | {value}) for x in add.groups())

        assert index(bound[2], 0) == 0 and index(bound[3], 0) == keys // 4 and index(bound[4], 0) == 1
        carries = re.findall(r'(' + token + r') = (' + token + r')', region['header'].split('iter_args(', 1)[1].split(') ->', 1)[0])
        carry_index = 0 if maximum else 1
        assert len(carries) == carry_index + 2
        assert carries[carry_index][1] == carries[carry_index + 1][1]
        seed = outer[carries[carry_index][1]]
        if maximum:
            assert 'dense<0xFF800000>' in seed and 'vector<64xf32>' in seed
        else:
            zero = re.search(r'dense<([^>]+)> : vector<64xf32>', seed)
            assert zero and float(zero[1]) == 0 and math.copysign(1, float(zero[1])) > 0
        reads = {}
        score_views = {}
        for value, expression in local.items():
            match = re.match(r'tensor.extract_slice (' + token + r')\[(' + token + r'), 0\] \[1, 64\] \[1, 1\] : tensor<' + str(keys) + r'x64xf32>', expression)
            if match:
                score_views[value] = match[2]
        assert len(score_views) == 4, score_views
        for value, expression in local.items():
            match = re.match(r'vector.transfer_read (' + token + r')\[', expression)
            if match and match[1] in score_views:
                reads[value] = score_views[match[1]]
        assert len(reads) == 4
        observed = Counter(index(key, i) for i in range(keys // 4) for key in reads.values())
        assert observed == Counter({key: 1 for key in range(keys)}), observed

        def dependencies(value, kind, visiting=None):
            if kind == 'key' and value in reads:
                return {index(reads[value], 0)}
            if kind == 'carry' and value in {x[0] for x in carries[carry_index:]}:
                return {value}
            if value not in local:
                return set()
            visiting = set() if visiting is None else visiting
            assert value not in visiting
            expression = local[value]
            if kind == 'key':
                assert not expression.startswith('arith.truncf '), 'FP32 sum uses rounded P'
            return set().union(*(dependencies(x, kind, visiting | {value})
                                 for x in re.findall(token, expression)))

        yield_line = next(line for line in region['body'].splitlines() if 'scf.yield ' in line)
        yielded = re.findall(token, yield_line.split(' : ', 1)[0])
        assert len(yielded) == carry_index + 2
        for chain in (0, 1):
            result = yielded[carry_index + chain]
            assert dependencies(result, 'carry') == {carries[carry_index + chain][0]}
            assert dependencies(result, 'key') == {chain * (keys // 4), chain * (keys // 4) + keys // 2}
        refs = [region['value'] + '#' + str(carry_index + x) for x in (0, 1)]
        merge_op = 'arith.maximumf' if maximum else 'arith.addf'
        merge = [expression for expression in outer.values() if expression.startswith(merge_op + ' ')
                 and re.findall(token, expression) == refs]
        assert len(merge) == 1, merge
        if not maximum:
            assert region['body'].count('arith.truncf ') == 4
            assert region['body'].count('ave.hir.vdintlv ') == 2
            assert len(re.findall(r'ave.hir.vdintlv[^\n]*DENSE', region['body'])) == 2
            packed_views = []
            for value, expression in local.items():
                match = re.match(r'tensor.extract_slice (' + token + r')\[0, (' + token + r'), 0\] \[8, 1, 16\]', expression)
                if match:
                    packed_views.append((value, match[1], match[2]))
            assert len(packed_views) == 2
            written = Counter(index(key, i) for i in range(keys // 4) for _, _, key in packed_views)
            assert written == Counter({key: 1 for key in range(keys // 2)})
            inserts = re.findall(r'(' + token + r') = tensor.insert_slice ' + token + r' into (' + token + r')\[0,', region['body'])
            assert len(inserts) == 2
            assert packed_views[0][1] == inserts[0][1] == carries[0][0]
            assert packed_views[1][1] == inserts[1][1] == inserts[0][0]
            assert yielded[0] == inserts[1][0]


def maximum(x, y):
    if math.isnan(x) or math.isnan(y):
        return float('nan')
    if x == y == 0:
        return 0.0 if math.copysign(1, x) > 0 or math.copysign(1, y) > 0 else -0.0
    return x if x > y else y


for values in itertools.product([float('-inf'), -1.0, -0.0, 0.0, 1.0, float('inf'), float('nan')], repeat=4):
    sequential = float('-inf')
    for value in values:
        sequential = maximum(sequential, value)
    interleaved = maximum(maximum(values[0], values[2]), maximum(values[1], values[3]))
    assert (math.isnan(sequential) and math.isnan(interleaved)) or struct.pack('f', sequential) == struct.pack('f', interleaved)

manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    source = run(path.read_text(), [])
    run(source, [prefix + 'reduction-chains=2'], 'CVP two-chain reduction requires')
    run(source, [prefix + base + ' enable-dense-p=true reduction-chains=2 allow-sum-reassociation=false'], 'explicit FP32 sum reassociation')
    if item['arm'] == 'baseline':
        run(source, [prefix + two], 'CVP dense P requires a closed full-key P producer')
        continue
    for scaling in (False, True):
        flags = base + ' enable-max-scaling=' + str(scaling).lower()
        control = run(source, [prefix + flags + ' enable-dense-p=true reduction-chains=1 require-match=true'])
        candidate = run(source, [prefix + flags + ' enable-dense-p=true reduction-chains=2 require-match=true'])
        validate(control, candidate, item['BN'])
    parent = run(source, [prefix + base + ' require-match=true'])
    fs, caller, producer = roles(parent)
    body = fs[producer]
    # Unsupported already-packed input rejects instead of silently running mode1.
    run(control, [prefix + two], 'CVP dense P requires a closed full-key P producer')
    max_loop, p_loop = loops(body)
    for loop in (max_loop, p_loop):
        carry = re.findall(r'(' + token + r') = (' + token + r')', loop['header'].split('iter_args(', 1)[1].split(') ->', 1)[0])[-1][0]
        old_body = loop['body']
        extra = '      %test_feedback = arith.negf ' + carry + ' : vector<64xf32>\n'
        position = old_body.index('      scf.yield ')
        changed = body.replace(old_body, old_body[:position] + extra + old_body[position:], 1)
        expected = 'one closed full-key maximum' if loop is max_loop else 'closed full-key P producer'
        run(parent.replace(body, changed, 1), [prefix + two], expected)
    changed = body.replace('dense<0xFF800000>', 'dense<0.000000e+00>', 1)
    assert changed != body
    run(parent.replace(body, changed, 1), [prefix + two], 'one closed full-key maximum')
    # Fusion can introduce several equal zero constants. Mutate the actual
    # sum initializer named by the P loop, not the first textual zero.
    p_carries = re.findall(r'(' + token + r') = (' + token + r')',
                          p_loop['header'].split('iter_args(', 1)[1].split(') ->', 1)[0])
    sum_seed = p_carries[-1][1]
    seed_line = re.search(r'^    ' + re.escape(sum_seed) + r' = arith.constant dense<0[.]000000e[+]00> : vector<64xf32>$', body, re.M)
    assert seed_line, (sum_seed, body)
    changed = body.replace(seed_line[0], seed_line[0].replace('dense<0.000000e+00>', 'dense<-0.000000e+00>'), 1)
    assert changed != body
    run(parent.replace(body, changed, 1), [prefix + two], 'closed full-key P producer')
    read = next(line for line in p_loop['body'].splitlines() if 'vector.transfer_read ' in line and 'tensor<1x64xf32>' in line)
    changed = body.replace(read, '      %test_mask = arith.constant dense<true> : vector<1x64xi1>\n' + read.replace(' {in_bounds', ', %test_mask {in_bounds', 1), 1)
    run(parent.replace(body, changed, 1), [prefix + two], 'closed full-key P producer')
    caller_body = scope(fs[caller])
    producer_calls = re.findall(r'^\s*(' + token + r')(?::(\d+))? = func.call @' + re.escape(producer) + r'\(', caller_body, re.M)
    assert len(producer_calls) == 1
    call_value, arity = producer_calls[0]
    p_value = call_value + '#0' if int(arity or 1) > 1 else call_value
    publications = [value for value, source in re.findall(r'(' + token + r') = tensor.expand_shape (' + token + r') ', caller_body)
                    if source == p_value]
    assert len(publications) == 1
    copies = [line for line in caller_body.splitlines() if 'hivm.hir.copy ' in line
              and re.search(r'ins\(\s*' + re.escape(publications[0]) + r'\s*:', line)]
    assert len(copies) == 1 and 'xf16>' in copies[0]
    copy = copies[0]
    assert fs[caller].count(copy) == 1 and parent.count(fs[caller]) == 1
    changed_caller = fs[caller].replace(copy, copy + '\n' + copy, 1)
    run(parent.replace(fs[caller], changed_caller, 1), [prefix + two], 'closed full-key P producer')
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', source)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    control = run(renamed, [prefix + base + ' enable-dense-p=true reduction-chains=1 require-match=true'])
    candidate = run(renamed, [prefix + two + ' require-match=true'])
    validate(control, candidate, item['BN'])
assert not unexpected_negative_cases, json.dumps(unexpected_negative_cases, indent=2)
print('Dense two-chain IR passed: actual key/read/store coverage, independent FP32 carries, neutral seeds/final merges, original packing/publication/anchors, rejection guards and maximum NaN/zero/infinity semantics; native/NPU remain unqualified')

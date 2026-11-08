"""Check independent maximum/sum chains and their numerical policy."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--opt', required=True)
parser.add_argument('--fixtures', type=Path, required=True)
parser.add_argument('--artifacts', type=Path)
args = parser.parse_args()
if args.artifacts:
    args.artifacts.mkdir(parents=True, exist_ok=False)
counter = 0
prefix = '--hivm-cvp-attention-query-vector='
remat = 'enable-score-rematerialization=true'
unsupported = 'CVP score rematerialization requires a closed key-major'


import itertools, math, struct

def run(source, options, expected=None):
    global counter
    command = [args.opt, *options, '-verify-each']
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if args.artifacts:
        stem = f'{counter:03d}'
        for suffix, value in [('input.mlir', source), ('output.mlir', result.stdout),
                              ('stderr.txt', result.stderr)]:
            (args.artifacts / (stem + '.' + suffix)).write_text(value)
        (args.artifacts / (stem + '.json')).write_text(json.dumps({
            'argv': command, 'exit': result.returncode, 'expected_error': expected
        }, indent=2) + '\n')
    counter += 1
    if expected is not None:
        assert result.returncode != 0 and expected in result.stderr, result.stderr
        return None
    assert result.returncode == 0, result.stderr
    return result.stdout

def functions(text):
    matches = list(re.finditer(r'^  func.func [^\n]+', text, re.M))
    return {re.search(r'@([^ (]+)', match[0])[1]:
            text[match.start():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
            for i, match in enumerate(matches)}

def scope(body):
    found = []
    for match in re.finditer(r'^[ \t]*(?:%\w+(?::\d+)? = )?scope.scope [^\n]*\{\n', body, re.M):
        end, depth = match.end(), 1
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        suffix = body[end:body.find('\n', end)]
        if '#hivm.tcore_type<VECTOR>' in suffix and 'hivm.preload_num = 2 : i32' in suffix:
            found.append(body[match.end():end - 1])
    assert len(found) == 1
    return found[0]

def protected_lines(text):
    return [line.strip() for line in text.splitlines()
            if 'hivm.hir.sync_block' in line or 'hivm.hir.anchor' in line]

chains = 'reduction-chains=4 allow-sum-reassociation=true'
unsupported = 'CVP four-chain reduction requires closed full-key maximum and FP32 P-sum loops'


def loops(body):
    result = []
    for match in re.finditer(r'(%[\w.$]+)(?::(\d+))? = scf.for ([^\n]+)\{\n', body):
        end, depth = match.end(), 1
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        result.append({'value': match[1], 'results': int(match[2] or 1),
                       'header': match[3], 'body': body[match.end():end - 1],
                       'tail': body[end:body.find('\n', end)],
                       'start': match.start(), 'end': end})
    return result


def roles(text):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    caller = callers[0]
    maximum, total = [], []
    for name in re.findall(r'func.call @([^ (]+)', scope(fs[caller])):
        for loop in loops(fs[name]):
            if 'arith.maximumf ' in loop['body'] and loop['results'] in (1, 4):
                maximum.append((name, loop))
            if 'arith.truncf ' in loop['body'] and 'arith.addf ' in loop['body']:
                total.append((name, loop))
    assert len(maximum) == len(total) == 1
    return fs, caller, maximum[0], total[0]


def validate(before, after, keys):
    fs, caller, maximum, total = roles(before)
    changed, changed_caller, new_maximum, new_total = roles(after)
    assert caller == changed_caller
    assert 'hivm.cvp_attention.four_chain_loops = 2 : i64' in after
    for original, transformed, is_maximum in [(maximum, new_maximum, True), (total, new_total, False)]:
        name, loop = transformed
        assert name == original[0]
        assert loop['results'] == (4 if is_maximum else 5)
        assert 'hivm.cvp_attention.reduction_chains = 4 : i64' in loop['tail']
        constants = {name: int(value) for name, value in re.findall(r'(%[\w.$]+) = arith.constant (-?\d+) : index', changed[name])}
        bounds = re.search(r'(%[\w.$]+) = (%[\w.$]+) to (%[\w.$]+) step (%[\w.$]+)', loop['header'])
        assert constants[bounds[2]] == 0 and constants[bounds[3]] == keys and constants[bounds[4]] == 4
        induction = bounds[1]
        carries = re.findall(r'(%[\w.$]+) = (%[\w.$]+)', loop['header'].split('iter_args(', 1)[1].split(') ->', 1)[0])
        offset = 0 if is_maximum else 1
        operation = 'arith.maximumf' if is_maximum else 'arith.addf'
        reductions = re.findall(r'(%[\w.$]+) = ' + re.escape(operation) + r' (%[\w.$]+), (%[\w.$]+)', loop['body'])
        yielded = re.search(r'scf.yield (.+?) : ', loop['body'])[1].split(', ')
        assert len(reductions) == 4 and len(yielded) == 4 + offset
        for index, (value, lhs, rhs) in enumerate(reductions):
            assert carries[index + offset][0] in (lhs, rhs)
            assert yielded[index + offset] == value
        # The original complete-key walk is still covered once in key order.
        offsets = {induction: 0}
        for value, lhs, rhs in re.findall(r'(%[\w.$]+) = arith.addi (%[\w.$]+), (%[\w.$]+)', loop['body']):
            if lhs == induction and rhs in constants:
                offsets[value] = constants[rhs]
        reads = re.findall(r'tensor.extract_slice %[\w.$]+\[(%[\w.$]+), 0\] \[1, 64\]', loop['body'])
        assert [offsets[value] for value in reads] == [0, 1, 2, 3]
        assert [base + i for base in range(0, keys, 4) for i in range(4)] == list(range(keys))
        suffix = changed[name][loop['end']:]
        pairs = []
        for first in (offset, offset + 2):
            pattern = r'(%[\w.$]+) = ' + re.escape(operation) + ' ' + re.escape(loop['value'] + '#' + str(first)) + ', ' + re.escape(loop['value'] + '#' + str(first + 1))
            match = re.search(pattern, suffix)
            assert match, pattern
            pairs.append(match[1])
        assert re.search(re.escape(operation + ' ' + pairs[0] + ', ' + pairs[1]), suffix)
        if not is_maximum:
            assert loop['body'].count('arith.truncf ') == 4
            assert loop['body'].count('#hfusion.round_mode<rint>') == 4
            inserts = re.findall(r'(%[\w.$]+) = tensor.insert_slice %[\w.$]+ into (%[\w.$]+)', loop['body'])
            assert len(inserts) == 4
            for index, (value, destination) in enumerate(inserts):
                assert destination == (carries[0][0] if index == 0 else inserts[index - 1][0])
            assert yielded[0] == inserts[-1][0]
    assert protected_lines(before) == protected_lines(after)
    assert scope(fs[caller]) == scope(changed[caller]), 'Caller/copy/publication changed'
    for name, body in fs.items():
        if name not in (maximum[0], total[0]):
            assert changed[name] == body, name


def maximum_semantics(x, y):
    if math.isnan(x) or math.isnan(y):
        return float('nan')
    if x == 0 and y == 0:
        return 0.0 if math.copysign(1, x) > 0 or math.copysign(1, y) > 0 else -0.0
    return x if x > y else y


# Independent check of the arith.maximumf reassociation property. This does not
# establish that a target's native implementation honors the IR contract.
values = [float('-inf'), -1.0, -0.0, 0.0, 1.0, float('inf'), float('nan')]
for sequence in itertools.product(values, repeat=4):
    sequential = float('-inf')
    for value in sequence:
        sequential = maximum_semantics(sequential, value)
    balanced = maximum_semantics(maximum_semantics(sequence[0], sequence[1]), maximum_semantics(sequence[2], sequence[3]))
    assert (math.isnan(sequential) and math.isnan(balanced)) or struct.pack('f', sequential) == struct.pack('f', balanced)

manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    before = run(path.read_text(), [])
    assert run(before, ['--hivm-cvp-attention-query-vector']) == before
    assert run(before, [prefix + 'reduction-chains=1']) == before
    run(before, [prefix + 'reduction-chains=4'], 'requires explicit FP32 sum reassociation permission')
    run(before, [prefix + 'reduction-chains=2'], 'CVP two-chain reduction requires dense P, fused maximum/P/state')
    if item['arm'] == 'baseline':
        assert run(before, [prefix + chains]) == before
        run(before, [prefix + chains + ' require-match=true'], unsupported)
        continue
    after = run(before, [prefix + chains + ' require-match=true'])
    validate(before, after, item['BN'])
    assert run(after, [prefix + chains]) == after
    parents = [
        'enable-score-rematerialization=true enable-max-scaling=true',
        'enable-score-rematerialization=true fuse-probability=true fuse-state=true',
        'enable-score-rematerialization=true enable-max-scaling=true fuse-probability=true fuse-state=true fuse-maximum=true',
    ]
    for parent in parents:
        control = run(before, [prefix + parent + ' require-match=true'])
        combined = run(before, [prefix + parent + ' ' + chains + ' require-match=true'])
        validate(control, combined, item['BN'])
        assert run(control, [prefix + chains + ' require-match=true']) == combined
        assert run(combined, [prefix + parent + ' ' + chains]) == combined
    fs, caller, maximum, total = roles(before)
    enabled = prefix + chains + ' require-match=true'
    for function, loop in (maximum, total):
        old = fs[function]
        carry = re.findall(r'(%[\w.$]+) = (%[\w.$]+)', loop['header'].split('iter_args(', 1)[1].split(') ->', 1)[0])[-1][0]
        # Put the extra use before the existing yield, preserving valid IR.
        extra = '      %test_feedback = arith.negf ' + carry + ' : vector<64xf32>\n'
        body = loop['body'];position = body.index('      scf.yield ')
        changed = old.replace(body, body[:position] + extra + body[position:], 1)
        run(before.replace(old, changed, 1), [enabled], unsupported)
        changed = old.replace('arith.constant 1 : index', 'arith.constant 2 : index', 1)
        assert changed != old
        run(before.replace(old, changed, 1), [enabled], unsupported)
    old = fs[total[0]]
    changed = old.replace('dense<0.000000e+00> : vector<64xf32>', 'dense<-0.000000e+00> : vector<64xf32>', 1)
    assert changed != old
    run(before.replace(old, changed, 1), [enabled], unsupported)
    old = fs[maximum[0]]
    changed = old.replace('dense<0xFF800000>', 'dense<0.000000e+00>', 1)
    run(before.replace(old, changed, 1), [enabled], unsupported)
    old = fs[total[0]]
    read_line = next(line for line in old.splitlines() if 'vector.transfer_read ' in line and 'tensor<1x64xf32>' in line)
    masked = old.replace(read_line, '      %test_mask = arith.constant dense<true> : vector<1x64xi1>\n' + read_line.replace(' {in_bounds', ', %test_mask {in_bounds', 1), 1)
    run(before.replace(old, masked, 1), [enabled], unsupported)
    call = next(line for line in scope(fs[caller]).splitlines() if 'func.call @' + total[0] + '(' in line)
    duplicate = re.sub(r'%[\w.$]+:2 = func.call', '%test_shared:2 = func.call', call, count=1)
    run(before.replace(call, call + '\n' + duplicate, 1), [enabled], unsupported)
    sync = next(line.strip() for line in before.splitlines() if 'hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]' in line)
    run(before.replace(call, call + '\n              ' + sync, 1), [enabled], unsupported)
    header = fs[caller].splitlines()[0]
    wrong = header.replace('numerical_policy = "reassociate-fp32-sum"', 'numerical_policy = "strict"')
    assert wrong != header
    run(before.replace(header, wrong, 1), [enabled], unsupported)
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', before)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate(renamed, run(renamed, [enabled]), item['BN'])
print('Four-chain IR: actual independent max/sum carries, merge trees, key/P coverage, '
      'compositions, numerical permission, closed feedback, handoffs and renaming passed; '
      '2401 abstract maximum NaN/zero/infinity patterns passed; native chains remain unverified')

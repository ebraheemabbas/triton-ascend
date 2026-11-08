"""Check key-major score rematerialization alone and with P fusion."""
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

def chain(text):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    caller = callers[0]
    region = scope(fs[caller])
    calls = re.findall(r'func.call @([^ (]+)', region)
    states = [name for name in calls if 'scf.for ' not in fs[name]
              and fs[name].count('math.exp ') == 1 and fs[name].count('arith.addf ') == 1]
    assert len(states) == 1
    state = states[0]
    state_line = next(line for line in region.splitlines() if 'func.call @' + state + '(' in line)
    producers = []
    for name in calls:
        line = next(line for line in region.splitlines() if 'func.call @' + name + '(' in line)
        result = re.search(r'(%\w+):2 = func.call', line)
        if result and fs[name].count('scf.for ') == 1 and result[1] + '#0' in state_line:
            producers.append((name, line, result[1]))
    assert len(producers) == 1
    producer, line, value = producers[0]
    return fs, caller, producer, state, line, state_line, value


def validate(before, after):
    fs, caller, producer, state, *_ = chain(before)
    changed = functions(after)
    assert state not in changed
    assert 'hivm.cvp_attention.sum_materializations_removed = 1 : i64' in after
    old, new = fs[producer], changed[producer]
    assert new.count('scf.for ') == old.count('scf.for ') == 1
    assert new.count('math.exp ') == old.count('math.exp ') + 1
    assert new.count('arith.addf ') == old.count('arith.addf ') + 1
    assert new.count('arith.truncf ') == old.count('arith.truncf ')
    assert new.count('#hfusion.round_mode<rint>') == old.count('#hfusion.round_mode<rint>')
    assert new.count('vector.transfer_write ') == old.count('vector.transfer_write ') + 1
    carry = re.search(r'(%\w+):2 = scf.for ', new)[1] + '#1'
    assert any(carry in line for line in new.splitlines() if 'arith.addf ' in line)
    assert not any(carry in line for line in new.splitlines() if 'vector.transfer_write ' in line)
    assert protected_lines(before) == protected_lines(after)
    old_scope, new_scope = scope(fs[caller]), scope(changed[caller])
    assert old_scope.count('func.call ') == new_scope.count('func.call ') + 1
    old_copies = [line.split(' outs(', 1)[1] for line in old_scope.splitlines() if 'hivm.hir.copy' in line]
    new_copies = [line.split(' outs(', 1)[1] for line in new_scope.splitlines() if 'hivm.hir.copy' in line]
    assert old_copies == new_copies
    for name, body in fs.items():
        if name not in (caller, producer, state):
            assert changed[name] == body, name
    return producer, state


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
enabled = prefix + 'fuse-state=true require-match=true'
unsupported = 'CVP state fusion requires a closed key-major'
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    before = run(path.read_text(), [])
    assert run(before, ['--hivm-cvp-attention-query-vector']) == before
    if item['arm'] == 'baseline':
        assert run(before, [prefix + 'fuse-state=true']) == before
        run(before, [enabled], unsupported)
        continue
    after = run(before, [enabled])
    validate(before, after)
    assert run(after, [prefix + 'fuse-state=true']) == after
    for parent in ['fuse-probability=true',
                   'enable-score-rematerialization=true',
                   'fuse-probability=true enable-score-rematerialization=true enable-max-scaling=true']:
        control = run(before, [prefix + parent + ' require-match=true'])
        combined = run(before, [prefix + parent + ' fuse-state=true require-match=true'])
        validate(control, combined)
        assert run(control, [enabled]) == combined
        assert run(combined, [prefix + parent + ' fuse-state=true']) == combined
    fs, caller, producer, state, line, state_line, value = chain(before)
    extra = (line + '\n              %test_zero = arith.constant 0 : index\n'
             '              %test_sum = tensor.extract ' + value + '#0[%test_zero] : tensor<64xf32>')
    run(before.replace(line, extra, 1), [enabled], unsupported)
    extra_call = re.sub(r'%\w+:2 = func.call', '%test_shared:2 = func.call', state_line, count=1)
    run(before.replace(state_line, state_line + '\n' + extra_call, 1), [enabled], unsupported)
    sync = next(l.strip() for l in before.splitlines()
                if 'hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]' in l)
    run(before.replace(line, line + '\n              ' + sync, 1), [enabled], unsupported)
    # A P use before the state call cannot be dominated by the new fused call.
    early = ('\n              %test_early = tensor.expand_shape ' + value + '#1 [[0], [1, 2], [3]] output_shape [4, '
             + str(item['BN'] // 16) + ', 16, 16] : tensor<4x' + str(item['BN']) + 'x16xf16> into tensor<4x'
             + str(item['BN'] // 16) + 'x16x16xf16>')
    run(before.replace(line, line + early, 1), [enabled], unsupported)
    for function in (producer, state):
        old = fs[function]
        typ = 'tensor<' + str(item['BN']) + 'x64xf32>' if function == producer else 'tensor<64xf32>'
        attributed = old.replace(typ + ',', typ + ' {test.arg = 1 : i64},', 1)
        assert attributed != old
        run(before.replace(old, attributed, 1), [enabled], unsupported)
        call = line if function == producer else state_line
        annotated = call.replace('{hivm.vector_function,', '{test.index = 0 : i64, hivm.vector_function,', 1)
        run(before.replace(call, annotated, 1), [enabled], unsupported)
    for initializer in ['1.000000e+00', '-0.000000e+00']:
        old = fs[producer]
        changed = old.replace('dense<0.000000e+00> : vector<64xf32>', 'dense<' + initializer + '> : vector<64xf32>', 1)
        assert changed != old
        run(before.replace(old, changed, 1), [enabled], unsupported)
    old = fs[state]
    read_line = next(line for line in old.splitlines() if 'vector.transfer_read ' in line)
    masked = old.replace(read_line, '    %test_mask = arith.constant dense<true> : vector<64xi1>\n' +
                          read_line.replace(' {in_bounds', ', %test_mask {in_bounds', 1), 1)
    run(before.replace(old, masked, 1), [enabled], unsupported)
    tagged = re.sub(r'(arith.mulf %\w+, %\w+)( : vector)', r'\1 {test.state_attr = "kept"}\2', old, count=1)
    assert tagged != old
    result = run(before.replace(old, tagged, 1), [enabled])
    assert 'test.state_attr = "kept"' in functions(result)[producer]
    caller_header = fs[caller].splitlines()[0]
    wrong = caller_header.replace('score_layout = "key-major"', 'score_layout = "query-major"')
    run(before.replace(caller_header, wrong, 1), [enabled], unsupported)
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', before)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate(renamed, run(renamed, [enabled]))
print('State fusion: direct FP32 sum forwarding, standalone/combined, default-off, '
      'idempotence, extra readers, shared callees, domination, handoffs, metadata and renaming passed')

"""Check maximum/row-combine fusion with the complete P/state chain."""
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

base = 'fuse-probability=true enable-score-rematerialization=true fuse-state=true'


def chain(text):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    caller = callers[0]
    region = scope(fs[caller])
    calls = re.findall(r'func.call @([^ (]+)', region)
    rows = [name for name in calls if 'hfusion.has_fill' in fs[name].splitlines()[0]]
    maxima = [name for name in calls if fs[name].count('scf.for ') == 1
              and 'arith.maximumf ' in fs[name] and 'math.exp ' not in fs[name]]
    probabilities = [name for name in calls if fs[name].count('scf.for ') == 1
                     and fs[name].count('math.exp ') == 2]
    assert len(rows) == len(maxima) == len(probabilities) == 1
    return fs, caller, maxima[0], rows[0], probabilities[0]


def validate(before, after):
    fs, caller, maximum, row, probability = chain(before)
    changed = functions(after)
    assert maximum not in changed and row not in changed
    assert 'hivm.cvp_attention.tile_max_materializations_removed = 1 : i64' in after
    old, new = fs[probability], changed[probability]
    assert new.count('scf.for ') == 2
    assert new.count('arith.maximumf ') == 2
    assert new.count('math.exp ') == old.count('math.exp ') == 2
    assert new.count('arith.addf ') == old.count('arith.addf ')
    assert new.count('arith.mulf ') == old.count('arith.mulf ') + fs[maximum].count('arith.mulf ')
    assert new.count('arith.truncf ') == old.count('arith.truncf ')
    assert new.count('#hfusion.round_mode<rint>') == old.count('#hfusion.round_mode<rint>')
    # Retain new_m for the outer recurrence, remove tile_m and the dead zero fill.
    assert new.count('vector.transfer_write ') == old.count('vector.transfer_write ') + 1
    assert new.count('vector.transfer_read ') == old.count('vector.transfer_read ')
    assert protected_lines(before) == protected_lines(after)
    old_scope, new_scope = scope(fs[caller]), scope(changed[caller])
    assert old_scope.count('func.call ') == new_scope.count('func.call ') + 2
    old_copies = [line.split(' outs(', 1)[1] for line in old_scope.splitlines() if 'hivm.hir.copy' in line]
    new_copies = [line.split(' outs(', 1)[1] for line in new_scope.splitlines() if 'hivm.hir.copy' in line]
    assert old_copies == new_copies
    call = next(line for line in new_scope.splitlines() if ':4 = func.call @' + probability + '(' in line)
    value = re.search(r'(%\w+):4 = func.call', call)[1]
    assert 'scope.return ' + value + '#2, ' + value + '#3, ' + value + '#1' in new_scope
    for name, body in fs.items():
        if name not in (caller, maximum, row, probability):
            assert changed[name] == body, name


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    before = run(path.read_text(), [])
    assert run(before, ['--hivm-cvp-attention-query-vector']) == before
    run(before, [prefix + 'fuse-maximum=true'], 'CVP maximum fusion requires score rematerialization')
    if item['arm'] == 'baseline':
        assert run(before, [prefix + base + ' fuse-maximum=true']) == before
        continue
    for scaling in (False, True):
        options = base + ' enable-max-scaling=' + str(scaling).lower()
        parent = run(before, [prefix + options + ' require-match=true'])
        result = run(before, [prefix + options + ' fuse-maximum=true require-match=true'])
        validate(parent, result)
        # A separate run uses optional matching because its prerequisites have
        # already been applied; strict mode requires a new match for each flag.
        enabled = prefix + options + ' fuse-maximum=true'
        assert run(parent, [enabled]) == result
        assert run(result, [enabled]) == result
    parent = run(before, [prefix + base + ' require-match=true'])
    enabled = prefix + base + ' fuse-maximum=true'
    fs, caller, maximum, row, probability = chain(parent)
    region = scope(fs[caller])
    maximum_line = next(line for line in region.splitlines() if 'func.call @' + maximum + '(' in line)
    row_line = next(line for line in region.splitlines() if 'func.call @' + row + '(' in line)
    probability_line = next(line for line in region.splitlines() if 'func.call @' + probability + '(' in line)
    tile_value = re.search(r'(%\w+) = func.call', maximum_line)[1]
    row_value = re.search(r'(%\w+):2 = func.call', row_line)[1]

    def unchanged(mutated):
        canonical = run(mutated, [])
        assert run(canonical, [enabled]) == canonical

    extra = (maximum_line + '\n              %test_zero = arith.constant 0 : index\n'
             '              %test_tile = tensor.extract ' + tile_value + '[%test_zero] : tensor<64xf32>')
    unchanged(parent.replace(maximum_line, extra, 1))
    extra = (row_line + '\n              %test_zero = arith.constant 0 : index\n'
             '              %test_fill = tensor.extract ' + row_value + '#0[%test_zero] : tensor<64xf32>')
    unchanged(parent.replace(row_line, extra, 1))
    extra = (row_line + '\n              %test_zero = arith.constant 0 : index\n'
             '              %test_early = tensor.extract ' + row_value + '#1[%test_zero] : tensor<64xf32>')
    unchanged(parent.replace(row_line, extra, 1))
    # A new_m reader after the fused call is supported through its fourth result.
    late = (probability_line + '\n              %test_zero = arith.constant 0 : index\n'
            '              %test_late = tensor.extract ' + row_value + '#1[%test_zero] : tensor<64xf32>')
    late_parent = run(parent.replace(probability_line, late, 1), [])
    validate(late_parent, run(late_parent, [enabled]))
    arguments = re.search(r'func.call @[^ (]+\(([^)]+)\)', row_line)[1].split(', ')
    assert len(arguments) == 4
    mismatched = row_line.replace('(' + arguments[0] + ',', '(' + arguments[2] + ',', 1)
    unchanged(parent.replace(row_line, mismatched, 1))
    for line in (maximum_line, row_line, probability_line):
        annotated = line.replace('{hivm.vector_function,', '{test.index = 0 : i64, hivm.vector_function,', 1)
        unchanged(parent.replace(line, annotated, 1))
    extra_call = re.sub(r'%\w+ = func.call', '%test_shared = func.call', maximum_line, count=1)
    unchanged(parent.replace(maximum_line, maximum_line + '\n' + extra_call, 1))
    sync = next(line.strip() for line in parent.splitlines()
                if 'hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]' in line)
    unchanged(parent.replace(row_line, row_line + '\n              ' + sync, 1))
    old = fs[maximum]
    for initializer in ('dense<0.000000e+00>', 'dense<1.000000e+00>'):
        changed = old.replace('dense<0xFF800000>', initializer, 1)
        assert changed != old
        unchanged(parent.replace(old, changed, 1))
    changed = old.replace('arith.constant 1 : index', 'arith.constant 2 : index', 1)
    assert changed != old
    unchanged(parent.replace(old, changed, 1))
    tagged = re.sub(r'(arith.maximumf %\w+, %\w+)( : vector)', r'\1 {test.maximum = "kept"}\2', old, count=1)
    assert tagged != old
    result = run(parent.replace(old, tagged, 1), [enabled])
    assert 'test.maximum = "kept"' in functions(result)[probability]
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', parent)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate(renamed, run(renamed, [enabled]))
print('Maximum fusion: full composition, retained new_m recurrence, removed tile maximum/fill, '
      'strict prerequisites, idempotence, extra readers, domination, metadata, coverage and renaming passed')

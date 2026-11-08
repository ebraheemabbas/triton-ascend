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


def chain(text, bn):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    caller = callers[0]
    region = scope(fs[caller])
    calls = re.findall(r'func.call @([^ (]+)', region)
    producers = [name for name in calls if fs[name].count('scf.for ') == 2
                 and 'arith.maximumf ' in fs[name]
                 and '-> (tensor<' + str(bn) + 'x64xf32>, tensor<64xf32>)' in fs[name].splitlines()[0]]
    assert len(producers) == 1
    producer = producers[0]
    line = next(line for line in region.splitlines() if 'func.call @' + producer + '(' in line)
    value = re.search(r'(%\w+):2 = func.call', line)[1] + '#0'
    consumers = [name for name in calls if 'func.call @' + name + '(' + value + ',' in region]
    assert len(consumers) == 1
    return fs, caller, producer, consumers[0], line, value


def protected_lines(text):
    return [line.strip() for line in text.splitlines()
            if 'hivm.hir.sync_block' in line or 'hivm.hir.anchor' in line]


def loop_body(body):
    start = body.index('scf.for ')
    start = body.index('{', start) + 1
    end, depth = start, 1
    while depth:
        depth += (body[end] == '{') - (body[end] == '}')
        end += 1
    return body[start:end - 1]


def validate(before, after, bn, max_scale):
    fs, caller, producer, consumer, old_call, _ = chain(before, bn)
    changed = functions(after)
    assert 'hivm.cvp_attention.query_score_materializations_removed = 1 : i64' in after
    assert ('hivm.cvp_attention.query_max_scalings = 1 : i64' in after) == max_scale
    body = changed[producer]
    assert body.count('scf.for ') == 1 and body.count('arith.maximumf ') == 1
    assert body.count('vector.transfer_write ') == 1
    assert body.count('tensor.insert_slice ') == 0
    assert body.count('arith.mulf ') == 1
    assert ('arith.mulf ' in loop_body(body)) != max_scale
    assert '-> tensor<64xf32> attributes' in body.splitlines()[0]
    assert protected_lines(before) == protected_lines(after)
    reader = changed[consumer]
    assert reader.count('arith.mulf ') == fs[consumer].count('arith.mulf ') + 1
    multiply = re.search(r'(%\w+) = arith.mulf ', reader)[1]
    assert re.search(r'arith.subf ' + re.escape(multiply) + r',', reader)
    assert reader.count('math.exp ') == fs[consumer].count('math.exp ')
    assert reader.count('arith.addf ') == fs[consumer].count('arith.addf ')
    assert reader.count('arith.truncf ') == fs[consumer].count('arith.truncf ')
    old_region, new_region = scope(fs[caller]), scope(changed[caller])
    assert old_region.count('func.call ') == new_region.count('func.call ')
    old_copies = [line.split(' outs(', 1)[1] for line in old_region.splitlines() if 'hivm.hir.copy' in line]
    new_copies = [line.split(' outs(', 1)[1] for line in new_region.splitlines() if 'hivm.hir.copy' in line]
    assert old_copies == new_copies
    raw = re.search(r'func.call @[^ (]+\((%\w+)', old_call)[1]
    # Removing an unused tensor.empty may renumber SSA names. Compare actual
    # dataflow in the transformed caller instead of assuming names are stable.
    new_call = next(line for line in new_region.splitlines() if 'func.call @' + producer + '(' in line)
    raw = re.search(r'func.call @[^ (]+\((%\w+)', new_call)[1]
    assert 'func.call @' + consumer + '(' + raw + ',' in new_region
    for name, original in fs.items():
        if name not in (caller, producer, consumer):
            assert changed[name] == original, name


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    before = run(path.read_text(), [])
    assert run(before, ['--hivm-cvp-attention-query-vector']) == before
    run(before, [prefix + 'enable-max-scaling=true'],
        'CVP query max scaling requires score rematerialization')
    if item['arm'] == 'baseline':
        assert run(before, [prefix + remat]) == before
        run(before, [prefix + remat + ' require-match=true'], unsupported)
        continue
    for max_scale in (False, True):
        options = remat + ' enable-max-scaling=' + str(max_scale).lower()
        after = run(before, [prefix + options + ' require-match=true'])
        validate(before, after, item['BN'], max_scale)
        assert run(after, [prefix + options]) == after
        fused = run(before, [prefix + 'fuse-probability=true require-match=true'])
        together = run(before, [prefix + options + ' fuse-probability=true require-match=true'])
        validate(fused, together, item['BN'], max_scale)
        assert run(fused, [prefix + options + ' require-match=true']) == together
        assert run(together, [prefix + options + ' fuse-probability=true']) == together
    fs, caller, producer, consumer, line, value = chain(before, item['BN'])
    enabled = prefix + remat + ' require-match=true'
    fills = [name for name in re.findall(r'func.call @([^ (]+)', scope(fs[caller]))
             if 'hfusion.has_fill' in fs[name].splitlines()[0]]
    assert len(fills) == 1
    fill = fs[fills[0]]
    malformed = fill.replace('hfusion.has_fill,', 'hfusion.has_fill = false,', 1)
    assert malformed != fill
    run(before.replace(fill, malformed, 1), [enabled], unsupported)
    unknown = fill.replace('hfusion.has_fill,', 'hfusion.has_fill, test.unknown = 1 : i64,', 1)
    run(before.replace(fill, unknown, 1), [enabled], unsupported)
    old = fs[producer]
    for replacement in ('-5.000000e-01', '2.500000e-01'):
        changed = old.replace('dense<5.000000e-01>', 'dense<' + replacement + '>', 1)
        assert changed != old
        run(before.replace(old, changed, 1), [enabled], unsupported)
    changed = old.replace('dense<0xFF800000>', 'dense<0.000000e+00>', 1)
    assert changed != old
    run(before.replace(old, changed, 1), [enabled], unsupported)
    changed = old.replace('arith.maximumf ', 'arith.maxnumf ', 1)
    assert changed != old
    run(before.replace(old, changed, 1), [enabled], unsupported)
    for function in (producer, consumer):
        body = fs[function]
        changed = body.replace('arith.constant 1 : index', 'arith.constant 2 : index', 1)
        assert changed != body
        run(before.replace(body, changed, 1), [enabled], unsupported)
        typ = 'tensor<' + str(item['BN']) + 'x64xf32>'
        changed = body.replace(typ + ',', typ + ' {test.arg = 1 : i64},', 1)
        assert changed != body
        run(before.replace(body, changed, 1), [enabled], unsupported)
        call = next(l for l in scope(fs[caller]).splitlines() if 'func.call @' + function + '(' in l)
        changed = call.replace('{hivm.vector_function,', '{test.operand = 0 : i64, hivm.vector_function,', 1)
        assert changed != call
        run(before.replace(call, changed, 1), [enabled], unsupported)
    extra = (line + '\n              %test_zero = arith.constant 0 : index\n'
             '              %test_read = tensor.extract ' + value + '[%test_zero, %test_zero] : tensor<' + str(item['BN']) + 'x64xf32>')
    run(before.replace(line, extra, 1), [enabled], unsupported)
    extra_call = re.sub(r'%\w+:2 = func.call', '%test_shared:2 = func.call', line, count=1)
    run(before.replace(line, line + '\n' + extra_call, 1), [enabled], unsupported)
    sync = next(l.strip() for l in before.splitlines()
                if 'hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]' in l)
    run(before.replace(line, line + '\n              ' + sync, 1), [enabled], unsupported)
    reader = fs[consumer]
    read_line = next(l for l in reader.splitlines() if 'vector.transfer_read' in l and ': tensor<1x64xf32>' in l)
    masked = reader.replace(read_line, '      %test_mask = arith.constant dense<true> : vector<1x64xi1>\n' +
                            read_line.replace(' {in_bounds', ', %test_mask {in_bounds', 1), 1)
    run(before.replace(reader, masked, 1), [enabled], unsupported)
    slice_value = re.search(r'vector.transfer_read (%\w+)\[', read_line)[1]
    extra_read = re.sub(r'%\w+ = vector.transfer_read', '%test_extra = vector.transfer_read', read_line, count=1)
    run(before.replace(reader, reader.replace(read_line, read_line + '\n' + extra_read, 1), 1), [enabled], unsupported)
    slice_line = next(l for l in reader.splitlines() if slice_value + ' = tensor.extract_slice' in l)
    strided = reader.replace(slice_line, slice_line.replace('[1, 1]', '[2, 1]', 1), 1)
    run(before.replace(reader, strided, 1), [enabled], unsupported)
    caller_header = fs[caller].splitlines()[0]
    changed = caller_header.replace('score_layout = "key-major"', 'score_layout = "query-major"')
    run(before.replace(caller_header, changed, 1), [enabled], unsupported)
    tagged = re.sub(r'(arith.mulf %\w+, %\w+)( : vector)', r'\1 {test.scale = "kept"}\2', old, count=1)
    assert tagged != old
    for max_scale in (False, True):
        result = run(before.replace(old, tagged, 1),
                     [prefix + remat + ' enable-max-scaling=' + str(max_scale).lower() + ' require-match=true'])
        changed = functions(result)
        assert 'test.scale = "kept"' in changed[producer] and 'test.scale = "kept"' in changed[consumer]
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', before)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate(renamed, run(renamed, [enabled]), item['BN'], False)
print('Score rematerialization: standalone/combined, max scaling, default-off, idempotence, '
      'closed uses, attributes, handoffs, key coverage, masks and renaming passed')

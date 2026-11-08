"""Verify first-state specialization on captured production modules."""
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
prefix = '--hivm-cvp-attention-boundary='
state_option = 'enable-first-state=true require-match=true'
output_options = 'enable-first-output=true enable-last-output=true require-match=true'
all_options = output_options + ' enable-first-state=true'
rejection = 'CVP boundary specialization requires a proved Skew logical boundary'
ssa = r'%[\w.$]+(?:#\d+)?'


def run(source, options, expected=None):
    global counter
    command = [args.opt, '--mlir-disable-threading', *options, '-verify-each']
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if args.artifacts:
        stem = f'{counter:04d}'
        for suffix, value in [('input.mlir', source), ('output.mlir', result.stdout), ('stderr.txt', result.stderr)]:
            (args.artifacts / (stem + '.' + suffix)).write_text(value)
        (args.artifacts / (stem + '.json')).write_text(json.dumps({
            'argv': command, 'exit': result.returncode, 'expected_error': expected
        }, indent=2) + '\n')
    counter += 1
    if expected:
        assert result.returncode != 0 and expected in result.stderr, result.stderr
        return None
    assert result.returncode == 0, result.stderr
    return result.stdout


def block(text, opening):
    assert text[opening] == '{'
    depth, end = 1, opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return opening + 1, end - 1, end


def functions(text):
    result = {}
    for match in re.finditer(r'^  func.func [^\n]+', text, re.M):
        header = match[0]
        if not header.endswith('{'):
            continue
        begin, end, closing = block(text, match.end() - 1)
        name = re.search(r'@([^ (]+)', header)[1]
        result[name] = {'header': header, 'body': text[begin:end], 'text': text[match.start():closing]}
    return result


def caller_info(text):
    fs = functions(text)
    names = [name for name, f in fs.items() if 'hacc.entry' in f['header']
             and '#hivm.func_core_type<AIV>' in f['header'] and 'hivm.backup_function' not in f['header']]
    assert len(names) == 1
    name = names[0]
    loops = []
    for match in re.finditer(r'scf.for (' + ssa + ') = (' + ssa + ') to (' + ssa + ') step (' + ssa + r') iter_args\(([^\n]+?)\) ->', fs[name]['body']):
        carried = re.findall('(' + ssa + ') = (' + ssa + ')', match[5])
        if len(carried) == 3:
            loops.append({'iv': match[1], 'lower': match[2], 'upper': match[3], 'step': match[4], 'carried': carried})
    assert len(loops) == 1
    regions = []
    body = fs[name]['body']
    for match in re.finditer(r'scope.scope [^\n]+\{\n', body):
        begin, end, closing = block(body, match.end() - 2)
        suffix = body[closing:body.find('\n', closing)]
        if 'hivm.preload_num = 2 : i32' in suffix and '#hivm.tcore_type<VECTOR>' in suffix:
            regions.append(body[begin:end])
    assert len(regions) == 1
    return fs, name, loops[0], regions[0]


def calls(text):
    return [(m[1], [v.strip() for v in m[2].split(',')] if m[2].strip() else [], m[0])
            for m in re.finditer(r'(?<![\w.])(?:func\.)?call @([^ (]+)\(([^)]*)\)[^\n]*', text)]


def arguments(function):
    return re.findall(r'(%[\w.$]+)\s*:', function['header'])


def normalize(body, argument_map=None):
    names = dict(argument_map or {})
    def replace(match):
        value = match[0]
        if value not in names:
            names[value] = '%v' + str(len(names))
        return names[value]
    body = re.sub(ssa, replace, body)
    return '\n'.join(line.strip() for line in body.splitlines() if line.strip())


def initializer(body, memory):
    found = [line for line in body.splitlines() if any(ops == [memory] for _, ops, _ in calls(line))]
    found += [line for line in body.splitlines() if 'hivm.hir.copy ' in line and 'outs(' + memory + ' :' in line]
    assert len(found) == 1, (memory, found)
    return found[0]


def fill_initializer(body, memory):
    seen = set()
    while memory not in seen:
        seen.add(memory)
        line = initializer(body, memory)
        direct = calls(line)
        if direct:
            assert len(direct) == 1 and len(direct[0][1]) == 1
            return direct[0][0], memory, line
        copied = re.search(r'hivm.hir.copy ins\((' + ssa + r') :', line)
        assert copied
        memory = copied[1]
    raise AssertionError('Initializer copy cycle')


def validate_state(before, after):
    old, name, loop, region = caller_info(before)
    new, new_name, new_loop, new_region = caller_info(after)
    assert name == new_name
    assert 'hivm.cvp_attention.first_state_specializations = 1 : i64' in after
    maximum, total = [int(re.search(key + r' = (\d+) : i64', old[name]['header'])[1]) for key in ['max_state', 'sum_state']]
    old_states = {loop['carried'][maximum][0]: '0xFF800000', loop['carried'][total][0]: '1.000000e+00'}
    new_states = {new_loop['carried'][maximum][0]: '0xFF800000', new_loop['carried'][total][0]: '1.000000e+00'}
    sites = [(fn, ops) for fn, ops, _ in calls(region) if any(x in old_states for x in ops)]
    assert sites
    for fn, _ in sites:
        assert old[fn]['text'] == new[fn]['text'], 'Regular producer changed'
    projected = new[name]['body']
    replacements, conditions, first_functions = [], [], []
    for match in re.finditer(r'scf.if (' + ssa + r') \{', new_region):
        begin, end, closing = block(new_region, match.end() - 1)
        following = re.match(r' else \{', new_region[closing:])
        assert following
        eb, ee, ec = block(new_region, closing + following.end() - 1)
        first, regular = calls(new_region[begin:end]), calls(new_region[eb:ee])
        assert len(first) == len(regular) == 1
        first_name, first_args, _ = first[0]
        regular_name, regular_args, regular_line = regular[0]
        assert 'hivm.cvp_attention.boundary = "first_state"' in new[first_name]['header']
        removed = {i: new_states[v] for i, v in enumerate(regular_args) if v in new_states}
        assert removed and first_args == [v for i, v in enumerate(regular_args) if i not in removed]
        original_args = arguments(new[regular_name])
        first_arg_names = arguments(new[first_name])
        remaining = [i for i in range(len(original_args)) if i not in removed]
        assert len(first_arg_names) == len(remaining)
        expected_body = new[regular_name]['body']
        for index, literal in removed.items():
            arg = re.escape(original_args[index])
            pattern = r'(?m)^([ \t]*)(%[\w.$]+) = vector.transfer_read ' + arg + r'\[[^\n]+vector<64xf32>[ \t]*$'
            expected_body, count = re.subn(pattern, lambda m: m[1] + m[2] + ' = arith.constant dense<' + literal + '> : vector<64xf32>', expected_body)
            assert count >= 1
        expected_map = {original_args[i]: '%a' + str(i) for i in remaining}
        actual_map = {value: '%a' + str(i) for value, i in zip(first_arg_names, remaining)}
        assert normalize(expected_body, expected_map) == normalize(new[first_name]['body'], actual_map), 'First clone changed more than initialized reads'
        condition = match[1]
        definition = next(line for line in new[name]['body'].splitlines() if re.search(re.escape(condition) + r' = arith.cmpi ', line))
        assert 'arith.cmpi eq, ' + new_loop['iv'] + ', ' + new_loop['lower'] in definition
        conditions.append(definition)
        replacements.append((new_region[match.start():ec], regular_line))
        first_functions.append(first_name)
    assert len(first_functions) == len(sites) and len(set(first_functions)) == len(sites)
    for original, replacement in replacements:
        assert projected.count(original) == 1
        projected = projected.replace(original, replacement, 1)
    for line in conditions:
        projected = projected.replace(line + '\n', '', 1)
    expected = old[name]['body']
    for slot in [maximum, total]:
        line = initializer(expected, loop['carried'][slot][1])
        expected = expected.replace(line + '\n', '', 1)
    assert normalize(expected) == normalize(projected), 'Caller changed beyond first-call dispatch and closed initializers'
    return old, name, loop, region, sites


def replace_function(text, function, replacement):
    assert text.count(function['text']) == 1
    return text.replace(function['text'], replacement, 1)


def test_negatives(before):
    fs, name, loop, region = caller_info(before)
    max_slot = int(re.search(r'max_state = (\d+) : i64', fs[name]['header'])[1])
    sum_slot = int(re.search(r'sum_state = (\d+) : i64', fs[name]['header'])[1])
    state = loop['carried'][max_slot][0]
    enabled = prefix + state_option
    for slot, old_literal, replacement in [(max_slot, 'dense<0xFF800000>', 'dense<0.000000e+00>'),
                                            (sum_slot, 'dense<1.000000e+00>', 'dense<2.000000e+00>')]:
        fn, _, _ = fill_initializer(fs[name]['body'], loop['carried'][slot][1])
        changed = fs[fn]['text'].replace(old_literal, replacement, 1)
        assert changed != fs[fn]['text']
        run(replace_function(before, fs[fn], changed), [enabled], rejection)
    wrong_slot = fs[name]['text'].replace(f'max_state = {max_slot} : i64', 'max_state = 0 : i64', 1)
    run(replace_function(before, fs[name], wrong_slot), [enabled], rejection)
    site = next((fn, operands, line) for fn, operands, line in calls(region) if state in operands)
    fn, operands, call = site
    changed = fs[fn]['text'].replace('attributes {', 'attributes {test_argument_index = 0 : i64, ', 1)
    run(replace_function(before, fs[fn], changed), [enabled], rejection)
    # An additional memory reader of a loop state is outside the proved call group.
    memory_type = 'memref<64xf32, #hivm.address_space<ub>>'
    extra = ('%test_sink = memref.alloc() : ' + memory_type + '\n'
             'hivm.hir.copy ins(' + state + ' : ' + memory_type + ') outs(%test_sink : ' + memory_type + ')\n')
    changed_caller = fs[name]['text'].replace(call, extra + call, 1)
    run(replace_function(before, fs[name], changed_caller), [enabled], rejection)
    # A copy of the initialized allocation must retain the initializing writer.
    initial_memory = loop['carried'][max_slot][1]
    init_line = initializer(fs[name]['body'], initial_memory)
    escaped = ('\n%test_escaped = memref.alloc() : ' + memory_type + '\n'
               'hivm.hir.copy ins(' + initial_memory + ' : ' + memory_type + ') outs(%test_escaped : ' + memory_type + ')')
    changed_caller = fs[name]['text'].replace(init_line, init_line + escaped, 1)
    run(replace_function(before, fs[name], changed_caller), [enabled], rejection)
    # Route an existing state value to a writing callee argument of the same type.
    fn_args = arguments(fs[fn])
    stores = re.findall(r'vector.transfer_write [^\n]*?, (%[\w.$]+)\[', fs[fn]['body'])
    writable = [fn_args.index(value) for value in stores if value in fn_args
                and re.search(re.escape(value) + r': memref<64xf32,', fs[fn]['header'])]
    assert writable
    aliased = list(operands);aliased[writable[0]] = state
    changed_call = call.replace('(' + ', '.join(operands) + ')', '(' + ', '.join(aliased) + ')', 1)
    assert changed_call != call
    run(replace_function(before, fs[name], fs[name]['text'].replace(call, changed_call, 1)), [enabled], rejection)
    changed_caller = fs[name]['text'].replace('hivm.preload_num = 2 : i32', 'hivm.preload_num = 1 : i32', 1)
    run(replace_function(before, fs[name], changed_caller), [enabled], rejection)
    # A wide transfer has observable padding: upper lanes are zero, not initial m.
    index = operands.index(state)
    argument = fn_args[index]
    write_arg = fn_args[writable[0]]
    injected = (f'\n    %test_zero = arith.constant 0.000000e+00 : f32\n'
                f'    %test_index = arith.constant 0 : index\n'
                f'    %test_wide = vector.transfer_read {argument}[%test_index], %test_zero {{in_bounds = [false]}} : {memory_type}, vector<128xf32>\n'
                f'    %test_upper = vector.extract_strided_slice %test_wide {{offsets = [64], sizes = [64], strides = [1]}} : vector<128xf32> to vector<64xf32>\n')
    function = fs[fn]['text'].replace(fs[fn]['header'], fs[fn]['header'] + injected, 1)
    function = function.replace('    return', f'    vector.transfer_write %test_upper, {write_arg}[%test_index] {{in_bounds = [true]}} : vector<64xf32>, {memory_type}\n    return', 1)
    run(replace_function(before, fs[fn], function), [enabled], rejection)
    # Shared constant source: only the copy into this loop's buffer may disappear.
    _, fill_memory, fill_line = fill_initializer(fs[name]['body'], initial_memory)
    source_fill = fill_line.replace(fill_memory, '%test_constant_source')
    source_fill = re.sub(r'(?<![\w.])call @', 'func.call @', source_fill)
    setup = (f'%test_constant_source = memref.alloc() : {memory_type}\n'
             f'%test_other_copy = memref.alloc() : {memory_type}\n' + source_fill + '\n'
             f'hivm.hir.copy ins(%test_constant_source : {memory_type}) outs(%test_other_copy : {memory_type})\n'
             f'hivm.hir.copy ins(%test_constant_source : {memory_type}) outs({initial_memory} : {memory_type})')
    copied = replace_function(before, fs[name], fs[name]['text'].replace(init_line, setup, 1))
    copied = run(copied, [])
    output = run(copied, [enabled])
    validate_state(copied, output)


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
assert len(manifest) == 12
for row in manifest:
    path = args.fixtures / row['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
    before = run(path.read_text(), [])
    assert run(before, ['--hivm-cvp-attention-boundary']) == before
    after = run(before, [prefix + state_option])
    validate_state(before, after)
    assert run(after, [prefix + 'enable-first-state=true']) == after
    outputs = run(before, [prefix + output_options + ' retain-provenance=true'])
    composed = run(outputs, [prefix + state_option])
    validate_state(outputs, composed)
    combined = run(before, [prefix + all_options])
    assert combined == composed, 'Separate and combined boundary transforms differ'
    for token in ['first_output', 'last_output', 'first_state']:
        assert 'hivm.cvp_attention.' + token + '_specializations = 1 : i64' in combined
    test_negatives(before)
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', before)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate_state(renamed, run(renamed, [prefix + state_option]))
print('Key-major first-state: 12 production inputs, exact clone/caller projection, output composition, renamed IR, alias/escape/padding/metadata/boundary guards and shared copied-initializer controls passed; native/NPU remain pending')

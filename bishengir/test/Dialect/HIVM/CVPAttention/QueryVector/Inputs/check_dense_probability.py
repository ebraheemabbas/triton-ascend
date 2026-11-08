"""Check dense P key-half pairing, original FP32 values, and both L1 copies."""
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
dense = 'enable-dense-p=true allow-sum-reassociation=true'
unsupported = 'CVP dense P requires a closed full-key P producer and canonical L1 publication'


def run(source, options, expected=None):
    global counter
    command = [args.opt, *options, '-verify-each']
    result = subprocess.run(command, input=source, text=True, capture_output=True)
    if args.artifacts:
        stem = f'{counter:03d}'
        for suffix, value in [('input.mlir', source), ('output.mlir', result.stdout), ('stderr.txt', result.stderr)]:
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
    return {re.search(r'@([^ (]+)', m[0])[1]: text[m.start():matches[i+1].start() if i+1 < len(matches) else len(text)]
            for i, m in enumerate(matches)}


def scope(body):
    found = []
    for m in re.finditer(r'^[ \t]*(?:%\w+(?::\d+)? = )?scope.scope [^\n]*\{\n', body, re.M):
        end, depth = m.end(), 1
        while depth:
            depth += (body[end] == '{') - (body[end] == '}')
            end += 1
        suffix = body[end:body.find('\n', end)]
        if '#hivm.tcore_type<VECTOR>' in suffix and 'hivm.preload_num = 2 : i32' in suffix:
            found.append(body[m.end():end-1])
    assert len(found) == 1
    return found[0]


def roles(text):
    fs = functions(text)
    callers = [name for name, body in fs.items() if 'hacc.entry' in body.splitlines()[0]
               and '#hivm.func_core_type<AIV>' in body.splitlines()[0]
               and 'hivm.backup_function' not in body.splitlines()[0]]
    assert len(callers) == 1
    caller = callers[0]
    producers = [name for name in re.findall(r'func.call @([^ (]+)', scope(fs[caller]))
                 if 'arith.truncf ' in fs[name] and 'scf.for ' in fs[name]]
    assert len(producers) == 1
    return fs, caller, producers[0]


def validate(before, after, keys):
    old, caller, producer = roles(before)
    new, new_caller, new_producer = roles(after)
    assert (caller, producer) == (new_caller, new_producer)
    half = keys // 2
    body = new[producer]
    assert 'hivm.cvp_attention.dense_p_layouts = 1 : i64' in after
    assert 'padding_required = true' in body.splitlines()[0]
    assert f'tensor<8x{half}x16xf16>' in body.splitlines()[0]
    packing = re.search(r'(%[\w.$]+(?::2|, %[\w.$]+)) = ave.hir.vdintlv (%[\w.$]+), (%[\w.$]+).*DENSE.*vector<64xf16>, vector<128xf16>', body)
    assert packing, body
    packed_result = packing[1].split(', ', 1)[0] if ', ' in packing[1] else packing[1][:-2] + '#0'
    # Both input vectors must derive from FP16 casts of the two original FP32
    # values, and the pair sum must use those FP32 values before either cast.
    casts = re.findall(r'(%[\w.$]+) = arith.truncf (%[\w.$]+).*#hfusion.round_mode<rint>', body)
    assert len(casts) == 2
    for index, (cast, value) in enumerate(casts):
        assert re.search(re.escape(packing[index+2] + ' = vector.shape_cast ' + cast) + r' : vector<1x64xf16> to vector<64xf16>', body)
    reshape = {v: x for v, x in re.findall(r'(%[\w.$]+) = vector.shape_cast (%[\w.$]+) : vector<1x64xf32> to vector<64xf32>', body)}
    sums = re.findall(r'(%[\w.$]+) = arith.addf (%[\w.$]+), (%[\w.$]+) : vector<64xf32>', body)
    pairs = [(v, a, b) for v, a, b in sums if reshape.get(a) == casts[0][1] and reshape.get(b) == casts[1][1]]
    assert len(pairs) == 1
    assert any(pairs[0][0] in (a, b) for v, a, b in sums if v != pairs[0][0])
    assert re.search(r'vector.shape_cast ' + re.escape(packed_result) + r' : vector<128xf16> to vector<8x1x16xf16>', body)
    constants = {n: int(v) for n, v in re.findall(r'(%[\w.$]+) = arith.constant (-?\d+) : index', body)}
    matched = []
    for m in re.finditer(r'(%[\w.$]+) = (%[\w.$]+) to (%[\w.$]+) step (%[\w.$]+) iter_args', body):
        if constants[m[3]] == half:
            matched.append(m)
    assert len(matched) == 1
    bounds = matched[0]
    assert constants[bounds[2]] == 0 and constants[bounds[4]] == 1
    induction = bounds[1]
    paired = [(v, rhs) for v, lhs, rhs in re.findall(r'(%[\w.$]+) = arith.addi (%[\w.$]+), (%[\w.$]+)', body)
              if lhs == induction and constants.get(rhs) == half]
    assert len(paired) == 1
    for key in (induction, paired[0][0]):
        assert re.search(r'tensor.extract_slice %[\w.$]+\[' + re.escape(key) + r', 0\] \[1, 64\]', body)
    definitions = dict(re.findall(r'(%[\w.$]+) = ([^\n]+)', body))
    key_reads = {}
    for value, source in re.findall(r'(%[\w.$]+) = vector.transfer_read (%[\w.$]+)', body):
        view = definitions.get(source, '')
        match = re.match(r'tensor.extract_slice %[\w.$]+\[(%[\w.$]+), 0\] \[1, 64\]', view)
        if match and match[1] in (induction, paired[0][0]):
            key_reads[value] = match[1]

    def leaf_keys(value, visiting=None):
        if value in key_reads:
            return {key_reads[value]}
        visiting = set() if visiting is None else visiting
        assert value not in visiting, 'Cycle in P expression'
        expression = definitions.get(value)
        if expression is None:
            return set()
        return set().union(*(leaf_keys(v, visiting | {value}) for v in re.findall(r'%[\w.$]+', expression)))

    assert leaf_keys(casts[0][1]) == {induction}
    assert leaf_keys(casts[1][1]) == {paired[0][0]}
    assert f'[0, {induction}, 0] [8, 1, 16] [1, 1, 1]' in body
    region = scope(new[caller])
    # The actual P result is marked, not a separate allocation or a row state.
    p_call = next(line for line in region.splitlines() if 'func.call @' + producer + '(' in line)
    call_value = re.search(r'(%[\w.$]+):\d+ = func.call', p_call)[1]
    output_types = p_call.split(' -> ', 1)[1]
    p_index = re.findall(r'tensor<[^>]+>', output_types).index(f'tensor<8x{half}x16xf16>')
    marker = next(line for line in region.splitlines() if 'annotation.mark ' + call_value + '#' + str(p_index) + ' ' in line)
    assert 'hivm.stride_align_dims = array<i32: 1>' in marker
    assert f'hivm.stride_align_value_in_byte = array<i32: {2*(half+1)}>' in marker
    assert region.count('hivm.hir.copy ') == scope(old[caller]).count('hivm.hir.copy ') + 1
    copies = [line for line in region.splitlines() if 'hivm.hir.copy ' in line and f'tensor<4x{half//16}x16x16xf16>' in line]
    assert len(copies) == 2
    definitions = dict(re.findall(r'(%[\w.$]+) = ([^\n]+)', region))
    copy_maps = []
    for line in copies:
        operands = re.search(r'ins\((%[\w.$]+) : .*outs\((%[\w.$]+) : ', line)
        assert operands, line
        expand = re.match(r'tensor.expand_shape (%[\w.$]+) \[\[0\], \[1, 2\], \[3\]\]', definitions[operands[1]])
        assert expand
        source = re.match(r'tensor.extract_slice (%[\w.$]+#\d+)\[(\d+), 0, 0\]', definitions[expand[1]])
        target = re.match(r'memref.subview (%[\w.$]+)\[0, (\d+), 0, 0\]', definitions[operands[2]])
        assert source and target
        assert source[1] == call_value + '#' + str(p_index)
        copy_maps.append((int(source[2]), target[1], int(target[2])))
    assert copy_maps[0][1] == copy_maps[1][1], 'Copies target different L1 tiles'
    # Decode the actual source/target views and check each destination element
    # independently against logical P[q,k], for both query subblocks.
    for subblock in (0, 1):
        observed = {}
        for source_plane, _, target_key_block in copy_maps:
            assert source_plane in (0, 4)
            for group in range(4):
                for key in range(half):
                    for lane in range(16):
                        query = 64 * subblock + 16 * group + lane
                        destination = (query, 16 * target_key_block + key)
                        assert destination not in observed
                        observed[destination] = (query, key + (source_plane // 4) * half)
        assert len(observed) == 64 * keys
        assert all(observed[(q, k)] == (q, k) for q in range(64*subblock, 64*(subblock+1)) for k in range(keys))
    for plane in (0, 1):
        assert re.search(r'tensor.extract_slice %[\w.$]+(?:#\d+)?\[' + f'{4*plane}, 0, 0' + r'\] \[' + f'4, {half}, 16' + r'\] \[1, 1, 1\]', region)
        assert re.search(r'memref.subview %[\w.$]+\[' + f'0, {plane*half//16}, 0, 0' + r'\] \[' + f'4, {half//16}, 16, 16' + r'\] \[1, 1, 1, 1\]', region)
    protected = lambda text: [l.strip() for l in text.splitlines() if 'hivm.hir.sync_block' in l or 'hivm.hir.anchor' in l]
    assert protected(before) == protected(after)
    for name in old:
        if name not in (caller, producer):
            assert old[name] == new[name], name


manifest = json.loads((args.fixtures / 'MANIFEST.json').read_text())
for item in manifest:
    path = args.fixtures / item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
    before = run(path.read_text(), [])
    assert run(before, [prefix + 'enable-dense-p=false']) == before
    run(before, [prefix + 'enable-dense-p=true'], 'requires one- or two-chain policy and explicit FP32 sum reassociation permission')
    run(before, [prefix + dense + ' reduction-chains=4'], 'requires one- or two-chain policy and explicit FP32 sum reassociation permission')
    enabled = prefix + dense + ' require-match=true'
    if item['arm'] == 'baseline':
        assert run(before, [prefix + dense]) == before
        run(before, [enabled], unsupported)
        continue
    after = run(before, [enabled])
    validate(before, after, item['BN'])
    assert run(after, [prefix + dense]) == after
    for parent in ['fuse-probability=true',
                   'enable-score-rematerialization=true fuse-probability=true fuse-state=true',
                   'enable-score-rematerialization=true enable-max-scaling=true fuse-probability=true fuse-state=true fuse-maximum=true']:
        control = run(before, [prefix + parent + ' require-match=true'])
        combined = run(before, [prefix + parent + ' ' + dense + ' require-match=true'])
        validate(control, combined, item['BN'])
        assert run(control, [enabled]) == combined
        assert run(combined, [prefix + parent + ' ' + dense]) == combined
    fs, caller, producer = roles(before)
    copy = next(line for line in scope(fs[caller]).splitlines() if 'hivm.hir.copy ' in line and 'tensor<4x' in line)
    changed_caller = fs[caller].replace(copy, copy + '\n' + copy, 1)
    mutated = before.replace(fs[caller], changed_caller, 1)
    assert functions(mutated)[caller].count(copy) == fs[caller].count(copy) + 1
    assert all(functions(mutated)[name] == body for name, body in fs.items() if name != caller)
    run(mutated, [enabled], unsupported)
    header = fs[caller].splitlines()[0]
    wrong = header.replace('numerical_policy = "reassociate-fp32-sum"', 'numerical_policy = "strict"')
    assert wrong != header
    run(before.replace(header, wrong, 1), [enabled], unsupported)
    call = next(line for line in scope(fs[caller]).splitlines() if 'func.call @' + producer + '(' in line)
    duplicate = re.sub(r'%[\w.$]+:2 = func.call', '%shared_test:2 = func.call', call, count=1)
    assert duplicate != call
    run(before.replace(call, call + '\n' + duplicate, 1), [enabled], unsupported)
    function_header = fs[producer].splitlines()[0]
    wrong = function_header.replace('attributes {', 'attributes {test_index_metadata = 1 : i64, ', 1)
    run(before.replace(function_header, wrong, 1), [enabled], unsupported)
    renamed = re.sub(r'@([A-Za-z_][\w.$]*)', r'@renamed_\1', before)
    renamed = re.sub(r'%([A-Za-z_]\w*|\d+)', r'%renamed_\1', renamed)
    renamed = run(renamed, [])
    validate(renamed, run(renamed, [enabled]), item['BN'])
print('Dense P tensor IR passed: key-half pairing, FP32 sum before rounding, DENSE packing, two L1 ranges, compositions and refusal guards; physical padding/native/NPU remain unqualified')

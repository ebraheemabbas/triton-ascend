"""Check paired forwarding, preserved initialization/events and unsafe refusals."""
from pathlib import Path
import re
import subprocess
import sys

source = Path(sys.argv[1]).read_text()
opt = sys.argv[2]
flag = '--hivm-cvp-query-scalar-input-forward'

def replace(text, old, new, count=1):
    assert text.count(old) == count, (old, text.count(old))
    return text.replace(old, new)

def run(text, enabled=True, required=False):
    args = [opt] + ([flag + ('=require-match' if required else '')] if enabled else [])
    return subprocess.run(args, input=text, text=True, capture_output=True, timeout=60)

def protocol(text):
    return [line.strip() for line in text.splitlines()
            if 'hivm.hir.anchor' in line or 'hivm.hir.sync_' in line]

if len(sys.argv) > 3 and sys.argv[3] == 'captured':
    baseline = run(source, False)
    result = run(source, required=True)
    assert baseline.returncode == result.returncode == 0, (baseline.stderr, result.stderr)
    assert 'hivm.cvp_query.scalar_input_copies_forwarded = 6 : i64' in result.stdout
    for operation in ('hivm.hir.copy', 'memref.alloc()'):
        assert baseline.stdout.count(operation) - result.stdout.count(operation) == 6, operation
    assert protocol(baseline.stdout) == protocol(result.stdout)
    def definitions(text):
        result = {}
        parts = re.split(r'(?=^  func.func )', text, flags=re.M)[1:]
        for part in parts:
            symbol = re.search(r'func.func (?:private )?@([^ (]+)', part).group(1)
            result[symbol] = part.rstrip().removesuffix('}').rstrip() if part is parts[-1] else part.rstrip()
        return result
    old, new = definitions(baseline.stdout), definitions(result.stdout)
    assert old.keys() == new.keys()
    changed = [name for name in old if old[name] != new[name]]
    assert len(changed) == 1, changed
    header = new[changed[0]].splitlines()[0]
    assert 'hacc.entry' in header and '#hivm.func_core_type<AIV>' in header
    scalar_banks = [line for line in result.stdout.splitlines() if 'state_bank' in line
                    and any('role = "' + role + '"' in line for role in ('sum','maximum'))]
    assert len(scalar_banks) == 4 and all('alignment = 64 : i64' in line for line in scalar_banks)
    print('PASS: actual captured P form; six copies/allocations removed, all other function bodies and explicit events unchanged')
    raise SystemExit(0)

def positive(text, alignment=64):
    baseline = run(text, False)
    assert baseline.returncode == 0, baseline.stderr
    result = run(text, required=True)
    assert result.returncode == 0, result.stderr
    assert 'hivm.cvp_query.scalar_input_copies_forwarded = 6 : i64' in result.stdout
    assert baseline.stdout.count('hivm.hir.copy') - result.stdout.count('hivm.hir.copy') == 6
    assert baseline.stdout.count('memref.alloc()') - result.stdout.count('memref.alloc()') == 6
    assert protocol(baseline.stdout) == protocol(result.stdout)
    before = baseline.stdout.split('  func.func @case(', 1)
    after = result.stdout.split('  func.func @case(', 1)
    # No outlined math changes. Only the module counter differs in this prefix.
    assert re.sub(r'module(?: attributes \{[^\n]*\})? \{', 'module {', before[0]) == re.sub(
        r'module(?: attributes \{[^\n]*\})? \{', 'module {', after[0])
    body = after[1]
    banks = re.findall(r'(%[\w]+) = memref.alloc\(\) \{alignment = (\d+) : i64, hivm.cvp_query.state_bank', body)
    assert len(banks) == 4 and all(int(value) >= alignment for _, value in banks), banks
    args = re.search(r'call @(?:produce|renamed_produce)\(([^)]*)\)', body).group(1).split(', ')
    assert len(args) == 7
    for index in (2, 3):
        match = re.search(re.escape(args[index]) + r' = arith.select %arg0, (%[\w]+), (%[\w]+)', body)
        assert match, (index, body)
        seed, selected = match.groups()
        assert seed not in {bank for bank, _ in banks}
        pair = re.search(re.escape(selected) + r' = arith.select %arg1, (%[\w]+), (%[\w]+)', body)
        assert pair and all(value in {bank for bank, _ in banks} for value in pair.groups())
    assert body.count('hivm.hir.copy') == 6  # Four initializations + two output copies.
    assert 'hivm.multi_buffer = 3 : i32' in body

positive(source)
positive(source.replace('@produce', '@renamed_produce'))
positive(source.replace('alignment = 64 : i64', 'alignment = 128 : i64'), 128)
positive(source.replace('arith.select %slot, %l1, %l0', 'arith.select %slot, %l0, %l1')
               .replace('arith.select %slot, %m1, %m0', 'arith.select %slot, %m0, %m1'))

# Match the real function's seed before its ordinary outer loop.
minus_line = '    %minusinf = memref.alloc() {alignment = 64 : i64} : memref<4xf32, #ub>\n'
outer_seed = replace(source, minus_line, '')
outer_seed = replace(outer_seed, '    %ones = memref.alloc()',
    minus_line + '    scf.for %outer = %c0 to %c1 step %c1 {\n    %ones = memref.alloc()')
outer_seed = replace(outer_seed, '    return\n  }\n}\n', '    }\n    return\n  }\n}\n')
positive(outer_seed)

call = '        func.call @produce(%score, %packed, %old_l, %old_m, %new_l, %alpha, %new_m)'
write_l = '        hivm.hir.copy ins(%new_l : memref<4xf32, #ub>) outs(%selected_l : memref<4xf32, #ub>)\n'
copy_l = '        hivm.hir.copy ins(%lb : memref<4xf32, #ub>) outs(%lc : memref<4xf32, #ub>)\n'
negatives = {
    'missing-state-storage': replace(source, ', hivm.cvp_query.state_storage', ''),
    'wrong-phase': replace(source, 'hivm.preload_num = 2 : i32', 'hivm.preload_num = 1 : i32'),
    'wrong-depth': replace(source, 'hivm.max_preload_num = 4 : i32', 'hivm.max_preload_num = 3 : i32'),
    'unroll': replace(source, 'pipeline_mode = "skew"', 'pipeline_mode = "unroll"'),
    'U2': replace(source, 'key_unroll = 1 : i32', 'key_unroll = 2 : i32'),
    'one-query-slot': replace(source, 'query_slots = 2 : i32', 'query_slots = 1 : i32'),
    'wrong-bank-clock': replace(source, 'clock = "query"', 'clock = "key"', 4),
    'wrong-bank-role': replace(source, 'role = "maximum"', 'role = "sum"', 2),
    'wrong-bank-slots': replace(source, 'required_slots = 2 : i32', 'required_slots = 1 : i32', 4),
    'missing-unique': replace(source, '    annotation.mark %m0 {mem_unique} : memref<4xf32, #ub>\n', ''),
    'mismatched-slot-direction': replace(source, 'arith.select %slot, %m1, %m0', 'arith.select %slot, %m0, %m1'),
    'mismatched-slot-condition': replace(source, 'arith.select %slot, %m1, %m0', 'arith.select %first, %m1, %m0'),
    'mismatched-first-key': replace(source, 'arith.select %first, %minusinf, %mc', 'arith.select %slot, %minusinf, %mc'),
    'opaque-snapshot-reader': replace(source, copy_l, copy_l + '        func.call @opaque(%lc) : (memref<4xf32, #ub>) -> ()\n'),
    'opaque-bank-reader': replace(source, call, '        func.call @opaque(%selected_l) : (memref<4xf32, #ub>) -> ()\n' + call),
    'intervening-effect': replace(source, call, '        func.call @effect() : () -> ()\n' + call),
    'bank-written-before-reader': replace(source, write_l, '').replace(call, write_l + call),
    'duplicate-snapshot-writer': replace(source, copy_l, copy_l + copy_l),
    'incomplete-pair': replace(replace(source, copy_l, ''), '%old_l = arith.select %first, %ones, %lc', '%old_l = arith.select %first, %ones, %lb'),
    'destination-aliases-bank': replace(source, call, call.replace('%new_l, %alpha', '%selected_l, %alpha')),
    'seed-aliases-bank': replace(source, 'arith.select %first, %ones, %lc', 'arith.select %first, %l0, %lc'),
    'temporary-annotation': replace(source, '%lc = memref.alloc() {alignment = 64 : i64}',
                                  '%lc = memref.alloc() {alignment = 64 : i64, unrelated_constraint}'),
    'hidden-vf-call': replace(source, '    %sum = arith.addf', '    func.call @effect() : () -> ()\n    %sum = arith.addf'),
    'writable-input': replace(source, '    %sum = arith.addf',
        '    vector.transfer_write %l, %old_l[%c0] {in_bounds = [true]} : vector<4xf32>, memref<4xf32, #ub>\n    %sum = arith.addf'),
}
for offset in (0, 1):
    view_type = ('memref<3xf32, #ub>' if offset == 0 else
                 'memref<3xf32, strided<[1], offset: 1>, #ub>')
    partial = ('        %part_lb = memref.subview %lb[' + str(offset) + '] [3] [1] : memref<4xf32, #ub> to ' + view_type + '\n'
               '        %part_lc = memref.subview %lc[' + str(offset) + '] [3] [1] : memref<4xf32, #ub> to ' + view_type + '\n'
               '        hivm.hir.copy ins(%part_lb : ' + view_type + ') outs(%part_lc : ' + view_type + ')\n')
    negatives['partial-copy-offset-' + str(offset)] = replace(source, copy_l, partial)
other_stage = ('      scope.scope : () -> () {\n'
               '        hivm.hir.copy ins(%score : memref<4xf32, #ub>) outs(%l0 : memref<4xf32, #ub>)\n'
               '        scope.return\n'
               '      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 0 : i32, hivm.max_preload_num = 4 : i32}\n')
negatives['writer-in-another-stage'] = replace(source,
    '      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1',
    other_stage + '      hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 1')
negatives['extra-bank-annotation'] = replace(source,
    'annotation.mark %l0 {mem_unique}', 'annotation.mark %l0 {mem_unique, unexpected_metadata}')
negatives['extra-bank-read-effect'] = replace(source,
    'annotation.mark %l0 {mem_unique}', 'annotation.mark %l0 {effects = ["write", "read"], mem_unique}')
inner_seed = replace(source, '        %old_m = arith.select',
    '        %inner_seed = memref.alloc() : memref<4xf32, #ub>\n        %old_m = arith.select')
negatives['seed-inside-P-stage'] = replace(inner_seed,
    'arith.select %first, %minusinf, %mc', 'arith.select %first, %inner_seed, %mc')
negatives['seed-across-non-loop-region'] = replace(outer_seed,
    'scf.for %outer = %c0 to %c1 step %c1 {', 'scf.if %slot {')
for name, text in negatives.items():
    baseline = run(text, False)
    assert baseline.returncode == 0, (name, 'invalid negative fixture', baseline.stderr)
    result = run(text)
    assert result.returncode == 0, (name, result.stderr)
    assert 'hivm.cvp_query.scalar_input_copies_forwarded = 0 : i64' in result.stdout, name
    without_counter = re.sub(r' attributes \{hivm.cvp_query.scalar_input_copies_forwarded = 0 : i64\}', '', result.stdout, count=1)
    assert without_counter == baseline.stdout, (name, 'refused module changed')
    required = run(text, required=True)
    assert required.returncode != 0, (name, 'missing explicit refusal')
    assert 'requires a proved paired six-copy readonly P-state contract' in required.stderr, (name, required.stderr)
print('PASS: five positives, ' + str(len(negatives)) + ' valid unsafe modules and explicit paired refusals')

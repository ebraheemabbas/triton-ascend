"""Check authentic HD128 chunked forwarding and valid unsafe counterexamples."""
from pathlib import Path
import re
import subprocess
import sys

source = Path(sys.argv[1]).read_text()
opt = sys.argv[2]
validate_only = sys.argv[3:] == ['--validate-fixtures-only']
assert sys.argv[3:] in ([], ['--validate-fixtures-only'])
ENTRY = '_attn_fwd_mix_aiv'
UPDATE = '_attn_fwd_mix_aiv_outlined_vf_5'
FIRST = '__cvp_attention_query_first_output'
LAST = '__cvp_attention_last_output'
STATE = '--hivm-cvp-query-state-copy-forward'
FINAL = '--hivm-cvp-query-final-output-forward'
DIRECT = '--hivm-cvp-query-output-copy-forward'
function_pattern = re.compile(r'^  func\.func @([^\s(]+)\(.*?^  }\n', re.M | re.S)


def run(text, flags=(), expected=0):
    result = subprocess.run([opt, *flags], input=text, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == expected, (flags, result.returncode, result.stderr)
    return result.stdout


def functions(text):
    aliases = dict(re.findall(r'^(#\w+) = (.+)$', text, re.M))
    return {item[1]: re.sub(r'#[A-Za-z_]\w*', lambda match: aliases.get(match[0], match[0]), item[0])
            for item in function_pattern.finditer(text)}


def body(text, name):
    return next(item[0] for item in function_pattern.finditer(text) if item[1] == name)


def change(text, name, mutate):
    before = body(text, name)
    after = mutate(before)
    assert before != after and text.count(before) == 1
    return text.replace(before, after)


def operands(text, name):
    values = re.findall(r'func.call @' + re.escape(name) + r'\(([^)]*)\)', text)
    assert len(values) == 1, (name, len(values))
    return values[0].split(', ')


def attr(text, name):
    match = re.search(re.escape(name) + r' = (\d+) : i64', text)
    assert match, name
    return int(match[1])


def anchors(text):
    return re.findall(r'hivm.hir.anchor \{id = \d+ : i64\}', text)


def bank(text):
    found = re.search(r'(%\w+) = memref.alloc\(\)[^\n]*hivm.cvp_query.state_bank = '
                      r'\{[^}]*role = "accumulator"', body(text, ENTRY))
    assert found
    return found[1]


def math_vfs(text):
    return {name: value for name, value in functions(text).items()
            if 'hivm.vector_function' in value.split('\n')[0]}


def constant(function, name, value):
    assert name not in function
    index = function.index('\n') + 1
    return function[:index] + '    ' + name + ' = arith.constant ' + str(value) + ' : index\n' + function[index:]


def loop_headers(function):
    loops = list(re.finditer(r'scf.for (%\w+) = (%\w+) to (%\w+) step (%\w+) \{', function))
    assert len(loops) == 2
    return loops


def change_loop(function, loop_index, field, replacement, added_constant=None):
    if added_constant is not None:
        function = constant(function, replacement, added_constant)
    loop = loop_headers(function)[loop_index]
    groups = list(loop.groups())
    groups[field] = replacement
    changed = 'scf.for ' + groups[0] + ' = ' + groups[1] + ' to ' + groups[2] + ' step ' + groups[3] + ' {'
    return function[:loop.start()] + changed + function[loop.end():]


def shifted_pv(function):
    rows, columns = loop_headers(function)
    needle = 'memref.subview %arg1[' + rows[1] + ', ' + columns[1] + ']'
    assert function.count(needle) == 1
    return function.replace(needle, 'memref.subview %arg1[' + rows[1] + ', %c0]')


def masked_write(function, element):
    function = constant_mask(function)
    matches = [line for line in function.splitlines(True)
               if 'vector.transfer_write' in line and ': vector<1x64x' + element + '>' in line]
    assert len(matches) == 1
    before = matches[0]
    after, count = re.subn(r'(vector\.transfer_write [^\n]*?\[[^]]*\])', r'\1, %cvp_mask', before, count=1)
    assert count == 1 and '], %cvp_mask {' in after
    return function.replace(before, after)


def constant_mask(function):
    assert '%cvp_mask' not in function
    index = function.index('\n') + 1
    return function[:index] + '    %cvp_mask = arith.constant dense<false> : vector<1x64xi1>\n' + function[index:]


def masked_pv(function):
    function = constant_mask(function)
    match = re.search(r'(%\w+) = memref.subview %arg1\[', function)
    assert match
    needle = r'(vector\.transfer_read ' + re.escape(match[1]) + r'\[[^]]*\], %\w+)( \{)'
    function, count = re.subn(needle, r'\1, %cvp_mask\2', function, count=1)
    assert count == 1
    return function


def before_copy(text, statement):
    function = body(text, ENTRY)
    call = function.index('func.call @' + UPDATE + '(')
    copy = function.index('hivm.hir.copy', call)
    start = function.rfind('\n', 0, copy) + 1
    indent = function[start:copy]
    return text.replace(function, function[:start] + indent + statement + '\n' + function[start:])


def extra_reader(text):
    declaration = '  func.func private @unexpected_chunk_reader(memref<64x128xf32, #hivm.address_space<ub>>)\n'
    index = next(function_pattern.finditer(text)).start()
    text = text[:index] + declaration + text[index:]
    return before_copy(text, 'func.call @unexpected_chunk_reader(' + bank(text) +
                       ') : (memref<64x128xf32, #hivm.address_space<ub>>) -> ()')


state = run(source, [STATE + '=require-match=true', '--canonicalize'])
assert attr(state, 'hivm.cvp_query.snapshot_copies_forwarded') == 1
assert anchors(source) == anchors(state)

negatives = []
for name in (FIRST, UPDATE, LAST):
    target = FINAL if name == LAST else DIRECT
    for label, mutate in (
        ('missing-column-chunk', lambda text: change_loop(text, 1, 2, '%c64')),
        ('column-gap', lambda text: change_loop(text, 1, 3, '%c128')),
        ('column-overlap', lambda text: change_loop(text, 1, 3, '%cvp_step32', 32)),
        ('shifted-column-start', lambda text: change_loop(text, 1, 1, '%c64')),
        ('missing-rows', lambda text: change_loop(text, 0, 2, '%cvp_rows32', 32)),
        ('row-gap', lambda text: change_loop(text, 0, 3, '%cvp_rows2', 2)),
        ('masked-write', lambda text, element='f16' if name == LAST else 'f32': masked_write(text, element)),
    ):
        negatives.append((name + '-' + label, change(state, name, mutate), target))
for name in (FIRST, UPDATE):
    negatives.append((name + '-shifted-PV-chunk', change(state, name, shifted_pv), DIRECT))
    negatives.append((name + '-masked-PV-read', change(state, name, masked_pv), DIRECT))
negatives.append(('non-unit-chunk-stride', change(state, UPDATE, lambda text:
    text.replace('[1, 64] [1, 1]', '[1, 64] [1, 2]').replace('strided<[128, 1]', 'strided<[128, 2]')), DIRECT))
negatives.append(('extra-bank-reader', extra_reader(state), DIRECT))
negatives.append(('fence-before-O-copyback', before_copy(state, 'hivm.hir.pipe_barrier[<PIPE_V>]'), DIRECT))

if validate_only:
    for name, text, target in negatives:
        run(text)
    print('PASS: authentic state control and ' + str(len(negatives)) + ' negative fixtures parse; no new-C++ positive claim')
    raise SystemExit(0)

final = run(state, [FINAL + '=require-match=true', '--canonicalize'])
combined = run(final, [DIRECT + '=require-match=true', '--canonicalize'])
assert attr(final, 'hivm.cvp_query.final_output_copies_forwarded') == 1
assert attr(combined, 'hivm.cvp_query.output_copies_forwarded') == 1
assert final.count('hivm.hir.copy') == state.count('hivm.hir.copy') - 1
assert combined.count('hivm.hir.copy') == state.count('hivm.hir.copy') - 2
accumulator = bank(combined)
raw_pv = operands(state, UPDATE)[1]
assert operands(combined, UPDATE)[1] == raw_pv and operands(combined, UPDATE)[3] == accumulator
assert operands(combined, FIRST)[1] == raw_pv and operands(combined, FIRST)[2] == accumulator
assert operands(combined, LAST)[4] == accumulator and operands(combined, LAST)[7] == raw_pv
for text in (state, final, combined):
    assert anchors(source) == anchors(text)
    assert math_vfs(state) == math_vfs(text), 'Changed a VF math body'
    for policy in ('default', 'largest-first'):
        run(text, ['--hivm-plan-memory-regbase=disable-tightly-coupled-buffer-reuse=true '
                   'plan-memory-strategy=' + policy])
for flag in (FINAL, DIRECT):
    repeated = run(combined, [flag])
    assert functions(repeated) == functions(combined), 'Repeated pass changed a function'

for name, text, target in negatives:
    normalized = run(text)
    if target == DIRECT:
        normalized = run(normalized, [FINAL + '=require-match=true', '--canonicalize'])
    optional = run(normalized, [target])
    suffix = 'final_output_copies_forwarded' if target == FINAL else 'output_copies_forwarded'
    assert attr(optional, 'hivm.cvp_query.' + suffix) == 0, name
    assert functions(optional) == functions(normalized), (name, 'Partial mutation on refusal')
    refused = subprocess.run([opt, target + '=require-match=true'], input=normalized,
                             text=True, capture_output=True, timeout=120)
    assert refused.returncode != 0, (name, 'Required rejection missing')

print('PASS: authentic HD128 state/final/combined, unchanged VF math/anchors/rawPV, '
      'stock memory/idempotence; ' + str(len(negatives)) + ' valid unsafe chunk/alias/fence cases rejected')

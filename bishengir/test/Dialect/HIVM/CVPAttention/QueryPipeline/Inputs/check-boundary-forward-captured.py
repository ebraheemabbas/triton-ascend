"""Exercise real first/final O forms, full forwarding and unsafe counterexamples."""
from pathlib import Path
import re
import subprocess
import sys

source = Path(sys.argv[1]).read_text()
opt = sys.argv[2]
UPDATE = '_attn_fwd_mix_aiv_fused_6_outlined_vf_0'
FIRST = '__cvp_attention_query_first_output'
LAST = '__cvp_attention_last_output'
NORMAL = '_attn_fwd_mix_aiv_fused_4_outlined_merged_vf_0'
ENTRY = '_attn_fwd_mix_aiv'
DIRECT = '--hivm-cvp-query-output-copy-forward'
SCALAR = '--hivm-cvp-query-scalar-input-forward=require-match=true'
function_pattern = re.compile(r'^  func\.func @([^\s(]+)\(.*?^  }\n', re.M | re.S)


def run(text, flags=(), expected=0):
    result = subprocess.run([opt, *flags], input=text, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == expected, (flags, result.stderr)
    return result.stdout


def functions(text):
    aliases = dict(re.findall(r'^(#\w+) = (.+)$', text, re.M))
    return {m[1]: re.sub(r'#[A-Za-z_]\w*', lambda x: aliases.get(x[0], x[0]), m[0])
            for m in function_pattern.finditer(text)}


def raw_function(text, name):
    return next(m[0] for m in function_pattern.finditer(text) if m[1] == name)


def change_function(text, name, mutate):
    old = raw_function(text, name)
    new = mutate(old)
    assert old != new and text.count(old) == 1
    return text.replace(old, new)


def operands(text, name):
    matches = re.findall(r'func.call @' + re.escape(name) + r'\(([^)]*)\)', text)
    assert len(matches) == 1, (name, len(matches))
    return matches[0].split(', ')


def change_operands(text, name, mutate):
    args = operands(text, name)
    before = 'func.call @' + name + '(' + ', '.join(args) + ')'
    after = 'func.call @' + name + '(' + ', '.join(mutate(list(args))) + ')'
    assert before != after and text.count(before) == 1
    return text.replace(before, after)


def bank(text):
    body = raw_function(text, ENTRY)
    match = re.search(r'(%\w+) = memref.alloc\(\)[^\n]*hivm.cvp_query.state_bank = '
                      r'\{[^}]*role = "accumulator"', body)
    assert match
    return match[1]


def attr(text, name):
    match = re.search(re.escape(name) + r' = (\d+) : i64', text)
    assert match, name
    return int(match[1])


def anchors(text):
    return re.findall(r'hivm.hir.anchor \{id = \d+ : i64\}', text)


def insert_before_copy(text, statement):
    body = raw_function(text, ENTRY)
    call_at = body.index('func.call @' + UPDATE + '(')
    copy_at = body.index('hivm.hir.copy', call_at)
    line_start = body.rfind('\n', 0, copy_at) + 1
    indent = body[line_start:copy_at]
    return text.replace(body, body[:line_start] + indent + statement + '\n' + body[line_start:])


def consumer(text, actual):
    first = next(function_pattern.finditer(text)).start()
    declaration = '  func.func private @unexpected_consumer(memref<64x64xf32, #hivm.address_space<ub>>)\n'
    text = text[:first] + declaration + text[first:]
    return insert_before_copy(text, 'func.call @unexpected_consumer(' + actual +
                              ') : (memref<64x64xf32, #hivm.address_space<ub>>) -> ()')


staged = {}
for first, last in ((False, False), (True, False), (False, True), (True, True)):
    flags = []
    if first or last:
        flags += ['--hivm-cvp-attention-boundary=enable-first-output=' + str(first).lower() +
                  ' enable-last-output=' + str(last).lower() + ' require-match=true', '--canonicalize']
    flags += ['--hivm-cvp-query-state-copy-forward=require-match=true', '--canonicalize',
              '--hivm-cvp-query-final-output-forward=require-match=true', '--canonicalize']
    before = run(source, flags)
    staged[(first, last)] = before
    after = run(before, [DIRECT + '=require-match=true', '--canonicalize', SCALAR, '--canonicalize'])
    assert attr(after, 'hivm.cvp_query.snapshot_copies_forwarded') == 1
    assert attr(after, 'hivm.cvp_query.final_output_copies_forwarded') == 1
    assert attr(after, 'hivm.cvp_query.output_copies_forwarded') == 1
    assert attr(after, 'hivm.cvp_query.scalar_input_copies_forwarded') == 6
    assert anchors(before) == anchors(after)
    assert before.count('hivm.hir.copy') - after.count('hivm.hir.copy') == 7
    before_vfs = {k: v for k, v in functions(before).items() if 'hivm.vector_function' in v.split('\n')[0]}
    after_vfs = {k: v for k, v in functions(after).items() if 'hivm.vector_function' in v.split('\n')[0]}
    assert before_vfs == after_vfs, 'Forwarding changed a VF body'
    accumulator = bank(after)
    raw_pv = operands(before, UPDATE)[1]
    updated = operands(after, UPDATE)
    assert updated[1] == raw_pv and updated[3] == accumulator and raw_pv != accumulator
    if first:
        initial = operands(after, FIRST)
        assert initial[1] == raw_pv and initial[2] == accumulator
    if last:
        final = operands(after, LAST)
        assert final[4] == accumulator and final[7] == raw_pv
    else:
        assert operands(after, NORMAL)[4] == accumulator
    for policy in ('default', 'largest-first'):
        run(after, ['--hivm-plan-memory-regbase=disable-tightly-coupled-buffer-reuse=true '
                    'plan-memory-strategy=' + policy])
    repeated = run(after, [DIRECT])
    assert attr(repeated, 'hivm.cvp_query.output_copies_forwarded') == 0
    assert functions(repeated) == functions(after), 'Repeated pass changed a function'

first = staged[(True, False)]
last = staged[(False, True)]
first_body = raw_function(first, FIRST)
assert first_body.count('to %c64 step %c1') == 1
negatives = {
    'partial-first-rows': change_function(first, FIRST, lambda s: s.replace('to %c64 step %c1', 'to %c1 step %c1')),
    'partial-first-columns': change_function(first, FIRST, lambda s: s.replace('[1, 64]', '[1, 32]').replace('1x64xf32', '1x32xf32')),
    'cross-row-first-PV': change_function(first, FIRST, lambda s: re.sub(r'memref.subview %arg1\[%\w+, 0\]', 'memref.subview %arg1[%c0, 0]', s, count=1)),
    'missing-first-writer': re.sub(r'^.*func.call @' + FIRST + r'\([^\n]*\n', '', first, count=1, flags=re.M),
    'missing-ordinary-writer': re.sub(r'^.*func.call @' + UPDATE + r'\([^\n]*\n', '', first, count=1, flags=re.M),
    'wrong-first-destination': change_operands(first, FIRST, lambda a: a[:2] + [bank(first)]),
    'extra-PV-consumer': consumer(first, operands(first, UPDATE)[1]),
    'extra-bank-consumer': consumer(first, bank(first)),
    'fence-before-copy': insert_before_copy(first, 'hivm.hir.pipe_barrier[<PIPE_V>]'),
    'final-raw-PV-is-old-O': change_operands(last, LAST, lambda a: a[:7] + [bank(last)]),
    'final-old-O-is-raw-PV': change_operands(last, LAST, lambda a: a[:4] + [a[7]] + a[5:]),
    'partial-final-rows': change_function(last, LAST, lambda s: s.replace('to %c64 step %c1', 'to %c1 step %c1')),
    'fence-in-nonfinal-branch': insert_before_copy(last, 'hivm.hir.pipe_barrier[<PIPE_V>]'),
}


def masked_first(body):
    start = body.index('\n') + 1
    body = body[:start] + '    %cvp_mask = arith.constant dense<false> : vector<1x64xi1>\n' + body[start:]
    result, count = re.subn(r'(vector\.transfer_write [^\n]*?\[[^]]*\])', r'\1, %cvp_mask', body, count=1)
    assert count == 1
    assert '], %cvp_mask {' in result
    return result


negatives['masked-first-write'] = change_function(first, FIRST, masked_first)
for name, text in negatives.items():
    normalized = run(text)  # A malformed mutation is never a passing rejection.
    optional = run(text, [DIRECT])
    assert attr(optional, 'hivm.cvp_query.output_copies_forwarded') == 0, name
    assert functions(optional) == functions(normalized), (name, 'partial mutation')
    required = subprocess.run([opt, DIRECT + '=require-match=true'], input=text,
                              text=True, capture_output=True, timeout=120)
    assert required.returncode != 0, (name, 'required refusal missing')
    assert 'requires a proved row-local full O update and closed final read' in required.stderr, (name, required.stderr)

print('PASS: authentic control/first/last/both, unchanged VF/anchors, stock packing, idempotence; '
      + str(len(negatives)) + ' valid unsafe counterexamples rejected transactionally')

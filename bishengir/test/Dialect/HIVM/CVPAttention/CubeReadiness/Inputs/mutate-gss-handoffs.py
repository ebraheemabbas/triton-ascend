"""Create invalid local-event contracts and verify preserved GSS handoffs."""
from pathlib import Path
import argparse
import re
import sys


def handoffs(text):
    owner = None
    result = []
    for line in text.splitlines():
        function = re.match(r'\s{2}func\.func(?:\s+private)?\s+@([^ (]+)', line)
        if function:
            owner = function.group(1)
        if 'hivm.hir.sync_block_' in line:
            result.append((owner, len(line) - len(line.lstrip()), line.strip()))
    return result


def mutate(text, kind):
    lines = text.splitlines(keepends=True)
    first = next(i for i, line in enumerate(lines)
                 if re.match(r'\s{2}func\.func @_attn_fwd_mix_aic\(', line))
    last = next(i for i in range(first + 1, len(lines)) if lines[i].rstrip() == '  }')
    qk = [i for i in range(first, last) if 'hivm.hir.mmadL1' in lines[i] and 'role = "qk"' in lines[i]]
    pv = [i for i in range(first, last) if 'hivm.hir.mmadL1' in lines[i] and 'role = "pv"' in lines[i]]
    assert len(qk) == len(pv) == 1
    qk, pv = qk[0], pv[0]
    indent = re.match(r'\s*', lines[pv]).group(0)
    wait = 'hivm.hir.sync_block_wait[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 0'
    assert lines[pv - 1].strip() == wait
    if kind == 'interposed-write':
        body = ''.join(lines[first:last])
        assert '%c0_i64 = arith.constant 0 : i64' in body
        assert '%c0 = arith.constant 0 : index' in body
        assert '%alloca = memref.alloca() : memref<1xi64>' in body
        lines.insert(pv, indent + 'memref.store %c0_i64, %alloca[%c0] : memref<1xi64>\n')
    elif kind == 'unknown-direction':
        lines[pv - 1] = lines[pv - 1].replace('<PIPE_MTE3>, <PIPE_MTE1>', '<PIPE_MTE2>, <PIPE_MTE1>')
    elif kind == 'foreign-copy':
        qk_a = re.search(r'\bins\((%[^, ]+),', lines[qk]).group(1)
        pv_b = re.search(r'\bins\(%[^, ]+, (%[^, ]+),', lines[pv]).group(1)
        copy = max(i for i in range(first, pv) if 'hivm.hir.nd2nz' in lines[i])
        assert qk_a != pv_b and 'outs(' + pv_b + ' :' in lines[copy]
        # Both descriptors have the same blocked shape; the copy now targets
        # the K bank instead of the actual PV V descriptor. IR stays valid.
        lines[copy] = lines[copy].replace('outs(' + pv_b + ' :', 'outs(' + qk_a + ' :')
    else:
        raise ValueError(kind)
    return ''.join(lines)


parser = argparse.ArgumentParser()
sub = parser.add_subparsers(dest='command', required=True)
mutation = sub.add_parser('mutate')
mutation.add_argument('kind', choices=('interposed-write', 'unknown-direction', 'foreign-copy'))
mutation.add_argument('input', type=Path)
check = sub.add_parser('check')
check.add_argument('input', type=Path)
check.add_argument('output', type=Path)
args = parser.parse_args()
if args.command == 'mutate':
    sys.stdout.write(mutate(args.input.read_text(), args.kind))
else:
    before = handoffs(args.input.read_text())
    after = handoffs(args.output.read_text())
    assert before and before == after, 'GSS handoff owner/order/flag/indentation changed'

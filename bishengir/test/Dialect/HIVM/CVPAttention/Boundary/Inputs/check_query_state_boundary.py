"""Exercise real bank/snapshot graphs through the boundary pass, without NPU use."""
from pathlib import Path
import argparse
import itertools
import re
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--opt', required=True)
p.add_argument('--input', type=Path, required=True)
a = p.parse_args()
source = a.input.read_text()
diagnostic = 'CVP boundary specialization requires a proved Skew logical boundary'


def replace(text, before, after):
    # Anchor IDs are local to each core function. Mutate only the captured
    # AIV function; the earlier AIC function can legitimately reuse an ID.
    start = text.index('  func.func @_attn_fwd_mix_aiv(')
    assert text.count(before, start) == 1, (before, text.count(before, start))
    at = text.index(before, start)
    return text[:at] + after + text[at + len(before):]


def run(path, *, first=True, last=True, state=True, require=True, transform=True):
    flags = ['--verify-each']
    if transform:
        flags.append('--hivm-cvp-attention-boundary=' + ' '.join([
            f'require-match={str(require).lower()}',
            f'enable-first-output={str(first).lower()}',
            f'enable-last-output={str(last).lower()}',
            f'enable-first-state={str(state).lower()}',
            'retain-provenance=true']))
    return subprocess.run([a.opt, str(path), *flags], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


with tempfile.TemporaryDirectory(prefix='cvp-query-boundary-') as tmp:
    tmp = Path(tmp)
    original = tmp/'original.mlir'
    original.write_text(source)
    baseline = run(original, transform=False)
    assert baseline.returncode == 0, baseline.stderr
    positives = 0
    for first, last, state in itertools.product((False, True), repeat=3):
        if not any((first, last, state)):
            continue
        result = run(original, first=first, last=last, state=state)
        assert result.returncode == 0, ((first, last, state), result.stderr)
        for enabled, count, kind in (
            (first, 'first_output_specializations', 'query_first_output'),
            (last, 'last_output_specializations', 'query_last_output'),
            (state, 'first_state_specializations', 'query_first_state')):
            assert (f'hivm.cvp_attention.{count} = 1 : i64' in result.stdout) == enabled
            assert (f'hivm.cvp_attention.boundary = "{kind}"' in result.stdout) == enabled
        assert 'hivm.cvp_query.state_storage' in result.stdout
        path = tmp/f'positive-{positives}.mlir'
        path.write_text(result.stdout)
        repeated = run(path, first=first, last=last, state=state, require=False)
        assert repeated.returncode == 0, repeated.stderr
        assert repeated.stdout == result.stdout, 'Specialization must not duplicate on a second pass'
        positives += 1

    ub = '#hivm.address_space<ub>'
    matrix = f'memref<64x64xf32, {ub}>'
    flat = f'memref<4096xf32, {ub}>'
    o_start = 'scope.scope : () -> () {\n          %alloc_15 ='
    update = ('          func.call @_attn_fwd_mix_aiv_fused_6_outlined_vf_0'
              '(%23, %alloc_12, %38, %alloc_12)')
    assert source.count(o_start) == source.count(update) == 1
    write = (f'          %collapse_shape_20 = memref.collapse_shape %alloc_12 [[0, 1]] : '
             f'{matrix} into {flat}\n'
             f'          hivm.hir.copy ins(%collapse_shape_20 : {flat}) '
             f'outs(%collapse_shape_11 : {flat})\n')
    assert source.count(write) == 1
    init = (f'      hivm.hir.copy ins(%collapse_shape : {flat}) '
            f'outs(%collapse_shape_11 : {flat})\n')
    assert source.count(init) == 1
    mutations = {
        'false-unique-marker': replace(source,
            f'annotation.mark %alloc_6 {{mem_unique}} : {matrix}',
            f'annotation.mark %alloc_6 {{mem_unique = false}} : {matrix}'),
        'early-unknown-raw-capture': replace(source,
            '        hivm.hir.anchor {id = 53 : i64}',
            f'        func.call @unexpected_raw(%alloc_12) : ({matrix}) -> ()\n'
            '        hivm.hir.anchor {id = 53 : i64}'),
        'raw-allocated-outside-stream': replace(replace(source,
            f'        %alloc_12 = memref.alloc() : {matrix}\n', ''),
            '      scf.for %arg11 = %c0_i64 to %13 step %c256_i64',
            f'      %alloc_12 = memref.alloc() : {matrix}\n'
            '      scf.for %arg11 = %c0_i64 to %13 step %c256_i64'),
        'missing-state-marker': replace(source, ', hivm.cvp_query.state_storage,', ','),
        'missing-unique': replace(source,
            f'      annotation.mark %alloc_6 {{mem_unique}} : {matrix}\n', ''),
        'wrong-bank-slots': replace(source,
            'clock = "O", required_slots = 1 : i32, role = "accumulator"',
            'clock = "O", required_slots = 2 : i32, role = "accumulator"'),
        'wrong-bank-clock': replace(source,
            'clock = "O", required_slots = 1 : i32, role = "accumulator"',
            'clock = "query", required_slots = 1 : i32, role = "accumulator"'),
        'wrong-local-boundary': replace(source,
            '%22 = arith.cmpi eq, %15, %c256_i64', '%22 = arith.cmpi eq, %15, %c0_i64'),
        'foreign-bank-read': replace(source, o_start,
            'scope.scope : () -> () {\n'
            f'          %foreign_snapshot = memref.alloc() : {flat}\n'
            f'          hivm.hir.copy ins(%collapse_shape_11 : {flat}) '
            f'outs(%foreign_snapshot : {flat})\n          %alloc_15 ='),
        'duplicate-bank-write': replace(source, write,
            f'          hivm.hir.copy ins(%alloc_17 : {flat}) '
            f'outs(%collapse_shape_11 : {flat})\n' + write),
        'changed-snapshot': replace(source, update,
            f'          hivm.hir.copy ins(%collapse_shape_11 : {flat}) '
            f'outs(%alloc_15 : {flat})\n' + update),
        'escaping-snapshot': replace(source, update,
            f'          func.call @unexpected(%alloc_18) : ({matrix}) -> ()\n' + update),
        'raw-view-escape': replace(source,
            f'        %alloc_12 = memref.alloc() : {matrix}\n',
            f'        %alloc_12 = memref.alloc() : {matrix}\n'
            '        %raw_alias = memref.subview %alloc_12[0, 0] [64, 64] [1, 1] : '
            f'{matrix} to memref<64x64xf32, strided<[64, 1]>, {ub}>\n'),
        'raw-late-read': replace(replace(source, o_start,
            f'scope.scope : () -> () {{\n          %foreign_raw = memref.alloc() : {flat}\n'
            '          %alloc_15 ='), write, write +
            f'          hivm.hir.copy ins(%collapse_shape_20 : {flat}) '
            f'outs(%foreign_raw : {flat})\n'),
        'write-before-update': replace(replace(source, write, ''), update, write + update),
        'duplicate-bank-initialization': replace(source, init, init + init),
    }
    mutations['escaping-snapshot'] = re.sub(r'\n}\s*$',
        f'\n  func.func private @unexpected({matrix})\n}}\n', mutations['escaping-snapshot'])
    mutations['early-unknown-raw-capture'] = re.sub(r'\n}\s*$',
        f'\n  func.func private @unexpected_raw({matrix})\n}}\n',
        mutations['early-unknown-raw-capture'])
    for name, text in mutations.items():
        path = tmp/(name+'.mlir')
        path.write_text(text)
        valid = run(path, transform=False)
        assert valid.returncode == 0, (name, 'invalid fixture', valid.stderr)
        rejected = run(path)
        assert rejected.returncode != 0 and diagnostic in rejected.stderr, (name, rejected.stderr)
        unchanged = run(path, require=False)
        assert unchanged.returncode == 0 and unchanged.stdout == valid.stdout, (name, 'partial rewrite')
    print(f'Explicit query-state boundary cases: {positives}; valid adversarial graphs rejected transactionally: {len(mutations)}')

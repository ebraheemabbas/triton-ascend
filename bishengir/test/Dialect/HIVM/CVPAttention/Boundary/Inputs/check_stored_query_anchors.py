"""Check the captured Boundary/delayed-GSS contract and closed stored graphs."""
from pathlib import Path
import argparse
import itertools
import re
import subprocess
import tempfile


def inventory(text):
    """Record marker descriptors and enclosing anchored region identities."""
    functions = {}
    name = None
    stack = []
    for line in text.splitlines():
        statement = line.strip()
        if statement.startswith('func.func '):
            name = re.search(r'@([^ (]+)', statement)[1]
            functions[name] = []
            stack = [dict(kind='function', ids=[])]
            continue
        if name is None:
            continue
        if statement.startswith('}'):
            frame = stack.pop()
            if frame['kind'] == 'scope':
                phase = re.search(r'hivm.preload_num = (\d+)', statement)
                frame['phase'] = int(phase[1]) if phase else None
            if 'else {' in statement:
                stack.append(dict(kind='else', ids=[]))
            elif frame['kind'] == 'function':
                name = None
            continue
        marker = re.fullmatch(r'hivm.hir.anchor\s*\{id = (\d+) : i64\}', statement)
        if marker:
            key = int(marker[1])
            for frame in stack:
                frame['ids'].append(key)
            functions[name].append((statement, stack[1:].copy()))
        if statement.endswith('{'):
            kind = ('for' if 'scf.for ' in statement else
                    'if' if 'scf.if ' in statement else
                    'scope' if 'scope.scope' in statement else 'other')
            stack.append(dict(kind=kind, ids=[]))
    return {name: [(descriptor, [(f['kind'], f.get('phase'), tuple(f['ids']))
                                for f in frames])
                   for descriptor, frames in rows]
            for name, rows in functions.items() if rows}


def replace_aiv(text, before, after):
    start = text.index('  func.func @_attn_fwd_mix_aiv(')
    assert text.count(before, start) == 1, before
    at = text.index(before, start)
    return text[:at] + after + text[at + len(before):]


def run(opt, path, *, first=True, last=True, state=True, require=True,
        transform=True):
    flags = ['--verify-each']
    if transform:
        flags.append('--hivm-cvp-attention-boundary=' + ' '.join([
            f'require-match={str(require).lower()}',
            f'enable-first-output={str(first).lower()}',
            f'enable-last-output={str(last).lower()}',
            f'enable-first-state={str(state).lower()}', 'retain-provenance=true']))
    return subprocess.run([opt, str(path), *flags], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def remove_false_anchor(source, remove_else):
    # Change MIX, AIC and AIV consistently. IDs remain consecutive in each
    # core; the negative is not an arbitrary mismatch against the frozen MIX.
    text, count = re.subn(
        r'(?m)^[ ]*hivm.hir.anchor \{id = 71 : i64\}\n', '', source)
    assert count == 3, count
    if remove_else:
        # The frozen tensor MIX if has results and must keep its yield/else.
        # Both no-result live-core else blocks can be absent in valid SCF.
        text, count = re.subn(r'(?m)^([ ]*)} else \{\n\1}', r'\1}', text)
        assert count == 2, count
    return re.sub(r'(hivm.hir.anchor \{id = )(\d+)( : i64\})',
                  lambda m: m[1] + str(int(m[2]) - (int(m[2]) > 71)) + m[3], text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--opt', required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--expect-old-regression', action='store_true')
    a = p.parse_args()
    source = a.input.read_text()
    diagnostic = 'CVP boundary specialization requires a proved Skew logical boundary'
    with tempfile.TemporaryDirectory(prefix='cvp-stored-anchor-') as directory:
        tmp = Path(directory)
        original = tmp / 'original.mlir'
        original.write_text(source)
        baseline = run(a.opt, original, transform=False)
        assert baseline.returncode == 0, baseline.stderr
        reference = inventory(baseline.stdout)
        assert set(reference) == {'_attn_fwd_backup', '_attn_fwd_mix_aic',
                                  '_attn_fwd_mix_aiv'}
        assert all(len(rows) == 76 for rows in reference.values())
        if a.expect_old_regression:
            result = run(a.opt, original)
            assert result.returncode == 0, result.stderr
            actual = inventory(result.stdout)
            assert actual['_attn_fwd_backup'] == reference['_attn_fwd_backup']
            assert actual['_attn_fwd_mix_aic'] == reference['_attn_fwd_mix_aic']
            assert actual['_attn_fwd_mix_aiv'] != reference['_attn_fwd_mix_aiv']
            print('Original stored-query anchor regression reproduced')
            return
        positives = 0
        for first, last, state in itertools.product((False, True), repeat=3):
            if not any((first, last, state)):
                continue
            result = run(a.opt, original, first=first, last=last, state=state)
            assert result.returncode == 0, ((first, last, state), result.stderr)
            assert inventory(result.stdout) == reference, 'Marker owner/order changed'
            for enabled, count, kind in (
                (first, 'first_output_specializations', 'query_first_output'),
                (last, 'last_output_specializations', 'query_last_output'),
                (state, 'first_state_specializations', 'query_first_state')):
                assert (f'hivm.cvp_attention.{count} = 1 : i64' in result.stdout) == enabled
                assert (f'hivm.cvp_attention.boundary = "{kind}"' in result.stdout) == enabled
            path = tmp / f'positive-{positives}.mlir'
            path.write_text(result.stdout)
            repeated = run(a.opt, path, first=first, last=last, state=state,
                           require=False)
            assert repeated.returncode == 0, repeated.stderr
            assert repeated.stdout == result.stdout, 'Specialization duplicated'
            positives += 1

        ub = '#hivm.address_space<ub>'
        matrix = f'memref<64x128xf32, {ub}>'
        flat = f'memref<8192xf32, {ub}>'
        mutations = {
            'absent-else-anchor': remove_false_anchor(source, remove_else=True),
            'unanchored-else': remove_false_anchor(source, remove_else=False),
            'escaping-tail-definition': replace_aiv(source,
                '          hivm.hir.anchor {id = 72 : i64}',
                f'          annotation.mark %collapse_shape_18 {{test_escape}} : {flat}\n'
                '          hivm.hir.anchor {id = 72 : i64}'),
            'unknown-raw-capture': replace_aiv(source,
                '          hivm.hir.anchor {id = 61 : i64}',
                '          hivm.hir.anchor {id = 61 : i64}\n'
                f'          func.call @unexpected_raw(%alloc_12) : ({matrix}) -> ()'),
        }
        mutations['unknown-raw-capture'] = re.sub(r'\n}\s*$',
            f'\n  func.func private @unexpected_raw({matrix})\n}}\n',
            mutations['unknown-raw-capture'])
        for name, text in mutations.items():
            path = tmp / (name + '.mlir')
            path.write_text(text)
            valid = run(a.opt, path, transform=False)
            assert valid.returncode == 0, (name, 'invalid fixture', valid.stderr)
            rejected = run(a.opt, path)
            assert rejected.returncode != 0 and diagnostic in rejected.stderr, (name, rejected.stderr)
            unchanged = run(a.opt, path, require=False)
            assert unchanged.returncode == 0 and unchanged.stdout == valid.stdout, (name, 'partial rewrite')
            if name in ('absent-else-anchor', 'unanchored-else'):
                first_only = run(a.opt, path, first=True, last=False, state=False)
                assert first_only.returncode == 0, (name, 'first-only changed', first_only.stderr)
        print(f'Stored query anchor cases: {positives}; valid adversarial graphs rejected transactionally: {len(mutations)}')


if __name__ == '__main__':
    main()

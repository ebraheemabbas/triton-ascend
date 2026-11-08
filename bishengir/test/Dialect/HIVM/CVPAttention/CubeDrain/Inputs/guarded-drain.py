"""Bounded, valid-IR mutations and preservation checks for the GSS drain case."""
from pathlib import Path
import re
import sys


def function(text, core):
    start = text.index('  func.func @_attn_fwd_mix_ai' + core + '(')
    end = re.search(r'^  }\s*$', text[start:], re.M)
    assert end
    return start, start + end.end(), text[start:start + end.end()]


def stream(text):
    match = re.search(r'^      scf.for [^\n]+ : i64 \{\n', text, re.M)
    assert match
    end = re.search(r'^      }\s*$', text[match.end():], re.M)
    assert end
    return match, text[match.end():match.end() + end.start()]


def alpha(text):
    names = {}
    def token(match):
        return names.setdefault(match[0], '%v' + str(len(names)))
    return re.sub(r'%[-\w.$]+', token, text).strip()


def preserve(before, after):
    _, _, cube_before = function(before, 'c')
    _, _, cube_after = function(after, 'c')
    assert alpha(stream(cube_before)[1]) == alpha(stream(cube_after)[1]), 'Cube body changed'
    assert alpha(function(before, 'v')[2]) == alpha(function(after, 'v')[2]), 'Vector changed'
    assert 'hivm.cvp_cube.credit_drain_loop' not in after
    assert after.count('hivm.cvp_cube.derived_drain = "guarded-cross-v1"') == 1


def change_one(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def domain(text, total):
    # Change only the CPU proof fixture's scalar work domain. Do not change the
    # shared physical-pointer constants or claim a runnable attention kernel.
    header_end = text.index('\n') + 1
    text = text[:header_end] + (
        '    %test_total = arith.constant ' + str(total) + ' : i64\n'
        '    %test_start = arith.constant 0 : i32\n') + text[header_end:]
    text, count = re.subn(r'(arith.subi )%c131072_i64(, %[-\w.$]+ : i64)',
                          r'\1%test_total\2', text)
    assert count == 1
    text, count = re.subn(r'(arith.extsi )%[-\w.$]+( \{hivm.cvp_query.coordinate = "query_start"})',
                          r'\1%test_start\2', text)
    assert count == 1
    return text


def mutate(text, mode):
    a, b, cube = function(text, 'c')
    if mode in ('zero-work', 'one-query', 'three-queries'):
        total = {'zero-work': 0, 'one-query': 1, 'three-queries': 57}[mode]
        for core in ('v', 'c'):
            a, b, body = function(text, core)
            text = text[:a] + domain(body, total) + text[b:]
        return text
    if mode == 'tail-barrier':
        m, _ = stream(cube)
        cube = cube[:m.end()] + '        hivm.hir.pipe_barrier[<PIPE_ALL>]\n' + cube[m.end():]
    elif mode == 'unguarded-handoff':
        m, _ = stream(cube)
        cube = cube[:m.end()] + '        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4\n' + cube[m.end():]
    elif mode == 'missing-return':
        cube = change_one(cube,
            '    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4\n', '')
    elif mode == 'foreign-sibling-handoff':
        m, _ = stream(cube)
        extra = ('      scf.for %test_iv = %c0_i64 to %c1_i64 step %c1_i64 : i64 {\n'
                 '        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 4\n'
                 '      }\n')
        cube = cube[:m.start()] + extra + cube[m.start():]
    elif mode == 'observable-counter':
        load = re.search(r'^        (%[-\w.$]+) = memref.load %alloca\[%c0\] : memref<1xi64>\n', cube, re.M)
        assert load
        cube = cube[:load.end()] + '        %test_observable = arith.cmpi eq, ' + load[1] + ', %c0_i64 : i64\n' + cube[load.end():]
    elif mode == 'odd-extra':
        # W+768 keeps valid IR, but removes one scheduler tick for nonzero W
        # and three for zero W. Both cases must reject.
        return text.replace('%c1024_i64 = arith.constant 1024 : i64',
                            '%c1024_i64 = arith.constant 768 : i64')
    elif mode == 'wrong-main-bound':
        m, _ = stream(cube)
        upper = re.search(r' to (%[-\w.$]+) step ', m[0])
        assert upper
        selected = re.search(r'^      ' + re.escape(upper[1]) + r' = arith.select (%[-\w.$]+), (%[-\w.$]+), (%[-\w.$]+) : i64$', cube, re.M)
        assert selected
        cube = cube[:selected.start(3)] + '%c256_i64' + cube[selected.end(3):]
    elif mode == 'wrong-policy':
        cube = change_one(cube, 'hivm.cvp_cube.derived_drain = "guarded-cross-v1"',
                          'hivm.cvp_cube.derived_drain = "cross-only-v1"')
    elif mode == 'vector-work-mismatch':
        va, vb, vector = function(text, 'v')
        vector = domain(vector, 57)
        return text[:va] + vector + text[vb:]
    else:
        raise ValueError(mode)
    assert cube != text[a:b]
    return text[:a] + cube + text[b:]


if __name__ == '__main__':
    action, *args = sys.argv[1:]
    if action == 'preserve':
        preserve(*(Path(path).read_text() for path in args))
    elif action == 'mutate':
        mode, source, destination = args
        before = Path(source).read_text()
        after = mutate(before, mode)
        assert after != before
        Path(destination).write_text(after)
    else:
        raise ValueError(action)

"""Corrupt only explicitly selected fields of a generated, valid drain fixture."""
from pathlib import Path
import re
import sys

mode, source, destination = sys.argv[1:]
text = Path(source).read_text()
start = text.index('  func.func @_attn_fwd_mix_aic(')
close = re.search(r'^  }\s*$', text[start:], re.M)
assert close
end = start + close.end()
cube = text[start:end]
marker = '} {hivm.cvp_cube.credit_drain_loop}'
assert cube.count(marker) == 1
suffix_end = cube.index(marker)
suffix_start = cube.rfind('      scf.for ', 0, suffix_end)
assert suffix_start >= 0
suffix = cube[suffix_start:suffix_end]
wait = '        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1\n'
ready = '        hivm.hir.sync_block_set[<CUBE>, <PIPE_FIX>, <PIPE_V>] flag = 1\n'
if mode == 'drop-suffix-signal':
    assert suffix.count(wait) == 1
    changed = suffix.replace(wait, '', 1)
elif mode == 'reorder-suffix-signals':
    assert wait + ready in suffix
    changed = suffix.replace(wait + ready, ready + wait, 1)
elif mode == 'truncate-suffix':
    bound = re.search(r'(scf.for %[^ ]+ = )(%[^ ]+)( to )(%[^ ]+)', suffix)
    assert bound
    changed = suffix[:bound.start(4)] + bound.group(2) + suffix[bound.end(4):]
elif mode == 'drop-both-cube-signals':
    target = '\n        hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 0\n'
    assert cube.count(target) == 2
    cube = cube.replace(target, '\n')
    changed = None
elif mode == 'drop-return-credit':
    target = '\n    hivm.hir.sync_block_wait[<CUBE>, <PIPE_V>, <PIPE_FIX>] flag = 1\n'
    assert cube.count(target) == 1
    cube = cube.replace(target, '\n', 1)
    changed = None
else:
    raise ValueError(mode)
if changed is not None:
    cube = cube[:suffix_start] + changed + cube[suffix_end:]
assert cube != text[start:end]
Path(destination).write_text(text[:start] + cube + text[end:])

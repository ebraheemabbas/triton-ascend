"""Compare the complete printed frozen backup and require live provenance cleanup."""
from pathlib import Path
import difflib
import sys

def functions(path):
    lines = Path(path).read_text().splitlines()
    result = []
    for start, line in enumerate(lines):
        if not line.startswith('  func.func '):
            continue
        end = next(i for i in range(start + 1, len(lines)) if lines[i] == '  }')
        result.append('\n'.join(lines[start:end + 1]))
    return result

before = functions(sys.argv[1])
after = functions(sys.argv[2])
old_backups = [f for f in before if 'hivm.backup_function' in f.splitlines()[0]]
new_backups = [f for f in after if 'hivm.backup_function' in f.splitlines()[0]]
assert len(old_backups) == len(new_backups) == 1
assert 'hivm.cvp_attention.verified' in old_backups[0].splitlines()[0]
if old_backups != new_backups:
    raise AssertionError('Frozen backup changed:\n' + '\n'.join(difflib.unified_diff(
        old_backups[0].splitlines(), new_backups[0].splitlines(),
        fromfile='before', tofile='after')))
for function in after:
    header = function.splitlines()[0]
    if 'hivm.part_of_mix' in header and 'hivm.backup_function' not in header:
        assert 'hivm.cvp_attention.verified' not in header, 'Live-kernel provenance was not cleaned up'

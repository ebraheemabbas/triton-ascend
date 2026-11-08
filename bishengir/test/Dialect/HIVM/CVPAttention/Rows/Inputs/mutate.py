"""Verified-input mutations for the independent-row scheduling contract."""
from pathlib import Path
import re,sys

mode=sys.argv[1];text=Path(sys.argv[2]).read_text()
if mode=='renamed':
    text=re.sub(r'([%@])([A-Za-z][A-Za-z0-9_]*)',lambda m:m[1]+'renamed_'+m[2],text)
elif mode=='odd':
    text=text.replace('arith.constant 64 : index','arith.constant 63 : index')
elif mode=='effect':
    text=text.replace('module {','module {\n  func.func private @side_effect()',1)
    text=text.replace('scf.for %row = %c0 to %c64 step %c1 {',
                      'scf.for %row = %c0 to %c64 step %c1 {\n      func.call @side_effect() : () -> ()')
elif mode=='overlap':
    # The input remains valid MLIR, but distinct rows alias physically.
    text=text.replace('memref<64x64xf32, #hivm.address_space<ub>>',
                      'memref<64x64xf32, strided<[1, 1]>, #hivm.address_space<ub>>')
    text=text.replace('strided<[64, 1], offset: ?>','strided<[1, 1], offset: ?>')
elif mode=='carry':
    text=text.replace('    scf.for %row = %c0 to %c64 step %c1 {',
                      '    %zero = arith.constant dense<0.0> : vector<1x64xf32>\n'
                      '    %result = scf.for %row = %c0 to %c64 step %c1 iter_args(%carry = %zero) -> (vector<1x64xf32>) {')
    text=text.replace('%updated = arith.addf %pv_value, %scaled',
                      '%with_carry = arith.addf %pv_value, %carry : vector<1x64xf32>\n'
                      '      %updated = arith.addf %with_carry, %scaled')
    text=text.replace('    }\n    return\n  }','      scf.yield %updated : vector<1x64xf32>\n    }\n    return\n  }')
elif mode=='no-route':
    text=re.sub(r', hivm\.cvp_attention\.verified = \[\{[^\n]+?\}\]', '', text)
else:
    raise ValueError(mode)
sys.stdout.write(text)

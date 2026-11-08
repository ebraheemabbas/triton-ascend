"""Well-formed counterexamples to the final write-forwarding ownership proof."""
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
# Drop RUN/check comments so each mutation changes executable IR only.
text = '\n'.join(line for line in text.splitlines() if not line.startswith('//'))+'\n'
mode = sys.argv[2]
call = '        func.call @convert(%input, %temporary) {hivm.vector_function} : (memref<4x64xf32, #ub>, memref<4x64xf16, #ub>) -> ()'
copy = '        hivm.hir.copy ins(%temporary : memref<4x64xf16, #ub>) outs(%selected : memref<4x64xf16, #ub>)'
if mode == 'rename':
    text = text.replace('@convert','@renamed_convert').replace('%temporary','%private_output')
elif mode == 'stronger-alignment':
    text = text.replace('alignment = 64 : i64','alignment = 128 : i64')
elif mode == 'read-output':
    text = text.replace('      %halves = arith.truncf',
        '      %old = vector.transfer_read %outrow[%c0, %c0], %halfzero {in_bounds = [true, true]} : memref<1x64xf16, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf16>\n      %halves = arith.truncf')
    text = text.replace('    %zero = arith.constant 0.0 : f32',
        '    %zero = arith.constant 0.0 : f32\n    %halfzero = arith.constant 0.0 : f16')
elif mode == 'partial-rows':
    text = text.replace('scf.for %row = %c0 to %c4','scf.for %row = %c0 to %c3')
elif mode == 'partial-columns':
    text = text.replace('[1, 64]','[1, 32]').replace('1x64x','1x32x')
elif mode == 'temporary-escape':
    text = text.replace(copy,copy+'\n        func.call @opaque(%temporary) : (memref<4x64xf16, #ub>) -> ()')
elif mode == 'unknown-between':
    text = text.replace(call,call+'\n        func.call @global_effect() : () -> ()')
elif mode == 'fence-between':
    text = text.replace(call,call+'\n        hivm.hir.pipe_barrier[<PIPE_V>]')
elif mode == 'wrong-clock':
    text = text.replace('clock = "query"','clock = "O"')
elif mode == 'same-bank':
    text = text.replace('%selected = arith.select %slot, %bank1, %bank0','%selected = arith.select %slot, %bank0, %bank0')
elif mode == 'missing-storage':
    text = text.replace(', hivm.cvp_query.output_storage','')
elif mode == 'bank-escape':
    text = text.replace(copy,copy+'\n        func.call @opaque(%bank0) : (memref<4x64xf16, #ub>) -> ()')
elif mode == 'later-condition':
    text = text.replace(call,call+'\n        %later = arith.xori %slot, %true : i1')
    text = text.replace('%selected = arith.select %slot,','%selected = arith.select %later,')
elif mode in ('scoped-input', 'scoped-input-unknown'):
    allocation = '    %input = memref.alloc() : memref<4x64xf32, #ub>'
    assert text.count(allocation) == 1
    body = ('    %input = scope.scope : () -> memref<4x64xf32, #ub> {\n'
            '      %private_input = memref.alloc() : memref<4x64xf32, #ub>\n')
    if mode == 'scoped-input-unknown':
        body += ('      %unknown_region = scf.if %slot -> memref<4x64xf32, #ub> {\n'
                 '        scf.yield %private_input : memref<4x64xf32, #ub>\n'
                 '      } else {\n'
                 '        scf.yield %private_input : memref<4x64xf32, #ub>\n'
                 '      }\n')
    body += ('      scope.return %' + ('unknown_region' if mode.endswith('unknown') else 'private_input') +
             ' : memref<4x64xf32, #ub>\n    }')
    text = text.replace(allocation, body)
elif mode in ('scoped-extra-fresh', 'scoped-extra-alias', 'scoped-extra-select', 'scoped-extra-view'):
    formal = '%output: memref<4x64xf16, #ub>)'
    assert text.count(formal) == 1 and text.count(call) == 1
    text = text.replace(formal, '%output: memref<4x64xf16, #ub>, %extra: memref<4x64xf16, #ub>)')
    extra_call = ('        func.call @convert(%input, %temporary, %scoped_extra) {hivm.vector_function} : '
                  '(memref<4x64xf32, #ub>, memref<4x64xf16, #ub>, memref<4x64xf16, #ub>) -> ()')
    text = text.replace(call, extra_call)
    scope = ('    %scoped_extra = scope.scope : () -> memref<4x64xf16, #ub> {\n')
    root = '%bank0'
    if mode in ('scoped-extra-fresh', 'scoped-extra-select'):
        scope += '      %fresh_extra = memref.alloc() : memref<4x64xf16, #ub>\n'
        root = '%fresh_extra'
    if mode == 'scoped-extra-select':
        scope += '      %alias_choice = arith.select %slot, %bank0, %fresh_extra : memref<4x64xf16, #ub>\n'
        root = '%alias_choice'
    elif mode == 'scoped-extra-view':
        scope += ('      %flat_alias = memref.collapse_shape %bank0 [[0, 1]] : memref<4x64xf16, #ub> into memref<256xf16, #ub>\n'
                  '      %whole_alias = memref.expand_shape %flat_alias [[0, 1]] output_shape [4, 64] : memref<256xf16, #ub> into memref<4x64xf16, #ub>\n')
        root = '%whole_alias'
    scope += '      scope.return ' + root + ' : memref<4x64xf16, #ub>\n    }\n'
    assert text.count('    scf.for %key =') == 1
    text = text.replace('    scf.for %key =', scope + '    scf.for %key =')
else:
    raise ValueError(mode)
print(text,end='')

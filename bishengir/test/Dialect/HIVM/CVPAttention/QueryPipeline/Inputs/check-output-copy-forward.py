"""Exercise the actual pass's row-alias proof and updated-O use closure."""
from pathlib import Path
import re
import subprocess
import sys

source=Path(sys.argv[1]).read_text()
opt=sys.argv[2]
flag='--hivm-cvp-query-output-copy-forward'

def replace(text,old,new,count=1):
    assert text.count(old)==count,(old,text.count(old))
    return text.replace(old,new)

def run(text,required=False):
    args=[opt,flag+('=require-match' if required else '')]
    return subprocess.run(args,input=text,text=True,capture_output=True,timeout=60)

def inspect_positive(text, has_initialization=False):
    result=run(text,True)
    assert result.returncode==0,result.stderr
    assert 'hivm.cvp_query.output_copies_forwarded = 1 : i64' in result.stdout
    body=result.stdout.split('func.func @case(',1)[1]
    bank=re.search(r'(%[\w]+) = memref.alloc\(\) \{hivm.cvp_query.state_bank',body).group(1)
    args=re.search(r'call @(?:update|renamed_update)\(([^)]*)\)',body).group(1).split(', ')
    finish=re.search(r'call @finish\(([^)]*)\)',body).group(1).split(', ')
    assert len(args)==4 and args[3]==bank and args[1]!=bank
    assert len(finish)==6 and finish[4]==bank
    # The first-key select still selects the independent zero root. It is not
    # the new destination, and its false arm is the persistent bank.
    selected=re.search(r'(%[\w]+) = arith.select %arg0, (%[\w]+), '+re.escape(bank),body)
    assert selected and selected.group(1)==args[2] and selected.group(2)!=bank
    assert body.count('hivm.hir.copy') == int(has_initialization)
    assert 'hivm.multi_buffer = 2 : i32' in body
    assert 'hivm.preload_local_buffer = 1 : i32' in body
    assert 'hivm.hir.anchor {id = 62 : i64}' in body

inspect_positive(source)
inspect_positive(source.replace('@update','@renamed_update'))

init = ('    %flat_zero = memref.collapse_shape %zero [[0, 1]] : memref<4x64xf32, #ub> into memref<256xf32, #ub>\n'
        '    %flat_bank = memref.collapse_shape %bank [[0, 1]] : memref<4x64xf32, #ub> into memref<256xf32, #ub>\n'
        '    hivm.hir.copy ins(%flat_zero : memref<256xf32, #ub>) outs(%flat_bank : memref<256xf32, #ub>)\n')
initialized = replace(source, '    scf.for %key =', init + '    scf.for %key =')
scope = ('      %alpha_value = scope.scope : () -> memref<4xf32, #ub> {\n'
         '        %alpha_bank = memref.alloc() : memref<4xf32, #ub>\n'
         '        annotation.mark %alpha_bank {hivm.multi_buffer = 3 : i32} : memref<4xf32, #ub>\n'
         '        scope.return %alpha_bank : memref<4xf32, #ub>\n'
         '      } {hivm.loop_core_type = #hivm.tcore_type<VECTOR>, hivm.preload_num = 2 : i32, hivm.max_preload_num = 4 : i32}\n')
def scoped_alpha(text):
    return replace(replace(text, '      scope.scope : () -> () {', scope + '      scope.scope : () -> () {'),
                   'func.call @update(%alpha,', 'func.call @update(%alpha_value,')
inspect_positive(initialized, True)
inspect_positive(scoped_alpha(source))
inspect_positive(scoped_alpha(initialized), True)

copy='        hivm.hir.copy ins(%pv : memref<4x64xf32, #ub>) outs(%bank : memref<4x64xf32, #ub>)\n'
finish='          func.call @finish(%scalar, %scalar, %scalar, %scalar, %pv, %out)'
late_read=('      %late = vector.transfer_read %orow[%c0, %c0], %zero '
           '{in_bounds = [true, true]} : memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>, vector<1x64xf32>\n')
negatives={
 'partial-rows':source.replace('scf.for %row = %c0 to %c4 step %c1 {',
                              'scf.for %row = %c0 to %c3 step %c1 {',1),
 'cross-row-old-O':replace(source,'memref.subview %old[%row, 0]',
                          'memref.subview %old[%c0, 0]'),
 'cross-row-output':replace(source,'memref.subview %out[%row, 0]',
                           'memref.subview %out[%c0, 0]',2),
 'extra-old-O-read-after-store':replace(source,'    }\n    return\n  }\n  func.func @finish',
                                       late_read+'    }\n    return\n  }\n  func.func @finish'),
 'hidden-effect':replace(source,'      %scaled = arith.mulf',
                         '      func.call @effect() : () -> ()\n      %scaled = arith.mulf'),
 'intervening-effect':replace(source,copy,'        func.call @effect() : () -> ()\n'+copy),
 'unknown-PV-consumer':replace(source,copy,copy+
     '        func.call @opaque(%pv) : (memref<4x64xf32, #ub>) -> ()\n'),
 'unknown-bank-consumer':replace(source,copy,copy+
     '        func.call @opaque(%bank) : (memref<4x64xf32, #ub>) -> ()\n'),
 'duplicate-copyback':replace(source,copy,copy+copy),
 'missing-unique':replace(source,'    annotation.mark %bank {mem_unique} : memref<4x64xf32, #ub>\n',''),
 'wrong-clock':replace(source,'clock = "O"','clock = "query"'),
 'nonstored':replace(source,', hivm.cvp_query.state_storage, hivm.cvp_query.output_storage',''),
 'unroll':replace(source,'pipeline_mode = "skew"','pipeline_mode = "unroll"'),
 'U2':replace(source,'key_unroll = 1 : i32','key_unroll = 2 : i32'),
 'wrong-phase':replace(source,'hivm.preload_num = 0 : i32','hivm.preload_num = 1 : i32'),
 'wrong-depth':replace(source,'hivm.max_preload_num = 4 : i32','hivm.max_preload_num = 3 : i32'),
 'wrong-old-selection':replace(source,'arith.select %first, %zero, %bank',
                              'arith.select %first, %bank, %zero'),
 'missing-final-read':replace(source,finish,finish.replace('%pv, %out','%zero, %out')),
 'writable-final-input':replace(source,'      %half = arith.truncf %value',
     '      vector.transfer_write %value, %inrow[%c0, %c0] {in_bounds = [true, true]} : vector<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>, #ub>\n      %half = arith.truncf %value'),
 'alias-alpha':replace(replace(source,'        %old = arith.select',
     '        %alpha_alias = memref.subview %bank[0, 0] [1, 4] [1, 1] : memref<4x64xf32, #ub> to memref<4xf32, #ub>\n        %old = arith.select'),
     'func.call @update(%alpha,','func.call @update(%alpha_alias,'),
 'noalias-formal':replace(source,'%out: memref<4x64xf32, #ub>) attributes',
                         '%out: memref<4x64xf32, #ub> {llvm.noalias}) attributes'),
}
unknown_scope = replace(scoped_alpha(source), 'func.func @case(%first: i1, %last: i1)',
                        'func.func @case(%first: i1, %last: i1, %unknown_alpha: memref<4xf32, #ub>)')
unknown_scope = replace(unknown_scope, 'scope.return %alpha_bank :', 'scope.return %unknown_alpha :')
wrong_init = replace(initialized, '%flat_zero = memref.collapse_shape %zero',
                     '%flat_zero = memref.collapse_shape %foreign')
wrong_init = replace(wrong_init, '    %zero = memref.alloc()',
                     '    %foreign = memref.alloc() : memref<4x64xf32, #ub>\n    %zero = memref.alloc()')
inside_init = replace(source, copy, '        hivm.hir.copy ins(%zero : memref<4x64xf32, #ub>) outs(%bank : memref<4x64xf32, #ub>)\n' + copy)
negatives.update({'unknown-scoped-alpha-root': unknown_scope,
                  'wrong-initialization-source': wrong_init, 'initialization-inside-stream': inside_init})
for name,text in negatives.items():
    result=run(text)
    assert result.returncode==0,(name,result.stderr)
    assert 'hivm.cvp_query.output_copies_forwarded = 0 : i64' in result.stdout,name
    assert 'hivm.hir.copy' in result.stdout.split('func.func @case(',1)[1],name
    required=run(text,True)
    assert required.returncode!=0,(name,'required refusal missing')
    assert 'requires a proved row-local full O update and closed final read' in required.stderr,(name,required.stderr)
print('PASS: five positives; '+str(len(negatives))+' valid negative modules and explicit refusals')

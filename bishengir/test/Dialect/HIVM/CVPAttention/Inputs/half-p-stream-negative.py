"""Require semantic/use-closure rejection and unchanged failed-pass IR."""
from pathlib import Path
import re
import subprocess
import sys

tool,filename=sys.argv[1:]
source=Path(filename).read_text()
def once(text,before,after):
    assert text.count(before)==1,(before,text.count(before))
    return text.replace(before,after)
def producer(text,transform):
    begin=text.index('  func.func @_attn_fwd_mix_aiv_outlined_vf_3(')
    end=text.index('\n  }',begin)+len('\n  }')
    return text[:begin]+transform(text[begin:end])+text[end:]
def caller(text,transform):
    begin=text.index('  func.func @_attn_fwd_mix_aiv(')
    end=text.index('\n  }',begin)+len('\n  }')
    return text[:begin]+transform(text[begin:end])+text[end:]
def first_of_two(text,before,after):
    assert text.count(before)==2,(before,text.count(before))
    return text.replace(before,after,1)
cases={
 'changed-maximum-scale':producer(source,lambda t:once(t,'dense<5.000000e-01> : vector<64xf32>',
                                                       'dense<2.500000e-01> : vector<64xf32>')),
 'changed-exp':producer(source,lambda t:once(t,'%16 = math.exp %15','%16 = math.tanh %15')),
 'changed-SSA-inputs':producer(source,lambda t:once(t,'%15 = arith.subf %14, %5','%15 = arith.subf %5, %14')),
 'changed-rounding-contract':producer(source,lambda t:once(t,
   '%18 = arith.truncf %16 {round_mode = #hfusion.round_mode<rint>}', '%18 = arith.truncf %16')),
 'non-complete-transfer-contract':producer(source,lambda t:first_of_two(t,
   '%13 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [true, true]}',
   '%13 = vector.transfer_read %subview[%c0, %c0], %cst_3 {in_bounds = [false, true]}')),
 'changed-key-offset':once(source,'#map3 = affine_map<()[s0] -> (s0 + 64)>',
                                '#map3 = affine_map<()[s0] -> (s0 + 63)>'),
 'unsupported-P-padding':source.replace('8x129x16x1','8x130x16x1').replace('2064','2080').replace('8256','8320'),
 'partial-key-tile':source.replace('arith.constant 1024 : i32','arith.constant 1025 : i32'),
 'unsupported-preload':source.replace('hivm.preload_num = 2 : i32','hivm.preload_num = 3 : i32'),
 'aliased-new-L-M':caller(source,lambda t:once(t,
   '(%alloc_11, %subview_13, %arg14, %arg15, %alloc_14, %alloc_15, %alloc_16)',
   '(%alloc_11, %subview_13, %arg14, %arg15, %alloc_14, %alloc_15, %alloc_14)')),
 'old-L-as-new-L':caller(source,lambda t:once(t,
   '(%alloc_11, %subview_13, %arg14, %arg15, %alloc_14, %alloc_15, %alloc_16)',
   '(%alloc_11, %subview_13, %arg14, %arg15, %arg14, %alloc_15, %alloc_16)')),
}
anchor='            hivm.hir.anchor {id = 37 : i64}\n'
cases['extra-P-reader']=caller(source,lambda t:once(t,anchor,anchor+
    '            memref.copy %subview_13, %subview_13 : memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>> to memref<8x128x16xf16, strided<[2064, 16, 1]>, #hivm.address_space<ub>>\n'))
cases['nested-sync']=caller(source,lambda t:once(t,anchor,anchor+
    '            hivm.hir.sync_block_wait[<VECTOR>, <PIPE_MTE1>, <PIPE_MTE3>] flag = 0\n'))
cases['partial-state-extra-user']=caller(source,lambda t:once(t,'            hivm.hir.anchor {id = 38 : i64}\n',
    '            memref.copy %alloc_14, %alloc_14 : memref<64xf32, #hivm.address_space<ub>> to memref<64xf32, #hivm.address_space<ub>>\n'
    '            hivm.hir.anchor {id = 38 : i64}\n'))
flag='--hivm-cvp-attention-p-stream=allow-sum-reassociation=true require-match=true'
for name,text in cases.items():
    assert text!=source,name
    normalized=subprocess.run([tool,'-','--mlir-disable-threading'],input=text,text=True,capture_output=True)
    assert normalized.returncode==0,(name,'negative input is invalid',normalized.stderr)
    refused=subprocess.run([tool,'-',flag,'--mlir-disable-threading','--mlir-print-ir-after-failure'],
                           input=text,text=True,capture_output=True)
    assert refused.returncode==1,(name,refused.returncode,refused.stderr)
    match=re.search(r'^// -----// IR Dump After .*CVPAttentionPStream.*Failed[^\n]*\n',refused.stderr,re.M)
    assert match,(name,'not rejected by the intended pass',refused.stderr)
    lines=[]
    for line in refused.stderr[match.end():].splitlines():
        lines.append(line)
        if line=='}':break
    assert '\n'.join(lines).strip()==normalized.stdout.strip(),(name,'rejection changed the module')
print('PASS: %d valid hostile inputs rejected transactionally'%len(cases))

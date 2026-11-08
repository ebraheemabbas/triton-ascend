"""Component-only mutations of actual pre-memory attention IR."""
from pathlib import Path
import re,sys

mode=sys.argv[1];text=Path(sys.argv[2]).read_text()
if mode=='renamed':
    text=re.sub(r'([%@])([A-Za-z_][A-Za-z0-9_]*)',lambda m:m[1]+'renamed_'+m[2],text)
elif mode.startswith('keys-'):
    count=int(mode.split('-')[1]);assert count in [1,2,3,4,5,6,8]
    pieces=re.split(r'(?m)(?=^  func.func )',text)
    changed=0
    for i,piece in enumerate(pieces):
        old='to %c4096_i32 step %c128_i32'
        if old not in piece:continue
        changed+=piece.count(old)
        piece=piece.replace(old,'to %boundary_key_count step %c128_i32')
        first=piece.index('\n')
        piece=piece[:first+1]+f'    %boundary_key_count = arith.constant {count*128} : i32\n'+piece[first+1:]
        pieces[i]=piece
    assert changed==3
    text=''.join(pieces).replace('key_count = 4096 : i64',f'key_count = {count*128} : i64')
elif mode=='release-inside-output':
    needle='            func.call @_attn_fwd_mix_aiv_outlined_vf_3('
    assert text.count(needle)==1
    text=text.replace(needle,'            hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>] flag = 99\n'+needle)
elif mode=='state-initial-two':
    needle='dense<1.000000e+00> : vector<64xf32>'
    assert text.count(needle)==1
    text=text.replace(needle,'dense<2.000000e+00> : vector<64xf32>')
elif mode in ['initial-output-escape','final-output-extra-reader']:
    ty='memref<64x128xf32, #hivm.address_space<ub>>'
    header=re.search(r'(?m)^module[^\n]*\n',text)
    assert header is not None
    start=header.end()
    text=text[:start]+f'  func.func private @observe_output({ty})\n'+text[start:]
    if mode=='initial-output-escape':
        needle='        func.call @_attn_fwd_mix_aiv_outlined_vf_5(%alloc_1)'
        index=text.index('\n',text.index(needle))
        text=text[:index+1]+f'        func.call @observe_output(%alloc_1) : ({ty}) -> ()\n'+text[index+1:]
    else:
        needle='        func.call @_attn_fwd_mix_aiv_fused_5_outlined_merged_vf_0('
        index=text.index('\n',text.index(needle))
        text=text[:index+1]+f'        func.call @observe_output(%23#0) : ({ty}) -> ()\n'+text[index+1:]
elif mode=='output-index-metadata':
    needle='  func.func @_attn_fwd_mix_aiv_outlined_vf_3('
    start=text.index(needle);end=text.index('\n',start)
    line=text[start:end];assert 'attributes {' in line
    line=line.replace('attributes {','attributes {test.argument_index = 2 : i64, ',1)
    text=text[:start]+line+text[end:]
else:
    raise ValueError(mode)
sys.stdout.write(text)

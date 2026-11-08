"""Check the complete QK/P/PV map on real scheduled attention fixtures."""

import argparse,itertools,json,re,subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--opt',required=True)
parser.add_argument('--fixtures',type=Path,required=True)
parser.add_argument('--artifacts',type=Path)
args=parser.parse_args()
if args.artifacts is not None:args.artifacts.mkdir(parents=True,exist_ok=False)
invocation=0
verify='--hivm-verify-cvp-attention=retain-provenance=true drop-annotations=false'
layout='--hivm-cvp-attention-layout=allow-fp32-sum-reassociation=true require-match=true'

def run(source,options,expected=None):
    global invocation
    result=subprocess.run([args.opt,*options,'-verify-each'],input=source,text=True,capture_output=True)
    if args.artifacts is not None:
        stem=f'{invocation:03d}'
        (args.artifacts/(stem+'.input.mlir')).write_text(source)
        (args.artifacts/(stem+'.output.mlir')).write_text(result.stdout)
        (args.artifacts/(stem+'.stderr.txt')).write_text(result.stderr)
        (args.artifacts/(stem+'.json')).write_text(json.dumps({
            'argv':[args.opt,*options,'-verify-each'],'exit':result.returncode,'expected_error':expected},indent=2)+'\n')
    invocation+=1
    if expected is None:
        assert result.returncode==0,result.stderr
        return result.stdout
    assert result.returncode!=0,(expected,result.stdout)
    assert expected in result.stderr,(expected,result.stderr)
    return result.stderr

def matrices(text):
    pattern=r'(%[\w]+) = hivm.hir.mmadL1 \{([^}]*)\} ins\(([^:]+) : ([^\n]+)'
    result=[]
    for match in re.finditer(pattern,text):
        operands=[value.strip() for value in match[3].split(',')]
        result.append({'attrs':match[2],'operands':operands,'line':match[0]})
    assert len(result)==2,result
    return result

def constants(text):
    return {m[1]:int(m[2]) for m in re.finditer(r'(%[\w]+) = arith.constant (\d+) : index',text)}

def scopes(text):
    result=list(re.finditer(
        r'^[ \t]*%\w+(?::\d+)? = scope.scope [^\n]*\{\n(.*?)'
        r'^[ \t]*\} \{hivm.loop_core_type = ([^\n]+)',text,re.M|re.S))
    assert len(result)==4,len(result)
    return result

def inputs(text,key_major):
    """Follow the actual views in their lexical scopes, not cross-print SSA IDs.

    This also distinguishes Q from K in square tiles with identical types.
    Q is loaded before the scheduled scopes; K and V are loaded in their
    respective Cube scopes. P is the view published between the vector and PV.
    """
    blocks=scopes(text)
    view=r'(%\w+) = bufferization.to_tensor (%\w+) restrict writable : ([^\n]+)'
    query=re.findall(view,text[:blocks[0].start()])[-1][0]
    key_views=re.findall(view,blocks[0][1])
    value_views=re.findall(view,blocks[2][1])
    assert len(key_views)==len(value_views)==1
    probability=re.findall(view,text[blocks[1].end():blocks[2].start()])
    assert len(probability)==1
    qk,pv=matrices(text)
    assert qk['operands'][:2]==([key_views[0][0],query] if key_major
                              else [query,key_views[0][0]])
    assert pv['operands'][:2]==[probability[0][0],value_views[0][0]]

def transpose_flat(values,shape,permutation):
    """Interpret the tensor transpose, independently of the Cube NZ decoder."""
    output=[]
    for target in itertools.product(*(range(shape[axis]) for axis in permutation)):
        source=[0]*len(shape)
        for new_axis,old_axis in enumerate(permutation):
            source[old_axis]=target[new_axis]
        offset=0
        for coordinate,size in zip(source,shape):offset=offset*size+coordinate
        output.append(values[offset])
    return output

def check_probability_coordinates(producer,bn):
    # Expand-shape preserves element order. Interpret the actual transpose
    # between the two expansions; do not fill and read with the same NZ formula.
    permutation=tuple(map(int,re.search(r'permutation = \[([0-9, ]+)\]',producer)[1].split(',')))
    logical=[(query,key) for key in range(bn) for query in range(128)]
    packed=transpose_flat(logical,(bn,8,16),permutation)
    seen=[set(),set()]
    for offset,label in enumerate(packed):
        # Decode the Cube's normalized [N1,M1,M0,N0] representation.
        remaining,n0=divmod(offset,16)
        remaining,m0=divmod(remaining,16)
        n1,m1=divmod(remaining,bn//16)
        row,column=16*m1+m0,16*n1+n0
        # PV has transpose-A, so physical P[row,column] is logical P[q,k].
        query,key=column,row
        assert label==(query,key),(offset,label,(query,key))
        assert (query,key) not in seen[query//64]
        seen[query//64].add((query,key))
    for shard in range(2):
        assert seen[shard]=={(q,k) for q in range(shard*64,(shard+1)*64) for k in range(bn)}
    # The independent interpreter must expose a missing transpose, including
    # the square BM=BN=128 case where types alone cannot expose the error.
    assert transpose_flat(logical,(bn,8,16),(0,1,2))!=packed

def check_mapping(before,after,bn,hd):
    inputs(before,False)
    inputs(after,True)
    new=matrices(after)
    values=constants(after)
    assert [values[name] for name in new[0]['operands'][3:6]]==[bn,hd,128]
    assert 'b_transpose' in new[0]['attrs'] and 'a_transpose' not in new[0]['attrs']
    assert [values[name] for name in new[1]['operands'][3:6]]==[128,bn,hd]
    assert 'a_transpose' in new[1]['attrs'] and 'b_transpose' not in new[1]['attrs']
    assert 'physical_score_shape = array<i64: '+str(bn)+', 128>' in after
    assert 'score_layout = "key-major"' in after
    assert 'numerical_policy = "reassociate-fp32-sum"' in after
    assert 'physical_lifetimes_verified = false' in after
    assert len(re.findall(r'reduce_dims = \[0\]',after))==2
    assert 'reduce_dims = [1]' not in after
    assert 'broadcast = [0]' in after
    assert 'tensor<8x%dx16x16xf32>'%(bn//16) in new[0]['line']
    assert 'memref<8x%dx16x16xf16, #hivm.address_space<cbuf>>'%(bn//16) in after
    assert 'memref<%dx128xf32, #hivm.address_space<ub>>'%bn in after
    assert [s[2] for s in scopes(before)]==[s[2] for s in scopes(after)],'Scope schedule changed'
    producer=scopes(after)[1][1]
    assert 'output_shape [%d, 8, 16]'%bn in producer
    assert 'permutation = [1, 0, 2]' in producer
    assert 'output_shape [8, %d, 16, 16]'%(bn//16) in producer
    assert producer.count('reduce_dims = [0]')==2
    assert 'tensor<1x128xf32>' in producer
    check_probability_coordinates(producer,bn)

def insert_after_line(text,match,operation):
    end=text.index('\n',match.end())
    return text[:end]+'\n'+operation+text[end:]

# No proof means no rewrite; an explicit require-match request must diagnose it.
empty=run('module {}',[])
assert run(empty,['--hivm-cvp-attention-layout'])==empty
run(empty,[layout],'requires a verified mixed Skew recurrence')

for item in json.loads((args.fixtures/'MANIFEST.json').read_text()):
    original=(args.fixtures/item['file']).read_text()
    # Function-pass dumps omit the surrounding module's target. Restore the
    # original Ascend950 target before running the ordinary IR verifier.
    original='module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {\n'+original+'\n}\n'
    prepared=run(original,[verify])
    changed=run(prepared,[layout])
    check_mapping(prepared,changed,item['BN'],item['shape'][-1])
    repeated=run(changed,[layout])
    assert repeated==changed,'Layout pass is not idempotent'
    renamed=original.replace('@_attn_fwd','@different_kernel')
    names=set(re.findall(r'%[A-Za-z_][\w]*|%\d+',renamed))
    mapping={name:'%renamed_'+name[1:] for name in names}
    renamed=re.sub(r'%[A-Za-z_][\w]*|%\d+',lambda m:mapping[m[0]],renamed)
    renamed_before=run(renamed,[verify])
    renamed_after=run(renamed_before,[layout])
    check_mapping(renamed_before,renamed_after,item['BN'],item['shape'][-1])
    run(prepared,['--hivm-cvp-attention-layout=require-match=true'],
        'explicit FP32 sum reassociation permission')
    corrupted=prepared.replace('arith.constant 5.000000e-01 : f32','arith.constant 7.500000e-01 : f32',1)
    assert corrupted!=prepared
    run(corrupted,[layout],'CVP attention layout rejected')
    stale=prepared.replace('block_m = 128 : i64','block_m = 64 : i64',1)
    assert stale!=prepared
    run(stale,[layout],'typed recurrence differs from provenance')
    bound=prepared.replace('    %cst =', '    %already_bound = hivm.hir.get_sub_block_idx -> i64\n    %cst =',1)
    if bound==prepared:
        marker=re.search(r'^\s+%\w+ = arith.constant',prepared,re.M)
        assert marker
        bound=prepared[:marker.start()]+'\n  %already_bound = hivm.hir.get_sub_block_idx -> i64\n'+prepared[marker.start():]
    run(bound,[layout],'before subblock binding')
    # A read-only extra user is still outside the proved score consumer set.
    source=re.search(r'(%\w+) = bufferization.to_tensor (%\w+) restrict writable : memref<128x'+str(item['BN'])+r'xf32, #hivm.address_space<ub>>',prepared)
    assert source
    extra='        %extra_zero = arith.constant 0 : index\n        %extra_read = tensor.extract '+source[1]+'[%extra_zero, %extra_zero] : tensor<128x'+str(item['BN'])+'xf32>'
    run(insert_after_line(prepared,source,extra),[layout],'score/P data escapes')
    shift=re.search(r'(%\w+) = hivm.hir.vsub [^\n]+ -> tensor<128x'+str(item['BN'])+'xf32>',prepared)
    assert shift
    extra='        %extra_zero = arith.constant 0 : index\n        %shift_read = tensor.extract '+shift[1]+'[%extra_zero, %extra_zero] : tensor<128x'+str(item['BN'])+'xf32>'
    run(insert_after_line(prepared,shift,extra),[layout],'score/P data escapes')
    # Even an otherwise unused alias invalidates the private-buffer proof.
    score_type='memref<128x%dxf32, #hivm.address_space<ub>>'%item['BN']
    alias='        %score_alias = memref.cast '+source[2]+' : '+score_type+' to '+score_type
    run(insert_after_line(prepared,source,alias),[layout],'private score/P buffer has another use or alias')
    p_type='memref<%dx8x16x16xf16, #hivm.address_space<cbuf>>'%(item['BN']//16)
    p_view=re.search(r'(%\w+) = bufferization.to_tensor (%\w+) restrict writable : '+re.escape(p_type),prepared)
    assert p_view
    alias='        %p_alias = memref.cast '+p_view[2]+' : '+p_type+' to '+p_type
    run(insert_after_line(prepared,p_view,alias),[layout],'private score/P buffer has another use or alias')
    p_tensor='tensor<%dx8x16x16xf16>'%(item['BN']//16)
    extra='        %extra_zero = arith.constant 0 : index\n        %p_read = tensor.extract '+p_view[1]+'[%extra_zero, %extra_zero, %extra_zero, %extra_zero] : '+p_tensor
    run(insert_after_line(prepared,p_view,extra),[layout],'score/P data escapes')
print('QK/P/PV maps, both query shards, square tiles, renaming, idempotence, permission, stale-proof and escape controls passed')

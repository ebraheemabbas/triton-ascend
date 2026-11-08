"""Exercise probability fusion on actual generated tensor VF modules."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--opt',required=True)
parser.add_argument('--fixtures',type=Path,required=True)
parser.add_argument('--artifacts',type=Path)
args=parser.parse_args()
if args.artifacts:args.artifacts.mkdir(parents=True,exist_ok=False)
counter=0
enabled='--hivm-cvp-attention-query-vector=fuse-probability=true require-match=true'
unsupported='requires a closed key-major tensor VF pair'

def run(source,options,expected=None):
    global counter
    command=[args.opt,*options,'-verify-each']
    result=subprocess.run(command,input=source,text=True,capture_output=True)
    if args.artifacts:
        stem=f'{counter:03d}'
        for suffix,value in [('input.mlir',source),('output.mlir',result.stdout),('stderr.txt',result.stderr)]:
            (args.artifacts/(stem+'.'+suffix)).write_text(value)
        (args.artifacts/(stem+'.json')).write_text(json.dumps({
            'argv':command,'exit':result.returncode,'expected_error':expected},indent=2)+'\n')
    counter+=1
    if expected is None:
        assert result.returncode==0,result.stderr
        return result.stdout
    assert result.returncode!=0 and expected in result.stderr,result.stderr

def functions(text):
    matches=list(re.finditer(r'^  func.func [^\n]+',text,re.M))
    return {re.search(r'@([^ (]+)',m[0])[1]:text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]
            for i,m in enumerate(matches)}

def header(body):
    return body.splitlines()[0]

def scope(body):
    found=[]
    for match in re.finditer(r'^[ \t]*(?:%\w+(?::\d+)? = )?scope.scope [^\n]*\{\n',body,re.M):
        end=match.end();depth=1
        while depth:
            depth+=(body[end]=='{')-(body[end]=='}');end+=1
        suffix=body[end:body.find('\n',end)]
        if '#hivm.tcore_type<VECTOR>' in suffix and 'hivm.preload_num = 2 : i32' in suffix:
            found.append(body[match.end():end-1])
    assert len(found)==1
    return found[0]

def chain(text,bn):
    fs=functions(text)
    callers=[name for name,body in fs.items() if 'hacc.entry' in header(body)
             and '#hivm.func_core_type<AIV>' in header(body) and 'hivm.backup_function' not in header(body)]
    assert len(callers)==1
    caller=callers[0];region=scope(fs[caller])
    calls=re.findall(r'func.call @([^ (]+)',region)
    producers=[name for name in calls if 'math.exp' in fs[name]
               and '-> tensor<'+str(bn)+'x64xf32> attributes' in header(fs[name])]
    assert len(producers)==1
    producer=producers[0]
    line=next(line for line in region.splitlines() if 'func.call @'+producer+'(' in line)
    value=re.search(r'(%\w+) = func.call',line)[1]
    consumers=[name for name in calls if 'func.call @'+name+'('+value+',' in region]
    assert len(consumers)==1
    return fs,caller,producer,consumers[0],line,value

def protected_lines(text):
    return [line.strip() for line in text.splitlines()
            if 'hivm.hir.sync_block' in line or 'hivm.hir.anchor' in line]

def validate(before,after,bn):
    fs,caller,producer,consumer,_line,_value=chain(before,bn)
    changed=functions(after)
    assert producer not in changed
    assert 'hivm.cvp_attention.p32_materializations_removed = 1 : i64' in after
    old_scope,new_scope=scope(fs[caller]),scope(changed[caller])
    assert old_scope.count('func.call ')==new_scope.count('func.call ')+1
    assert protected_lines(before)==protected_lines(after)
    body=changed[consumer]
    assert body.count('math.exp ')==1
    assert body.count('arith.addf ')==1 and body.count('arith.truncf ')==1
    assert '#hfusion.round_mode<rint>' in body
    exp=re.search(r'(%\w+) = math.exp ',body)[1]
    assert re.search(r'vector.shape_cast '+re.escape(exp)+r' : vector<1x64xf32> to vector<64xf32>',body)
    assert re.search(r'arith.truncf '+re.escape(exp)+r' ',body)
    # The conversion and FP32 sum both use exp's FP32 result.
    assert body.count('scf.for ')==fs[consumer].count('scf.for ')==1
    old_copies=[line.split(' outs(',1)[1] for line in old_scope.splitlines() if 'hivm.hir.copy' in line]
    new_copies=[line.split(' outs(',1)[1] for line in new_scope.splitlines() if 'hivm.hir.copy' in line]
    assert old_copies==new_copies,'P publication destination/type/attributes changed'
    for name,original in fs.items():
        if name not in (caller,producer,consumer):
            assert changed[name]==original,'Unrelated or backup function changed: '+name
    return caller,producer,consumer

manifest=json.loads((args.fixtures/'MANIFEST.json').read_text())
for item in manifest:
    path=args.fixtures/item['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    before=run(path.read_text(),[])
    assert run(before,['--hivm-cvp-attention-query-vector'])==before
    if item['arm']=='baseline':
        assert run(before,['--hivm-cvp-attention-query-vector=fuse-probability=true'])==before
        run(before,[enabled],unsupported)
        continue
    after=run(before,[enabled])
    validate(before,after,item['BN'])
    assert run(after,['--hivm-cvp-attention-query-vector=fuse-probability=true'])==after
    fs,caller,producer,consumer,line,value=chain(before,item['BN'])
    # An additional P32 reader prevents eliminating the producer result.
    read=(line+'\n              %test_zero = arith.constant 0 : index\n'
          '              %test_read = tensor.extract '+value+'[%test_zero, %test_zero] : tensor<'+str(item['BN'])+'x64xf32>')
    run(before.replace(line,read,1),[enabled],unsupported)
    # An otherwise unused second call still makes the callee shared.
    extra=re.sub(r'%\w+ = func.call','%test_shared = func.call',line,count=1)
    run(before.replace(line,line+'\n'+extra,1),[enabled],unsupported)
    # A handoff inside the scheduled P scope must never be crossed.
    sync=next(l.strip() for l in before.splitlines()
              if 'hivm.hir.sync_block_set[<VECTOR>, <PIPE_V>, <PIPE_FIX>]' in l)
    run(before.replace(line,line+'\n              '+sync,1),[enabled],unsupported)
    # Both loops must cover every key with unit stride.
    for function in (producer,consumer):
        old=fs[function]
        changed=old.replace('arith.constant 1 : index','arith.constant 2 : index',1)
        assert changed!=old
        run(before.replace(old,changed,1),[enabled],unsupported)
    for initializer in ('1.000000e+00','-0.000000e+00'):
        old=fs[consumer]
        changed=old.replace('dense<0.000000e+00> : vector<64xf32>',
                            'dense<'+initializer+'> : vector<64xf32>',1)
        assert changed!=old
        run(before.replace(old,changed,1),[enabled],unsupported)
    old=fs[producer]
    read_line=next(l for l in old.splitlines() if 'vector.transfer_read' in l and ': tensor<1x64xf32>' in l)
    mask_line=read_line.replace(' {in_bounds',', %test_mask {in_bounds',1)
    assert mask_line!=read_line
    masked=old.replace(read_line,'      %test_mask = arith.constant dense<true> : vector<1x64xi1>\n'+mask_line,1)
    run(before.replace(old,masked,1),[enabled],unsupported)
    source_slice=re.search(r'vector.transfer_read (%\w+)\[',read_line)[1]
    slice_line=next(l for l in old.splitlines() if source_slice+' = tensor.extract_slice' in l)
    strided=old.replace(slice_line,slice_line.replace('[1, 1]','[2, 1]',1),1)
    assert strided!=old
    run(before.replace(old,strided,1),[enabled],unsupported)
    argument_type='tensor<'+str(item['BN'])+'x64xf32>'
    attributed=old.replace(argument_type+',',argument_type+' {test.arg_attr = 1 : i64},',1)
    assert attributed!=old
    run(before.replace(old,attributed,1),[enabled],unsupported)
    # Updating a call's operands must not leave unknown operand-index metadata.
    for callee in (producer,consumer):
        call_line=next(l for l in scope(fs[caller]).splitlines() if 'func.call @'+callee+'(' in l)
        annotated=call_line.replace('{hivm.vector_function,','{test.operand_map = [0 : i64], hivm.vector_function,',1)
        assert annotated!=call_line
        run(before.replace(call_line,annotated,1),[enabled],unsupported)
    caller_header=header(fs[caller])
    wrong_proof=caller_header.replace('score_layout = "key-major"','score_layout = "query-major"')
    assert wrong_proof!=caller_header
    run(before.replace(caller_header,wrong_proof,1),[enabled],unsupported)
    # Keep the original arithmetic attributes when the expression is cloned.
    old=fs[producer]
    tagged=re.sub(r'(math.exp %\w+)( : vector)',r'\1 {test.fusion_attr = "kept"}\2',old,count=1)
    assert tagged!=old
    tagged_result=run(before.replace(old,tagged,1),[enabled])
    assert 'test.fusion_attr = "kept"' in functions(tagged_result)[consumer]
    # Recognition must not depend on generated function or SSA names.
    renamed=re.sub(r'@([A-Za-z_][\w.$]*)',r'@renamed_\1',before)
    renamed=re.sub(r'%([A-Za-z_]\w*|\d+)',r'%renamed_\1',renamed)
    renamed=run(renamed,[])
    validate(renamed,run(renamed,[enabled]),item['BN'])
print('Probability fusion: real fixtures, default-off, idempotence, closed uses, shared callee, handoff, key coverage, attributes and renaming passed')

"""Mutate only the generated Cube guard; preserve the rest of the module."""
from pathlib import Path
import re,sys
source,mode,destination=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
assert mode in ['commuted','false','active-wait','missing-wait']
text=source.read_text()
start=text.index('func.func @_attn_fwd_mix_aic(')
following=re.search(r'\n\s*func\.func ',text[start+1:])
end=start+1+following.start() if following else len(text)
cube=text[start:end]
pattern=r'(?m)^(?P<indent>\s*)scf\.if (?P<condition>%[\w.$]+) \{\n\s*hivm\.hir\.(?P<kind>set|wait)_flag\[<PIPE_FIX>, <PIPE_M>, <EVENT_ID0>\]\s*\n(?P=indent)\}'
guards=list(re.finditer(pattern,cube))
assert len(guards)==2 and {g['kind'] for g in guards}=={'set','wait'}
condition=guards[0]['condition']
assert all(g['condition']==condition for g in guards)
expression=re.search(r'(?m)^\s*'+re.escape(condition)+r' = arith\.xori (?P<a>%[\w.$]+), (?P<b>%[\w.$]+) : i1\s*$',cube)
assert expression
constants=[]
for name in ['a','b']:
    matches=list(re.finditer(r'(?m)^\s*'+re.escape(expression[name])+r' = arith\.constant (?P<bit>true)\s*$',cube))
    if matches:
        assert len(matches)==1
        constants.append((name,matches[0]))
assert len(constants)==1
name,constant=constants[0]
active=expression['b' if name=='a' else 'a']
if mode=='commuted':
    original=expression.group()
    swapped=original.replace(expression['a']+', '+expression['b'],expression['b']+', '+expression['a'])
    assert swapped!=original
    cube=cube[:expression.start()]+swapped+cube[expression.end():]
elif mode=='false':
    cube=cube[:constant.start('bit')]+'false'+cube[constant.end('bit'):]
else:
    wait=next(g for g in guards if g['kind']=='wait')
    if mode=='active-wait':
        cube=cube[:wait.start('condition')]+active+cube[wait.end('condition'):]
    else:
        cube=cube[:wait.start()]+cube[wait.end():]
assert cube!=text[start:end]
assert not destination.exists()
destination.write_text(text[:start]+cube+text[end:])

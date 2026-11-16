from core112 import *
assert jread(H112/'ENGINEERING.json')['n']==12 and len(ARMS112)==89 and len(NEW112)==14
old=jread(H112/'experiment/ENGINEERING_CODE_LOCK.json')
for p,h in old['files'].items():assert sha(R112/p)==h
fs={p:sha(R112/p) for p in old['files']};fs.update({str(p.relative_to(R112)):sha(p) for p in H112.glob('*.py')});o=H112/'experiment/CODE_LOCK.json';assert not o.exists();jdump(o,{'files':fs,'upstream_dependency_sha256':old['upstream_dependency_sha256'],'design_sha256':sha(H112/'experiment/DESIGN_LOCK.json'),'engineering_sha256':sha(H112/'ENGINEERING.json'),'primary':PRIMARY112,'arms':ARMS112});print('FORMAL_LOCK112',len(fs),len(ARMS112))

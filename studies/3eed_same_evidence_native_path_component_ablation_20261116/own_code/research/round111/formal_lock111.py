from core111 import *
assert len(ARMS111)==75 and len(SWAPS111)==5 and jread(H111/'ENGINEERING.json')['n']==12
old=jread(H111/'experiment/ENGINEERING_CODE_LOCK.json')
for p,h in old['files'].items():assert sha(R111/p)==h
fs={p:sha(R111/p) for p in old['files']};fs.update({str(p.relative_to(R111)):sha(p) for p in H111.glob('*.py')});o=H111/'experiment/CODE_LOCK.json';assert not o.exists();jdump(o,{'files':fs,'upstream_dependency_sha256':old['upstream_dependency_sha256'],'design_sha256':sha(H111/'experiment/DESIGN_LOCK.json'),'engineering_sha256':sha(H111/'ENGINEERING.json'),'arms':ARMS111,'primary':PRIMARY111});print('FORMAL_LOCK111',len(fs),len(ARMS111))

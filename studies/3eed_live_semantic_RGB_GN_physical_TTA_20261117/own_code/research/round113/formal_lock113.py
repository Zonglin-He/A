from core113 import *
assert jread(H113/'ENGINEERING.json')['n']==12
old=jread(H113/'experiment/ENGINEERING_CODE_LOCK.json')
for p,h in old['files'].items():assert sha(R113/p)==h,p
fs={p:sha(R113/p) for p in old['files']};fs.update({str(p.relative_to(R113)):sha(p) for p in H113.glob('*.py')});o=H113/'experiment/CODE_LOCK.json';assert not o.exists();jdump(o,{'files':fs,'upstream_dependency_sha256':old['upstream_dependency_sha256'],'design_sha256':sha(H113/'experiment/DESIGN_LOCK.json'),'engineering_sha256':sha(H113/'ENGINEERING.json'),'primary':PRIMARY113,'arms':ARMS113});print('FORMAL_LOCK113',len(fs),len(ARMS113))

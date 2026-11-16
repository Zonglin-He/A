"""Pin user-confirmed full R95 with no target metric selection."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round98';rd=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert rd(H/'LEGACY_R95_PARITY_AUDIT.json')['state'].startswith('PASS_')
for src,tgt in [('waymo','drone'),('drone','quad'),('quad','waymo')]:
 p=R/f'crossplatform/round98/engineering_{src}_to_{tgt}/SEALED.json';assert rd(p)['n']==2
files=set(H.glob('*.py'));files.add(R/'tools/manual_r98_guard.py')
files.update(R/f for f in rd(R/'research/round97/experiment/TARGET_CODE_LOCK.json')['code_sha256'])
files.update(R/f for f in ['research/round92/common92.py','research/round93/core93.py','research/round94/core94.py','research/round94/solver94.py','research/round95/core95.py'])
p=H/'experiment/CODE_LOCK.json';assert not p.exists();p.write_text(json.dumps({'method':'R95_complete_lm_FC','code_sha256':{str(f.relative_to(R)):sha(f) for f in sorted(files)},'inputs_sha256':{str(p.relative_to(R)):sha(p) for p in [H/'experiment/DESIGN_LOCK.json',R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json',R/'research/round96/SOURCE_MODEL_COLLECTION_PRIVATE.json']},'no_source_training':True,'FINAL_access':False},indent=2)+'\n');print({'pinned_programs':len(files)})

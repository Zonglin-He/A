"""Administrative addition of postseal descriptive summaries, no prediction changes."""
from pathlib import Path
import json,hashlib,argparse
p=argparse.ArgumentParser();p.add_argument('--attempt',type=int,default=1);a=p.parse_args();R=Path(__file__).resolve().parents[2];H=R/'research/round109';O=H/('publication/public_export_attempt'+str(a.attempt).zfill(3));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name in ['RESIDUAL_PAIR_DIAGNOSTIC.json','ATLAS_REVIEW_SUMMARY.json','ADMIN_FAILURES.json']:(O/name).write_bytes((H/name).read_bytes())
m=json.loads((O/'MANIFEST.json').read_text());m['files']=[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file() and f.name!='MANIFEST.json'];(O/'MANIFEST.json').write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n');print('SUPPLEMENT_EXPORT109',len(m['files'])+1)

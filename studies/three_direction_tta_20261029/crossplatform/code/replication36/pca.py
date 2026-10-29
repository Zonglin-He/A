from pathlib import Path
import json,hashlib,time,numpy as np
from common import refine,reconstruct_points
R=Path(__file__).resolve().parent;plan=json.loads((R/'PLAN_LOCK.json').read_text())
for name,h in plan['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
source=R/'source';seal=json.loads((source/'sealed.json').read_text());manifest=json.loads((R/'manifest.json').read_text());out=R/'pca';out.mkdir(exist_ok=False);records=[]
for i,s in enumerate(manifest):
 name=s['id']+'.npz';assert hashlib.sha256((source/name).read_bytes()).hexdigest()==seal['files'][name];start=time.perf_counter();src=np.load(source/name)['source7d'];points=reconstruct_points(s,20261029+i);b,p,diag=refine(points,src);assert np.array_equal(b,src);np.savez_compressed(out/name,source7d=b,refined7d=p);rec={'id':s['id'],'platform':s['platform'],'scene':s['scene'],**diag,'seconds':time.perf_counter()-start};records.append(rec);print(json.dumps(rec),flush=True)
(out/'trace.json').write_text(json.dumps(records,indent=2));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'n':len(records),'plan_sha256':hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest(),'source_seal_sha256':hashlib.sha256((source/'sealed.json').read_bytes()).hexdigest(),'label_access':False},indent=2));print('PCA SEALED',len(records),flush=True)

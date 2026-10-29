from pathlib import Path
import sys,json,hashlib,time,numpy as np
R=Path(__file__).resolve().parent;P=R.parent;sys.path.insert(0,str(P/'replication36'))
from common import reconstruct_points
from refine import refine
plan=json.loads((R/'PLAN_LOCK.json').read_text());assert hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest()==(R/'PLAN_LOCK.sha256').read_text().strip()
for n,h in plan['source_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
out=R/'predictions';out.mkdir(exist_ok=False);records=[]
for cohort,manifest_path,pred_dir in [('development9',P/'manifest.json',P/'runs/geometry_pca_v1'),('replication36',P/'replication36/manifest.json',P/'replication36/pca')]:
 manifest=json.loads(manifest_path.read_text());seal=json.loads((pred_dir/'sealed.json').read_text())
 for i,s in enumerate(manifest):
  name=s['id']+'.npz';assert hashlib.sha256((pred_dir/name).read_bytes()).hexdigest()==seal['files'][name];d=np.load(pred_dir/name);start=time.perf_counter();points=reconstruct_points(s,20261029+i);selected,diag=refine(points,d['source7d'],d['refined7d']);np.savez_compressed(out/name,source7d=d['source7d'],pca7d=d['refined7d'],surface7d=selected);record={'id':s['id'],'cohort':cohort,'platform':s['platform'],**diag,'seconds':time.perf_counter()-start};records.append(record);print(json.dumps(record),flush=True)
(out/'trace.json').write_text(json.dumps(records,indent=2));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'n':len(records),'plan_sha256':hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest(),'label_access':False},indent=2));assert len(records)==45;print('SEALED45',flush=True)

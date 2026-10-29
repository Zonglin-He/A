from pathlib import Path
import json,hashlib,time,numpy as np,importlib.util
from common import reconstruct_points
R=Path(__file__).resolve().parent;P=R.parent;plan=json.loads((R/'PLAN_LOCK.json').read_text());assert hashlib.sha256((P/'surface_dev45/refine.py').read_bytes()).hexdigest()==plan['surface_refine_sha256'];spec=importlib.util.spec_from_file_location('locked_surface_refine',P/'surface_dev45/refine.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
for n,h in plan['source_sha256'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
pca=R/'pca';seal=json.loads((pca/'sealed.json').read_text());manifest=json.loads((R/'manifest.json').read_text());out=R/'surface';out.mkdir(exist_ok=False);rows=[]
for i,s in enumerate(manifest):
 n=s['id']+'.npz';assert hashlib.sha256((pca/n).read_bytes()).hexdigest()==seal['files'][n];d=np.load(pca/n);start=time.perf_counter();points=reconstruct_points(s,20261029+i);b,diag=module.refine(points,d['source7d'],d['refined7d']);np.savez_compressed(out/n,source7d=d['source7d'],pca7d=d['refined7d'],surface7d=b);row={'id':s['id'],'platform':s['platform'],'scene':s['scene'],**diag,'seconds':time.perf_counter()-start};rows.append(row);print(json.dumps(row),flush=True)
(out/'trace.json').write_text(json.dumps(rows,indent=2));hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.npz')};(out/'sealed.json').write_text(json.dumps({'files':hashes,'n':len(rows),'plan_sha256':hashlib.sha256((R/'PLAN_LOCK.json').read_bytes()).hexdigest(),'PCA_seal_sha256':hashlib.sha256((pca/'sealed.json').read_bytes()).hexdigest(),'label_access':False},indent=2));assert len(rows)==36;print('SURFACE SEALED36',flush=True)

"""Fixed cached-input small EdgeTAM evaluation, no new Source or expert."""
import os,sys,json,time,hashlib,shutil
from pathlib import Path
from functools import lru_cache
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent
sys.path[:0]=[str(R/'studies/review_strengthening_v1'),str(R),str(R/'tools')]
import review_common as parent
sys.path.insert(0,str(S))
read=parent.read;sha=parent.sha;atomic=parent.atomic;identity=parent.identity;digest=parent.digest;fixed=lru_cache(None)(read)
D=R/'outputs/edgetam_comparison_v1';P=R/'results/edgetam_comparison_v1'
ARMS=['SourceMatchedTime','ExpertInterpolation','EdgeTAM','Ours'];PAIRS=[('EdgeTAM','SourceMatchedTime'),('EdgeTAM','ExpertInterpolation'),('Ours','EdgeTAM')]
def rows(ds):return fixed(parent.D/'selection.json')[ds]
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(f.stat().st_size for f in D.rglob('*')if f.is_file())<160*2**30
def validate():
 p=read(D/'pilot_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 parent.validate();return p
def save(path,obj):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def payload(ds,row):
 import numpy as np,torch
 from stvg.decode import decode
 from stvg.teacher_bridge import evidence_key
 cap,origin=parent.selected.suite().load_parent(ds,row);frames,ids=decode(row['metadata'],ds);assert ids==cap['frame_ids'];pos=cap['positions8'];native=np.asarray(cap['geometry']['native_boxes']);runtime=fixed(parent.RUN/'plan.json')['provider_runtime_fingerprint']
 key,ctx=evidence_key(frames[pos],row['metadata']['caption'],positions=pos,frame_ids=ids,fingerprint=runtime,dataset=ds);roots=[parent.PROVIDER/'outputs/boxfixed_full_v1/teacher_cache',R.parent/'stvg-mainline/outputs/tastvg_paper_v1/teacher_cache'];hits=[p for p in roots if(p/(key+'.json')).exists()];assert hits;tp=hits[0]/(key+'.pt');meta=read(tp.with_suffix('.json'));assert meta['key']==key and meta['context']==ctx and sha(tp)==meta['sha256'];teacher=torch.load(tp,map_location='cpu',weights_only=False,mmap=True);assert np.array_equal(teacher['boxes'].numpy(),native)
 pe,ce=parent.paths(ds,row['id'])
 with np.load(parent.RUN/ds/pe['path'])as old,np.load(parent.selected.D/ce['path'])as cur:
  arrays=dict(SourceMatchedTime=cap['Source'].copy(),ExpertInterpolation=old['Interpolation'].copy(),Ours=cur['FullMAP'].copy());iv=cur['FullMAP_interval'].copy()
 for arm in list(arrays):arrays[arm+'_interval']=iv.copy()
 arrays.update(frame_ids=np.asarray(ids),positions8=np.asarray(pos))
 proof=dict(parent=origin,original_teacher_payload_path=str(tp),original_teacher_payload_sha256=meta['sha256'],original_teacher_receipt_path=str(tp.with_suffix('.json')),original_teacher_receipt_sha256=sha(tp.with_suffix('.json')),evidence_key=key,actual_native_RGB_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),actual_eight_RGB_sha256=hashlib.sha256(frames[pos].tobytes()).hexdigest(),current_prediction_sha256=ce['sha256'],current_receipt_sha256=ce['receipt_sha256'],original_interpolation_prediction_sha256=pe['sha256'],GT_read=False)
 return frames,pos,native,arrays,proof
def committed(ds,key):
 rp=D/'pilot_v1'/ds/'receipts'/(key+'.json');e=read(rp)
 for f,h in [('path','sha256'),('support_path','support_sha256')]:assert sha(D/e[f])==e[h]
 return dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))

"""Posthoc output movement from sealed tensors; no GT, GPU or new predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,time,hashlib
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'artifacts/tastvg_directional_preference_v1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run():
 torch.set_num_threads(1);bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert bar['cells']==2304;result={};allrows={}
 for ds in ['vidstg','hc2']:
  result[ds]={};allrows[ds]={}
  for arm in ['A','E','F']:
   out=BASE/ds/arm;pb=out/'PREDICTION_BARRIER.json';assert sha(pb)==bar['files'][f'{ds}/{arm}/PREDICTION_BARRIER.json'];vals=[]
   for f in sorted((out/'online').rglob('*.pt')):
    if f.parts[-3]=='clean':continue
    assert sha(f)==read(f.with_suffix('.json'))['sha256'];x=torch.load(f,map_location='cpu',weights_only=False)
    for j,s in enumerate(x['update_steps']):
     u=s['update']
     if u is None:continue
     boxdelta=float((s['post_prediction']['boxes']-s['prediction']['boxes']).abs().mean());norm=u['actual_step_norm'];gn=u['global_gradient_norm']
     vals.append(dict(condition=x['condition'],order=x['order'],arrival=x['arrival'],step=j,box_mean_absolute_coordinate_change=boxdelta,actual_parameter_step=norm,box_change_per_parameter_norm=boxdelta/norm if norm>0 else None,RKL_norm_fraction_in_four_axes=u['rkl_probe_subspace_norm']/gn if gn>0 else None))
   allrows[ds][arm]=vals;result[ds][arm]=dict(eligible_corrupt_steps=len(vals),statistics={k:dict(mean=float(np.mean([x[k] for x in vals if x[k] is not None])),median=float(np.median([x[k] for x in vals if x[k] is not None]))) for k in ['box_mean_absolute_coordinate_change','actual_parameter_step','box_change_per_parameter_norm','RKL_norm_fraction_in_four_axes']})
 previous=read(BASE/'FUNCTIONAL_MOVEMENT.json')['datasets']
 for ds in result:
  for arm in result[ds]:
   assert result[ds][arm]['eligible_corrupt_steps']==previous[ds][arm]['eligible_corrupt_steps']
   for k,v in result[ds][arm]['statistics'].items():
    for sub,n in v.items():assert abs(n-previous[ds][arm]['statistics'][k][sub])<1e-12
 assert not torch.cuda.is_initialized()
 rowsfile=BASE/'FUNCTIONAL_STEP_ROWS.json';rowsfile.write_text(json.dumps(allrows,indent=2)+'\n')
 audit=dict(status='pass',datasets=result,rows_sha256=sha(rowsfile),code_sha256=sha(__file__),global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),GT_read=False,model_execution=False,GPU_initialized=False,scope='posthoc normalized cxcywh output coordinate movement, mean over frames and four coordinates; repeated scalar computation matches first receipt',time=time.time())
 (BASE/'FUNCTIONAL_MOVEMENT_AUDITED.json').write_text(json.dumps(audit,indent=2)+'\n');print('Audited',sum(len(v) for d in allrows.values() for v in d.values()),'sealed step movements; no GT or model execution')
if __name__=='__main__':run()

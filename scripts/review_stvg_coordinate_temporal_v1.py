"""Independent NumPy gradient/Adam/readout verification for all shared-head trials."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_coordinate_v1'
def verify(case,fit):
 import numpy as np
 from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
 from scripts.stvg_sparse_r1_student_v1 import _target_pair
 from methods.decota_final_simplified_v1.objectives import project
 from scripts.stvg_opd_p6_existing_common001 import config
 hs=[h.numpy().astype('float64')for h in _hidden_from_source_case(case)];ev=project(case['zero']['logits'],case['zero']['actions'],case['records'],config(case['dataset']));features=[];bases=[]
 for j in fit['valid_candidate_indices']:
  bf=[];ff=[]
  for h,e,rec in zip(hs,ev['offsets'],case['records']):
   s,t,_=_target_pair(rec['frame_ids'],case['temporal_candidates'][j]['interval_frames']);ij=e['ij'].cpu().numpy();at=np.flatnonzero((ij[0]==s)&(ij[1]==t))[0];bf.append(float(e['score'][at]));ff.append(np.stack([h[s],h[t]],1))
  bases.append(np.mean(bf));features.append(np.mean(ff,axis=0))
 base=np.asarray(bases);feat=np.asarray(features);reward=fit['rewards'].numpy();cfg=fit['config'];theta=np.zeros((256,2));m=np.zeros_like(theta);v=np.zeros_like(theta)
 def softmax(x):e=np.exp(x-x.max());return e/e.sum()
 target=softmax(base+reward/cfg['tau_T'])if len(base)else base
 if fit['no_information']:target=softmax(base)if len(base)else base
 assert np.allclose(target,fit['target'].numpy(),atol=2e-15,rtol=0)
 maxerr=0.;checks=0
 for row in fit['path']:
  diff=float(np.max(abs(theta-row['theta'].numpy())));maxerr=max(maxerr,diff);assert diff<2e-9
  probs=softmax(base+(feat*theta).sum((1,2)))if len(base)else base
  assert np.allclose(probs,row['candidate_probabilities'].numpy(),atol=2e-10,rtol=0)
  iv=[]
  for h,e,rec in zip(hs,ev['offsets'],case['records']):
   ij=e['ij'].cpu().numpy();scores=e['score'].cpu().numpy().astype('float64')+h[ij[0]]@theta[:,0]+h[ij[1]]@theta[:,1];at=int(scores.argmax());s,t=ij[:,at];iv.append([rec['frame_ids'][s],rec['frame_ids'][t]+1])
  assert [min(x[0]for x in iv),max(x[1]for x in iv)]==row['physical_interval'];checks+=3
  if row['step']==cfg['steps']or fit['no_information']:continue
  g=((probs-target)[:,None,None]*feat).sum(0)+2*cfg['l2']*theta;m=.9*m+.1*g;v=.999*v+.001*g*g;k=row['step']+1;theta=theta-cfg['lr']*(m/(1-.9**k))/(np.sqrt(v/(1-.999**k))+1e-8)
 return checks,maxerr

def main():
 import torch
 from scripts.run_stvg_weightfree_expand_v1 import roster,load_case
 torch.set_num_threads(2);cases={r['probe_alias']:load_case(r)[0]for r in roster()};out=[]
 for stage in read(B/'TEMPORAL_DESIGN_LOCK002.json')['stages']:
  base=B/stage['name'];bar=read(base/'GLOBAL_PREDICTION_BARRIER.json');n=checks=0;err=0.
  for item in bar['files']:
   p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False)
   for f in d['fits'].values():
    c,e=verify(cases[d['case_id']],f);checks+=c;err=max(err,e);n+=1
  out.append(dict(stage=stage['name'],fit_count=n,check_count=checks,max_theta_abs_error=err))
 if (B/'confirmation/TEMPORAL_GLOBAL_BARRIER.json').exists():
  from scripts.run_stvg_coordinate_confirmation_v1 import roster as hr,load_case as hl
  n=checks=0;err=0.
  for r in hr():
   c,_=hl(r);p=B/'confirmation/temporal'/(r['probe_alias']+'.pt');d=torch.load(p,map_location='cpu',weights_only=False)['temporal'];f=d.get('selected_temporal_fit')
   if f is not None:k,e=verify(c,f);checks+=k;err=max(err,e);n+=1
  out.append(dict(stage='confirmation',fit_count=n,check_count=checks,max_theta_abs_error=err))
 write(B/'ROOT_TEMPORAL_ARITHMETIC_READBACK.json',dict(status='pass',records=out,CPU_only=True,GT_read=False,scope='independent NumPy shared-head gradient, Adam from zero state, every grid-decode interval; original frozen feature/project inputs'))
 print(out)
if __name__=='__main__':main()

"""Finish immutable CPU outputs with native JSON scalars; no new measurements."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,time
import numpy as np
from scripts.stvg_opd_observation_response_projection_common_v1 import *

def run():
 import torch
 torch.set_num_threads(2);assert not torch.cuda.is_initialized();verify()
 b=read(BASE/'QUALIFICATION_BARRIER.json');root=read(BASE/'ROOT_QUALIFICATION.json');assert root['status']=='pass'
 rows=read(OUT/'QUALIFICATION_ROWS.json');assert len(rows)==144
 lookup={(r['parent_id'],r['arm'],r['boundary'],r['repeat']):r for r in rows}
 design={r['index']:r for r in read(BASE/'DESIGN_LOCK.json')['records']};fd=[];checks=0;gradient_checks=0
 for rel,hashed in b['files'].items():
  assert sha(ROOT/rel)==hashed;z=load(ROOT/rel)
  if 'witness' not in z:continue
  w=z['witness'];r=design[z['index']];row=lookup[z['index'],z['arm'],z['boundary'],z['repeat']]
  A=load(ROOT/r['A_path'])['fit'];k=z['boundary']
  assert row['updated']==w['updated'] and row['fallback']==w.get('fallback',False);checks+=2
  for field in ['seconds','observed_relative_error','original_norm','candidate_norm','D_A','D_candidate','lambda_R']:
   assert row[field]==w.get(field);checks+=1
  if 'derivative_checks' in w:
   fd.append(old.canonical(dict(parent_id=z['index'],dataset=z['dataset'],boundary=k,**w['derivative_checks'])))
  if not w['updated']:continue
  assert torch.equal(w['original_OPD_gradient'],A['path'][k]['update']['gradient'])
  assert old.digest(w['rollout'])==old.digest(A['rounds'][k]['rollout'])
  gradient_checks+=1792+w['rollout']['samples'].numel()
  if z['arm'] in ['P','F']:
   expect=w['J_O'].numpy().T@w['autograd_mean_gradient'].numpy().ravel()
   assert np.linalg.norm(expect-w['original_OPD_gradient'].numpy())<2e-6+2e-5*np.linalg.norm(expect)
   gradient_checks+=1792
  else:
   f=load(BASE/'qualification'/f"{z['index']:05}"/f'F_k{k}_repeat{z["repeat"]}.pt')['witness']
   Ju=f['J_U'].numpy();U=f['unobserved_positions'];e=(z['entry']['boxes'].numpy()[U]-A['before'].numpy()[U]).ravel().astype(float)
   expected=Ju.T@e/max(1,len(U));recorded=w['consistency_gradient'].numpy().astype(float)
   assert np.linalg.norm(recorded-expected)<2e-6+2e-5*np.linalg.norm(expected)
   expected_total=(w['original_OPD_gradient'].numpy()+w['lambda_R']*w['consistency_gradient'].numpy()).astype(np.float32)
   assert np.array_equal(expected_total,w['gradient'].numpy());gradient_checks+=3584
 decisions={}
 for ds in ['hc2','vidstg']:
  probes=[r for r in rows if r['dataset']==ds and r['arm']=='F' and r['updated']];accepted=sum(r['accepted'] for r in probes)
  fraction=accepted/len(probes) if probes else None
  decisions[ds]=dict(informative_probes=len(probes),accepted=accepted,fallback=len(probes)-accepted,fraction=fraction,
    technical_response_condition=bool(probes and fraction>=.5))
 repeat_failed=sum(r['repeat_scientific_status']=='failed_bitwise' for r in rows)
 eligible=all(v['technical_response_condition'] for v in decisions.values()) and all(x['status']=='pass' for x in fd) and not repeat_failed
 assert eligible==root['formal_eligible'] and repeat_failed==root['repeat_failed']
 write(OUT/'QUALIFICATION_SUMMARY.json',dict(status='complete_actual_qualification',datasets=decisions,counts=root['counts'],formal_eligible=eligible,
  repeat_failed=repeat_failed,actual_derivative_checks=fd,GT_read=False,new_DINO_calls=0,historically_exposed=True))
 write(BASE/'TECHNICAL_DECISION.json',dict(status='eligible_fixed192_panel' if eligible else 'stop_current_projection_recipe_before_formal',
  scope='native AD graph Jacobian not qualified as full physical output derivative; all fixed local measurements retained; no GT efficacy judgment',
  datasets=decisions,formal_eligible=eligible,formal_fits_executed=0,repeat_failed=repeat_failed,
  full_functional_Jacobian_qualified=all(x['status']=='pass' for x in fd),GT_read=False,CPU_only=True,time=time.time()))
 write(BASE/'ROOT_ANONYMOUS_SCALAR_READBACK.json',dict(status='pass',scope='all144 private local probe scalar fields/source likelihood and consistency VJPs/complete anonymous rows and failed FD statuses',
  count=144,total=144,scalar_checks=checks,gradient_and_sample_coordinates=gradient_checks,CPU_only=True,GT_read=False,time=time.time()))
 write(BASE/'CPU_completion_scalar_dispatch007/RESOLUTION.json',dict(status='pass',scope='numpy bool metadata converted to native JSON scalars; all prior root/source/measurements retained; no new model or scoring',
  count=144,CPU_only=True,GT_read=False,time=time.time()))
 print('SCALAR_COMPLETION',eligible,decisions,'repeat_failed',repeat_failed)

if __name__=='__main__':run()

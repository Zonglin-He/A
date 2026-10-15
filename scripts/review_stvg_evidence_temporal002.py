"""Independent NumPy objectives and full-grid decode for every saved temporal iterate."""
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def verify(case,f):
 import torch,numpy as np
 from scripts.stvg_table1_temporal_adapter_v1 import _hidden_from_source_case
 from scripts.stvg_sparse_r1_student_v1 import _target_pair
 from scripts.stvg_opd_p6_existing_common001 import config
 from methods.decota_final_simplified_v1.objectives import project
 ev=project(case['zero']['logits'],case['zero']['actions'],case['records'],config(case['dataset']));hs=[x.numpy().astype('float64')for x in _hidden_from_source_case(case)];valid=f['valid_candidate_indices'];r=np.asarray(f['rewards']);tau=f['config'].get('tau_T',1.);off=[]
 def lp(x):m=x.max();return x-m-np.log(np.exp(x-m).sum())
 def KL(q,lq,p):return float(np.sum(q*(lq-p)))
 for h,e,rec in zip(hs,ev['offsets'],case['records']):
  ij=e['ij'].numpy();b=e['score'].numpy().astype('float64');at=[];feat=[]
  for j in valid:
   s,t,_=_target_pair(rec['frame_ids'],case['temporal_candidates'][j]['interval_frames']);k=int(np.flatnonzero((ij[0]==s)&(ij[1]==t))[0]);at.append(k);feat.append(np.stack([h[s],h[t]],1))
  c=b[at];ftr=np.stack(feat)if feat else np.empty((0,256,2));q=lp(c+r/tau)if len(c)else c
  if f['no_information']and len(c):q=lp(c)
  off.append((h,ij,b,c,ftr,q,rec))
 maxerr=0.;checks=0;prev=None
 for row in f['path']:
  theta=np.asarray(row['theta']);intervals=[];gridkl=[];candkl=[]
  for h,ij,b,c,feat,q,rec in off:
   score=b+h[ij[0]]@theta[:,0]+h[ij[1]]@theta[:,1];k=int(score.argmax());s,t=ij[:,k];intervals.append([rec['frame_ids'][s],rec['frame_ids'][t]+1]);g0=lp(b);gridkl.append(KL(np.exp(g0),g0,lp(score)))
   candkl.append(KL(np.exp(q),q,lp(c+(feat*theta).sum((1,2))))if len(c)else 0.)
  assert [min(v[0]for v in intervals),max(v[1]for v in intervals)]==row['physical_interval'];checks+=2
  if f['variant']=='offset_trust':L=sum(candkl)+sum(gridkl)+.001*float(np.square(theta).sum())
  else:
   c=(off[0][3]+off[1][3])/2;feat=(off[0][4]+off[1][4])/2
   if len(c):q=lp(c+r/tau);q=lp(c)if f['no_information']else q;L=KL(np.exp(q),q,lp(c+(feat*theta).sum((1,2))))+.001*float(np.square(theta).sum())
   else:L=0.
  maxerr=max(maxerr,abs(L-row['loss']));assert abs(L-row['loss'])<2e-9
  if f['variant']in ['stable','offset_trust']and prev is not None:assert row['loss']<prev
  prev=row['loss'];checks+=1
 assert np.count_nonzero(np.asarray(f['path'][0]['theta']))==0
 if f['no_information']or f['variant']=='projection':assert f['optimizer_steps']==0 and np.count_nonzero(np.asarray(f['path'][-1]['theta']))==0
 return checks,maxerr

def main():
 import torch,numpy as np
 phase=sys.argv[1];b=BASE/phase;cohort='confirmation'if phase=='confirmation'else'development';torch.set_num_threads(2);lookup={r['probe_alias']:r for r in roster(cohort)};records=[];count=checks=0;error=0.
 for r in roster(cohort):
  case,_=load_case(r);cid=r['probe_alias'];p=b/('student'if phase=='round1'else'temporal')/(cid+'.pt');d=torch.load(p,map_location='cpu',weights_only=False);fits=d['fits']if phase=='round1'else d['fits']
  for name,f in fits.items():
   n,e=verify(case,f);checks+=n;error=max(error,e);count+=1;records.append(dict(case_id=cid,dataset=case['dataset'],variant=name,optimizer_steps=f['optimizer_steps'],nonzero_parameters=int(np.count_nonzero(np.asarray(f['path'][-1]['theta']))),interval_changed_from_projection=f['final_interval']!=case['Projection_interval'],loss_initial=f['path'][0]['loss'],loss_final=f['path'][-1]['loss'],fixed_objective_decreased=f['path'][-1]['loss']<f['path'][0]['loss']-1e-12,no_information=f['no_information'],per_offset_grid_kl=f['per_offset_grid_kl'],boundary_jumps=(f['diagnostics']['boundary_jump_count']if 'boundary_jump_count'in f['diagnostics']else sum(int(a['start_index']!=z['start_index']and z['start_boundary'])+int(a['end_index']!=z['end_index']and z['end_boundary'])for a,z in zip(f['diagnostics']['initial_offset_decodes'],f['diagnostics']['final_offset_decodes'])))))
 write(b/'ROOT_TEMPORAL_READBACK.json',dict(status='pass',case_count=32,fit_count=count,check_count=checks,max_objective_error=error,independent_NumPy_loss_and_every_grid_decode=True,GT_read=False,CPU_only=True));write(b/'TEMPORAL_MECHANISM_DIAGNOSTICS.json',dict(status='complete',records=records,GT_read=False));print('temporal root',count,checks,error)
if __name__=='__main__':main()

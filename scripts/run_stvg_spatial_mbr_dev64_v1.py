"""Frozen final-support spatial MBR, all64 outputs sealed before offline GT."""
import os,sys,time,collections
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=ROOT/'artifacts/stvg_spatial_mbr_dev64_v1';P=ROOT/'results/stvg_error_state/2026-10-19/spatial_mbr_dev64';OLD=ROOT/'artifacts/stvg_pair_direction_dev64_v1'
ARMS=['E_MBR','Gaussian','Last_center','P_MBR','Q_MBR'];EPS=1e-12;VER=None
CONTRASTS=[(a,'E_MBR')for a in ARMS[1:]]+[('Q_MBR',b)for b in ['Gaussian','Last_center','P_MBR']]+[('P_MBR',b)for b in ['Gaussian','Last_center']]+[('Last_center','Gaussian')]
ENDPOINTS=['v','t','s','unobserved_actual_s','observed_actual_s','benchmark_interpolated_s']

def verify(full=False):
 global VER
 from main_method.runtime import PinVerifier,resource_guard
 p=read(D/'PLAN_LOCK.json');r=read(D/'RUNTIME_LOCK.json');assert sha(D/'PLAN_LOCK.json')==r['plan_sha256']
 pins=dict(r['pins'])
 if (D/'RECOVERY_LOCK.json').exists():
  fix=read(D/'RECOVERY_LOCK.json');assert sha(D/'RUNTIME_LOCK.json')==fix['original_runtime_sha256'];assert sha(D/'PLAN_LOCK.json')==fix['original_plan_sha256'];pins.update(fix['replacement_pins'])
 if VER is None:VER=PinVerifier(pins)
 VER.verify(full=full);resource_guard();return p

def iou_matrix(boxes):
 import numpy as np
 b=np.asarray(boxes,float);assert b.shape==(17,4)and np.isfinite(b).all()and(b[:,2:]>0).all();lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;area=b[:,2:].prod(-1);inter=np.maximum(0,np.minimum(hi[:,None],hi[None])-np.maximum(lo[:,None],lo[None])).prod(-1);return inter/(area[:,None]+area[None]-inter)

def decide(matrix,weight,valid):
 import numpy as np
 w=np.asarray(weight,float).copy();v=np.asarray(valid,bool);assert np.isfinite(w).all()and(w>=0).all()
 if not v.any():return 0,dict(no_support=True,utility=None,tie=False)
 assert abs(w[~v]).sum()<EPS;w[~v]=0;assert w.sum()>0;w/=w.sum();u=matrix@w;best=float(u[v].max());ties=np.flatnonzero(v&(u>=best-EPS));ix=0 if 0 in ties else int(ties[0]);return ix,dict(no_support=False,utility=float(u[ix]),tie=len(ties)>1,all_utilities=u,normalized_weights=w)

def properties():
 import numpy as np
 b=np.array([[.5,.5,.2,.2]]*17);b[1,0]=.7;m=iou_matrix(b);v=np.ones(17,bool);w=np.zeros(17);w[1]=1;assert decide(m,w,v)[0]==1
 assert decide(np.ones((17,17)),np.ones(17)/17,v)[0]==0
 v[0]=False;w=np.zeros(17);w[2]=1;ix,_=decide(m,w,v);assert ix==2
 ix,_=decide(m,w,v);assert ix==decide(m,w.copy(),v)[0]
 assert decide(m,np.ones(17)/17,np.zeros(17,bool))[0]==0
 return dict(status='pass',pointmass_selects_same_box=True,tie_current_priority=True,invalid_action_excluded=True,identical_weights_same_action=True,empty_support_noop=True)

def prepare():
 assert not D.exists();D.mkdir();P.mkdir();old=read(OLD/'PLAN_LOCK.json');records=old['records'];assert len(records)==len({r['parent']for r in records})==64
 files=['scripts/run_stvg_spatial_mbr_dev64_v1.py','scripts/continue_stvg_spatial_mbr_dev64_v1.py','protocols/stvg_spatial_mbr_dev64_v1.md','configs/stvg_fine_center_dataset_candidate_v1.json','methods/CURRENT_METHOD.json','methods/MAIN_METHOD.json','methods/PAPER_METHOD.json','scripts/run_stvg_dino_local_residual32_v1.py','scripts/score_stvg_opd_p1_v1.py','scripts/score_stvg_dino_mask_align32_v1.py','vg_tta/tastvg_oracle_event5_v1.py','scripts/score_stvg_joint_robust_correction32_v1.py','artifacts/stvg_pair_direction_dev64_v1/PLAN_LOCK.json','results/stvg_error_state/2026-10-19/pair_direction_dev64/RESULTS_PUBLIC.json','results/stvg_error_state/2026-10-19/fine_dataset_formal64/RESULTS_PUBLIC.json','artifacts/stvg_fine_dataset_formal64_v1/GLOBAL_PREDICTION_BARRIER.json']+list({r['path']for r in records}|{r['result_path']for r in records})
 put(D/'PLAN_LOCK.json',dict(status='locked',count=64,records=records,arms=ARMS,contrasts=CONTRASTS,final_round_only=True,EPS=EPS,readout='argmax expected unclipped normalized-box IoU on identical saved valid support',tie='current if tied and valid, otherwise lowest original index',gate='original allowed and informative; false retains saved Last_center',frozen_HCstep1_Vidstep05_tau01=True,new_GPU_models_experts_features_sampling_training_OPD=0,formal64_excluded=True,GT_offline_after_global64_barrier=True,development_only=True,CPU_cap_seconds=900,created_epoch=time.time()))
 put(D/'RUNTIME_LOCK.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins={f:sha(ROOT/f)for f in files}));put(D/'CPU_PROPERTIES.json',properties());print('prelocked64',flush=True)

def same(a,b):
 import numpy as np,torch
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(same(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
 return a==b

def measure():
 import numpy as np,torch
 torch.set_num_threads(4);p=verify(full=True);start=time.perf_counter();files=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(ROOT/r['path']);fit=x['fits'][r['arm']];assert len(fit['rounds'])==2;t=fit['rounds'][-1];b=t['candidates'].numpy();center=fit['path'][-2]['boxes'].numpy();E=np.asarray(x['outputs']['E_MBR']);final=fit['final'].numpy();assert np.array_equal(center,b[:,0]);active=x['allowed'].numpy()&t['feedback']['informative'].numpy();valid=t['feedback']['valid'].numpy();q=t['feedback']['prob'].numpy();sp=t['student_prob'].numpy();out=dict(E_MBR=E,Gaussian=final,Last_center=center,P_MBR=center.copy(),Q_MBR=center.copy());decisions=[]
  for j in range(len(b)):
   rec=dict(node=j,active=bool(active[j]),valid_candidates=int(valid[j].sum()))
   if active[j]:
    m=iou_matrix(b[j]);rec['matrix_symmetry_error']=float(abs(m-m.T).max());assert rec['matrix_symmetry_error']<1e-15;assert abs(m.diagonal()-1).max()<1e-12
    for arm,w in [('P_MBR',sp[j]),('Q_MBR',q[j])]:
     ix,a=decide(m,w,valid[j]);out[arm][j]=b[j,ix];rec[arm]=dict(index=ix,**a)
   else:assert np.array_equal(final[j],center[j])
   decisions.append(rec)
  for arm,z in out.items():assert np.array_equal(z[~x['allowed'].numpy()],E[~x['allowed'].numpy()])and np.array_equal(z[:,2:],E[:,2:])and np.isfinite(z).all()
  payload=dict(alias=r['alias'],dataset=r['dataset'],panel=r['panel'],active=bool(t['feedback']['per_anchor_rewards'].shape[0]),frame_ids=x['frame_ids'],planned_positions=x['planned_positions'],interval=x['interval'],allowed=x['allowed'],active_gate=active,outputs=out,decisions=decisions,GT_read=False,new_OPD=False)
  if 'active'in x:assert x['active']==payload['active']
  dest=D/'predictions'/(r['alias']+'.pt')
  if dest.exists():
   assert same(load(dest),payload);files.append(read(dest.with_suffix('.receipt.json')))
  else:files.append(save(dest,payload))
  print(r['alias'],'five outputs sealed',flush=True)
 verify(full=True);put(D/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=64,files=files,all_outputs_before_GT=True,CPU_seconds=time.perf_counter()-start,new_GPU_models_experts_training_OPD=0));print('global64 seal complete',flush=True)

def aggregate(rows):
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 return dict(parents=len(rows),active=sum(r['active']for r in rows),means={a:{e:mean([r['values'][a][e]for r in rows])for e in ENDPOINTS}for a in ARMS},paired={a+'_minus_'+b:{e:paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rows])for e in ENDPOINTS}for a,b in CONTRASTS})

def score():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all_outputs_before_GT'];start=time.perf_counter()
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[];nodes=[]
 for r in p['records']:
  verify();assert time.perf_counter()-start<900;x=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];m=metrics(x,plans[ds][q],gt[ds][0][q],gt[ds][1][q]);old=next(z for z in read(ROOT/r['result_path'])['rows']if z['alias']==r['alias'])
  for a,b in [('E_MBR','E_MBR'),('Gaussian',r['arm'])]:
   for e in ENDPOINTS:
    u,v=m['values'][a][e],old['values'][b][e];assert u is None and v is None or abs(u-v)<1e-10
  assert len({m['values'][a]['t']for a in ARMS})==1
  public=[]
  for d in x['decisions']:
   nd={k:v for k,v in d.items()if k not in ['P_MBR','Q_MBR']};nd.update(alias=r['alias'],dataset=ds,panel=r['panel'])
   for a in ['P_MBR','Q_MBR']:
    if a in d:nd[a]={k:v for k,v in d[a].items()if k not in ['all_utilities','normalized_weights']}
   public.append(nd)
  rows.append(dict(alias=r['alias'],dataset=ds,panel=r['panel'],active=x['active'],active_nodes=int(x['active_gate'].sum()),decisions=dict(q_changed_center=sum(d.get('Q_MBR',{}).get('index',0)!=0 for d in public),p_changed_center=sum(d.get('P_MBR',{}).get('index',0)!=0 for d in public),q_p_different=sum(d.get('Q_MBR',{}).get('index',0)!=d.get('P_MBR',{}).get('index',0)for d in public)),**m));nodes.extend(public);print(r['alias'],'scored',flush=True)
 groups={pan:{ds:aggregate([r for r in rows if r['dataset']==ds and(pan=='combined'or r['panel']==pan)])for ds in gt}for pan in ['original','selection','combined']}
 verify(full=True);put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=64,arms=ARMS,groups=groups,rows=rows,all_outputs_before_GT=True,formal64_excluded=True,new_OPD=False,development_exposed=True,independent_confirmation=False));put(P/'NODE_DECISIONS_PUBLIC.json',dict(status='complete',nodes=nodes,no_raw_boxes_GT_or_parent_identity=True));put(D/'SCORE_COMPLETION.json',dict(status='complete',CPU_seconds=time.perf_counter()-start,count=64));print('64 score complete',flush=True)
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','measure','score']);arg=ap.parse_args()
 try:globals()[arg.stage]()
 except BaseException as e:
  D.mkdir(exist_ok=True);put(D/('FAILURE_'+str(time.time_ns())+'.json'),dict(stage=arg.stage,error_type=type(e).__name__,error=str(e),all_partials_preserved=True));raise

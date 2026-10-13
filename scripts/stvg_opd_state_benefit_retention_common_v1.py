"""Cache-only arithmetic and immutable scope for the retention audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,gzip,json,math,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load,save
from scripts import stvg_opd_LN_state_intervention_common_v1 as old
BASE=ROOT/'artifacts/stvg_opd_state_benefit_retention_audit_v1'
OUT=ROOT/'results/stvg_opd_state_benefit_retention_audit/2026-10-13'
ARMS=old.ARMS;PAIRS=[('Source','Inherited_order1'),('Source','Inherited_order2'),('Inherited_order1','Inherited_order2')]
MASKS=['own_U','common_U','observed'];BOOT=10000;SEED=20261013;EPS=1e-12
SCIENCE=['protocols/stvg_opd_state_benefit_retention_audit_v1.md',
 'scripts/stvg_opd_state_benefit_retention_common_v1.py','scripts/run_stvg_opd_state_benefit_retention_audit_v1.py',
 'scripts/score_stvg_opd_state_benefit_retention_audit_v1.py','scripts/audit_stvg_opd_state_benefit_retention_audit_v1.py']
def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(x) for x in f]
def row_writer(path):
 assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
 return gzip.GzipFile(filename='',fileobj=path.open('xb'),mode='wb',mtime=0)
def emit(stream,row):stream.write((json.dumps(row,allow_nan=False,separators=(',',':'))+'\n').encode())
def write_rows(path,items):
 with row_writer(path) as f:
  for r in items:emit(f,r)
def sign(x):return 'unknown' if x is None else 'positive' if x>EPS else 'negative' if x< -EPS else 'neutral'
def mean(v):return math.fsum(v)/len(v) if len(v) else None
def difference(a,b):return None if a is None or b is None else a-b
def estimate(values,independent=False):
 v=np.array([x for x in values if x is not None],float);n=len(v)
 if not n:return dict(defined_parents=0,mean=None,ci95=None)
 idx=np.random.default_rng(SEED).integers(n,size=(BOOT,n))
 if independent:
  weights=np.stack([(idx==i).sum(1) for i in range(n)],1);means=weights@v/n;avg=math.fsum(v)/n
 else:means=v[idx].mean(1);avg=float(v.mean())
 return dict(defined_parents=n,mean=avg,ci95=np.quantile(means,[.025,.975]).tolist())
def distribution(v,independent=False):
 valid=[x for x in v if x is not None]
 return dict(total_parents=len(v),unknown_parents=len(v)-len(valid),positive_parents=sum(x>EPS for x in valid),
  negative_parents=sum(x< -EPS for x in valid),neutral_parents=sum(abs(x)<=EPS for x in valid),
  negative5pp=sum(x<-.05 for x in valid),negative20pp=sum(x<-.2 for x in valid),estimate=estimate(v,independent))
def iou(a,b,independent=False,floor=0.):
 a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
 if independent:
  from vg_tta.tastvg_oracle_event5_v1 import box_iou
  if not floor:return np.array([float(box_iou(x,y)) for x,y in zip(a,b)])
  out=[]
  for x,y in zip(a,b):
   w=max(0,min(x[2],y[2])-max(x[0],y[0]));h=max(0,min(x[3],y[3])-max(x[1],y[1]))
   inter=w*h;union=max(0,x[2]-x[0])*max(0,x[3]-x[1])+max(0,y[2]-y[0])*max(0,y[3]-y[1])-inter
   out.append(inter/max(union,floor))
  return np.array(out)
 lo=np.maximum(a[:,:2],b[:,:2]);hi=np.minimum(a[:,2:],b[:,2:]);inter=np.maximum(hi-lo,0).prod(1)
 union=np.maximum(a[:,2:]-a[:,:2],0).prod(1)+np.maximum(b[:,2:]-b[:,:2],0).prod(1)-inter
 return np.divide(inter,np.maximum(union,floor),out=np.zeros(len(a)),where=union>0)
def corners(boxes,width=1.,height=1.,clip=False,independent=False):
 boxes=np.asarray(boxes,np.float64)
 if independent:
  from vg_tta.tastvg_paper48_metrics_v1 import xyxy
  out=xyxy(boxes,width,height)
 else:
  wh=np.array([width,height]);out=np.concatenate(((boxes[:,:2]-boxes[:,2:]/2)*wh,(boxes[:,:2]+boxes[:,2:]/2)*wh),1)
 return np.maximum(out,0) if clip else out
def distances(a,b,independent=False):
 if independent:
  rms=np.array([math.sqrt(math.fsum((float(x)-float(y))**2 for x,y in zip(p,q))/4) for p,q in zip(a,b)])
 else:rms=np.sqrt(np.square(np.asarray(a,float)-np.asarray(b,float)).mean(1))
 return rms,iou(corners(a,independent=independent),corners(b,independent=independent),independent,floor=1e-7)
def vector(state):return np.concatenate([v.numpy().ravel() for v in state.values()]).astype(float)
def norm(x):return float(np.linalg.norm(x))
def cosine(x,y):
 n=norm(x)*norm(y);return float(np.dot(x,y)/n) if n else None
def verify():
 r=read(BASE/'RUNTIME.json');assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
 for n,h in {**r['pins'],**r['bindings'],**r['protected']}.items():assert sha(ROOT/n)==h,n
 old.verify();assert read(old.BASE/'FINAL_COMPLETION.json')['status']=='complete'
 from scripts.run_stvg_opd_LN_state_intervention_v1 import standby
 standby();return r
def require_seal():
 verify();b=read(BASE/'GLOBAL_READOUT_BARRIER.json');assert b['status']=='sealed' and not b['GT_read']
 assert b['parents']==64 and b['fits']==192 and b['arm_boundaries']==4992 and b['saved_updates']==4800
 assert b['runtime_sha256']==sha(BASE/'RUNTIME.json')
 for rel,h in b['files'].items():assert sha(ROOT/rel)==h
 return b
def contracts():
 # Pure arithmetic fixtures; no model, gradients, optimizer or training is run.
 assert abs(difference(.02,.05)+.03)<1e-15
 assert abs((.03-.01)-(.04-.06)-((.03-.04)-(.01-.06)))<1e-15
 assert .6-.5>0 and .7-.5>.6-.5  # advantage erosion with inherited improvement
 b=np.array([[.5,.5,.4,.4],[.1,.1,.2,.2]])
 assert np.allclose(iou(corners(b),corners(b)),1)
 a=np.array([[0.,0.,.2,.2]]);c=np.array([[1.,1.,.2,.2]])
 assert distances(a,c)[0][0]>0 and mean([.3,.7])==mean([.5,.5])
 assert cosine(np.zeros(2),np.ones(2)) is None
 assert mean([]) is None and sign(None)=='unknown' and sign(0)=='neutral'
 assert estimate([None,None])['mean'] is None
 from scripts.audit_stvg_opd_LN_state_intervention_v1 import near
 near(estimate([.2,.3,None,.4]),estimate([.2,.3,None,.4],True))
 try:require_seal()
 except (FileNotFoundError,AssertionError):pass
 else:raise AssertionError('Preseal label guard failed')
 return dict(status='pass',count=10,scope='CPU arithmetic/identity/source catch-up vs inherited drop/output vs mean quality/unknown zero cosine/independent parent bootstrap/preseal refusal',CPU_only=True,GT_read=False,time=time.time())
def prepare():
 import torch
 assert not torch.cuda.is_initialized() and not BASE.exists() and not OUT.exists();BASE.mkdir();OUT.mkdir(parents=True)
 from scripts.score_stvg_opd_LN_state_intervention_v1 import require_seal as prior_seal
 b=prior_seal();records=read(old.BASE/'DESIGN_LOCK.json')['records'];assert len(records)==64
 bindings={}
 for rel,h in b['files'].items():
  assert sha(ROOT/rel)==h;receipt=(ROOT/rel).with_suffix('.json');r=read(receipt);assert r['sha256']==h
  bindings[rel]=h;bindings[str(receipt.relative_to(ROOT))]=sha(receipt)
 for n in ['DESIGN_LOCK.json','RUNTIME.json','GLOBAL_PREDICTION_BARRIER.json','ROOT_QUALIFICATION.json','ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','FINAL_COMPLETION.json','CLOSING_FINAL_GITHUB_RECEIPT.json']:
  p=old.BASE/n;bindings[str(p.relative_to(ROOT))]=sha(p)
 write(BASE/'DESIGN_LOCK.json',dict(status='locked_before_new_offline_label_readout',records=records,parents=64,fits=192,arms=ARMS,
  saved_updates=4800,arm_boundaries=4992,historical_GT_exposure=True,bootstrap=BOOT,seed=SEED,
  primary='own_U real GT-event decoder frames; all Omega0 visits excluded',secondary='common_U and observed real frames; all-frame output convergence',
  initial_correct_IoU=.5,advantage_tiers=[0,.01,.05],retention_min_initial_advantage=.01,
  early_quarter_steps={'hc2':10,'vidstg':3},persistent_final_quarter_start={'hc2':30,'vidstg':8},
  no_GT_step_or_parameter_selection=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_optimizer_steps=0,production_promoted=False,time=time.time()))
 write(BASE/'CPU_CONTRACTS.json',contracts())
 bindings[str((BASE/'CPU_CONTRACTS.json').relative_to(ROOT))]=sha(BASE/'CPU_CONTRACTS.json')
 protected={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'methods/CURRENT_METHOD.json',ROOT/'methods/decota_spatial_opd_v1/configs.json',
  ROOT/'artifacts/stvg_opd_temporal_guided_spatial_candidate_v1/P7_A/FINAL_COMPLETION.json',old.S/'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json']}
 shared=['scripts/score_stvg_opd_LN_state_intervention_v1.py','scripts/audit_stvg_opd_LN_state_intervention_v1.py',
  'scripts/stvg_opd_LN_state_intervention_common_v1.py','scripts/stvg_opd_legacy_input_bridge003.py',
  'vg_tta/decota_fixed_full_audit_v1.py','vg_tta/stvg_opd_common_state_geometry_arithmetic004.py']
 write(BASE/'RUNTIME.json',dict(status='locked_CPU_cache_only',pins={n:sha(ROOT/n) for n in SCIENCE+shared},bindings=bindings,protected=protected,
  design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_only=True,GT_read=False,time=time.time()))
 verify();status(BASE/'STATUS.json',dict(status='prepared_pending_all_cached_readouts',done=0,total=192,CPU_only=True,GT_read=False,time=time.time()))

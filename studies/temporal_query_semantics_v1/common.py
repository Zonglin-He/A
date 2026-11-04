"""Same-target different-action temporal diagnostics; no current-method changes."""
from pathlib import Path
import os,sys,json,time,hashlib,shutil,inspect
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
R=Path(__file__).resolve().parents[2];S=Path(__file__).parent;D=R/'outputs/temporal_query_semantics_v1';P=R/'results/temporal_query_semantics_v1';OLD=R/'outputs/reviewer_key_controls_v1'
sys.path[:0]=[str(R/'studies/reviewer_key_controls_v1'),str(R),str(R/'tools')]
import common_v1 as old
import numpy as np,torch
from stvg.prototype import one_step,update,inclusive_scores
from stvg.current_shared_roi import map_interval
read=old.read;sha=old.sha;atomic=old.atomic;digest=old.digest;identity=old.identity;fixed=old.fixed
ARMS=['SourceNative','JointPrior','SourceROI','CorrectedROI','CrossFeatures','CrossPrior','CrossBoth','CrossEvidence','CommonPrior']
def resources(start):
 assert time.perf_counter()-start<259200 and shutil.disk_usage(R).free>=80*2**30
 assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:'))>=8*2**30
 assert sum(p.stat().st_size for p in D.rglob('*')if p.is_file())<160*2**30
def validate():
 p=read(D/'plan.json');assert digest({k:v for k,v in p.items()if k!='fingerprint'})==p['fingerprint']
 for grp in ['code_sha256','input_sha256']:
  for f,h in p[grp].items():assert sha(R/f)==h,f
 return p
def guard():old.infer_guard()
def source_inputs(key):
 rr={r['id']:r for r in fixed(R/'outputs/temporal_query_semantics_v1/records.json')};c,proof=old.source('vidstg',rr[key]);e=fixed(OLD/'temporal/vidstg/sealed.json')['files'][key];assert sha(OLD/e['path'])==e['sha256']and sha(OLD/e['receipt_path'])==e['receipt_sha256']
 with np.load(OLD/e['path'])as z:
  q,d=one_step(c['features']['AuxE_ROI'],c['q0'],kappa=0,beta=1);qs,ds=one_step(c['features']['AuxSourceROI'],c['q0'],kappa=0,beta=1)
  assert np.array_equal(d['L'],z['evidence_CorrectedROI'])and np.array_equal(ds['L'],z['evidence_SourceROI'])and np.array_equal(c['q0'],z['q0'])
  assert np.array_equal(np.asarray(map_interval(c['ids'],q)),z['Ours_interval'])and np.array_equal(np.asarray(map_interval(c['ids'],qs)),z['SourceTR_interval'])
  return dict(ids=np.asarray(c['ids']),pos=np.asarray(c['pos']),q0=c['q0'].copy(),F=c['features']['AuxE_ROI'].copy(),Fs=c['features']['AuxSourceROI'].copy(),native=z['Source_interval'].copy()),dict(compact_sha256=proof['compact_sha256'],parent_capture_sha256=c['parent_capture_sha256'],old_temporal_prediction_sha256=e['sha256'],old_temporal_receipt_sha256=e['receipt_sha256'])
def derive(a,b,caption=''):
 assert np.array_equal(a['ids'],b['ids']);q0=a['q0'];F=a['F'];base,dd=one_step(F,q0,kappa=0,beta=1);other,od=one_step(b['F'],b['q0'],kappa=0,beta=1);src,sd=one_step(a['Fs'],q0,kappa=0,beta=1);cf,fd=one_step(b['F'],q0,kappa=0,beta=1);cp,pd=one_step(F,b['q0'],kappa=0,beta=1);common,cd=one_step(F,(q0+b['q0'])/2,kappa=0,beta=1);ce=update(q0,inclusive_scores(od['L']))
 # Metadata-only caption changes do not enter one_step: direct input-contract control,
 # not a new end-to-end text counterfactual through the frozen Source network.
 altered,ad=one_step(F,q0,kappa=0,beta=1);assert np.array_equal(base,altered)and np.array_equal(dd['L'],ad['L'])
 neutral,nd=one_step(np.zeros_like(F),q0,kappa=0,beta=1);assert np.array_equal(neutral,q0)and not np.any(nd['L'])
 qq=dict(JointPrior=q0,SourceROI=src,CorrectedROI=base,CrossFeatures=cf,CrossPrior=cp,CrossBoth=other,CrossEvidence=ce,CommonPrior=common)
 ll=dict(SourceROI=sd['L'],CorrectedROI=dd['L'],CrossFeatures=fd['L'],CrossPrior=pd['L'],CrossBoth=od['L'],CommonPrior=cd['L'],CrossEvidence=od['L'])
 out=dict(ids=a['ids'],positions8=a['pos'],input_F=F,input_Fs=a['Fs'],input_q0=q0,input_otherF=b['F'],input_otherq0=b['q0'],SourceNative_interval=a['native'])
 for k,q in qq.items():out[k+'_q']=q;out[k+'_interval']=np.asarray(map_interval(a['ids'],q))
 for k,L in ll.items():out[k+'_L']=L
 out['text_metadata_control_q']=altered;out['text_metadata_control_L']=ad['L'];out['neutral_features_q']=neutral
 return out

def bootstrap(rows,fields):
 if not rows:return dict(pairs=0,statistics={})
 n=len(rows);x=np.array([[np.nan if r.get(k)is None else r[k]for k in fields]for r in rows]);w=np.random.default_rng(20261104).multinomial(n,np.ones(n)/n,size=10000);out={}
 for j,k in enumerate(fields):
  v=x[:,j];mask=np.isfinite(v);num=w[:,mask]@v[mask];den=w[:,mask].sum(1);b=np.divide(num,den,out=np.full(10000,np.nan),where=den>0);b=b[np.isfinite(b)];out[k]=dict(mean=float(np.mean(v[mask]))if mask.any()else None,CI95=np.quantile(b,[.025,.975]).tolist()if len(b)else None,pairs=int(mask.sum()))
 return dict(pairs=n,independent_videos=n,statistics=out)
def tiou(iv,gt):
 i=max(0,min(float(iv[1]),float(gt[1]))-max(float(iv[0]),float(gt[0])));u=max(float(iv[1]),float(gt[1]))-min(float(iv[0]),float(gt[0]));return i/u if u>0 else 0.

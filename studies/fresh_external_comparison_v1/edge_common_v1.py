"""Original fixed256 EdgeTAM protocol with audited cache-optional current inputs."""
from pathlib import Path
import os,sys,time,json,hashlib,shutil
sys.path.insert(0,str(Path(__file__).parent))
import current_evidence_runtime_v2 as preparation
import current_evidence_v1 as current
from current_evidence_v1 import R,N,read,sha,atomic,identity,digest,fixed,resources
S=R/'studies/edgetam_comparison_v1';OLD=R/'outputs/edgetam_comparison_v1';D=N/'edgetam_v1';P=R/'results/fresh_external_comparison_v1/edgetam_v1';parent=current.rc
sys.path.insert(0,str(S))
ARMS=['SourceMatchedTime','ExpertInterpolation','EdgeTAM','Ours'];PAIRS=[('EdgeTAM','SourceMatchedTime'),('EdgeTAM','ExpertInterpolation'),('Ours','EdgeTAM')]
def rows(ds):return current.pilot_rows(ds)
def validate():
 preparation.validate_execution();p=read(N/'edge_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 return p
def release():
 q=read(N/'current_audit_completion_v1.json');assert q['status']=='verified_all256_current_expert_contexts_and_matched_baselines_independent_no_GT_audit';assert q['totals']==dict(queries=256,cached_current_expert=250,new_current_expert=6)
 for ds in ['hc2','vidstg']:
  e=q['domains'][ds];assert sha(R/e['path'])==e['sha256']and read(R/e['path'])['queries']==128
 for f in ['current_evidence','current_audit']:
  name='worker_v1.json'if f=='current_evidence'else'current_audit_worker_v1.json';x=read(N/name);stat=Path('/proc')/str(x['pid'])/'stat';assert not stat.exists()or int(stat.read_text().split(') ',1)[1].split()[19])!=x['startticks']
 assert not(N/'failure_v1.json').exists()and not(N/'current_audit_failure_v1.json').exists()
 return dict(current_whole256_audit_sha256=sha(N/'current_audit_completion_v1.json'),cached_current_expert=250,actual_recomputed_current_expert=6,new_Source=0,CG_gamma_corruption_paused=True)
def save(path,obj):
 import torch
 path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def payload(ds,row):
 import numpy as np,torch
 cap,raw,frames,proof=current.inputs(ds,row);e=fixed(N/ds/'sealed.json')['files'][row['id']];assert proof==e['input_binding'];tp=Path(e['current_teacher']['payload_path']);assert sha(tp)==e['current_teacher']['payload_sha256']and sha(e['current_teacher']['receipt_path'])==e['current_teacher']['receipt_sha256'];native=torch.load(tp,map_location='cpu',weights_only=False,mmap=True)['boxes'].numpy();assert sha(N/e['path'])==e['sha256']and sha(N/e['receipt_path'])==e['receipt_sha256']
 with np.load(N/e['path'])as z:arrays=dict(SourceMatchedTime=cap['Source'].copy(),ExpertInterpolation=z['Simple'].copy(),Ours=z['Ours'].copy());iv=z['Ours_interval'].copy()
 for arm in list(arrays):arrays[arm+'_interval']=iv.copy()
 arrays.update(frame_ids=np.asarray(cap['frame_ids']),positions8=np.asarray(cap['positions8']))
 binding=dict(actual_input_binding=proof,current_teacher=e['current_teacher'],current_baseline_prediction_sha256=e['sha256'],current_baseline_receipt_sha256=e['receipt_sha256'],current_baseline_whole_audit_sha256=sha(N/ds/'current_audit_receipt_v1.json'),original_fixed128_parents=True,GT_read=False)
 return frames,cap['positions8'],native,arrays,binding
def committed(ds,key):
 rp=D/'pilot_v1'/ds/'receipts'/(key+'.json');e=read(rp)
 for f,h in [('path','sha256'),('support_path','support_sha256')]:assert sha(D/e[f])==e[h]
 return dict(e,receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))

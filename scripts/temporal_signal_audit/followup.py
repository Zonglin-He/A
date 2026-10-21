"""Explicitly post-hoc sensitivity of the observed hidden-change association.

No additional time decoder call. New label-free positional/residual curves seal
before the sensitivity evaluator reads GT. These are diagnostics, not a readout.
"""
import os,sys,json,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import OUT,PUBLIC,read,write,sha,arr,cosine_change
from scripts.temporal_signal_audit.evaluate import auc,ci

def run():
 torch.set_num_threads(1);os.nice(15);plan=read(OUT/'plan.json');sealed=read(OUT/'sealed.json');assert not(OUT/'sensitivity_plan.json').exists()
 protocol=dict(status='posthoc_after_primary_results',questions=['Fixed positional-query change as negative control','Within-query label-free residual after positional/nuisance factors','Residual after native boundary mass: information not already expressed by the existing head','Exclude GT boundaries within2s of captured edges'],
 boundary_near_seconds=1,boundary_far_seconds=2,minimum_per_class=2,minimum_boundary_distance_from_capture_edges_seconds=2,
 additional_decoder_calls=0,new_parameters_or_predictions=0,source_sha256=sha(Path(__file__)),original_signal_seal_sha256=sha(OUT/'sealed.json'),primary_report_sha256=sha(PUBLIC/'report.json'))
 write(OUT/'sensitivity_plan.json',protocol);paths=[]
 for item,se in zip(plan['items'],sealed['files']):
  c=torch.load(ROOT/item['capture'],map_location='cpu',weights_only=False);z=np.load(OUT/se['path'],allow_pickle=False);T=len(z['frame_ids']);v=np.zeros((T,256));q=np.zeros_like(v)
  for o in c['temporal_cache']['offsets']:
   ix=np.asarray(o['frame_positions'],int);v[ix]=arr(o['inputs']['query_time'][:,0]);q[ix]=arr((o['inputs']['query_tgt']+o['inputs']['query_time'])[:,0])
  pos=cosine_change(v);query=cosine_change(q);x=z['time_position'];X=np.column_stack([np.ones(T),x,x*x,z['anchor_proximity'],np.log(z['area'].clip(1e-8)),pos,query]);h=z['hidden_change'];r=h-X@np.linalg.lstsq(X,h,rcond=None)[0];Y=np.column_stack([X,z['native_boundary_mass']]);res=h-Y@np.linalg.lstsq(Y,h,rcond=None)[0]
  p=OUT/'sensitivity_signals'/f"{item['anonymous']}.npz";p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,positional_change=pos,initial_query_change=query,hidden_change=h,hidden_without_position_nuisance=r,hidden_without_position_and_head=res,head_boundary_mass=z['native_boundary_mass'])
  paths.append(dict(anonymous=item['anonymous'],path=str(p.relative_to(OUT)),sha256=sha(p)))
 write(OUT/'sensitivity_sealed.json',dict(files=paths,GT_read=False,additional_decoder_calls=0,CUDA_initialized=torch.cuda.is_initialized(),plan_sha256=sha(OUT/'sensitivity_plan.json')))
 labels={}
 for cohort,v in plan['labels'].items():
  assert sha(ROOT/v['path'])==v['sha256'];labels[cohort]={g['id']:g for g in [json.loads(l)for l in (ROOT/v['path']).read_text().splitlines()if l.strip()]}
 rows=[]
 for item,se,ss in zip(plan['items'],sealed['files'],paths):
  assert sha(OUT/ss['path'])==ss['sha256'];z=np.load(OUT/se['path']);v=np.load(OUT/ss['path']);t=z['frame_ids']/item['fps'];s,e=np.asarray(labels[item['cohort']][item['id']]['interval'])/item['fps'];unobs=~z['observed']
  for scope in ['all_captured_boundaries','both_boundaries_interior2s']:
   if scope=='both_boundaries_interior2s'and not(t[0]+2<s<e<t[-1]-2):continue
   if not(t[0]<=s<=t[-1]and t[0]<=e<=t[-1]+1/item['fps']):continue
   d=np.minimum(abs(t-s),abs(t-e));mask=((d<=1)|(d>2))&unobs;y=(d<=1)[mask]
   for name in v.files:
    a=auc(y,v[name][mask])
    if a is not None:rows.append(dict(cohort=item['cohort'],query=item['anonymous'],scope=scope,signal=name,auc=a))
 summary={}
 for c in ['HC64','HC32','Vid32']:
  summary[c]={}
  for scope in ['all_captured_boundaries','both_boundaries_interior2s']:
   summary[c][scope]={k:ci([r['auc']for r in rows if r['cohort']==c and r['scope']==scope and r['signal']==k])for k in ['positional_change','initial_query_change','hidden_change','hidden_without_position_nuisance','hidden_without_position_and_head','head_boundary_mass']}
 write(PUBLIC/'sensitivity.json',dict(protocol=protocol,summary=summary,queries_anonymous=rows,all_new_curves_sealed_before_GT=True,signal_seal_sha256=sha(OUT/'sensitivity_sealed.json')));print(json.dumps(summary,indent=2))
if __name__=='__main__':run()

"""Saved-state, independent ranking and noninterruption verification only."""
import os,sys,json,csv,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.temporal_signal_audit.extract import OUT,PUBLIC,read,write,sha

def run():
 p=read(OUT/'plan.json');s=read(OUT/'sealed.json');report=read(PUBLIC/'report.json');original=read(OUT/'evaluation_receipt.json')
 assert sha(PUBLIC/'report.json')==original['report_sha256']and sha(PUBLIC/'signed_queries.tsv')==original['signed_sha256'];assert sha(OUT/'plan.json')==s['plan_sha256']
 for k,v in p['sources'].items():assert sha(ROOT/k)==v
 labels={cohort:{r['id']:r for r in [json.loads(l)for l in (ROOT/v['path']).read_text().splitlines()if l.strip()]}for cohort,v in p['labels'].items()}
 for v in p['labels'].values():assert sha(ROOT/v['path'])==v['sha256']
 with open(PUBLIC/'signed_queries.tsv')as f:rows=list(csv.DictReader(f,delimiter='\t'))
 index={(r['query'],r['analysis'],r['signal']):r for r in rows};checks=0;error=0.
 for item,file in zip(p['items'],s['files']):
  for name in ['capture','posterior','receipt']:assert sha(ROOT/item[name])==item[name+'_sha256']
  assert sha(OUT/file['path'])==file['sha256'];z=np.load(OUT/file['path']);ids=z['frame_ids'];assert (np.diff(ids)>0).all()and z['observed'].sum()==8
  assert abs(z['temporal_key_salience'].mean()-1)<1e-6
  assert np.max(abs(z['attention_placement']-(z['roi_mass']-z['permuted_roi_mass'])))<1e-12
  for k in ['roi_mass','permuted_roi_mass','visual_mass']:assert z[k].min()>=-1e-12 and z[k].max()<=1+1e-6
  t=ids/item['fps'];a,b=np.array(labels[item['cohort']][item['id']]['interval'])/item['fps'];inside=(t>=a)&(t<b);d=np.minimum(abs(t-a),abs(t-b));eligible=((d<=1)|(d>2))&~z['observed']
  for analysis,key,y,mask,covered in [('event_unobserved','attention_placement',inside,~z['observed'],True),('boundary_either_unobserved','hidden_change',d<=1,eligible,t[0]<=a<=t[-1]and t[0]<=b<=t[-1]+1/item['fps'])]:
   y=y[mask];v=z[key][mask];n=int(y.sum());m=len(y)-n;r=index.get((item['anonymous'],analysis,key))
   if not covered or min(n,m)<2:assert r is None;continue
   rank=rankdata(v,method='average');value=(rank[y].sum()-n*(n+1)/2)/(n*m);e=abs(value-float(r['auc']));assert e<2e-14;error=max(error,e);checks+=1
 # Group independence uses pre-existing manifest metadata, not private ID spelling.
 records=[json.loads(l)for l in (ROOT/'data/evaluation/development.jsonl').read_text().splitlines()if l.strip()];records={r['id']:r for r in records}
 hc={records[x['id']]['metadata']['source']for x in p['items']if x['cohort']=='HC64'}
 selection=read(ROOT/'outputs/interval_pool/confirmation_selection.json')['development_recheck_selection'];hcr={r['record']['metadata']['source']for r in selection['hc2']};vid={r['record']['metadata']['source']for r in selection['vidstg']};assert len(hc)==64 and len(hcr)==32 and len(vid)==32 and not hc&hcr
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert all(sha(ROOT/k)==v for k,v in formal['sources'].items());assert sha(ROOT/'outputs/boxfixed_full_v1/plan.json')==p['formal_plan_sha256']
 progress=read(ROOT/'outputs/boxfixed_full_v1/vidstg/progress.json');pid=progress['pid'];stat=Path(f'/proc/{pid}/stat').read_text();ticks=int(stat[stat.rfind(')')+2:].split()[19]);assert ticks==progress['startticks']and progress['completed']>=4763
 result=dict(status='pass',queries=128,input_capture_posterior_receipt_signal_hashes_rechecked=True,independent_rank_sum_checks=checks,rank_sum_max_error=error,attention_placement_identity_and_probability_bounds=True,eight_observation_masks_exact=True,HC64_parents=64,HC32_parents=32,Vid32_parents=32,HC_parent_intersection=0,frozen_formal_sources_unchanged=len(formal['sources']),formal_worker_identity_verified=True,formal_progress_advanced_from=4763,formal_progress_at_verification=progress['completed'],formal_fingerprint=progress['fingerprint'])
 write(PUBLIC/'verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':run()

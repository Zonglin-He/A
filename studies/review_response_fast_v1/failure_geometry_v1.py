"""Whole-target output-geometry census, no new GT labels or identity claims."""
from common import *
import argparse,csv

def prepare():
 assert not(D/'failure_geometry_plan_v1.json').exists();inputs={str((selected.D/f'components_{ds}_sealed.json').relative_to(R)):sha(selected.D/f'components_{ds}_sealed.json')for ds in ['hc2','vidstg']}
 inputs.update({str((selected.D/f'components_{ds}_reporting_receipt.json').relative_to(R)):sha(selected.D/f'components_{ds}_reporting_receipt.json')for ds in ['hc2','vidstg']})
 p=dict(schema='output_geometry_full_census_v1',scope=dict(hc2=1902,vidstg=10303),arms=['Source','FullMAP','FinalIndependent'],definitions='Finite normalized cxcywh only. Empty=width<=0 or height<=0. Partial out-of-image=any rectangle coordinate outside[0,1], strict with no epsilon. Fully outside=positive-area box but zero image intersection. Separate all-native and own-predicted-event support. All are geometric proxies, not identity switch/occlusion/correspondence truth. No GT file, label scoring, fallback, clipping or inference.',frozen_current_outputs=True,code_sha256={str(Path(__file__).relative_to(R)):sha(Path(__file__))},input_sha256=inputs);p['fingerprint']=digest(p);atomic(D/'failure_geometry_plan_v1.json',p);atomic(P/'failure_geometry_protocol_v1.json',p)

def derive(ds):
 import numpy as np
 p=read(D/'failure_geometry_plan_v1.json');assert p['fingerprint']==digest({k:v for k,v in p.items()if k!='fingerprint'})
 for field in ['code_sha256','input_sha256']:
  for f,h in p[field].items():assert sha(R/f)==h,f
 gt_guard();start=time.perf_counter();seal=read(selected.D/f'components_{ds}_sealed.json');out=[];arms=p['arms'];counts={a:{s:dict(frames=0,queries=0,nonfinite_frames=0,empty_frames=0,partial_outside_frames=0,fully_outside_frames=0,queries_with_partial_outside=0,queries_with_fully_outside=0,queries_with_empty=0)for s in ['all_native','own_event']}for a in arms}
 for i,(key,e)in enumerate(seal['files'].items(),1):
  guard(start);assert sha(selected.D/e['path'])==e['sha256']and sha(selected.D/e['receipt_path'])==e['receipt_sha256']
  with np.load(selected.D/e['path'])as z:
   ids=z['frame_ids'];pos=z['positions8'];obs=np.zeros(len(ids),bool);obs[pos]=True
   for arm in arms:
    b=z[arm];iv=z[arm+'_interval'];assert b.shape==(len(ids),4);finite=np.isfinite(b).all(1);empty=(b[:,2]<=0)|(b[:,3]<=0);lo=b[:,:2]-b[:,2:]/2;hi=b[:,:2]+b[:,2:]/2;partial=finite&((lo<0).any(1)|(hi>1).any(1));intersection=np.maximum(0,np.minimum(hi,1)-np.maximum(lo,0));full=finite&~empty&(intersection.prod(1)==0)
    for scope,use in [('all_native',np.ones(len(ids),bool)),('own_event',(ids>=iv[0])&(ids<=iv[1]))]:
     c=counts[arm][scope];c['frames']+=int(use.sum());c['queries']+=1
     for name,mask in [('nonfinite',~finite),('empty',finite&empty),('partial_outside',partial),('fully_outside',full)]:c[name+'_frames']+=int((use&mask).sum())
     for name,mask in [('partial_outside',partial),('fully_outside',full),('empty',finite&empty)]:c['queries_with_'+name]+=int((use&mask).any())
     out.append(dict(query_hash=hashlib.sha256((ds+':'+key).encode()).hexdigest(),arm=arm,support=scope,frames=int(use.sum()),observed_frames=int((use&obs).sum()),partial_outside_frames=int((use&partial).sum()),fully_outside_frames=int((use&full).sum()),empty_frames=int((use&finite&empty).sum()),nonfinite_frames=int((use&~finite).sum()),unobserved_partial_outside_frames=int((use&~obs&partial).sum()),unobserved_fully_outside_frames=int((use&~obs&full).sum())))
  if i%250==0:atomic(D/ds/'failure_geometry_progress.json',dict(completed=i,total=len(seal['files']),worker=identity(),GT_read=False))
 return out,counts,time.perf_counter()-start

def run(ds):
 rr,summary,seconds=derive(ds);out=P/'failure_geometry'/ds;out.mkdir(parents=True,exist_ok=True)
 with(out/'all_query_geometry_counts.tsv').open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rr[0]),delimiter='\t');w.writeheader();w.writerows(rr)
 atomic(D/ds/'failure_geometry_private_rows_v1.json',rr);atomic(out/'geometry_counts_v1.json',dict(queries=len(rr)//6,counts=summary,scope='Whole-target exact output-geometry census. Strict normalized bounds; partial outside does not imply whole prediction lost. Own intervals differ, denominators reported. No identity/occlusion labels, Source fallback or method change.'))
 atomic(D/ds/'failure_geometry_derive_v1.json',dict(queries=len(rr)//6,rows=len(rr),summary=summary,TSV_sha256=sha(out/'all_query_geometry_counts.tsv'),worker=identity(),elapsed_wall_seconds=seconds,new_GT_scoring=0,new_model=0))

def audit(ds):
 rr,summary,seconds=derive(ds);old=read(D/ds/'failure_geometry_private_rows_v1.json');rec=read(D/ds/'failure_geometry_derive_v1.json');assert rr==old and summary==rec['summary'];assert sha(P/'failure_geometry'/ds/'all_query_geometry_counts.tsv')==rec['TSV_sha256'];atomic(P/'failure_geometry'/ds/'reporting_receipt_v1.json',dict(status='verified_all_full_target_Source_Full_Independent_geometry_counts_prediction_receipt_SHA_no_GT',queries=rec['queries'],all_rows_counts_rederived_exact=True,derive_sha256=sha(D/ds/'failure_geometry_derive_v1.json'),worker=identity(),audit_wall_seconds=seconds,new_model=0,new_GT_scoring=0))
 text=f'# {ds}: exact full-target output geometry census\n\n{rec["queries"]} complete previously audited query outputs; no new model, ground-truth file or official score. Values are geometric failures/proxies, not tracked identity-switch, occlusion or expert-correspondence labels. A box crossing an image edge is distinct from a positive-area box wholly outside the image; both denominators are given. Original out-of-image predictions are never silently clipped.\n\n|Arm|Support|Frames|Queries with partial outside|Fully outside frames|Queries with fully outside|Empty frames|\n|---|---|---:|---:|---:|---:|---:|\n'
 for arm,x in summary.items():
  for support,c in x.items():text+=f'|{arm}|{support}|{c["frames"]}|{c["queries_with_partial_outside"]}|{c["fully_outside_frames"]}|{c["queries_with_fully_outside"]}|{c["empty_frames"]}|\n'
 text+='\nAll-query anonymized counts include observed/unobserved support and all native versus each arm\'s own predicted event. Source/Full/Independent time intervals can differ; comparisons of own-event fractions therefore are descriptive, not a fixed-support causal attribution. To claim true identity-switch/occlusion prevalence requires independent suitable labels; no such rates are fabricated from low IoU or trajectory jumps. Existing complete signed severe-loss/rescue and actual negative images are retained. No fallback or threshold was selected on scores.\n';(P/'failure_geometry'/ds/'ROOT_FINDINGS_v1.md').write_text(text)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','audit']);p.add_argument('--dataset',choices=['hc2','vidstg']);a=p.parse_args();globals()[a.command](*([]if a.command=='prepare'else[a.dataset]))

"""Separate post-seal official/independent metrics and exact parent-paired deltas; no bootstrap."""
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 import torch
 from scripts import review_stvg_sparse_r1_v2 as scorer
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 phase=sys.argv[1];cohort='confirmation'if phase=='confirmation'else'development';b=BASE/phase;seal=read(b/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete'and seal['count']==32 and not seal['GT_read']and seal['scorer_sha256']==sha(__file__)
 assert seal['design_sha256']==sha(BASE/'DESIGN_LOCK.json')
 assert read(b/'ROOT_TEMPORAL_READBACK.json')['status']=='pass'
 if phase!='round1':assert read(b/'ROOT_SPATIAL_READBACK.json')['status']=='pass'and read(b/'ROOT_INPUT_READBACK.json')['status']=='pass'
 for p in seal['files']:assert sha(ROOT/p['path'])==p['sha256']
 cases={};out={}
 for rec in roster(cohort):
  c,p=load_case(rec);cid=rec['probe_alias'];d=torch.load(b/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);assert d['source_case_sha256']==c['source_case_sha256']and not d['GT_read']and d['source_restored'];cases[cid]=c;out[cid]=d
 assert len(cases)==32
 write(b/'GT_EXPOSURE.json',dict(status='after_all_predictions_sealed',seal_sha256=sha(b/'GLOBAL_PREDICTION_BARRIER.json'),GT_read=True,time=time.time()))
 truth,provenance=scorer._truths_after_seal(cases);official,independent,_=scorer._metric_functions();rows=[]
 for cid,d in out.items():
  if 'reference_records'in d:support={a['position']for r in d['reference_records'].values()for a in r['anchors']}
  else:support={a['position']for a in d['reference_record']['original']['anchors']}
  for name,pred in d['arms'].items():
   m,audit=metric._score_student_arm(pred,cases[cid],truth[cid],support,official,independent);rows.append(dict(case_id=cid,dataset=d['dataset'],arm=name,metrics=m,support=audit['support'],metric_audit=audit['metric_audit']))
 summary=[];paired=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];ids=sorted({r['case_id']for r in rr});lookup={(r['case_id'],r['arm']):r for r in rr};names=sorted({r['arm']for r in rr});assert len(ids)==16 and len({out[c]['parent_sha256']for c in ids})==16
  for name in names:
   for key in rr[0]['metrics']:
    vals=[lookup[c,name]['metrics'][key]for c in ids if lookup[c,name]['metrics'][key]is not None];summary.append(dict(dataset=ds,arm=name,metric=key,count=len(vals),mean=math.fsum(vals)/len(vals)if vals else None))
  pairs=[(a,'F')for a in names if a!='F']
  if phase=='round1':pairs +=[(a,b)for a,b in [('T_stable','T_projection'),('T_offset_trust','T_projection'),('TS_stable','TS_projection'),('TS_offset_trust','TS_projection'),('T_stable','T_original'),('T_offset_trust','T_original')]]
  else:
   pairs +=[(a,'T')for a in names if a.startswith('TS_')]+[(a,'TS_old_crop')for a in names if a.startswith('TS_')and a!='TS_old_crop']
  for left,right in pairs:
   if left not in names or right not in names:continue
   d={c:(lookup[c,left]['metrics']['m_vIoU']-lookup[c,right]['metrics']['m_vIoU'])*100 for c in ids};vv=list(d.values());paired.append(dict(dataset=ds,left=left,right=right,metric='m_vIoU',mean_delta_pp=math.fsum(vv)/16,positive_sources=sum(x>1e-10 for x in vv),negative_sources=sum(x< -1e-10 for x in vv),neutral_sources=sum(abs(x)<=1e-10 for x in vv),worst_delta_pp=min(vv),best_delta_pp=max(vv),per_source_delta_pp=d))
 write(b/'POSTSEAL_ROWS.json',dict(status='complete',rows=rows,GT_read=True));write(b/'POSTSEAL_SUMMARY.json',dict(status='complete',summary=summary,GT_read=True,development_selection=cohort=='development'));write(b/'PARENT_PAIRED_DELTAS.json',dict(status='complete',records=paired,GT_read=True,bootstrap_executed=False))
 for x in summary:
  if x['metric']=='m_vIoU':print(x,flush=True)
if __name__=='__main__':main()

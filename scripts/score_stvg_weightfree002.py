"""Only after complete predictions: matched official metrics and paired dev deltas."""
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_weightfree_spatial_v1/consensus002'
def main():
 import torch,numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 from scripts import review_stvg_sparse_r1_v2 as scorer
 seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete' and not seal['GT_read'] and seal['scorer_sha256']==sha(__file__)
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 roster=read(ROOT/'artifacts/stvg_native_instance_spatial_opd_v1/ROSTER_LOCK.json')['records'];cases={};outputs={}
 for r in roster:
  cid=r['case_id'];d=torch.load(BASE/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);assert not d['GT_read'] and d['source_restored'];case=load_source_case(ROOT/r['source_case_path']);assert list(case['frame_ids'])==d['frame_ids'];cases[cid]=case;outputs[cid]=d
 assert len(cases)==8
 write(BASE/'GT_EXPOSURE.json',dict(status='after_all_predictions_sealed',GT_read=True,time=time.time(),seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')))
 truth,provenance=scorer._truths_after_seal(cases);official,independent,contract=scorer._metric_functions();rows=[]
 for cid,d in outputs.items():
  for name,a in d['arms'].items():
   ref=d['reference_record'].get(name.removeprefix('T_'),{});support={x['position']for x in ref.get('anchors',[])}
   m,audit=metric._score_student_arm(a,cases[cid],truth[cid],support,official,independent)
   rows.append(dict(case_id=cid,dataset=d['dataset'],arm=name,metrics=m,support=audit['support'],metric_audit=audit['metric_audit']))
 summaries=[];contrasts=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];lookup={(r['case_id'],r['arm']):r for r in rr}
  for name in outputs[next(c for c in cases if outputs[c]['dataset']==ds)]['arms']:
   subset=[r for r in rr if r['arm']==name]
   for key in subset[0]['metrics']:
    vals=[r['metrics'][key]for r in subset if r['metrics'][key]is not None];summaries.append(dict(dataset=ds,arm=name,metric=key,mean=math.fsum(vals)/len(vals)if vals else None,count=len(vals)))
  for l,r in [(name,base) for name in ['T_consensus8','T_nested8','T_consensus_nested8'] for base in ['T','T_validated_loss_weighted_iou']]:
   for key in rr[0]['metrics']:
    vals=[lookup[c,l]['metrics'][key]-lookup[c,r]['metrics'][key]for c in cases if (c,l)in lookup and lookup[c,l]['metrics'][key]is not None and lookup[c,r]['metrics'][key]is not None]
    contrasts.append(dict(dataset=ds,left=l,right=r,metric=key,mean_delta_pp=100*math.fsum(vals)/len(vals)if vals else None,count=len(vals),positive=sum(v>1e-12 for v in vals),negative=sum(v< -1e-12 for v in vals),neutral=sum(abs(v)<=1e-12 for v in vals)))
 write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',GT_read=True,rows=rows));write(BASE/'POSTSEAL_SUMMARY.json',dict(status='complete',GT_read=True,summary=summaries,contrasts=contrasts,historically_exposed_development=True))
 for x in summaries:
  if x['metric']=='m_vIoU':print(x)
if __name__=='__main__':main()

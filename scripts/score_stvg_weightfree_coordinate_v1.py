"""Only after complete predictions: matched official metrics and paired dev deltas."""
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1'/sys.argv[1]
def main():
 import torch,numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 from scripts import review_stvg_sparse_r1_v2 as scorer
 seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert seal['status']=='complete' and not seal['GT_read'] and seal['scorer_sha256']==sha(__file__)
 for f in seal['files']:assert sha(ROOT/f['path'])==f['sha256']
 from scripts.run_stvg_weightfree_expand_v1 import roster,load_case
 cases={};outputs={}
 for r in roster():
  cid=r['probe_alias'];d=torch.load(BASE/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);assert not d['GT_read'] and d['source_restored'];case,_=load_case(r);assert list(case['frame_ids'])==d['frame_ids'];cases[cid]=case;outputs[cid]=d
 assert len(cases)==32
 write(BASE/'GT_EXPOSURE.json',dict(status='after_all_predictions_sealed',GT_read=True,time=time.time(),seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')))
 truth,provenance=scorer._truths_after_seal(cases);official,independent,contract=scorer._metric_functions();rows=[]
 for cid,d in outputs.items():
  for name,a in d['arms'].items():
   # All six arms use the same prelocked reference support for supervised/unsupervised attribution.
   assert d['reference_record']['original']['anchors']==d['reference_record']['selected']['anchors']
   support={x['position']for x in d['reference_record']['original']['anchors']}
   m,audit=metric._score_student_arm(a,cases[cid],truth[cid],support,official,independent)
   rows.append(dict(case_id=cid,dataset=d['dataset'],arm=name,metrics=m,support=audit['support'],metric_audit=audit['metric_audit']))
 summaries=[];contrasts=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];lookup={(r['case_id'],r['arm']):r for r in rr}
  for name in outputs[next(c for c in cases if outputs[c]['dataset']==ds)]['arms']:
   subset=[r for r in rr if r['arm']==name]
   for key in subset[0]['metrics']:
    vals=[r['metrics'][key]for r in subset if r['metrics'][key]is not None];summaries.append(dict(dataset=ds,arm=name,metric=key,mean=math.fsum(vals)/len(vals)if vals else None,count=len(vals)))
 write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',GT_read=True,rows=rows));write(BASE/'POSTSEAL_SUMMARY.json',dict(status='complete',GT_read=True,summary=summaries,contrasts=contrasts,historically_exposed_project_sources=True,development_selection=True,prior_confirmation_reclassified=True))
 for x in summaries:
  if x['metric']=='m_vIoU':print(x)
if __name__=='__main__':main()

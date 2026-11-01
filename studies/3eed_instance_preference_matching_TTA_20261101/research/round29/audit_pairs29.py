import sys,numpy as np,collections,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round29'),str(ROOT/'research/round5/audit')]
from common29 import *
from independent_core_v2 import candidate_ious,transform_gt

def main():
 out=BASE/'gate154_attempt001';out.mkdir(exist_ok=False)
 allrows=rows()[:208];pred=[];qq=[]
 for r in allrows:
  z=feature(r);pairs=old(r,'pairs')['pairs'];raw=read(ROOT/'crossplatform/round22/pair208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['pairs'];assert len(pairs)==len(raw)
  for pair,p in zip(pairs,raw):
   assert pair['qA_PRIVATE']==int(z['scores'].argmax());pred.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'pair_alias':pair['pair_alias'],'qA_PRIVATE':pair['qA_PRIVATE'],'qB_PRIVATE':pair['qB_PRIVATE'],'mask_indices_PRIVATE':pair['mask_indices_PRIVATE'],'policies':policies(p),'metrics':metrics(p)})
 dump(out/'PREDICTIONS_PRIVATE.json',pred);seal(out,'R29_cached_pair_policy_predictions_before_current_task_audit',n_pairs=len(pred),new_model_calls=0)
 idx={x['id']:x for x in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};lookup={r['id']:r for r in allrows};containers={}
 for x in pred:
  r=lookup[x['id_PRIVATE']];i=idx[r['id']];path=i['meta_path']
  if path not in containers:assert sha(path)==i['metadata_container_sha256'];containers[path]=read(path)
  gt=transform_gt(containers[path]['ground_info'][i['object_index']]['bbox_3d'],r['platform']);z=feature(r);b=z['boxes'][[x['qA_PRIVATE'],x['qB_PRIVATE']]].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,gt)[0];x['iou_A'],x['iou_B']=map(float,v);x['delta_B_pp']=float((v[1]-v[0])*100)
 dump(EXP/'PAIR154_ROWS_PRIVATE.json',pred);summary={}
 for pl in ['all','drone','quad']:
  rr=[x for x in pred if pl=='all' or x['platform']==pl];summary[pl]={}
  for arm in pred[0]['policies']:
   pp=[x for x in rr if x['policies'][arm]=='B'];summary[pl][arm]={'A':sum(x['policies'][arm]=='A' for x in rr),'B':len(pp),'U':sum(x['policies'][arm]=='U' for x in rr),'B_task_better':sum(x['delta_B_pp']>1e-8 for x in pp),'B_task_worse':sum(x['delta_B_pp']< -1e-8 for x in pp),'B_task_tie':sum(abs(x['delta_B_pp'])<=1e-8 for x in pp),'sum_B_delta_pp':sum(x['delta_B_pp'] for x in pp),'rejected_vs_raw_B_better':sum(x['policies']['raw_context_AB']=='B' and x['policies'][arm]!='B' and x['delta_B_pp']>1e-8 for x in rr),'rejected_vs_raw_B_worse':sum(x['policies']['raw_context_AB']=='B' and x['policies'][arm]!='B' and x['delta_B_pp']< -1e-8 for x in rr)}
 dump(EXP/'GATE154_SUMMARY.json',summary);print(__import__('json').dumps(summary,indent=2))
if __name__=='__main__':main()

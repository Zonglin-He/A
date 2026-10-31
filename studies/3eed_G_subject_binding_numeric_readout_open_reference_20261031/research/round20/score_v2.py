"""Sealed direct-use G evaluation; research TTA gate never feeds G decisions."""
from pathlib import Path
import sys,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round20'),str(ROOT/'research/round5/audit')]
from common_v2 import *
from independent_core_v2 import transform_gt,candidate_ious
def summarize(rows,seed):
 output={}
 for phase in ('D','W','Z'):
  output[phase]={}
  for platform in ('drone','quad'):
   rr=[r for r in rows if r['phase']==phase and r['platform']==platform];arms={};scenes=sorted({r['scene_PRIVATE'] for r in rr})
   for arm in rr[0]['ious']:
    v=np.array([r['ious'][arm] for r in rr]);s=np.array([r['ious']['source'] for r in rr]);d=v-s;by={scene:np.array([x for r,x in zip(rr,d) if r['scene_PRIVATE']==scene]) for scene in scenes};rng=np.random.default_rng(seed);boot=[]
    for _ in range(10000):boot.append(float(np.concatenate([by[scenes[j]] for j in rng.integers(0,len(scenes),len(scenes))]).mean()*100))
    arms[arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_pp':float(d.mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'ties':int((abs(d)<=1e-10).sum()),'changed':sum(r['changed'][arm] for r in rr),'scene_clusters':len(scenes),'equal_scene_gain_pp':float(np.mean([x.mean() for x in by.values()])*100),'scene_cluster_bootstrap95_pp':list(map(float,np.quantile(boot,[.025,.975]))),'Acc25_rescues':int(((s<=.25)&(v>.25)).sum()),'Acc25_harms':int(((s>.25)&(v<=.25)).sum()),'Acc50_rescues':int(((s<=.5)&(v>.5)).sum()),'Acc50_harms':int(((s>.5)&(v<=.5)).sum())}
   output[phase][platform]=arms
 return output
def main():
 locked('SCORE_v2_LOCK.json');pred=ROOT/'crossplatform/round20/direct208_attempt002';verify(pred);verify(ROOT/'crossplatform/round20/dino144_attempt001');verify(ROOT/'crossplatform/round20/visual144_attempt001');out=EXP/'scored208_attempt002';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'prediction_seal_sha256':sha(pred/'SEALED.json'),'program_sha256':sha(__file__),'all208predictions_sealed_before_current_labels':True,'no_GT_into_G_or_per_case_acceptance':True})
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};rows=[]
 for r in read(EXP.parent/'input/ALL208_PRIVATE.json'):
  with np.load(pred/'predictions_PRIVATE'/(r['id']+'.npz')) as z:names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();qs=z['query_indices'].copy()
  ix=index[r['id']];p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
  if p not in containers:containers[p]=read(p)
  gt=transform_gt(containers[p]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6));values=candidate_ious(boxes,gt)[0];case=read(pred/'cases_PRIVATE'/(r['id']+'.json'));rows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'stage':r['stage'],'phase':r['stage'] if r['stage'] in ('W','Z') else 'D','platform':r['platform'],'scene_PRIVATE':r['scene'],'ious':dict(zip(names,map(float,values))),'changed':dict(zip(names,map(bool,qs!=qs[0]))),'G_intervenes':case['fullG_intervenes'],'diagnostic_pair_count':len(case['pair_diagnostics'])})
 spec=read(EXP/'G_SPEC_LOCK.json');summary=summarize(rows,spec['scene_bootstrap']['seed']);gate={}
 for platform in ('drone','quad'):
  d=summary['D'][platform]['direct_fullG'];w=summary['Z'][platform]['direct_fullG'];base=summary['Z'][platform]['source'];checks={'positive_dev_gain':d['gain_pp']>0,'positive_Z_gain':w['gain_pp']>0,'Z_cluster_lower_positive':w['scene_cluster_bootstrap95_pp'][0]>0,'Z_Acc25_not_lower':w['Acc25_percent']>=base['Acc25_percent'],'Z_Acc50_not_lower':w['Acc50_percent']>=base['Acc50_percent'],'Z_has_rescue':w['gains']>0};gate[platform]={'eligible_for_finite_TTA_pilot':all(checks.values()),'checks':checks,'not_a_label_free_deployment_gate':True,'few_scene_scope':True}
 dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',summary);dump(out/'TTA_PROGRESSION_GATE.json',gate);seal(out,'R20_direct_G_scored_after_sealed_predictions',n=208,all_negatives_retained=True);print('R20_DIRECT_SCORED',summary,gate)
if __name__=='__main__':main()

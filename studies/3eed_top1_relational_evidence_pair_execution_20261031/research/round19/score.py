"""CPU-only task/identity-proxy audit after every prediction artifact seals."""
from pathlib import Path
import sys,collections,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round19'),str(ROOT/'research/round5/audit')]
from common import *
from features_v2 import feature
from independent_core_v2 import transform_gt,candidate_ious
def identity_proxy(box,targets,target):
 # Geometry matching is explicitly not independent semantic identity gold.
 bb=np.array(box,np.float32)[None];bb[:,3:]=np.maximum(bb[:,3:],1e-6);ious=np.array([candidate_ious(bb,t)[0][0] for t in targets]);ix=int(ious.argmax());best=float(ious[ix]);same=np.array_equal(np.array(targets[ix]),np.array(target));target_iou=float(candidate_ious(bb,target)[0][0]);return {'status':'target_geometry_match' if best>=.25 and same else 'other_GT_geometry_match' if best>=.25 else 'unresolved_low_overlap','best_overlap':best,'target_overlap':target_iou,'semantic_identity_gold':False}
def main():
 locked('SCORE_LOCK.json');closed=ROOT/'crossplatform/round19/closed112_attempt002';verify(closed);verify(ROOT/'crossplatform/round19/G112_attempt004');verify(ROOT/'crossplatform/round19/pair_fit112_attempt002');verify(ROOT/'crossplatform/round19/historical_pair_interface_attempt001');out=EXP/'scored112_attempt001';out.mkdir(exist_ok=False);dump(out/'PRELABEL_LOCK.json',{'closed_prediction_seal_sha256':sha(closed/'SEALED.json'),'G_seal_sha256':sha(ROOT/'crossplatform/round19/G112_attempt004/SEALED.json'),'scorer_sha256':sha(__file__),'all112_predictions_sealed_before_current_scoring':True})
 index={r['id']:r for r in read(ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json')};containers={};rows=[];pair_rows=[]
 fittrace={r['id_PRIVATE']:r for r in read(ROOT/'crossplatform/round19/pair_fit112_attempt002/TRACE_PRIVATE.json')}
 for r in read(EXP.parent/'input/ALL112_v2_PRIVATE.json'):
  with np.load(closed/'predictions_PRIVATE'/(r['id']+'.npz')) as z:names=z['arm_names'].tolist();boxes=z['arm_boxes_raw6'].copy();queries=z['query_indices'].copy()
  ix=index[r['id']];p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
  if p not in containers:containers[p]=read(p)
  objects=containers[p]['ground_info'];gt=transform_gt(objects[ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));targets=[]
  for obj in objects:
   b=transform_gt(obj['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'))
   if not any(np.array_equal(b,x) for x in targets):targets.append(b)
  raw=boxes.copy();boxes[:,3:]=np.maximum(boxes[:,3:],np.float32(1e-6));values=candidate_ious(boxes,gt)[0];g=read(ROOT/'crossplatform/round19/G112_attempt004/cases_PRIVATE'/(r['id']+'.json'));pref=g['preference'];accepted=pref['status']=='usable';diag=fittrace[r['id']]['diagnostics'];probes=g['probes'];responses={}
  shadow=[]
  for pair in g['pairs']:
   qb=pair['pair']['qB_PRIVATE'];qa=pair['pair']['qA_PRIVATE'];zz=feature(r);pb=zz['boxes'][[qa,qb]].copy();pb[:,3:]=np.maximum(pb[:,3:],1e-6);pv=candidate_ious(pb,gt)[0];pd=pair['preference'];record={'alias':r['alias'],'pair_alias':pair['pair']['pair_alias'],'stage':r['stage'],'platform':r['platform'],'accepted':pd['status']=='usable','hypothetical_competitor_minus_q0_pp':float(100*(pv[1]-pv[0])),'A_IoU':float(pv[0]),'B_IoU':float(pv[1]),'discriminating_relations':pd['distinct_discriminating_relations'],'B_contradicted':pd['B_contradicted'],'appearance_B':pd['appearance_PRIVATE'].get(str(pair['pair']['mask_indices_PRIVATE'][1]),'U'),'identity_A_proxy':identity_proxy(pb[0],targets,gt),'identity_B_proxy':identity_proxy(pb[1],targets,gt)};shadow.append(record);pair_rows.append(record)
  chosen_shadow=next((x for x in shadow if x['accepted']),shadow[0] if shadow else None)
  for key in ('attribute_desaturate','reference_occlude'):
   if key in probes and key+'_area_control' in probes:
    orig=probes['original']['conditional_four_label_probabilities']['B'];e=probes[key]['conditional_four_label_probabilities']['B'];c=probes[key+'_area_control']['conditional_four_label_probabilities']['B'];responses[key]={'pB_original':orig,'pB_evidence_destroyed':e,'pB_area_control':c,'selective_pB_drop':float(c-e),'area_matched':probes[key+'_area_control']['area_matched'],'used_as_gate':False,'shadow_candidate_delta_pp':chosen_shadow['hypothetical_competitor_minus_q0_pp'] if chosen_shadow else None,'shadow_not_a_selected_prediction':True}
  rows.append({'alias':r['alias'],'id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'stage':r['stage'],'accepted':accepted,'reason':pref['reason'],'ious':dict(zip(names,map(float,values))),'query_changed':dict(zip(names,map(bool,queries!=queries[0]))),'box_changed':dict(zip(names,map(bool,[not np.array_equal(b,raw[0]) for b in raw]))),'steps':diag['steps'],'pair_margin_satisfied':diag.get('after',{}).get('pair_margin_satisfied',False),'B_above_A':diag.get('after',{}).get('B_above_A',False),'B_global_first_cached':diag.get('after',{}).get('B_global_first',False),'unverified_C_top1':diag.get('unverified_C_top1',False),'identity_proxy':{n:identity_proxy(b,targets,gt) for n,b in zip(names,raw)} if accepted else {},'evidence_responses':responses})
 summary={};feedback={}
 for stage in ('A','H','V'):
  summary[stage]={};feedback[stage]={}
  for platform in ('drone','quad'):
   rr=[r for r in rows if r['stage']==stage and r['platform']==platform];aa=[r for r in rr if r['accepted']];cells={}
   for arm in rows[0]['ious']:
    vals=np.array([r['ious'][arm] for r in rr]);source=np.array([r['ious']['source'] for r in rr]);diff=vals-source;sc=collections.defaultdict(list)
    for r,v in zip(rr,diff):sc[r['scene_PRIVATE']].append(float(v))
    cells[arm]={'n':len(rr),'mIoU_percent':float(vals.mean()*100),'Acc25_percent':float((vals>.25).mean()*100),'Acc50_percent':float((vals>.5).mean()*100),'gain_pp':float(diff.mean()*100),'gains':int((diff>1e-10).sum()),'harms':int((diff< -1e-10).sum()),'ties':int((abs(diff)<=1e-10).sum()),'query_changed':sum(r['query_changed'][arm] for r in rr),'scenes':len(sc),'equal_scene_gain_pp':float(np.mean([np.mean(v) for v in sc.values()])*100)}
   summary[stage][platform]=cells;dd=np.array([r['ious']['direct_pair_no_update']-r['ious']['source'] for r in aa]);feedback[stage][platform]={'n':len(rr),'accepted':len(aa),'direct_beneficial':int((dd>1e-10).sum()),'direct_harmful':int((dd< -1e-10).sum()),'direct_tie':int((abs(dd)<=1e-10).sum()),'direct_intervened_mean_gain_pp':float(dd.mean()*100) if len(dd) else None,'pair_margin_satisfied':sum(r['pair_margin_satisfied'] for r in aa),'B_above_A':sum(r['B_above_A'] for r in aa),'B_global_first_cached':sum(r['B_global_first_cached'] for r in aa),'unverified_C_top1':sum(r['unverified_C_top1'] for r in aa),'actual_steps':sum(r['steps'] for r in aa),'Acc25_rescues':sum(r['ious']['source']<=.25<r['ious']['direct_pair_no_update'] for r in aa),'Acc25_harms':sum(r['ious']['direct_pair_no_update']<=.25<r['ious']['source'] for r in aa)}
 dump(out/'ROWS_PRIVATE.json',rows);dump(out/'PAIR_DIAGNOSTIC.json',pair_rows);dump(out/'SUMMARY.json',summary);dump(out/'FEEDBACK_EXECUTION.json',feedback);seal(out,'R19_scored_only_after_all_predictions_sealed_all_negatives_kept',n=112);print('R19_SCORED',feedback)
if __name__=='__main__':main()

"""Separate CPU scoring. A labels gate supervision; B labels only after seal.
Candidate IoU preference is task-utility evidence, not physical identity gold.
All A/B rows and signed tails retained; no per-query independent-sample claim.
"""
from pathlib import Path
import sys,argparse,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round16'),str(ROOT/'research/round5/audit')]
from common import EXP,sha,read,dump,verify,seal,check_plan
from independent_core_v2 import transform_gt,candidate_ious
INDEX=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json'
def ious_for(row,z,index,containers):
 ix=index[row['id']];assert ix['platform']==row['platform'] and ix['scene']==row['scene']
 p=ix['meta_path'];assert sha(p)==ix['metadata_container_sha256']
 if p not in containers:containers[p]=read(p)
 raw=containers[p]['ground_info'][ix['object_index']]['bbox_3d'];target=transform_gt(raw,row['platform'],ix.get('waymo_view_suffix'))
 pool=np.array(z,copy=True);pool[:,3:]=np.maximum(pool[:,3:],np.float32(1e-6));return candidate_ious(pool,target)[0]
def summary(rows,arms):
 result={}
 for p in ('drone','quad'):
  pp=[r for r in rows if r['platform']==p];s={}
  for a in arms:
   v=np.array([r['ious'][a] for r in pp]);base=np.array([r['ious']['source'] for r in pp]);diff=v-base
   scenes=collections.defaultdict(list)
   for r,d in zip(pp,diff):scenes[r['scene_PRIVATE']].append(float(d))
   s[a]={'n':len(pp),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_pp':float(diff.mean()*100),'gains':int((diff>1e-10).sum()),'harms':int((diff< -1e-10).sum()),'ties':int((np.abs(diff)<=1e-10).sum()),'query_changes':sum(r['queries'][a]!=r['queries']['source'] for r in pp),'scene_clusters':len(scenes),'equal_scene_gain_pp':float(np.mean([np.mean(x) for x in scenes.values()])*100),'signed_delta_all':diff.tolist()}
  result[p]=s
 return result
def main(stage):
 check_plan();spec=read(EXP/'FIT_SPEC_LOCK.json')
 for rel,h in spec['programs'].items():assert sha(ROOT/rel)==h
 froot=ROOT/'crossplatform/round16/features64_attempt001';verify(froot);groot=ROOT/'crossplatform/round16/G_A32_attempt004';verify(groot)
 predroot=ROOT/'crossplatform/round16/B_predictions_attempt001'
 if stage=='B':
  verify(predroot);closure=read(EXP/'B_CLOSURE.json');assert closure['prediction_seal_sha256']==sha(predroot/'SEALED.json') and closure['passed']
 out=EXP/(stage+'_scoring_attempt001');out.mkdir(exist_ok=False)
 dump(out/'PRELABEL_LOCK.json',{'stage':stage,'features_seal_sha256':sha(froot/'SEALED.json'),'G_seal_sha256':sha(groot/'SEALED.json'),'fit_spec_sha256':sha(EXP/'FIT_SPEC_LOCK.json'),'B_prediction_seal_sha256':sha(predroot/'SEALED.json') if stage=='B' else None,'program_sha256':sha(__file__),'index_sha256':sha(INDEX),'label_boundary_below':True})
 index={r['id']:r for r in read(INDEX)};containers={};manifest=read(EXP.parent/'input'/(stage+'32_PRIVATE.json'));rows=[]
 for i,r in enumerate(manifest):
  with np.load(froot/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:boxes=z['boxes'].copy();scores=z['scores'].copy()
  values=ious_for(r,boxes,index,containers);q0=int(scores.argmax())
  if stage=='A':
   pref=read(groot/'cases_PRIVATE'/(r['id']+'.json'))['preference'];qg=pref['direct_query'];names=['source','direct_G'];qs=[q0,qg];details={'preference_status':pref['status'],'reason':pref['reason'],'relation_status':pref['relation_status'],'positive_n':len(pref['positive_queries']),'negative_n':len(pref['negative_queries'])}
   if pref['status']=='usable':
    pos=values[pref['positive_queries']];neg=values[pref['negative_queries']];delta=pos[:,None]-neg[None,:];w=int((delta>1e-10).sum());l=int((delta< -1e-10).sum());details.update(pair_wins=w,pair_losses=l,pair_ties=int((np.abs(delta)<=1e-10).sum()),strict_pair_win_fraction=w/(w+l) if w+l else None,mean_positive_IoU=float(pos.mean()),mean_negative_IoU=float(neg.mean()),representative_pair_delta=float(values[pref['pair_positive_query']]-values[pref['pair_negative_query']]))
  else:
   with np.load(predroot/'predictions_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:
    names=z['arm_names'].tolist();qs=z['query_indices'].tolist();assert np.array_equal(z['source_pool'],boxes) and np.array_equal(z['boxes'],boxes[qs])
   details={}
  rows.append({'alias':f'{stage}_{r["platform"]}_{i:03d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'queries':dict(zip(names,map(int,qs))),'ious':{a:float(values[q]) for a,q in zip(names,qs)},'oracle_IoU':float(values.max()),'details':details})
 sums=summary(rows,names);dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',sums)
 if stage=='A':
  gates={}
  for platform in ('drone','quad'):
   pp=[r for r in rows if r['platform']==platform];eligible=[r for r in pp if r['details']['preference_status']=='usable'];ratios=[r['details']['strict_pair_win_fraction'] for r in eligible if r['details']['strict_pair_win_fraction'] is not None]
   s=sums[platform]['direct_G'];pd=[r['details']['representative_pair_delta'] for r in eligible];criteria={'enough_usable_A':len(eligible)>=4,'strict_pairs_favor_positive':bool(ratios and np.mean(ratios)>.5),'positive_representative_pair_gain':bool(pd and np.mean(pd)>0),'local_direct_not_harmful':s['gain_pp']>=-1e-10 and s['gains']>=s['harms']}
   gates[platform]={'passed':all(criteria.values()),'criteria':criteria,'usable_A':len(eligible),'strict_preference_evaluable_A':len(ratios),'mean_sample_pair_win_fraction':float(np.mean(ratios)) if ratios else None,'direct_all_A_gain_pp':s['gain_pp'],'note':'Predeclared exposed-A development gate; no GT supplies training preference or hyperparameters. Not identity correctness or independent confirmation.'}
  dump(out/'GATE.json',{'G_seal_sha256':sha(groot/'SEALED.json'),'fit_spec_sha256':sha(EXP/'FIT_SPEC_LOCK.json'),'platforms':gates,'B_task_labels_read':False})
 seal(out,'R16_'+stage+'_CPU_scored_after_complete_prediction_seal',n=len(rows),all_negatives_retained=True,stage=stage)
 print(stage,sums,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('A','B'),required=True);main(p.parse_args().stage)

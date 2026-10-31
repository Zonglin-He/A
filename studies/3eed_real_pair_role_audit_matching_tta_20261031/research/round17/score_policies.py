"""Sealed CPU utility of fixed role policies; no individual GT pseudo labels."""
from pathlib import Path
import sys,argparse,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round17'),str(ROOT/'research/round5/audit')]
from common import EXP,read,dump,sha,verify,seal,check_plan
from independent_core_v2 import transform_gt,candidate_ious
INDEX=ROOT/'crossplatform/round5/baseline_engineering/SCORING_INDEX_PRIVATE.json'
POLICIES=('strict_roles','winner_role','pair_only')
def utility(pref,values):
 out={'status':pref['status'],'reason':pref['reason'],'positive_n':len(pref['positive_queries']),'negative_n':len(pref['negative_queries'])}
 if pref['status']=='usable':
  pos=values[pref['positive_queries']];neg=values[pref['negative_queries']];delta=pos[:,None]-neg[None,:];wins=int((delta>1e-10).sum());losses=int((delta< -1e-10).sum());out.update(pair_wins=wins,pair_losses=losses,pair_ties=int((np.abs(delta)<=1e-10).sum()),strict_pair_win_fraction=wins/(wins+losses) if wins+losses else None,representative_pair_delta=float(values[pref['pair_positive_query']]-values[pref['pair_negative_query']]))
 return out
def summarize(rows):
 result={}
 for platform in ('drone','quad'):
  rr=[r for r in rows if r['platform']==platform];result[platform]={}
  for arm in rr[0]['ious']:
   v=np.array([r['ious'][arm] for r in rr]);base=np.array([r['ious']['source'] for r in rr]);d=v-base;sc=collections.defaultdict(list)
   for r,delta in zip(rr,d):sc[r['scene_PRIVATE']].append(float(delta))
   result[platform][arm]={'n':len(rr),'mIoU_percent':float(v.mean()*100),'Acc25_percent':float((v>.25).mean()*100),'Acc50_percent':float((v>.5).mean()*100),'gain_pp':float(d.mean()*100),'gains':int((d>1e-10).sum()),'harms':int((d< -1e-10).sum()),'ties':int((np.abs(d)<=1e-10).sum()),'query_changes':sum(r['queries'][arm]!=r['queries']['source'] for r in rr),'scene_clusters':len(sc),'equal_scene_gain_pp':float(np.mean([np.mean(x) for x in sc.values()])*100),'signed_delta_all':d.tolist()}
 return result
def main(stage):
 check_plan();spec=read(EXP/'FIT_SPEC_LOCK.json')
 for rel,h in spec['programs'].items():assert sha(ROOT/rel)==h
 features=ROOT/'crossplatform/round17/features96_attempt001';verify(features);groot=ROOT/'crossplatform/round17/G_variants64_attempt001';verify(groot);original=ROOT/'crossplatform/round17/G64_attempt001';verify(original);preds=ROOT/'crossplatform/round17/B_predictions_attempt001'
 if stage=='B':
  verify(preds);cl=read(EXP/'B_CLOSURE.json');assert cl['passed'] and cl['prediction_seal_sha256']==sha(preds/'SEALED.json')
 out=EXP/(stage+'_scoring_attempt001');out.mkdir(exist_ok=False)
 dump(out/'PRELABEL_LOCK.json',{'stage':stage,'features_seal_sha256':sha(features/'SEALED.json'),'G_variants_seal_sha256':sha(groot/'SEALED.json'),'fit_spec_sha256':sha(EXP/'FIT_SPEC_LOCK.json'),'B_prediction_seal_sha256':sha(preds/'SEALED.json') if stage=='B' else None,'scorer_sha256':sha(__file__),'index_sha256':sha(INDEX)})
 # Labels only below: all G calls, policy variants and activation rules fixed.
 index={r['id']:r for r in read(INDEX)};containers={};rows=[];pair_audits=[];inputs=read(EXP.parent/'input'/('A64_PRIVATE.json' if stage=='A' else 'B32_PRIVATE.json'))
 for i,r in enumerate(inputs):
  with np.load(features/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:boxes=z['boxes'].copy();scores=z['scores'].copy()
  ix=index[r['id']];assert ix['platform']==r['platform'] and ix['scene']==r['scene'];mp=ix['meta_path'];assert sha(mp)==ix['metadata_container_sha256']
  if mp not in containers:containers[mp]=read(mp)
  target=transform_gt(containers[mp]['ground_info'][ix['object_index']]['bbox_3d'],r['platform'],ix.get('waymo_view_suffix'));b=boxes.copy();b[:,3:]=np.maximum(b[:,3:],np.float32(1e-6));values=candidate_ious(b,target)[0];q0=int(scores.argmax());details={}
  if stage=='A':
   x=read(groot/'cases_PRIVATE'/(r['id']+'.json'));names=['source']+list(POLICIES);qs=[q0]+[x['variants'][p]['direct_query'] for p in POLICIES];details={p:utility(x['variants'][p],values) for p in POLICIES};raw=read(original/'cases_PRIVATE'/(r['id']+'.json'))
   for rec in raw['pairs']:
    base=rec['variants']['main'];pa={'alias':x['alias'],'platform':r['platform'],'subset':x['subset'],'pair_alias':rec['pair']['pair_alias'],'pair_consensus':base['target_consensus'],'role_original_accepted':base['role_accepted'],'mapping_main':rec['pair']['main_mapping_both'],'mapping_old':rec['pair']['old_mapping_both'],'mapping_relaxed':rec['pair']['relaxed_mapping_both'],'main_reason':base['reason'],'task_pair_delta_main':None,'task_pair_delta_relaxed':None}
    if base['target_consensus']:
     ids=rec['pair']['mask_indices'];win=base['positive_mask'];lose=base['negative_mask']
     for key,name in [('query_groups','main'),('relaxed_query_groups','relaxed')]:
      pos=rec['pair'][key][ids.index(win)];neg=rec['pair'][key][ids.index(lose)]
      if pos and neg:
       qp=max(pos,key=lambda q:(float(scores[q]),-q));qn=max(neg,key=lambda q:(float(scores[q]),-q));pa['task_pair_delta_'+name]=float(values[qp]-values[qn])
    pair_audits.append(pa)
   subset=x['subset']
  else:
   with np.load(preds/'predictions_PRIVATE'/(r['id']+'.npz'),allow_pickle=False) as z:names=z['arm_names'].tolist();qs=z['query_indices'].tolist();assert np.array_equal(z['source_pool'],boxes) and np.array_equal(z['boxes'],boxes[qs])
   subset='new_R17_B32'
  rows.append({'alias':f'{stage}_{r["platform"]}_{i:03d}','id_PRIVATE':r['id'],'platform':r['platform'],'scene_PRIVATE':r['scene'],'subset':subset,'queries':dict(zip(names,map(int,qs))),'ious':{a:float(values[q]) for a,q in zip(names,qs)},'oracle_IoU':float(values.max()),'details':details})
 sums=summarize(rows);dump(out/'ROWS_PRIVATE.json',rows);dump(out/'SUMMARY.json',sums)
 if stage=='A':
  policies={};platforms={}
  for platform in ('drone','quad'):
   pp=[r for r in rows if r['platform']==platform];policies[platform]={}
   for policy in POLICIES:
    elig=[r for r in pp if r['details'][policy]['status']=='usable'];ratios=[r['details'][policy]['strict_pair_win_fraction'] for r in elig if r['details'][policy]['strict_pair_win_fraction'] is not None];pd=[r['details'][policy]['representative_pair_delta'] for r in elig];s=sums[platform][policy];criteria={'enough_usable_A':len(elig)>=4,'enough_comparable_A':len(ratios)>=3,'pair_direction_useful':bool(ratios and np.mean(ratios)>.6),'positive_representative_pair_delta':bool(pd and np.mean(pd)>0),'local_direct_not_harmful':s['gain_pp']>=-1e-10 and s['gains']>=s['harms']}
    policies[platform][policy]={'passed':all(criteria.values()),'criteria':criteria,'usable_A':len(elig),'comparable_A':len(ratios),'mean_sample_pair_win_fraction':float(np.mean(ratios)) if ratios else None,'mean_representative_delta_pp':float(np.mean(pd)*100) if pd else None,'direct_A_gain_pp':s['gain_pp']}
   chosen=next((p for p in POLICIES if policies[platform][p]['passed']),None);platforms[platform]={'passed':chosen is not None,'selected_policy':chosen}
  dump(out/'GATE.json',{'platforms':platforms,'policies':policies,'G_variants_seal_sha256':sha(groot/'SEALED.json'),'fit_spec_sha256':sha(EXP/'FIT_SPEC_LOCK.json'),'B_task_labels_read':False,'deployment_gate_claim':False,'selection':'First passing strict_roles>winner_role>pair_only; global exposed target-development selection, no per-sample GT targets.'})
  dump(out/'PAIR_GATE_AUDIT.json',{'scope':'Post-seal descriptive task utility, not physical identity gold. Pairs share observations/scenes.','rows':pair_audits});dump(out/'SUBSET_SUMMARIES.json',{s:summarize([r for r in rows if r['subset']==s]) for s in sorted({r['subset'] for r in rows})})
 seal(out,'R17_'+stage+'_CPU_scored_after_complete_prediction_seal',n=len(rows),all_negatives_retained=True,stage=stage);print('SCORED',stage,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=('A','B'),required=True);main(p.parse_args().stage)

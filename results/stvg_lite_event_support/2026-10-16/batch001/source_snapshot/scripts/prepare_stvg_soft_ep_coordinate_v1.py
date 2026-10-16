"""Select 128 independent video parents without scores; preserve old dev exposure."""
from pathlib import Path
import sys,copy,collections,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def main():
 assert not (BASE/'ROSTER_LOCK.json').exists();BASE.mkdir(parents=True,exist_ok=True)
 records=[]
 for reg,group in [('DEVELOPMENT_ROSTER_LOCK.json','previous_development'),('CONFIRMATION_ROSTER_LOCK.json','former_confirmation_now_development')]:
  for row in read(OLD/reg)['records']:
   r=copy.deepcopy(row);cid=r['probe_alias'];r.setdefault('source_case_path','artifacts/stvg_final_table1_v1/source_cases/'+cid+'.pt')
   parent=OLD/('round2_prompt009'if group=='previous_development'else'confirmation')
   r.update(expansion_group=group,cached_parent=str(parent.relative_to(ROOT)),cached_temporal_path=str((parent/'temporal'/(cid+'.pt')).relative_to(ROOT)))
   records.append(r)
 used={(r['dataset'],r['parent_sha256'])for r in records};assert len(records)==len(used)==64
 official=ROOT/'artifacts/stvg_final_table1_v1/FULL/FULL_SOURCE_ROSTER_LOCK.json';pool=read(official)['records']
 for ds,prefix in [('hc2','EH'),('vidstg','EV')]:
  per={}
  for r in pool:
   k=(ds,r['parent_sha256'])
   if r['dataset']!=ds or k in used:continue
   previous=per.get(r['parent_sha256'])
   if previous is None or r['query_key_sha256']<previous['query_key_sha256']:per[r['parent_sha256']]=r
  choices=sorted(per.values(),key=lambda r:r['parent_sha256'])[:32];assert len(choices)==32
  for i,row in enumerate(choices,1):
   r=copy.deepcopy(row);r['full_roster_probe_alias']=r['probe_alias'];r['probe_alias']=f'{prefix}{i:03d}'
   r.update(expansion_group='additional_development_historical_global_exposure_not_excluded',source_case_path=str((BASE/'source_cases'/(r['probe_alias']+'.pt')).relative_to(ROOT)))
   records.append(r);used.add((ds,r['parent_sha256']))
 records.sort(key=lambda r:(r['dataset'],r['expansion_group'],r['probe_alias']))
 assert len(records)==len(used)==128 and collections.Counter(r['dataset']for r in records)==dict(hc2=64,vidstg=64)
 write(BASE/'ROSTER_LOCK.json',dict(status='locked',count=128,unique_parent_count=128,domain_counts=dict(hc2=64,vidstg=64),records=records,source_roster_sha256=sha(official),selection='one query per unique parent; 64 previously measured sources plus 64 parent-hash ordered additional sources; no GT/score selection',fresh_confirmation_claim=False,globally_unseen_claim=False,former_confirmation_used_for_selection=True,GT_read=False))
 settings=read(ROOT/'artifacts/stvg_soft_ep_opd_v1/DESIGN_LOCK.json')['response_settings']
 write(BASE/'SEARCH_DESIGN.json',dict(status='locked',scope='expanded development coordinate search, not Table1',count=128,baseline_mu=.25,base_response_settings=settings,coordinate_order=['mu','lr','sigma','tau','steps','response_ratio'],grids=dict(mu=[.0625,.25,1.],lr_multiplier=[.25,.5,1.,2.,4.],sigma_multiplier=[.5,1.,2.],tau_multiplier=[.5,1.,2.],steps=[3,5,8],response_ratio=[.5,.75,1.]),selection='per-domain paired full-tube development mean; include all 64 sources including empty references; freeze winner of measured coordinate before next; one midpoint log refinement around interior winner at most; save all attempts; no GT online/teacher selection',max_refinement_per_coordinate=1,raw_training_GT=False,score_only_after_each_complete_prediction_seal=True,bootstrap=False,CURRENT_promoted=False,full_evaluation_resume_authorized=False,independent_confirmation_pending=True,GT_read=False,time=time.time()))
 print('locked roster128, parents128, hc64/vid64; old confirmation now development')
if __name__=='__main__':main()

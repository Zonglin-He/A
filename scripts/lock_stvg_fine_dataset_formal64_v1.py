"""Identity-only prospective parent lock, excluding every listed method/parameter panel."""
from pathlib import Path
import json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 A=ROOT/'artifacts';D=A/'stvg_fine_dataset_formal64_v1';assert not D.exists();D.mkdir();pool=read(A/'stvg_dino_supported_source256_v1/PLAN_LOCK.json')['records'];excluded={'hc2':set(),'vidstg':set()};sources=[]
 for f in sorted(A.glob('stvg_*/PLAN_LOCK.json')):
  if any(x in f.parent.name for x in ['source256','expanded256','evidence_strata','spatial_controls256','query_consistency_diagnosis256','fine_dataset_formal64']):continue
  p=read(f);rows=[]
  for k in ['records','target','source']:
   z=p.get(k,[])
   if isinstance(z,list):rows+=z
  touched=0
  for r in rows:
   if isinstance(r,dict)and r.get('dataset')in excluded and 'parent'in r:excluded[r['dataset']].add(r['parent']);touched+=1
  if touched:sources.append(dict(path=str(f.relative_to(ROOT)),sha256=digest(f),rows=touched))
 prior=read(A/'stvg_fine_step_confirm32_v1/ROSTER_LOCK.json')['records'];assert all(r['parent']in excluded[r['dataset']]for r in prior)
 records=[];available={};seed='frozen-dataset-formal64-v1|20261019'
 for ds,prefix in [('hc2','FH'),('vidstg','FV')]:
  eligible=[r for r in pool if r['dataset']==ds and r['parent']not in excluded[ds]];available[ds]=len(eligible);eligible.sort(key=lambda r:hashlib.sha256((seed+'|'+ds+'|'+r['parent']).encode()).hexdigest());assert len(eligible)>=32
  for i,r in enumerate(eligible[:32]):
   prioralias=r['alias'];src=A/'stvg_dino_supported_source256_v1/source'/(prioralias+'.pt');pred=A/'stvg_dino_supported_source256_v1/predictions'/(prioralias+'.pt');assert src.is_file()and pred.is_file()
   records.append(dict(alias=prefix+f'{i+1:03}',dataset=ds,parent=r['parent'],query_ordinal=r['query_ordinal'],prior_alias=prioralias,source_path=str(src.relative_to(ROOT)),prediction_path=str(pred.relative_to(ROOT)),source_sha256=digest(src),prediction_sha256=digest(pred)))
 assert all(len({r['parent']for r in records if r['dataset']==ds})==32 for ds in excluded)
 out=dict(status='roster_locked_before_feature_or_GT',count=64,per_domain=32,records=records,seed=seed,eligible_counts=available,excluded_parent_counts={d:len(s)for d,s in excluded.items()},excluded_manifests=sources,excluded_CH_CV_previous_parameter_panel=True,selection='SHA256 identity-only; no score/GT/quality/cache-based replacement',historical_Source_E_expanded_evaluation_exposure=True,not_completely_unexposed=True,not_used_in_listed_Fine_method_parameter_selection_panels=True,new_Source_extractions_required=64,new_DINO_calls=0,new_training=0,created_epoch=time.time(),GPU_cap_seconds=900,arms=['E_MBR','Fixed','Live'],frozen_configuration='configs/stvg_fine_center_dataset_candidate_v1.json',frozen_configuration_sha256=digest(ROOT/'configs/stvg_fine_center_dataset_candidate_v1.json'))
 (D/'ROSTER_LOCK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['status','count','eligible_counts','excluded_parent_counts','new_Source_extractions_required']}))
if __name__=='__main__':main()

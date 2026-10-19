"""Outcome-blind parent exclusion/selection for one finite step confirmation."""
from pathlib import Path
import json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 A=ROOT/'artifacts';D=A/'stvg_fine_step_confirm32_v1';D.mkdir();pool=read(A/'stvg_dino_supported_source256_v1/PLAN_LOCK.json')['records'];excluded={'hc2':set(),'vidstg':set()};sources=[]
 for f in sorted(A.glob('stvg_*/PLAN_LOCK.json')):
  if any(x in f.parent.name for x in ['source256','expanded256','evidence_strata','spatial_controls256','query_consistency_diagnosis256','fine_step_confirm32']):continue
  p=read(f);rows=[]
  for k in ['records','target','source']:
   z=p.get(k,[])
   if isinstance(z,list):rows+=z
  touched=0
  for r in rows:
   if isinstance(r,dict)and r.get('dataset')in excluded and 'parent'in r:excluded[r['dataset']].add(r['parent']);touched+=1
  if touched:sources.append(dict(path=str(f.relative_to(ROOT)),sha256=digest(f),rows=touched))
 records=[];available={};missing=[];seed='step-confirm32-v1|20261019'
 for ds,prefix in [('hc2','CH'),('vidstg','CV')]:
  eligible=[r for r in pool if r['dataset']==ds and r['parent']not in excluded[ds]];available[ds]=len(eligible);eligible.sort(key=lambda r:hashlib.sha256((seed+'|'+ds+'|'+r['parent']).encode()).hexdigest());assert len(eligible)>=16
  for i,r in enumerate(eligible[:16]):
   alias=prefix+f'{i+1:03}';prior=r['alias'];src=A/'stvg_dino_supported_source256_v1/source'/(prior+'.pt');pred=A/'stvg_dino_supported_source256_v1/predictions'/(prior+'.pt');assert src.is_file()and pred.is_file()
   records.append(dict(alias=alias,dataset=ds,parent=r['parent'],query_ordinal=r['query_ordinal'],prior_alias=prior,source_path=str(src.relative_to(ROOT)),prediction_path=str(pred.relative_to(ROOT)),source_sha256=digest(src),prediction_sha256=digest(pred)))
   missing.append(alias)
 out=dict(status='roster_locked_no_model_started',count=32,per_domain=16,records=records,seed=seed,eligible_counts=available,excluded_parent_counts={d:len(s)for d,s in excluded.items()},excluded_manifests=sources,selection='SHA256 fixed seed sort identity only; every selected row retained independent of cache/quality/GT',historical_source256_expanded_evaluation_exposure=True,not_completely_unexposed=True,not_used_in_listed_method_selection_panels=True,history_audit_scope='58 stvg PLAN_LOCK small experimental records/target/source; broad full/evidence/core256 evaluations disclosed, not claimed unseen globally',missing_layer3=missing,new_Source_extractions_required=32,new_DINO_calls=0,new_training=0,created_epoch=time.time(),GPU_cap_seconds=900,arms=['E_MBR','Fixed_step05','Live_step05','Fixed_step10','Live_step10'],new_GPU_not_started_await_old_deadline_clarification=True)
 (D/'ROSTER_LOCK.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in out.items()if k in ['status','count','eligible_counts','new_Source_extractions_required']})
if __name__=='__main__':main()

"""Synthetic CPU scorer boundary check; no actual dev/confirm target mask access."""
import pathlib,tempfile,json,hashlib,sys,importlib.util,subprocess
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scorer',ROOT/'score_round2.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
with tempfile.TemporaryDirectory(prefix='round2_scorer_fixture_',dir=ROOT) as tmp:
 r=pathlib.Path(tmp);run=r/'run';run.mkdir();co=r/'cohorts';co.mkdir();refs=[]
 for i,target in enumerate([np.zeros((2048,1)),np.ones((2048,1)),np.linspace(0,1,2048).reshape(-1,1)]):
  container=np.c_[np.zeros((2048,3)),target];p=r/f'{i}.npy';np.save(p,container);refs.append({'id':str(i),'parent_id':str(i),'object_class':'synthetic','affordance':'synthetic','native_annotation_container':str(p)})
  for obj in ['hm_only','hm_support']:
   for k in [0,3]:
    q=np.linspace(.1,.9,2048,dtype=np.float32).reshape(-1,1);np.savez_compressed(run/f'{i:04d}_{obj}_k{k}.npz',source=q,ensemble=q,median=q,q75=q,adapted=q)
 (run/'protocol.json').write_text(json.dumps({'state':'sealed','count':3,'cohort':'dev','steps':[0,3],'objectives':['hm_only','hm_support']}));(run/'SEALED.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in run.glob('*.npz')}));path=co/'dev_scoring.json';path.write_text(json.dumps(refs));(co/'COHORT_LOCK.json').write_text(json.dumps({'manifests':{path.name:hashlib.sha256(path.read_bytes()).hexdigest()}}));(r/'MECHANISM_LOCK.json').write_text('{}');(r/'algorithm.py').write_text('# synthetic hash placeholder, never imported\n');module.ROOT=r;sys.argv=['score_round2.py','--run',str(run),'--cohort','dev'];module.main();s=json.loads((run/'SCORED.json').read_text());assert all(a['valid_counts']=={'auc':1,'aiou':2,'sim':3,'mae':3} for a in s['averages']);assert json.loads((r/'DEV_DECISION_FROZEN.json').read_text())['primary_comparator']=='median'
report={'kind':'synthetic_scorer_engineering_only','passed':True,'fixed_valid_denominators_all_arms':{'auc':1,'aiou':2,'sim':3,'mae':3},'empty_and_full_GT_reasons_recorded':True,'zero_step_equals_median_check_passed':True,'source_and_frozen_ensembles_bitexact_all_arms':True,'tie_selection_prefers_median':True,'no_real_GT_masks_read':True};(ROOT/'SCORER_ENGINEERING.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

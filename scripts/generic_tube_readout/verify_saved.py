"""Read-only independent saved-output checks; never executes a model or scoring."""
import os,json,hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 total=0;timechecks=0;err=0.;panels={}
 for name in ['interval_state_readout_v1','action_interval_readout_v1','rank_interval_readout_v1','generic_tube_readout_v1']:
  folder=ROOT/'outputs'/name;p=read(folder/'plan.json');seal=read(folder/'sealed.json');report=read(ROOT/'results'/name/'report.json');assert len(seal['files'])==128 and seal['GT_read'] is False and sha(folder/'plan.json')==seal['plan_sha256']
  for rel,h in p['sources'].items():assert sha(ROOT/rel)==h
  assert report['proof']['official_dense_max_error']<2e-10
  assert report['proof']['all128_new_predictions_before_new_GT_scores']
  for item,receipt,row in zip(p['items'],seal['files'],report['queries_anonymous']):
   assert item['anonymous']==receipt['query']==row['query'];assert sha(folder/receipt['path'])==receipt['sha256'];z=np.load(folder/receipt['path']);old=np.load(ROOT/'outputs/joint_boundary_readout_v2/predictions'/f"{item['anonymous']}.npz")
   for key in ['Source','Source_interval','Baseline','Baseline_interval','frame_ids','positions8','candidate_intervals','candidate_utilities']:assert np.array_equal(z[key],old[key]);total+=1
   assert row['changed']['PriorMatched']==row['changed']['RandomMatched']
   if name=='generic_tube_readout_v1':
    for arm in ['ExpectedTube','FixedTrackTube','MeanQuality','ShiftedQuality','PriorMatched','RandomMatched']:
     k=receipt['selected'][arm];assert np.array_equal(z[arm+'_interval'],z['candidate_intervals'][k]);timechecks+=1
    for control in ['PriorMatched','RandomMatched']:assert row['changed'][control]==row['changed']['ExpectedTube']
    assert np.isfinite(z['qualities']).all() and z['qualities'].min()>=0 and z['qualities'].max()<=1+1e-12
  for c,g in report['cohorts'].items():
   rows=[r for r in report['queries_anonymous']if r['cohort']==c]
   for arm,paired in g['subsets']['all']['paired'].items():
    for field in ['v','t']:
     actual=100*np.mean([r['values'][arm][field]-r['values']['Baseline'][field]for r in rows]);err=max(err,abs(actual-paired[field]['mean']));assert abs(actual-paired[field]['mean'])<1e-10
  panels[name]=dict(predictions=128,official_dense_checks=report['proof']['official_dense_checks'],not_adopted=not report['decision']['validated_candidate'])
 formal=read(ROOT/'outputs/boxfixed_full_v1/plan.json');assert len(formal['sources'])==434
 for name,h in formal['sources'].items():assert sha(ROOT/name)==h
 progress=read(ROOT/'outputs/boxfixed_full_v1/vidstg/progress.json');state=read(ROOT/'outputs/boxfixed_full_v1/state.json');identities=[]
 for x in [state['controller'],*state['children']]:
  pid=x['pid'];raw=Path(f'/proc/{pid}/stat').read_text();actual=int(raw.rsplit(')',1)[1].split()[19]);assert actual==x['startticks'];identities.append(dict(pid=pid,startticks=actual))
 assert state['jobs']==dict(hc2='scored',vidstg='running')
 result=dict(status='pass',panels=panels,unchanged_source_baseline_and_candidate_arrays=total,saved_interval_candidate_index_checks=timechecks,paired_aggregation_maxerror=err,no_new_labels_read=True,no_model_execution=True,frozen_formal_sources_unchanged=434,formal_plan_sha256=sha(ROOT/'outputs/boxfixed_full_v1/plan.json'),formal_identities_verified=identities,Vid_progress=progress['completed'],generic_constant_quality_temporal_utility_maxerror=read(ROOT/'outputs/generic_tube_readout_v1/sealed.json')['constant_quality_temporal_utility_max_error'])
 dest=ROOT/'results/generic_tube_readout_v1/verification.json';dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()

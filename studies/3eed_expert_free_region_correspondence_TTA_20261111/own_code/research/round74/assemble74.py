"""Reuse sealed pilot/remaining artifacts; one stronger direct region reader."""
from common74 import *
from direct_region74 import direct_region
import os,time
def main():
 p0=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');pb=read(H/'expanded/experiment/PREDICT_PLAN_LOCK_attempt002.json');allrows=read(ROOT/'research/round69/input/ALL2304_PRIVATE.json');pilot={r['id'] for r in rows()};out=ROOT/'crossplatform/round74/all2304';assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();roots={k:out/k for k in ['predictions','evidence','fits']}
 for root,folders in [(roots['predictions'],['samples_PRIVATE','aux_PRIVATE','cases_PRIVATE']),(roots['evidence'],['samples_PRIVATE','cases_PRIVATE']),(roots['fits'],['cases_PRIVATE'])]:
  for folder in folders:(root/folder).mkdir(parents=True,exist_ok=True)
 statuses=[];costrows=[];TASK_ARMS=None;cohort=read(ROOT/'research/round69/input/COHORT_BY_ID_PRIVATE.json');cache={}
 for pp in [p0,pb]:
  sr0=ROOT/pp['prediction_root'];ev0=ROOT/pp['evidence_root'];fr0=ROOT/pp['fit_root'];cache[pp['prediction_root']]=(read(sr0/'SEALED.json'),read(ev0/'SEALED.json'),read(fr0/'SEALED.json'),{x['id']:x['arm_statuses'] for x in read(sr0/'READOUT_STATUS_PRIVATE.json')})
 for i,r in enumerate(allrows):
  sid=r['id'];p=p0 if sid in pilot else pb;rel='samples_PRIVATE/'+sid+'.npz';sr=ROOT/p['prediction_root'];se,es,fs,stmap=cache[p['prediction_root']];assert sha(sr/rel)==se['files'][rel];ev=ROOT/p['evidence_root'];assert sha(ev/rel)==es['files'][rel];froot=ROOT/p['fit_root'];z=load(sr/rel);e=load(ev/rel);score,cr=direct_region(ab(r),e);names=list(z['arm_names']);idx=names.index('frozen_AB_repr')+1;newnames=names[:idx]+['frozen_region_consensus']+names[idx:];values=np.insert(z['scores'],idx,score,axis=0);q=values.argmax(1);np.savez_compressed(roots['predictions']/rel,arm_names=np.array(newnames),scores=values,queries=q,source_boxes=z['source_boxes'],selected_original_boxes=z['source_boxes'][q]);TASK_ARMS=newnames
  for folder in ['aux_PRIVATE','cases_PRIVATE']:
   sub=folder+'/'+sid+('.npz' if folder=='aux_PRIVATE' else '.json');assert sha(sr/sub)==se['files'][sub];os.link(sr/sub,roots['predictions']/sub)
  for folder in ['samples_PRIVATE','cases_PRIVATE']:
   sub=folder+'/'+sid+('.npz' if folder=='samples_PRIVATE' else '.json');assert sha(ev/sub)==es['files'][sub];os.link(ev/sub,roots['evidence']/sub)
  for kind in ADAPT:
   sub='cases_PRIVATE/'+sid+'_'+kind+'.json';assert sha(froot/sub)==fs['files'][sub];os.link(froot/sub,roots['fits']/sub)
  oldstatus=stmap[sid].copy();oldstatus['frozen_region_consensus']={'unknown':not bool(e['RGB_negatives'].any()),'fallback':cr['unique_region_query_updates']==0,'no_parameter_update':True};statuses.append({'id':sid,'arm_statuses':oldstatus});costrows.append({'row':'ALL_'+str(i+1).zfill(5),'cohort':cohort[sid],'phase':'PILOT64' if sid in pilot else 'REMAINDER2240','platform':r['platform'],**cr})
 dump(roots['predictions']/'READOUT_STATUS_PRIVATE.json',statuses)
 for k,root in roots.items():seal(root,'R74_LINKED_EXISTING_PILOT_AND_FIXED_EXPANSION_'+k.upper(),n=2304,arms=TASK_ARMS if k=='predictions' else None)
 dump(H/'ALL_DIRECT_REGION_COST_ROWS.json',costrows);plan={**p0,'n':2304,'evidence_root':roots['evidence'].relative_to(ROOT).as_posix(),'fit_root':roots['fits'].relative_to(ROOT).as_posix(),'prediction_root':roots['predictions'].relative_to(ROOT).as_posix(),'all_task_arms':TASK_ARMS,'pilot64_and_remaining2240_not_independent_datasets':True,'supplemental_direct_reader':'same_RGB_qualified_fixed_regions_AB_centroid_half_shift_unique_query_membership;no_gradient_no_C_no_tuning;registered_postpilot_pre_remaining_score'};dump(H/'experiment/ALL2304_PLAN_LOCK.json',plan);dump(H/'ASSEMBLY_RECEIPT.json',{'state':'PASS_R74_64_PLUS2240_UNION_NO_FIT_RERUN','n':2304,'pilot_n':64,'remaining_n':2240,'all_original12_readout_vectors_preserved':True,'extra_fixed_direct_reader_only':True,'seconds':time.perf_counter()-start,'prediction_seal_sha256':sha(roots['predictions']/'SEALED.json')});print(read(H/'ASSEMBLY_RECEIPT.json'))
if __name__=='__main__':main()

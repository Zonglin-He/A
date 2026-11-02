"""Separately granted oldDEV2 CPU engineering check of the generic frontend.

No full manifest, new detector/VLM/grounder, task score, label or closed FINAL.
The two old input-hash rows are fixed in the plan before executing this check.
"""
import sys,time
from pathlib import Path
import numpy as np
from full_common50 import ROOT,EXP,read,sha,dump,npz,seal
from full_cpu50 import observation_from_files,geometry_arrays

def main():
 import argparse
 p=argparse.ArgumentParser(description=__doc__)
 for k in ('plan','authorization'):p.add_argument('--'+k,required=True)
 a=p.parse_args();plan=read(a.plan);grant=read(a.authorization)
 assert grant['state']=='ROOT_AUTHORIZED_R50_OLDDEV2_GENERIC_FRONTEND_ENGINEERING' and grant['plan_sha256']==sha(a.plan)
 assert plan['n']==2 and plan['no_full_manifest_or_GT_or_new_GPU_models'] is True
 for group in ('programs_sha256','inputs_sha256'):
  for rel,h in plan[group].items():
   assert 'round46/final' not in rel and 'FINAL384' not in rel and 'full5912_gpu_inputs' not in rel
   assert sha(ROOT/rel)==h,rel
 old=read(ROOT/'research/round24/input/ALL336_PRIVATE.json')
 salt=plan['input_hash_selection_salt'];expected=[min([r for r in old if r['platform']==pl],key=lambda r:__import__('hashlib').sha256((salt+r['id']).encode()).hexdigest()) for pl in ('drone','quad')]
 rows=read(ROOT/plan['rows_path']);assert rows==expected
 out=ROOT/plan['output_root'];assert out.is_relative_to(ROOT/'crossplatform/round50') and not out.exists();out.mkdir(parents=True)
 sys.path[:0]=[str(ROOT/'crossplatform/round49'),str(ROOT/'crossplatform/round23'),str(ROOT/'crossplatform/round20')]
 from ownership49 import frame_route
 from parse23 import parse
 from textZ import pipeline
 from stages49 import relation_value
 from pool_weights50 import exact
 A={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
 AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
 prep=ROOT/'crossplatform/round50/prepared336_attempt001';ps=read(prep/'SEALED.json')
 bp=prep/'BINDINGS_PRIVATE.json';assert sha(bp)==ps['files'][bp.name];original_bindings=read(bp)
 sourcefolder=ROOT/'crossplatform/round24/capture336_attempt001';source_seal=read(sourcefolder/'SEALED.json')
 nlp=pipeline();results=[];start=time.perf_counter()
 for r in rows:
  sid=r['id'];fr,mp=frame_route(r,A,AF,RF);fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
  if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
  cp=out/(sid+'_camera_PRIVATE.json');dump(cp,fields)
  binding=dict(image_sha256=fr['image_sha256'],point_sha256=fr['point_sha256'],metadata_sha256=fr['metadata_sha256'],mask_cache_sha256=sha(mp),camera_cache_sha256=sha(cp))
  assert sha(mp)==original_bindings[mp.relative_to(ROOT).as_posix()]
  frame,scene,three=observation_from_files(r,mp,cp,binding,with_rgb=False)
  expected_casepath=prep/'cases_PRIVATE'/(sid+'.json');assert sha(expected_casepath)==ps['files'][expected_casepath.relative_to(prep).as_posix()]
  expected_case=read(expected_casepath);textpath=ROOT/expected_case['actual_text_source_PRIVATE'];assert sha(textpath)==original_bindings[textpath.relative_to(ROOT).as_posix()];g=read(textpath)
  freshg=parse(r,nlp);assert freshg==g,'Original text parse must reproduce before generic frontend adoption'
  sourcepath=sourcefolder/'samples_PRIVATE'/(sid+'.npz');assert sha(sourcepath)==source_seal['files'][sourcepath.relative_to(sourcefolder).as_posix()];source=npz(sourcepath)
  # Existing source bridge already proved R24/R8 equal. This engineering test
  # deliberately reads the old source only, never opens the full registration.
  source['R8_file_sha256']=sha(sourcepath)
  ar,c=geometry_arrays(r,source,frame,scene,three,g,textpath)
  ep=prep/'arrays_PRIVATE'/(sid+'.npz');assert sha(ep)==ps['files'][ep.relative_to(prep).as_posix()];target=npz(ep)
  keys=['source_boxes','source_scores','old_pool','wide_pool','old_weights','wide_old_weights','mask_sha256','detector_boxes','mask_counts','allvisible','unassigned','ambiguous','legacy_support','legacy_coverage','legacy_projection_iou','legacy_depth_trim_count','legacy_quality','legacy_valid','historical_taxonomy_compatible','corrected_taxonomy_compatible']
  for key in keys:exact(ar[key],target[key],key)
  suffix='M64' if r['stage']=='M' else '272';base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23')
  evidencepath=base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json');assert sha(evidencepath)==original_bindings[evidencepath.relative_to(ROOT).as_posix()];evidence=read(evidencepath)
  W,H=scene['image_size_wh'];pr=np.zeros((len(scene['objects']),2))
  for m,obj in enumerate(scene['objects']):
   if obj['mask_geometry_defined']:pr[m]=relation_value(g,obj,evidence,W,H)
  exact(pr,target['raw_mask_PR'],'raw_mask_PR')
  np.savez_compressed(out/(sid+'_arrays_PRIVATE.npz'),**ar)
  results.append({'id_PRIVATE':sid,'platform':r['platform'],'all_tested_arrays_bitexact':True,'original_parse_graph_equal':True,'fullfile_not_opened':True,'n_compared_arrays':len(keys)+1,'raw_RGB_read':False,'raw_XYZ_read':True,'no_GT_or_taskscore':True})
 dump(out/'RESULT.json',{'state':'PASS_R50_OLDDEV2_GENERIC_FRONTEND','n':2,'results_PRIVATE':results,'seconds':time.perf_counter()-start,'GPU_forwards':0,'optimizer_steps':0})
 seal(out,'R50_OLDDEV2_GENERIC_FRONTEND_ENGINEERING_SEALED',n=2,plan_sha256=sha(a.plan));print('R50_GENERIC_FRONTEND_ENGINEERING_PASS',flush=True)
if __name__=='__main__':main()

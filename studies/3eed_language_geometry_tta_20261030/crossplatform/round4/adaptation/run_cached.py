"""Parentlocked episodicCPU updates on sealedGTfree sourcecache only."""
from pathlib import Path
import argparse,importlib.util,json,hashlib,time
import numpy as np,torch
from interfaces import validate_cache_arrays
from caption_regions import parse_target_horizontal_region
from spatial_teacher import spatial_posterior
from query_update import cached_views,source_aug,probability,run_query,aabb_iou
from geometry_update import interval_teacher,direct_center,run_center,box_from_delta
R=Path(__file__).resolve().parent;P=R.parent;S=P/'source';C=S/'calibration/CORRECTION';MECHANISM=P/'MECHANISM_PLAN_LOCK.json'
ARMS=['frozen_raw_diagnosticonly','frozen_PCA_primarybaseline','TTAug_3fullviews_PCA_no_update','nounonly_3views_matchedref_PCA_no_update','spatial_teacher_direct_PCA_no_update','query0_control','query1','query3_primary','query_no_spatial3_control','geometry_direct_PCA_no_update','geometry1','geometry3_primary','geometry_no_observation3_control']
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
 if isinstance(x,list):return [plain(v) for v in x]
 return x

def main(cohort):
 torch.set_num_threads(4);lockpath=R/'CODE_LOCK.json';lock=json.loads(lockpath.read_text());assert sha(lockpath)==lockpath.with_suffix('.sha256').read_text().split()[0];assert sha(MECHANISM)==lock['mechanism_sha256']
 for f,h in lock['source_hashes'].items():assert sha(f)==h,f
 if cohort=='confirm':
  release=json.loads((R/'CONFIRM_RELEASE.json').read_text());assert release['code_lock_sha256']==sha(lockpath) and release['mechanism_sha256']==sha(MECHANISM),'parent finalrelease required'
 source=S/f'{cohort}_source';seal=json.loads((source/'sealed.json').read_text());assert seal['source_plan_sha256']==lock['source_plan_sha256'];mp=S/'inputs'/f'{cohort}_manifest.json';assert sha(mp)==seal['manifest_sha256'];samples=json.loads(mp.read_text());camera_seal=json.loads((C/'ADAPTER_INPUT_SEAL_v2.json').read_text());assert sha(C/'ADAPTER_INPUT_SEAL_v2.json')==lock['camera_input_seal_sha256'];camera_manifest=C/f'{cohort}_adapter_manifest.json';assert sha(camera_manifest)==camera_seal['manifests'][cohort]['sha256'];cameras={x['id']:x for x in json.loads(camera_manifest.read_text())};helper_path=C/'camera_helper.py';assert sha(helper_path)==camera_seal['helper_sha256'];spec=importlib.util.spec_from_file_location('round4_verified_camera_helper',helper_path);camhelper=importlib.util.module_from_spec(spec);spec.loader.exec_module(camhelper)
 out=R/f'{cohort}_predictions';out.mkdir(exist_ok=False);(out/'diagnostics').mkdir();traces=[]
 for sample in samples:
  start=time.perf_counter();sid=sample['id'];cache={}
  for v in range(3):
   for cond in ['full','noun']:
    name=f'{sid}_v{v}_{cond}.npz';assert sha(source/name)==seal['files'][name]
    with np.load(source/name,allow_pickle=False) as z:cache[v,cond]={k:z[k].copy() for k in z.files}
    validate_cache_arrays(cache[v,cond])
  full=[cache[v,'full'] for v in range(3)];canonical=full[0]['boxes_pca'];assert np.array_equal(canonical[:,:6],full[0]['boxes']) and canonical.shape==(256,7)
  source_query=int(full[0]['semantic_scores'].argmax());views,weights=cached_views(full);aug=source_aug(views,weights);prior=probability(aug).numpy();region,parse=parse_target_horizontal_region(sample['text']);camera=camhelper.load_camera(cameras[sid]);uv,depth,valid=camhelper.project_source_points(full[0]['boxes'][:,:3],camera);teacher,td=spatial_posterior(prior,uv,valid,camera['image_size_wh'],region,(full[0]['boxes'][:,3:6]>0).all(-1));informative=td['informative']
  query,qd=run_query(views,weights,teacher,informative);control,qc=run_query(views,weights,teacher,informative,control=True)
  noun_scores=[];noun_weights=[]
  for v in range(3):
   a=cache[v,'noun'];idx=a['aligned_to_full_view0'].astype(int);noun_scores.append(a['semantic_scores'][idx]);noun_weights.append(aabb_iou(full[0]['boxes'],a['boxes'][idx]))
  ns=np.stack(noun_scores);nw=np.stack(noun_weights);den=nw.sum(0);noun=np.where(den>0,(ns*nw).sum(0)/np.maximum(den,1e-30),full[0]['semantic_scores'])
  base=canonical[source_query].copy();raw=np.r_[full[0]['boxes'][source_query],np.float32(0)].astype('float32');selected={'frozen_raw_diagnosticonly':source_query,'frozen_PCA_primarybaseline':source_query,'TTAug_3fullviews_PCA_no_update':int(aug.argmax()),'nounonly_3views_matchedref_PCA_no_update':int(noun.argmax()),'spatial_teacher_direct_PCA_no_update':int(teacher.argmax()),'query0_control':int(query[0].argmax()),'query1':int(query[1].argmax()),'query3_primary':int(query[3].argmax()),'query_no_spatial3_control':int(control[3].argmax())};arrays={a:canonical[q].copy() for a,q in selected.items()};arrays['frozen_raw_diagnosticonly']=raw
  view0=sample['views'][0];pp=Path(view0['point_path']);assert sha(pp)==view0['point_sha256'];points=np.load(pp,allow_pickle=False);origin=np.linalg.inv(camera['source_to_camera'])[:3,3];lo,hi,observed,gd=interval_teacher(points,base,origin);pose,gtrace=run_center(base,lo,hi,observed);gcontrol,gct=run_center(base,lo,hi,observed,control=True);arrays.update({'geometry_direct_PCA_no_update':box_from_delta(base,direct_center(lo,hi,observed)),'geometry1':pose[1],'geometry3_primary':pose[3],'geometry_no_observation3_control':gcontrol[3]})
  arrays={a:arrays[a] for a in ARMS};assert np.array_equal(arrays['query0_control'],arrays['TTAug_3fullviews_PCA_no_update']);assert np.array_equal(arrays['query_no_spatial3_control'],arrays['TTAug_3fullviews_PCA_no_update']);assert np.array_equal(arrays['geometry_no_observation3_control'],base)
  for a,b in arrays.items():
   assert b.shape==(7,) and np.isfinite(b).all()
   if a.startswith('geometry'):assert np.array_equal(b[3:7],base[3:7]);selected[a]=source_query
   elif a!='frozen_raw_diagnosticonly':assert np.array_equal(b,canonical[selected[a]])
  np.savez_compressed(out/(sid+'.npz'),**arrays);np.savez_compressed(out/'diagnostics'/(sid+'_scores.npz'),source=full[0]['semantic_scores'],TTAug=aug.numpy(),noun_only=noun,teacher=teacher,query0=query[0],query1=query[1],query3=query[3],query_no_spatial3=control[3],spatial_penalty=td['penalty'],projected_uv=uv,projection_valid=valid,geometry_interval_lower=lo,geometry_interval_upper=hi,geometry_observed=observed)
  traces.append(plain({'id':sid,'platform':sample['platform'],'split':sample['split'],'scene':sample['scene'],'selected_queries':selected,'caption_region':region.__dict__ if region else None,'parser':parse,'spatial_teacher':td,'geometry_correspondence_IoU_weights':weights.numpy(),'query':qd,'query_no_spatial':qc,'geometry_teacher':gd,'geometry':gtrace,'geometry_control':gct,'seconds':time.perf_counter()-start,'noun_unmatched_reference_candidates':int((den==0).sum()),'canonical_pool_and_sizes_and_yaw_immutable':True}));(out/'progress.json').write_text(json.dumps(traces,indent=2));print(json.dumps({'id':sid,'region':region.relation if region else None,'spatial_teacher_informative':informative,'geometry_axes':int(observed.sum()),'seconds':traces[-1]['seconds']}),flush=True)
 (out/'trace.json').write_text(json.dumps(traces,indent=2));files={p.name:sha(p) for p in out.glob('*.npz')};assert len(files)==len(samples);result={'state':'sealed','n':len(samples),'arms':ARMS,'files':files,'trace_sha256':sha(out/'trace.json'),'diagnostic_files':{p.name:sha(p) for p in (out/'diagnostics').glob('*.npz')},'mechanism_sha256':sha(MECHANISM),'code_lock_sha256':sha(lockpath),'source_seal_sha256':sha(source/'sealed.json'),'camera_input_seal_sha256':sha(C/'ADAPTER_INPUT_SEAL_v2.json'),'numeric_GT_or_candidate_IoU_outcomes_read':False,'actual_small_parameter_updates':True,'source_model_parameters_or_cache_modified':False,'BBS_controls':'separatelysealedfrozenhead outputs supplied bysource/parent; not repeatedhere','scope':'DEVunlabeled updates beforeparentofflineGTscore' if cohort=='dev' else 'lockedconfirmation'};(out/'SEALED.json').write_text(json.dumps(result,indent=2));print('SEALED',len(samples),sha(out/'SEALED.json'),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['dev','confirm'],required=True);main(p.parse_args().cohort)

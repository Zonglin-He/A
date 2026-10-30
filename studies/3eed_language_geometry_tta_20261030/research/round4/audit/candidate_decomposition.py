"""Independent stage1 offline CPU diagnostics. GT-labelled outputs never feed adaptation.
Only reference full/view0 candidates; ALL192 GT-free source caches verified before targets.
"""
import pathlib,json,hashlib,math,sys,traceback
from collections import Counter,defaultdict
import numpy as np
R=pathlib.Path(__file__).resolve().parent;B=R.parents[2];C=B/'crossplatform';S=C/'round4/source';sys.path.insert(0,str(B/'tools'))
from obb_iou import oriented_iou
KS=[1,5,10,32,256];TH=[.25,.5]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def rot(t):
 c,s=math.cos(float(t)),math.sin(float(t));return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def metric(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float)
 assert a.shape==b.shape==(7,) and np.isfinite(a).all() and np.isfinite(b).all() and (b[3:6]>0).all()
 if (a[3:6]<=0).any():return 0.
 return float(oriented_iou(a[:3],a[3:6],rot(a[6]),b[:3],b[3:6],rot(b[6])))
def core(raw,pca,scores,gt):
 raw=np.asarray(raw);pca=np.asarray(pca);scores=np.asarray(scores);assert raw.shape==(256,6) and pca.shape==(256,7) and scores.shape==(256,)
 assert np.isfinite(raw).all() and np.isfinite(pca).all() and np.isfinite(scores).all() and np.array_equal(raw,pca[:,:6])
 order=np.lexsort((np.arange(256),-scores));q=int(order[0]);raw7=np.c_[raw,np.zeros(256)];result={};ious={}
 for name,boxes in [('rawyaw0',raw7),('fixedPCA',pca)]:
  values=np.array([metric(b,gt) for b in boxes]);ious[name]=values;g={'selected_query':q,'top1_iou':float(values[q]),'best256_iou_label_assisted':float(values.max()),'best_query_label_assisted':int(values.argmax()),'invalid_size_count':int(np.any(raw[:,3:6]<=0,axis=1).sum()),'recall':{},'decomposition':{}}
  for t in TH:
   key=str(t);g['recall'][key]={str(k):bool(np.max(values[order[:k]])>t) for k in KS};where=np.flatnonzero(values[order]>t);first=int(where[0]+1) if len(where) else None
   g['decomposition'][key]={'first_passing_rank':first,'selected_success':bool(values[q]>t),'rankrecoverable_top1fails_any256passes':bool(values[q]<=t and values.max()>t),'candidate_coverage_failure_all256fail':bool(values.max()<=t),'not_exact_instance_identity':True}
  result[name]=g
 base=pca[q].astype(float);inter={}
 for label,sl in [('center',slice(0,3)),('size',slice(3,6)),('yaw',slice(6,7))]:
  box=base.copy();box[sl]=gt[sl];v=metric(box,gt);inter[label]={'iou':v,'delta_fixedPCA':v-result['fixedPCA']['top1_iou'],'same_selected_query':q,'interpretation':'singleGTparameterinjection, otherfields fixed; conditional notcausal/identityproof, no PCA recrop'}
 error={'center_distance_m':float(np.linalg.norm(base[:3]-gt[:3])),'center_normalized_by_GTdiag':float(np.linalg.norm(base[:3]-gt[:3])/np.linalg.norm(gt[3:6])),'size_mean_abs_logratio':float(np.abs(np.log(base[3:6]/gt[3:6])).mean()) if (base[3:6]>0).all() else None,'yaw_abs_error_modPI':float(abs((base[6]-gt[6]+np.pi/2)%np.pi-np.pi/2)),'invalid_source_size':bool((base[3:6]<=0).any())}
 return result,inter,error,ious

def fixture():
 gt=np.array([0.,0.,0.,2.,2.,2.,0.]);raw=np.tile([100.,0.,0.,2.,2.,2.],(256,1));raw[0]=[0,0,0,1,1,2];raw[1]=[0,0,0,1,2,2];raw[6]=gt[:6];pca=np.c_[raw,np.zeros(256)];scores=1-np.arange(256)*.001
 g,_,_,_=core(raw,pca,scores,gt);assert not g['rawyaw0']['recall']['0.25']['1'] and g['rawyaw0']['recall']['0.25']['5'];assert not g['rawyaw0']['recall']['0.5']['5'] and g['rawyaw0']['recall']['0.5']['10'];assert g['fixedPCA']['decomposition']['0.5']['first_passing_rank']==7
 scores[1]=scores[0];assert np.lexsort((np.arange(256),-scores))[0]==0
 return {'synthetic_only':True,'strict_thresholds_and_stable_ties_passed':True,'rank7_not_top5_passed':True,'identity_not_inferred':True}

def aggregate(rows):
 out=[]
 for platform in ['waymo','drone','quad']:
  rr=[x for x in rows if x['platform']==platform];scenes=sorted({x['scene'] for x in rr});item={'platform':platform,'research_scope':'primary_Waymo_VAL' if platform=='waymo' else 'TRAIN_union_pretraining_exposed_auxiliary','n':len(rr),'scene_clusters':len(scenes),'geometries':{},'single_GT_interventions':{}}
  for geom in ['rawyaw0','fixedPCA']:
   g={'top1_mean_iou':float(np.mean([x['geometries'][geom]['top1_iou'] for x in rr])),'best256_mean_iou_label_assisted':float(np.mean([x['geometries'][geom]['best256_iou_label_assisted'] for x in rr])),'recall_counts':{},'rankrecoverable_counts':{},'coverage_failure_counts':{}}
   for t in TH:
    key=str(t);g['recall_counts'][key]={str(k):sum(x['geometries'][geom]['recall'][key][str(k)] for x in rr) for k in KS};g['rankrecoverable_counts'][key]=sum(x['geometries'][geom]['decomposition'][key]['rankrecoverable_top1fails_any256passes'] for x in rr);g['coverage_failure_counts'][key]=sum(x['geometries'][geom]['decomposition'][key]['candidate_coverage_failure_all256fail'] for x in rr)
   vals=np.array([np.mean([x['geometries'][geom]['top1_iou'] for x in rr if x['scene']==s]) for s in scenes]);g['top1_scene_macro_iou']=float(vals.mean());g['top1_scene_bootstrap95_descriptive']=np.quantile(np.random.default_rng(20261030).choice(vals,(10000,len(vals)),replace=True).mean(1),[.025,.975]).tolist() if len(scenes)>1 else None;item['geometries'][geom]=g
  for kind in ['center','size','yaw']:
   v=[x['interventions'][kind] for x in rr];item['single_GT_interventions'][kind]={'mean_iou':float(np.mean([x['iou'] for x in v])),'mean_delta_fixedPCA':float(np.mean([x['delta_fixedPCA'] for x in v])),'positive':sum(x['delta_fixedPCA']>1e-10 for x in v),'negative':sum(x['delta_fixedPCA']< -1e-10 for x in v),'same':sum(abs(x['delta_fixedPCA'])<=1e-10 for x in v)}
  item['annotated_object_list_sizes']=dict(Counter(x['annotated_object_count'] for x in rr));out.append(item)
 return out

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--run-dev-after-root-authorization',action='store_true');a=ap.parse_args();engineering=fixture();(R/'ENGINEERING.json').write_text(json.dumps(engineering,indent=2));assert a.run_dev_after_root_authorization,'Default performs fixtures only; real targets require explicit parent grant'
 dest=R/'dev_decomposition_v1';dest.mkdir(exist_ok=False);status={'state':'verifying_before_any_target_access','scope':'DEV ONLY; no confirmation GT read','source_seal_expected':'b8a1553ad0c9c9811176928f90a859d3eeb3a2d61f84ae89e2de5342beff7842'};(dest/'STATUS.json').write_text(json.dumps(status,indent=2))
 try:
  master=C/'round4/STAGE1_PLAN_LOCK.json';assert sha(master)=='6cd4bef3f1bae5556169d2fc7eab596d738cad1f935ccfb9c9c93c09cb21982c';source=S/'dev_source';seal=read(source/'sealed.json');assert sha(source/'sealed.json')==status['source_seal_expected'] and seal['n_samples']==32 and seal['n_forward_caches']==192 and len(seal['files'])==192
  for name,h in seal['files'].items():assert sha(source/name)==h
  assert sha(source/'repeat_control.json')==seal['repeat_control_sha256'];plan=read(S/'SOURCE_PLAN.json');assert sha(S/'SOURCE_PLAN.json')==seal['source_plan_sha256'];assert sha(S/'PATH_LOCK.json')==plan['path_lock_sha256']
  for name,h in plan['local_code_sha256'].items():assert sha(S/name)==h
  for name,h in plan['upstream_code_sha256'].items():assert sha(C/name)==h
  mp=S/'inputs/dev_manifest.json';assert sha(mp)==seal['manifest_sha256']==plan['input_manifest_sha256']['dev'];inputs=read(mp);receipt=read(S/'INPUT_RECEIPT.json');index=S/'dev_scoring_index_PRIVATE.json';assert sha(index)==receipt['cohorts']['dev']['scoring_index_sha256'];samples=read(index);assert len(samples)==32 and {s['id'] for s in samples}=={s['id'] for s in inputs}
  lock={'scorer_sha256':sha(pathlib.Path(__file__)),'independent_3Dconvex_sha256':sha(B/'tools/obb_iou.py'),'master_sha256':sha(master),'source_plan_sha256':sha(S/'SOURCE_PLAN.json'),'all192_source_and_canonicalPCA_files_verified_before_targets':True,'source_seal_sha256':sha(source/'sealed.json'),'dev_scoring_index_sha256':sha(index),'confirm_GT_authorized':False,'fixed_schema':'rawboxes256x6,scores256,PCA256x7,referencefullview0; stablequeryindex ties','GT_access_authorization':'Parent explicitly authorized DEV32 after source seal; no confirmation access'};(dest/'SCORER_LOCK.json').write_text(json.dumps(lock,indent=2));rows=[]
  for i,s in enumerate(samples):
   sid=s['id'];
   with np.load(source/(sid+'_v0_full.npz'),allow_pickle=False) as z:raw=z['boxes'].copy();pca=z['boxes_pca'].copy();scores=z['semantic_scores'].copy()
   meta=read(s['meta_path']);objects=meta['ground_info'];gt=np.array(objects[s['object_index']]['bbox_3d'][:7],float)
   if s['platform']=='drone':gt[2]+=1.8
   if s['platform']=='waymo':
    angle=math.radians([0,-45,45,-90,90][s['waymo_view_suffix']]);gt[:3]=rot(angle)@gt[:3];gt[6]+=angle
   g,inter,err,ious=core(raw,pca,scores,gt);row={'id':sid,'platform':s['platform'],'split':s['split'],'scene':s['scene'],'geometries':g,'interventions':inter,'parameter_error_conditioned_on_selected_query':err,'annotated_object_count':len(objects),'instance_identity':'unknown; ground_info records not proven exhaustive instance inventory and geometric recall doesnot prove identity'};rows.append(row);np.savez_compressed(dest/(sid+'_GT_IOU_PRIVATE.npz'),**ious);print('DEV_SCORED',i+1,s['platform'],sid,flush=True)
  result={'scope':'DEV32only firstdiagnostic; WaymoVAL primary, DroneQuadTRAIN unionexposed auxiliary; notsourceonly','n':32,'scorer_lock_sha256':sha(dest/'SCORER_LOCK.json'),'summaries':aggregate(rows),'rows':rows,'limits':['Recall/candidateoracle is geometry/ranking headroom, not confirmed realinstance identity','GTcenter/size/yaw injection keepsselectedq andotherfields, not causal attribution ordeployable solution','GTyaw not strictmaximum; noGTcrop orPCArecrop','No confirmation GT read','No TTAfeedback selected by this scorer']};(dest/'RESULTS_PRIVATE.json').write_text(json.dumps(result,indent=2,allow_nan=False));(dest/'SEALED.json').write_text(json.dumps({'n':32,'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='STATUS.json'}},indent=2));status.update(state='completed',measurement='valid_independent3Dconvex_DEVonly');(dest/'STATUS.json').write_text(json.dumps(status,indent=2));print(json.dumps(result['summaries'],indent=2))
 except Exception as e:status.update(state='failed_retained',error=repr(e),traceback=traceback.format_exc());(dest/'STATUS.json').write_text(json.dumps(status,indent=2));raise
if __name__=='__main__':main()

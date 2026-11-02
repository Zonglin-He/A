"""Independent no-GT R50 prepared-array replay; no author R50 algorithm import."""
from pathlib import Path
import sys,json,hashlib,time,collections,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;EXP=ROOT/'research/round50/experiment'
PREP=ROOT/'crossplatform/round50/prepared336_attempt001';STAGE=ROOT/'crossplatform/round49/stage336_attempt001';OWN=ROOT/'crossplatform/round49/ownership336_attempt001';OUT=HERE/'prepared_audit_attempt001'
sys.path[:0]=[str(ROOT/'research/round49/independent_audit'),str(ROOT/'crossplatform/round7')]
from reference49 import extract_original_relation,image_only_scene
from camera_projection import CameraSpec,project_model_points
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
SEALS={}
def checked(p,folder):
 if folder not in SEALS:SEALS[folder]=read(folder/'SEALED.json')
 assert sha(p)==SEALS[folder]['files'][p.relative_to(folder).as_posix()]
 return load(p) if p.suffix=='.npz' else read(p)
def exact(a,b,name):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
def iou(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);inter=np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod();union=np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter
 return float(inter/union) if union>0 else 0.
def matrices(points,owners,depth,boxes,camera,wh,rects):
 M=len(rects);S=np.zeros((256,M),int);C=np.zeros((256,M));I=np.zeros((256,M));trim=[]
 ids=[]
 for m in range(M):
  idx=np.flatnonzero(owners==m)
  if len(idx)>=4:
   dd=depth[idx];lo,hi=np.quantile(dd,[.25,.75]);spread=max(hi-lo,.1);idx=idx[(dd>=lo-1.5*spread)&(dd<=hi+1.5*spread)]
  ids.append(idx);trim.append(len(idx))
 signs=np.asarray(list(itertools.product((-1,1),repeat=3)))
 for q,b in enumerate(boxes):
  if (b[3:]<=0).any():continue
  corners=np.asarray(b[:3],float)+signs*np.asarray(b[3:],float)/2
  pr=project_model_points(corners,camera,wh);valid=bool((pr['depth']>.1).all() and np.isfinite(pr['uv']).all())
  rect=np.r_[pr['uv'].min(0),pr['uv'].max(0)] if valid else None
  for m,idx in enumerate(ids):
   count=int(np.count_nonzero((np.abs(points[idx]-b[:3])<=b[3:]/2+.1).all(1)));S[q,m]=count;C[q,m]=count/max(len(idx),1)
   if valid:I[q,m]=iou(rect,rects[m])
 return S,C,I,np.asarray(trim,dtype=np.int64)
def oldw(pool,S,C,I,boxes):
 valid=(S>=3)&(C>=.15)&(I>=.1)&(boxes[:,3:]>0).all(1)[:,None];quality=np.where(valid,C*np.sqrt(np.maximum(I,0)),0.);w=quality[:,pool].copy() if len(pool) else np.zeros((256,0));w/=np.maximum(w.sum(1,keepdims=True),1e-12);return w,quality,valid
def oldread(s,w,pool,e,pr):
 v=np.zeros(256);p=np.zeros(256);r=np.zeros(256);active=w.sum(1)>0
 if len(pool):
  cue=w@e[pool];rr=w@pr[pool];q0=int(s.argmax());av=cue[q0] if active[q0] else float(np.median(e[pool]));ar=rr[q0] if active[q0] else np.median(pr[pool],axis=0)
  v[active]=np.clip((cue[active]-av)/2,-2,2);p[active]=np.clip(rr[active,0]-ar[0],-2,2);r[active]=np.clip(rr[active,1]-ar[1],-2,2)
 return {'vision':v,'position':p,'relation':r,'scores':s.astype(float)+2*v+2*p+2*r}
def classflags(category,labels):
 families=[({'person'},{0},{1}),({'bicycle','bike','cyclist','biker'},{1},{2}),({'motorcycle'},{3},{4}),({'bus'},{5},{6}),({'truck','trailer'},{7},{8}),({'car','vehicle','sedan','van','taxi','suv','jeep'},{2,7},{3,8})]
 for names,old,new in families:
  if category in names:return np.array([int(x) in old for x in labels]),np.array([int(x) in new for x in labels])
 return np.ones(len(labels),bool),np.ones(len(labels),bool)
def main():
 start=time.perf_counter();auth=read(EXP/'ROOT_INDEPENDENT_PREP_AUDIT_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R50_INDEPENDENT_DEV_PREP_AUDIT_NO_GT' and auth['program_sha256']==sha(__file__) and auth['prepared_seal_sha256']==sha(PREP/'SEALED.json') and auth['prep_plan_sha256']==sha(EXP/'PREP_PLAN_LOCK.json')
 for rel,h in auth['independent_reference_sha256'].items():assert sha(ROOT/rel)==h
 plan=read(EXP/'PREP_PLAN_LOCK.json')
 for rel,h in {**plan['programs_sha256'],**plan['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 seal=read(PREP/'SEALED.json');assert seal['n']==336 and seal['new_model_calls']==0
 for f,h in seal['files'].items():assert sha(PREP/f)==h
 assert read(ROOT/'research/round49/independent_audit/OWNERSHIP_AUDIT_RESULT.json')['state'].startswith('PASS_')
 assert not OUT.exists();OUT.mkdir(parents=True);rows=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');AC={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
 relfn=extract_original_relation((ROOT/'crossplatform/round23/rank23.py').read_text(),(ROOT/'crossplatform/round20/detect.py').read_text());resultrows=[];totals=collections.Counter()
 try:
  for i,row in enumerate(rows):
   rid=row['id'];z=checked(PREP/'arrays_PRIVATE'/(rid+'.npz'),PREP);case=checked(PREP/'cases_PRIVATE'/(rid+'.json'),PREP);old=checked(STAGE/'samples_PRIVATE'/(rid+'.npz'),STAGE);oc=checked(OWN/'cases_PRIVATE'/(rid+'.json'),OWN);od=checked(OWN/'samples_PRIVATE'/(rid+'.npz'),OWN);fr=checked(OWN/'frames_PRIVATE'/(oc['frame_key_PRIVATE']+'.npz'),OWN);sc=checked(STAGE/'cases_PRIVATE'/(rid+'.json'),STAGE)
   mp=ROOT/sc['mask_cache_path_PRIVATE'];masks=checked(mp,mp.parent);M=len(masks['class_ids']);pool=old['pool'];boxes=old['source_boxes'];s=old['source_scores']
   for a,b in [('source_boxes',boxes),('source_scores',s),('old_pool',pool),('mask_counts',od['mask_counts']),('allvisible',od['allvisible']),('unassigned',od['unassigned']),('ambiguous',od['ambiguous']),('mask_sha256',fr['mask_sha256']),('native_instance_id',fr['native_instance_id']),('native_detector_class',masks['class_ids']),('detector_boxes',masks['boxes_xyxy'])]:exact(z[a],b,a)
   if row['stage']=='A':camera_data=AF[AC[rid]['frame_alias']]
   else:
    matches=[r for r in RF.values() if all(r.get(k)==row[k] for k in ('imagepath','pointpath','metapath'))];assert len(matches)==1;camera_data=matches[0]
   fields={k:camera_data[k] for k in ('image_intrinsic','image_extrinsic')}
   if camera_data['camera_distortion'] is not None:fields['camera_distortion']=camera_data['camera_distortion']
   camera=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared');H,W=map(int,masks['original_hw'])
   S,C,I,trim=matrices(fr['visible_points_model'],fr['point_owner'],fr['depth_camera'],boxes,camera,(W,H),masks['boxes_xyxy'])
   for key,val in [('legacy_support',S),('legacy_coverage',C),('legacy_projection_iou',I),('legacy_depth_trim_count',trim)]:exact(z[key],val,key)
   nonempty=np.asarray([mask.any() for mask in masks['masks_bool']],dtype=bool);crop=np.zeros(M,bool)
   for m,(x1,y1,x2,y2) in enumerate(masks['boxes_xyxy']):crop[m]=nonempty[m] and x1<x2 and y1<y2 and min(W,int(x2))>max(0,int(x1)) and min(H,int(y2))>max(0,int(y1))
   exact(crop,z['crop_valid'],'crops');assert crop[pool].all();wide=np.array(sorted(set(pool.tolist())|{m for m in range(M) if crop[m] and od['mask_counts'][:,m].any()}),np.int64);exact(wide,z['wide_pool'],'wide')
   ow,quality,valid=oldw(pool,S,C,I,boxes);ww,_,_=oldw(wide,S,C,I,boxes)
   for key,val in [('old_weights',ow),('wide_old_weights',ww),('legacy_quality',quality),('legacy_valid',valid)]:exact(z[key],val,key)
   exact(ow,old['weights'],'R49W')
   base=ROOT/'crossplatform'/('round24' if row['stage']=='M' else 'round23');suffix='M64' if row['stage']=='M' else '272';text=checked(base/('text'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'),base/('text'+suffix+'_attempt001'));ev=checked(base/('evidence'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'),base/('evidence'+suffix+'_attempt001'));prep=checked(base/('prepared'+suffix+'_attempt001')/'cases_PRIVATE'/(rid+'.json'),base/('prepared'+suffix+'_attempt001'))
   width,height,objs=image_only_scene(masks);pr=np.zeros((M,2))
   for m in range(M):
    if nonempty[m]:pr[m]=relfn(text,objs[m],ev,width,height)
   exact(pr,z['raw_mask_PR'],'fullmaskPR');exact(nonempty,z['relation_geometry_defined'],'relationdefined');exact(pr[pool],old['mask_relations'],'oldPR')
   for key,oldkey in [('image_e','image_e'),('image_full','image_full'),('image_gray','image_gray')]:
    exact(z[key][pool],old[oldkey],key);assert np.isnan(z[key][np.array([m not in set(pool.tolist()) for m in range(M)],dtype=bool)]).all()
   expected_missing=np.array([m for m in wide if m not in set(pool.tolist())],np.int64);exact(expected_missing,z['needs_original_visual_score'],'needs_score');exact(np.isfinite(z['image_e']),z['old_scored_masks'],'scoredmask')
   states=oldread(s,ow,pool,z['image_e'],pr)
   for key in ('vision','position','relation'):exact(states[key],old[key],'old '+key)
   exact(states['scores'],old['arm_scores'][7],'old F')
   wrong,correct=classflags(prep['category'],masks['class_ids']);exact(wrong,z['historical_taxonomy_compatible'],'oldtax');exact(correct,z['corrected_taxonomy_compatible'],'correcttax')
   c=od['mask_counts'];n=od['allvisible'];u=od['unassigned'];a=od['ambiguous'];assert np.array_equal(c.sum(1)+u+a,n)
   for k in (3,10):
    for tag,pp in [('old',pool),('wide',wide)]:
     selected=c[:,pp].astype(float);num=selected.sum(1);den=n.astype(float)+k;cond=np.zeros_like(selected)
     for q in np.flatnonzero(num>0):cond[q]=selected[q]/num[q]
     fields={'conditional_association':cond,'rho':num/den,'unknown_mass':1-num/den,'unknown_prior_mass':k/den,'out_of_pool_foreground_mass':(c.sum(1)-c[:,pp].sum(1))/den,'unassigned_mass':u/den,'ambiguous_mass':a/den}
     for key,val in fields.items():exact(z[f'{tag}_k{k}_{key}'],val,tag+key)
   extra=np.array([m for m in wide if m not in set(pool.tolist())],np.int64);rescued=int((~wrong[extra]&correct[extra]).sum())
   assert case['additional_mask_count']==len(extra) and case['needs_new_visual_mask_count']==len(expected_missing) and case['new_numeric_forward_upper_bound']==4*len(expected_missing) and case['new_masks_old_taxonomy_reject_correct_taxonomy_accept_count']==rescued
   totals.update(n=1,new_masks=len(expected_missing),numeric_upper_bound=4*len(expected_missing),full_matrix_cells=256*M,taxonomy_new_recovered=rescued)
   resultrows.append({'row':f'D{i+1:04d}','platform':row['platform'],'old_pool_count':len(pool),'wide_pool_count':len(wide),'new_masks':len(expected_missing),'new_taxonomy_compatibility_recovered':rescued,'all_checks_PASS':True})
   if (i+1)%32==0:print('INDEPENDENT_PREP50',i+1,flush=True)
 except Exception as exc:
  dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':totals['n'],'no_method_or_GT_change':True});raise
 counts=read(PREP/'COUNTS.json');assert totals['n']==336 and totals['new_masks']==counts['needs_new_visual_masks']==1239 and totals['numeric_upper_bound']==counts['new_numeric_forward_upper_bound']==4956
 dump(OUT/'ANONYMOUS_ROWS.json',resultrows);res={'state':'PASS_R50_INDEPENDENT_PREP_GEOMETRY_ENTRY_ASSOCIATION_QUALITY_NO_GT','n':336,'full_geometry_cells_exact':totals['full_matrix_cells'],'new_masks':totals['new_masks'],'numeric_call_upper_bound':totals['numeric_upper_bound'],'new_masks_rejected_by_oldwrong_but_correct_taxonomy_compatible':totals['taxonomy_new_recovered'],'source_oldW_oldVPR_fullscore_bitexact':True,'new_conditional_a_rho_unknown_parts_recomputed_exact':True,'allmask_old_geometry_and_PR_recomputed':True,'actual_cached_detector_label_flags_recomputed':True,'old_unscored_e_NaN_not_zero':True,'imports_author_R50_algorithm':False,'no_GT_or_media_rawXYZ_models_or_protected_FINAL_read':True,'prepared_seal_sha256':sha(PREP/'SEALED.json'),'program_sha256':sha(__file__),'seconds':time.perf_counter()-start};dump(OUT/'AUDIT_RESULT.json',res);print(json.dumps(res))
if __name__=='__main__':main()

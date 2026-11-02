"""R51 DIAGNOSTIC ONLY hybrid preparation. Never a final-safe feedback input.

Copies old external controls, measures old-mask colors from NEW raw-safe
all-cloud visible pixels, and writes new-instance target numeric work items.
Reference selection is a separately locked stage, not guessed here.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from attributes51 import color_support, COLOR_RGB_PROTOTYPES, MINIMUM_COLOR_PIXELS, COLOR_BETA_PSEUDOCOUNT, COLOR_LOGODDS_CLIP
ROOT=Path(__file__).resolve().parents[2]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);a=ap.parse_args()
 plan=read(a.plan);auth=read(a.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_DIAGNOSTIC_HYBRID_PREPARATION_NO_GT' and auth['plan_sha256']==sha(a.plan)
 assert plan['diagnostic_only'] is True and plan['n']==336 and plan['references_stage']=='NOT_RUN_PENDING_PURE_REFERENCE_SELECTOR'
 for g in ('programs_sha256','inputs_sha256'):
  for f,h in plan[g].items():assert sha(path(f))==h,('pin',f)
 for f in (__file__,Path(__file__).with_name('attributes51.py')):assert plan['programs_sha256'][Path(f).resolve().relative_to(ROOT).as_posix()]==sha(f)
 assert plan['manifest'] in plan['inputs_sha256']
 rows=read(path(plan['manifest']));assert plan['manifest']=='research/round24/input/ALL336_PRIVATE.json' and len(rows)==336
 roots={k:path(plan[k]) for k in ('rawsafe_root','r50_prepared_root','r50_prediction_root','r50_visual_root','r49_stage_root')}
 seals={k:read(v/'SEALED.json') for k,v in roots.items()}
 for k,v in roots.items():assert plan['inputs_sha256'][(v/'SEALED.json').relative_to(ROOT).as_posix()]==sha(v/'SEALED.json')
 def bound(k,rel):
  p=roots[k]/rel;assert sha(p)==seals[k]['files'][rel];return p
 out=path(plan['output_root']);assert out.is_relative_to(ROOT/'crossplatform/round51') and not out.exists();out.mkdir(parents=True)
 for d in ('controls_PRIVATE','oldmask_attributes_PRIVATE','target_instances_PRIVATE','cases_PRIVATE'):(out/d).mkdir()
 started=time.perf_counter();items=[];done=[];unknown=0
 try:
  for row in rows:
   sid=row['id'];safe=load(bound('rawsafe_root','arrays_PRIVATE/'+sid+'.npz'))
   attrs=read(bound('rawsafe_root','attributes_PRIVATE/'+sid+'.json'));rc=read(bound('rawsafe_root','cases_PRIVATE/'+sid+'.json'))
   old=load(bound('r50_prepared_root','arrays_PRIVATE/'+sid+'.npz'));pred=load(bound('r50_prediction_root','samples_PRIVATE/'+sid+'.npz'))
   original=read(bound('r49_stage_root','cases_PRIVATE/'+sid+'.json'));visual=read(bound('r50_visual_root','cases_PRIVATE/'+sid+'.json'))
   assert exact(safe['source_boxes'],old['source_boxes']) and exact(safe['source_scores'],old['source_scores'])
   assert exact(safe['source_boxes'],pred['source_boxes']) and exact(safe['source_scores'],pred['source_scores'])
   mp=path(original['mask_cache_path_PRIVATE']);assert sha(mp)==original['mask_cache_sha256']==visual['mask_cache_sha256']
   masks=load(mp);rgbsha=sha(row['imagepath']);assert rgbsha==rc['raw_bindings']['image_sha256']==visual['image_sha256']
   with Image.open(row['imagepath']) as im:rgb=np.asarray(im.convert('RGB')).copy()
   mb=masks['masks_bool'];assert mb.dtype==bool and tuple(masks['original_hw'])==rgb.shape[:2] and mb.shape[1:]==rgb.shape[:2]
   graph=attrs['text_graph_unchanged'];subject=graph['subject_query'];assert hashlib.sha256(subject.encode()).hexdigest()==visual['actual_subject_query_sha256']
   control={k:masks[k] for k in ('masks_bool','boxes_xyxy','original_hw')}
   if 'labels' in masks:control['labels']=masks['labels']
   for k in ('source_boxes','source_scores','old_pool','wide_pool','old_weights','wide_old_weights'):control[k]=old[k]
   for k in ('image_e','image_full','image_gray','raw_mask_PR'):control[k]=pred[k]
   cp=out/'controls_PRIVATE'/(sid+'.npz');np.savez_compressed(cp,**control)
   dump(out/'controls_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'diagnostic_only':True,'finalsafe_must_not_consume':True,'old_targets_PRIVATE':visual['targets_PRIVATE'],'old_references':visual['references'],'source_arrays_byte_bridge':True,'mask_cache_sha256':sha(mp),'r50_visual_case_sha256':sha(roots['r50_visual_root']/'cases_PRIVATE'/(sid+'.json'))})
   # Reassign only globally visible NEW raw point pixels to OLD masks. No old
   # owner, old point cache or source-box filtering enters this ownership.
   xy=safe['visible_pixels_xy'];hits=mb[:,xy[:,1],xy[:,0]];n=hits.sum(0)
   owner=np.full(len(xy),-1,np.int64)
   if len(mb):owner[n==1]=hits[:,n==1].argmax(0)
   owner[n>1]=-2
   parts=[np.flatnonzero(owner==j) for j in range(len(mb))];offset=np.r_[0,np.cumsum([len(x) for x in parts])].astype(np.int64)
   flat=np.concatenate(parts) if parts else np.empty(0,np.int64)
   fake={'visible_pixels_xy':xy,'visible_depth_camera_m':safe['visible_depth_camera_m'],'visible_point_owner_instance_ids':owner,
    'instance_crop_mask_pixel_offsets':offset,'instance_crop_mask_pixel_xy':xy[flat],
    'instance_visible_point_offsets':offset.copy(),'instance_visible_point_indices':safe['visible_raw_point_indices'][flat]}
   # Reuse the exact color descriptors; do not fabricate XYZ for the color-only branch.
   scene=color_support(rgb[xy[:,1],xy[:,0]]);measured=[];ns=len(xy)
   for j,part in enumerate(parts):
    px=xy[part];pixels=rgb[px[:,1],px[:,0]];nc=len(px);enough=nc>=MINIMUM_COLOR_PIXELS
    local=color_support(pixels);corrected={name:None for name in COLOR_RGB_PROTOTYPES}
    if enough and ns>=MINIMUM_COLOR_PIXELS:
     for name in corrected:
      pp=(nc*local[name]+COLOR_BETA_PSEUDOCOUNT)/(nc+2*COLOR_BETA_PSEUDOCOUNT);bb=(ns*scene[name]+COLOR_BETA_PSEUDOCOUNT)/(ns+2*COLOR_BETA_PSEUDOCOUNT)
      corrected[name]=float(np.clip(np.log(pp/(1-pp))-np.log(bb/(1-bb)),-COLOR_LOGODDS_CLIP,COLOR_LOGODDS_CLIP))
    measured.append({'instance_id':j,'n_unique_visible_rgb_pixels':nc,'color_status':'MEASURED_sparse_visible_support' if enough else 'UNKNOWN_insufficient_visible_pixels','color_soft_support':local if enough else {k:None for k in COLOR_RGB_PROTOTYPES},'color_logodds_relative_scene_visible':corrected,'color_evidence_reliability':float(nc/(nc+MINIMUM_COLOR_PIXELS)) if enough else 0.,'geometry':{'status':'UNKNOWN_color_only_branch_no_geometry_inferred'},'semantic_category':None,'complex_appearance':None,'absence_of_attribute_evidence_is_not_negative':True})
   ma={'schema':'R51_oldmask_new_visible_color_only_v1','instances':measured,'text_graph_unchanged':graph,'entity_attribute_requests':attrs['entity_attribute_requests'],'scene_visible_color_soft_support':scene,'n_scene_visible_pixels':ns,'point_support_pixels_only':True}
   ma.update(diagnostic_only=True,old_mask_overlap_owner_UNKNOWN=True,new_rawsafe_visible_point_binding_sha256=sha(roots['rawsafe_root']/'arrays_PRIVATE'/(sid+'.npz')))
   dump(out/'oldmask_attributes_PRIVATE'/(sid+'.json'),ma)
   targetkeys=['instance_rect_xyxy','instance_crop_mask_pixel_offsets','instance_crop_mask_pixel_xy','source_boxes']
   tp=out/'target_instances_PRIVATE'/(sid+'.npz');np.savez_compressed(tp,**{k:safe[k] for k in targetkeys})
   offsets=safe['instance_crop_mask_pixel_offsets']
   for j,rect in enumerate(safe['instance_rect_xyxy']):
    visible_pixels=int(offsets[j+1]-offsets[j]);valid=rect[2]>rect[0] and rect[3]>rect[1]
    status=('UNKNOWN_INSUFFICIENT_VISIBLE_SUPPORT' if visible_pixels<MINIMUM_COLOR_PIXELS else 'READY' if valid and subject.strip() else 'UNKNOWN_EMPTY_CROP_OR_SUBJECT');unknown+=status!='READY'
    items.append({'id_PRIVATE':sid,'source_instance_id':j,'kind':'target','recipe':'TARGET_DUAL_CROP_FULL_GRAY','query':subject,'status':status,
     'imagepath':row['imagepath'],'image_sha256':rgbsha,'instance_arrays':tp.relative_to(ROOT).as_posix(),'instance_arrays_sha256':sha(tp),
     'n_exclusive_visible_pixels':visible_pixels,'minimum_exclusive_visible_pixels':MINIMUM_COLOR_PIXELS,'rawsafe_array_sha256':sha(roots['rawsafe_root']/'arrays_PRIVATE'/(sid+'.npz')),'pure_text_graph_sha256':rc['pure_text_graph_sha256']})
   done.append(sid);dump(out/'cases_PRIVATE'/(sid+'.json'),{'id_PRIVATE':sid,'diagnostic_only':True,'n_old_masks':len(mb),'n_new_instances':len(offsets)-1,'new_owner_unassigned':int((owner==-1).sum()),'new_owner_ambiguous':int((owner==-2).sum()),'no_old_owner_reused':True})
  dump(out/'TARGET_WORK_ITEMS_PRIVATE.json',items)
  dump(out/'COUNTS.json',{'n':len(done),'target_items':len(items),'target_UNKNOWN_items':unknown,'target_UNKNOWN_insufficient_visible_support_items':sum(x['status']=='UNKNOWN_INSUFFICIENT_VISIBLE_SUPPORT' for x in items),'minimum_exclusive_visible_pixels':MINIMUM_COLOR_PIXELS,'target_new_numeric_call_cap':4*sum(x['status']=='READY' for x in items),'reference_items':0,'reference_stage':'NOT_RUN_PENDING_PURE_REFERENCE_SELECTOR','new_models_or_GT_or_optimizer':False})
  dump(out/'SEALED.json',{'state':'R51_DIAGNOSTIC_HYBRID_OLDMASK_COLOR_AND_TARGET_WORKLIST_COMPLETE','n':len(done),'diagnostic_only':True,'finalsafe_must_not_consume':True,'plan_sha256':sha(a.plan),'seconds':time.perf_counter()-started,'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as e:dump(out/'FAILED.json',{'error_type':type(e).__name__,'error':str(e),'completed_n':len(done),'no_automatic_retry':True});raise
if __name__=='__main__':main()

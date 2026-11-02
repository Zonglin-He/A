"""DRAFT: full336 CPU geometry/evidence-inventory preparation, no new inference.

Reads only old exposed source/camera/masks, sealed R49 visible-point arrays and
legacy text/numeric evidence. Does not decode RGB, raw XYZ, GT or FINAL. Root
must separately lock and grant this real-cache job; module synthetic checks
are not such a grant. Existing R49 artifacts are never edited.
"""
from pathlib import Path
import sys,json,hashlib,time,datetime,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round49'),str(ROOT/'crossplatform/round13'),str(ROOT/'crossplatform/round7')]
from stages49 import mask_scene_lite,relation_value
from ownership49 import frame_route
from association import box_corners
from camera_projection import CameraSpec,project_model_points
from pool_weights50 import (exact,native_crop_valid,wide_entry,taxonomy_flags,legacy_geometry_from_visible,
                           legacy_weights,visible_mass_weights,legacy_operator)
EXP=ROOT/'research/round50/experiment';OUT=ROOT/'crossplatform/round50/prepared336_attempt001'
STAGE=ROOT/'crossplatform/round49/stage336_attempt001';OWN=ROOT/'crossplatform/round49/ownership336_attempt001'
BINDINGS={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def bound(p,folder,sealname='SEALED.json'):
    p=Path(p);folder=Path(folder);sp=folder/sealname;seal=read(sp);digest=sha(p)
    assert seal['files'][p.relative_to(folder).as_posix()]==digest
    BINDINGS[sp.relative_to(ROOT).as_posix()]=sha(sp);BINDINGS[p.relative_to(ROOT).as_posix()]=digest
    return digest

def check():
    plan=read(EXP/'PREP_PLAN_LOCK.json');auth=read(EXP/'PREP_EXECUTION_AUTHORIZATION.json')
    assert auth['state']=='ROOT_AUTHORIZED_R50_DEV336_CACHE_GEOMETRY_PREP'
    assert auth['plan_sha256']==sha(EXP/'PREP_PLAN_LOCK.json')
    assert plan['n']==336 and plan['k_values']==[3,10] and plan['new_model_calls_allowed'] is False
    for group in ('programs_sha256','inputs_sha256'):
        for p,h in plan[group].items():assert sha(ROOT/p)==h,p
    return plan

def project_boxes_original(boxes,camera,size_wh):
    rectangles=np.zeros((256,4));valid=np.zeros(256,bool)
    for q,b in enumerate(boxes):
        if (b[3:]<=0).any():continue
        p=project_model_points(box_corners(b),camera,size_wh)
        valid[q]=bool(np.all(p['depth']>.1) and np.isfinite(p['uv']).all())
        if valid[q]:rectangles[q]=np.r_[p['uv'].min(0),p['uv'].max(0)]
    return rectangles,valid

def main():
    plan=check();assert not OUT.exists();OUT.mkdir()
    for name in ('arrays_PRIVATE','cases_PRIVATE'):(OUT/name).mkdir()
    rr=read(ROOT/'research/round24/input/ALL336_PRIVATE.json');assert len(rr)==336 and len({r['id'] for r in rr})==336
    split={r['id']:s for s in ('D208','V64','M64') for r in read(ROOT/'research/round24/input'/(s+'_PRIVATE.json'))}
    A={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
    AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
    legacybindings=read(ROOT/'research/round49/inventory_attempt001/SOURCE_BINDINGS_PRIVATE.json')
    count=collections.Counter();cases=[];start=time.perf_counter()
    try:
        for i,r in enumerate(rr):
            sid=r['id'];sp=STAGE/'samples_PRIVATE'/(sid+'.npz');cp=STAGE/'cases_PRIVATE'/(sid+'.json')
            op=OWN/'samples_PRIVATE'/(sid+'.npz');ocp=OWN/'cases_PRIVATE'/(sid+'.json')
            for p,f in [(sp,STAGE),(cp,STAGE),(op,OWN),(ocp,OWN)]:bound(p,f)
            old=load(sp);oldcase=read(cp);ownerdata=load(op);owncase=read(ocp)
            exact(old['source_boxes'],ownerdata['source_boxes'],'R49_source_boxes')
            boxes=old['source_boxes'];scores=old['source_scores'];oldpool=old['pool']
            fp=OWN/'frames_PRIVATE'/(owncase['frame_key_PRIVATE']+'.npz');bound(fp,OWN);frame=load(fp)
            fr,mp=frame_route(r,A,AF,RF);assert sha(mp)==owncase['frame_mask_cache_sha256']==oldcase['mask_cache_sha256']
            bound(mp,mp.parent);masks=load(mp);H,W=map(int,masks['original_hw'])
            exact(frame['mask_sha256'],ownerdata['mask_sha256'],'mask_identity_order')
            fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
            if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
            camera=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared')
            rectangles,projected_valid=project_boxes_original(boxes,camera,(W,H))
            geometry=legacy_geometry_from_visible(frame['visible_points_model'],frame['point_owner'],frame['depth_camera'],
                boxes,rectangles,projected_valid,masks['boxes_xyxy'])
            support=geometry['point_support_count'];coverage=geometry['point_support_fraction'];pi=geometry['projection_iou']
            oldassoc=legacy_weights(oldpool,support,coverage,pi,boxes)
            exact(oldassoc['weights'],old['weights'],'original_oldpool_W')
            lite=mask_scene_lite(masks)
            crops=native_crop_valid(masks['boxes_xyxy'],(W,H),[x['mask_geometry_defined'] for x in lite['objects']])
            entry=wide_entry(oldpool,ownerdata['mask_counts'],crops);wide=entry['pool']
            wideassoc=legacy_weights(wide,support,coverage,pi,boxes)
            suffix='M64' if r['stage']=='M' else '272';base=ROOT/'crossplatform'/('round24' if r['stage']=='M' else 'round23')
            paths={kind:base/(kind+suffix+'_attempt001')/'cases_PRIVATE'/(sid+'.json') for kind in ('text','evidence','prepared')}
            for path in paths.values():
                digest=sha(path);assert legacybindings[path.relative_to(ROOT).as_posix()]==digest
                BINDINGS[path.relative_to(ROOT).as_posix()]=digest
            text=read(paths['text']);evidence=read(paths['evidence']);prep=read(paths['prepared'])
            assert prep['mask_pool_PRIVATE']==oldpool.tolist()
            M=len(masks['masks_bool']);relation_all=np.zeros((M,2))
            relation_defined=np.array([x['mask_geometry_defined'] for x in lite['objects']],bool)
            for m,obj in enumerate(lite['objects']):
                if relation_defined[m]:relation_all[m]=relation_value(text,obj,evidence,W,H)
            exact(relation_all[oldpool],old['mask_relations'],'legacy_rawmask_PR')
            e=np.full(M,np.nan);full=np.full(M,np.nan);gray=np.full(M,np.nan)
            e[oldpool]=old['image_e'];full[oldpool]=old['image_full'];gray[oldpool]=old['image_gray']
            oldread=legacy_operator(scores,oldassoc['weights'],oldpool,e,relation_all)
            for name in ('vision','position','relation'):exact(oldread[name],old[name],'old_'+name)
            exact(oldread['scores'],old['arm_scores'][7],'old_fullG_endpoint')
            flags=taxonomy_flags(prep['category'],masks['class_ids'])
            need=np.array([m for m in wide if not np.isfinite(e[m])],np.int64)
            payload=dict(source_boxes=boxes,source_scores=scores,old_pool=oldpool,wide_pool=wide,
                mask_sha256=frame['mask_sha256'],native_instance_id=frame['native_instance_id'],
                native_detector_class=masks['class_ids'],detector_boxes=masks['boxes_xyxy'],crop_valid=crops,
                mask_counts=ownerdata['mask_counts'],allvisible=ownerdata['allvisible'],unassigned=ownerdata['unassigned'],ambiguous=ownerdata['ambiguous'],
                legacy_support=support,legacy_coverage=coverage,legacy_projection_iou=pi,
                legacy_depth_trim_count=geometry['point_count_after_depth_trim'],legacy_quality=oldassoc['quality'],legacy_valid=oldassoc['valid'],
                old_weights=oldassoc['weights'],wide_old_weights=wideassoc['weights'],raw_mask_PR=relation_all,
                relation_geometry_defined=relation_defined,image_e=e,image_full=full,image_gray=gray,
                old_scored_masks=np.isfinite(e),needs_original_visual_score=need,
                historical_taxonomy_compatible=flags['historical_taxonomy_compatible'],corrected_taxonomy_compatible=flags['corrected_taxonomy_compatible'])
            for k in (3,10):
                for tag,pool in [('old',oldpool),('wide',wide)]:
                    nw=visible_mass_weights(pool,ownerdata['mask_counts'],ownerdata['allvisible'],ownerdata['unassigned'],ownerdata['ambiguous'],boxes,k)
                    for field in ('conditional_association','rho','unknown_mass','unknown_prior_mass','out_of_pool_foreground_mass','unassigned_mass','ambiguous_mass'):
                        payload[f'{tag}_k{k}_{field}']=nw[field]
            np.savez_compressed(OUT/'arrays_PRIVATE'/(sid+'.npz'),**payload)
            record={'id_PRIVATE':sid,'platform':r['platform'],'split':split[sid],
                'old_pool_count':len(oldpool),'wide_pool_count':len(wide),'additional_mask_count':len(entry['additional_masks']),
                'needs_new_visual_mask_count':len(need),'new_numeric_forward_upper_bound':int(4*len(need)),
                'entry_reasons':entry['entry_reasons'],'old_endpoint_W_VPR_scores_bitexact':True,
                'new_masks_old_taxonomy_reject_correct_taxonomy_accept_count':int((~flags['historical_taxonomy_compatible'][entry['additional_masks']]&flags['corrected_taxonomy_compatible'][entry['additional_masks']]).sum()),
                'actual_subject_query_sha256':hashlib.sha256(text['subject_query'].encode()).hexdigest(),
                'actual_text_source_PRIVATE':paths['text'].relative_to(ROOT).as_posix(),
                'legacy_reference_numeric_empty':not bool(evidence['references']),
                'reference_numeric_not_newly_acquired':True,'new_model_calls':0,
                'image_pixels_rawXYZ_GT_FINAL_read':False}
            dump(OUT/'cases_PRIVATE'/(sid+'.json'),record);cases.append(record)
            count['n']+=1;count['additional_masks']+=len(entry['additional_masks']);count['needs_new_visual_masks']+=len(need)
            count['new_numeric_forward_upper_bound']+=int(4*len(need))
            if (i+1)%16==0:print('PREPARE50',i+1,dict(count),flush=True)
    except Exception as exc:
        dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':count['n'],
            'STOP_no_new_predictions_or_score':True,'original_attempt_preserved':True});raise
    dump(OUT/'CASES_PRIVATE.json',cases);dump(OUT/'COUNTS.json',dict(count));dump(OUT/'BINDINGS_PRIVATE.json',BINDINGS)
    dump(OUT/'SEALED.json',{'state':'R50_DEV336_FULL_MASK_GEOMETRY_AND_UNSCORED_ENTRANCE_BEFORE_NEW_MODEL_OR_TASK_SCORE',
        'n':336,'files':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        'plan_sha256':sha(EXP/'PREP_PLAN_LOCK.json'),'seconds':time.perf_counter()-start,
        'new_model_calls':0,'optimizer_steps':0,'no_GT_or_new_readout_prediction':True,
        'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print('PREPARE50_SEALED',dict(count),flush=True)

if __name__=='__main__':main()

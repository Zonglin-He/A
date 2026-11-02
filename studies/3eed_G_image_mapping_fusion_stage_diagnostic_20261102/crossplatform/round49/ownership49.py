"""Proposed full336 target-independent visible-point ownership cache.

Only original raw XYZ, cached calibration/masks and original source6D boxes.
No image pixels, description text, task annotations, method predictions or GT.
No model imports. Requires a new root plan and execution authorization.
"""
from pathlib import Path
import sys,json,hashlib,datetime,collections
import ijson
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round7')]
from scene_observation import build_scene_observation
from camera_projection import CameraSpec
EXP=ROOT/'research/round49/experiment'
OUT=ROOT/'crossplatform/round49/ownership336_attempt001'
ALLOWED_ROW={'id','platform','stage','scene','pointpath','metapath','imagepath'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)

def manifest_identity_only(path):
    """Stream allowlisted scalar fields; never materialize caption strings.

    The JSON tokenizer necessarily traverses bytes of other fields, but their
    values are discarded immediately and never stored, displayed or consumed.
    """
    out=[];row=None
    with Path(path).open('rb') as f:
        for prefix,event,value in ijson.parse(f):
            if prefix=='item' and event=='start_map':row={}
            elif prefix=='item' and event=='end_map':
                assert set(row)==ALLOWED_ROW,(set(row),ALLOWED_ROW);out.append(row);row=None
            elif row is not None and prefix.startswith('item.') and prefix[5:] in ALLOWED_ROW and event=='string':
                row[prefix[5:]]=value
    assert len(out)==336 and len({x['id'] for x in out})==336
    return out

def checked(p,folder,bindings):
    p=Path(p);folder=Path(folder);sp=folder/'SEALED.json';seal=read(sp);h=sha(p)
    assert seal['files'][str(p.relative_to(folder))]==h,('sealed_artifact_changed',str(p))
    bindings[str(sp.relative_to(ROOT))]=sha(sp);bindings[str(p.relative_to(ROOT))]=h
    return h

def frame_route(r,A,AF,RF):
    if r['stage']=='A':
        case=A[r['id']];fr=AF[case['frame_alias']]
        assert case['platform']==r['platform'] and case['scene_PRIVATE']==r['scene']
        path=ROOT/case['mask_dir']/(case['frame_alias']+'.npz')
    else:
        matches=[(k,f) for k,f in RF.items() if all(f.get(k2)==r[k2] for k2 in ('imagepath','pointpath','metapath'))]
        assert len(matches)==1,('nonunique_frame_path_join',r['id'],len(matches))
        key,fr=matches[0];path=ROOT/'crossplatform/round12/masks501_attempt001'/(key+'.npz')
    assert fr['platform']==r['platform']
    for key in ('imagepath','pointpath','metapath'):assert fr[key]==r[key]
    return fr,path

def ownership_counts(points,owner,boxes,M):
    """R28 strict yaw0 inside test; visible surfaces never become box bounds.

    Every globally visible point contributes to exactly one of exclusive mask,
    unassigned(-1), ambiguous(-2). Geometric box membership is not identity.
    """
    points=np.asarray(points);owner=np.asarray(owner);boxes=np.asarray(boxes)
    assert points.dtype==np.float32 and boxes.dtype==np.float32
    assert points.ndim==2 and points.shape[1]==3 and owner.shape==(len(points),)
    assert boxes.shape==(256,6) and np.isfinite(boxes).all()
    counts=np.zeros((256,M),np.int64);unassigned=np.zeros(256,np.int64);ambiguous=np.zeros(256,np.int64);allvisible=np.zeros(256,np.int64)
    nonpositive=(boxes[:,3:]<=0).any(1)
    for q,box in enumerate(boxes):
        if nonpositive[q]:continue  # Explicit UNKNOWN, not negative identity evidence.
        inside=(abs(points-box[:3])<=box[3:]/2).all(1)
        values=owner[inside];positive=values[values>=0]
        counts[q]=np.bincount(positive,minlength=M) if M else np.zeros(0,np.int64)
        unassigned[q]=int((values==-1).sum());ambiguous[q]=int((values==-2).sum());allvisible[q]=int(inside.sum())
    assert np.array_equal(counts.sum(1)+unassigned+ambiguous,allvisible)
    assert (owner>=-2).all() and (owner[owner>=0]<M).all()
    status=np.where(nonpositive,'UNKNOWN_nonpositive_source_size','DEFINED_yaw0_visible_support')
    return dict(mask_counts=counts,unassigned=unassigned,ambiguous=ambiguous,allvisible=allvisible,
                nonpositive_source_size=nonpositive,geometry_status=status)

def check():
    plan=read(EXP/'OWNERSHIP_PLAN_LOCK.json');auth=read(EXP/'OWNERSHIP_EXECUTION_AUTHORIZATION.json')
    assert auth['state']=='ROOT_AUTHORIZED_R49_FULL336_VISIBLE_OWNERSHIP'
    assert auth['plan_sha256']==sha(EXP/'OWNERSHIP_PLAN_LOCK.json')
    assert plan['n']==336 and plan['no_description_GT_media_or_model_inputs'] is True
    for group in ('programs_sha256','inputs_sha256'):
        for p,h in plan[group].items():assert sha(ROOT/p)==h,p
    # Pin the prior inventory files, not just a newly mutable source seal.
    inv=ROOT/'research/round49/inventory_attempt001';seal=read(inv/'SEALED.json')
    for p,h in seal['files'].items():assert sha(inv/p)==h
    oldbindings=read(inv/'SOURCE_BINDINGS_PRIVATE.json')
    return plan,oldbindings

def main():
    plan,oldbindings=check();assert not OUT.exists();OUT.mkdir()
    for d in ('frames_PRIVATE','samples_PRIVATE','cases_PRIVATE'):(OUT/d).mkdir()
    rows=manifest_identity_only(ROOT/'research/round24/input/ALL336_PRIVATE.json')
    A={r['id']:r for r in read(ROOT/'research/round17/input/A_CASES_PRIVATE.json')}
    AF=read(ROOT/'research/round17/input/A_FRAME_INDEX_PRIVATE.json');RF=read(ROOT/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json')
    bindings={};frames={};records=[];counter=collections.Counter()
    try:
        for i,r in enumerate(rows):
            fr,maskpath=frame_route(r,A,AF,RF);maskhash=checked(maskpath,maskpath.parent,bindings)
            pointpath=Path(r['pointpath']);pointsha=sha(pointpath);assert pointsha==fr['point_sha256']
            bindings[str(pointpath)]=pointsha  # Local private absolute raw-input binding.
            fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
            if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
            # Frame-index JSON uses Python float, matching original read_json.
            camera=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared')
            keydata={'platform':r['platform'],'point_sha256':pointsha,'mask_sha256':maskhash,'camera_fields':fields,'image_sha256':fr['image_sha256']}
            key=hashlib.sha256(json.dumps(keydata,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            if key not in frames:
                with np.load(maskpath,allow_pickle=False) as z:
                    ins={k:z[k].copy() for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')};H,W=map(int,z['original_hw'])
                raw=np.load(pointpath,allow_pickle=False) if pointpath.suffix=='.npy' else np.fromfile(pointpath,np.float32).reshape(-1,4)
                assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
                raw=np.ascontiguousarray(raw[:,:3],np.float32)
                scene=build_scene_observation(raw_points=raw,camera_spec=camera,image_size_wh=(W,H),instances=ins,image_sha256=fr['image_sha256'])
                mh=np.array([o['mask_sha256'] for o in scene['objects']]);native=np.array([o['native_instance_id'] for o in scene['objects']],np.int64)
                np.savez_compressed(OUT/'frames_PRIVATE'/(key+'.npz'),visible_points_model=scene['visible_points_model'],point_owner=scene['point_owner'],
                    visible_uv=scene['visible_uv'],depth_camera=scene['depth_camera'],physical_raw_ids=scene['physical_raw_ids'],
                    mask_sha256=mh,native_instance_id=native,image_size_wh=np.array([W,H],np.int64))
                # Reuse only minimal visible-point arrays. Retaining every
                # full R15 scene would also retain all high-resolution masks
                # and redundant object arrays across the entire cohort.
                frames[key]={'points':scene['visible_points_model'],'owner':scene['point_owner'],
                    'visibility_sha256':scene['visibility_sha256'],'nmask':len(scene['objects']),
                    'mask_sha256':mh,'native':native,'keydata':keydata}
                counter['unique_frames']+=1
            frame=frames[key]
            sourcepath=ROOT/'crossplatform/round24/capture336_attempt001/samples_PRIVATE'/(r['id']+'.npz')
            sourcesha=checked(sourcepath,sourcepath.parent.parent,bindings)
            assert oldbindings[str(sourcepath.relative_to(ROOT))]==sourcesha
            with np.load(sourcepath,allow_pickle=False) as z:boxes=z['boxes'].copy()
            result=ownership_counts(frame['points'],frame['owner'],boxes,frame['nmask'])
            np.savez_compressed(OUT/'samples_PRIVATE'/(r['id']+'.npz'),source_boxes=boxes,mask_sha256=frame['mask_sha256'],native_instance_id=frame['native'],
                **result)
            case={'id_PRIVATE':r['id'],'platform':r['platform'],'stage':r['stage'],'frame_key_PRIVATE':key,
                'frame_mask_cache_sha256':maskhash,'raw_XYZ_sha256':pointsha,'source_cache_sha256':sourcesha,
                'visible_point_n':len(frame['owner']),'independent_mask_n':frame['nmask'],
                'visibility_sha256':frame['visibility_sha256'],'source_boxes_unchanged':True,
                'support_is_proxy_not_GT_identity':True,'UNKNOWN_preserved_no_target_label_or_method_selection':True,
                'caption_field_materialized_or_used':False,'caption_bytes_tokenized_then_discarded':True,
                'mask_class_scores_used_only_for_original_helper_validation_not_supervision_or_selection':True,
                'nonpositive_query_UNKNOWN_n':int(result['nonpositive_source_size'].sum()),
                'raw_read_format':'original_R15_npy_or_float32_binary_Nx4_then_XYZfloat32',
                'image_pixels_read':False,'GT_container_read':False,'model_forward':False}
            dump(OUT/'cases_PRIVATE'/(r['id']+'.json'),case);records.append(case);counter['n']+=1
            if (i+1)%16==0:print('OWNERSHIP49',i+1,'frames',counter['unique_frames'],flush=True)
        assert counter['n']==336
    except Exception as exc:
        dump(OUT/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc),'completed_n':counter['n'],
            'no_automatic_subset_switch':True,'no_prediction_or_task_scoring':True});raise
    dump(OUT/'COUNTS.json',dict(counter));dump(OUT/'ROWS_PRIVATE.json',records);dump(OUT/'BINDINGS_PRIVATE.json',bindings)
    dump(OUT/'SEALED.json',{'state':'FULL336_TARGET_INDEPENDENT_VISIBLE_POINT_OWNERSHIP_BEFORE_IDENTITY_JOIN',
        'n':336,'files':{p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},
        'plan_sha256':sha(EXP/'OWNERSHIP_PLAN_LOCK.json'),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'no_GT_RGB_or_model':True,'caption_value_not_materialized_or_used':True,'caption_bytes_tokenized_then_discarded':True,
        'no_box_or_parameter_change':True})
    print('OWNERSHIP49_SEALED',dict(counter),flush=True)

if __name__=='__main__':main()

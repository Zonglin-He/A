"""Actual R10 geometry from sealed source and input observations. No GT API."""
from pathlib import Path
import argparse,dataclasses,hashlib,json,sys,time
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path[:0]=[str(D),str(D.parent/'round7'),str(D.parent/'round7/experiment')]
from runner_helpers import sha,read,validate_manifest,eval_boxes,to_json,array_sha
from camera_projection import read_camera_allowlist,CameraSpec

ARMS=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')

def main(a):
    a.plan=a.plan.resolve();a.output=a.output.resolve()
    assert sha(a.plan)==a.plan_sha256;plan=read(a.plan)
    assert plan['stage']=='geometry_direct' and plan['GT_read_permitted'] is False
    source=Path(plan['source_dir']);sealpath=source/'SEALED.json'
    assert sha(sealpath)==plan['source_seal_sha256'];s=read(sealpath)
    for name,digest in s['files'].items():assert sha(source/name)==digest,name
    assert s['source_unchanged'] and s['source_repeat_bitexact']
    manifest=Path(plan['manifest']);assert sha(manifest)==plan['manifest_sha256']
    rows=validate_manifest(read(manifest));assert len(rows)==s['n_samples']==plan['expected_samples']
    assert sha(manifest)==s['manifest_sha256']
    source_plan=read(source/'registered_plan.json')
    source_traces={r['id']:r for r in read(source/'trace.json')}
    for name,digest in plan['program_sha256'].items():assert sha(ROOT/name)==digest,name
    from visibility_geometry import generate_geometry_targets,GeometryConfig
    from rgb_support import make_rgb_support,RGBSupportConfig
    import cv2,scipy
    cv2.setNumThreads(1)
    gc=GeometryConfig();rc=RGBSupportConfig()
    assert dataclasses.asdict(gc)==plan['geometry_config']
    assert dataclasses.asdict(rc)==plan['rgb_config']
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    (a.output/'manifest_input.json').write_bytes(manifest.read_bytes())
    start=time.perf_counter();files={};traces=[];camera_cache={}
    for i,row in enumerate(rows):
        tick=time.perf_counter();cp=source/'source_cache'/f'{row["id"]}.npz'
        op=source/'observations'/f'{row["id"]}.npz'
        with np.load(cp,allow_pickle=False) as z:cache={k:z[k].copy() for k in z.files}
        with np.load(op,allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
        pc=obs['sampled_points_model'];ix=obs['raw_indices'];boxes=cache['boxes_raw6'][0];scores=cache['text_scores'][0];q=int(cache['query_index'])
        assert boxes.shape==(256,6) and scores.shape==(256,) and q==int(scores.argmax())
        assert cache['point_sample_sha256'].tobytes().hex()==array_sha(pc)
        assert cache['raw_indices_sha256'].tobytes().hex()==array_sha(ix)
        assert np.array_equal(cache['point_sample_sha256'],obs['point_sample_sha256'])
        assert cache['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()
        assert cache['actual_model_text_sha256'].tobytes()==hashlib.sha256((row['caption']+source_plan['sentinel']).encode()).digest()
        assert sha(row['imagepath'])==source_traces[row['id']]['image_sha256']
        assert sha(row['metapath'])==source_traces[row['id']]['camera_container_sha256']
        with Image.open(row['imagepath']) as im:rgb=np.array(im.convert('RGB'))
        camera_error=None
        try:
            if row['metapath'] not in camera_cache:camera_cache[row['metapath']]=read_camera_allowlist(row['metapath'])
            camera=CameraSpec.from_camera_fields(row['platform'],camera_cache[row['metapath']],waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
            camera.validated()
        except ValueError as error:
            camera=None;camera_error=str(error)
        # Model sampling can repeat physical raw points. Geometry counts unique
        # observations, never 16384 replacement rows as independent support.
        observed_unique,unique_first=np.unique(pc,axis=0,return_index=True)
        observed_raw_indices=ix[unique_first]
        if camera is None:
            reason='camera_contract_invalid: '+camera_error
            seg={'mask2d':np.zeros(rgb.shape[:2],bool),'valid':False,'diagnostics':{'reason':reason}}
            result={'arm_names':ARMS,'arm_boxes_raw6':np.broadcast_to(boxes[q],(4,6)).copy(),
                    'pre_constraint_geometry6':np.broadcast_to(boxes[q],(4,6)).astype(np.float64).copy(),
                    'acceptance_mask6':np.zeros((4,6),bool),
                    'point_support_sha256':np.array([hashlib.sha256(b'').hexdigest()]*4),
                    'provenance':np.full((4,6),'source_fallback_camera_contract'),
                    'arm_status':['source_q0']+[reason]*3,'diagnostics':{'reason':reason,'G_not_called':True}}
        else:
            seg=make_rgb_support(rgb=rgb,points_fit=observed_unique,source_box6=boxes[q],camera_spec=camera,config=rc)
            assert seg['mask2d'].shape==rgb.shape[:2] and seg['mask2d'].dtype==bool
            result=generate_geometry_targets(points_fit=observed_unique,raw_indices=observed_raw_indices,source_box6=boxes[q],q0=q,camera_spec=camera,rgb_mask2d=seg['mask2d'],segmentation_valid=seg['valid'],config=gc)
        assert tuple(result['arm_names'])==ARMS
        raw=np.asarray(result['arm_boxes_raw6'],np.float32)
        acceptance=np.asarray(result['acceptance_mask6'],bool)
        provenance=np.asarray(result['provenance'])
        assert raw.shape==acceptance.shape==provenance.shape==(4,6)
        assert np.isfinite(raw).all() and np.array_equal(raw[0],boxes[q])
        before=np.asarray(result['pre_constraint_geometry6'],np.float64)
        assert before.shape==(4,6) and np.isfinite(before).all()
        assert np.array_equal(acceptance[1],acceptance[2]) and np.array_equal(acceptance[1],acceptance[3])
        support=np.asarray(result['point_support_sha256'])
        assert support.shape==(4,) and support[1]==support[2]
        assert np.array_equal(raw[~acceptance],np.broadcast_to(boxes[q],raw.shape)[~acceptance])
        changed=raw!=boxes[q]
        status=np.asarray(result.get('arm_status',['source_q0']+['geometry_eligible' if acceptance[j].any() else 'source_fallback_no_eligible_coordinate' for j in range(1,4)]))
        z={'sample_id':np.array(row['id']),'source_boxes_raw6':boxes,'source_boxes_eval6':eval_boxes(boxes),
           'source_text_scores256':scores,
           'q_source':np.array(q,np.int64),'arm_names':np.array(ARMS),'arm_query_index':np.full(4,q,np.int64),
           'arm_status':status,'arm_boxes_raw6':raw,'arm_boxes_eval6':eval_boxes(raw),
           'pre_constraint_geometry6':before,
           'acceptance_mask6':acceptance,'actual_changed_axes':changed,'point_support_sha256':support,
           'provenance':provenance,'point_sample_sha256':cache['point_sample_sha256'],
           'raw_indices_sha256':cache['raw_indices_sha256'],
           'source_cache_sha256':np.frombuffer(bytes.fromhex(sha(cp)),np.uint8),
           'image_file_sha256':np.frombuffer(bytes.fromhex(sha(row['imagepath'])),np.uint8),
           'RGB_mask2d_private':seg['mask2d'],
           'RGB_mask_sha256':np.frombuffer(bytes.fromhex(array_sha(seg['mask2d'])),np.uint8)}
        name=row['id']+'.npz';np.savez_compressed(a.output/name,**z);files[name]=sha(a.output/name)
        trace={'id':row['id'],'platform':row['platform'],'scene':row['scene'],'q_source':q,
               'source_cache_file_sha256':sha(cp),'observation_file_sha256':sha(op),
               'RGB':seg['diagnostics'],'geometry':{k:v for k,v in result.items() if k not in ['arm_boxes_raw6','pre_constraint_geometry6','acceptance_mask6','provenance','mask2d']},
               'camera_fields_only_allowlist':True,'camera_container_sha256':sha(row['metapath']),
               'language_conditions_original_frozen_q0_only_no_new_text_localizer':True,
               'model_input_rows':len(pc),'unique_raw_indices':len(np.unique(ix)),
               'geometry_unique_XYZ_rows':len(observed_unique),
               'geometry_unique_XYZ_sha256':array_sha(observed_unique),
               'seconds':time.perf_counter()-tick}
        traces.append(to_json(trace))
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(traces[-1],allow_nan=False)+'\n')
        if i%10==0 or i+1==len(rows):print(json.dumps({'completed':i+1,'n':len(rows),'segmentation_valid':bool(seg['valid']),'active_axes':acceptance[1].tolist(),'full_actual_changed':bool(changed[1].any()),'elapsed_seconds':time.perf_counter()-start}),flush=True)
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    for n in ['trace.json','trace.jsonl','registered_plan.json','manifest_input.json']:files[n]=sha(a.output/n)
    out={'state':'sealed_R10_fixedq0_direct_geometry_before_GT','n_samples':len(rows),'n_arms':4,
         'manifest_sha256':sha(manifest),'source_seal_sha256':sha(sealpath),'plan_sha256':sha(a.plan),
         'source_checkpoint_sha256':s['source_checkpoint_sha256'],
         'source_model_state_fingerprint':s['source_model_state_fingerprint'],
         'strict_FP32_flags':s['strict_FP32_flags'],'program_sha256':plan['program_sha256'],
         'program_sha256_base':'workspace_root','same_support_and_acceptance_full_noRGB':True,
         'all_queries_fixed_original_q0':True,'no_GT_or_parameter_update':True,
         'geometry_config':plan['geometry_config'],'rgb_config':plan['rgb_config'],
         'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},
         'elapsed_seconds':time.perf_counter()-start,'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(out,indent=2)+'\n')
    print('R10_GEOMETRY_SEALED',sha(a.output/'SEALED.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())

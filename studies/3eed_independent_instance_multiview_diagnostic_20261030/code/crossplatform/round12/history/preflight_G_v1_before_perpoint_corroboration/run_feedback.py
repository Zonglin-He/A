"""R12 CPU8-arm direct geometry from sealed source and full-image masks.
No model/GPU/GT consumer. Original input-only frame index supplies K/E; raw pose
is not passed to primary registration. No target annotation container is parsed.
"""
from pathlib import Path
from functools import lru_cache
from dataclasses import asdict
import argparse,hashlib,json,sys,time
import numpy as np
from PIL import Image
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
sys.path[:0]=[str(D),str(D.parent/'round7'),str(D.parent/'round7/experiment')]
from runner_helpers import sha,read,validate_manifest,eval_boxes,to_json,array_sha
from camera_projection import CameraSpec
from multiview_feedback import build_geometry_feedback,FeedbackConfig,ARMS
from temporal_association import TemporalConfig
from geometry_feedback import GeometryConfig
from rgb_support import RGBSupportConfig
MASK_KEYS=('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')
SOURCE_KEYS=('boxes_raw6','text_scores','query_index','sample_id','source_checkpoint_sha256','point_sample_sha256','raw_indices_sha256','manifest_row_sha256','actual_model_text_sha256')
def _safe_member(base,name):
    p=Path(name)
    assert not p.is_absolute() and '..' not in p.parts and p.suffix in ('.json','.npz')
    return base/p

def main(a):
    entry_start=time.perf_counter()
    a.plan=a.plan.resolve();a.output=a.output.resolve();assert sha(a.plan)==a.plan_sha256
    plan=read(a.plan);assert plan['stage']=='multiview_direct_geometry' and plan['GT_read_permitted'] is False
    assert plan['model_or_GPU_execution_permitted'] is False
    for name,digest in plan['program_sha256'].items():assert sha(ROOT/name)==digest,name
    inputs={}
    for key in ('manifest','pairmap','frame_index'):
        path=Path(plan[key]);assert sha(path)==plan[key+'_sha256'];inputs[key]=read(path)
    rows=validate_manifest(inputs['manifest']);pairs=inputs['pairmap'];index=inputs['frame_index']
    assert len(rows)==pairs['anchor_count']==plan['expected_samples'] and 0<len(rows)<=160
    assert pairs['anchor_manifest_sha256']==plan['manifest_sha256'] and pairs['frame_index_sha256']==plan['frame_index_sha256']
    pairs={x['anchor_id_PRIVATE']:x for x in pairs['rows_PRIVATE']};assert set(pairs)=={r['id'] for r in rows}
    source=Path(plan['source_dir']);masks=Path(plan['mask_dir'])
    assert sha(source/'SEALED.json')==plan['source_seal_sha256'] and sha(masks/'SEALED.json')==plan['mask_seal_sha256']
    ss=read(source/'SEALED.json');ms=read(masks/'SEALED.json');sourceplan=read(source/'registered_plan.json')
    assert ss['state']=='sealed_R12_source_forward_and_input_observations_no_G_no_GT' and ss['n_samples']==len(rows) and ss['manifest_sha256']==plan['manifest_sha256']
    assert ss['source_unchanged'] and ss['source_repeat_bitexact']
    assert ms['state']=='sealed_R12_independent_whole_image_instance_masks_no_q0_caption_or_GT' and ms['trained_or_updated'] is False and ms['mask_returned_original_image_pixels']
    for base,seal in ((source,ss),(masks,ms)):
        for name,h in seal['files'].items():assert sha(_safe_member(base,name))==h,name
    mt={x['frame_alias']:x for x in read(masks/'TRACE.json')};st={x['id']:x for x in read(source/'trace.json')}
    for key,cfg in [('feedback_config',FeedbackConfig()),('temporal_config',TemporalConfig()),('geometry_config',GeometryConfig()),('RGB_config',RGBSupportConfig())]:assert asdict(cfg)==plan[key]
    import cv2,scipy
    cv2.setNumThreads(1)
    @lru_cache(maxsize=6)
    def loadframe(alias):
        row=index[alias];mp=alias+'.npz';assert mp in ms['files'] and alias in mt
        for key,hkey in [('imagepath','image_sha256'),('pointpath','point_sha256'),('metapath','metadata_sha256')]:assert sha(row[key])==row[hkey],(alias,key)
        assert row['image_sha256']==mt[alias]['image_sha256']
        rgb=np.array(Image.open(row['imagepath']).convert('RGB'),copy=True)
        pp=Path(row['pointpath']);raw=np.load(pp,allow_pickle=False) if pp.suffix=='.npy' else np.fromfile(pp,np.float32).reshape(-1,4)
        assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all();raw=np.ascontiguousarray(raw[:,:3],np.float32)
        fields={k:row[k] for k in ('image_intrinsic','image_extrinsic')}
        if row['camera_distortion'] is not None:fields['camera_distortion']=row['camera_distortion']
        camera=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared');camera.validated()
        with np.load(masks/mp,allow_pickle=False) as z:
            inst={k:z[k].copy() for k in MASK_KEYS};hw=z['original_hw'].copy()
        assert tuple(hw)==rgb.shape[:2] and len(inst['scores'])==mt[alias]['instance_n'] and np.all(inst['scores']>=.5)
        assert [hashlib.sha256(m.tobytes()).hexdigest() for m in inst['masks_bool']]==mt[alias]['mask_sha256']
        return {'frame_alias':alias,'rgb':rgb,'raw_points':raw,'camera':camera,'instances':inst}
    integrity_validation_seconds=time.perf_counter()-entry_start
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
    (a.output/'manifest_input.json').write_bytes(Path(plan['manifest']).read_bytes())
    files={};traces=[];start=time.perf_counter()
    for i,row in enumerate(rows):
        tick=time.perf_counter();p=pairs[row['id']];alias=p['anchor_frame_alias'];fr=index[alias]
        assert p['platform']==row['platform']==fr['platform'] and p['scene_PRIVATE']==row['scene']==fr['scene_PRIVATE']
        assert all(fr[k]==row[k] for k in ('pointpath','metapath','imagepath'))
        assert fr['image_sha256']==st[row['id']]['image_sha256'] and fr['point_sha256']==st[row['id']]['point_file_sha256'] and fr['metadata_sha256']==st[row['id']]['camera_container_sha256']
        cp=source/'source_cache'/(row['id']+'.npz')
        with np.load(cp,allow_pickle=False) as z:c={k:z[k].copy() for k in SOURCE_KEYS}
        assert str(c['sample_id'])==row['id'] and str(c['source_checkpoint_sha256'])==ss['source_checkpoint_sha256']
        assert c['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()
        assert c['actual_model_text_sha256'].tobytes()==hashlib.sha256((row['caption']+sourceplan['sentinel']).encode()).digest()
        boxes=c['boxes_raw6'][0];scores=c['text_scores'][0];q=int(c['query_index']);assert boxes.shape==(256,6) and scores.shape==(256,) and q==int(scores.argmax())
        anchor=loadframe(alias);neighbors=[];roles=[]
        for role in ('previous','next'):
            neighbor=p[role]
            if neighbor is None:roles.append({'role':role,'status':'missing_prelocked_no_replacement'});continue
            other=neighbor['frame_alias'];nr=index[other]
            assert nr['platform']==row['platform'] and nr['scene_PRIVATE']==row['scene'] and nr['physical_group_sha256']!=fr['physical_group_sha256']
            neighbors.append(loadframe(other));roles.append({'role':role,'frame_alias':other,'counter_delta_not_seconds':neighbor['counter_delta'],'same_scene':True,'physical_time_seconds_or_sync_verified':False})
        wrong=None
        if p['wrongview'] is not None:
            other=p['wrongview']['frame_alias'];wr=index[other];assert wr['platform']==row['platform'] and wr['scene_PRIVATE']!=row['scene'] and wr['physical_group_sha256']!=fr['physical_group_sha256'];wrong=loadframe(other)
        g=build_geometry_feedback(anchor=anchor,neighbors=neighbors,wrong_neighbor=wrong,source_box6=boxes[q],q0=q,caption=row['caption'],seed=int(row['seed']))
        raw=np.asarray(g['arm_boxes_raw6'],np.float32);accept=np.asarray(g['acceptance_mask6'],bool)
        assert tuple(g['arm_names'])==ARMS and raw.shape==accept.shape==(8,6) and np.isfinite(raw).all() and np.array_equal(raw[0],boxes[q])
        assert np.all(g['arm_query_index']==q)
        assert np.array_equal(raw[5],boxes[q]+np.float32(.5)*(raw[2]-boxes[q])) and np.array_equal(raw[6],boxes[q]+np.float32(.5)*(raw[3]-boxes[q]))
        payload={'sample_id':np.array(row['id']),'source_boxes_raw6':boxes,'source_text_scores256':scores,'q_source':np.array(q,np.int64),'arm_names':g['arm_names'],'arm_query_index':g['arm_query_index'],'arm_status':np.asarray(g['arm_status']),'arm_boxes_raw6':raw,'arm_boxes_eval6':eval_boxes(raw),'acceptance_mask6':accept,'actual_changed_axes':raw!=boxes[q],'point_support_sha256':g['point_support_sha256'],'provenance':g['provenance'],'point_sample_sha256':c['point_sample_sha256'],'raw_indices_sha256':c['raw_indices_sha256'],'source_cache_sha256':np.frombuffer(bytes.fromhex(sha(cp)),np.uint8),'anchor_image_file_sha256':np.frombuffer(bytes.fromhex(fr['image_sha256']),np.uint8),'anchor_raw_point_file_sha256':np.frombuffer(bytes.fromhex(fr['point_sha256']),np.uint8)}
        name=row['id']+'.npz';np.savez_compressed(a.output/name,**payload);files[name]=sha(a.output/name)
        tr={'id':row['id'],'platform':row['platform'],'scene_PRIVATE':row['scene'],'anchor_frame_alias':alias,'q_source':q,'roles':roles,'wrong_frame_alias':None if wrong is None else wrong['frame_alias'],'source_cache_sha256':sha(cp),'mask_cache_sha256':ms['files'][alias+'.npz'],'anchor_raw_full_points':len(anchor['raw_points']),'geometry_uses_raw_full_points_not_source_resampling':True,'model_coordinate_Drone_z_shift_once':True,'provided_pose_not_primary_input':True,'direct_no_parameter_update':True,'diagnostics':g['diagnostics'],'seconds':time.perf_counter()-tick}
        traces.append(to_json(tr))
        with (a.output/'trace.jsonl').open('a') as f:f.write(json.dumps(traces[-1],allow_nan=False)+'\n')
        if i%10==0 or i+1==len(rows):print(json.dumps({'completed':i+1,'n':len(rows),'elapsed_seconds':time.perf_counter()-start,'single_status':g['arm_status'][2],'multiview_status':g['arm_status'][3]}),flush=True)
    (a.output/'trace.json').write_text(json.dumps(traces,indent=2,allow_nan=False)+'\n')
    for n in ('trace.json','trace.jsonl','registered_plan.json','manifest_input.json'):files[n]=sha(a.output/n)
    seal={'state':'sealed_R12_8arm_independent_instance_multiview_direct_before_GT','n_samples':len(rows),'n_arms':8,'arm_names':ARMS,'source_seal_sha256':sha(source/'SEALED.json'),'mask_seal_sha256':sha(masks/'SEALED.json'),'source_checkpoint_sha256':ss['source_checkpoint_sha256'],'mask_weights_sha256':ms['weights_sha256'],'manifest_sha256':plan['manifest_sha256'],'pairmap_sha256':plan['pairmap_sha256'],'frame_index_sha256':plan['frame_index_sha256'],'plan_sha256':sha(a.plan),'program_sha256':plan['program_sha256'],'program_sha256_base':'workspace_root','all_queries_fixed_original_q0':True,'GT_model_GPU_or_neural_update_executed':False,'clock_unknown_verified_overlapping_view_hypothesis_only':True,'provided_pose_used_for_registration_or_target_fusion':False,'missing_or_negative_cases_preserved':True,'elapsed_seconds':time.perf_counter()-start,'initial_integrity_validation_seconds':integrity_validation_seconds,'whole_runner_elapsed_seconds':time.perf_counter()-entry_start,'runtime':{'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads()},'files':files}
    (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print('R12_GEOMETRY_SEALED',sha(a.output/'SEALED.json'),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

"""R13 fixed-q0 real-data direct assay; no model, GPU, annotation or GT reader."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys,time,hashlib
import numpy as np
D=Path(__file__).resolve().parent;B=D.parents[1]
sys.path[:0]=[str(D),str(D.parent/'round12'),str(D.parent/'round7/experiment')]
from input_io import read,sha,verify_seal,load_case
from geometry_observation import generate_observation_targets,DEFAULT_CONFIG as OBS_CONFIG
from support import extract_fixed_q0_support,DEFAULT_CONFIG as SUPPORT_CONFIG
from multiview_feedback import _geometry
from runner_helpers import to_json,eval_boxes
ARMS=('source','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
def main(a):
 assert sha(a.plan)==a.plan_sha256;p=read(a.plan);assert p['stage']=='R13_fixed_q0_partial_positive_direct' and not p['GT_permitted']
 for rel,h in p['program_sha256'].items():assert sha(B/rel)==h,rel
 for key in ('manifest','cases','frame_index'):assert sha(p[key])==p[key+'_sha256']
 for folder,h in p['capture_seals'].items():assert sha(B/folder/'SEALED.json')==h;verify_seal(B/folder)
 assert p['observation_config']==asdict(OBS_CONFIG) and p['support_config']==asdict(SUPPORT_CONFIG)
 rows=read(p['manifest']);cases=read(p['cases']);index=read(p['frame_index']);assert len(rows)==len(cases)==116
 assert not a.output.exists();a.output.mkdir(parents=True);(a.output/'registered_plan.json').write_bytes(a.plan.read_bytes())
 files={};trace=[];start=time.perf_counter()
 for i,(row,case) in enumerate(zip(rows,cases)):
  tick=time.perf_counter();frame,c,bindings=load_case(row,case,index);q=int(c['query_index']);pool=c['boxes_raw6'][0];src=pool[q]
  support=extract_fixed_q0_support(frame=frame,source6=src,q0=q,caption=row['caption'])
  g=generate_observation_targets(points_support_model=support['points_support_model'],mask_bool=support['mask_bool'],camera_spec=frame['camera'],source6=src,q0=q,segmentation_valid=support['segmentation_valid'],physical_raw_ids=support['physical_raw_ids'])
  boxes=np.tile(src,(5,1));boxes[:4]=g['arm_boxes_raw6'];status=list(g['arm_status']);active=np.zeros((5,6),bool);active[:4]=g['active_mask6'];old=None
  if support['segmentation_valid'] and np.all(src[3:]>0):
   old=_geometry(frame,src,q,support['mask_bool'],frame['raw_points']);boxes[4]=old['arm_boxes_raw6'][1];status.append(old['arm_status'][1]);active[4]=old['acceptance_mask6'][1]
  else:status.append('source_fallback:old_surface_no_compatible_mask_or_valid_source')
  assert boxes.dtype==np.float32 and np.isfinite(boxes).all() and np.array_equal(boxes[0],src)
  assert np.array_equal(boxes[3],src+np.float32(.5)*(boxes[1]-src))
  if np.all(src[3:]>0):assert np.all(boxes[1:4,3:]>=src[3:])
  name=row['id']+'.npz';np.savez_compressed(a.output/name,sample_id=np.array(row['id']),case_alias=np.array(case['case_alias']),cohort=np.array(case['role']),source_boxes_raw6=pool,source_text_scores256=c['text_scores'][0],q_source=np.array(q),arm_names=np.array(ARMS),arm_query_index=np.full(5,q),arm_boxes_raw6=boxes,arm_boxes_eval6=eval_boxes(boxes),arm_status=np.array(status),active_coordinate_mask=active,actual_changed_axes=boxes!=src,B_support_raw_ids=support['physical_raw_ids'],B_support_model_points=support['points_support_model'],selected_mask_index=np.array(-1 if support['selected_mask_index'] is None else support['selected_mask_index']),source_cache_sha256=np.array(bindings['source_cache_sha256']),mask_cache_sha256=np.array(bindings['mask_cache_sha256']))
  files[name]=sha(a.output/name)
  tr={'id':row['id'],'case_alias':case['case_alias'],'cohort':case['role'],'platform':row['platform'],'scene_PRIVATE':row['scene'],'q_source':q,'frame_alias':case['frame_alias'],'bindings':bindings,'support':{k:v for k,v in support.items() if k not in ('mask_bool','points_support_model','points_support_raw')},'inequality':g['diagnostics'],'old_surface':old,'seconds':time.perf_counter()-tick}
  trace.append(to_json(tr))
  if (i+1)%20==0 or i+1==len(rows):print(json.dumps({'completed':i+1,'n':len(rows),'seconds':time.perf_counter()-start}),flush=True)
 (a.output/'TRACE_PRIVATE.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
 for n in ('TRACE_PRIVATE.json','registered_plan.json'):files[n]=sha(a.output/n)
 seal={'state':'sealed_R13_fixedq0_inequality_vs_source_and_surface_direct_no_GT','n':116,'new_n':108,'regression_n':8,'arm_names':ARMS,'files':files,'program_sha256':p['program_sha256'],'plan_sha256':sha(a.plan),'input_manifest_sha256':p['manifest_sha256'],'capture_seals':p['capture_seals'],'all_q_fixed':True,'no_neural_update_or_GPU_or_GT':True,'seconds':time.perf_counter()-start}
 (a.output/'SEALED.json').write_text(json.dumps(seal,indent=2)+'\n');print('SEALED',sha(a.output/'SEALED.json'))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())

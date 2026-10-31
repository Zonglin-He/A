"""R15 synthetic engineering checks only. No files containing real observations.

These verify code/controls and preserve known non-identifiability; no task claim.
"""
from dataclasses import asdict
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round7'))
from camera_projection import CameraSpec
from scene_observation import build_scene_observation,SceneConfig,config_sha256 as scene_config_sha
from joint_geometry import generate_joint_targets,map_scene_to_candidate_groups,JointConfig,config_sha256 as joint_config_sha

def camera(platform='quad'):
 return CameraSpec(platform,np.array([[40.,0.,32.],[0.,40.,32.],[0.,0.,1.]]),np.eye(4),np.zeros(5))
def fixture(platform='quad'):
 x=np.linspace(-.5,.5,5);left=np.array([[a-1.,b,10.] for a in x for b in x]);right=np.array([[a+1.,b,10.] for a in x for b in x]);raw=np.r_[left,right]
 masks=np.zeros((2,64,64),bool);masks[0,30:35,26:31]=True;masks[1,30:35,34:39]=True
 inst={'boxes_xyxy':np.array([[26,30,31,35],[34,30,39,35]],float),'class_ids':np.array([1,1]),'scores':np.array([.8,.9]),'masks_bool':masks,'instance_indices':np.array([10,20])}
 scene=build_scene_observation(raw_points=raw,camera_spec=camera(platform),image_size_wh=(64,64),instances=inst,image_sha256='a'*64)
 z=np.float32(11.8 if platform=='drone' else 10.)
 boxes=np.array([[-1,0,z,1.4,1.4,1],[1,0,z,1.4,1.4,1],[1,0,z,1.4,1.4,1],[0,0,z,4.4,1.4,1],[0,4,z,1,1,1]],np.float32)
 return raw,inst,scene,boxes,np.array([.4,.5,.7,10.,.1])
def graph(mid='M00020'):return {'status':'unique_supported','selected_target_mask_id':mid,'search_complete':True}
def scene_from(raw,inst,camera_spec=None):return build_scene_observation(raw_points=raw,camera_spec=camera_spec or camera(),image_size_wh=(64,64),instances=inst,image_sha256='a'*64)

def checks():
 c={};raw,inst,scene,boxes,scores=fixture()
 assert len(scene['visible_points_model'])==50 and all(o['n_exclusive']==25 for o in scene['objects'])
 assert scene['objects'][0]['mask_id']=='M00010' and scene['objects'][1]['mask_id']=='M00020'
 assert all(o['depth_camera']['defined'] and o['depth_camera']['median']==10 for o in scene['objects'])
 c['fullscene_masks_and_exclusive_camera_depth']=True
 # Camera depth is not shifted model z on Drone.
 _,_,drone,dbox,dscores=fixture('drone')
 assert all(o['depth_camera']['median']==10 for o in drone['objects']) and (drone['visible_points_model'][:,2]==np.float32(11.8)).all()
 assert np.array_equal(drone['visible_raw_points'],scene['visible_raw_points'])
 c['Drone_float32_inplace_z_once_camera_depth_stays_raw_camera_z']=True
 # More raw rows and point permutation cannot multiply unique evidence.
 duplicate=scene_from(np.repeat(raw,3,axis=0),inst);permuted=scene_from(raw[::-1],inst)
 for s in [duplicate,permuted]:
  assert np.array_equal(s['visible_points_model'],scene['visible_points_model']) and np.array_equal(s['point_owner'],scene['point_owner'])
  assert [o['n_exclusive'] for o in s['objects']]==[25,25]
 c['rawduplicates_and_permutation_do_not_multiply_support']=True
 # Global zbuffer runs before masks, so a farther return cannot become visible
 # by applying a mask after ignoring the nearer surface.
 zi={'boxes_xyxy':np.array([[30,30,35,35]]),'class_ids':np.array([1]),'scores':np.array([.9]),'masks_bool':np.ones((1,64,64),bool),'instance_indices':np.array([1])}
 zs=scene_from(np.array([[0.,0.,10.],[0.,0.,20.]]),zi)
 assert len(zs['visible_raw_points'])==1 and zs['depth_camera'][0]==10
 c['global_nearest_depth_before_any_mask']=True
 # Correct continuous FOV/floor pixel endpoints.
 uv=np.array([-.4,0.,63.9,64.]);edge=np.c_[(uv-32.)*.25,np.zeros(4),np.full(4,10.)]
 es=scene_from(edge,zi);assert sorted(es['visible_pixels'][:,0].tolist())==[0,63]
 c['continuousFOV_floor_keeps_right_edge_rejects_negative_roundin']=True
 # Independent masks overlap -> ambiguous ownership; no false foreign evidence.
 overlap={**inst,'masks_bool':np.stack([inst['masks_bool'][0],inst['masks_bool'][0]])}
 os=scene_from(raw,overlap)
 assert (os['point_owner']==-2).sum()==25 and sum(o['n_exclusive'] for o in os['objects'])==0
 c['mask_overlap_not_counted_as_exclusive_positive_or_foreign']=True
 # Whole-cloud mapping, actual q0 giant compatibility does not protect q0.
 mapping=map_scene_to_candidate_groups(scene=scene,boxes_raw6=boxes,text_scores=scores)
 assert mapping['membership'][1]==mapping['membership'][2]
 row=next(r for r in mapping['mask_mappings'] if r['mask_id']=='M00020')
 assert row['canonical_query_index']==2 and mapping['pollution_rejected'][3,1]
 assert mapping['positive_feasible'][3,1] and row['foreign_gate_changed_mapping']
 assert mapping['foreign_exclusive_counts'][3,1]==25 and mapping['foreign_fraction'][3,1]==.5
 out=generate_joint_targets(scene=scene,boxes_raw6=boxes,text_scores=scores,q0=3,graph_match=graph())
 assert out['association_accepted'] and out['q_association']==2 and out['arm_query_index'][0]==3
 assert np.array_equal(out['arm_boxes_raw6'][0],boxes[3]) and np.array_equal(out['arm_boxes_raw6'][1],boxes[2])
 assert np.array_equal(out['arm_boxes_raw6'][4],boxes[2]+np.float32(.5)*(out['arm_boxes_raw6'][2]-boxes[2]))
 c['foreign_object_rejects_giant_q0_no_source_priority_and_duplicate_queries_collapsed']=True
 # Source score changes can alter member readout only, never groups/edges.
 scores2=scores.copy();scores2[1]=100
 m2=map_scene_to_candidate_groups(scene=scene,boxes_raw6=boxes,text_scores=scores2)
 assert np.array_equal(mapping['membership'],m2['membership']) and np.array_equal(mapping['feasible'],m2['feasible'])
 assert next(r for r in m2['mask_mappings'] if r['mask_id']=='M00020')['canonical_query_index']==1
 c['source_scores_only_choose_within_confirmed_group']=True
 # Same visible support, incompatible amodal alternatives remain ambiguity.
 amb=np.array([boxes[1],boxes[1]],np.float32);amb[1,5]=3
 am=map_scene_to_candidate_groups(scene=scene,boxes_raw6=amb,text_scores=np.array([1.,100.]))
 assert next(r for r in am['mask_mappings'] if r['mask_id']=='M00020')['status']=='NA_multiple_compatible_groups'
 c['no_score_arbitration_of_distinct_positive_groups']=True
 # Single detected object and no foreign evidence does NOT certify giant-box
 # correctness; retain this precise known limitation rather than fake a fix.
 one={k:(v[1:2].copy() if isinstance(v,np.ndarray) else v) for k,v in inst.items()};one_scene=scene_from(raw,one)
 gm=map_scene_to_candidate_groups(scene=one_scene,boxes_raw6=boxes[3:4],text_scores=np.array([1.]))
 assert gm['mask_mappings'][0]['canonical_query_index']==0 and gm['diagnostics']['single_mask_or_missing_foreign_points_cannot_exclude_giant_box']
 c['single_mask_giant_box_limitation_explicit_not_resolved']=True
 # Positive partial observations already inside source: no shrink or forced shift.
 assert np.array_equal(out['arm_boxes_raw6'][1:],np.tile(boxes[2],(4,1)))
 assert np.array_equal(out['geometry']['point_support_sha256'][1:3],np.repeat(out['geometry']['point_support_sha256'][1],2))
 assert np.array_equal(out['geometry']['acceptance_mask6'][1],out['geometry']['acceptance_mask6'][2])
 c['partial_contained_observations_stay_source_and_full_noRGB_share_gate_support']=True
 # A mapping can succeed while the selected anchor's projection corners are
 # behind the camera. All geometric failure deliveries must keep q1/anchor,
 # never silently roll back to the original q0 or select another query.
 large_depth=np.array([boxes[0],boxes[1]],np.float32);large_depth[1,5]=30.
 gf=generate_joint_targets(scene=scene,boxes_raw6=large_depth,text_scores=np.array([10.,1.]),q0=0,graph_match=graph())
 assert gf['association_accepted'] and gf['q_association']==1
 assert np.all(gf['arm_query_index'][1:]==1)
 assert np.array_equal(gf['arm_boxes_raw6'][1:],np.tile(large_depth[1],(4,1)))
 assert gf['geometry']['diagnostics']['gate_passed'] is False and 'projection' in gf['arm_status'][2]
 c['geometry_failure_keeps_same_associated_q1_anchor_not_originalq0']=True
 # Near point violations can produce a bounded increment after mapping. No
 # point mean or visible extent is assigned as an amodal target.
 shifted=boxes.copy();shifted[2,0]=1.08;shifted[2,3]=1.
 so=generate_joint_targets(scene=scene,boxes_raw6=shifted,text_scores=scores,q0=3,graph_match=graph())
 assert so['association_accepted'] and so['q_association']==2
 assert (so['arm_boxes_raw6'][2,3:]>=shifted[2,3:]).all() and not np.array_equal(so['arm_boxes_raw6'][2],shifted[2])
 assert so['geometry']['diagnostics']['solver']['direct_full']['final_total']<=so['geometry']['diagnostics']['solver']['direct_full']['initial_total']+1e-7
 c['bounded_geometry_increment_after_unique_mapping_without_visibleface_equality']=True
 # Exact pollution threshold: 3/30=.10 is retained; 4/30 is rejected.
 cloud=np.array([[x,y,10.] for x in np.linspace(-.625,.625,6) for y in np.linspace(-.5,.5,5)])
 pixels=np.floor(cloud[:,:2]*4.+32.).astype(int)
 for foreign_n,rejected in [(3,False),(4,True)]:
  masks=np.zeros((2,64,64),bool)
  for j,(u,v) in enumerate(pixels):masks[int(j>=30-foreign_n),v,u]=True
  ii={**inst,'masks_bool':masks}
  ts=scene_from(cloud,ii);tb=np.array([[0,0,10,2,2,1]],np.float32)
  tm=map_scene_to_candidate_groups(scene=ts,boxes_raw6=tb,text_scores=np.ones(1))
  assert tm['foreign_exclusive_counts'][0,0]==foreign_n and bool(tm['pollution_rejected'][0,0]) is rejected
 c['foreign_threshold_three_points_and_strict_fraction_above_tenpercent']=True
 # Complete-link cannot chain A-B and B-C into a group if A-C is incompatible.
 chain=np.array([[0,0,10,1.4,1.4,1],[.3,0,10,1.4,1.4,1],[.6,0,10,1.4,1.4,1]],np.float32)
 cm=map_scene_to_candidate_groups(scene=scene,boxes_raw6=chain,text_scores=np.array([3.,2.,1.]))
 assert cm['membership'][0]!=cm['membership'][2]
 c['complete_link_no_transitive_neighbor_chain']=True
 assert scene['objects'][0]['mask_extent_norm2'].shape==(2,)
 assert scene['objects'][0]['mask_bbox_norm4'].shape==(4,)
 assert scene['objects'][0]['mask_centroid_defined'] and scene['objects'][0]['mask_bbox_defined']
 c['explicit_centroid_widthheight_and_bbox_fields_no_shape_alias']=True
 # Fewer than8 exclusive observations cannot map or feed geometry.
 small=scene_from(raw[:4],inst);sm=generate_joint_targets(scene=small,boxes_raw6=boxes,text_scores=scores,q0=0,graph_match=graph('M00010'))
 assert not sm['association_accepted'] and np.array_equal(sm['arm_boxes_raw6'],np.tile(boxes[0],(5,1)))
 c['min8_same_support_geometry_gate_failclosed']=True
 for g in [{'status':'ambiguous','selected_target_mask_id':None,'search_complete':True},{'status':'unique_supported','selected_target_mask_id':'M00020','search_complete':False}]:
  r=generate_joint_targets(scene=scene,boxes_raw6=boxes,text_scores=scores,q0=0,graph_match=g);assert not r['association_accepted'] and np.array_equal(r['arm_boxes_raw6'],np.tile(boxes[0],(5,1)))
 c['semantic_unknown_or_incomplete_not_forced_unique']=True
 proposal={'status':'proposal_control_unique','selected_target_mask_id':'M00020','search_complete':True,'relation_check':False}
 rejected=generate_joint_targets(scene=scene,boxes_raw6=boxes,text_scores=scores,q0=3,graph_match=proposal)
 accepted=generate_joint_targets(scene=scene,boxes_raw6=boxes,text_scores=scores,q0=3,graph_match=proposal,semantic_branch='proposal_control')
 assert not rejected['association_accepted'] and accepted['association_accepted']
 assert accepted['diagnostics']['semantic_original_status']=='proposal_control_unique' and accepted['diagnostics']['semantic_branch']=='proposal_control'
 assert np.array_equal(accepted['arm_boxes_raw6'],out['arm_boxes_raw6'])
 c['proposal_control_explicit_branch_same_kernel_without_status_retag']=True
 # Strict no annotation keys accepted; no optional sourceROI/GT mask override.
 bad={**inst,'GT_bbox':np.ones(7)}
 try:scene_from(raw,bad)
 except ValueError:pass
 else:raise AssertionError('GT/annotation field accepted')
 try:build_scene_observation(raw_points=raw,camera_spec=camera(),image_size_wh=(64,64),instances=inst,image_sha256='a'*64,q0=0)
 except TypeError:pass
 else:raise AssertionError('q0 accepted by whole-scene interface')
 c['GT_injection_and_q0ROI_interface_rejected']=True
 # Empty cloud, zero independent masks, invalid dimensions never manufacture evidence.
 zero={k:(np.zeros((0,64,64),bool) if k=='masks_bool' else v[:0]) for k,v in inst.items()}
 empty=scene_from(np.empty((0,3)),zero);em=map_scene_to_candidate_groups(scene=empty,boxes_raw6=boxes,text_scores=scores)
 assert not len(em['mask_mappings']) and not em['feasible'].any()
 c['empty_scene_stays_undefined']=True
 cfg=JointConfig();base=Path(__file__).resolve().parent
 return {'state':'PASS_R15_PURE_SYNTHETIC_SCENE_MAPPING_AND_GEOMETRY_ENGINEERING_NOT_TASK_EFFECT',
  'checks':c,'check_count':len(c),'scene_config':asdict(SceneConfig()),'scene_config_sha256':scene_config_sha(),
  'joint_config':asdict(cfg),'joint_config_sha256':joint_config_sha(cfg),
  'source_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [base/'scene_observation.py',base/'joint_geometry.py',base/'check_geometry_cpu.py']},
  'real_GT_scores_images_points_model_or_GPU_accessed':False,'known_singlemask_giant_limit_preserved':True}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=checks()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'state':r['state'],'checks':r['check_count']}))

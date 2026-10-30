"""Independent synthetic R11 A/B dataflow and missing-status checks only.
No target data/labels, model, GPU, weights, images or cache file input.
"""
from pathlib import Path
import sys,copy,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sys.path[:0]=[str(B/'crossplatform/round11'),str(B/'crossplatform/round7')]
import reliability as R
from camera_projection import CameraSpec
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 # Exact duplicates and coincident distinct raw returns cannot cross A/B roles.
 rng=np.random.default_rng(713);p=rng.uniform(-1,1,(300,3)).astype(np.float32);raw=np.arange(len(p),dtype=np.int64)
 pp=np.r_[p,p[:20],p[:5]];rr=np.r_[raw,raw[:20],np.arange(300,305)]
 q,ids,role,diag=R.physical_partitions(pp,rr,'synthetic_case')
 order=rng.permutation(len(pp));q2,ids2,role2,_=R.physical_partitions(pp[order],rr[order],'synthetic_case')
 assert np.array_equal(q,q2) and np.array_equal(ids,ids2) and np.array_equal(role,role2)
 assert len(q)==300 and not set(ids[role==0])&set(ids[role==1]) and diag['dedup']['distinct_rawindices_same_XYZ']==5
 # Actual pure APIs plus spies establish each normal/seed call sees its own role.
 uv=np.linspace(-.8,.8,17);x,y=np.meshgrid(uv,uv);plane=np.c_[x.ravel(),y.ravel(),np.full(x.size,4.)].astype(np.float32);raw=np.arange(len(plane),dtype=np.int64)
 rgb=np.zeros((100,120,3),np.uint8)+30;rgb[29:72,39:82]=[180,90,50]
 cam=CameraSpec('quad',np.array([[100,0,60],[0,100,50],[0,0,1]],float),np.eye(4),np.zeros(5));source=np.array([0,0,5,2,2,2],np.float32);target=source.copy();target[2]-=.05
 base={'acceptance_mask6':np.tile(np.array([False,False,True,False,False,False]),(4,1)),'arm_boxes_raw6':np.stack([source,target,target,target]),'diagnostics':{}}
 seed_calls=[];normal_calls=[];orig_seed=R.make_rgb_support;orig_normal=R.normals_from_fit_cloud
 def seed_spy(**kw):seed_calls.append(kw['points_fit'].copy());return orig_seed(**kw)
 def normal_spy(points,query_ids,cfg):normal_calls.append(points.copy());return orig_normal(points,query_ids,cfg)
 R.make_rgb_support=seed_spy;R.normals_from_fit_cloud=normal_spy
 try:out=R.compute_reliability(sample_id='synthetic_plane',points=plane,raw_indices=raw,rgb=rgb,source_box6=source,camera_spec=cam,base_mask2d=np.ones((100,120),bool),base_result=base)
 finally:R.make_rgb_support=orig_seed;R.normals_from_fit_cloud=orig_normal
 canon,_,parity,_=R.physical_partitions(plane,raw,'synthetic_plane')
 assert len(seed_calls)==2 and len(normal_calls)==3
 assert np.array_equal(seed_calls[0],canon[parity==0]) and np.array_equal(seed_calls[1],canon[parity==1])
 assert np.array_equal(normal_calls[0],canon[parity==0]) and np.array_equal(normal_calls[1],canon[parity==1]) and np.array_equal(normal_calls[2],canon)
 # Shared whole-source baseline graph is allowed only in the base denominator;
 # it is not used as either A/B normal pool. Correlation remains explicit.
 assert out['features']['RGB_and_source_shared_across_partitions'] and out['features']['mask_stability_alone_is_not_identity_evidence']
 zero=np.zeros((8,8),bool);undefined=R.mask_stability(zero,zero,3.);assert undefined['defined'] is False and undefined['symmetric_boundary_distance_px'] is None
 accepted=np.array([False,False,True,False,False,False]);changed=accepted.copy()
 for status in ('association_undefined','association_conflicted'):
  o=copy.deepcopy(out);o['association_status']=status;o=R.apply_correspondence_gate(o,accepted,changed)
  assert all(v==0 for v in o['face_weights'].values()) and o['RGB_weight']==0 and not np.any(o['axis_reliability6'])
  assert o['case_reliability'] is None if status=='association_undefined' else o['case_reliability']==0
  if status=='association_undefined':assert not np.any(o['axis_reliability_defined6']) and o['RGB_weight_defined'] is False
 # Fixed coverage can select NA only as an explicit hash tie, not imputed score.
 units=[{'id':f'fixture{i}','platform':'drone','eligible':True,'scores':{'reliability':None}} for i in range(3)]
 sel=R.select_exact_half(units,'reliability');assert len(sel['selected_ids'])==2 and sel['by_platform']['drone']['undefined_score_count']==3
 report={'state':'independent_synthetic_R11_AB_dataflow_and_undefined_semantics_pass','program_sha256':sha(__file__),'reliability_sha256':sha(B/'crossplatform/round11/reliability.py'),'physical_partition_permutation_duplicate_coincident_grouping_pass':True,'partition_seed_and_normal_pools_spy_exact_pass':True,'base_whole_source_graph_only_shared_denominator_not_AB_graph':True,'association_undefined_weightszero_and_caseNA_conflicted_defined0':True,'empty_masks_undefined_not_evidence0':True,'fixedcoverage_NA_hashselection_not_defined_confidence':True,'no_actual_target_media_points_GT_cache_model_or_GPU':True,'synthetic_not_method_quality_or_identity_evidence':True}
 p=D/'RELIABILITY_INDEPENDENT_CPU_FIXTURE.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()

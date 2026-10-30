"""Synthetic CPU engineering fixtures only. No real RGB/P/model/GT/cache access."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[3];sys.path.insert(0,str(B/'crossplatform/round10'));sys.path.insert(0,str(B/'crossplatform/round7'))
from rgb_support import segment_q0_support,make_rgb_support,DEFAULT_CONFIG,config_sha256,split_point_observations,frame_split_seed,visible_point_indices
from camera_projection import CameraSpec

def main():
 image=np.full((160,200,3),45,np.uint8);image[45:116,60:145]=[210,55,35]
 cuv=np.array([[55,40],[150,40],[55,120],[150,120],[58,43],[147,43],[58,117],[147,117]],float);cd=np.ones(8)*5
 x,y=np.meshgrid(np.linspace(70,130,8),np.linspace(55,105,6));uv=np.c_[x.ravel(),y.ravel()];depth=np.ones(len(uv))*4.5;crop=np.ones(len(uv),bool)
 a=segment_q0_support(image,box_corners_uv=cuv,box_corner_depth=cd,fit_uv=uv,fit_depth=depth,fit_source_crop_mask=crop,seed=123)
 b=segment_q0_support(image,box_corners_uv=cuv,box_corner_depth=cd,fit_uv=uv,fit_depth=depth,fit_source_crop_mask=crop,seed=123)
 assert a['valid'] and np.array_equal(a['mask2d'],b['mask2d']) and a['mask2d'].dtype==bool
 target=np.zeros((160,200),bool);target[45:116,60:145]=True;iou=float((a['mask2d']&target).sum()/(a['mask2d']|target).sum());assert iou>.95,iou
 color=segment_q0_support(image[:,:,::-1].copy(),box_corners_uv=cuv,box_corner_depth=cd,fit_uv=uv,fit_depth=depth,fit_source_crop_mask=crop,seed=123);assert color['valid'] and np.array_equal(a['mask2d'],color['mask2d'])
 no_seed=segment_q0_support(image,box_corners_uv=cuv,box_corner_depth=cd,fit_uv=uv,fit_depth=depth,fit_source_crop_mask=np.zeros(len(uv),bool),seed=123);assert not no_seed['valid'] and not no_seed['mask2d'].any() and not no_seed['diagnostics']['fake_all_foreground_fallback']
 behind=segment_q0_support(image,box_corners_uv=cuv,box_corner_depth=np.array([-.1]+[5]*7),fit_uv=uv,fit_depth=depth,fit_source_crop_mask=crop,seed=123);assert not behind['valid']
 tiny=segment_q0_support(image,box_corners_uv=np.zeros((8,2)),box_corner_depth=cd,fit_uv=uv,fit_depth=depth,fit_source_crop_mask=crop,seed=123);assert not tiny['valid']
 # Negative tiny ray must not round into pixel0; nearest-depth/XYZ tie stable.
 points=np.array([[0.,0.,4.],[1.,0.,4.],[0.,0.,3.],[0.,0.,2.]])
 test_uv=np.array([[-.01,4.],[4.2,4.],[4.2,4.],[4.2,4.]])
 ids=visible_point_indices(points,test_uv,points[:,2],(10,10));assert np.array_equal(ids,[3])
 rng=np.random.default_rng(42);p=rng.normal(size=(100,3)).astype(np.float32);p=np.r_[p,p[:15],[[0.,-0.,0.],[0.,0.,0.]]].astype(np.float32);seed=frame_split_seed('synthetic_case');split=split_point_observations(p,seed);assert np.array_equal(split['fit_mask'][:15],split['fit_mask'][100:115]) and split['fit_mask'][-1]==split['fit_mask'][-2]
 perm=rng.permutation(len(p));permuted=split_point_observations(p[perm],seed);assert np.array_equal(permuted['fit_mask'],split['fit_mask'][perm]);assert np.all(split['fit_mask']^split['eval_mask'])
 camera=CameraSpec('quad',np.array([[100.,0,100.],[0,100.,80.],[0,0,1.]]),np.eye(4),np.zeros(4))
 xx,yy=np.meshgrid(np.linspace(-1.5,1.5,8),np.linspace(-1,1,6));pc=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,4.5)]
 wrapped=make_rgb_support(rgb=image,points_fit=pc,source_box6=[0,0,5,4,3,1],camera_spec=camera);assert wrapped['valid'] and wrapped['mask2d'].shape==(160,200)
 receipt={'state':'synthetic_CPU_GrabCut_projection_split_engineering_pass_no_actual_data','rgb_module_sha256':hashlib.sha256((B/'crossplatform/round10/rgb_support.py').read_bytes()).hexdigest(),'fixture_program_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'config_sha256':config_sha256(),'opencv_version':a['diagnostics']['opencv_version'],'synthetic_rectangle_mask_IoU_only_not_task_result':iou,'fixed_seed_repeat_bitexact':True,'synthetic_channel_permutation_not_color_hard_rule':True,'no_seed_returns_invalid_zero_mask_no_fakefg':True,'nearplane_and_degenerateROI_explicit_failure':True,'continuousFOV_negative_ray_rejected_nearestdepth_correct':True,'duplicate_signedzero_and_permutation_invariant_pointsplit':True,'wrapper_CameraSpec_projection_API_pass':True,'no_actual_RGB_points_metadata_GT_index_sidecar_scores_cache_model_or_GPU_access':True,'scope':'Synthetic API/geometry/algorithm mechanics only; no taskquality/reliability evidence.'}
 out=Path(__file__).with_name('RGB_SUPPORT_CPU_FIXTURE.json');assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

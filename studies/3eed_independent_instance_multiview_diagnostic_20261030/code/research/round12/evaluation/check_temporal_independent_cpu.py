"""R12 finite independent synthetic review; no actual images/P/GT/models."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
B=Path(__file__).resolve().parents[3];D=B/'crossplatform/round12';O=Path(__file__).resolve().parent;sys.path.insert(0,str(D))
import temporal_association as t
from camera_projection import CameraSpec,raw_to_model_points

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 lineB=np.r_[np.c_[np.linspace(0,10,31),np.zeros((31,2))],[[5,1,0],[3,-1,0],[8,1,0]]];lineA=lineB.copy();lineA[-3:,1]*=1.7
 fixed=t.estimate_from_correspondences(points_A=lineA,points_B=lineB,seed=0)
 assert not fixed['accepted'] and fixed['T_raw_B_to_A'] is None and fixed['status']=='degenerate_fit_consensus' and fixed['n_fit']==27 and fixed['n_holdout']==7
 rng=np.random.default_rng(381);b=rng.uniform(-4,4,(180,3));b[:,2]+=10;theta=.173;c,s=np.cos(theta),np.sin(theta);T=np.eye(4);T[:3,:3]=[[1,0,0],[0,c,-s],[0,s,c]];T[:3,3]=[.23,-.31,.12];a=t.transform_points(b,T)
 ordinary=t.estimate_from_correspondences(points_A=a,points_B=b,seed=9);assert ordinary['accepted'] and np.max(abs(ordinary['T_raw_B_to_A']-T))<1e-12
 wrong=np.eye(4);wrong[:3,3]=[100,-23,55];pose=t.estimate_from_correspondences(points_A=a,points_B=b,seed=9,provided_pose_A=wrong,provided_pose_B=np.eye(4));assert np.array_equal(ordinary['T_raw_B_to_A'],pose['T_raw_B_to_A']) and pose['accepted']==ordinary['accepted']
 h=t.spatial_split(b,9);bad=a.copy();bad[h]+=[2,3,-1];failed=t.estimate_from_correspondences(points_A=bad,points_B=b,seed=9);assert not failed['accepted'] and np.array_equal(failed['T_raw_B_to_A'],ordinary['T_raw_B_to_A'])
 cells=np.floor(b/.5).astype(int);assert not {tuple(x) for x in cells[h]}&{tuple(x) for x in cells[~h]}
 K=np.array([[100.,0,64],[0,100.,64],[0,0,1.]]);camera=CameraSpec('drone',K,np.eye(4),np.zeros(5));Tm=t.raw_pair_to_model_transform(T,camera,camera);left=raw_to_model_points(a,'drone');right=t.transform_points(raw_to_model_points(b,'drone'),Tm);assert np.max(abs(left-right))<1e-12
 assert np.max(abs(left-t.transform_points(raw_to_model_points(b,'drone'),T)))>.1
 assert t.check_static_object_support(points_A=a,points_B=b,T_raw_B_to_A=T)['accepted'];assert not t.check_static_object_support(points_A=a+[1,0,0],points_B=b,T_raw_B_to_A=T)['accepted']
 lock=json.loads((D/'IMPLEMENTATION_LOCK.json').read_text());assert all(sha(B/k)==v for k,v in lock['files_sha256'].items())
 receipt={'state':'PASS_R12_INDEPENDENT_CPU_SYNTHETIC_INTERFACE_CHECKS','module_sha256':sha(D/'temporal_association.py'),'implementation_lock_sha256':sha(D/'IMPLEMENTATION_LOCK.json'),'reviewer_fixture_sha256':sha(Path(__file__)),'degenerate_final_consensus_NA_repair_verified':True,'raw_B_to_A_known_transform_verified':True,'provided_pose_changes_do_not_change_estimate_or_acceptance':True,'holdout_only_qualifies_rejects_same_fitted_T':True,'B_spatial_voxel_fit_holdout_disjoint':True,'Drone_model_shift_conjugation_verified_under_nonZ_rotation':True,'moving_wrong_support_rejected_without_refit':True,'no_actual_data_images_points_GT_model_GPU_or_scoring':True}
 (O/'TEMPORAL_INDEPENDENT_CPU_FIXTURE.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()

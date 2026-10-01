import numpy as np
from scripts.score_native_support_fig1_v1 import component_scores
from vg_tta.tastvg_paper48_metrics_v1 import official_functions

def test_fixed_support_missing_frames_and_branch_isolation():
 p=dict(box_frame_ids=[2,4],boxes=[[[0,0,1,1],[0,0,1,1]]]*6,spatial_valid=[True]*6,temporal_valid=[True]*6,intervals=[[2,5],[0,6],[0,1],[1,6],[3,4],[4,5]])
 truth={i:[0,0,10,10] for i in range(6)};r=component_scores(p,truth,[0,6],10,10,official_functions())
 np.testing.assert_allclose(r['spatial'],[.5]*6);assert r['temporal'][1]==1 and r['temporal'][0]==.5
 assert r['max_kernel_error']<1e-10

def test_degenerate_frame_not_entire_tube():
 p=dict(box_frame_ids=[0,1,2],boxes=[[[0,0,1,1],[0,0,0,0],[0,0,1,1]]]*6,spatial_valid=[True]*6,temporal_valid=[True]*6,intervals=[[0,3]]*6)
 r=component_scores(p,{i:[0,0,10,10] for i in range(3)},[0,3],10,10,official_functions());np.testing.assert_allclose(r['spatial'],[2/3]*6);assert r['invalid_interpolated_box_frames']==[1]*6

def test_missing_native_format_is_zero_retained_slot():
 p=dict(box_frame_ids=[],boxes=[[]]*6,spatial_valid=[False]*6,temporal_valid=[False]*6,intervals=[None]*6)
 r=component_scores(p,{0:[0,0,10,10]},[0,1],10,10,official_functions());assert r['spatial']==r['temporal']==[0.]*6

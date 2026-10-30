"""Synthetic fixtures only, no model/data/sourceVALprediction access."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from core import HEADS,MODES,score_arrays,validate_boxes,native_top,metrics,independent_selected_iou
from batch_yaw_iou import many_iou
R=Path(__file__).resolve().parent

def expect_failure(fn):
 try:fn()
 except (ValueError,AssertionError):return True
 raise AssertionError('Invalidinput didnotfail')
def main():
 gt=np.array([0,0,0,2,2,2,0],np.float32);b=np.tile([100,0,0,2,2,2],(2,256,1)).astype(np.float32);b[:,0]=[0,0,0,1,1,2];b[:,1]=[0,0,0,1,2,2];b[:,6]=gt[:6];z={'boxes_raw':b.copy(),'boxes':b.copy(),'head_names':np.array(HEADS)}
 for mode in MODES:
  z[mode]=np.tile((1-np.arange(256)*.001).astype(np.float32),(2,1));z[mode+'_top10']=np.tile(np.arange(10,dtype=np.int64),(2,1))
 # A genuine native nonstable tied order changes selectedquery; never argsort it on CPU.
 for mode in ['official_bbs','text_bbs']:
  z[mode][:]=1;z[mode+'_top10'][:]=[6,0,1,2,3,4,5,7,8,9]
 arms,values,ind,bad=score_arrays(z,gt);a=arms['official_full_last_bbf'];assert a['iou']==.25 and not a['topk']['1']['acc25'];assert a['topk']['5']['acc25'] and not a['topk']['5']['acc50'] and a['topk']['10']['acc50'];assert arms['official_full_last_bbs']['query']==6 and arms['official_full_last_bbs']['iou']==1;assert ind['strict_threshold_booleans_checked']==16
 raw=b.copy();raw[0,3,3]=-1;raw[1,4,4]=0;cl=raw.copy();cl[...,3:6]=np.maximum(cl[...,3:6],np.float32(1e-6));bb=validate_boxes(raw,cl);assert bb['raw_negative_candidates_by_head']==[1,0] and bb['raw_nonpositive_candidates_by_head']==[1,1]
 assert expect_failure(lambda:validate_boxes(raw,b));badtop=np.arange(10);badtop[2]=1;assert expect_failure(lambda:native_top(z['official_bbf'][0],badtop));assert expect_failure(lambda:native_top(np.arange(256),np.arange(10)));assert expect_failure(lambda:metrics(np.full(256,np.nan),z['official_bbf'][0],np.arange(10)))
 rng=np.random.default_rng(20261030);maximum=0.;decisions=0
 for i in range(128):
  center=rng.normal(0,10,3);target=np.r_[center,rng.uniform(.3,6,3),rng.uniform(-np.pi,np.pi)].astype(np.float32).astype(np.float64);pred=np.r_[center+rng.normal(0,1.5,3),rng.uniform(.3,6,3)].astype(np.float32).astype(np.float64);primary=float(many_iou(np.r_[pred,0][None],target)[0]);independent,_=independent_selected_iou(pred,target);maximum=max(maximum,abs(primary-independent));assert abs(primary-independent)<=1e-10
  for t in [.25,.5]:assert (primary>t)==(independent>t);decisions+=1
 # Float32 source input must be promoted before volume product, not only corner construction.
 target=np.array([0,0,0,4.37,1.93,2.11,.34],np.float32);pred=np.array([.1,.2,.3,3.13,2.49,1.57],np.float32);x=independent_selected_iou(pred,target)[0];y=independent_selected_iou(pred.astype(float),target.astype(float))[0];assert x==y
 receipt={'state':'all_synthetic_contracts_pass','strict25_50_andTop1_5_10_256':True,'nativeGPUtieorder_preserved':True,'raw_negative_zero_clamp_retained':True,'malformed_inputs_fail_not_skip':True,'readout_and_head_axis_contract':True,'independent_general3D_vs_primary2D_random_cases':128,'max_random_absdifference':maximum,'strict_random_threshold_booleans':decisions,'sourcefloat32_promoted_before_volume_product':True,'dataset_GT_or_actual_model_predictions_read':False,'GPU_or_model_or_training_calls':False,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'core.py',R/'score_source_val.py',Path(__file__),R/'batch_yaw_iou.py',R/'obb_iou.py']}}
 (R/'SYNTHETIC_FIXTURES.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

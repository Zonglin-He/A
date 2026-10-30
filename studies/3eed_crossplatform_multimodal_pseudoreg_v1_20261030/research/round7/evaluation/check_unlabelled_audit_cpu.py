"""Synthetic trace-contract fixture only, not actual gradient/model evidence."""
from pathlib import Path
import copy,json
import numpy as np
import score_crossdomain as S
import audit_unlabelled_predictions as U

def main():
 source=np.array([0.,0.,5.,2.,2.,2.],np.float32);pool=np.tile(source,(256,1));scores=1-np.arange(256)*.001;arms=np.tile(source,(16,1));targets=np.tile(source,(3,1));targets[0,0]=.1;w=np.zeros((3,6));w[0,0]=1
 plan={'alpha':.25,'beta_metres':.1};trace={'query_index':0,'feature_norm_squared_plus_bias':{'center':1.,'size':1.},'lr':.025,'modes':{}}
 for mi,mode in enumerate(S.MODES):
  steps={}
  for k in (1,3):
   ts=[];current=source.copy()
   for j in range(k if mi==0 else 0):
    before=current.copy();g=(current-targets[mi])/.1;current=current-.025*g
    delta={p:0. for p in U.PARAMETERS};delta['prediction_heads.5.center_residual_head.net.8.bias']=float(abs(current[0]-source[0]))
    ts.append({'step':j+1,'selected_box_before':before.tolist(),'selected_box_after':current.tolist(),'loss':float(abs(g[0])),'parameter_gradient_norm':float(abs(g[0])),'nonzero_gradient_parameter_tensors':int(abs(g[0])>0),'clip_enabled':False,'clip_factor':1.,'weighted_SmoothL1_box_gradient6':g.tolist(),'parameter_delta_max_by_name':delta,'output_space_replay_max_abs_error':0.})
   steps[str(k)]={'trace':ts,'feature_norms':{'center_self_kernel_with_bias':1.,'size_self_kernel_with_bias':1.},'spillover':{}}
   for kind in ('param'+str(k),'outputspace'+str(k)):arms[U.NAMES.index(mode+'__'+kind)]=current
  arms[U.NAMES.index(mode+'__direct_G')]=targets[mi]
  trace['modes'][mode]={'reason':'synthetic_nonzero' if mi==0 else 'synthetic_zero_support','steps':steps}
 a={'names':list(U.NAMES),'q':0,'source_boxes_raw6':pool,'source_boxes_eval6':S.evaluation_box(pool),'source_native_top10':np.arange(10),'mode_names_list':list(S.MODES),'pseudo_target6':targets,'pseudo_reliability6':w,'arm_boxes_raw6':arms}
 result=U.audit_sample(a,trace,plan);assert result['modes']['rgb_text']['executed_parameter_steps_total_one_plus_three']==4 and result['modes']['rgb_text']['nonzero_parameter_update_runs']==2;assert result['modes']['pc_only']['identity_w0']
 bad=copy.deepcopy(a);bad['arm_boxes_raw6'][U.NAMES.index('rgb_text__param3'),0]=.5;bad['arm_boxes_raw6'][U.NAMES.index('rgb_text__outputspace3'),0]=.5
 try:U.audit_sample(bad,trace,plan);raise AssertionError('Out of interval accepted')
 except AssertionError as error:assert 'outside source-to-pseudo interval' in str(error)
 out=Path(__file__).resolve().parent;receipt={'state':'pure_synthetic_unlabelled_audit_contract_pass','auditor_sha256':S.sha(Path(U.__file__)),'fixture_sha256':S.sha(Path(__file__)),'synthetic_normalized_alpha_point25_interval_checked':True,'synthetic_sizew0_identity_and_one_three_reset_checked':True,'out_of_interval_selected_update_rejected':True,'actual_parameter_model_gradient_evidence':False,'no_real_model_GPU_or_target_GT_data_access':True};S.dump(out/'SYNTHETIC_UNLABELLED_AUDIT_FIXTURE.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()

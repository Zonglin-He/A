"""Pure-NumPy checks for native-score units and q0-anchored raw memory."""
import json,numpy as np
from common36 import norm
from coordinate_projection32 import project_group_preferences,synthetic_checks

def main():
 s0=np.array([-.1,-2.,-3.]);alpha=np.array([.2,.5,0.]);r0=np.array([1.,2.,3.]);r1=r0+np.array([.3,-.2,4.]);base=s0-alpha*r0
 assert np.allclose(s0+alpha*(r1-r0),base+alpha*r1);assert np.array_equal(s0+alpha*(r0-r0),s0)
 pr=project_group_preferences(s0,[([1],[0])],margin=.1,q0=0);assert pr['raw_query_correction'][2]==0.;assert pr['projected_scores'][1]-pr['projected_scores'][0]>=.1-1e-12
 raw=np.array([2.,1.,-.4]);q0=1;assert np.allclose((raw+19)-(raw+19)[q0],raw-raw[q0]);q=np.array([[1.,0.],[0.,2.]]);assert np.allclose(norm(q),norm(q*3))
 result={'state':'PASSED','checks':['native_residual_delta_equals_actual_base_plus_rank','source_initialization_exact','unjudged_query_not_new_negative','transferred_raw_anchored_current_q0_gauge_invariant','positive_scale_key_normalization'],'QP_checks':synthetic_checks(),'no_model_or_GPU_or_real_data':True};print(json.dumps(result,indent=2))
if __name__=='__main__':main()

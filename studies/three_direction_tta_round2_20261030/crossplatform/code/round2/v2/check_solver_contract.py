"""Analytic zero-bridge, mask and optimizer-function-space checks only."""
from pathlib import Path
import json,sys
import numpy as np
import torch
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));from adapter_solver import run,CONFIG
def main():
    rng=np.random.default_rng(20261030);features=rng.normal(size=(8,288)).astype('float32');boxes=np.tile([0,0,0,4,2,2],(8,1)).astype('float32');scores=np.linspace(.8,.1,8).astype('float32');target=np.c_[boxes,np.zeros(8)];target[0,:3]+=[.04,.02,0];target[0,6]=.05;t={'query_indices':np.arange(8),'accepted':np.array([True]+[False]*7),'target_boxes7':target,'logscore_bonus':np.zeros(8)};rngbefore=torch.get_rng_state().clone()
    constrained,trace=run(features,boxes,scores,t,'constrained');adam,at=run(features,boxes,scores,t,'adam');control,ct=run(features,boxes,scores,t,'constrained',control=True)
    assert torch.equal(rngbefore,torch.get_rng_state());assert np.array_equal(constrained[0]['box7'],np.r_[boxes[0],np.float32(0)]);assert all(np.array_equal(v['box7'],control[0]['box7']) for v in control.values()) and all(x['parameter_delta_l2']==0 for x in ct)
    for x in trace:
        assert x['loss_after']<=x['loss_before']+CONFIG['objective_tolerance'];assert x['max_center_step_m']<=CONFIG['center_step_m']+1e-6;assert x['max_center_total_m']<=CONFIG['center_total_m']+1e-6;assert x['max_yaw_step_radians']<=CONFIG['yaw_step_radians']+1e-6;assert x['max_yaw_total_radians']<=CONFIG['yaw_total_radians']+1e-6;assert x['max_logscore_step']<=CONFIG['logscore_step']+1e-6;assert x['max_logscore_total']<=CONFIG['logscore_total']+1e-6;assert x['unaccepted_output_exact_source']
    assert trace[-1]['parameter_delta_l2']>0;assert at[0]['max_yaw_step_radians']>CONFIG['yaw_step_radians'];assert at[0]['loss_after']>at[0]['loss_before']
    result={'scope':'synthetic optimizer engineering, not dataset metric','zero_bridge_bitexact':True,'source_control3_bitexact':True,'unaccepted_queries_exact_source':True,'RNG_restored':True,'old_Adam_fanin_overshoot_reproduced':True,'constrained_loss_monotonic':True,'all_physical_step_and_total_limits_met':True,'actual_nonzero_parameter_update':True,'constrained_trace':trace,'original_masked_Adam_trace':at,'control_trace':ct};(R/'SOLVER_ENGINEERING_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if 'trace' not in k},indent=2))
if __name__=='__main__':main()

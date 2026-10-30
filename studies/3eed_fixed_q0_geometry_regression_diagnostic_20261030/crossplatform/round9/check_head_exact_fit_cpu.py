"""Synthetic cached-feature exact-fit CPU fixture; no real model/GT file."""
from pathlib import Path
import sys,json,hashlib
import torch
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D.parent/'round7'))
from check_regression_adapter_cpu import Synthetic,snapshot,assert_state
from regression_adapter import capture_source
from head_exact_fit import exact_fit_same_head

def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(2);torch.manual_seed(9);model=Synthetic().eval();model.requires_grad_(False)
    initial=snapshot(model);inp={'point_clouds':torch.randn(1,7,3),'text':['the car']};cache=capture_source(model,inp,torch.tensor([False,False,True,False]),query_index=2)
    target=cache.boxes_raw6[0,2].clone()+torch.tensor([2.,-1.,3.,.5,.4,.3]);results={}
    for name,mask in [('C',[1,1,1,0,0,0]),('S',[0,0,0,1,1,1]),('CS',[1,1,1,1,1,1])]:
        r=exact_fit_same_head(model,cache,target,mask)
        assert r['supervised_max_abs_coordinate_residual_m']<=5e-5 and r['inactive_coordinates_bitexact_source'] and r['observed_parameter_change'] and r['real_backward_and_optimizer_step_executed']
        assert_state(model,initial);results[name]={k:v for k,v in r.items() if k!='boxes_raw6'}
    assert not torch.cuda.is_initialized()
    out={'state':'synthetic_CPU_exactfit_real_backward_same_head_pass','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'head_exact_fit.py',Path(__file__)]},'results':results,'no_real_model_dataset_GT_index_or_GPU_used':True}
    (D/'CPU_EXACTFIT_FIXTURE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()

"""Actual CPU torch final Conv backward/SGD synthetic fixture, no model/data."""
from pathlib import Path
import torch,json
TOL=5e-5;torch.manual_seed(901);torch.set_num_threads(1)
records={}
for name,mask in [('C',[1,1,1,0,0,0]),('S',[0,0,0,1,1,1]),('CS',[1]*6)]:
 c=torch.nn.Conv1d(288,3,1);s=torch.nn.Conv1d(288,3,1);hc=torch.randn(1,288,1);hs=torch.randn(1,288,1)*3;base=torch.randn(1,1,3)
 def readout():return torch.cat([base+c(hc).transpose(1,2),s(hs).transpose(1,2)],-1)[0,0]
 before=readout().detach().clone();target=torch.tensor([1.,2.,3.,4.,2.,1.]);w=torch.tensor(mask,dtype=torch.float32);direct=torch.where(w>0,target,before)
 original={n:p.detach().clone() for h,m in [('c',c),('s',s)] for n,p in [(h+'.'+n,p) for n,p in m.named_parameters()]}
 nc=float(hc.square().sum()+1);ns=float(hs.square().sum()+1)
 opt=torch.optim.SGD([{'params':c.parameters(),'lr':1/nc},{'params':s.parameters(),'lr':1/ns}],momentum=0,weight_decay=0)
 opt.zero_grad(set_to_none=True);loss=.5*((readout()-target).square()*w).sum();loss.backward();grad={h+'.'+n:float(p.grad.abs().max()) for h,m in [('c',c),('s',s)] for n,p in m.named_parameters()};opt.step();after=readout().detach();error=float((after-direct).abs().max());assert error<TOL
 delta={h+'.'+n:float((p-original[h+'.'+n]).abs().max()) for h,m in [('c',c),('s',s)] for n,p in m.named_parameters()}
 inactive='s' if name=='C' else 'c' if name=='S' else None
 if inactive:assert all(v==0 for k,v in delta.items() if k.startswith(inactive+'.'))
 assert any(v>0 for v in grad.values()) and any(v>0 for v in delta.values())
 records[name]={'max_actual_output_error_vs_direct_m':error,'gradient_max_by_parameter':grad,'parameter_delta_max':delta,'center_selfkernel':nc,'size_selfkernel':ns,'inactive_block_exactly_unchanged':True if inactive else None}
p=Path(__file__).with_name('CPU_ACTUAL_BACKWARD_FIXTURE.json');assert not p.exists();p.write_text(json.dumps({'state':'synthetic_CPU_actual_Conv_backward_pass','records':records,'tolerance_m':TOL,'synthetic_only_not_actual_3EED_head_or_task_proof':True,'no_model_checkpoint_data_GT_GPU':True},indent=2)+'\n');print(json.dumps({'state':'pass','errors':{k:v['max_actual_output_error_vs_direct_m'] for k,v in records.items()}}))

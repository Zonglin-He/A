"""Four input-hash fixed observations x two source modes, true full inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,time,numpy as np
from common36 import *
from infer25 import FrozenTrimodalGrounder

def main():
 import torch
 check_plan();verify(BASE/'fit272_attempt001');out=BASE/'closure4_attempt001';out.mkdir(exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;selection=read(EXP/'CLOSURE_INPUTS_PRIVATE.json');lookup={r['id']:r for r in rows()+rows('DEV64')};records=[];start=time.perf_counter()
 for mode in MODES:
  model=FrozenTrimodalGrounder(source_path(mode),'residual_rank');outside=fp_nonrank(model.fusion);original={k:v.detach().clone() for k,v in model.fusion.rank.state_dict().items()};hook=[];hd=model.fusion.rank.register_forward_pre_hook(lambda m,x:hook.append(x[0].detach().clone()))
  for selected in selection:
   r=lookup[selected['id_PRIVATE']];model.fusion.rank.load_state_dict(original);hook.clear();f=model.predict({k:v for k,v in r.items() if k not in ['stage','alias','frame_alias']},mode=mode);u=hook[-1][0].cpu().numpy();c=captured(r,mode);p=load_npz(BASE/'fit272_attempt001/predictions_PRIVATE'/(r['id']+'_'+mode+'.npz'));sourceerr=float(abs(f['scores']-c['native_scores']).max());uerr=float(abs(u-c['rankinput']).max());state=load_npz(BASE/'fit272_attempt001/states_PRIVATE'/(r['platform']+'_'+mode+'.npz'));model.fusion.rank.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});a=model.predict({k:v for k,v in r.items() if k not in ['stage','alias','frame_alias']},mode=mode);adaptederr=float(abs(a['scores']-p['native_scores'][1]).max());q0=grouped_readout(r,f['scores'],c['native_scores']);qt=grouped_readout(r,a['scores'],c['native_scores']);boxsame=np.array_equal(f['boxes'],c['boxes']) and np.array_equal(a['boxes'],c['boxes']);rec={'alias':r['alias'],'mode':mode,'source_native_maxerror':sourceerr,'rankinput_maxerror':uerr,'adapted_native_maxerror':adaptederr,'frozen_query_same':q0==int(p['queries'][0]),'adapted_query_same':qt==int(p['queries'][1]),'boxes_bitexact':boxsame};records.append(rec);dump(out/'RUNNING_RECEIPTS.json',records);assert sourceerr<2e-5 and uerr<2e-5 and adaptederr<2e-5 and boxsame and rec['frozen_query_same'] and rec['adapted_query_same'],rec;assert fp_nonrank(model.fusion)==outside and model.unchanged()
  model.fusion.rank.load_state_dict(original);hd.remove();del model;torch.cuda.empty_cache()
 dump(out/'CLOSURE_AUDIT.json',{'state':'PASSED','fixed_observations':4,'source_modes':2,'actual_full_grounding_RGB_fusion_forwards':16,'nonrank_and_backbone_frozen':True,'all_boxes_and_argmax_reproduced':True,'rows':records});seal(out,'R36_input_hash_fixed_ADEV_full_RGB_pointtext_source_and_adapted_closure',n=4,forwards=16,seconds=time.perf_counter()-start,plan_sha256=sha(EXP/'PLAN_LOCK.json'));print('CLOSURE36_SEALED')
if __name__=='__main__':main()

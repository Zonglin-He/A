"""Four fixed inputs, runtime/memory only. No GT, hyperparameter search or scores."""
import sys,time,hashlib
import numpy as np
import torch
from common import ROOT,EXP,read,dump,sha,seal,check_plan
from runner import Reset
from algorithms import tent,sar,memo,detach
from reg_inspired import run_reg
sys.path[:0]=[str(ROOT/'crossplatform/round9'),str(ROOT/'crossplatform/round7'),str(ROOT/'crossplatform/round7/experiment')]
from run_diagnostic import load_model,mask_for,fingerprint
from runner_helpers import sample_points
def main():
 check_plan();lock=read(EXP/'COST_PROBE_LOCK.json');assert lock['program_sha256']==sha(__file__);rows=read(EXP.parent/'input/ALL96_PRIVATE.json');rows=[r for p in ('drone','quad') for r in [x for x in rows if x['platform']==p][:2]];out=ROOT/'crossplatform/round18/cost_probe4_attempt001';out.mkdir(exist_ok=False)
 torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;model,_=load_model(read(ROOT/'research/round15/experiment/SOURCE_CAPTURE_PLAN_LOCK.json'));fp=fingerprint(model);reset=Reset(model);results=[]
 for i,r in enumerate(rows):
  pc,_,_=sample_points(r);pc=torch.tensor(pc,device='cuda')[None];text=r['caption']+' . not mentioned';m,_=mask_for(model,text,'all_content_tokens');mask=torch.as_tensor(m,device='cuda');seed=int(hashlib.sha256(('R18|'+r['id']).encode()).hexdigest()[:8],16)%2**31
  reset.reset(seed)
  with torch.no_grad():source=detach(model({'point_clouds':pc,'text':[text]}),mask)
  rng=np.random.default_rng(seed);views=np.stack([pc[0].cpu().numpy()[rng.permutation(16384)]+rng.normal(0,.01,(16384,3)).clip(-.03,.03).astype(np.float32) for _ in range(4)])
  for method in ('Tent','SAR','MEMO','Reg_inspired'):
   reset.reset(seed);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tick=time.perf_counter()
   if method=='Tent':_,_,d,sel=tent(model,pc,[text],mask)
   elif method=='SAR':_,_,d,sel=sar(model,pc,[text],mask)
   elif method=='MEMO':_,_,_,d,sel=memo(model,torch.tensor(views,device='cuda'),[text]*4,mask,dict(source,points=pc[0].cpu().numpy()))
   else:_,_,_,_,d,sel,_,_,_=run_reg(model,pc,text,mask,seed)
   torch.cuda.synchronize();results.append({'case_alias':f'cost_{r["platform"]}_{i:03d}','platform':r['platform'],'method':method,'seconds':time.perf_counter()-tick,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'selected_parameters':sum(p.numel() for n,p in sel),'gradient_parameters':d['gradient_parameters'],'steps':d['steps']})
 reset.reset();assert fingerprint(model)==fp;dump(out/'RESULTS.json',results);seal(out,'R18_cost4_no_task_labels_or_method_selection',n=4,calls=16,program_sha256=sha(__file__));print('COST_PROBE_SEALED')
if __name__=='__main__':main()

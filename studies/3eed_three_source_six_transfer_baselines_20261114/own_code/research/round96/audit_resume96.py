"""CPU exact next-step checkpoint/RNG roundtrip for both source provenance tags."""
from pathlib import Path
import sys,json,random
R=Path(__file__).resolve().parents[2];H=R/'research/round96';B=R/'crossplatform/round96';sys.path.insert(0,str(B));from factory import torch,np,sha;from checkpointing import save_resume,load_resume
O=B/'resume_fixture_attempt001';O.mkdir();rows=[]
for pl in ['drone','quad']:
 torch.manual_seed(96);np.random.seed(96);random.seed(96);m=torch.nn.Linear(3,2);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);sch=torch.optim.lr_scheduler.MultiStepLR(opt,[2,3],gamma=.1);g=torch.Generator().manual_seed(0);plan_sha=sha(H/'experiment'/(pl.upper()+'_TRAIN_LOCK.json'));p={'trained_platforms':[pl],'plan_sha256':plan_sha};f=O/(pl+'.pth');x=torch.randn(4,3);loss=m(x).square().mean();loss.backward();opt.step();sch.step();save_resume(m,opt,sch,g,f,1,1,p)
 def step(m,opt,sch,g):
  x=torch.randn(4,3)+float(np.random.rand())+random.random()+float(torch.rand((),generator=g));y=m(x).square().mean();opt.zero_grad();y.backward();opt.step();sch.step();return x.clone(),y.detach().clone(),{k:v.clone() for k,v in m.state_dict().items()}
 a=step(m,opt,sch,g);m2=torch.nn.Linear(3,2);o2=torch.optim.AdamW(m2.parameters(),lr=1e-4);s2=torch.optim.lr_scheduler.MultiStepLR(o2,[2,3],gamma=.1);g2=torch.Generator();ep,st=load_resume(f,m2,o2,s2,g2,plan_sha);b=step(m2,o2,s2,g2);assert ep==2 and st==1 and torch.equal(a[0],b[0]) and torch.equal(a[1],b[1]) and all(torch.equal(a[2][k],b[2][k]) for k in a[2]);ck=torch.load(f,weights_only=True);assert ck['trained_platforms']==[pl];rows.append({'platform':pl,'safe_load':True,'identical_next_input_loss_AdamW_update':True,'all_CPU_RNG_restored':True})
out={'state':'PASS_R96_BOTH_SOURCE_TAGS_AND_EXACT_CPU_RESUME','rows':rows,'CUDA_initialized':torch.cuda.is_initialized(),'trained_task_model':False};assert not out['CUDA_initialized'];(H/'experiment/RESUME_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

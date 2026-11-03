"""Real episodic matching projection LoRA under frozen joint contradictions."""
import argparse,math,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from common55 import *
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint,native_matching
sys.path.insert(0,str(ROOT/'research/round54'));from adapt54 import fingerprint_in_order
class LowRank(nn.Module):
 def __init__(self,w):
  super().__init__();assert w.shape==(64,288)
  with torch.random.fork_rng(devices=[w.device.index] if w.is_cuda else []):
   torch.manual_seed(5500);self.A=nn.Parameter(w.new_empty(4,288));self.B=nn.Parameter(w.new_zeros(64,4));nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))
 def forward(self,w):return w+self.B@self.A
class Interface:
 def __init__(self,m):
  self.model=m;self.layer=m.contrastive_align_projection_image[4];self.adapter=LowRank(self.layer.weight);parametrize.register_parametrization(self.layer,'weight',self.adapter)
  self.initial={n:x.detach().clone() for n,x in self.adapter.named_parameters()};self.versions={n:x._version for n,x in m.named_parameters() if not x.requires_grad};assert sum(x.numel() for x in self.adapter.parameters())==1408
 def reset(self,trainable):
  with torch.no_grad():
   for n,x in self.adapter.named_parameters():x.copy_(self.initial[n]);x.grad=None;x.requires_grad_(trainable)
  assert torch.equal(self.layer.weight,self.layer.parametrizations.weight.original)
 def check(self):
  for n,x in self.model.named_parameters():
   if n in self.versions:assert not x.requires_grad and x.grad is None and x._version==self.versions[n],n
  assert all(not m.training for m in self.model.modules())
 def delta(self):return float(torch.sqrt(sum((p.detach()-self.initial[n]).square().sum() for n,p in self.adapter.named_parameters())))
def role_scores(model,hidden,tokens,masks):
 q=F.normalize(model.contrastive_align_projection_image[4](hidden),p=2,dim=-1)
 # Native ALL-token denominator/logodds; each role uses its own literal mask.
 pairs=[native_matching(q,tokens,mask) for mask in masks]
 return pairs[0][0],torch.stack([x[1] for x in pairs])
def distributions(z,C,z0):
 p0=z0.double().softmax(-1)
 if not len(C):return p0+z.sum()*0
 known=torch.cat([C.any(0).any(-1)[None],C.any(1)],0)
 groups=[]
 for k in range(len(z)):
  output=z[k].double()*0
  for mask in [known[k],~known[k]]:
   if mask.any():output=output.masked_scatter(mask,z[k,mask].double().softmax(-1)*p0[k,mask].sum())
  groups.append(output)
 return torch.stack(groups)
def conflict_loss(z,C,z0):
 p=distributions(z,C,z0)
 if not len(C):return z.sum()*0,p.new_empty(0)
 mass=torch.einsum('i,rj,rij->r',p[0],p[1:],C.double());assert (mass>=-1e-12).all() and (mass<1).all()
 return -torch.log1p(-mass).mean(),mass

def joint_readout(z,C,z0):
 p=distributions(z,C,z0)
 if not len(C):return p[0].clamp_min(1e-300).log()
 valid=1-torch.einsum('rj,rij->ri',p[1:],C.double());assert (valid>0).all()
 return p[0].clamp_min(1e-300).log()+valid.log().sum(0)

def objective(model,interface,hidden,tokens,masks,C,z0):
 p,z=role_scores(model,hidden,tokens,masks);conf,mass=conflict_loss(z,C,z0);anchor=z0.argmax(-1,keepdim=True);d=z-z.gather(1,anchor);d0=z0-z0.gather(1,anchor);keep=(d-d0).square().mean();dw=interface.adapter.B@interface.adapter.A;prior=dw.square().mean();loss=conf+keep+.01*prior
 return loss,{'conflict':conf,'keep':keep,'parameter_prior':prior},p,z,mass

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--limit',type=int);ap.add_argument('--resume',action='store_true');a=ap.parse_args();plan=read(a.plan);pin(plan);rows=read(ROOT/plan['manifest']);out=ROOT/plan['output_root'];binding=sha(a.plan)
 if out.exists():assert a.resume and read(out/'RUN_BINDING.json')['plan_sha256']==binding and not (out/'SEALED.json').exists()
 else:out.mkdir(parents=True);dump(out/'RUN_BINDING.json',{'plan_sha256':binding})
 for f in ['samples_PRIVATE','cases_PRIVATE','fitted_PRIVATE']:(out/f).mkdir(exist_ok=True)
 torch.manual_seed(0);start=time.perf_counter();model=load_source(read(ROOT/plan['source_plan']));fp=fingerprint(model);order=list(model.state_dict());interface=Interface(model);ls=read(ROOT/plan['latent_root']/'SEALED.json');es=read(ROOT/plan['evidence_root']/'SEALED.json');done=[];statuses=[];steps_total=0
 try:
  for row in rows:
   sid=row['id'];cf=out/'cases_PRIVATE'/(sid+'.json');sf=out/'samples_PRIVATE'/(sid+'.npz')
   if cf.exists():
    rec=read(cf);assert a.resume and rec['state']=='COMPLETED' and sha(sf)==rec['sample_sha256'];done.append(sid);statuses.append(rec['status_row']);steps_total+=rec['actual_optimizer_steps'];continue
   rel='samples_PRIVATE/'+sid+'.npz';assert sha(ROOT/plan['latent_root']/rel)==ls['files'][rel] and sha(ROOT/plan['evidence_root']/rel)==es['files'][rel]
   cache=load(ROOT/plan['latent_root']/rel,['h','tokens','scores','logodds','boxes']);ev=load(ROOT/plan['evidence_root']/rel,['role_masks','conflicts']);device=next(model.parameters()).device;T=lambda x:torch.as_tensor(x,device=device)
   interface.reset(False)
   with torch.no_grad():
    hidden=model.contrastive_align_projection_image[:4](T(cache['h'])[None]);tokens=T(cache['tokens'])[None];masks=T(ev['role_masks']).bool();p0,z0=role_scores(model,hidden,tokens,masks);assert torch.equal(p0,T(cache['scores'])) and torch.equal(z0[0],T(cache['logodds']))
   result={'source':cache['scores'].copy(),'zero_step':p0.cpu().numpy()};diag={};fit={};st={a:{'unknown':False,'fallback':False,'no_parameter_update':True} for a in ['source','zero_step']};actual=0
   for variant,Cnp in zip(VARIANTS,ev['conflicts']):
    interface.reset(True);C=T(Cnp);optimizer=torch.optim.Adam(interface.adapter.parameters(),lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=0)
    active=bool(C.any());unknown=not active
    with torch.no_grad():
     initial,initialterms,pp,zz,mass0=objective(model,interface,hidden,tokens,masks,C,z0);direct=joint_readout(z0,C,z0);result[variant+'_direct']=direct.cpu().numpy();st[variant+'_direct']={'unknown':unknown,'fallback':False,'no_parameter_update':True}
    trajectory=[]
    for step in range(1,4):
     optimizer.zero_grad(set_to_none=True);loss,terms,_,_,_=objective(model,interface,hidden,tokens,masks,C,z0)
     if active:
      loss.backward();grad=float(torch.sqrt(sum(x.grad.square().sum() for x in interface.adapter.parameters() if x.grad is not None)));torch.nn.utils.clip_grad_norm_(interface.adapter.parameters(),1);optimizer.step();actual+=1
     else:grad=0.
     interface.check()
     with torch.no_grad():
      final,fterms,p,z,mass=objective(model,interface,hidden,tokens,masks,C,z0);joint=joint_readout(z,C,z0);delta=interface.delta()
      trajectory.append({'step':step,'loss_before':float(loss),'loss_after':float(final),'conflict_after':float(fterms['conflict']),'keep_after':float(fterms['keep']),'gradient_norm':grad,'parameter_delta_L2':delta,'target_score_delta_maxabs':float((z[0]-z0[0]).abs().max()),'reference_score_delta_maxabs':float((z[1:]-z0[1:]).abs().max()) if len(z)>1 else 0.,'conflict_mass_mean':float(mass.mean()) if len(mass) else 0.,'active':active})
      if step in [1,3]:
       for kind,scores in [('raw',distributions(z,C,z0)[0].clamp_min(1e-300).log()),('joint',joint)]:
        name=variant+'_tta'+str(step)+'_'+kind;result[name]=scores.cpu().numpy();st[name]={'unknown':unknown,'fallback':False,'no_parameter_update':delta==0}
       fit[variant+'_'+str(step)+'__A']=interface.adapter.A.detach().cpu().numpy().copy();fit[variant+'_'+str(step)+'__B']=interface.adapter.B.detach().cpu().numpy().copy();fit[variant+'_'+str(step)+'__role_scores']=z.cpu().numpy().copy()
    diag[variant]={'initial_loss':float(initial),'initial_conflict_mass_mean':float(mass0.mean()) if len(mass0) else 0.,'initial_top1_target_conflict_mass':float((C.double()@z0[1:].double().softmax(-1).unsqueeze(-1)).squeeze(-1)[:,int(p0.argmax())].mean()) if len(C) else 0.,'steps':trajectory,'nonzero_conflict_pairs':int((C>0).sum()),'unknown_measurements_not_trainable':True}
    optimizer.zero_grad(set_to_none=True);del optimizer
   assert set(result)==set(ARMS);stack=np.stack([result[k] for k in ARMS]);queries=stack.argmax(1).astype(np.int64);np.savez_compressed(sf,arm_names=np.asarray(ARMS),scores=stack,queries=queries,selected_original_boxes=cache['boxes'][queries],source_boxes=cache['boxes'],source_scores=cache['scores']);np.savez_compressed(out/'fitted_PRIVATE'/(sid+'.npz'),**fit)
   status={'id':sid,'arm_statuses':st};rec={'state':'COMPLETED','id_PRIVATE':sid,'platform':row['platform'],'sample_sha256':sha(sf),'fitted_sha256':sha(out/'fitted_PRIVATE'/(sid+'.npz')),'statistics':diag,'status_row':status,'actual_optimizer_steps':actual,'source_score_bridge_bitexact':True};dump(cf,rec);done.append(sid);statuses.append(status);steps_total+=actual
   if len(done)%16==0:print('ADAPT55',len(done),336,flush=True)
   if a.limit is not None and len(done)>=a.limit:break
  interface.reset(False);interface.check();parametrize.remove_parametrizations(interface.layer,'weight',leave_parametrized=False);model.requires_grad_(False);assert fingerprint_in_order(model,order)==fp
  counts={'n':len(done),'seconds_this_process':time.perf_counter()-start,'actual_optimizer_steps':steps_total,'trainable_parameters':1408,'source_fingerprint':fp,'source_model_restored_exact':True,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'new_full_grounder_forwards':0}
  if len(done)==336:
   dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',counts);seal(out,'R55_JOINT_CONFLICT_MATCHING_TTA_PREDICTIONS_NO_GT',**counts,arm_names=ARMS,plan_sha256=binding)
  else:dump(out/'ENGINEERING_STOP.json',counts)
  print(json.dumps(counts),flush=True)
 except BaseException as e:dump(out/('FAILED_'+str(time.time_ns())+'.json'),{'type':type(e).__name__,'error':str(e),'n':len(done),'no_GT':True});raise
if __name__=='__main__':main()

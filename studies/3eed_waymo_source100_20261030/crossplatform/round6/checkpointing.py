"""Atomic source-only epoch-boundary checkpoints with safe resumable RNG payloads."""
from pathlib import Path
import os,random,json
import numpy as np
import torch
from factory import sha

def rng_state(generator):
 n=np.random.get_state();return {'python':random.getstate(),'numpy':{'kind':n[0],'keys':torch.tensor(n[1].astype(np.int64)),'position':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])},'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),'loader_shared_generator':generator.get_state()}
def restore_rng(state,generator):
 random.setstate(state['python']);n=state['numpy'];np.random.set_state((n['kind'],n['keys'].cpu().numpy().astype(np.uint32),n['position'],n['has_gauss'],n['cached_gaussian']));torch.set_rng_state(state['torch_cpu'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in state['torch_cuda']]);generator.set_state(state['loader_shared_generator'].cpu())
def atomic_save(obj,path,replace=False):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);assert replace or not path.exists();temp=path.with_name(path.name+'.writing');assert not temp.exists();torch.save(obj,temp)
 with open(temp,'rb') as f:os.fsync(f.fileno())
 os.replace(temp,path);return sha(path)
def model_state_cpu(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def save_model(model,path,epoch,provenance):
 return atomic_save({'model':model_state_cpu(model),'epoch':epoch,'trained_platforms':['waymo'],'initialization':'generic RoBERTa; random visual/grounding; no task checkpoint','source_provenance':provenance},path)
def save_resume(model,opt,sched,generator,path,epoch,step,provenance):
 return atomic_save({'model':model_state_cpu(model),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'rng':rng_state(generator),'epoch':epoch,'next_epoch':epoch+1,'global_step':step,'trained_platforms':['waymo'],'source_provenance':provenance,'resume_scope':'epoch_boundary_after_scheduled_source_validation','joint_checkpoint_initialization':False},path,replace=True)
def load_resume(path,model,opt,sched,generator,plan_sha):
 ck=torch.load(path,map_location='cpu',weights_only=True);assert ck['trained_platforms']==['waymo'] and ck['joint_checkpoint_initialization'] is False and ck['source_provenance']['plan_sha256']==plan_sha;assert ck['resume_scope']=='epoch_boundary_after_scheduled_source_validation';model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);restore_rng(ck['rng'],generator);return ck['next_epoch'],ck['global_step']

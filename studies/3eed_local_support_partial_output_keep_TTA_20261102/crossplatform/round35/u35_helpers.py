"""Deferred U128 interfaces: no U access at import, no training or feedback edits."""
import json,hashlib,numpy as np
from common35 import ROOT,BASE,EXP,read,dump,sha,seal,verify,load_npz
CONFIGS=['R30_U_rows','U_rows_matched_denominator','partial_output_keep'];ARMS=['source']+CONFIGS
DEFAULT_LOCK=ROOT/'research/NIGHT_20261101/SELECTED_NIGHT_CANDIDATE_BEFORE_U_LOCK.json'
U_OUT=BASE/'predictionsU128_attempt001';U_CLOSURE=BASE/'closureU128_attempt001'

def check_u_lock(lockpath):
 p=read(lockpath);assert p['R35_plan_sha256']==sha(EXP/'PLAN_LOCK.json');assert p['R35_training_seal_sha256']==sha(BASE/'partial_keep_attempt001/SEALED.json');assert isinstance(p['selected_primary'],str) and p['U_not_for_selection'] is True
 assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED';verify(BASE/'partial_keep_attempt001')
 plan=read(EXP/'PLAN_LOCK.json')
 for n,h in plan['implementation_sha256'].items():assert sha(BASE/n)==h,n
 assert plan['configurations']==CONFIGS
 interface=read(EXP/'CONFIRMATION_INTERFACE_LOCK.json')
 for n,h in interface['program_sha256'].items():assert sha(BASE/n)==h,n
 return p

def U_rows():
 return read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json')
def U_feature(r):
 assert r['stage']=='U'
 return load_npz(ROOT/'crossplatform/round33/sourceU128_attempt001/samples_PRIVATE'/(r['id']+'.npz'))
def source_head():
 d=load_npz(ROOT/'crossplatform/round24/capture336_attempt001/projection_source_PRIVATE.npz');return {k.removeprefix('image.'):v for k,v in d.items() if k.startswith('image.')}
def state_for(pl,name):return load_npz(BASE/'partial_keep_attempt001/states_PRIVATE'/(pl+'_'+name+'.npz'))
def GPU_probabilities_and_logodds(B,state):
 import torch
 F=torch.nn.functional;m=torch.nn.Sequential(torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,288),torch.nn.ReLU(),torch.nn.Linear(288,64)).cuda().eval().requires_grad_(False);m.load_state_dict({k:torch.tensor(v,device='cuda') for k,v in state.items()});zz=[U_feature(r) for r in B];T=max(len(z['tokens']) for z in zz);tok=np.zeros((len(B),T,64),np.float32);padding=np.ones((len(B),T),bool);mask=np.zeros((len(B),T),bool)
 for j,z in enumerate(zz):tok[j,:len(z['tokens'])]=z['tokens'];padding[j,:len(z['tokens'])]=False;mask[j,:len(z['tokens'])]=z['token_mask']
 with torch.inference_mode():
  q=F.normalize(m(torch.tensor(np.stack([z['h'] for z in zz]),device='cuda')),dim=-1);v=q@torch.tensor(tok,device='cuda').transpose(-1,-2)/.07;pad=torch.tensor(padding,device='cuda');positive=torch.tensor(mask,device='cuda')&~pad;negative=~positive&~pad;vv=v.masked_fill(pad[:,None],-torch.inf);prob=(vv.softmax(-1)*positive[:,None]).sum(-1);ell=torch.logsumexp(v.masked_fill(~positive[:,None],-torch.inf),-1)-torch.logsumexp(v.masked_fill(~negative[:,None],-torch.inf),-1)
 return prob.cpu().numpy(),ell.cpu().numpy()

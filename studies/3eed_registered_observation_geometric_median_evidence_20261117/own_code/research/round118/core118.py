"""Frozen native grounding with source-compatible synchronized context providers.
No training, model selection, task labels, source/media decoding or external model.
"""
from pathlib import Path
import json,hashlib,numpy as np,torch,time
R=Path(__file__).resolve().parents[2];H=R/'research/round118';B=R/'crossplatform/round118'
rd=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
NEW=[f'F_l{l}_r{r}' for l in ['NF','SF','SA'] for r in ['zero','NF','SF','SA']]+['F_lSA_rSAshuffled','F_lSAshuffled_rSAshuffled','A_lSA_rSA','C_lSA_rSA','ABC_lSA_rSA','ABC_lSA_rzero','ABC_native']
REF=['original_source','prepared_A','prepared_F','native_F','repaired_F','lateOnly_F'];ARMS=REF+NEW;PRIMARY='F_lSA_rSA';assert len(NEW)==19 and len(ARMS)==25
REFMAP={'DEV':dict(zip(REF,['original_source','prepared_frozen','real_frozen_fusedABC','native_real_fusedABC','neither_correct_fusedABC','anchor_lateOnly_fusedABC'])), 'EXP':dict(zip(REF,['original_source','prepared_frozen','real_BBF_fusedABC','native_fusedABC','repaired_fusedABC','lateOnly_fusedABC']))}
def olddir(co,name):return R/('crossplatform/round113' if co=='DEV' else 'crossplatform/round117/formal')/name
def rows(co,tgt):return rd(R/f'research/round{99 if co=="DEV" else 109}/input/TARGET_{tgt.upper()}{128 if co=="DEV" else 256}_PRIVATE.json')
def head(src):
 item=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'][src];p=R/item['state_path'];assert sha(p)==item['state_sha256']
 with np.load(p) as z:return {k[6:]:torch.from_numpy(z[k].copy()) for k in z.files if k.startswith('match.')}
def components(co,name,sid):
 files=[];N={};S={}
 if co=='DEV':
  p=olddir(co,name)/'features_PRIVATE'/(sid+'.npz');files.append(p)
  with np.load(p) as z:
   for v in ['A','C','fusedABC']:
    N[v]={k:z['frozen_anchor_'+v+'_'+k][0] for k in ['h','u','late']};N[v].update({k:z['frozen_anchor_'+v+'_'+k] for k in ['mask','r']})
   for v in ['A','fusedABC']:S[v]={k:z['frozen_full_'+v+'_'+k][0] for k in ['h','u','late']};S[v].update({k:z['frozen_full_'+v+'_'+k] for k in ['mask','r']})
  p=R/'crossplatform/round111'/name/'features_PRIVATE'/(sid+'.npz');files.append(p)
  with np.load(p) as z:N['B']={'h':z['real_B_h'],'u':z['real_B_tokens'],'mask':z['real_B_mask']}
 else:
  p=olddir(co,name)/'features_PRIVATE'/(sid+'.npz');files.append(p)
  with np.load(p) as z:N={v:{k:z['native_'+v+'_'+k] for k in ['h','u','late','mask','r']} for v in ['A','B','C','fusedABC']}
  p=R/'crossplatform/round109'/name/'features_PRIVATE'/(sid+'.npz');files.append(p)
  with np.load(p) as z:S={v:{k:z['real_'+v+'_'+k] for k in ['h','u','late','mask','r']} for v in ['A','fusedABC']}
 assert all(np.array_equal(N['A']['mask'],z['mask']) for z in N.values())
 return N,S,{str(p.relative_to(R)):sha(p) for p in files}
def project(h,W):
 import torch.nn.functional as f
 x=torch.from_numpy(h)
 for k in ['0','2']:x=f.relu(f.linear(x,W[k+'.weight'],W[k+'.bias']))
 return f.linear(x,W['4.weight'],W['4.bias'])
def read(raw,u,mask,late,res):
 import torch.nn.functional as f
 q=f.normalize(raw+torch.from_numpy(late),dim=-1);s=q@torch.from_numpy(u).T/.07;m=torch.from_numpy(mask).bool();s=s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1)
 return (s.double()+torch.from_numpy(res).double()).softmax(0).numpy()
def predictions(co,name,sid,N,S,W):
 raw={v:project(z['h'],W) for v,z in N.items()};L={'NF':N['fusedABC']['late'],'SF':S['fusedABC']['late'],'SA':S['A']['late']};res={'zero':np.zeros(256,np.float32),'NF':N['fusedABC']['r'],'SF':S['fusedABC']['r'],'SA':S['A']['r']};Fz=N['fusedABC'];out={}
 for l in L:
  for r in res:out[f'F_l{l}_r{r}']=read(raw['fusedABC'],Fz['u'],Fz['mask'],L[l],res[r])
 seed=int(hashlib.sha256(('R118|'+co+'|'+name+'|'+sid).encode()).hexdigest()[:16],16);order=np.roll(np.random.default_rng(seed).permutation(256),1);perm=np.empty(256,int);perm[order]=np.roll(order,1);assert np.all(perm!=np.arange(256))
 out['F_lSA_rSAshuffled']=read(raw['fusedABC'],Fz['u'],Fz['mask'],L['SA'],res['SA'][perm]);out['F_lSAshuffled_rSAshuffled']=read(raw['fusedABC'],Fz['u'],Fz['mask'],L['SA'][perm],res['SA'][perm])
 ps={};pn={};pz={}
 for v in ['A','B','C']:
  z=N[v];ps[v]=read(raw[v],z['u'],z['mask'],L['SA'],res['SA']);pz[v]=read(raw[v],z['u'],z['mask'],L['SA'],res['zero']);pn[v]=read(raw[v],z['u'],z['mask'],L['SA']*0,res['zero'])
 for v in ['A','C']:out[v+'_lSA_rSA']=ps[v]
 out['ABC_lSA_rSA']=np.mean(list(ps.values()),axis=0);out['ABC_lSA_rzero']=np.mean(list(pz.values()),axis=0);out['ABC_native']=np.mean(list(pn.values()),axis=0)
 assert set(out)==set(NEW)
 # Original providers exactly reproduce their trained/readout definitions.
 checks={}
 for v in ['A','fusedABC']:
  z=S[v];checks['prepared_'+v]=read(project(z['h'],W),z['u'],z['mask'],z['late'],z['r'])
 checks['native_A']=pn['A'];checks['native_F']=read(raw['fusedABC'],Fz['u'],Fz['mask'],L['SA']*0,res['zero'])
 return out,checks,perm

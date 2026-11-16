"""Causal ablation of active source point-adapter versus early cross-modal exchange."""
from pathlib import Path
import sys,contextlib
R112=Path(__file__).resolve().parents[2];H112=R112/'research/round112';B112=R112/'crossplatform/round112';sys.path.insert(0,str(R112/'research/round111'));from core111 import *
VAR112=['full','noPA','noEarlyRGB','neither'];BIND112=['correct','wrong'];VIEWS112=['A','fusedABC'];PRIMARY112='noPA_correct_fusedABC';REFS112=jread(B111/'MATRIX_SEALED.json')['arms'];NEW112=[v+'_'+b+'_'+view for v in VAR112 for b in BIND112 for view in VIEWS112 if not(v=='full' and b=='correct')];ARMS112=REFS112+NEW112
ALLOWED112=['point_adapter.up.weight','point_adapter.up.bias','gate_P']

def pair112(src,r):
 name=src+'_to_'+r['platform'];sid=r['id'];D=B111/name;z0=load(D/'prefix_PRIVATE'/(sid+'.npz'));key=hashlib.sha256(z0['xyz'].tobytes()+z0['point_indices'].tobytes()).hexdigest();f=B97/'shared_physical_geometry_PRIVATE'/(sid+'_'+key[:20]+'.npz');assert f.is_file();proof=jread(f.with_suffix('.json'));assert proof['geometry_seed_and_sample_sha256']==key;g=load(f);g['query_anchor_indices']=np.asarray(z0['query_points_sample_inds'],np.int64);obs=load(D/'observations_PRIVATE'/(sid+'.npz'));oo=[]
 for v in ['B','C']:
  if 'real_'+v+'_vis' in obs:oo.append({k:torch.as_tensor(obs['real_'+v+'_'+k],device='cuda') for k in ['vis','valid']})
 z=tensor_prefix(z0);return z0,g,{'A':z,'fusedABC':prefix106(z,oo,True)},{'geometry_cache_sha256':sha(f),'R111_prefix_sha256':sha(D/'prefix_PRIVATE'/(sid+'.npz')),'R111_observations_sha256':sha(D/'observations_PRIVATE'/(sid+'.npz')),'no_new_sensor_read':True,'supported_observations':len(oo)}

def edit112(m,S,variant):
 restore(m,S)
 with torch.no_grad():
  if variant in ['noPA','neither']:m.point_adapter.up.weight.zero_();m.point_adapter.up.bias.zero_()
  if variant in ['noEarlyRGB','neither']:m.gate_P.zero_()
 ss=state(m);changed=[k for k,v in S.items() if not np.array_equal(ss[k],v)];assert set(changed)<=set(ALLOWED112);return {'changed_tensor_names':changed,'changed_values':sum(int(np.count_nonzero(ss[k]!=S[k])) for k in changed),'parameter_L2':float(np.sqrt(sum(np.square(ss[k].astype(float)-S[k].astype(float)).sum() for k in changed))),'TTA':False}

@contextlib.contextmanager
def functional112(m,variant):
 hs=[]
 if variant in ['noPA','neither']:hs.append(m.point_adapter.register_forward_hook(lambda _m,a,o:a[0]))
 if variant in ['noEarlyRGB','neither']:hs.append(m.image_to_point.register_forward_hook(lambda _m,a,o:torch.zeros_like(o)))
 try:yield
 finally:
  for h in hs:h.remove()

def forward112(m,base,z,d,route):
 m.route=route;m.use_interface=True;m.variant='full';m.pool_mode='strict'
 with torch.no_grad():p,q,e=m.main(base,z,d,'bidirectional')
 f={k:e[j].detach() for k,j in [('h','query_hidden'),('u','projected_text'),('late','late_RGB_delta'),('r','residual')]};f.update(mask=z['token_mask'].bool(),p=p.detach());return f

def cpu112(h,u,mask,late,residual,S):
 import torch.nn.functional as fn
 x=torch.from_numpy(h)
 for j in [0,2]:x=fn.relu(fn.linear(x,torch.from_numpy(S[str(j)+'.weight']),torch.from_numpy(S[str(j)+'.bias'])))
 raw=fn.linear(x,torch.from_numpy(S['4.weight']),torch.from_numpy(S['4.bias']))+torch.from_numpy(late);q=fn.normalize(raw,dim=-1);s=q@torch.from_numpy(u).T/.07;m=torch.from_numpy(mask).bool();s=s[:,m].logsumexp(-1)-s[:,~m].logsumexp(-1);return (s.double()+torch.from_numpy(residual).double()).softmax(0).numpy()

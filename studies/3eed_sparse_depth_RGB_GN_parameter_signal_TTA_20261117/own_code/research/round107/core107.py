"""Fixed R106 evidence/head/solver: supervision source x fused-view preservation."""
from pathlib import Path
import sys
R107=Path(__file__).resolve().parents[2];H107=R107/'research/round107';B107=R107/'crossplatform/round107';sys.path.insert(0,str(R107/'research/round106'));from core106 import *
FAMS107=['real','randomA','wrong'];FITS107=['mean_keep','native2','native_keep'];PRIMARY107='real_mean_keep_fusedABC'
def cached107(z,k,v):return {f:torch.as_tensor(z[k+'_'+v+'_'+f],device='cuda') for f in ['h','u','late','mask','r','p']}
def targets107(A,B,AB,variant):
 mean=(A['p']+B['p'])/2;q=AB['p'] if variant.startswith('native') else mean;views=[A,B];targets=[q,q]
 if variant.endswith('keep'):views.append(AB);targets.append(AB['p'])
 return views,torch.stack(targets).log()
def normmatch107(S,old,new):
 od={k:old[k].astype(np.float64)-S[k].astype(np.float64) for k in S};nd={k:new[k].astype(np.float64)-S[k].astype(np.float64) for k in S};on=np.sqrt(sum(np.square(v).sum() for v in od.values()));nn=np.sqrt(sum(np.square(v).sum() for v in nd.values()));ratio=float(nn/on) if on>0 else 0.;st={k:(S[k].astype(np.float64)+ratio*od[k]).astype(np.float32) for k in S};actual=np.sqrt(sum(np.square(st[k].astype(np.float64)-S[k].astype(np.float64)).sum() for k in S));cos=float(sum((od[k]*nd[k]).sum() for k in S)/(on*nn)) if on*nn>0 else None;assert abs(actual-nn)<1e-5 or on==0;return st,{'old_parameter_L2':float(on),'protected_parameter_L2':float(nn),'normmatched_parameter_L2':float(actual),'scale_old_direction':ratio,'direction_cosine':cos,'old_zero_direction':bool(on==0),'C_or_GT_used':False}
def obtain_AB107(base,m,S,r,ar,pairs,oldrec,engineer=False):
 # R106 did not save aligned pre-language B features. Re-extract that missing
 # state with exact same sensor hashes/sampling. No C extraction in this API.
 z0,g,_=capture98(base,r,source97(src_now107)['source']);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);obs={};proof={};ab={}
 for k in ['real','randomA']:
  ob,pf=extra106(base,ar['A'],pairs[ar['B_pair']],z0,k,'B');assert ob is not None;prior=oldrec['families'][k]['support']['B'];assert all(pf[a]==prior[a] for a in ['point_sha256','sample_sha256','registration_sha256','aligned_seed_n']);obs[k]=ob;proof[k]=pf
 obs['wrong']=wrong106(obs['real'],r,'B')
 with torch.no_grad():
  for k in FAMS107:ab[k]=feature106(m,base,z,d,route,S,[obs[k]],True)
 return ab,proof,(z0,g,z,d,route,obs)
src_now107=None
def setsource107(s):
 global src_now107
 src_now107=s;setsource97(s)
def draw107(m,views,st):
 put94(m,st)
 with torch.no_grad():p={v:read105(m,f) for v,f in views.items()};p['ensemble4']=torch.stack([p[v] for v in ['A','B','AB','C']]).mean(0);return p

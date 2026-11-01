"""R34 contracts. Merely importing this module does not read U/FINAL manifests."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'crossplatform/round33'),str(ROOT/'crossplatform/round32'),str(ROOT/'crossplatform/round30'),str(ROOT/'crossplatform/round24')]
import common33 as prior
from common33 import read,dump,sha,seal,verify,arrsha,load_npz
BASE=ROOT/'crossplatform/round34';EXP=ROOT/'research/round34/experiment'
DIAGNOSTIC_SPLITS=['A208','DEV64','R64','S64','T128']
KEYS=['h288','noun64'];ARMS=['source','R30_ungated','h_scoped_R30','noun_scoped_R30','h_samegate_QP','noun_samegate_QP','h_global_mean','h_global_norm','noun_global_mean','noun_global_norm']
EPS=1e-6;MARGIN=.1;TEMPERATURE=.1

def rows(split='A208'):
 if split=='diagnostic':return [dict(r,split=s) for s in DIAGNOSTIC_SPLITS for r in prior.rows(s)]
 if split in DIAGNOSTIC_SPLITS or split=='U128':return prior.rows(split)
 raise ValueError('Only prespecified A/DEV/R/S/T diagnostic and U128 are permitted')
def feature(r):return prior.feature(r)
def output(split):
 if split not in ['diagnostic','U128']:raise ValueError(split)
 return BASE/('scope_'+split+'_attempt001')
def check_plan():
 plan=read(EXP/'PLAN_LOCK.json')
 amendment=EXP/'EXECUTION_SCOPE_AMENDMENT_LOCK.json';programs=dict(plan['program_sha256'])
 if amendment.exists():
  a=read(amendment);assert a['original_plan_sha256']==sha(EXP/'PLAN_LOCK.json');programs.update(a['new_program_sha256'])
 for p,h in programs.items():assert sha(ROOT/p)==h,p
 for p,h in plan['immutable_input_sha256'].items():assert sha(ROOT/p)==h,p
 return plan

def plan_receipt():
 a=EXP/'EXECUTION_SCOPE_AMENDMENT_LOCK.json'
 return {'plan_sha256':sha(EXP/'PLAN_LOCK.json'),'execution_scope_amendment_sha256':sha(a) if a.exists() else None}

def closure_rows(split):
 rr=rows(split)
 if split=='U128':return rr,[]
 assert split=='diagnostic';selected=[];receipt=[]
 for ss in DIAGNOSTIC_SPLITS:
  for pl in ['drone','quad']:
   pool=[r for r in rr if r['split']==ss and r['platform']==pl];assert pool
   def inputhash(r):
    inp={k:v for k,v in r.items() if k not in ('alias','frame_alias','stage','split')}
    return hashlib.sha256(json.dumps(inp,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
   r=min(pool,key=lambda r:(inputhash(r),r['id']));selected.append(r);receipt.append({'split':ss,'platform':pl,'alias':r['alias'],'input_manifest_sha256':inputhash(r),'selection':'minimum canonical input manifest SHA256, no GT or score'})
 assert len(selected)==10
 return selected,receipt

def normalized(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
def make_key(z,kind):
 if kind=='h288':a=z['h']
 elif kind=='noun64':a=z['query']*normalized(z['tokens'][z['token_mask']].mean(0))
 else:raise ValueError(kind)
 return normalized(a).astype(np.float32)
def relative_logodds(p,q0):
 # This is the R32 inference coordinate; it is not the exact training log-odds.
 p=np.asarray(p,np.float64);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
 ell=np.log(p+EPS)-np.log(1+EPS-p);return ell-ell[q0]
def group_readout(r,s):
 if r['stage'] in ['Q','R','S','T','U']:return int(np.argmax(s))
 groups=read(ROOT/'crossplatform/round29/groups208_attempt001/cases_PRIVATE'/(r['id']+'.json'))['groups_PRIVATE'];used=set();options=[];p=feature(r)['scores']
 for ix in groups.values():
  used.update(ix);rep=max(ix,key=lambda i:(float(p[i]),-i));options.append((float(np.max(s[ix])),-rep,rep))
 for i in range(256):
  if i not in used:options.append((float(s[i]),-i,i))
 return int(max(options)[2])
def parent_probability(r):
 # Existing official float32 probabilities are retained on exposed diagnostic rows.
 if r['stage']=='T':return load_npz(ROOT/'crossplatform/round32/selectedT128_attempt001/predictions_PRIVATE'/(r['id']+'.npz'))['adapted_probability']
 if not hasattr(parent_probability,'cache'):
  parent_probability.cache={}
  for pl in ['drone','quad']:
   z=load_npz(ROOT/'crossplatform/round31/loo19_attempt002/predictions_PRIVATE'/(pl+'_all.npz'))
   parent_probability.cache.update({rid:z['scores'][j] for j,rid in enumerate(z['row_ids_PRIVATE'].tolist())})
 return parent_probability.cache[r['id']]
def readout_scores(p0,pT,library,neighbors):
 """All arithmetic follows fixed R32 q0-anchored score logic; NN indices are given."""
 q0=int(p0.argmax());d0=relative_logodds(p0,q0);dT=relative_logodds(pT,q0);delta=dT-d0
 obs=neighbors['obs'][:,:1];pos=neighbors['pos'][:,:1];w=neighbors['weights'][:,:1].astype(np.float64)
 g=library['endpoint'][obs,pos].max(-1).astype(np.float64)
 raw=(library['raw'][obs,pos]*w).sum(-1)/np.maximum(w.sum(-1),1e-12);md=raw-raw[q0]
 mean=float(g.mean());n=float(np.linalg.norm(delta));alpha=float(np.linalg.norm(g*delta)/max(n,1e-12))
 local=d0+g*delta;local[g==0]=d0[g==0];local[g==1]=dT[g==1]
 return {'scoped_R30':local,'samegate_QP':d0+g*md,'global_mean':d0+mean*delta,'global_norm':d0+alpha*delta},{'gate':g,'raw_retrieved':raw,'relative_memory':md,'gate_fraction':mean,'delta_norm':n,'gated_delta_norm':float(np.linalg.norm(g*delta)),'global_norm_alpha':alpha,'q0_gate':float(g[q0])}
def all_scores(p0,pT,libraries,neighbors):
 q0=int(p0.argmax());s={'source':relative_logodds(p0,q0),'R30_ungated':relative_logodds(pT,q0)};process={}
 for key in KEYS:
  v,process[key]=readout_scores(p0,pT,libraries[key],neighbors[key]);prefix='h_' if key=='h288' else 'noun_';s.update({prefix+k:vv for k,vv in v.items()})
 return {a:s[a] for a in ARMS},process

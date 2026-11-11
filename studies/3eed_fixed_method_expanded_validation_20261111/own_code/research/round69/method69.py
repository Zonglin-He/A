"""Unmodified R66v2 equation and R67/R68 random variants on new input rows."""
import sys,torch,numpy as np,hashlib
from common69 import ROOT
sys.path.insert(0,str(ROOT/'research/round66'));import interface66 as F,groups66 as G
from evidence_query66 import matched_controls
from fit_query66 import data,one_step
sys.path.insert(0,str(ROOT/'research/round58'));import numerics58 as N
from readout66 import direct,set_direct
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import cpu_head,readouts
sys.path.insert(0,str(ROOT/'research/round67'));from sampling67 import suffix

def evidence(model,r,cc,raw):
 z={v:{k[2:]:val for k,val in cc.items() if k.startswith(v+'_')} for v in 'AB'};ids=np.unique(np.r_[z['A']['point_indices'],z['B']['point_indices']]);x=np.array(raw[ids,:3],np.float32,copy=True)
 if r['platform']=='drone':x[:,2]+=1.8
 af,_,_=G.affinity(z['A']['boxes'].astype(float),z['B']['boxes'].astype(float),x.astype(float));W=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));base=G.native(z['A']);mass=float((W@G.native(z['B'])).sum());pAB=(base+G.transport(G.native(z['B']),W,mass,base))/2;groups,pool=G.make_groups(z['A']['boxes'].astype(float),x.astype(float),pAB);K=len(pool);members=np.array([np.isin(np.arange(256),groups[k]) for k in pool]);regions={v:np.array([G.support(z[v]['xyz'][0].astype(float),z['A']['boxes'][groups[k]].astype(float)).any(0) for k in pool]) for v in 'AB'};baseline=np.zeros((2,2,256));pert=np.zeros((2,2,2,K,256));valid=np.zeros((2,K),bool);proof=[];calls=0;payload={'B_W':W,'B_fixed_mass':np.array(mass),'base':base,'members':members,'group_ids':np.array(pool,np.int64)}
 for vi,v in enumerate('AB'):
  for ti,cat in enumerate([False,True]):
   y=F.replay(model,z[v],category=cat);calls+=1
   if not cat:
    for k in ['h','query','tokens']:assert np.array_equal(y[k],z[v][k]),('zero',v,k)
   baseline[vi,ti]=y['p'] if v=='A' else G.transport(y['p'],W,mass,base)
  controls=matched_controls(z[v]['xyz'][0].astype(float),regions[v],z[v]['query_points_sample_inds']);payload[v+'_region_masks']=regions[v];payload[v+'_control_masks']=np.array([c for c,ok in controls])
  for j,(control,ok) in enumerate(controls):
   valid[vi,j]=ok and bool(cc['category_available'])
   for ti,cat in enumerate([False,True]):
    for ki,mask in enumerate([regions[v][j],control]):
     if valid[vi,j]:
      y=F.replay(model,z[v],mask,cat);calls+=1;pert[vi,ti,ki,j]=y['p'] if v=='A' else G.transport(y['p'],W,mass,base);proof.append({'view':v,'pool_position':j,'kind':ki,'category':cat,**y['proof']})
     else:pert[vi,ti,ki,j]=baseline[vi,ti]
 drops=np.zeros((2,2,2,K))
 for vi in range(2):
  for ti in range(2):
   for ki in range(2):
    for j in range(K):drops[vi,ti,ki,j]=np.log(max(float(baseline[vi,ti,members[j]].sum()),1e-300))-np.log(max(float(pert[vi,ti,ki,j,members[j]].sum()),1e-300))
 full=drops[:,0,0]-drops[:,0,1];cat=drops[:,1,0]-drops[:,1,1];E=full-cat;pos=valid.all(0)&(E>1e-6).all(0);neg=valid.all(0)&(E< -1e-6).all(0);stable=pos|neg;S=members[pos].any(0) if pos.any() else np.zeros(256,bool);C=members[stable].any(0) if stable.any() else np.zeros(256,bool);payload.update(baseline=baseline,perturbed=pert,drops=drops,valid=valid,E_views=E,E=E.mean(0),full_only=full.mean(0),category_only=cat.mean(0),control_only=(drops[:,0,1]-drops[:,1,1]).mean(0),stable=stable,positive=pos,negative=neg,tested_query_mask=C,positive_query_mask=S,update_allowed=np.array(bool(pos.any() and neg.any())));return payload,proof,calls

def fit(initial,ab,e):
 model=N.head(initial).requires_grad_(True)
 if bool(e['update_allowed']):
  with torch.enable_grad():record=one_step(model,data(ab,e),e)
 else:record={'accepted':False,'scale':0.,'loss_before':0.,'loss_after':0.,'trials':[],'gradient_norm':0.,'step_L2':0.,'before_hash':N.phash(model),'after_hash':N.phash(model),'C_used':False}
 return N.state(model),record

def random_indices(native,key,rep):
 prefix='R67random' if rep==0 else 'R68random_repeat'+str(rep);ss=int(hashlib.sha256((prefix+'|'+key).encode()).hexdigest()[:16],16);base=native[:256].tolist();assert len(set(base))==256;seen=set(base)
 for j in np.random.default_rng(ss).permutation(1024):
  if int(j) not in seen:base.append(int(j));seen.add(int(j))
  if len(base)==512:break
 return np.array(base,np.int64)

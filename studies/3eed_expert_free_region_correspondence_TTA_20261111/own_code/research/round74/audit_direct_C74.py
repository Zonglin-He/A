"""Separate full C-region correspondence and AB representation-reader audit."""
from common74 import *
from scipy.special import softmax,logsumexp
from scipy.spatial import cKDTree
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def main():
 p=read(H/'experiment/PREDICT_PLAN_LOCK_attempt002.json');out=H/'independent_direct_C_audit_attempt001';assert not out.exists();out.mkdir();maxerr=0.;validrows=0
 for r in rows():
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';z=ab(r);c=cview(r);e=load(ROOT/p['evidence_root']/rel);a=load(ROOT/p['prediction_root']/'aux_PRIVATE'/(sid+'.npz'));b=load(ROOT/p['prediction_root']/rel);scores=dict(zip(b['arm_names'],b['scores']));raw=np.fromfile(r['pointpath'],np.float32).reshape(-1,4)[:,:3].copy()
  if r['platform']=='drone':raw[:,2]+=np.float32(1.8)
  ids=np.unique(c['point_indices']);dist,_=cKDTree(raw[ids].astype(float)).query(c['query_anchor_xyz'].astype(float));assert dist.max()<1e-5;boxes=e['region_boxes'];inside=(np.abs(c['query_anchor_xyz'][None]-boxes[:,None,:3])<=boxes[:,None,3:]/2).all(2);PC=inside/np.maximum(inside.sum(1)[:,None],1);assert np.array_equal(PC,a['C_pool']);common=[]
  for box in boxes:
   poola=np.unique(z['A']['point_indices']);poolc=np.unique(c['point_indices']);sa=poola[(np.abs(raw[poola]-box[:3])<=box[3:]/2).all(1)];sc=poolc[(np.abs(raw[poolc]-box[:3])<=box[3:]/2).all(1)];common.append(len(np.intersect1d(sa,sc)))
  good=(inside.sum(1)>0)&(np.array(common)>=4);assert np.array_equal(a['C_shared_raw_counts'],common) and np.array_equal(good,a['C_eligible']);N=e['geometry_negatives']&good[:,None]&good[None,:];assert np.array_equal(N,a['eval_negatives']);validrows+=int(N.any(1).sum())
  W=e['B_W'];qA=unit(z['A']['query'].astype(float));qB=unit(z['B']['query'].astype(float));af=W/(1-W.sum(0))[None];ra=np.clip(af.max(1),0,1);rb=np.clip(af.max(0),0,1);target=unit(qA+ra[:,None]*(W@qB/np.maximum(W.sum(1)[:,None],1e-12)));qa=unit(qA*(1-ra[:,None])+target*ra[:,None]);mapped=W.T@target/np.maximum(W.sum(0)[:,None],1e-12);valid=np.linalg.norm(mapped,axis=1)>1e-12;rb*=valid;mapped=unit(mapped);mapped[~valid]=qB[~valid];qb=unit((1-rb[:,None])*qB+rb[:,None]*mapped)
  pp={}
  for v,q in [('A',qa),('B',qb)]:
   mat=q@z[v]['tokens'].astype(float).T/.07;mask=z[v]['token_mask'];pp[v]=softmax(logsumexp(mat[:,mask],1)-logsumexp(mat[:,~mask],1))
  pb=W@pp['B'];mass=float(e['B_fixed_mass']);pb=mass*pb/max(pb.sum(),1e-300)+(1-mass)*e['base'];expected=(pp['A']+pb)/2;err=float(np.max(abs(expected-scores['frozen_AB_repr'])));maxerr=max(maxerr,err);assert err<1e-12
 rec={'state':'PASS_R74_ALL64_HELDOUT_C_RAW_REGION_IDENTITY_AND_STRONG_AB_DIRECT','n':64,'valid_C_retrieval_anchor_rows':validrows,'max_AB_fusion_probability_error':maxerr,'no_task_GT_or_fit_operations':True};dump(out/'AUDIT_RESULT.json',rec);print(rec)
if __name__=='__main__':main()

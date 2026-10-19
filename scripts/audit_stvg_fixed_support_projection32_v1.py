"""Independent convex KKT/Pythagorean identity and official metric verification."""
from pathlib import Path
import os,sys,collections
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fixed_support_projection32_v1 import D,P,CENTER,SENS,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import numpy as np,torch
 from scipy.special import softmax,logsumexp
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 p=verify();b=read(D/'GLOBAL_PREDICTION_BARRIER.json');z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};cnt=collections.Counter();maxg=maxpyth=maxnull=0.;gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for f in b['files']:assert sha(ROOT/f['path'])==f['sha256']
 for r in p['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));s=load(SENS/'predictions'/(a+'.pt'));c=load(CENTER/'predictions'/(a+'.pt'));fits=dict(Center_live=c['fits']['Fine_live'],**s['fits']);E=s['outputs']['E_MBR'];allow=s['allowed'].numpy()
  for mode,f in fits.items():
   for st,t in enumerate(f['rounds']):
    path=o['paths'][mode][st];valid=t['feedback']['valid'].numpy();active=allow&t['feedback']['informative'].numpy()&valid.any(1);assert np.array_equal(active,path['active']);mu=path['mu'];before=t['eta_after'].numpy();assert np.array_equal(mu[~active],before[~active]);out=E.copy();out[:,:2]+=.25*E[:,2:]*mu;out[~allow]=E[~allow];assert np.array_equal(out,o['outputs'][mode+'_project'+str(st+1)])
    cert={v['node']:v for v in path['certificates']}
    for i in np.flatnonzero(active):
     A=t['standardized_candidates'][i].numpy()[valid[i]];q=t['feedback']['prob'][i].numpy()[valid[i]];_,sing,V=np.linalg.svd(A-A[0],full_matrices=False);rank=int((sing>1e-12*max(1.,sing[0]if len(sing)else 0.)).sum());V=V[:rank];h=-.5*(A*A).sum(-1);oldlog=h+A@before[i];oldlog-=logsumexp(oldlog);newlog=h+A@mu[i];newlog-=logsumexp(newlog);prob=np.exp(newlog);g=V@((prob-q)@A);er=float(abs(g).max())if rank else 0.;assert er<=1.01e-10;maxg=max(maxg,er);nu=mu[i]-V.T@(V@mu[i]);oldnu=before[i]-V.T@(V@before[i]);en=float(np.linalg.norm(nu-oldnu));maxnull=max(maxnull,en);assert en<1e-10
     klold=float(np.sum(q*(np.log(q)-oldlog)));klnew=float(np.sum(q*(np.log(q)-newlog)));klproj=float(np.sum(prob*(newlog-oldlog)));err=abs(klold-klnew-klproj);maxpyth=max(maxpyth,err);assert err<1e-8
     assert abs(klnew-cert[i]['KL_optimum'])<1e-10 and klnew<=klold+1e-12;cnt['KKT_null_and_Pythagorean_certificates']+=1
    cnt['frozen_support_counterfactual_paths']+=1
  for name,v in s['outputs'].items():assert np.array_equal(o['outputs'][name],v);cnt['original_outputs_exact_reuse']+=1
  for out in o['outputs'].values():assert np.array_equal(out[~allow],E[~allow])and np.array_equal(out[:,2:],E[:,2:]);cnt['scope_sizes_planned_exact']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rows[a],cnt)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,ep in g['paired'].items():
   a,b=key.split('_minus_')
   for e,want in ep.items():assert paired([None if r['values'][a][e]is None or r['values'][b][e]is None else 100*(r['values'][a][e]-r['values'][b][e])for r in rr])==want;cnt['parent_bootstrap']+=1
 put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(cnt),max_projected_gradient=maxg,max_null_preservation_error=maxnull,max_Pythagorean_identity_error=maxpyth,KKT_tolerance_certificate_not_strict_duality_gap=True,no_new_model_or_teacher=True));print(dict(cnt),maxg,maxpyth,flush=True)
if __name__=='__main__':main()

"""One fixed QCQP test: old response M0, KL-budget projection M1, target M2.

No change to the frozen paper pipeline. Dense Jacobians are engineering here;
no learned network, GT tuning, extra Teacher, schedule or finite shrink search.
"""
from pathlib import Path
import copy,hashlib,inspect,json,math,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write,guard
BASE=ROOT/'artifacts/stvg_output_budget32_v1';OLD=ROOT/'artifacts/stvg_opd_anneal_and_tube64_v1'
def put(p,d,replace=False):return write(p,d,mutable=replace)
def verify():
 r=read(BASE/'EXECUTION_RUNTIME001.json');assert time.time()<r['deadline_epoch']
 for p,h in r['pins'].items():assert sha(ROOT/p)==h,p
 return r

def quadratic(A,U,target,d,kind,epsilon=.01,gamma=1.):
 """Solve SPD QCQP using output-space Woodbury and a bounded scalar dual."""
 import numpy as np
 from scipy.linalg import cho_factor,cho_solve
 A,U,target,d=[np.asarray(x,dtype=np.float64) for x in (A,U,target,d)];n=len(d);assert A.shape[1]==U.shape[1]==n and epsilon>0 and gamma>0
 if kind=='M1':F=np.empty((0,n));rhs=d;g=1.
 else:assert kind=='M2';F=A;rhs=A.T@target;g=gamma
 def solve(lam):
  B=np.concatenate([F,math.sqrt(lam)*U]);x=rhs/g if not len(B) else (rhs-B.T@cho_solve(cho_factor(g*np.eye(len(B))+B@B.T,lower=True),B@rhs))/g;cost=float(np.sum((U@x)**2)/2);return x,cost
 z,cost=solve(0.);lo=hi=0.;iterations=0
 if cost>epsilon*(1+1e-10):
  hi=1.
  for _ in range(40):
   z,cost=solve(hi);iterations+=1
   if cost<=epsilon:break
   hi*=2
  else:raise RuntimeError('fixed dual bracket exhausted')
  for _ in range(24):
   mid=(lo+hi)/2;x,c=solve(mid);iterations+=1
   if c>epsilon:lo=mid
   else:hi=mid;z,cost=x,c
 lam=hi;grad=g*z+F.T@(F@z)-rhs+lam*U.T@(U@z);res=float(np.linalg.norm(grad)/max(1.,np.linalg.norm(rhs)));assert res<1e-6 and cost<=epsilon*(1+1e-8)
 return z,dict(status='pass',kind=kind,epsilon=epsilon,gamma=g,lambda_dual=lam,dual_bracket=[lo,hi],linear_KL=cost,stationarity_relative=res,dual_iterations=iterations,active_constraint=lam>0.,linear_target_error=float(np.linalg.norm(A@z-target)),solution=z.tolist())

def actual_KL(before,after,U,sigma):
 import torch
 if not U:return 0.
 b,a=before[U],after[U]
 if not bool(torch.isfinite(a).all() and ((a>0)&(a<1)).all()):return math.inf
 return float((torch.logit(a)-torch.logit(b)).double().square().sum()/(2*sigma*sigma*len(U)))

def direction(boxes,params,adam,policy,Apos,Upos,sigma,kind):
 import torch,numpy as np
 assert bool(((boxes>0)&(boxes<1)).all())
 # Every row differentiates the actual decoder graph; no detached box optimizer.
 out=torch.logit(boxes).flatten();rows=[]
 for k in range(len(out)):
  grads=torch.autograd.grad(out[k],params,retain_graph=True,allow_unused=True);rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)]).detach().cpu().double().numpy())
 J=np.stack(rows).reshape(len(boxes),4,-1);JA=J[Apos].reshape(-1,J.shape[-1])/(sigma*math.sqrt(len(Apos)));JU=J[Upos].reshape(-1,J.shape[-1])/(sigma*math.sqrt(len(Upos))) if Upos else np.empty((0,J.shape[-1]));target=(policy['weights'][:,:,None]*(policy['samples']-policy['mean'][:,None,:])).sum(1).detach().cpu().double().numpy().reshape(-1)/(sigma*math.sqrt(len(Apos)));d=adam.detach().cpu().double().numpy();z,proof=quadratic(JA,JU,target,d,kind)
 # Matrices are private audit payload; only derived scalar receipts are exported.
 proof.update(JA=JA,JU=JU,target=target,adam=d,GT_read=False)
 return torch.as_tensor(z,device=adam.device,dtype=adam.dtype),proof

def fitter(kind):
 from scripts import stvg_spatial_lite_fit_v1 as lite
 src=inspect.getsource(lite.fit);start=src.index('                    solver = None');end=src.index('                    if qualification and step == 0',start)
 src=src[:start]+'''                    _sync(device)
                    solver_started = time.perf_counter()
                    direction, solver = new_direction(before_graph, params, adam_delta, policy, active_positions, protection, cfg["sigma"], NEW_KIND)
                    _sync(device)
                    solver_seconds = time.perf_counter() - solver_started
                    solver_seconds_total += solver_seconds
                    solver_calls += 1

'''+src[end:]
 needle='candidate_accepted = finite and reference_pass';assert src.count(needle)==1;src=src.replace(needle,'actual_cost = output_KL(boxes_before, candidate_boxes, protection, cfg["sigma"])\n                        budget_pass = actual_cost <= .01 * (1 + 1e-7)\n                        candidate_accepted = finite and reference_pass and budget_pass')
 src=src.replace('"reference_pass": bool(reference_pass),','"reference_pass": bool(reference_pass), "actual_output_KL": actual_cost if math.isfinite(actual_cost) else None, "output_budget_pass": budget_pass,')
 ns=dict(lite.__dict__);ns.update(new_direction=direction,output_KL=actual_KL,NEW_KIND=kind);exec(compile(src,__file__,'exec'),ns);return ns['fit']

def contracts():
 import numpy as np
 from scipy.optimize import minimize
 rng=np.random.default_rng(20261017);checks=0
 for kind in ('M1','M2'):
  for k in range(4):
   A=rng.normal(size=(3,7));U=rng.normal(size=(4,7));v=rng.normal(size=3);d=rng.normal(size=7);z,p=quadratic(A,U,v,d,kind)
   fun=(lambda x:.5*np.sum((x-d)**2)) if kind=='M1' else (lambda x:.5*np.sum((A@x-v)**2)+.5*np.sum(x*x))
   result=minimize(fun,np.zeros(7),method='SLSQP',constraints=[dict(type='ineq',fun=lambda x:.01-.5*np.sum((U@x)**2))],options=dict(ftol=1e-11,maxiter=1000));assert result.success and abs(fun(z)-result.fun)<1e-5;checks+=1
 fitter('M1');fitter('M2');return dict(status='pass',count=checks,independent_SLSQP_objective_agreement=True,CPU_only=True,GT_read=False)

def setup():
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 import torch,numpy as np
 BASE.mkdir(exist_ok=False);rr=read(OLD/'ROSTER_LOCK.json')['records'];chosen=[r for ds in ('hc2','vidstg') for r in [r for r in rr if r['dataset']==ds][:16]];put(BASE/'ROSTER_LOCK.json',dict(status='locked',count=32,per_domain=16,records=chosen,selection='first16 original parent-hash positions per domain from prelocked development64, before chart/fit/result inspection',GT_read=False));index={r['case_id']:r for r in read(OLD/'INPUT_LOCK.json')['records']};rows=[]
 for r in chosen:
  cid=r['probe_alias'];x=index[cid];c,_=load_case(r);b=c['zero']['boxes'];valid=bool(((b>0)&(b<1)).all());rows.append({**x,'whole_native_logit_chart_valid':valid,'GT_read':False});del c;gc=__import__('gc');gc.collect()
 put(BASE/'INPUT_LOCK.json',dict(status='locked',count=32,records=rows,GT_read=False));put(BASE/'CPU_CONTRACT_COMPLETION.json',contracts());put(BASE/'METHOD_LOCK.json',dict(status='locked',count=32,epsilon_S=.01,gamma=1.,same_budget_both_domains_and_steps=True,duality='M1 Euclidean projection of original raw Adam direction; M2 normalized OPD target Jacobian QCQP with same U KL budget',M0='exact sealed old original-response OPD, not plain-Adam ablation',M1_M2='same original finite/reference acceptance AND actual nonlinear U KL<=epsilon; original fullstep and reject rollback/early-stop, no extra shrink schedule',invalid_logit='whole Query Source no-op retained; pre-adaptation chart-valid diagnostic separate',params=1792,Gaussian_samples=32,steps_sigma_tau_LR_and_references_original=True,M2_LR_note='new target solver replaces direction; original Adam trace retained but its LR is not an effective M2 direction parameter',no_GT_tuning=True,no_new_Teacher_or_Temporal=True,not_paper_method=True,one_fixed_round_only=True,GT_read=False))
 paths=[Path(__file__),ROOT/'scripts/run_stvg_output_budget32_v1.py',ROOT/'scripts/score_stvg_output_budget32_v1.py',ROOT/'scripts/continue_stvg_output_budget32_v1.py',BASE/'ROSTER_LOCK.json',BASE/'INPUT_LOCK.json',BASE/'METHOD_LOCK.json',ROOT/'scripts/stvg_spatial_lite_fit_v1.py',ROOT/'scripts/stvg_soft_ep_response_v1.py',ROOT/'methods/CURRENT_METHOD.json'];put(BASE/'EXECUTION_RUNTIME001.json',dict(status='locked',count=32,pins={str(p.relative_to(ROOT)):sha(p) for p in paths},deadline_epoch=time.time()+57600,compute_seconds_cap=14400,maximum_new_fit_calls=64,wait_for='stvg-paper-t2-b-001.service normal exit and256 complete barriers; no outcome gate',GT_read=False));put(BASE/'STATUS.json',dict(status='prepared',stage='await_paper_integrated_measurement',count=32,GT_read=False));print('OUTPUT_BUDGET32_LOCKED')
if __name__=='__main__':setup()

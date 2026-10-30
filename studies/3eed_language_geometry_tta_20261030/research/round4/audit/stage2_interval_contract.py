"""Independent synthetic convex interval/anchor/cap oracle, no data/GT inputs."""
import numpy as np,json,pathlib,hashlib
from scipy.optimize import minimize
R=pathlib.Path(__file__).resolve().parent

def analytic(c0,lower,upper,observed,lam=.25,cap=.15):
 c0=np.asarray(c0,float);observed=np.asarray(observed,bool);d=(np.clip(c0,lower,upper)-c0)/(1+lam);d=np.where(observed,d,0.);n=np.linalg.norm(d)
 if n>cap:d*=cap/n
 return c0+d

def main():
 rng=np.random.default_rng(10);checks=[];solver_status=[];kkt=[]
 for i in range(24):
  c0=rng.normal(size=3);mid=c0+rng.uniform(-.4,.4,3);width=rng.uniform(.01,.2,3);lo=mid-width;hi=mid+width;obs=rng.random(3)>.25
  expected=analytic(c0,lo,hi,obs)
  def objective(d):
   c=c0+d;v=np.maximum(np.maximum(lo-c,c-hi),0.)*obs;return float(v@v+.25*(d@d))
  result=minimize(objective,np.zeros(3),method='SLSQP',bounds=[(-.15,.15) if obs[j] else (0.,0.) for j in range(3)],constraints=[{'type':'ineq','fun':lambda d:.15**2-float(d@d)}],options={'ftol':1e-13,'maxiter':200})
  d=expected-c0;c=expected;gradient=2*np.where(c<lo,c-lo,np.where(c>hi,c-hi,0.))*obs+.5*d
  multiplier=max(0.,-float(gradient@d)/(2*float(d@d))) if np.linalg.norm(d)>.15-1e-10 else 0.
  residual=float(np.max(abs(gradient+2*multiplier*d)));assert residual<1e-10;kkt.append(residual)
  solver_status.append({'case':i,'success':bool(result.success),'message':str(result.message),'objective_difference':float(objective(expected-c0)-result.fun)})
  assert abs(objective(expected-c0)-result.fun)<1e-10
  assert np.linalg.norm(expected-c0)<=.15+1e-12 and np.array_equal(expected[~obs],c0[~obs]);checks.append(abs(objective(expected-c0)-result.fun))
 out={'status':'passed_synthetic_only','cases':24,'max_objective_difference_vs_independent_SLSQP':max(checks),'analytic_KKT_stationarity_max':max(kkt),'comparator_solver_status':solver_status,'comparator_status_failures_retained':sum(not x['success'] for x in solver_status),'definition':'min sum_observed dist(center_axis,[lower,upper])^2 +.25norm(center-source)^2 subject physicalL2cap.15; unobserved axes frozen','proof_condition':'All active observed axes have same curvature1.25. Unconstrained optimum moves toward nearest interval bound but never crosses it; radial clipping therefore is exact. Different peraxis weights/extentnormalization would break this argument.','no_real_targets_points_or_models':True,'source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()};(R/'STAGE2_INTERVAL_CONTRACT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()

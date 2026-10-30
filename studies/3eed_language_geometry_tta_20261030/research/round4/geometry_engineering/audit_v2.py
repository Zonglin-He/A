"""Independent SYNTHETIC-only teacher and convex center-latent engineering checks.
No real point inputs, annotations, embeddings, GPU or model imports are used.
"""
from pathlib import Path
import json,hashlib,importlib.util
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree as IndependentTree
R=Path(__file__).resolve().parent;MODULE=R.parents[2] / 'crossplatform/round4/adaptation/geometry_update.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module():
 spec=importlib.util.spec_from_file_location('audited_geometry_update',MODULE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def main():
 code_before=sha(MODULE);m=module();source=np.array([0.,0.,0.,4.,2.,1.,0.]);x,y=np.meshgrid(np.linspace(-1.6,1.6,41),np.linspace(-.8,.8,21));roof=np.c_[x.ravel(),y.ravel(),np.full(x.size,.6)]
 y,z=np.meshgrid(np.linspace(-.8,.8,33),np.linspace(-.4,.4,17));wall=np.c_[np.full(y.size,2.2),y.ravel(),z.ravel()]
 cases=[('roof_camera_above',roof,[0,0,5],[False,False,True]),('frontwall_camera_front',wall,[10,0,0],[True,False,False]),('ground_camera_above',roof*np.array([1,1,-1]),[0,0,5],[False]*3),('backwall_camera_front',wall*np.array([-1,1,1]),[10,0,0],[False]*3),('camera_inside_vertical_slab',roof,[0,0,.1],[False]*3),('camera_inside_horizontal_slab',wall,[.1,0,0],[False]*3),('camera_exact_source_center',roof,[0,0,0],[False]*3),('camera_exact_vertical_slab_boundary',roof,[0,0,.5],[False]*3),('fewer_than8_points',roof[:7],[0,0,5],[False]*3)]
 # Two separated positive-Z planes have valid unsigned normals but high plane MAD.
 bimodal=np.r_[roof*np.array([1,1,1/6]),roof];cases.append(('parallel_faces_MAD_majority_limitation',bimodal,[0,0,5],None))
 jitter=np.random.default_rng(811).normal(0,.001,(len(roof),2));jroof=roof.copy();jroof[:,:2]+=jitter;jboth=np.r_[jroof*np.array([1,1,1/6]),jroof];cases.append(('balanced_parallel_faces_highMAD_gate',jboth,[0,0,5],[False]*3))
 teacher=[]
 for name,points,camera,expected in cases:
  lo,hi,obs,diag=m.interval_teacher(points,source,np.array(camera));teacher.append({'case':name,'expected_observed':expected,'actual_observed':obs.tolist(),'pass':bool(np.array_equal(obs,expected)) if expected is not None else None,'lower':lo.tolist(),'upper':hi.tolist(),'diagnostics':diag})
 # Instrument the KDTree interface to establish its graph uses full cloud, including uncropped points.
 graph={};allpoints=np.r_[roof,np.array([[20.+i,20.,20.] for i in range(31)])]
 class SpyTree:
  def __init__(self,p):graph['tree_points']=len(p);self.tree=IndependentTree(p)
  def query(self,q,k):graph['query_points']=len(q);graph['k']=k;return self.tree.query(q,k=k)
 original=m.cKDTree
 try:m.cKDTree=SpyTree;m.interval_teacher(allpoints,source,np.array([0,0,5]))
 finally:m.cKDTree=original
 graph['full_cloud_graph_pass']=graph['tree_points']==len(np.unique(allpoints,axis=0)) and graph['query_points']<graph['tree_points'] and graph['k']==16
 # Independent convex numerical solver. Equal metric and quarter-strength center prior.
 # We independently form squared distance to intervals; use explicit L2 ball constraint.
 rng=np.random.default_rng(3401);numeric=[]
 for j in range(64):
  center=rng.uniform(-1,1,3);half=rng.uniform(.02,.4,3);lo=center-half;hi=center+half;obs=rng.random(3)>.3;active=np.flatnonzero(obs);direct=m.direct_center(lo,hi,obs)
  def obj(v):
   d=np.zeros(3);d[active]=v;dist=d-np.minimum(np.maximum(d,lo),hi);return float((dist[obs]**2).sum()+.25*(d**2).sum())
  def jac(v):
   d=np.zeros(3);d[active]=v;dist=d-np.minimum(np.maximum(d,lo),hi);return (2*dist+.5*d)[active]
  if len(active):
   result=minimize(obj,np.zeros(len(active)),jac=jac,method='SLSQP',constraints=[{'type':'ineq','fun':lambda v:.15**2-float(v@v),'jac':lambda v:-2*v}],options={'ftol':1e-13,'maxiter':500});numeric_delta=np.zeros(3);numeric_delta[active]=result.x;success=bool(result.success);original_numeric_norm=float(np.linalg.norm(numeric_delta));
   if original_numeric_norm>.15:numeric_delta*=.15/original_numeric_norm
   value=obj(numeric_delta[active])
  else:numeric_delta=np.zeros(3);success=True;value=obj(np.array([]))
  diff=float(np.max(np.abs(direct-numeric_delta)));gap=float(obj(direct[active])-value);numeric.append({'fixture':j,'solver_success':success,'numeric_solution_feasibility_restored_by_radial_projection_if_needed':True,'max_delta_error_m':diff,'objective_gap_analytic_minus_numeric':gap,'analytic_L2':float(np.linalg.norm(direct)),'unobserved_exact_zero':bool(np.all(direct[~obs]==0)),'pass':bool(diff<2e-6 and abs(gap)<1e-9 and np.linalg.norm(direct)<=.15000000001 and np.all(direct[~obs]==0))})
 # Episodic/control invariance, fixed shape/PCA yaw and local-to-world conversion.
 source_yaw=np.array([2.,3.,4.,4.,2.,1.,.7]);lo=np.array([.17,-.3,0.]);hi=np.array([.23,-.2,0.]);obs=np.array([True,True,False]);runs,trace=m.run_center(source_yaw,lo,hi,obs);control,_=m.run_center(source_yaw,lo,hi,obs,control=True)
 invariants={'zero_step_source_exact':np.array_equal(runs[0],source_yaw),'shape_and_yaw_exact_all_steps':all(np.array_equal(v[3:],source_yaw[3:]) for v in runs.values()),'source_control_exact_all_steps':all(np.array_equal(v,source_yaw) for v in control.values()),'unobserved_local_axis_zero_all_steps':all(abs(((v[:3]-source_yaw[:3])@m.rotation(source_yaw[6]))[2])==0 for v in runs.values()),'projected_physical_L2_cap':all(np.linalg.norm(v[:3]-source_yaw[:3])<=.1500000001 for v in runs.values())}
 result={'scope':'synthetic engineering only; no GT/model/GPU/input caches','audited_code_sha256':sha(MODULE),'teacher_cases':teacher,'full_cloud_knn_instrumentation':graph,'convex_numeric_cases':numeric,'direct_analytic_numeric_all_pass':all(x['pass'] for x in numeric),'direct_max_delta_error_m':max(x['max_delta_error_m'] for x in numeric),'direct_max_abs_objective_gap':max(abs(x['objective_gap_analytic_minus_numeric']) for x in numeric),'invariants':invariants,'all_invariants_pass':all(invariants.values()),'all_teacher_cases_pass':all(x['pass'] for x in teacher if x['expected_observed'] is not None),'audit_script_sha256':sha(Path(__file__)),'code_unchanged_during_audit':sha(MODULE)==code_before}
 p=R/('AUDIT_V2_'+sha(MODULE)[:12]+'.json');assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'file':str(p),'code_sha256':sha(MODULE),'teacher_pass':result['all_teacher_cases_pass'],'failed_teacher_cases':[x['case'] for x in teacher if x['expected_observed'] is not None and not x['pass']],'graph_pass':graph['full_cloud_graph_pass'],'numeric_all_pass':result['direct_analytic_numeric_all_pass'],'direct_max_delta_error':result['direct_max_delta_error_m'],'max_objective_gap':result['direct_max_abs_objective_gap'],'invariants_all_pass':result['all_invariants_pass']},indent=2))
if __name__=='__main__':main()

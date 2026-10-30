"""Synthetic mechanisms/zero-information contracts; no realdata/model/GT."""
from pathlib import Path
import json,hashlib,numpy as np,torch
from scipy.optimize import minimize
from query_update import cached_views,source_aug,run_query,probability
from geometry_update import interval_teacher,direct_center,run_center
R=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);rng=np.random.default_rng(11);full=[]
 boxes=np.c_[rng.normal(size=(8,3)),np.full((8,3),2)].astype('float32')
 for v in range(3):
  q=torch.tensor(rng.normal(size=(8,64)),dtype=torch.float32);q=torch.nn.functional.normalize(q,dim=-1);t=torch.tensor(rng.normal(size=(12,64)),dtype=torch.float32);t=torch.nn.functional.normalize(t,dim=-1);L=q@t.T/.07;mask=np.zeros(12,bool);mask[1:3]=True;s=(L.softmax(-1)*torch.as_tensor(mask)[None]).sum(-1)
  full.append({'boxes':boxes,'proj_queries':q.numpy(),'proj_tokens':t.numpy(),'exact_logits':L.numpy(),'noun_mask':mask,'semantic_scores':s.numpy(),'aligned_to_full_view0':np.arange(8)})
 views,w=cached_views(full);base=source_aug(views,w);p=probability(base).numpy();teacher=.1*p;teacher[7]+=.9
 out,d=run_query(views,w,teacher,True);assert np.array_equal(out[0],base.numpy());assert d['trace'][-1]['parameter_delta_l2']>0 and not np.array_equal(out[3],out[0]);assert all(x['max_embedding_total_L2']<=.100001 and x['max_embedding_step_L2']<=.050001 for x in d['trace'])
 control,c=run_query(views,w,teacher,True,control=True);assert all(np.array_equal(v,base.numpy()) for v in control.values()) and all(x['parameter_delta_l2']==0 and x['gradient_norm']==0 for x in c['trace'])
 # A visibleplane atx=-1.9 supportsonecenteraxis with sourceextent4; no shrink.
 y,z=np.meshgrid(np.linspace(-1,1,25),np.linspace(-1,1,25));points=np.c_[np.full(y.size,-1.9),y.ravel(),z.ravel()];source=np.array([0.,0,0,4,2,2,0],dtype='float32');lo,hi,obs,g=interval_teacher(points,source,np.array([-10.,0,0]));assert obs[0] and not obs[1:].any(),g
 direct=direct_center(lo,hi,obs);pose,pdiag=run_center(source,lo,hi,obs);assert np.array_equal(pose[0],source);assert np.array_equal(pose[3][3:],source[3:]);assert pdiag['trace'][-1]['latent_delta_L2_m']>0;assert all(t['total_loss_after']<=t['total_loss_before']+1e-12 for t in pdiag['trace'])
 npose,nd=run_center(source,lo,hi,obs,control=True);assert all(np.array_equal(x,source) for x in npose.values()) and all(x['gradient_norm']==0 for x in nd['trace'])
 # Camera inside source Xslab cannot qualifythat plane regardlessof normal/support.
 _,_,inside,inside_d=interval_teacher(points,source,np.array([.5,0,0]));assert not inside.any(),inside_d
 # Independentconvex oracle validates direct interval+prior+ball optimum.
 direct_errors=[]
 for k in range(15):
  center=rng.uniform(-.5,.5,3);width=rng.uniform(.02,.1,3);l=center-width;h=center+width;m=rng.random(3)>.25;delta=direct_center(l,h,m)
  def obj(x):return np.sum((np.maximum(np.maximum(l-x,x-h),0)*m)**2)+.25*np.sum(x*x)
  res=minimize(obj,np.zeros(3),method='SLSQP',bounds=[(None,None) if a else (0,0) for a in m],constraints=[{'type':'ineq','fun':lambda x:.15**2-x@x}],options={'maxiter':200,'ftol':1e-12});err=abs(obj(delta)-res.fun);assert err<1e-8,(k,err,res);direct_errors.append(err)
 tiny=np.array([.005,0,0]);tiny_hi=np.array([.015,0,0]);_,tiny_diag=run_center(source,tiny,tiny_hi,np.array([True,False,False]));assert all(t['total_loss_after']<=t['total_loss_before']+1e-12 for t in tiny_diag['trace']);assert any(t['backtrack_halvings']>0 for t in tiny_diag['trace'])
 report={'tiny_interval_backtracking_validated':True,'state':'passed','scope':'synthetic noactualmodel/data/GT/GPU','query512_real_grad_parameter_and_outputs_change':True,'query0_exact_TTAug_bridge':True,'query_episodic_reset_exact':True,'no_spatial_gradient_parameter_output_zero':True,'embedding_step_total_caps':[.05,.1],'visible_singleface_axis_support':True,'camera_inside_slab_rejected':True,'geometry_size_and_PCAyaw_exact':True,'geometry3_real_center_update':True,'geometry_no_observation_zero_gradient_and_drift':True,'direct_interval_vs_independent_convex_oracle_cases':15,'max_objective_difference':max(direct_errors),'query_diagnostics':d,'geometry_diagnostics':pdiag,'code_hashes':{p.name:hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [R/'interfaces.py',R/'caption_regions.py',R/'spatial_teacher.py',R/'query_update.py',R/'geometry_update.py']}}
 (R/'MECHANISM_FIXTURES.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['query_diagnostics','geometry_diagnostics']},indent=2))
if __name__=='__main__':main()

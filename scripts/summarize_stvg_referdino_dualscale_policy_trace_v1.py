"""Aggregate exploratory geometry and teacher weight mass, without GT.

Process statistics, never selection scores or localization-accuracy claims.
No raw boxes, textual queries or parent identifiers enter the public export.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_referdino_dualscale_hc128_v1 import D,P,verify

def main():
 import torch
 torch.set_num_threads(4);p=verify(True);assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==128;start=time.perf_counter();groups={}
 for r in p['records']:
  z=load(D/'predictions'/(r['alias']+'.pt'))
  for name,fit in z['fits'].items():
   policy=name.split('_')[0];key=policy+'_'+fit['arm']
   if key not in groups:groups[key]=dict(fits=0,rounds=0,candidates=0,outside_image=0,zero_image_overlap=0,far_mass=0.,ESS=0.,near_sq=np.zeros(4),far_sq=np.zeros(4),component_candidates=0,gradient_norm_sum=0.,gradient_norm_max=0.,initial_anchor_cost=0.,final_anchor_cost=0.,actual_chart_max_abs_error=0.)
   g=groups[key];g['fits']+=1;g['initial_anchor_cost']+=fit['path'][0]['anchor_loss'];g['final_anchor_cost']+=fit['path'][-1]['anchor_loss']
   for step in fit['rounds']:
    rr=step['rollout'];bs=fit['path'][step['step']]['boxes'][fit['positions']].double();ch=torch.cat((bs[:,:2],bs[:,2:].log()),-1);ce=float((ch-step['current_policy_mean']).abs().max());assert ce<1e-12;g['actual_chart_max_abs_error']=max(g['actual_chart_max_abs_error'],ce);w=rr['weights'].double();delta=(rr['actions'].double()-rr['proposal_mean'].double()[:,None])/rr['D'].double()[:,None];g['rounds']+=1;g['candidates']+=w.numel();g['outside_image']+=rr['outside_image'];g['zero_image_overlap']+=rr['zero_image_overlap'];g['far_mass']+=float(w[:,16:].sum(-1).mean());g['ESS']+=float((1/w.square().sum(-1)).mean());g['near_sq']+=delta[:,:16].square().sum((0,1)).numpy();g['far_sq']+=delta[:,16:].square().sum((0,1)).numpy();g['component_candidates']+=delta.shape[0]*16
    if step['gradient']is not None:
     norm=float(step['gradient'].norm());g['gradient_norm_sum']+=norm;g['gradient_norm_max']=max(g['gradient_norm_max'],norm)
 out={}
 for name,g in groups.items():
  out[name]=dict(fits=g['fits'],rounds=g['rounds'],candidates=g['candidates'],outside_image_fraction=g['outside_image']/g['candidates'],zero_image_overlap_fraction=g['zero_image_overlap']/g['candidates'],positive_size_finite_candidates_verified=True,teacher_far_component_mean_probability_mass=g['far_mass']/g['rounds'],teacher_mean_ESS=g['ESS']/g['rounds'],near_normalized_delta_std_by_coord=np.sqrt(g['near_sq']/g['component_candidates']).tolist(),far_normalized_delta_std_by_coord=np.sqrt(g['far_sq']/g['component_candidates']).tolist(),mean_content_gradient_L2=g['gradient_norm_sum']/g['rounds'],maximum_content_gradient_L2=g['gradient_norm_max'],mean_initial_anchor_geometric_cost=g['initial_anchor_cost']/g['fits'],mean_final_anchor_geometric_cost=g['final_anchor_cost']/g['fits'],actual_chart_max_abs_error=g['actual_chart_max_abs_error'])
 assert sum(g['fits']for g in groups.values())==2304;assert sum(g['rounds']for g in groups.values())==92160;assert sum(g['candidates']for g in groups.values())==23592960
 put(P/'POLICY_TRACE_DIAGNOSTICS_PUBLIC.json',dict(status='completed_process_diagnostics',groups=out,not_used_for_policy_selection=True,no_GT_access=True,no_new_model_calls=True,not_a_teacher_calibration_or_task_accuracy_claim=True,far_mass_definition='sum q over the16 far-generated candidate indices, NOT latent mixture responsibility or native teacher probabilities',CPU_seconds=time.perf_counter()-start));audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');audit['schema_notes']=dict(native_chart='unused legacy schema counter; actual corrected cxcylogwh chart checked per round with1e-12 tolerance',actual_chart_max_abs_error=max(g['actual_chart_max_abs_error']for g in groups.values()));put(P/'ROOT_INDEPENDENT_AUDIT.json',audit,True);assert not torch.cuda.is_initialized();print('PROCESS DIAGNOSTICS',out,flush=True)
if __name__=='__main__':main()

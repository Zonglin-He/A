"""Matched parameter effects and process/teacher-quality explanations, offline only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,itertools,time,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import ROOT,D,P,BASE,read,put,sha,load,verify
from scripts.audit_stvg_spatial_strong_bidirectional256_v1 import array,giou

def boot(x,indices):
 x=np.array(x,float);return dict(mean=float(x.mean()),CI95=np.quantile(x[indices].mean(1),[.025,.975]).tolist(),positive=int((x>1e-10).sum()),negative=int((x< -1e-10).sum()))
def main(ds):
 import torch
 torch.set_num_threads(2)
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 p=verify();r=read(P/ds/'coarse/RESULTS_PUBLIC.json');assert read(P/ds/'coarse/ROOT_AUDIT_PUBLIC.json')['status']=='pass';bar=read(D/ds/'COARSE_FIT_BARRIER.json');records=[x for x in p['records']if x['dataset']==ds];rows={x['alias']:x for x in r['rows']};defs=r['definitions'];rng=np.random.default_rng(20261020);idx=rng.integers(0,128,(10000,128));lookup={(c['lr'],c['tau'],c['steps'],c['alpha']):a for a,c in defs.items()};effects=[];beg=time.perf_counter()
 def add(factor,a,b,fixed):
  x=[100*(v['values'][a]['v']-v['values'][b]['v'])for v in r['rows']];u=[100*(v['values'][a]['unobserved_actual_sIoU']-v['values'][b]['unobserved_actual_sIoU'])for v in r['rows']];effects.append(dict(factor=factor,new=a,reference=b,fixed=fixed,full_vIoU_pp=boot(x,idx),unobserved_sIoU_pp=boot(u,idx)))
 for lr,k,alpha in itertools.product([.003,.01,.03],p['coarse_steps'],p['coarse_alphas']):add('tau_0.5_minus_0.2',lookup[lr,.5,k,alpha],lookup[lr,.2,k,alpha],dict(lr=lr,steps=k,alpha=alpha))
 for tau,k,alpha,(lo,hi)in itertools.product([.2,.5],p['coarse_steps'],p['coarse_alphas'],[(.003,.01),(.01,.03),(.003,.03)]):add('lr_'+str(hi)+'_minus_'+str(lo),lookup[hi,tau,k,alpha],lookup[lo,tau,k,alpha],dict(tau=tau,steps=k,alpha=alpha))
 for lr,tau,alpha,(lo,hi)in itertools.product([.003,.01,.03],[.2,.5],p['coarse_alphas'],[(10,20),(20,30),(30,40)]):add('steps_'+str(hi)+'_minus_'+str(lo),lookup[lr,tau,hi,alpha],lookup[lr,tau,lo,alpha],dict(lr=lr,tau=tau,alpha=alpha))
 for lr,tau,k,(lo,hi)in itertools.product([.003,.01,.03],[.2,.5],p['coarse_steps'],[(.25,.5),(.5,1.),(1.,1.5),(1.5,2.)]):add('alpha_'+str(hi)+'_minus_'+str(lo),lookup[lr,tau,k,hi],lookup[lr,tau,k,lo],dict(lr=lr,tau=tau,steps=k))
 gt,spans,_=truths(ds,dict(parents=[x['query_ordinal']for x in records]));raw=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'];process={};quality=[];taucos=[]
 for rec in records:
  alias,q=rec['alias'],rec['query_ordinal'];old=load(BASE/f'predictions/{ds}/{alias}.pt');inp=load(BASE/f'inputs/{ds}/{alias}.pt');ids=old['frame_ids'];positions=old['planned_positions'];E=array(old['E_Ref']);target=array(inp['teacher_native']['boxes']);row=raw[q];pix=xyxy(target,row['input']['width'],row['input']['height']);epi=xyxy(E,row['input']['width'],row['input']['height']);pix=np.maximum(pix,0)if ds=='hc2'else pix;epi=np.maximum(epi,0)if ds=='hc2'else epi;valid=[j for j,pos in enumerate(positions)if ids[pos]in gt[q]];ti=[float(box_iou(pix[j],gt[q][ids[positions[j]]]))for j in valid];ei=[float(box_iou(epi[positions[j]],gt[q][ids[positions[j]]]))for j in valid]
  quality.append(dict(alias=alias,GT_available_anchor_count=len(valid),teacher_actual_anchor_IoU=float(np.mean(ti))if ti else None,ERef_same_anchor_IoU=float(np.mean(ei))if ei else None,teacher_minus_ERef_anchor_pp=100*float(np.mean(np.array(ti)-ei))if ti else None,ERef_full_vIoU=rows[alias]['values']['E_Strong']['v'],Strong_tIoU=rows[alias]['values']['E_Strong']['t']))
  fits={}
  for cid,cfg in bar['configs'].items():
   ent=bar['files'][cid+'/'+alias];f=load(ROOT/ent['path'])['fit'];fits[cid]=f;first=f['rounds'][0];grad=array(first['gradient']);firstw=array(first['rollout']['weights']);firstmetrics=dict(first_ESS=float(np.mean(1/np.square(firstw).sum(-1))),first_gradient_norm=float(np.linalg.norm(grad)),first_state_step_norm=float(np.linalg.norm(array(first['state_step']))))
   for k in p['coarse_steps']:
    b=array(f['path'][k]['boxes']);ref_cost=float(f['path'][k]['anchor_loss']);delta=(b[:,:2]-E[:,:2])/E[:,2:];unplanned=[i for i in range(len(b))if i not in positions];w=array(f['rounds'][k-1]['rollout']['weights']);a=lookup[cfg['lr'],cfg['tau'],k,1.];v=rows[alias]['values'];value=dict(alias=alias,**firstmetrics,step_ESS=float(np.mean(1/np.square(w).sum(-1))),anchor_reference_cost=ref_cost,anchor_reference_cost_change=ref_cost-f['path'][0]['anchor_loss'],normalized_all_node_center_movement=float(np.linalg.norm(delta,axis=-1).mean()),normalized_observed_center_movement=float(np.linalg.norm(delta[positions],axis=-1).mean()),normalized_unobserved_center_movement=float(np.linalg.norm(delta[unplanned],axis=-1).mean()),mean_abs_log_size_change=float(abs(np.log(b[:,2:]/E[:,2:])).mean()),full_vIoU_minus_E_pp=100*(v[a]['v']-v['E_Strong']['v']),unobserved_sIoU_minus_E_pp=100*(v[a]['unobserved_actual_sIoU']-v['E_Strong']['unobserved_actual_sIoU']))
    process.setdefault(cid+'_S'+str(k),[]).append(value)
  for lr in [.003,.01,.03]:
   lo=next(k for k,c in bar['configs'].items()if c['lr']==lr and c['tau']==.2);hi=next(k for k,c in bar['configs'].items()if c['lr']==lr and c['tau']==.5);g=array(fits[lo]['rounds'][0]['gradient']).ravel();h=array(fits[hi]['rounds'][0]['gradient']).ravel();taucos.append(dict(alias=alias,lr=lr,first_gradient_cosine=float(g@h/max(np.linalg.norm(g)*np.linalg.norm(h),1e-30))))
  if len(quality)%32==0:print('PARAMETER_MECHANISM',ds,len(quality),flush=True)
 summaries={}
 for key,values in process.items():
  keys=[k for k in values[0]if k!='alias'];m={k:float(np.mean([v[k]for v in values]))for k in keys};m['reference_fit_improved_but_full_worse']=sum(v['anchor_reference_cost_change']<0 and v['full_vIoU_minus_E_pp']<0 for v in values);m['unobserved_movement_above_observed']=sum(v['normalized_unobserved_center_movement']>v['normalized_observed_center_movement']for v in values);summaries[key]=m
 validq=[v for v in quality if v['teacher_minus_ERef_anchor_pp']is not None];groups={'teacher_better_than_ERef_at_available_anchors':[v['alias']for v in validq if v['teacher_minus_ERef_anchor_pp']>0],'teacher_no_better_at_available_anchors':[v['alias']for v in validq if v['teacher_minus_ERef_anchor_pp']<=0],'no_available_anchor_GT':[v['alias']for v in quality if v['teacher_minus_ERef_anchor_pp']is None],'zero_Strong_time_intersection':[v['alias']for v in quality if v['Strong_tIoU']==0]};strata={}
 for name,aliases in groups.items():
  st={}
  if aliases:
   for arm in defs:st[arm]=dict(full_minus_E_pp=float(np.mean([100*(rows[a]['values'][arm]['v']-rows[a]['values']['E_Strong']['v'])for a in aliases])),unobserved_minus_E_pp=float(np.mean([100*(rows[a]['values'][arm]['unobserved_actual_sIoU']-rows[a]['values']['E_Strong']['unobserved_actual_sIoU'])for a in aliases])))
  strata[name]=dict(parents=len(aliases),means=st,offline_explanation_only_not_selection=True)
 result=dict(status='completed_offline_parameter_effect_analysis',dataset=ds,parents=128,matched_effects=effects,process_summaries=summaries,process_per_parent=process,teacher_quality_per_parent=quality,teacher_quality_strata=strata,first_step_tau_gradient_cosines=taucos,interpretation_limits=['Matched comparisons isolate a configuration change within the fixed method; they do not uniquely identify every downstream mechanism.','Adam normalizes first-step coordinate magnitudes: softer teacher weights need not proportionally reduce the first state step.','Anchor loss and update magnitude are process measures, not localization accuracy.','Teacher GT quality groups are offline explanations only and never drive per-video parameter choice.','All intervals are exploratory development results with multiple comparisons, not independent confirmation.'],seconds=time.perf_counter()-beg)
 put(P/ds/'PARAMETER_EFFECTS_PUBLIC.json',result);print('PARAMETER_EFFECTS_COMPLETE',ds,len(effects),flush=True)
if __name__=='__main__':main(sys.argv[1])

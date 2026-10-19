"""Post-global-seal evaluation of conditional feedback and student uptake."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_referdino_condition_hc128_v1 import D,P,verify

def avg(v):
 v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None

def main():
 import torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import metrics
 from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_hc_anchor_content16_v1 import local_cost
 torch.set_num_threads(4);p=verify(True);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_outputs_before_new_GT'] and not torch.cuda.is_initialized();start=time.perf_counter()
 for stage in ['CAPTURE','TEACHER0','STUDENT1','TEACHER1','STUDENT2']:
  for rec in read(D/(stage+'_BARRIER.json'))['files']:assert sha(ROOT/rec['path'])==rec['sha256']
 qs=[r['query_ordinal']for r in p['records']];gt,spans,_=truths('hc2',dict(parents=qs));plans=read(ROOT/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];frames=[];ar=['E_MBR','S1','Old_feedback','Refreshed_feedback'];lineage=[];costs=[]
 for r in p['records']:
  a=r['alias'];q=r['query_ordinal'];row=plans[q];z=load(D/'student2'/(a+'.pt'));inp=load(D/'inputs'/(a+'.pt'));s=load(D/'student1'/(a+'.pt'));old=load(D/'teacher0'/(a+'.pt'));new=load(D/'teacher1'/(a+'.pt'));m=metrics(z,row,gt[q],spans[q]);native=old['native'];t0=old['conditional'];t1=new['conditional'];ns=[]
  for j,pos in enumerate(r['positions8']):
   fid=int(z['frame_ids'][pos]);known=fid in gt[q];bs=dict(E=z['outputs']['E_MBR'][pos],S1=z['outputs']['S1'][pos],Native=native['boxes'][j].numpy(),T0=t0['boxes'][j].numpy(),T1=t1['boxes'][j].numpy());vals={k:iou_gt(b,gt[q][fid],row,'hc2')if known else None for k,b in bs.items()}
   # Native candidates are actual final tracked boxes, not GT-selected online.
   def pair_iou(b,c):
    lo=np.maximum(b[:2]-b[2:]/2,c[:,:2]-c[:,2:]/2);hi=np.minimum(b[:2]+b[2:]/2,c[:,:2]+c[:,2:]/2);inter=np.maximum(hi-lo,0).prod(1);return inter/(b[2:].prod()+c[:,2:].prod(1)-inter)
   nearest={k:float(pair_iou(bs[k],native['trace']['all_boxes'][j].numpy()).max())for k in ['T0','T1']}
   ff=dict(alias=a,position=pos,original4=pos in r['positions4'],GT_known=known,values=vals,T0_intervened_selected=t0['trace']['selected_is_intervened'][j],T1_intervened_selected=t1['trace']['selected_is_intervened'][j],T0_survives_final=t0['trace']['survives_per_layer'][-1][j],T1_survives_final=t1['trace']['survives_per_layer'][-1][j],nearest_native_candidate_IoU=nearest);frames.append(ff);ns.append(ff)
  state=s['fit']['final_state'];center=np.linalg.norm((z['outputs']['S1'][:,:2]-z['outputs']['E_MBR'][:,:2])/np.maximum(z['outputs']['E_MBR'][:,2:],1e-12),axis=1);motion=dict(S1_state_L2=float(state.norm()),S1_changed_nodes=int(np.any(z['outputs']['S1']!=z['outputs']['E_MBR'],axis=1).sum()),S1_center_shift_Ewh_mean=float(center.mean()),S1_center_shift_Ewh_max=float(center.max()),T1_minus_T0_box_max_abs=float((t1['boxes']-t0['boxes']).abs().max()),T1_minus_T0_frame_mean_L1=float((t1['boxes']-t0['boxes']).abs().sum(-1).mean()))
  fitdiag={}
  for name,fit in [('S1',s['fit']),('Old_feedback',z['fixed']),('Refreshed_feedback',z['refresh'])]:fitdiag[name]=dict(updates=fit['actual_updates'],loss_start=fit['path'][0]['loss'],loss_end=fit['path'][-1]['loss'],state_step_L2=float((fit['final_state']-fit['path'][0]['state']).norm()))
  rows.append(dict(alias=a,dataset='hc2',**m,motion=motion,fit_diagnostics=fitdiag,teacher_means={k:avg([f['values'][k]for f in ns])for k in bs}))
  for label,t in [('T0',t0),('T1',t1)]:lineage.append(dict(alias=a,teacher=label,selected_query=t['selected_query'],intervened_slots=t['trace']['injected_slots'].tolist(),pruning_survival=t['trace']['survives_per_layer'],selected_is_intervened=t['trace']['selected_is_intervened'],intervened_track_indices=t['trace']['intervened_track_indices'],initial_reference_projected_elements=t['trace']['reference_projection_elements'],other_references_bitexact=t['trace']['other_references_bitexact']))
  costs.append(dict(alias=a,Source_seconds=inp['source_seconds'],native=native['cost'],T0=t0['cost'],T1=t1['cost'],student1=s['compute'],student2=z['compute']))
 ep=['v','t','s','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'];pairs=[('S1','E_MBR'),('Old_feedback','E_MBR'),('Refreshed_feedback','E_MBR'),('Refreshed_feedback','Old_feedback'),('Refreshed_feedback','S1'),('Old_feedback','S1')];group=dict(parents=128,means={a:{e:avg([x['values'][a][e]for x in rows])for e in ep}for a in ar},paired={a+'_minus_'+b:{e:paired([None if x['values'][a][e]is None or x['values'][b][e]is None else 100*(x['values'][a][e]-x['values'][b][e])for x in rows])for e in ep}for a,b in pairs})
 tp=[('Native','E'),('T0','E'),('T0','Native'),('T1','T0'),('T1','S1')];teacher_pairs={a+'_minus_'+b:paired([None if x['teacher_means'][a]is None else 100*(x['teacher_means'][a]-x['teacher_means'][b])for x in rows])for a,b in tp}
 # Continuous changes plus predeclared descriptive IoU=.5 recovery/damage.
 behavior={}
 for teacher,student in [('Native','E'),('T0','E'),('T1','S1')]:
  known=[x for x in frames if x['GT_known']];bad=[x for x in known if x['values'][student]<.5];good=[x for x in known if x['values'][student]>=.5];behavior[teacher]=dict(known=len(known),student_bad=len(bad),bad_recovered_ge05=sum(x['values'][teacher]>=.5 for x in bad),student_good=len(good),good_damaged_lt05=sum(x['values'][teacher]<.5 for x in good),bad_mean_delta_pp=avg([100*(x['values'][teacher]-x['values'][student])for x in bad]),good_mean_delta_pp=avg([100*(x['values'][teacher]-x['values'][student])for x in good]))
 put(P/'RESULTS_PUBLIC.json',dict(status='completed',count=128,groups=dict(HC128=group),rows=rows,all_outputs_before_new_GT=True,configuration_frozen=True,full_OPD_run=False,exposed_development=True,confirmation32_not_run=True))
 put(P/'TEACHER_FRAME_DIAGNOSTICS_PUBLIC.json',dict(rows=frames,teacher_parent_paired_pp=teacher_pairs,descriptive_recovery_damage=behavior,unknown_not_zero_filled=True,identity_labels_unavailable=True,alternate_selected_lineage_not_counted_as_intervened_correction=True))
 put(P/'QUERY_LINEAGE_PUBLIC.json',dict(rows=lineage,slots_are_internal_model_indices_not_identity_labels=True))
 put(P/'COST_BREAKDOWN.json',dict(GPU_allocations=[read(f)for f in sorted((D/'cost').glob('*.json'))],rows=costs,total_GPU_seconds=sum(read(f)['GPU_seconds']for f in(D/'cost').glob('*.json')),Source_forwards=128,Refer_queries=384+3,Refer_unique_physical_observations=128*8,engineering_repeat_queries=3,new_DINO_T_calls=0,student_state_updates=128*120,new_network_training=0,pretrained_weights_frozen=True,scoring_CPU_seconds=time.perf_counter()-start,model_loading_RGB_decode_old_cache_cost_separate=True))
 print('STUDENT', {k:{e:v[e]['mean']for e in ['v','unobserved_actual_s']}for k,v in group['paired'].items()},flush=True);print('TEACHER',teacher_pairs,flush=True)
if __name__=='__main__':main()

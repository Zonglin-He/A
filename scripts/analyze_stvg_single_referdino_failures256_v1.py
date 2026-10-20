"""Retained harmful-parent mechanisms tied to matched parameters, cached evidence only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_single_referdino_parameter_search256_common_v1 import D,P,BASE,read,put,load

def main():
 result={};checks=0;maxerr=0.;beg=time.perf_counter()
 for ds in ['hc2','vidstg']:
  r=read(P/ds/'coarse/RESULTS_PUBLIC.json');assert read(P/ds/'coarse/ROOT_AUDIT_PUBLIC.json')['status']=='pass';m=read(P/ds/'PARAMETER_EFFECTS_PUBLIC.json');qual={x['alias']:x for x in m['teacher_quality_per_parent']};process={a:{x['alias']:x for x in rows}for a,rows in m['process_per_parent'].items()};base='C02_S'+('40'if ds=='hc2'else'30')+'_A100';k=r['definitions'][base]['steps'];best=r['development_ranking'][0]
  variants={'baseline':base,'same_path_earlier20':'C02_S20_A100','same_path_later40':'C02_S40_A100','same_path_output_half':base.replace('A100','A050'),'same_lr_steps_softer_tau':base.replace('C02','C03'),'same_steps_tau_smaller_lr':base.replace('C02','C00'),'same_steps_tau_larger_lr':base.replace('C02','C04'),'whole_panel_coarse_winner':best};strength={x['alias']:x for x in read(P/'INITIALIZATION_STRENGTH_RESULTS_PUBLIC.json')['datasets'][ds]['rows']};rows=[]
  for row in r['rows']:
   a=row['alias'];old=load(BASE/f'predictions/{ds}/{a}.pt');q=qual[a];out=dict(alias=a,teacher_at_GT_available_anchors=q,variants={},initialization_only_strength_curve={g:100*v['v']for g,v in strength[a]['values'].items()},no_per_video_parameter_selection=True)
   with np.load(D/ds/'coarse/private_metric_receipts'/f'{a}.npz')as z:
    arms=z['arms'].tolist();e=arms.index('E_Strong');fids=z['dense_GT_ids'];u,v=z['GT_span'];actual=z['actual_ids'];obs=np.isin(actual,z['observed_ids']);st,en=z['interval'];support=(actual>=st)&(actual<en);unobs=~obs
    def vi(i,iv):
     st,en=iv;mask=(fids>=max(st,u))&(fids<min(en,v));return float(z['dense_GT_IoU'][i,mask].sum()/max(max(en,v)-min(st,u),1))
    for name,arm in variants.items():
     i=arms.index(arm);val=row['values'][arm];err=abs(vi(i,z['interval'])-val['v']);maxerr=max(maxerr,err);assert err<1e-10;checks+=1;delta=z['actual_IoU'][i]-z['actual_IoU'][e];parts={}
     for group,mask in [('observed',obs),('unobserved_in_Strong',unobs&support),('unobserved_outside_Strong',unobs&~support)]:parts[group]=dict(nodes=int(mask.sum()),mean_sIoU_change_pp=100*float(delta[mask].mean())if mask.any()else None)
     cfg=r['definitions'][arm];pkey=cfg['config_id']+'_S'+str(cfg['steps']);proc=process[pkey][a]
     out['variants'][name]=dict(arm=arm,config=cfg,full_vIoU_percent=100*val['v'],full_minus_E_pp=100*(val['v']-row['values']['E_Strong']['v']),full_minus_baseline_pp=100*(val['v']-row['values'][base]['v']),unobserved_sIoU_delta_pp=100*(val['unobserved_actual_sIoU']-row['values']['E_Strong']['unobserved_actual_sIoU']),same_boxes_Native_time_minus_E_pp=100*(vi(i,old['intervals']['Native'])-vi(e,old['intervals']['Native'])),same_boxes_Strong_time_minus_E_pp=100*(vi(i,z['interval'])-vi(e,z['interval'])),spatial_support_breakdown=parts,actual_fit_process=proc,process_is_raw_unscaled_rollout_alpha_applied_only_to_output=True)
   bv=out['variants']['baseline'];out['baseline_harmful']=bv['full_minus_E_pp']< -1e-10;out['supported_mechanism_flags']=dict(teacher_mean_no_better_than_E=q['teacher_minus_ERef_anchor_pp']is not None and q['teacher_minus_ERef_anchor_pp']<=0,teacher_mean_IoU_below_half=q['teacher_actual_anchor_IoU']is not None and q['teacher_actual_anchor_IoU']<.5,reference_fit_improved_while_full_worse=bv['actual_fit_process']['anchor_reference_cost_change']<0 and bv['full_minus_E_pp']<0,unobserved_motion_exceeds_observed=bv['actual_fit_process']['normalized_unobserved_center_movement']>bv['actual_fit_process']['normalized_observed_center_movement'],no_Strong_GT_overlap=q['Strong_tIoU']==0,halving_output_reduces_harm=out['baseline_harmful']and out['variants']['same_path_output_half']['full_minus_baseline_pp']>0,smaller_lr_reduces_harm=out['baseline_harmful']and out['variants']['same_steps_tau_smaller_lr']['full_minus_baseline_pp']>0,earlier_stop_reduces_harm=out['baseline_harmful']and out['variants']['same_path_earlier20']['full_minus_baseline_pp']>0,softer_tau_reduces_harm=out['baseline_harmful']and out['variants']['same_lr_steps_softer_tau']['full_minus_baseline_pp']>0);rows.append(out)
  harmed=[x for x in rows if x['baseline_harmful']];group_results={}
  for name in variants:
   def aggregate(rr):
    d=np.array([x['variants'][name]['full_minus_baseline_pp']for x in rr]);return dict(parents=len(rr),mean_change_pp=float(d.mean())if len(d)else None,improved=int((d>1e-10).sum()),worse=int((d< -1e-10).sum()))
   group_results[name]=dict(all_parents=aggregate(rows),baseline_harmed_offline_group=aggregate(harmed),other_parents_offline_group=aggregate([x for x in rows if not x['baseline_harmful']]))
  worst=sorted(rows,key=lambda x:x['variants']['baseline']['full_minus_E_pp'])[:12];result[ds]=dict(parents=128,baseline_arm=base,coarse_winner=best,harmed_parents=len(harmed),flags_within_harmed={key:sum(x['supported_mechanism_flags'][key]for x in harmed)for key in rows[0]['supported_mechanism_flags']},matched_parameter_changes=group_results,worst_baseline_aliases=[x['alias']for x in worst],all128_signed_parent_explanations=rows)
  print('HARM_LINKS',ds,'harmed',len(harmed),'flags',result[ds]['flags_within_harmed'],flush=True)
  for x in worst[:5]:print('HARM_CASE',ds,x['alias'],{n:round(v['full_minus_E_pp'],4)for n,v in x['variants'].items()},'teacher-minus-E',x['teacher_at_GT_available_anchors']['teacher_minus_ERef_anchor_pp'],flush=True)
 put(P/'FAILURE_PARAMETER_LINKS_PUBLIC.json',dict(status='completed_retained_harmful_parent_and_matched_parameter_analysis',datasets=result,cached_offline_evidence_only=True,new_GT_reads=False,new_model_or_expert_forward=False,all128_per_domain_retained=True,per_video_configuration_selection=False,identity_error_not_inferred_from_low_IoU_alone=True,full_metric_reconstruction_checks=checks,max_abs=maxerr,seconds=time.perf_counter()-beg))
 lines=['# Harmful videos and parameter links','', 'All analyses use already sealed coarse predictions, saved metric receipts and saved update processes. All128 parents per dataset remain in selection. Harmful-parent groups below are retrospective explanations, never online gates. Teacher quality means actual available-anchor IoU; low IoU alone does not identify an identity error. Time counterfactuals use the exact same saved boxes with Native versus Strong; neither time branch is changed.','']
 for ds,g in result.items():
  lines += [f'## {ds}: baseline {g["baseline_arm"]}', '',f'{g["harmed_parents"]}/128 parents have negative full-vIoU adaptation relative to the same E+Strong. Flags: `{g["flags_within_harmed"]}`.','', '| Alias | baseline−E pp | earlier20 | half-output | smaller-lr | softer-tau | larger-lr | coarse-winner | teacher−E anchor pp |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
  for a in g['worst_baseline_aliases']:
   x=next(v for v in g['all128_signed_parent_explanations']if v['alias']==a);vv=x['variants'];t=x['teacher_at_GT_available_anchors']['teacher_minus_ERef_anchor_pp'];names=['baseline','same_path_earlier20','same_path_output_half','same_steps_tau_smaller_lr','same_lr_steps_softer_tau','same_steps_tau_larger_lr','whole_panel_coarse_winner'];lines.append('| '+a+' | '+' | '.join(f'{vv[n]["full_minus_E_pp"]:+.4f}'for n in names)+' | '+('NA'if t is None else f'{t:+.4f}')+' |')
  lines += ['', '| Matched change vs original baseline | all128 mean pp | baseline harmed mean pp | remaining parents mean pp |','|---|---:|---:|---:|']
  for name,z in g['matched_parameter_changes'].items():
   if name=='baseline':continue
   lines.append(f'| {name} | {z["all_parents"]["mean_change_pp"]:+.6f} | {z["baseline_harmed_offline_group"]["mean_change_pp"]:+.6f} | {z["other_parents_offline_group"]["mean_change_pp"]:+.6f} |')
 lines += ['', 'The whole-panel tradeoff determines the dataset-wide parameter, not the individually best recovery shown in a row. Lower reference loss is not task correctness. If less update rescues harmful videos but removes larger gains elsewhere, shrinking globally need not be better. Initialization-only gamma curves are labeled separately from actual adapted results; nonunit gamma must receive actual OPD before entering the final adaptive comparison.']
 (P/'FAILURE_PARAMETER_LINKS.md').write_text('\n'.join(lines)+'\n')
 print('HARM_LINKS_COMPLETE',checks,maxerr,flush=True)
if __name__=='__main__':main()

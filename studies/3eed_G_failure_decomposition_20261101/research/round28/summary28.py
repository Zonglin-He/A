"""All findings remain offline descriptive; do not choose a deployed method on these labels."""
import sys,numpy as np,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round28'))
from common28 import *
def main():
 rr=read(EXP/'FUSION_ROWS_PRIVATE.json');ident=read(EXP/'IDENTITY24_ROWS.json');support=read(EXP/'POINT_SUPPORT24_ROWS.json');phasecounts={};effects={}
 for phase,rs in [('ALL528',rr),('OLD400',rr[:400]),('REPLAY256',rr[208:464]),('Q64',rr[464:])]:
  phasecounts[phase]={p:{'n':len([x for x in rs if x['platform']==p]),'pool':sum(x['has_pool'] for x in rs if x['platform']==p),'unmapped_original_with_pool':sum(x['has_pool'] and not x['original_mapped'] for x in rs if x['platform']==p),'mapped_only_changes':sum(x['q']['mapped_competition_only']!=x['q']['joint_original'] for x in rs if x['platform']==p)} for p in ['drone','quad']}
  for arm in ['mapped_competition_only','mean_zero','median_zero']:
   effects[phase+'_'+arm]={'balanced_gain_pp':float(np.mean([np.mean([x['ious'][arm]-x['ious']['joint_original'] for x in rs if x['platform']==p])*100 for p in ['drone','quad']])),'gains':sum(x['ious'][arm]>x['ious']['joint_original']+1e-10 for x in rs),'harms':sum(x['ious'][arm]<x['ious']['joint_original']-1e-10 for x in rs),'query_changes':sum(x['q'][arm]!=x['q']['joint_original'] for x in rs)}
 offsets={}
 for c in [-2,-1,-.5,-.25,0,.25,.5,1,2]:
  name='offset_'+str(c);offsets[str(c)]={'changes':sum(x['q'][name]!=x['q']['joint_original'] for x in rr),'mean_gain_pp':float(np.mean([x['ious'][name]-x['ious']['joint_original'] for x in rr])*100)}
 witnesses=[]
 for x in support:
  if x.get('chain_support_diagnosis')=='visual_and_mapped_cue_support_target_fusion_selects_other':
   f=x['details']['original_final'];g=x['details']['cue_max_source_tiebreak'];ds=f['source_score']-g['source_score'];de=2*(f['mapped_cue']-g['mapped_cue']);witnesses.append({'case':x['case'],'platform':x['platform'],'visual_margin':x['visual_margin'],'source_advantage_for_wrong_support':ds,'weighted_cue_advantage_for_wrong_support':de,'F_wrong_minus_correct_support':ds+de,'lambda_needed_to_reverse_pair':ds/(g['mapped_cue']-f['mapped_cue']) if g['mapped_cue']>f['mapped_cue'] else None,'competitor_chosen_by_cue_then_source_without_GT':True})
 margins=[x['visual_target_margin'] for x in ident if x['visual_target_margin'] is not None]
 duplicated=[x['max_cue_query_count'] for x in support if x.get('image_ranking_status')=='correct']
 # Exact geometric corroboration of manually resolved target mask(s), not an identity rewrite.
 corroboration=[]
 for x in ident:
  targets=x['visual_annotation']['target_labels']
  if not targets:continue
  target_support=[x['mask_point_support'][t] for t in targets];corroboration.append({'case':x['case'],'platform':x['platform'],'visual_confidence':x['visual_annotation']['confidence'],'target_mask_3D_annotation_supported':any(t['physical_support']=='target_supported' for t in target_support),'target_mask_point_fraction':max(t['target_point_fraction'] or 0 for t in target_support)})
 # Classifying center alone incorrectly calls some target-supported boxes unresolved.
 disagreements=[x['case'] for x in support if x.get('details',{}).get('original_final',{}).get('support_label')=='target_supported' and x.get('center_proxy_final')!='target']
 summary={'phasecounts':phasecounts,'readout_effects':effects,'offset_stress_fixed_not_selected':offsets,'source_prior_witnesses':witnesses,'visual_strict_ranking_counts':dict(collections.Counter(x['image_ranking_status'] for x in ident)),'visual_margins_abs_le_1e_3_descriptive':sum(abs(v)<=1e-3 for v in margins),'threshold_not_a_runtime_gate':True,'maxcue_plateau_queries_when_visual_correct':{'n':len(duplicated),'median':float(np.median(duplicated)),'min':min(duplicated),'max':max(duplicated)},'manual_target_mask_3D_geometry_corroboration':corroboration,'center_proxy_unresolved_but_visual_point_supported_cases':disagreements,'no_new_confirmation_or_additive_error_partition':True}
 dump(EXP/'DIAGNOSTIC_SUMMARY.json',summary);print('SUMMARY28',phasecounts,effects,witnesses)
if __name__=='__main__':main()

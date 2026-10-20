"""Root recomputation from saved scores, cost accounting, scope decisions."""
import sys,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np
from scripts.run_stvg_spatial_strong_bidirectional256_v1 import D,P,read,put,verify,sha

def main():
 p=verify();res=read(P/'RESULTS_PUBLIC.json');profiles=read(P/'COMPLETE_SINGLE_SEED_PROFILE_PUBLIC.json');maxerr=0.;root={};decisions={};cost={};newupdates=reusedupdates=0
 for ds,z in res['directions'].items():
  rr=z['rows'];assert len(rr)==128 and len({r['alias']for r in rr})==128;assert len([r for r in p['records']if r['dataset']==ds])==128;checks=0
  for a,mm in z['means'].items():
   for met,v in mm.items():
    actual=np.mean([r['values'][a][met]for r in rr if r['values'][a][met]is not None]);maxerr=max(maxerr,float(abs(actual-v)));assert abs(actual-v)<1e-13;checks+=1
  for r in rr:
   for metric in ['s','fixed_actual_sIoU','unobserved_actual_sIoU']:
    assert r['values']['Source'][metric]==r['values']['Temporal_only'][metric]
    assert r['values']['Spatial_only'][metric]==r['values']['Full'][metric]
   assert r['values']['Full']['t']==r['values']['Without_spatial_adaptation']['t']==r['values']['Temporal_only']['t']
   newupdates+=r['new_updates'];reusedupdates+=r['reused_updates']
  root[ds]=dict(recomputed_macro_scalars=checks,five_rows_complete=True,parents=128,seed_count=len(p['seeds']),no_spatial_change_from_temporal_switch=True,same_Strong_time_for_adaptation_contrast=True,actual_sIoU_defined_parents={a:sum(r['values'][a]['fixed_actual_sIoU']is not None for r in rr)for a in z['means']},unobserved_actual_sIoU_defined_parents=sum(r['values']['Full']['unobserved_actual_sIoU']is not None for r in rr))
  vv=np.array([100*(r['values']['Full']['v']-r['values']['Without_spatial_adaptation']['v'])for r in rr]);loo=(vv.sum()-vv)/127
  adaptation=z['paired']['Full_minus_Without_spatial_adaptation']['v'];total=z['paired']['Full_minus_Source']['v']
  decisions[ds]=dict(total_gain_pp=total,spatial_adaptation_gain_over_E_Strong_pp=adaptation,adaptation_leave_one_parent_out_range_pp=[float(loo.min()),float(loo.max())],decision='panel_supports_spatial_adaptation_contribution; needs independent confirmation'if adaptation['CI95'][0]>0 else 'spatial_adaptation_contribution_not_established_on_this_panel; total gain must not be attributed entirely to_OPD',independent_confirmation=False,automatic_method_promotion=False)
  measured=[r['cost']for r in rr if 'Source224_seconds'in r['cost']];fs=[float(c['single_seed_component_sum_seconds'][str(s)])for c in measured for s in p['seeds']];peaks=[c['peak_GPU_bytes']/2**30 for c in measured];cost[ds]=dict(measured_matrix_queries=len(measured),reused_queries=128-len(measured),actual_complete_single_seed_profile=profiles[ds],component_sum_seconds_mean=float(np.mean(fs)),component_sum_seconds_median=float(np.median(fs)),component_sum_seconds_range=[float(min(fs)),float(max(fs))],component_sum_is_not_direct_wall_measurement=True,measured_case_peak_allocated_GiB_range=[float(min(peaks)),float(max(peaks))],actual_new_DINO_image_forwards=sum(c['DINO_image_forwards']for c in measured),actual_new_ReferDINO_eight_frame_queries=len(measured),new_Source_offset_forwards=4*len(measured),new_spatial_updates=sum(r['new_updates']for r in rr),reused_spatial_updates=sum(r['reused_updates']for r in rr),parsing_cost='main matrix uses frozen subject cache; actual complete single-seed profiles re-run parser from original text and verify equality')
 cost['additional_profile_cost']=dict(queries=2,Source_offset_forwards=8,ReferDINO_eight_frame_queries=2,DINO_image_forwards=sum(x['capture']['DINO_image_forwards']for x in profiles.values()),spatial_updates=80)
 audit=dict(status='pass',directions=root,max_macro_scalar_error=maxerr,all256_parent_rows_present=True,new_matrix_updates=newupdates,reused_matrix_updates=reusedupdates,single_seed_profile_updates=80,registries_unchanged=p['protected'],postseal_official_metric_checks=res['official_metric_checks'],independent_rollout_audit=read(P/'INDEPENDENT_SPATIAL_AUDIT_PUBLIC.json'),pure_saved_score_review=True,new_GT_read=False)
 audit['primary_seed41_updates']=10240;audit['primary_new_updates']=129*40;audit['primary_reused_updates']=127*40;audit['unused_new_seed42_43_updates']=newupdates-129*40;audit['unused_historical_reused_seed42_43_updates']=reusedupdates-127*40
 put(P/'ROOT_AUDIT_PUBLIC.json',audit);put(P/'DECISION_PUBLIC.json',decisions);put(P/'COST_PUBLIC.json',cost)
 lines=['','## Root saved-score verification and decision scope','',f'All256 parents retained. {res["official_metric_checks"]} official/dense metric agreement checks. Independent NumPy audits256 selected seed41 fits/10240 updates; earlier unselected42/43 are historical compute. Actual matrix new updates {newupdates}; exact reused updates {reusedupdates}. Additional single-seed cost profiles80 updates are separately counted. No candidate selection by new GT and no registry promotion.']
 for ds,v in decisions.items():
  x=v['spatial_adaptation_gain_over_E_Strong_pp'];lines +=['',f"{ds}: Full−E+Strong {x['mean']:+.6f} pp, CI {x['CI95']}; {v['decision']}."]
 lines+=['','## Measured actual single-inference cost','','| Target | complete cold single seed wall seconds | Source cold load seconds | GroundingDINO forwards | ReferDINO frames | peak allocated GPU GiB |','|---|---:|---:|---:|---:|---:|']
 for ds,v in profiles.items():lines.append(f"| {ds}, {v['alias']}, seed41 | {v['actual_complete_single_seed_wall_seconds']:.6f} | {v['Source_load_seconds']:.6f} | {v['capture']['DINO_image_forwards']} | 8 | {v['peak_GPU_bytes']/2**30:.6f} |")
 lines+=['','Actual full single-seed wall profiles include original video decoding, text parsing, Source loading,224/448 Source inference, both experts including model loads/subprocess isolation, E/MBR and40 spatial updates. They are the first prelocked case per direction, not128-case deployment averages. Peak is the maximum PyTorch allocated bytes of isolated model processes, excluding unrelated services/context-reserved memory. For reverse128, all per-case components and three-seed wall costs are retained; component sums are explicitly distinguished from independently measured single-seed wall time. HC127 reused cases retain prior full compute provenance, not zero-cost deployment claims.','', 'The fixed actual-node sIoU averages only parents with GT-supported physical input nodes; counts are recorded in ROOT_AUDIT_PUBLIC.json. All256 parents remain in full vIoU/tIoU/dense sIoU and severe-tail analysis. Both directions are frozen development/exposure-disclosed panels; HC128 directly participated in spatial/Strong development, and reverse legacy exposure prevents claiming a pristine test set.']
 with (P/'REPORT.md').open('a')as f:f.write('\n'.join(lines)+'\n')
 print(decisions,flush=True)
if __name__=='__main__':main()

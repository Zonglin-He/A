"""ROOT review/export of complete source calibration, no new GT/model calls."""
from pathlib import Path
import sys,json,csv,time,numpy as np
R=Path(__file__).resolve().parents[3];S=R/'studies/expert_guided_ttft_v1'
sys.path[:0]=[str(S),str(R),str(R/'tools')]
import calibration_runtime_v1 as c
sys.path.insert(0,str(R/'studies/reviewer_key_controls_v1'))
from statistics_v1 import stats,pair_stats
def main():
 c.check();d=c.C;p=c.P;assert c.read(d/'controller_actual_exit.json')['actual_tool_wait'];assert c.read(d/'controller_actual_exit.json')['exit_code']==0
 ack=c.read(d/'audit_receipt.json');assert ack['report_sha256']==c.sha(p/'report.json');rows=c.read(d/'private_rows.json');rep=c.read(p/'report.json');assert c.report(rows)==rep
 diagnostics={};cost=[];traces=[]
 for target in c.TARGETS:
  rr=[r for r in rows if r['target']==target];idx=rep[target]['selected_index'];arm=f'Config{idx}';selection=c.config_list(target)[idx];assert selection==rep[target]['selected_config']
  assert idx==max(range(8),key=lambda j:(rep[target]['means'][f'Config{j}']['v'],-j))
  groups={g:{f'{arm}-SourceTR':pair_stats([r for r in rr if g=='all'or r['duration_group']==g],arm,'SourceTR'),f'{arm}-Source':pair_stats([r for r in rr if g=='all'or r['duration_group']==g],arm,'Source')}for g in ['all','short_le5s','long_gt5s']}
  trace_rows=[]
  for i,r in enumerate(rr,1):
   e=c.checked_record(target,r['id']);tt=c.read(d/e['trace_path']);cost.append(dict(target=target,query=f'Q{i:03}',Source_seconds=e['Source_seconds'],inherited_preflight_Source_nonzero=e['reused_preflight_Source'],query_wall_seconds=e['query_wall_seconds'],worker=e['worker']))
   for a,t in tt.items():
    assert t['reset_all_parameters_and_Source_boxes_exact']and t['checkpoint_cold_forward_exact'];assert t['parameter_delta_L2']>0 and t['spatial_changed_frames']>0
    assert all(np.isfinite(x['loss'])and x['gradient_norm_before_clip']>0 for x in t['trace'])
    row=dict(target=target,query=f'Q{i:03}',config=a,steps=t['optimizer_steps'],pseudo_loss_before=t['pseudo_loss_before'],pseudo_loss_after=t['pseudo_loss_after'],parameter_delta_L2=t['parameter_delta_L2'],active_gradient_tensors=t['trace'][-1]['gradient_parameter_tensors'],active_gradient_scalars=t['trace'][-1]['gradient_parameter_scalars'],spatial_changed_frames=t['spatial_changed_frames'],unobserved_changed_frames=t['unobserved_changed_frames'],all_sampled_frames=t['all_sampled_frames'],reset_exact=True,cold_exact=True)
    trace_rows.append(row);traces.append(row)
  diagnostics[target]=dict(selected_index=idx,selected_config=selection,groups=groups,functional_trace_rows=len(trace_rows),all576_trace_loss_parameter_reset_cold_finite=True,selected_pseudo_loss_increased_queries=sum(t['pseudo_loss_after']>t['pseudo_loss_before']for t in trace_rows if t['config']==arm))
 c.atomic(p/'ROOT_diagnostics.json',diagnostics);c.atomic(p/'all_query_acquisition_update_cost.json',dict(rows=cost,nested_not_additive=True,Source_preflight_reused_nonzero=True,cached_update_with_audit_IO_not_deployment=True,unique_model_loads=[c.read(d/t/'model_load.json')for t in c.TARGETS]))
 with(p/'all_source_signed_scalars.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['target','query','parent','group','arm','t_percent','v_percent','R03_percent','R05_percent','unobserved_IoU_percent'])
  for target in c.TARGETS:
   qq=[r for r in rows if r['target']==target];pm={x:f'P{i:03}'for i,x in enumerate(sorted({r['parent']for r in qq}),1)}
   for i,r in enumerate(qq,1):
    for a,z in r['values'].items():w.writerow([target,f'Q{i:03}',pm[r['parent']],r['duration_group'],a,*[None if z[k]is None else 100*z[k]for k in ['t','v','R03','R05','unobserved_actual_sIoU']]])
 with(p/'all_actual_update_diagnostics.tsv').open('w')as f:
  w=csv.DictWriter(f,list(traces[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(traces)
 lines=['# Source-only TTFT calibration review','', 'All72 source-validation queries/576 adapted states and all720 official/manual scalar checks passed. No target scores selected these configurations. The validation videos were historically exposed in the reverse transfer direction; HC source queries have34 scorer parents, not36. These are calibration outcomes and selected-setting CIs are descriptive, not independent validation.','']
 for t in c.TARGETS:
  z=rep[t];a=f"Config{z['selected_index']}";v=z['means'][a];q=z['paired'][a+'-SourceTR']['statistics'];lines += [f"## Target {t}: calibrate only on source {z['source_domain']}",'',f"Selected {a}: {json.dumps(z['selected_config'])}",f"SourceTR vIoU {100*z['means']['SourceTR']['v']:.6f} -> selected {100*v['v']:.6f}; delta {q['delta_v_pp']['mean']:.6f} pp, descriptive95%CI {q['delta_v_pp']['CI95']}. Temporal delta {q['delta_t_pp']['mean']:.6f}, CI {q['delta_t_pp']['CI95']}. All8 candidates, harms/severe/short-long and actual update traces remain in report/ROOT_diagnostics/signed files.",'']
 lines += ['The learned baseline is functional in both directions. Accept the source-selected configurations for the original128-query target pilot only. Target scores cannot change lr, loss weights or steps. No shared correction, transport or DP enters the TTFT baseline. Source/acquisition/update/checkpoint costs are retained; conditional cached wall is not deployment timing.']
 (p/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
 c.atomic(d/'root_acceptance.json',dict(status='ROOT_source_functional_numerical_and_source_only_selection_accepted_release_original128_target_pilot',source_functioning_accepted=True,no_target_scores_used=True,selected_configs={t:rep[t]['selected_config']for t in c.TARGETS},selected_indices={t:rep[t]['selected_index']for t in c.TARGETS},source_report_sha256=c.sha(p/'report.json'),source_audit_sha256=c.sha(d/'audit_receipt.json'),ROOT_diagnostics_sha256=c.sha(p/'ROOT_diagnostics.json'),source_actual_controller_exit_sha256=c.sha(d/'controller_actual_exit.json'),release_scope='Original fixed128/query/parent per domain pilot first. Full target plan requires separate bounded storage/execution plan after functionality; no target selection.',source_validation_not_independent_test=True,full_target_released=False,new_Source=0,new_expert=0,new_GT=0,worker=c.identity(),time=time.time()))
 print(json.dumps({t:diagnostics[t]['selected_config']for t in c.TARGETS}),flush=True)
if __name__=='__main__':main()

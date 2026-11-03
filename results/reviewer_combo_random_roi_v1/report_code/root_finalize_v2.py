"""Root scalar-only synthesis after all independently audited domains and actual exits."""
from pathlib import Path
import sys,json,hashlib,time,csv
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R/'studies/reviewer_combo_random_roi_v1'))
from study_common import *
def f(v):return f'{v:.4f}'
def diff(z):return f'{z["mean"]:+.4f} [{z["CI95"][0]:+.4f}, {z["CI95"][1]:+.4f}]'
validate();guard();outer=read(D/'controller_actual_exit.json');assert outer['actual_tool_wait']and outer['exit_code']==0
units={};checks=0;maxerr=0;rootchecks=0
for task in ['combination','regions']:
 for ds in COUNTS:
  d=D/task/ds;p=P/task/ds;a=read(d/'reporting_receipt.json');sc=read(d/'score_receipt.json');rr=read(d/'root_receipt.json');assert a['all_cold_arrays_exact']and rr['audit_sha256']==sha(d/'reporting_receipt.json')and rr['findings_sha256']==sha(p/'ROOT_FINDINGS.md')and sc['max_discrepancy']<2e-10
  checks+=sc['new_official_checks'];maxerr=max(maxerr,sc['max_discrepancy']);rootchecks+=rr['paired_group_CI_checks'];units[task+'_'+ds]=dict(report=read(p/'report.json'),report_sha256=sha(p/'report.json'),audit_sha256=sha(d/'reporting_receipt.json'),root_sha256=sha(d/'root_receipt.json'),prediction_wall_seconds=read(d/'sealed.json')['elapsed_wall_seconds'])
exits={p.stem:read(p)for p in (D/'exits').glob('*.json')};assert len(exits)==12 and all(e['actual_subprocess_wait']and e['exit_code']==0 for e in exits.values())
for e in list(exits.values())+[dict(worker=read(D/'controller.json'))]:
 who=e['worker'];q=Path('/proc')/str(who['pid'])/'stat';assert not q.exists()or int(q.read_text().split()[21])!=who['startticks']
assert checks==12461 and rootchecks==330
text='# Reviewer controls: full strict simple combination and random spatial ROI\n\nAll four whole-domain tasks sealed, separately officially scored (new arms only), independently cold-array/input/receipt/reused-metric/means audited and ROOT interpreted. Existing results were not rerun. Full HC1902queries/227parents, Vid10303/732parents; region128queries/128distinctparents per domain. No new Source/expert/model/GPU/optimizer, no method or paper change. All12actual stages and actual controller exited0.\n\n'
text+='## Full-target simple-combination comparison\n\n|Dataset|Configuration|Expert|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|\n|---|---|---|---:|---:|---:|---:|---:|\n'
for ds in COUNTS:
 x=units['combination_'+ds]['report']
 for a in MAIN:
  m=x['means'][a];text+='|'+ds+'|'+a+'|'+('no'if a in ['Source','SourceTemporal']else'yes')+'|'+'|'.join(f(100*m[k])if m[k]is not None else'missing'for k in ['t','v','vIoU_ge_03','vIoU_ge_05','unobserved_actual_sIoU'])+'|\n'
text+='\nSourceTemporal uses native Source boxes/SourceROI appearance with its own refined interval and no expert. SimpleCombination uses the exact same interval with strict raw8expert center/logsize interpolation and nearest-expert tails, no initialized-E tails/transport/DP. InitializedTemporal outputs original corrected E boxes and its own E-ROI refined interval, no spatial propagation/DP. Full uses that same E-ROI interval and the unchanged finalDP path. All expert configurations use the same eight physical observations; SourceTemporal does not consume them. Full-versus-Simple changes both pooling evidence and spatial trajectory; the following fixed-time comparison isolates the latter. Historical hybrid interpolation outputs remain separate.\n\n|Dataset|Paired comparison|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|\n|---|---|---|---|\n'
for ds in COUNTS:
 for key,v in units['combination_'+ds]['report']['paired'].items():text+=f'|{ds}|{key}|{diff(v["t"])}|{diff(v["v"])}|\n'
text+='\n## Spatial trajectories at the identical Full interval\n\n|Dataset|Trajectory|tIoU %|vIoU %|\n|---|---|---:|---:|\n'
for ds in COUNTS:
 text+='\n|Dataset|Trajectory|tIoU %|vIoU %|\n|---|---|---:|---:|\n'
 x=read(P/'combination'/ds/'fixed_interval/report.json')
 for a,m in x['means'].items():text+=f'|{ds}|{a}|{100*m["t"]:.4f}|{100*m["v"]:.4f}|\n'
 z=x['paired']['Full-PureExpertFixed']['v'];text+=f'\n{ds} Full minus strict pure interpolation at the same event: {diff(z)} pp. All per-query tIoUs are identical. Fixed-time Source is original Source boxes with the expert-derived Full interval, not an expert-free system.\n\n'
text+='## Region evidence on unchanged 128-parent panels\n\nAll region arms output the same corrected E trajectory, use the same q0/grid/update/readout, and recompute their own event interval. This is an evidence-pooling diagnostic; SourceROI here is not an expert-free complete system.\n\n|Dataset|Pooling region|tIoU %|vIoU %|\n|---|---|---:|---:|\n'
for ds in COUNTS:
 for a,m in units['regions_'+ds]['report']['means'].items():
  if a!='Source':text+=f'|{ds}|{a}|{100*m["t"]:.4f}|{100*m["v"]:.4f}|\n'
text+='\n|Dataset|Corrected ROI minus|Delta tIoU pp [95% CI]|Delta vIoU pp [95% CI]|\n|---|---|---|---|\n'
for ds in COUNTS:
 for k,v in units['regions_'+ds]['report']['paired'].items():
  if k.startswith('CorrectedROI-'):text+=f'|{ds}|{k.split("-",1)[1]}|{diff(v["t"])}|{diff(v["v"])}|\n'
text+='\nRandomRegion translates the exact expanded1.5ROI fractional-weight support template to a SHA-selected feasible valid-grid placement per physical frame (seed20261104). Area, shape and weight multiset remain bitwise identical; overlaps, identity and fallback cases remain in the denominator. Manifest committed before labels. This controls region support size without asserting that random regions never contain the object.\n\n'
for ds in COUNTS:
 z=read(P/'regions'/ds/'random_pooling_diagnostics.json');text+=f'{ds}: {z["frames"]}frames, {z["identity_frames"]}identity placements, {z["unchanged_feature_queries"]}unchanged-feature queries, {z["fallback_target_frames"]}whole-valid target fallback frames; mean support overlap {z["mean_frame_overlap"]:.6f}.\n\n'
text+='## Interpretation and limits\n\nHC corrected ROI improves tIoU over random regions, while its global comparison crosses zero; Vid corrected versus every alternative regional evidence t/v CI crosses zero. The intervention does not establish two-domain target-region necessity or language/action semantic recognition. A CI crossing zero does not prove equivalence. Existing AUROC is descriptive evidence, not causal proof of object specificity. No new parameter/seed choice follows these results.\n\nThe unchanged Full still permits corrected E versus rawexpert candidates at observed frames; it does not satisfy the intended hard-anchor contract. These results describe the existing soft-anchor implementation, not a repair or validation of a hard-anchor method. All observed-frame metrics include GT-supported frames outside predicted intervals; they are not all tube-inside damage. Source-to-SourceTemporal native gains combine legal jointMAP readout change and visual evidence; existing matched SourceJointMAP controls are exported separately.\n\nPointwise10000paired-parent CIs, query-equal means, exposed cohorts, no familywise/unknown-test/equivalence claims. GT only after whole seals. Short<=5s/long groups, all signed harms, severe t/v union once strictly over5pp, rescue counts, observed/unobserved and boundary diagnostics are retained in each ROOT report and all-signed TSV. Inherited Source/expert acquisition remains nonzero. New prediction/score/audit walls are cached CPU experimental processing, not deployment latency; stage windows and nested seal clocks must not be added as separate costs. Vid construction revalidates the full manifest repeatedly in its frozen entry; this affects CPU experiment wall, not predictions.\n'
(P/'ANALYSIS.md').write_text(text)
cost=dict(new_model=0,new_Source=0,new_expert=0,new_GPU=0,new_optimizer=0,inherited_Source_and_expert_acquisition_cost_nonzero=True,deployment_latency_measured_here=False,actual_stage_costs=exits,prediction_seal_clocks={k:z['prediction_wall_seconds']for k,z in units.items()},windows_nested_not_additive=True,root_outer_wait=read(D/'ROOT_actual_exits_v1.json'),report_failure_clock=None,report_failure_clock_reason='Outer v1 exporter result lost in tool response truncation; traceback preserved, not imputed0')
atomic(P/'cached_CPU_actual_costs.json',cost)
summary=dict(status='all_authorized_two_reviewer_controls_scored_audited_ROOTcomplete_PENDING_publication',scientific_fingerprint=validate()['fingerprint'],units={k:{f:v for f,v in z.items()if f!='report'}for k,z in units.items()},new_official_checks=checks,max_official_discrepancy=maxerr,root_paired_group_CI_checks=rootchecks,successful_actual_subprocess_stages=12,controller_actual_wait=outer,all_owned_incarnations_exited=True,new_Source=0,new_expert=0,new_model=0,new_GPU=0,new_optimizer=0,Current_observed_contract_changed=False,paper_changed=False,old_scopes_resumed=False,analysis_sha256=sha(P/'ANALYSIS.md'),cost_sha256=sha(P/'cached_CPU_actual_costs.json'),time=time.time())
atomic(D/'ROOT_scientific_completion_v1.json',summary);atomic(P/'completion_scalars.json',summary);print(json.dumps(summary))

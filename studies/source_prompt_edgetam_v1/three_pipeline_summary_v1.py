"""Join separately audited fixed128 single-expert pipelines; no new GT metric."""
from pathlib import Path
import json,hashlib,csv
R=Path(__file__).resolve().parents[2]
D=R/'outputs/source_prompt_edgetam_v1'
P=R/'results/source_prompt_edgetam_v1'
F=R/'outputs/fresh_external_comparison_v1/referformer_v4'
FP=R/'results/fresh_external_comparison_v1/referformer_v4'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
 pub=read(F/'complete_published_small_v1.json');assert pub['all_remote_verified']and sha(Path(pub['completion_path']))==pub['completion_sha256']
 tables={};proof={};lines=['# Single-expert complete pipelines: fixed128 parents per domain','', 'Each branch has one external pretrained expert. Source-prompt EdgeTAM uses only original Source boxes and Source native time, without ReferDINO, current posterior/time or current DP. ReferFormer substitutes the expert and rebuilds its own mask geometry, E-ROI features, posterior, event and final DP. The current method uses ReferDINO. These interfaces and compute are different; complete-tube differences do not isolate spatial propagation. The old serial ReferDINO-to-EdgeTAM result is an additional-model extension and is excluded from this single-expert table.','']
 for ds in ['hc2','vidstg']:
  sf=D/'pilot_v1'/ds;rf=F/'small_v1'/ds
  sa=read(sf/'reporting_receipt.json');ra=read(rf/'reporting_receipt.json')
  assert sa['status']=='verified_whole128_Source_prompt_only_all_mask_input_arrays_and_means'and ra['status']=='verified_fixed128_expert_generalization_all_arrays_Source_teacher_and_means'
  for f,a in [(sf,sa),(rf,ra)]:assert sha(f/'metrics.json')==a['metrics_sha256']
  ss={r['id']:r for r in read(sf/'metrics.json')['private_rows']};rr={r['id']:r for r in read(rf/'metrics.json')['private_rows']};assert len(ss)==len(rr)==128 and set(ss)==set(rr)
  for key,s in ss.items():
   r=rr[key];assert s['parent']==r['parent']and s['duration_group']==r['duration_group']
   for arm in ['Source','CurrentSimple','CurrentOurs']:assert s['values'][arm]==r['values'][arm],(ds,key,arm)
  sr=read(P/ds/'report.json');ref=read(FP/'small_v1'/ds/'report.json')
  tables[ds]={name:{k:100*m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']}for name,m in [('Source',sr['means']['Source']),('ReferDINO_current',sr['means']['CurrentOurs']),('Source_box_EdgeTAM',sr['means']['EdgeTAM']),('ReferFormer_current',ref['means']['AlternativeOurs'])]}
  lines.extend([f'## {ds}', '', '|Pipeline|tIoU %|vIoU %|R.3 %|R.5 %|','|---|---:|---:|---:|---:|'])
  for name,m in tables[ds].items():lines.append(f"|{name}|{m['t']:.6f}|{m['v']:.6f}|{m['vIoU_ge_03']:.6f}|{m['vIoU_ge_05']:.6f}|")
  lines.extend(['',f"Current-minus-Source-box-EdgeTAM v{sr['paired']['CurrentOurs-EdgeTAM']['v']['mean']:+.6f}pp, parent-paired95%CI{sr['paired']['CurrentOurs-EdgeTAM']['v']['CI95']}; ReferFormer-current minus ReferDINO-current v{ref['paired']['AlternativeOurs-CurrentOurs']['v']['mean']:+.6f}pp, CI{ref['paired']['AlternativeOurs-CurrentOurs']['v']['CI95']}.",''])
  proof[ds]=dict(queries=128,parents=128,all_perquery_Source_CurrentSimple_CurrentOurs_metrics_and_parent_duration_exact_across_both_studies=True,Source_Edge_audit_sha256=sha(sf/'reporting_receipt.json'),ReferFormer_audit_sha256=sha(rf/'reporting_receipt.json'),Source_Edge_root_export_sha256=sha(sf/'root_export_receipt_v1.json'),ReferFormer_root_cost_sha256=sha(rf/'root_cost_export_receipt_v1.json'))
 lines.extend(['Full signed/tails/short-long parent CIs are retained in each study. All CIs are pointwise; the fixed cohorts have prior development exposure. A CI crossing zero is not equivalence or expert-independence proof. No model/checkpoint/threshold/prompt direction is selected from these results. Source-only EdgeTAM has unchanged Source temporal readout by design, whereas the other complete methods have their own temporal inference. Single-expert count is controlled, not pretrained knowledge, processed frames, input resolution or compute.','', 'Both256 small groups are complete; no full12205 completion or complete uncached deployment latency is claimed. CG, gamma and corruption remain paused by the user; additional CG-HC cancelled.'])
 (P/'THREE_SINGLE_EXPERT_PIPELINES_v1.md').write_text('\n'.join(lines)+'\n');(P/'three_single_expert_means_v1.json').write_text(json.dumps(tables,indent=2)+'\n')
 with(P/'three_single_expert_means_v1.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['dataset','pipeline','tIoU_percent','vIoU_percent','R03_percent','R05_percent'])
  for ds,tab in tables.items():
   for name,m in tab.items():w.writerow([ds,name,*[m[k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]])
 receipt=dict(status='root_three_single_expert_pipeline_tables_crossjoined_perquery_metrics_exact',datasets=proof,code_sha256=sha(Path(__file__)),ReferFormer_publication_sha256=sha(F/'complete_published_small_v1.json'),new_GT_scoring=0,new_model=0,new_forward=0,full12205_complete=False)
 (D/'three_pipeline_root_receipt_v1.json').write_text(json.dumps(receipt,indent=2)+'\n');(P/'three_pipeline_root_receipt_v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
if __name__=='__main__':run()

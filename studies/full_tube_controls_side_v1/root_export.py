"""Report-only root export after complete official scoring and cold-array audit."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import runtime_v3 as runtime
C=runtime.original
import json,csv,hashlib,time,argparse
import numpy as np

def export(ds):
 start=time.perf_counter();runtime.validate();d=C.D/ds;p=C.P/ds;a=C.read(d/'reporting_receipt.json');score=C.read(d/'score_receipt.json');seal=C.read(d/'sealed.json');rows=C.read(d/'metrics.json')['private_rows'];report=C.read(p/'report.json')
 assert a['status']=='verified_whole_domain'and a['queries']==C.COUNTS[ds]and a['score_sha256']==C.sha(d/'score_receipt.json')and a['metrics_sha256']==C.sha(d/'metrics.json')and a['seal_sha256']==C.sha(d/'sealed.json')
 assert score['checks']==4*C.COUNTS[ds]and score['whole_domain_sealed_before_GT']and a['all_arrays_cold_rederived_exact']and a['all_means_recomputed']
 for name in C.ARMS:
  for field,v in report['means'][name].items():
   x=[r['values'][name][field]for r in rows if r['values'][name][field]is not None];assert(v is None and not x)or abs(v-sum(x)/len(x))<2e-12
 assert all(all(r['values'][name]['t']==r['values']['Source']['t']for name in C.NEW_ARMS)for r in rows)
 full_fields=['t','v','s','vIoU_ge_03','vIoU_ge_05','fixed_actual_sIoU','observed_actual_sIoU','unobserved_actual_sIoU','boundary_MAE_seconds']
 parents={v:f'P{i:04d}'for i,v in enumerate(sorted({r['parent']for r in rows}),1)}
 f=p/'all_metrics_signed_parent.tsv'
 with f.open('w')as h:
  w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['query','parent','duration_group']+[name+'_'+field for name in C.ARMS for field in full_fields]);w.writerows([[f'Q{i:05d}',parents[r['parent']],r['duration_group']]+[r['values'][name][field]for name in C.ARMS for field in full_fields]for i,r in enumerate(rows,1)])
 with f.open()as h:public=list(csv.DictReader(h,delimiter='\t'))
 assert len(public)==C.COUNTS[ds]and len({r['parent']for r in public})==len(parents)
 for name in C.ARMS:
  for field in full_fields:
   x=[float(r[name+'_'+field])for r in public if r[name+'_'+field]!=''];v=report['means'][name][field];assert(v is None and not x)or abs(v-sum(x)/len(x))<2e-12
 tails={}
 for left,right in C.PAIRS:
  vals=[(100*(r['values'][left]['t']-r['values'][right]['t']),100*(r['values'][left]['v']-r['values'][right]['v']))for r in rows]
  tails[left+'-'+right]=dict(severe_t=sum(t< -5 for t,v in vals),severe_v=sum(v< -5 for t,v in vals),severe_union_once=sum(t< -5 or v< -5 for t,v in vals),rescue_union_once=sum(t>5 or v>5 for t,v in vals),both_t_v_harmed=sum(t< -1e-10 and v< -1e-10 for t,v in vals),both_t_v_improved=sum(t>1e-10 and v>1e-10 for t,v in vals))
 C.atomic(p/'all_paired_tails.json',dict(paired=report['paired'],short_long=report['timegroups'],severe_rescues=tails,CI='Pointwise 95% parent-video clustered10000 bootstrap, query-equal means, no familywise correction. All target sets previously exposed.'))
 table=p/'full_tube_metrics.tsv'
 with table.open('w')as h:
  w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['configuration','tIoU_percent','vIoU_percent','sIoU_percent','R03_percent','R05_percent']);w.writerows([[name,*[100*report['means'][name][field]for field in ['t','v','s','vIoU_ge_03','vIoU_ge_05']]]for name in C.ARMS])
 with(p/'temporal_branch_ablation.tsv').open('w')as h:
  w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['configuration','temporal_output','tIoU_percent','vIoU_percent','R03_percent','R05_percent']);w.writerows([[name,'original Source per-offset MAP/hull'if name=='NoTemporalRefinement'else'own appearance-updated joint MAP',*[100*report['means'][name][k]for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]]for name in ['NoTemporalRefinement','FullMAP']])
 with(p/'table2_simple_fusion.tsv').open('w')as h:
  w=csv.writer(h,delimiter='\t',lineterminator='\n');w.writerow(['configuration','time_rule','tIoU_percent','vIoU_percent','sIoU_percent','R03_percent','R05_percent']);w.writerows([[name,'own full temporal refinement'if name=='FullMAP'else'original Source native MAP/hull',*[100*report['means'][name][k]for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05']]]for name in ['Source','EightReplacementNative','SharedCorrectionNative','ExpertInterpolationNative','FullMAP']])
 oldrows=C.read(C.selected.D/'components'/ds/'metrics.json')['private_rows'];oldmean={k:sum(r['values']['SpatialInference'][k]for r in oldrows)/len(oldrows)for k in ['t','v']}
 C.atomic(p/'simple_fusion_protocol_distinction.json',dict(current_full_tube_time='Original Source native per-offset MAP/hull for all three simple fusions',old_component_time='q0 joint-MAP readout; component ablation, not original native Source full-tube protocol',old_component_rows_not_renamed=True,previous_component_means_percent={a:{k:100*sum(r['values'][a][k]for r in oldrows)/len(oldrows)for k in ['t','v']}for a in ['EightReplacement','SharedCorrection']},new_full_tube_means_percent={a:{k:100*report['means'][a][k]for k in ['t','v']}for a in ['EightReplacementNative','SharedCorrectionNative']}))
 C.atomic(p/'temporal_control_semantics.json',dict(NoVisualUpdate=dict(definition='No appearance posterior update, still uses q0 legal joint-MAP readout and final spatial DP',inherited_previous_component_mean_percent={k:100*v for k,v in oldmean.items()},newly_scored_in_this_study=False),NoTemporalRefinement=dict(definition='Bypasses complete temporal refinement/readout replacement, original per-offset Source MAP/hull event and reexecuted interval-conditioned final spatial DP',newly_scored_in_this_study=True,mean_percent={k:100*report['means']['NoTemporalRefinement'][k]for k in ['t','v']}),these_are_distinct_controls=True))
 name_map={'Source':'Original Source','EightReplacementNative':'Eight-frame replacement + Source time','SharedCorrectionNative':'Shared correction + Source time','ExpertInterpolationNative':'Same-eight interpolation + Source time','NoTemporalRefinement':'Full spatial DP + Source time; temporal refinement entirely disabled','FullMAP':'Current full method'}
 text=f'# Complete full-tube controls: {ds}\n\n{len(rows)} complete queries, {len(parents)} parent videos. Predictions sealed before independent official CPU scoring; independent all-array cold replay, SHA/provenance and mean audit passed. No new model/Source/expert/optimizer/GPU. Original current method and all older frozen studies remain unchanged.\n\n| Configuration | tIoU % | vIoU % | sIoU % | R.3 % | R.5 % |\n|---|---:|---:|---:|---:|---:|\n'
 for name in C.ARMS:text+='|'+name_map[name]+'|'+'|'.join(f'{100*report["means"][name][k]:.6f}'for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05'])+'|\n'
 text+='\nThe first ablation bypasses all temporal refinement/readout replacement and uses original Source native per-offset MAP/hull time, then actually reruns the final interval-conditioned DP with the same MAIN candidate geometry, appearance/link scores and coefficients. Event-dependent eligibility and original normalization recompute under that interval. It reproduces the previously independently scored NoTemporalInference arrays and scalar metrics exactly. It is not the NoVisualUpdate / q0 joint-MAP row.\n\nThe three simple fusions use original Source time independently of Ours. Replacement changes exactly eight observed boxes and keeps every unobserved Source box. Shared correction uses the saved E trajectory. Interpolation uses the original center/logsize rule and E outside anchor coverage. No simple method receives our posterior or trajectory DP. All four controls must therefore have exactly the same tIoU as original Source. Their vIoU/R metrics are independently scored from the newly composed full tubes, not transplanted component-table values.\n\nHistorical cache naming: the old parent Source_interval and OursS use the historical added MBR readout. These interval fields are deliberately excluded; the original Source interval is verified against the corrected current/structural audits and actual saved native logits in the four-case bridge. No legacy output is silently relabeled.\n\n## Complete paired outcomes\n\n| Comparison | delta tIoU pp [95% CI] | delta vIoU pp [95% CI] | severe t/v/union | rescues |\n|---|---:|---:|---:|---:|\n'
 for key,pr in report['paired'].items():
  def cell(field):
   v=pr[field];return f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"if v['CI95']is not None else'not supported'
  z=tails[key];text+=f"|{key}|{cell('t')}|{cell('v')}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 text+='\nShort (GT duration <=5 seconds) and long paired CIs, all signed positive/negative query outcomes, all mean metrics and all severe losses/rescues are retained in the accompanying JSON/TSVs. CIs crossing zero do not establish equivalence. These historically exposed full sets do not constitute new blind validation and do not select parameters.\n\nCosts are cached derivation, label scoring and cold-audit elapsed wall, not process CPU time or deployment latency. Source and expert observations were acquired in earlier workers and remain nonzero inherited costs; the original Vid Source producer was shared with TTA and must not be counted twice. No claim of newly measured full-pipeline latency is made.\n'
 (p/'ANALYSIS.md').write_text(text)
 producer=C.selected.RUN/ds/'sealed.json';priorcol=C.selected.PREV/'spatial'/f'{ds}_sealed.json';lineage=dict(cache_prediction=dict(worker=seal['worker'],whole_elapsed_wall_seconds=seal['wall_seconds'],per_worker_sum_cached_query_wall_seconds=a['cached_query_wall_segments']),new_official_score=dict(worker=score['worker'],elapsed_wall_seconds=score['wall_seconds'],checks=score['checks']),independent_cold_audit=dict(worker=a['worker'],elapsed_wall_seconds=a['wall_seconds']),inherited_producer_seal_sha256=C.sha(producer),inherited_spatial_collector_seal_sha256=C.sha(priorcol),inherited_Source_and_expert_cost_nonzero=True,shared_Vid_Source_producer_with_TTA_never_double_counted=True,preflight_v1_v2_failed_wall_seconds=None,missing_clock_is_not_zero=True,nested_times_not_additive=True,deployment_latency_measured=False)
 C.atomic(p/'cost_lineage.json',lineage)
 rec=dict(status='root_exported_complete_domain',queries=len(rows),parents=len(parents),root_report_source_sha256=C.sha(Path(__file__)),audit_sha256=C.sha(d/'reporting_receipt.json'),score_sha256=C.sha(d/'score_receipt.json'),public_table_SHA256=C.sha(table),public_full_signed_SHA256=C.sha(f),public_TSV_means_independently_recomputed_exact=True,analysis_sha256=C.sha(p/'ANALYSIS.md'),elapsed_wall_seconds=time.perf_counter()-start,root_process=C.identity(),publication_pending=True)
 C.atomic(d/'root_export_receipt.json',rec);C.atomic(p/'root_export_receipt.json',rec);print(json.dumps(dict(dataset=ds,means={name:{k:100*v for k,v in report['means'][name].items()if k in ['t','v','s','vIoU_ge_03','vIoU_ge_05']}for name in C.ARMS},paired=report['paired'],tails=tails),indent=2),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=C.COUNTS);export(ap.parse_args().dataset)

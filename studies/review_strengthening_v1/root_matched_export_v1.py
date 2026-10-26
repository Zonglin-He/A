"""Root reads only fully audited scalars; no model/GT/prediction execution."""
from review_common import *
from matched_cpu import ARMS,PAIRS
import argparse,csv

def run(ds):
 import numpy as np
 validate();dest=D/'matched_cpu'/ds;folder=P/'matched_cpu'/ds;ap=dest/'reporting_receipt.json';sp=dest/'score_receipt.json';sealpath=dest/'sealed.json';audit=read(ap);score=read(sp);seal=read(sealpath)
 assert audit['status']=='verified_whole_domain'and audit['all_arrays_cold_rederived_exact']and audit['all_means_recomputed']and audit['all_temporal_IoUs_identical'];assert audit['score_sha256']==sha(sp)and score['seal_sha256']==sha(sealpath)and audit['seal_sha256']==sha(sealpath)
 rr=read(dest/'metrics.json')['private_rows'];r=read(folder/'report.json');assert len(rr)==COUNTS[ds]==seal['queries']and not seal['GT_read'];assert r['parents']=={'hc2':227,'vidstg':732}[ds]
 for row in rr:assert len({row['values'][a]['t']for a in ARMS})==1
 for a,mm in r['means'].items():
  for k,v in mm.items():
   vals=[row['values'][a][k]for row in rr if row['values'][a][k]is not None];assert(v is None and not vals)or abs(v-float(np.mean(vals)))<1e-12
 tails={}
 for a,b in PAIRS:
  dt=np.array([100*(row['values'][a]['t']-row['values'][b]['t'])for row in rr]);dv=np.array([100*(row['values'][a]['v']-row['values'][b]['v'])for row in rr]);tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()));assert np.array_equal(dt,np.zeros_like(dt))
 atomic(folder/'all_paired_tails.json',tails)
 costs=dict(actual_prediction_worker=seal['worker'],cached_prediction_elapsed_wall_seconds=seal['seconds'],actual_score_worker=score['worker'],actual_independent_audit_worker=audit['worker'],new_Source=0,new_expert=0,new_model=0,inherited_Source_and_expert_acquisition_nonzero=True,cost_scope='Saved predictions/evidence reused; cached conditional replay/checks/CPU official scoring are not uncached deployment. Source/teacher acquisition was previously paid and is nonzero. Original failed report score passed official per-query loops but was not saved; recovery repeated scoring, disclosed separately.',interface_failure=dict(preservation_sha256=sha(D/'preserved/report_interface_v1/preservation.json'),original_score_exit=read(D/'hc2_score_exit.json')if ds=='hc2'else None,original_score_wall_seconds=None,missing_clock='No standalone full-stage score clock was saved; no invented zero cost.'),SAM2_model_forward=False,full_external_propagation_comparison_complete=False)
 atomic(folder/'cost_segments.json',costs)
 def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"
 text=f'# Same eight observations, matched current temporal interval: {ds}\n\nComplete {len(rr)} queries / {r["parents"]} parents; {score["checks"]} official/dense checks, max error {score["max_error"]:.17g}; {audit["prediction_receipt_SHA_checks"]} prediction/receipt SHA checks plus bound parent/current references, exact cold arrays and all means. Root analysis reads audited scalar metrics only, zero new GT scoring/model/Source/expert.\n\n|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|---:|\n'
 for a,m in r['means'].items():text+=f'|{a}|'+ '|'.join(f'{100*m[k]:.6f}'for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05'])+'|\n'
 text+='\n|Comparison|Delta v pp [95% CI]|Severe v|Rescue union|\n|---|---:|---:|---:|\n'
 for key,x in r['paired'].items():text+=f'|{key}|{fmt(x["v"])}|{tails[key]["severe_v"]}|{tails[key]["rescue_union_once"]}|\n'
 text+='\nAll intervals and tIoU are exactly identical across these rows: same audited current selected q1 MAP interval. This isolates spatial trajectory conditional on Ours temporal evidence; it is not a claim that each competitor has independent equal-quality temporal inference. SourceMatchedTime is Source boxes under this common refined interval, not the native Source main-table baseline. EightReplacement keeps Source outside eight exact original teacher positions. ExpertInterpolation exactly reuses original center/log-size interpolation and E outside coverage; neither competitor receives our DP. FinalIndependent is the already-measured internal fixed-candidate control, not a new external propagation method. Ours and FinalIndependent reproduce the current main/final-DP references. SAM2 is not in this table and is still pending; no full mature-propagation superiority claim follows yet.\n\n|Duration / comparison|Queries|Delta v pp [95% CI]|\n|---|---:|---:|\n'
 for group,g in r['timegroups'].items():
  for key,x in g['paired'].items():text+=f'|{group} / {key}|{g["queries"]}|{fmt(x["v"])}|\n'
 text+='\nShort is GT duration <=5s; long >5s. Pointwise parent-cluster10000 bootstrap95%, not familywise; fullsets were previously exposed, no parameter/sample selection. All signed positive/negative metrics, fixed annotated support observed/unobserved sIoU, duration CIs, severe >5pp union once and rescues remain in report/TSVs. A positive mean is not every-query improvement. Native temporal Source-relative severe counts from the main table must not be confused with this fixed-time spatial table.\n\n## Costs and preserved engineering failures\n\n'+json.dumps(costs,indent=2)+'\n'
 (folder/'ANALYSIS.md').write_text(text)
 with(folder/'main_table.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['method','tIoU_percent','vIoU_percent','sIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*m[k]for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05']]]for a,m in r['means'].items()])
 rec=dict(status='root_complete_matched_temporal_spatial_analysis',dataset=ds,queries=len(rr),code_sha256=sha(Path(__file__)),audit_sha256=sha(ap),score_sha256=sha(sp),seal_sha256=sha(sealpath),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),all_paired_tails_duration_means_checked=True,SAM2_complete=False,new_model=0,new_GT_scoring=0)
 atomic(dest/'root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec);print(ds,'root matched-time full analysis complete')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);run(a.parse_args().dataset)

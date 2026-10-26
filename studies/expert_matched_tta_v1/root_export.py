"""Root report of already whole-domain audited full tubes, no new GT/model."""
from common import *
import argparse,csv
def run(ds):
 import numpy as np
 validate();rc.gt_guard();dest=D/ds;folder=P/ds;receipt=read(dest/'reporting_receipt.json');sc=read(dest/'score_receipt.json');se=read(dest/'sealed.json')
 assert receipt['status']=='verified_whole_domain'and receipt['all_arrays_cold_rederived_exact']and receipt['all_means_recomputed']and receipt['competitor_intervals_from_own_native_adapted_logits'];assert receipt['score_sha256']==sha(dest/'score_receipt.json')and sc['seal_sha256']==sha(dest/'sealed.json')and receipt['metrics_sha256']==sha(dest/'metrics.json')
 rows=read(dest/'metrics.json')['private_rows'];report=read(folder/'report.json');assert len(rows)==COUNTS[ds]and report['parents']=={'hc2':227,'vidstg':732}[ds]
 for a,m in report['means'].items():
  for k,v in m.items():
   x=[r['values'][a][k]for r in rows if r['values'][a][k]is not None];assert(v is None and not x)or abs(v-float(np.mean(x)))<1e-12
 tails={}
 for a,b in PAIRS:
  dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rows]);dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescues_union_once=int(((dt>5)|(dv>5)).sum()),v_improve=int((dv>0).sum()),v_harm=int((dv<0).sum()),v_tie=int((dv==0).sum()))
 atomic(folder/'all_paired_tails.json',tails)
 costs=dict(prediction_worker=se['worker'],cached_composition_elapsed_wall_seconds=se['seconds'],official_score_worker=sc['worker'],official_score_elapsed_wall_seconds=sc['elapsed_wall_seconds'],independent_audit_worker=receipt['worker'],audit_elapsed_wall_seconds=receipt['elapsed_wall_seconds'],original_TTA_costs=read(T/f'{ds}_reporting_receipt.json')['worker_cost_segments'],new_Source=0,new_expert=0,new_optimizer=0,original_expert_and_Source_acquisition_nonzero=True,uncached_complete_combination_latency_seconds=None,latency_status='Not yet measured for new combinations. Saved composition/score/audit elapsed wall is not deployment; inherited nested acquisition/optimizer/Source/teacher costs must not be added twice. Ours independently measured latency is not a substitute for these combinations.')
 atomic(folder/'cost_segments.json',costs)
 text=f'# Identical eight expert observations, independently inferred full tubes: {ds}\n\n{len(rows)} queries / {report["parents"]} parents; whole seal before {sc["checks"]} new official metric checks (Source and Ours metrics inherited with exact array proofs), maximum discrepancy {sc["max_error"]:.17g}; independent {receipt["prediction_receipt_SHA_checks"]} prediction/receipt SHA checks plus all-parent/TTA provenance and all arrays cold rederived exact.\n\n|Method|tIoU %|vIoU %|sIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|---:|\n'
 for a,m in report['means'].items():text+='|'+a+'|'+'|'.join(f'{100*m[k]:.6f}'for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05'])+'|\n'
 text+='\n|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe union|Rescue union|\n|---|---:|---:|---:|---:|\n'
 def fmt(z):return f'{z["mean"]:+.6f} [{z["CI95"][0]:.6f}, {z["CI95"][1]:.6f}]'
 for key,z in report['paired'].items():text+=f'|{key}|{fmt(z["t"])}|{fmt(z["v"])}|{tails[key]["severe_union_once"]}|{tails[key]["rescues_union_once"]}|\n'
 text+='\nUnlike the prior matched-time spatial table, each Source/Tent/SAR interpolation row uses its own original native per-offset MAP and physical-hull event interval. No Ours event interval or temporal-posterior correction is injected. All three share the exact original same-eight center/log-size interpolation tensor including the original E outside coverage; no DP is attached to interpolation. Ours retains its current structured spatial and temporal inference. The no-update SourceExpertInterpolation row separates expert fusion from optimizer effects. The current official pinned Tent/SAR STVG adapters update temporal-decoder LayerNorm affine only, with fixed Source boxes and five episodic steps; not canonical BN/full-network/continual method rankings. Actual parameter/logit changes and SAR empty reliable-filter cases are retained, never forced argmax changes.\n\nAll t/v/s/R.3/R.5 signed metrics, parent-cluster 10000 bootstrap pointwise confidence intervals (not familywise), severe >5pp t/v union once/rescues, short <=5s and long >5s are preserved. Full datasets are historically exposed, no parameter or sample sweep. These results address same-observation evidence usage, not equal compute budgets.\n\n## Duration-specific outcomes\n\n'
 for group,g in report['timegroups'].items():
  text+=f'### {group}, {g["queries"]} queries\n\n'
  for key,z in g['paired'].items():text+=f'- {key}: t {fmt(z["t"])}, v {fmt(z["v"])} pp.\n'
 text+='\n## Measured and missing costs\n\n'+json.dumps(costs,indent=2)+'\n';(folder/'ANALYSIS.md').write_text(text)
 with(folder/'main_table.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['method','tIoU_percent','vIoU_percent','sIoU_percent','R03_percent','R05_percent']);w.writerows([[a,*[100*m[k]for k in ['t','v','s','vIoU_ge_03','vIoU_ge_05']]]for a,m in report['means'].items()])
 out=dict(status='root_complete_full_tube_expert_matched_TTA_analysis',dataset=ds,queries=len(rows),code_sha256=sha(Path(__file__)),audit_sha256=sha(dest/'reporting_receipt.json'),score_sha256=sha(dest/'score_receipt.json'),seal_sha256=sha(dest/'sealed.json'),report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),independent_adapted_intervals_verified=True,all_signed_tails_duration_means_checked=True,new_model=0,new_GT_scoring=0)
 atomic(dest/'root_export_receipt.json',out);atomic(folder/'root_export_receipt.json',out);print(ds,'root complete')
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True);run(a.parse_args().dataset)

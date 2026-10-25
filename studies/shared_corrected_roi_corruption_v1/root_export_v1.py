"""Read-only root analysis after all384old corruption panel views/domain score and audit."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from panel_common import *
import argparse

def summarize(folder, rr, pairs):
 import numpy as np
 report=read(folder/'report.json');assert report['queries']==len(rr)
 for arm,means in report['means'].items():
  for metric,value in means.items():
   vals=[r['values'][arm][metric]for r in rr if r['values'][arm][metric]is not None]
   assert (value is None and not vals)or abs(value-float(np.mean(vals)))<1e-12
 tails={}
 for left,right in pairs:
  dt=np.asarray([100*(r['values'][left]['t']-r['values'][right]['t'])for r in rr]);dv=np.asarray([100*(r['values'][left]['v']-r['values'][right]['v'])for r in rr])
  tails[left+'-'+right]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()))
 atomic(folder/'all_paired_tails.json',tails)
 def ci(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:.6f}, {x['CI95'][1]:.6f}]"
 text='|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Severe t/v/union vs corrupted-or-native Source|\n|---|---:|---:|---:|---:|---|\n'
 for arm,m in report['means'].items():
  z=report['severe_vs_Source'][arm];text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|{z['t']}/{z['v']}/{z['union']}|\n"
 text+='\n|Signed comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union|Rescue union|\n|---|---:|---:|---|---:|\n'
 for key,paired in report['paired'].items():
  z=tails[key];text+=f"|{key}|{ci(paired['t'])}|{ci(paired['v'])}|{z['severe_t']}/{z['severe_v']}/{z['severe_union_once']}|{z['rescue_union_once']}|\n"
 text+='\n## Duration groups\n\nShort is GT duration <=5 seconds, long is >5 seconds. All intervals are pointwise parent-cluster bootstrap95%, not familywise.\n\n|Group / comparison|Queries|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---:|---:|\n'
 for group,g in report['timegroups'].items():
  for key,paired in g['paired'].items():text+=f"|{group} / {key}|{g['queries']}|{ci(paired['t'])}|{ci(paired['v'])}|\n"
 text+='\nEvery signed positive/negative result, fixed-support observed/unobserved metric and boundary error remains in report.json and the signed TSVs. Severe means a loss greater than5pp; union counts a query once. Rescue means an increase greater than5pp in t or v. Source is native per-offset legal MAP plus physical hull. Historically exposed cohorts and user-directed method selection are not unseen confirmation; zero-crossing intervals are not equivalence proofs.\n'
 return report,text

def run(ds):
 validate();ap=D/f'{ds}_reporting_receipt.json';sp=D/f'{ds}_score_receipt.json';sealed=D/f'{ds}_sealed.json';audit=read(ap);score=read(sp);seal=read(sealed)
 assert audit['status']=='verified_whole_domain' and audit['views']==384 and audit['all_arrays_cold_rederived_exact'] and audit['all_means_recomputed']
 assert audit['score_sha256']==sha(sp) and score['seal_sha256']==sha(sealed) and seal['views']==384 and not seal['GT_read']
 assert score['matched_clean_audit_sha256']==sha(selected.D/f'components_{ds}_reporting_receipt.json')
 rr=read(D/ds/'metrics.json')['private_rows'];assert len(rr)==384;view_receipts={}
 for view in VIEWS:
  folder=P/ds/view;vr=[r for r in rr if r['view']==view];report,body=summarize(folder,vr,PAIRS);assert len(vr)==128 and report['parents']==128
  costs=dict(domain_actual_counts=seal['counts'],actual_worker=seal['worker'],whole_worker_elapsed_seconds=seal['seconds'],model_load_once_for_actual_worker=seal['model_load'],actual_PID_start_domain_segments=audit['worker_segments'],new_full_Source=0,new_expert=0,original_Source_teacher_acquisition_nonzero=True,selected_first_auxiliary_DP=False,scope='Domain costs are repeated in three view reports for reference, not three independent acquisitions. Prefix/compatibility/selected-finish/pipeline clocks are nested; decoding/transforms/outer-loop live in whole-worker wall. Not deployment latency. Three views share128parents. Only own new committed prefix raw disposed; no arbitrary later repooling.')
  atomic(folder/'cost_segments.json',costs)
  text=f'# Old corruption panel synchronized to shared-corrected-box ROI: {ds} / {view}\n\nAll384views/domain sealed before independent official labels, then complete cold all-array/parent/teacher/SHA and mean audit. This view is128queries /128parents. Domain {score["checks"]} official metric checks, maximum error {score["max_error"]:.17g}, {audit["SHA_checks"]} prediction/compact/receipt SHA checks plus original provenance. This768view total panel study is not the219690-view full corruption study. Root reads audited scalars and costs only, no model/Source/prefix/expert/optimizer/GT scoring.\n\n'+body
  text+='\n## Matched inputs and clean reference\n\nFullMAP and Source use the same original corrupted RGB/input/frame grid/positions. HistoricalTwoDP is the immutable earlier kappa0 method. CleanFullMAP/CleanSource are audited selected clean references joined by the same query/parent/duration; they are not extra corrupted predictions. Complete same-corrupted-eight teacher context/key/payload SHA must match; no clean teacher substitution. Original appearance prefix reconstruction must exactly reproduce old ROI pools/arrays before actual new corrected-box pooling, newposterior/jointMAP/newintervalfinalDP. First auxiliary DP only executes in hidden historical compatibility, never selected inference. No claims of unconditional robustness; retain all clean-relative losses and each view separately.\n\n## Costs\n\n'+json.dumps(costs,indent=2)+'\n'
  (folder/'ANALYSIS.md').write_text(text);view_receipts[view]=dict(report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'))
 receipt=dict(status='root_complete_all_three_views_matched_clean_paired_tails_duration_cost_analysis',dataset=ds,code_sha256=sha(Path(__file__)),audit_sha256=sha(ap),score_sha256=sha(sp),seal_sha256=sha(sealed),views=view_receipts,new_model=0,new_Source=0,new_prefix=0,new_expert=0,new_optimizer=0,new_GT_scoring=0)
 atomic(D/f'{ds}_root_export_receipt.json',receipt);atomic(P/ds/'root_export_receipt.json',receipt);print(ds,'all three old corruption views root analysis complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=COUNTS,required=True);run(p.parse_args().dataset)

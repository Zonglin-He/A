"""Root-only signed128-panel reporting; no inference or label access."""
from subset_v1 import *

def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:+.6f}, {x['CI95'][1]:+.6f}]" if x.get('mean') is not None else 'unscorable'

def export(ds):
 validate_subset();ack=read(U/ds/'reporting_receipt.json');assert ack['status']=='verified_fixed128_ROI_uniform_gamma_CPU' and ack['report_sha256']==sha(V/ds/'report.json')
 rr=read(U/ds/'metrics.json')['private_rows'];rep=read(V/ds/'report.json');seal=read(U/ds/'sealed.json');tails={}
 for x,y in PAIRS:
  dt=np.asarray([100*(r['values'][x]['t']-r['values'][y]['t'])for r in rr]);dv=np.asarray([100*(r['values'][x]['v']-r['values'][y]['v'])for r in rr])
  tails[x+'-'+y]=dict(v_improve=int((dv>1e-10).sum()),v_harm=int((dv< -1e-10).sum()),v_tie=int((abs(dv)<=1e-10).sum()),v_gain_gt1=int((dv>1).sum()),v_loss_lt_minus1=int((dv< -1).sum()),v_loss_le_minus10=int((dv<=-10).sum()),severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()))
 query_clocks=[e['elapsed_wall_seconds']for e in seal['files'].values()]
 diag=[e['diagnostics']for e in seal['files'].values()]
 preserved=D/'full_to_subset_preserved_20261103';boundary=read(preserved/'safe_boundary.json')
 original=[]
 for f in (OLD_FULL/'hc2/receipts').glob('*.json'):original.append(read(f))
 costly=[e for e in original if e['raw_origin']['kind']=='new_Source']
 costs=dict(subset_cached_prediction_elapsed_wall_seconds=seal['elapsed_wall_seconds'],subset_query_elapsed_wall_sum_seconds=sum(query_clocks),subset_reuse_counts=seal['counts'],new_Source=0,new_expert=0,new_model=0,new_optimizer=0,GPU=False,
  original_full_acquisition=dict(status='intentionally_stopped_unscored_after_user_scope_reduction',completed_queries=boundary['progress']['completed'],counts=boundary['progress']['counts'],original_Source_forward_sum_seconds=sum(e['raw_origin']['Source_seconds']for e in costly),model_load=read(OLD_FULL/'hc2/model_load.json'),whole_worker_completion_clock=None,all_paid_commits_preserved=True,no_uncommitted_paid_attempt_at_stop=True),
  interpretation='Cached elapsed wall includes read/hash/pool/replay/serialization and is not deployment latency or process CPU time. Source/expert original acquisition nonzero. Original full-prefix Source sum is nested in original query wall; partial-prefix query costs can overlap this subset reuse and must not be added as disjoint acquisition. Original full worker was user-stopped, unknown complete-worker clock remains null.')
 diagnostics=dict(empty_background_frames=sum(e['empty_background_frames']for e in diag),shuffled_fallback_frames=sum(e['shuffled_fallback_frames']for e in diag),queries=128,parents=128)
 text=f'# Fixed128 reviewer regional and uniform-strength controls: {ds}\n\nUser explicitly replaced the remaining full-domain acquisition with a subset on 2026-11-03. This report uses the unchanged original matched-pipeline128-query/128-distinct-parent panel. It is previously exposed development data, not a new unknown test. The completed full CPU experiments remain separate and unchanged.\n\nThe original full dispatcher and GPU worker were intentionally stopped after289 committed HC queries (260 new Source forwards /29 retained raw /0 new expert), with no uncommitted paid attempt. Those outputs are preserved and unscored as a full set. This subset needs no GPU or new acquisition: all256 queries have retained exact raw grids.\n\n|Arm|tIoU %|vIoU %|\n|---|---:|---:|\n'
 for arm,m in rep['means'].items():text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|\n"
 text+='\n|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|Severe t/v/union-once|Rescue union-once|\n|---|---|---|---|---:|\n'
 for name,x in rep['paired'].items():
  t=tails[name];text+=f"|{name}|{fmt(x['t'])}|{fmt(x['v'])}|{t['severe_t']}/{t['severe_v']}/{t['severe_union_once']}|{t['rescue_union_once']}|\n"
 text+='\n## Short and long events\n\n|Group|Queries|Comparison|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---:|---|---|---|\n'
 for group,g in rep['timegroups'].items():
  for name,x in g['paired'].items():text+=f"|{group}|{g['queries']}|{name}|{fmt(x['t'])}|{fmt(x['v'])}|\n"
 text+='\nThe five region arms all output the SAME shared-corrected E trajectory and independently update their temporal posterior/interval. Source-box, corrected-box, global, background-outside-expanded-E and whole-box-permutation controls isolate temporal pooling support. They are diagnostic expert-derived fixed-space comparisons, not no-expert deployment rankings. Every whole shuffled box and size multiset is preserved; fixed output geometry is unchanged.\n\nBoth uniform gamma values .5 and1 are reported in BOTH domains. Each rebuilds shared-corrected boxes, actual native-grid ROI features, original postfusion propagation geometry, temporal posterior, its own interval and final DP. The default HC.5 / Vid1 output is array-exact Current. These are two fixed cross-domain controls, not a sweep, new selected setting, or source-only proof for the historical target-exposed gamma selection. No method, budget, source model or parameters were changed.\n\nOriginal per-query Source/region/Current scalar scores are reused only after their old independent audits, exact input identities and array equality; only the two newly derived gamma arms are newly officially scored after the whole128 seal. Another process replays all arrays and checks input/prediction/compact/receipt SHA and every reused score/mean. All positive and negative signed results and short/long groups are retained. Bootstrap is query-equal,10000 paired-parent resamples, pointwise95% intervals; no independent-frame, familywise, equivalence, action-semantic or full-target claim follows a crossing-zero CI.\n\n## Pooling diagnostics\n\n'+json.dumps(diagnostics,indent=2)+'\n\n## Cost and stopped full-prefix preservation\n\n'+json.dumps(costs,indent=2)+'\n\nMedia, captions, GT coordinates, raw features/masks, weights and credentials remain private. No manuscript/Fig1/current registry or old frozen source/plan changed.\n'
 (V/ds/'ANALYSIS.md').write_text(text);atomic(V/ds/'all_paired_tails.json',tails);atomic(V/ds/'actual_costs.json',costs);atomic(V/ds/'pooling_diagnostics.json',diagnostics)
 atomic(U/ds/'root_export_receipt.json',dict(status='root_fixed128_all_signed_duration_tails_cost_exported',code_sha256=sha(Path(__file__)),audit_sha256=sha(U/ds/'reporting_receipt.json'),analysis_sha256=sha(V/ds/'ANALYSIS.md'),worker=identity(),new_GT=0,new_Source=0,new_expert=0))
 print(ds,'root exported',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=COUNTS,required=True);export(ap.parse_args().dataset)

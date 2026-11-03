"""Root complete cache-pilot analysis; no Current change or new GPU release."""
from matching_pilot import *
def main():
 plan=check_pilot();torch.set_num_threads(1);outputs={}
 for ds in COUNTS:
  ack=read(M/ds/'reporting_receipt.json');assert ack['report_sha256']==sha(V/ds/'report.json')and ack['distance_report_sha256']==sha(V/ds/'distance_report.json')
  for stage in ['predict','score','audit']:
   e=read(M/(ds+'_'+stage+'_exit.json'));assert e['actual_subprocess_wait']and e['exit_code']==0
  rr=read(M/ds/'metrics.json')['private_rows'];rep=read(V/ds/'report.json');distance=read(V/ds/'distance_report.json');tails={}
  from full_score import cluster
  for a,b in PAIRS:
   dt=np.array([100*(r['values'][a]['t']-r['values'][b]['t'])for r in rr]);dv=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rr]);tails[a+'-'+b]=dict(severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),severe_union_once=int(((dt< -5)|(dv< -5)).sum()),rescue_union_once=int(((dt>5)|(dv>5)).sum()),v_gain_gt1=int((dv>1).sum()),v_loss_lt_minus1=int((dv< -1).sum()),v_loss_le_minus10=int((dv<=-10).sum()),recall_paired_CI=cluster(rr,a,b,[('vIoU_at_03',100),('vIoU_at_05',100)]))
  costs={stage:read(M/(ds+'_'+stage+'_exit.json'))for stage in ['predict','score','audit']};empty=0;native=0
  for e in read(M/ds/'sealed.json')['files'].values():empty+=e['proof']['diagnostics']['empty_unobserved_masks']
  for row in panel(ds):
   e=read(M/ds/'sealed.json')['files'][row['id']]
   with np.load(M/e['path'])as z:native+=len(z['frame_ids'])-8
  atomic(V/ds/'all_paired_tails.json',tails);atomic(V/ds/'actual_cached_costs.json',dict(stages=costs,empty_unobserved_masks=empty,all_native_unobserved_frames=native,new_Source=0,new_expert=0,new_GPU=0,inherited_Source_expert_cost_nonzero=True,scope='Cached prediction wall includes raw-file reads/SHA, Source-feature label propagation, serialization and cold replay. Independent scoring/audit are additional experiment costs. Not uncached deployment or spatial-only time; the requested controlled actual timing is still pending.'))
  text=f'''# Root matching-propagation cache128 pilot: {ds}

128 queries/128 original confirm parents; all prediction arrays sealed before128new official matching tube checks, then independent complete-grid/raw/parent/prediction/receipt SHA, all arrays cold replay, frame-IoU distance groups and means audit. No new Source/expert/model/optimizer/GPU/media. Source-only selection fixed temp.05/topK10/context20 before target output scoring; no target tuning. Current, native Source and pure scalars reused by audited identity, with Source native interval explicitly separate.

Matching adapts the pinned official VideoWalk label-propagation kernel to the SAME Source postfusion token grid and SAME8teacher softmask/rawbox/physical positions used by Current. It is not the original learned encoder reproduction. The source validation already found the best adapter inferior to interpolation in BOTH directions; this negative result remains part of the interpretation, not hidden behind large Current-versus-adapter differences.

SourceMatched/PureMatched/Matching/Current use exactly the same Current interval. Pure/Matching preserve all8rawexpert anchor outputs; Current keeps its original soft candidate-selection behavior, which can choose E at anchors. No anchor clamp or method alteration was applied. The full observed-frame audit documents the difference. Therefore this controls evidence INPUT, feature layer and event interval, but does not fix Current anchor OUTPUT to the same boxes. The observed/unobserved IoUs below use the same actual GT-supported native sampled frames; observed support is not restricted to the predicted event.

|Configuration|tIoU %|vIoU %|R.3 %|R.5 %|Unobserved IoU %|
|---|---:|---:|---:|---:|---:|
'''
  for arm,m in rep['means'].items():text+=f"|{arm}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_at_03']:.6f}|{100*m['vIoU_at_05']:.6f}|{100*m['unobserved_actual_sIoU']:.6f}|\n"
  text+='\n|Comparison|Delta v pp [95% CI]|Delta unobserved IoU pp [95% CI]|v improve/harm/tie|Severe v / rescue union|\n|---|---|---|---|---|\n'
  def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:+.6f}, {x['CI95'][1]:+.6f}]"if x.get('mean')is not None else'unscorable'
  for name,p in rep['paired'].items():t=tails[name];text+=f"|{name}|{fmt(p['v'])}|{fmt(p['unobserved_actual_sIoU'])}|{p['v']['improved']}/{p['v']['harmed']}/{p['v']['unchanged']}|{t['severe_v']}/{t['rescue_union_once']}|\n"
  text+='\n## Short/long queries\n\n|Group|Queries|Comparison|Delta v pp [95% CI]|\n|---|---:|---|---|\n'
  for group,g in rep['timegroups'].items():
   for name,v in g['paired'].items():text+=f"|{group}|{g['queries']}|{name}|{fmt(v['v'])}|\n"
  text+='\n## Same annotated native-frame distance support\n\nFrame counts are support counts; queries/parents, not frames, define the paired bootstrap. Empty query/bin support remains unscorable, not0. Distances use the original8physical expert frame times. No old E-tail-hybrid distance curve is renamed pure.\n\n|Scope/bin|Scorable queries|Annotated frames|Pure IoU %|Matching IoU %|Current IoU %|Current-Matching pp [95% CI]|Current-Pure pp [95% CI]|\n|---|---:|---:|---:|---:|---|---|\n'
  for group,g in distance.items():
   st=g['statistics']['statistics'];n=g['statistics']['queries']
   if not st:text+=f"|{group}|0|0|unscorable|unscorable|unscorable|unscorable|unscorable|\n";continue
   text+=f"|{group}|{n}|{g['annotated_frames']}|{st['PureMatched']['mean']:.6f}|{st['Matching']['mean']:.6f}|{st['Current']['mean']:.6f}|{fmt(st['Current-Matching'])}|{fmt(st['Current-PureMatched'])}|\n"
  text+=f'''\nSelected matching outputs zero-area boxes on {empty}/{native} unobserved native frames; no E/Source fallback, skipped query or new threshold. The source-defined settings are retained despite the negative target comparison against pure interpolation. All signed negative/severe/rescue/R.3/R.5/shortlong/distance outcomes public. This is the old exposed128panel, not an independent unseen cohort, and is not the full1902/10303 experiment.

Root interpretation: Current exceeds this particular weak module adapter, but that does not establish superiority to a strong learned VideoWalk model or every matched-feature propagation strategy. Pure interpolation is a stronger comparison in these source and target pilots. Full-target acquisition is NOT launched from this report; it still requires a separate frozen numerical/execution plan, current-method/Source exact compatibility, single-use paid checkpoints, actual first2root bridges and sufficient-state retention. No cancellation of the user-authorized six-item scope or resumption of old queues is implied. Actual five-arm ready-input/end-to-end timing and real Source density remain pending. Cached wall is not deployment latency; inherited acquisition cost is nonzero. Queryequal10000paired-parent95%CIs are pointwise, not familywise or equivalence. No Source/E/features/teacher/Current hyperparameters, manuscript or frozen source changed.
'''
  (V/ds/'ANALYSIS.md').write_text(text);receipt=dict(status='root_complete_matching_cache128_signed_tails_recall_shortlong_distance_cost_exported',dataset=ds,queries=128,audit_sha256=sha(M/ds/'reporting_receipt.json'),analysis_sha256=sha(V/ds/'ANALYSIS.md'),code_sha256=sha(Path(__file__)),worker=identity(),Current_method_changed=False,new_Source=0,new_expert=0,new_GPU=0,full_target_GPU_released=False,whole_six_complete=False);atomic(M/ds/'root_receipt.json',receipt);atomic(V/ds/'root_receipt.json',receipt);outputs[ds]=receipt;print(ds,'root matching pilot analysis complete',flush=True)
 atomic(M/'root_completion.json',dict(status='both_cache_matching_pilots_scored_audited_root_exported_PENDING_publication',datasets=outputs,new_Source=0,new_expert=0,new_GPU=0,Current_method_changed=False,full_target_GPU_release=False,whole_six_complete=False))
if __name__=='__main__':main()

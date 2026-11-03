"""Root source-validation analysis; release only a cache-only target pilot.

Source working-state tests passed, but the adapter is inferior to interpolation.
This is a qualified module baseline, never a claim of a strong reproduction.
"""
from source_calibration import *
def main():
 check_calibration();torch.set_num_threads(1);ack=read(K/'audit_receipt.json');assert ack['report_sha256']==sha(P/'source_validation'/'report.json');assert ack['target_release']is False
 for stage in ['controls','score','audit']:
  e=read(K/(stage+'_exit.json'));assert e['actual_subprocess_wait']and e['exit_code']==0
 assert read(C/'acquire_actual_exit.json')['actual_tool_wait_exit_code']==0
 rep=read(P/'source_validation'/'report.json');rr=read(K/'private_rows.json');cohort=read(D/'source_validation_cohort_v1.json');contract=read(D/'matching_contract_preflight.json')
 assert contract['moving_patch_positive_control_exact']and contract['empty_mask_no_Source_or_E_fallback']and contract['all_eight_observation_boxes_exact']
 assert all(x['official_indices_exact']and x['maximum_weight_error']==0 for x in contract['kernel_cases'])
 costs={};seeds={};parents={};configs={}
 for target in COUNTS:
  ds=rep[target]['source_domain'];qq=[r for r in rr if r['target']==target];truth=SEL.labels(ds);metadata=cohort[target]['rows'];parents[target]=dict(queries=len(metadata),metadata_unique_video_ids=len({r['metadata']['original_video_id']for r in metadata}),video_sha_unique=len({r['metadata']['video_sha256']for r in metadata}),scorer_parent_clusters=len({str(truth[r['id']]['parent'])for r in metadata}));assert parents[target]['metadata_unique_video_ids']==36 and parents[target]['video_sha_unique']==36
  seal=read(C/target/'sealed.json');n=0;impossible=0;soft_low=0;ns=0.;qw=0.;empty=0;records=[]
  for key,e in seal['files'].items():
   assert sha(C/e['input_path'])==e['input_sha256'];v=torch.load(C/e['input_path'],map_location='cpu',weights_only=False,mmap=True);n+=len(v['frame_ids'])-8;impossible+=int(v['seeds'].max()<=.5);soft_low+=int((v['seeds'].reshape(8,-1).max(1).values<=.5).sum());ns+=e['Source_seconds'];qw+=e['query_elapsed_wall_seconds'];empty+=e['diagnostics'][rep[target]['selected_candidate']]['empty_unobserved_masks'];records.append(dict(query=len(records)+1,worker=e['worker'],Source_forward_seconds=e['Source_seconds'],query_elapsed_wall_seconds=e['query_elapsed_wall_seconds']))
  seeds[target]=dict(all_native_unobserved_frames=n,selected_candidate_empty_masks=empty,empty_percent=100*empty/n,queries_all8_seedmax_le_half=impossible,individual_seedmax_le_half=soft_low,total_seeds=288)
  costs[target]=dict(model_load=read(C/target/'model_load.json'),new_Source=36,new_expert=0,new_optimizer=0,Source_forward_sum_seconds=ns,query_elapsed_sum_seconds=qw,nested_not_additive=True,whole_worker_wall_seconds=None,scope='Source clock excludes Source-to-GPU transfer that precedes its start; query wall includes decode/parse/transfer/forward/CPUoffload/checkpoint/feature extraction/4cfg repeated propagation and IO, excludes subsequent root-bridge WAIT. Neither is uncached deployment latency. Model load once per checkpoint under the same PID; parser/whole-stage clocks not independently available remain null.',calls=records)
  configs[target]=rep[target]['config'];assert configs[target]==CONFIGS[0]
  assert rep[target]['paired']['Config0-UniformFeatures_Config0']['statistics']['delta']['CI95'][0]>0 and rep[target]['paired']['Config0-FixedTimePermutation_Config0']['statistics']['delta']['CI95'][0]>0
  assert rep[target]['paired']['Config0-PureExpert']['statistics']['delta']['CI95'][1]<0
 atomic(P/'source_validation'/'actual_source_costs.json',costs);atomic(P/'source_validation'/'mask_and_cluster_diagnostics.json',dict(seeds=seeds,parents=parents))
 fields=['Source','PureExpert']+[f'Config{i}'for i in range(4)]+[f'{c}_Config{i}'for c in CONTROL_NAMES for i in range(4)]
 with(P/'source_validation'/'all_source_query_scalars.tsv').open('w')as out:
  w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(['target_direction','query','source_parent_cluster','annotated_unobserved_frames',*[a+'_unobserved_IoU_percent'for a in fields],*[f'Config{i}_empty_unobserved_masks'for i in range(4)]])
  for target in COUNTS:
   qq=[r for r in rr if r['target']==target];pm={p:f'P{i:03d}'for i,p in enumerate(sorted({r['parent']for r in qq}),1)}
   for i,r in enumerate(qq,1):w.writerow([target,f'Q{i:03d}',pm[r['parent']],r['values']['Source']['annotated_unobserved_frames'],*[r['values'][a]['unobserved_IoU_percent']for a in fields],*[r['empty_counts'][f'Config{j}']for j in range(4)]])
 text='''# Root source-domain validation: functioning but weak propagation adapter

The 72 actually new Source extractions and all source controls/scores/independent audits are complete. Source queries, not target scores, select temperature0.05/topK10/context20 in BOTH directions. This is the original fixed4config candidate0, not an additional sweep or a target-selected choice. Source-validation36 inputs per direction were metadata/SHA selected; they are not the historical source-training36 calibration roster. The opposite transfer direction has exposed these videos before, so this is not unknown validation.

The official pinned VideoWalk affinity kernel agrees exactly; actual feature, control, saved state and source-only score checks pass. Uniform-feature and fixed-time-permutation interventions harm source unobserved IoU in both directions, so this is not a nonfunctional branch. However, the best candidate is inferior to pure interpolation on BOTH source cohorts. It must be called a Source-feature VideoWalk module adapter, not a strong baseline or reproduction of the paper's learned encoder. Raw postfusion feature resolution and diffuse teacher softmask readout are potential interface limitations, not established unique causes.

|Target direction / source validation|Source unobserved IoU %|Pure interpolation %|Selected adapter %|Adapter minus interpolation pp [95% CI]|Adapter minus time-permutation pp [95% CI]|
|---|---:|---:|---:|---|---|
'''
 for target,r in rep.items():
  f=lambda a:r['means'][a]['statistics']['value']['mean'];d=r['paired']['Config0-PureExpert']['statistics']['delta'];t=r['paired']['Config0-FixedTimePermutation_Config0']['statistics']['delta'];text+=f"|{target} / {r['source_domain']}|{f('Source'):.6f}|{f('PureExpert'):.6f}|{f('Config0'):.6f}|{d['mean']:+.6f} [{d['CI95'][0]:+.6f},{d['CI95'][1]:+.6f}]|{t['mean']:+.6f} [{t['CI95'][0]:+.6f},{t['CI95'][1]:+.6f}]|\n"
 text+='\n## Coverage and cohort definitions\n\n'
 for target in COUNTS:
  z=seeds[target];p=parents[target];text+=f"- Target {target}: 36 queries /36 unique metadata video IDs /36 unique video byte hashes, but **{p['scorer_parent_clusters']} scorer parent clusters**. Bootstrap uses that actual scorer parent grouping. Selected adapter returns empty masks on {z['selected_candidate_empty_masks']}/{z['all_native_unobserved_frames']} native unobserved frames ({z['empty_percent']:.3f}%). All8seed maxima<=.5 in {z['queries_all8_seedmax_le_half']} queries; {z['individual_seedmax_le_half']}/288 individual coarse seed masks never exceed.5. Empty outputs remain zero-area boxes, no E/Source fallback or skipped queries.\n"
 text+='''
The HC-source cohort contains36 different video byte hashes but34 scorer parent clusters; earlier wording claiming36 independent scorer parents for each source direction was too broad. No query was replaced after scores. All36 remain and the paired bootstrap already uses34 clusters. The Vid-source cohort uses36 clusters. Query-equal statistics/10000pointwise parent CIs are descriptive source validation, not familywise or causal action-semantics evidence.

Root decision: accept exact implementation functionality ONLY for the already authorized cache-only target128 pilot, with the source-selected configuration fixed. Preserve the negative source comparison and explicitly qualify baseline strength. **Do not release a new full-target GPU collection from this receipt.** The target pilot must verify its complete-grid interface, same physical eight observations and common Current interval, then whole128seal -> independent official score -> separate cold arrays/SHA/means/distance audit -> root analysis. Target outcomes may not alter temperature/topK/threshold/Current method. No claim that expert anchor OUTPUT is fixed in Current: the read-only observed-frame audit found Current can choose E at anchors, while interpolation/matching preserve their original expert boxes. This policy difference remains disclosed and unchanged, per the latest no-method-change instruction.

All4 configurations, Source/pure references, all8 feature-control variants, anonymous per-query signed source scores and empty counts, costs, audits and engineering preservation are retained. OriginalSource/checkpoint/eightteacher masks/model backend unchanged. No new teacher, optimizer, target prediction or annotation-driven input selection in source acquisition. Original raw72 remain private and preserved. Source model loads, forward/query clocks are nested and are not summed or advertised as deployment timing; unavailable whole-worker clock remains null.
'''
 (P/'source_validation'/'ANALYSIS.md').write_text(text)
 root=dict(status='source_functioning_verified_negative_source_strength_disclosed_cache_pilot_only',source_functioning_accepted=True,no_target_scores_used=True,selected_configs=configs,source_audit_sha256=sha(K/'audit_receipt.json'),source_report_sha256=sha(P/'source_validation'/'report.json'),source_root_analysis_sha256=sha(P/'source_validation'/'ANALYSIS.md'),baseline_strength='inferior to pure interpolation on both source cohorts; do not call strong or original learned-encoder reproduction',release_scope='cache-only target128 pilot; no new full-target GPU acquisition',full_new_GPU_release=False,Current_method_changed=False,code_sha256=sha(Path(__file__)),worker=identity());atomic(K/'root_acceptance.json',root);atomic(P/'source_validation'/'root_receipt.json',root);print(json.dumps(root),flush=True)
if __name__=='__main__':main()

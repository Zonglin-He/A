"""Read-only scalar/cost closeout of the user-reduced reviewer scope."""
from subset_v1 import *

def fmt(x):return f"{x['mean']:+.6f} [{x['CI95'][0]:+.6f}, {x['CI95'][1]:+.6f}]"

if __name__=='__main__':
 validate_subset();assert read(U/'state.json')['status']=='both_fixed128_domains_scored_audited_pending_root_analysis_publication'
 exits={};bound={};reports={}
 for ds in ['hc2','vidstg']:
  for stage in ['predict','score','audit']:
   p=U/ds/f'{stage}_exit.json';e=read(p);assert e['exit_code']==0 and e['actual_subprocess_wait'];exits[f'{ds}/{stage}']=e;bound[str(p.relative_to(R))]=sha(p)
  for name in ['reporting_receipt.json','root_export_receipt.json']:
   p=U/ds/name;assert p.exists();bound[str(p.relative_to(R))]=sha(p)
  reports[ds]=read(V/ds/'report.json')
 preserved=D/'full_to_subset_preserved_20261103';stopped=read(preserved/'owned_exit.json');assert stopped['no_paid_uncommitted_attempt_at_stop']
 for name in ['parent','child']:
  w=stopped[name];p=Path(f'/proc/{w["pid"]}/stat')
  if p.exists():
   z=p.read_text().rsplit(')',1)[1].split();assert z[0]=='Z' or int(z[19])!=w['start_ticks']
 prefix=sorted((OLD_FULL/'hc2/receipts').glob('*.json'),key=lambda p:p.stat().st_mtime)
 with(V/'stopped_full_prefix_actual_costs.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','origin','Source_forward_seconds_nested','query_elapsed_wall_seconds_nested','prediction_sha256','compact_sha256','receipt_sha256','GPU_pid','GPU_startticks'])
  for i,p in enumerate(prefix,1):
   e=read(p);w.writerow([f'Q{i:05}',e['raw_origin']['kind'],e['raw_origin'].get('Source_seconds'),e['query_elapsed_wall_seconds'],e['sha256'],e['compact_sha256'],sha(p),e['worker']['pid'],e['worker']['startticks']])
 assert len(prefix)==289
 with(V/'all256_cached_costs.tsv').open('w')as f:
  w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['dataset','query','origin','cached_elapsed_wall_seconds','prediction_sha256','compact_sha256','worker_pid','worker_startticks'])
  for ds in ['hc2','vidstg']:
   for i,(key,e)in enumerate(read(U/ds/'sealed.json')['files'].items(),1):w.writerow([ds,f'Q{i:05}',e['origin']['kind'],e['elapsed_wall_seconds'],e['sha256'],e['compact_sha256'],e['worker']['pid'],e['worker']['startticks']])
 public=dict(status='all_fixed128_CPU_controls_scored_audited_root_exported_pending_publication',supersedes='remaining full1902/10303 regional and uniform-gamma acquisition only; previous complete full CPU statistics stay complete',
  queries_per_domain=128,distinct_parents_per_domain=128,selection='Unchanged prior matched-pipeline panel. Scope reduction followed an explicit user request after full HC acquisition started. No new score-based selection. The previously published192-parent region controls remain alongside these nested128 controls; smaller-panel significance must not replace the primary192 null evidence.',
  full_stopped=dict(committed_HC_queries=289,new_Source=260,retained_raw=29,new_expert=0,all_outputs_preserved=True,uncommitted_paid_attempts=0,full_prefix_accuracy_scored=False,dispatcher_external_wait_code=None,predictor_external_wait_code=None),
  subset_new_Source=0,subset_new_expert=0,subset_new_model=0,subset_new_optimizer=0,subset_GPU=False,new_official_checks=512,reused_old_audited_metrics=True,all_six_independent_stage_waits_zero=True,
  protocol_fingerprint=read(U/'plan_v1.json')['fingerprint'],root_code_sha256=sha(Path(__file__)),private_completion_binding_SHA256=digest(bound),other_old_paused_scopes_resumed=False,current_method_or_paper_changed=False)
 atomic(V/'scope_and_preservation.json',public);atomic(U/'root_scientific_completion.json',dict(**public,bound_receipt_sha256=bound,actual_stage_exits=exits,worker=identity()))
 text='# Reviewer controls after requested subset reduction\n\nAll previously completed full CPU reviewer experiments remain complete and unchanged. The user cancelled the remaining full-target acquisition and requested subsets instead. Both fixed128-query/128-parent domains now have complete five-region and uniform-gamma controls, independent new gamma scores, array/input/score/mean audits and root signed/tail/short-long/cost reports. There are no new Source/expert/model/optimizer calls or GPU use in the subset successor.\n\n## Fixed cross-domain gamma\n\n|Domain|Gamma|tIoU %|vIoU %|Delta t versus current [95% CI]|Delta v versus current [95% CI]|\n|---|---:|---:|---:|---|---|\n'
 for ds,r in reports.items():
  for arm,gamma in core.GAMMAS:
   m=r['means'][arm];d=r['paired'][arm+'-Current'];text+=f"|{ds}|{gamma}|{100*m['t']:.6f}|{100*m['v']:.6f}|{fmt(d['t'])}|{fmt(d['v'])}|\n"
 text+='\nThe current default is gamma.5 on HC and gamma1 on Vid; matching-default rows are identity controls, not a statistical equivalence finding. Both fixed values are reported without adopting a new default. Existing historical gamma selection used exposed target development data; this subset does not retroactively make that selection source-only or an unknown-target validation.\n\n## Regional temporal evidence at identical E space\n\n|Domain|Corrected minus comparator|Delta t pp [95% CI]|Delta v pp [95% CI]|\n|---|---|---|---|\n'
 for ds,r in reports.items():
  for arm in ['SourceROI','GlobalAppearance','BackgroundAppearance','ShuffledBoxes']:
   d=r['paired']['CorrectedROI-'+arm];text+=f"|{ds}|{arm}|{fmt(d['t'])}|{fmt(d['v'])}|\n"
 text+='\nThe previously published192-parent region analyses are retained. These128 cases are the original matched-pipeline panel nested within those192, chosen through the user-requested scope change, not an independent replication or a replacement selected by significance. A pointwise positive subset contrast does not establish universal object-local necessity; nulls are not equivalence proofs. All methods, eight observations, kappa0/beta1/context1.5 and the last trajectory DP stay fixed.\n\nThe discontinued full run retains all289 committed HC predictions and their hashes/costs,260 actual new Source calls and29 raw-cache reuses, with no uncommitted paid attempt at stop. It is not scored or presented as full-domain accuracy. Original acquisition costs remain nonzero, partial-run whole-worker clocks unavailable/null, nested Source/query/IO costs not additive or deployment latency.\n\nAll signed gains/losses, severe t/v union-once/rescues, short/long and recall/spatial/boundary metrics, raw scalar costs and all numerical nulls/negatives are in the domain reports/TSVs. CIs are10000 paired-parent percentile pointwise95%, not familywise. Public outputs contain no captions/media/GT coordinates/features/masks/weights/credentials. Old paused queues, production, paper/Fig1 and every old frozen code/plan remain unchanged. Final remote publication verification is a separate completion gate.\n'
 (V/'ANALYSIS.md').write_text(text);print('root subset final scientific closeout complete',flush=True)

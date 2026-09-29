"""Full locked evaluation report, never a method selection step."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha,status
OUT=ROOT/'artifacts/tastvg_full_b1_v1'

def run():
 assert read(OUT/'SCORE_STATUS.json')['status']=='completed';s=read(OUT/'SUMMARY.json');excess=read(OUT/'CORRUPTION_EXCESS.json');audit=read(OUT/'AUDIT.json');cohort=read(OUT/'COHORT.json')
 from scripts.audit_tastvg_full_b1_public_v1 import run as audit_public
 check=audit_public(OUT);write(OUT/'PUBLIC_AUDIT.json',check)
 resources=[read(f) for f in (OUT/'allocations').glob('*.json')];cost=dict(total_GPU_process_seconds=sum(r['seconds'] for r in resources),by_stage={stage:sum(r['seconds'] for r in resources if r['stage']==stage) for stage in sorted({r['stage'] for r in resources})},failed_attempts=sum(r['status']=='failed' for r in resources));write(OUT/'RESOURCES.json',cost)
 def v(group,subset,key):return s[group][subset][key]['mean']*100
 def ci(x):return f"{x['mean']*100:+.6f} [{x['ci95'][0]*100:+.6f}, {x['ci95'][1]*100:+.6f}]"
 lines=['# B1: full fixed-method official VidSTG test evaluation','', 'All9411queries/670retained official-test sources,3prelocked source-hash orders,16conditions;451728online arrivals. Excludes62current-route development sources/892queries. Historical project exposure is retained: this is not globally untouched fresh confirmation. Method remains frozen from97bd8ab.','', '## Main corruption panel','', 'Average queries within source/condition/order, then15conditions, then3orders, then source macro. Metrics in percent; paired differences in percentage points. Source bootstrap10000,seed20260929; orders are repeated observations of the same sources.','', '| Arm | sIoU % | tIoU % | vIoU % | Δv vsFrozen pp [95% source CI] | Sources with >5pp mean v harm |','|---|---:|---:|---:|---:|---:|']
 for arm in ['Frozen','Fast-only','Slow-only','Final']:
  st=s['corruption']['all'][arm+'_minus_Frozen_v'];lines.append('| '+arm+' | '+' | '.join(f"{v('corruption','all',arm+'_'+m):.6f}" for m in ['s','t','v'])+f" | {ci(st)} | {st['source_harm_gt5pp']} / {st['sources']} |")
 lines+=['','## Future nonexpert transfer','', '| Subset | Sources | Final−Fast Δs pp [CI] | Final−Fast Δv pp [CI] | >5pp mean v harm sources |','|---|---:|---:|---:|---:|']
 for sub in ['nonexpert','first_source_nonexpert']:
  a=s['corruption'][sub]['Final_minus_Fast_s'];b=s['corruption'][sub]['Final_minus_Fast_v'];lines.append(f"| {sub} | {b['sources']} | {ci(a)} | {ci(b)} | {b['source_harm_gt5pp']} |")
 lines+=['','All-nonexpert includes queries preceded by other queries of the same source. The first-source-query subset separately reads inherited state from earlier sources. Eligibility and denominators are retained in source rows; missing first-source nonexpert observations are not zero-filled.','', '## Condition/severity breakdown','', '| Condition | Frozen v % | Final v % | Final−Frozen Δv pp [CI] | Future Final−Fast Δv pp [CI] |','|---|---:|---:|---:|---:|']
 for group in cohort['conditions']:
  lines.append(f"| {group} | {v(group,'all','Frozen_v'):.6f} | {v(group,'all','Final_v'):.6f} | {ci(s[group]['all']['Final_minus_Frozen_v'])} | {ci(s[group]['nonexpert']['Final_minus_Fast_v'])} |")
 lines+=['','## Clean and corruption excess','', 'Clean is a matched control; corruption-specific recovery is not assumed. Excess paired Final−Frozen gain (corruption minus clean), source-macro:', '', '| Readout | Δs pp [CI] | Δt pp [CI] | Δv pp [CI] |','|---|---:|---:|---:|']
 for sub in ['all','nonexpert','first_source_nonexpert']:lines.append('| '+sub+' | '+' | '.join(ci(excess[sub][m]) for m in ['s','t','v'])+' |')
 lines+=['','## Online schedule and cost','', f"Every fourth query arrival receives both experts in Final:2353/9411each order. Orders preserve source blocks and fixed query-hash order. Reset only at each of48independent order/condition streams; no16-arrival resets. Fast uses temporal only, Slow spatial only, Final both; equal availability is not equal total specialist cost. Actual GPU-process time including failures:{cost['total_GPU_process_seconds']:.3f}s; cached timings are not deployment latency.",'', f"Validation:{audit['links']}state links,{audit['SGD_coordinates']}independent SGD coordinates,{audit['KL_updates']}KL reconstructions,{audit['dual_metric_calls']}dual metric calls. All451728predictions sealed before GT scoring. Public source aggregation audit:{check['checks']}checks. All clean/condition/order outcomes and source/cell harm tails retained; no favorable schedule selection, gate or hyperparameter change.",'', 'PhaseA Random/Off/PL results remain separate development ablations. In particular, their failure to establish on-policy superiority is not erased by this full run. Production CURRENT_METHOD is unchanged. Secondary cross-domain stress test is a later, separately specified scope.','']
 (OUT/'REPORT.md').write_text('\n'.join(lines))
 write(OUT/'COMPLETION.json',dict(status='completed',cells=451728,sources=670,queries=9411,frozen_method_unchanged=True,publication='pending_remote_verification',summary_sha256=sha(OUT/'SUMMARY.json'),time=time.time()))
 status(OUT/'STATUS.json',dict(status='completed_pending_publication',time=time.time()));print('Full evaluation report completed',flush=True)
if __name__=='__main__':run()

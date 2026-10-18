"""Independent saved-scalar aggregation and compact anonymous export."""
import sys,csv,json,gzip,collections,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_evidence_strata_v1 import D,P,read,put,sha,mean,FEATURES,CATS,METRICS
import numpy as np

def report():
 full=read(P/'FULL_STRATA.json');tem=read(P/'TEMPORAL64.json');sp=read(P/'SPATIAL_SCALE_GATE.json');private=read(D/'FEATURES_PRIVATE.json');checks=0
 for ds in ['hc2','vidstg']:
  path=ROOT/'results/stvg_opd_p1_complete/2026-10-09/anonymous_rows'/ds/'spatial_opd/ROWS.jsonl.gz';rr=[json.loads(x)for x in gzip.open(path,'rt')];fby={f['query_ordinal']:f for f in private['rows'][ds]}
  for scope,rs in [('all',rr),('excluding_tuning_parents',[r for r in rr if fby[r['query_ordinal']]['primary']])]:
   s=full['groups'][ds][scope]['overall'];assert len(rs)==s['arrivals'];assert len({r['query_ordinal']for r in rs})==s['queries']
   for k in METRICS:
    a=[r[k]for r in rs if r.get(k)is not None];v=sum(a)/len(a)if a else None;assert v is None and s['query_macro'][k]is None or abs(v-s['query_macro'][k])<1e-10;checks+=1
    parents=collections.defaultdict(list)
    for r in rs:
     if r.get(k)is not None:parents[r['source_id']].append(r[k])
    v=mean([sum(g)/len(g)for g in parents.values()]);assert v is None and s['parent_macro'][k]is None or abs(v-s['parent_macro'][k])<1e-10;checks+=1
   # Exact Source/Before/After metric differences, independently averaged.
   for k,a,b in [('delta_current_v','After_v','Before_v'),('delta_inherited_v','Before_v','Frozen_v'),('delta_total_v','After_v','Frozen_v')]:
    assert abs(mean([r[a]-r[b]for r in rs])-s['query_macro'][k])<1e-12;checks+=len(rs)
 old=read(ROOT/'results/stvg_interval_decision/2026-10-18/cached64/RESULTS.json')
 for ds in ['hc2','vidstg']:
  for support in ['offset0','offset1','fused']:
   for a in ['Native','projection','native_MBR']:
    for k in ['v','t','s']:
     assert abs(tem['groups'][ds]['Frozen'][support]['overall'][a]['query_macro'][k]-old['groups'][ds][support][a][k])<1e-12;checks+=1
 # Anonymous per-query CSV, retaining every signed result without leaking exact GT features.
 public=read(P/'QUERY_BINS_AND_METRICS.json')['rows'];outfiles=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in public if r['dataset']==ds]
  for start in range(0,len(rr),1000):
   dest=P/'anonymous_query_bins'/f'{ds}_{start//1000+1:02}.csv';dest.parent.mkdir(exist_ok=True);outfiles.append(str(dest.relative_to(ROOT)));keys=['query_alias','parent_alias','primary']+FEATURES+[k for k in CATS if k!='arrival_exposure']+METRICS+[f'{o}_{k}'for o in ['order1','order2','order3']for k in ['arrival_exposure','delta_total_v','delta_current_v','delta_inherited_v']]
   with dest.open('w',newline='')as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
    for r in rr[start:start+1000]:
     x={k:r[k]for k in ['query_alias','parent_alias','primary']+[k for k in CATS if k!='arrival_exposure']};x.update(r['bins']);x.update(r['metrics'])
     for o in r['per_order']:
      for k in ['arrival_exposure','delta_total_v','delta_current_v','delta_inherited_v']:x[o['order']+'_'+k]=o[k]
     w.writerow({k:format(v,'.12g')if isinstance(v,float)else v for k,v in x.items()})
 # Coverage ties are not four distinct groups; report constant coverage explicitly.
 coverage={ds:dict(queries=len(fs),complete=sum(abs(f['annotation_coverage']-1)<1e-12 for f in fs),partial=sum(f['annotation_coverage']<1-1e-12 for f in fs),greater_than_one=sum(f['annotation_coverage']>1+1e-12 for f in fs))for ds,fs in private['rows'].items()}
 put(P/'VERIFICATION.json',dict(status='pass',independent_saved_row_and_paired_aggregate_checks=checks,temporal_official_checks=tem['independent_official_metric_checks'],online_rollout_checks=sp['checks'],all13785_Source_input_hashes_verified=True,annotation_coverage=coverage,full_raw_steps_retired=True,zero_new_model_calls=True,conditional_spatial_triggered=sp['passed'],registries_unchanged=True))
 lines=['# Full cached evidence stratification and fixed readout controls','', 'Original clean official cohort: HC3482/237 parents and Vid10303/732, three original online orders. Historically exposed, not a fresh confirmation. Full P1 uses Native timing; restored P7-A additionally uses projection. Saved original predictions and metrics are unchanged. No new model/Teacher/parameter fitting in this audit.','', '## Full online decomposition','', 'Values in percentage points; query macro averages the three locked orders. Parent macro gives each video parent equal weight. Current=After−Before; inherited=Before−Frozen. This arithmetic does not identify a causal transfer mechanism.','', '| Cohort | Domain | Queries / parents | Total query | Current query | Inherited query | Total parent | Current parent | Inherited parent |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for scope in ['all','excluding_tuning_parents']:
  for ds in ['hc2','vidstg']:
   s=full['groups'][ds][scope]['overall'];q=s['query_macro'];p=s['parent_macro'];lines.append(f'| {scope} | {ds} | {s["queries"]} / {s["parents"]} | '+ ' | '.join(f'{100*m[k]:+.6f}'for m in [q,p]for k in ['delta_total_v','delta_current_v','delta_inherited_v'])+' |')
 lines+=['','The previously quoted +2.409/+3.329 pp values are parent macro, not official query macro. Query weighting changes these figures; neither is substituted for the other.', '', '## Corrected feature definitions','', 'Event length and start are measured against the actual physical model-input frame-grid span [first,last+1). This may differ from the original full parent-video duration. Annotation coverage is annotated frame count / physical event frame count; input annotation fraction is a separate statistic. HC physical feature support uses metadata start plus bbox count; the legacy evaluator endpoint remains unchanged. Current binding gives complete within-event annotation for every query in both domains: missing-box risk is not observed in this official binding.','', 'Motion uses actual fid gaps / fps. Report image diagonals/sec and center displacement per geometric-mean target box size/sec, using adjacent pair mean target size. This is apparent 2D motion including camera motion, not recovered object velocity. Same pooled query quartile boundaries apply to both datasets; Vid contributes more queries to these boundaries. Constant annotation coverage has coincident boundaries, so its Q4 label means the sole constant group, not a high-versus-low contrast. Lexical relation/sequence markers and parser target category are diagnostic proxies, not verified identities or semantic annotations.','', 'First/repeated is assigned separately within each original order and actual parent; a query may be first in one order and repeated in another, so unique-query counts across these strata need not add to N. Exposure is descriptive, not a randomized intervention.', '', 'Public query files contain only bin labels, anonymous identifiers and sealed metrics. Exact per-query GT duration/start/coordinates and raw query text stay private. All strata, sample counts, gross harm thresholds, parent/query means and overall parent-cluster CIs are in the public FULL_hc2.json and FULL_vidstg.json. Per-order signed values remain in CSV; no stratum is used for online selection.','', '## Fixed temporal decisions on the exact64 cache','', 'Only this cache retains complete logits/actions. Full13785 Source inputs lack them; this is NOT a full-cohort MBR experiment. The same sealed decisions are crossed with Source, inherited Before and actual original online OPD After geometry. Observations, spatial states and parameters do not change. Final outputs preserve the original physical envelope of two offsets; per-offset outputs are also reported. No new history is simulated.','', '| Domain | Fixed geometry | Readout | vIoU % | tIoU % | Δv Native pp | added annotated frames/query | IoU on added known frames |','|---|---|---|---:|---:|---:|---:|---:|']
 for ds in ['hc2','vidstg']:
  for geo in ['Frozen','Before','After']:
   for a in ['Native','projection','native_MBR']:
    v=tem['groups'][ds][geo]['fused']['overall'][a]['query_macro'];s='unknown'if v['added_known_s']is None else f'{100*v["added_known_s"]:.3f}';lines.append(f'| {ds} | {geo} | {a} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {100*v["delta_v_vs_Native"]:+.6f} | {v["added_annotated_frames"]:.3f} | {s} |')
 lines+=['','The public TEMPORAL_hc2.json and TEMPORAL_vidstg.json contain every signed query result, event-length/start common-bin strata, lost/gained annotated support and added support without GT. Added-frame IoU is a macro over queries with known added frames, not zero-filled; coverage counts and unknowns are separate. Fixed geometry means sIoU is invariant to the temporal readout. MBR maximizes expected physical tIoU under the model distribution, not true GT overlap. Projection-length controls from the previous report are diagnostics still using Actionness length; equal per-offset length does not guarantee equal fused-envelope length.','', '## Exploration-scale audit and prelocked conditional gate','', f'Actual original ONLINE32 rollout upper/lower RMS quartile ratio: {sp["actual_RMS_upper_lower_quartile_ratio"]:.6f}. Full-cohort exploration scale is a Source-box delta-method PROXY; full original raw step traces were legally retired, not newly lost. Actual32 traces retain real inherited state and steps.']
 for ds in ['hc2','vidstg']:
  g=sp['groups'][ds];lines+=['',f'{ds}: conditions `{json.dumps(g["conditions"])}`.']
  if g['high']and g['low']:
   h,l=g['high']['query_macro'],g['low']['query_macro'];lines.append(f'Primary high/low proxy quartile current-query Δv: {100*h["delta_current_v"]:+.6f} / {100*l["delta_current_v"]:+.6f} pp; high unobserved GT-frame Δs: {100*h["unobserved_iou_delta"]:+.6f} pp. Parent/query means and CIs are retained; these means do not condition on successful updates.')
 if not sp['passed']:lines+=['','**Gate not passed: no target-relative-coordinate OPD was fitted.** Scale imbalance alone is insufficient to establish damage. The prelocked required conjunction did not hold. This stops this conditional branch, not all coordinate-normalized policies. Do not interpret domain/small-object correlations as causal evidence or replace the positive original full online result.']
 else:lines+=['','Gate passed; one matched online32 relative-coordinate variant is required before this task can close. No result for that variant is implied by this cache audit.']
 lines+=['','Flat reward, ESS, actual perturbation IoU/log-size, per-round reward-up/unobserved-actual-decoder-down and current-query effects are retained in the public SPATIAL_hc2.json and SPATIAL_vidstg.json. Per-round conflicts are intermediate path diagnostics, not final failures. Full strata do not recover retired raw samples; Source proxies are not mislabeled as actual inherited trajectories.','', 'No production registry promotion, new Actionness-free official claim, critic, diffusion, Qwen, semantic identity label, parameter sweep or official full restart. GT is used only offline. Existing negative/positive results remain unchanged.']
 lines+=['','CPU reader recovery retained three interface fixes: the existing receipt filename suffix, accepting the sealed NumPy array type, and converting cxcywh to the stored float64 xyxy chart for exact comparison. Original failures and pre-fix implementations remain in private recovery. No predictions, Student steps, scientific parameters or sealed metrics were changed; no model calls were added.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 # Small self-contained views split by domain to keep publication reviewable.
 files=['scripts/analyze_stvg_evidence_strata_v1.py','scripts/analyze_stvg_evidence_readouts_v1.py','scripts/report_stvg_evidence_strata_v1.py','protocols/stvg_evidence_strata_v1.md',str((P/'REPORT.md').relative_to(ROOT)),str((P/'VERIFICATION.json').relative_to(ROOT))]+outfiles
 for ds in ['hc2','vidstg']:
  p=P/f'FULL_{ds}.json';put(p,dict(groups=full['groups'][ds],common_bins=full['common_bins'],distributions=full['dataset_distributions'][ds],query_type_counts=full['query_type_counts'][ds]));files.append(str(p.relative_to(ROOT)))
  p=P/f'TEMPORAL_{ds}.json';put(p,dict(groups=tem['groups'][ds],rows=[r for r in tem['rows']if r['dataset']==ds]));files.append(str(p.relative_to(ROOT)))
  p=P/f'SPATIAL_{ds}.json';put(p,dict(gate=sp['groups'][ds],association=sp['within_domain_associations'][ds],rows=[r for r in sp['rows']if r['dataset']==ds],global_actual_quartile_ratio=sp['actual_RMS_upper_lower_quartile_ratio'],passed=sp['passed']));files.append(str(p.relative_to(ROOT)))
 put(D/'PUBLIC_FILES.json',dict(files=files));put(D/'CPU_COMPLETION.json',dict(status='complete',count=13785,temporal_count=64,online_trace_count=32,GT_read=True,new_model_calls=0,conditional_spatial_triggered=sp['passed']));print('REPORT_COMPLETE',checks,'gate',sp['passed'],flush=True)
if __name__=='__main__':report()

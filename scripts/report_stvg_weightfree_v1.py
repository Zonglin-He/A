"""Report and explicitly export complete anonymous development results."""
import sys,json,math,shutil,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_spatial_v1';C=B/'consensus002'
def alias(cid):return ('HC' if cid.startswith('hc2') else 'V')+f'{int(cid.rsplit("_",1)[1])+1:02d}'
def main():
 out=ROOT/'results/stvg_weightfree/2026-10-15';out.mkdir(parents=True,exist_ok=True)
 paths=[]
 def save(name,obj):
  p=out/name;write(p,obj);paths.append(p)
 datasets={};metric_error=0.;audit_count=0
 for phase,d in [('initial',B),('controlled_diagnostics',C)]:
  rows=read(d/'POSTSEAL_ROWS.json')['rows'];summ=read(d/'POSTSEAL_SUMMARY.json');bar=read(d/'GLOBAL_PREDICTION_BARRIER.json')
  for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
  assert not bar['GT_read'] and read(d/'GT_EXPOSURE.json')['seal_sha256']==sha(d/'GLOBAL_PREDICTION_BARRIER.json')
  for s in summ['summary']:
   vals=[r['metrics'][s['metric']] for r in rows if r['dataset']==s['dataset'] and r['arm']==s['arm'] and r['metrics'][s['metric']] is not None]
   assert len(vals)==s['count'];assert (s['mean'] is None and not vals) or abs(math.fsum(vals)/len(vals)-s['mean'])<1e-14;audit_count+=1
  public=[]
  for r in rows:
   e=r['metric_audit']['official_independent_max_abs_error'];metric_error=max(metric_error,e)
   public.append(dict(case=alias(r['case_id']),dataset=r['dataset'],arm=r['arm'],metrics=r['metrics'],support_counts={k:v for k,v in r['support'].items() if k.endswith('_count') or k=='source_grid_positions'},official_independent_max_abs_error=e))
  save(phase+'_all_results.json',dict(status='complete',historically_exposed_development=True,rows=public))
  save(phase+'_summary.json',summ)
  save(phase+'_root.json',read(d/'ROOT_READBACK.json'))
  datasets[phase]=summ
 execution=dict(status='complete',historical_cases=8,windows=32,new_detector_or_segmenter_weights=0,new_GT_oracle_fits=0,phases={})
 for phase,d in [('initial',B),('controlled_diagnostics',C)]:
  tb=read(d/'TEACHER_GLOBAL_BARRIER.json');sb=read(d/'STUDENT_GLOBAL_BARRIER.json');tr=[]
  for f in tb['files']:
   assert sha(ROOT/f['path'])==f['sha256'];tr.append(read(ROOT/f['path']))
  execution['phases'][phase]=dict(teacher_calls=len(tr),valid_output_count=sum(t['parsed']['status']=='valid' for t in tr),truncated_output_count=sum(t['output_truncated_at_cap'] for t in tr),teacher_generate_seconds=math.fsum(t['model_forward_generate_seconds'] for t in tr),teacher_request_seconds=math.fsum(t['end_to_end_request_seconds'] for t in tr),teacher_peak_allocated_bytes=max(t['GPU_peak_bytes'] for t in tr),input_tokens=sum(t['input_tokens'] for t in tr),visual_tokens=sum(t['visual_tokens'] for t in tr),output_tokens=sum(t['output_tokens'] for t in tr),new_fit_count=sum(f['actual_fit_count'] for f in sb['files']),new_full_redecode_count=sum(f['full_redecode_count'] for f in sb['files']),student_worker_wall_seconds=sb['wall_seconds'])
 execution['cached_window_candidate_seconds']=read(B/'CANDIDATE_INPUT_BARRIER.json')['proposal_seconds']
 execution['cost_scope']='Cached local windows; candidate time excludes RGB loading/export. R1 request time excludes model loading/hash verification. Student worker includes its own source loads/captures. Prior Temporal/source/video costs excluded. Not an end-to-end full-video comparison.'
 save('execution.json',execution)
 refs=read(C/'REFERENCE_LOCK.json')['records'];support=[]
 for r in refs:
  support.append(dict(case=alias(r['case_id']),anchors={v:len(r[v]['anchors']) for v in ['consensus8','nested8','consensus_nested8']},nested_rejections={v:len(r[v].get('nested_rejections',[])) for v in ['nested8','consensus_nested8']}))
 save('diagnostic_decisions.json',dict(status='complete',vote_counts=read(C/'REFERENCE_LOCK.json')['counts'],support=support,diagnostic_design_after_initial_development_scores=True,GT_used_for_online_decisions=False))
 save('metric_audit.json',dict(status='pass',summary_fields_recomputed=audit_count,official_independent_max_abs_error=metric_error,all_144_phase_case_arm_rows_preserved=True,independent_backward_replay=False,baseline_anchor_partition_not_comparable=True))
 table=[]
 names=[('F','Frozen Source','initial'),('T','Temporal QA','initial'),('T_validated_loss_weighted_iou','Temporal + original YOLO/SAM2/R1 OPD','initial'),('T_fast4','Temporal + weight-free 4','initial'),('T_diverse8','Temporal + weight-free 8','initial'),('T_consensus8','Temporal + 8 + three-order majority','controlled_diagnostics'),('T_nested8','Temporal + 8 + nested-region rejection','controlled_diagnostics'),('T_consensus_nested8','Temporal + 8 + majority + rejection','controlled_diagnostics')]
 for arm,label,phase in names:
  vals=[next(s['mean']*100 for s in datasets[phase]['summary'] if s['dataset']==ds and s['arm']==arm and s['metric']=='m_vIoU') for ds in ['hc2','vidstg']];table.append((label,*vals))
 body='\n'.join(f'| {l} | {a:.3f} | {b:.3f} |' for l,a,b in table)
 text=f'''# Weight-free Spatial proposals: bounded development experiment

Date: 2026-10-15. **Measured and complete for eight historical development cases; not adopted as a replacement for the frozen external-expert method.**

## Question and actual configuration

Can classical whole-image region generation replace YOLO/SAM2 while preserving full Student tube quality? Eight previously exposed sources (HC01–HC04, V01–V04), one query each, four QA-centered five-frame windows each. Student remains the original cross-domain TA-STVG source checkpoint, episodic Query/LN reset, 1792 spatial parameters, original five-step Gaussian IoU-OPD and training-reference acceptance. Temporal remains the saved four-candidate Caption–Query/512-parameter Adam result. Source/Temporal/original external-expert controls are exact saved results, not newly inferred controls.

Selective Search Fast proposes whole-image rectangles at maximum edge 320; first 256, appearance/size diversification, up to 24 GrabCut refinements, NMS 0.7, at most eight new regions. DIS adjacent forward/backward consistency propagates only the local five-frame mask, with two-pixel cycle filtering. Source is retained. Fixed budgets expose four or eight new candidates to frozen STVG-R1-7B. Candidate IDs are local to each window; there is no established long-range object identity. The 448-pixel crop-contact-video prompt is a changed R1 input protocol, not equivalent to native whole-video marked-instance input. OpenCV-contrib 4.11.0.86 is isolated in this artifact directory; no detector/segmenter weights were downloaded. R1 and TA-STVG remain learned models: only the proposal generator is weight-free.

No GT was used to generate regions, select references, update Student, or accept updates. Each phase sealed all predictions before official scoring. The second phase was designed after inspecting first-phase development results; it is exploratory and cannot count as independent confirmation. No GT Oracle fits, full Table1 execution, CURRENT promotion, or restart of stopped queues occurred.

## Full-tube results

Official query-macro vIoU in percent, four queries per domain. With one query per source here, these means also equal source-macro; eight exposed cases do not support a population claim. Every positive, negative, neutral and undefined result is retained in the accompanying JSON.

| Method | HC-STVG-v2 | VidSTG |
|---|---:|---:|
{body}

The four-proposal variant loses 3.747 pp HC and 6.552 pp Vid relative to the original full dual branch. Eight proposals approximately preserve HC (+0.059 pp) but lose 12.696 pp Vid. No one new recipe preserves both domains. Choosing four for Vid and eight for HC after these scores would be dataset-specific development selection and was not adopted.

## Controlled debugging and negative evidence

The second phase fixes all eight-proposal regions and tests two factors: (1) original, rotate-by-three and reversed tile/ID presentations, choosing a canonical proposal only with at least two votes; (2) reject a selected region if at least 90% of its area is inside Source while its area is below two-thirds of Source. The latter is purely predicted geometry and can also reject a useful correction to an oversized Source box.

Three-order voting yields 12 non-Source anchors, 11 Source choices and nine abstentions/disagreements. It improves the eight-proposal Vid mean but damages HC; stable majority is not evidence of semantic correctness. The nesting rule rejects **zero original eight-proposal anchors**, so it does not explain or cure the original negative updates. Combined with majority it rejects one V01 anchor and only partly reduces its harm. Both results, including the zero-effect control, are retained.

Private visual review of the HC01 first window and all eight cases at center-slot 2 shows body parts, texture and background proposals. This is evidence that the generator lacks an object-completeness constraint, not proof that every selected bad reference is a nested body-part box. We did not claim all 32 windows or all video frames were visually reviewed. The existing native teacher-reference agreement diagnostic is an imperfect reference-system comparison, not GT coverage or oracle truth.

Notable first-phase individual changes versus Temporal alone: HC03 -5.728 pp with four proposals versus +8.326 pp with eight; V01 -15.649 pp with eight; V04 +11.620/+11.986 pp with four/eight. Positive recovery and severe negative transfer coexist. More proposals and more voting do not monotonically help.

## Actual work and cost

32 windows yielded 256 new region hypotheses. Initial phase: 64 R1 calls and 12 new complete Student fits/redecodes. Diagnostic phase: 64 additional R1 calls and five new complete fits/redecodes. **Total 128 R1 calls and 17 new fits**, not 34 fits: the diagnostic reader's 22 logical fit records include exact original/same-reference reuse. Copied evidence and repeated controls are not independent inference.

Cached-window candidate generation took {execution['cached_window_candidate_seconds']:.3f} CPU wall seconds. Initial R1 generation took {execution['phases']['initial']['teacher_generate_seconds']:.3f} seconds (request total {execution['phases']['initial']['teacher_request_seconds']:.3f}); the additional 64 diagnostic calls took {execution['phases']['controlled_diagnostics']['teacher_generate_seconds']:.3f} generation seconds. Student worker totals were {execution['phases']['initial']['student_worker_wall_seconds']:.3f} and {execution['phases']['controlled_diagnostics']['student_worker_wall_seconds']:.3f} seconds. These exclude the already-existing Temporal results, original video preprocessing, cache preparation, and other separately scoped work. R1 times exclude checkpoint loading/hash verification. The short 96-token output protocol is different from the native teacher protocol. These component timings cannot establish a full-video speedup ratio or a reliable Table1 ETA. Units exited normally; no OOM was observed in these runs.

## Verification and decision

Eight synthetic CPU contracts passed. Source frames/pixels/frozen outputs are matched on recapture; actual updates are followed by complete model redecoding and cached/full decoder equality. Independent CPU readback checks reference binding, float64 IoU, acceptance, saved Adam/parameter rollback and final output reconstruction. It does **not** independently rerun backward gradients. Official versus independent scoring maximum absolute discrepancy is {metric_error:.3g}; all summary fields are independently reaggregated. Existing original-expert baseline has no new-reference supervision partition in these exports; compare overall metrics, not its empty supervised-frame split against new arms.

The tested pure-algorithm replacement is fast on cached windows but has not met the user's low-degradation objective. Preserve the original frozen A method and all positive/negative evidence; do not expand this replacement to Table1. These failures apply to these finite configurations, not all weight-free segmentation. A future speed experiment should isolate the native tracker's repeated suffix propagation/cache scheduling from its object proposal quality, with actual profiling and matched outputs before claiming equivalence. That change has not been implemented or benchmarked here.
'''
 report=ROOT/'docs/STVG_WEIGHTFREE_SPATIAL_REPORT.md';report.write_text(text);paths.append(report)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(11,5.7));labels=[x[0].replace('Temporal + ','T + ') for x in table];y=list(range(len(table)))
 ax.barh([i-.18 for i in y],[x[1]for x in table],height=.34,label='HC-STVG-v2');ax.barh([i+.18 for i in y],[x[2]for x in table],height=.34,label='VidSTG');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlabel('Official vIoU (%) — four exposed development sources/domain');ax.legend();ax.set_title('Fast classical regions do not preserve both-domain quality');fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig);paths.append(out/'comparison.png')
 sources=set(str(p.relative_to(ROOT))for p in (ROOT/'scripts').glob('*stvg_weightfree*.py'));sources.add('protocols/stvg_weightfree_spatial_v1.md')
 for d in [B,C]:
  for name in ['STUDENT_RUNTIME.json','TEACHER_RUNTIME.json']:
   for p,h in read(d/name).get('pins',{}).items():
    if p.startswith(('scripts/','vg_tta/','methods/','protocols/')) and Path(p).suffix in ['.py','.md']:
     assert sha(ROOT/p)==h,p;sources.add(p)
 save('source_bindings.json',dict(status='pass',sources={p:sha(ROOT/p)for p in sorted(sources)},phase_barriers={k:sha(d/'GLOBAL_PREDICTION_BARRIER.json')for k,d in [('initial',B),('controlled_diagnostics',C)]},protected_current_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
 for p in paths:
  if p.suffix=='.json':
   s=p.read_text();assert all(k not in s for k in ['/home/wwww/','query_text','raw_output','physical_frame_ids','video_path','box_cxcywh']),p
 for phase,d in [('initial',B),('controlled_diagnostics',C)]:assert read(out/(phase+'_summary.json'))==read(d/'POSTSEAL_SUMMARY.json')
 write(B/'ROOT_POSTSEAL_METRIC_READBACK.json',read(out/'metric_audit.json'))
 paths += [ROOT/p for p in sorted(sources)]
 stage=B/'public_stage';stage.mkdir(exist_ok=False);manifest=[]
 for p in paths:
  rel=str(p.relative_to(ROOT));dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),encoding='base64'if p.suffix=='.png'else'utf-8'))
 write(B/'PUBLIC_MANIFEST.json',manifest);write(B/'PUBLIC_SCALAR_AUDIT.json',dict(status='pass',phase_case_arm_rows=144,all_positive_negative_neutral_unknown_preserved=True,private_media_query_coordinates_excluded=True,file_count=len(manifest),bytes=sum(f['bytes']for f in manifest)))
 print(json.dumps(execution,indent=2));print('public',len(manifest),sum(f['bytes']for f in manifest))
if __name__=='__main__':main()

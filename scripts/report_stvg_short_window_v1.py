"""Complete anonymous evidence export and fixed finite-scope scientific figures."""
import json,hashlib,statistics
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];B=R/'artifacts/stvg_short_event_tracklets_v1';O=R/'results/stvg_short_event_tracklets/2026-10-14'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def alias(cid):return ('HC' if cid.startswith('hc2') else 'V')+f'{int(cid[-3:])+1:02d}'
def clean(v):
 if isinstance(v,dict):return {k:clean(x)for k,x in v.items()if k not in {'GT_source_bindings','private_path','raw_output'}}
 if isinstance(v,list):return [clean(x)for x in v]
 if isinstance(v,str):
  if v.startswith(('artifacts/','/home/','/tmp/')):return 'private_artifact_binding'
  for cid in [f'{s}_{i:03d}'for s in ['hc2','vidstg']for i in range(4)]:v=v.replace(cid,alias(cid))
 return v
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 d=read(B/'ROOT_POSTSEAL_REPORT.json');assert d['status']=='pass'and d['metric_count']==48
 refs=read(B/'METHOD_REFERENCE_LOCK.json');tb=read(B/'TEACHER_GLOBAL_BARRIER.json');calls=[]
 for f in tb['files']:
  p=R/f['path'];assert sha(p)==f['sha256'];r=read(p)
  calls.append({k:r[k]for k in ['status','job_id','case_id','dataset','kind','center_slot','parsed','output_truncated_at_cap','source_case_sha256','prompt_sha256','candidate_count','source_only_non_comparable','empty_candidate_pool','input_tokens','visual_tokens','output_tokens','model_forward_generate_seconds','end_to_end_request_seconds','GPU_peak_bytes','teacher_frozen','GT_read','mean_gap_encoder_pacing_approximation']})
 refdiag=[]
 for row in refs['records']:
  refdiag.append(dict(case_id=row['case_id'],dataset=row['dataset'],roles={role:dict(no_update=v['no_update'],reason=v['reason'],anchor_count=len(v['anchors']),source_grid_positions=[x['position']for x in v['anchors']],selected_candidate_ids=[x['selected_candidate_id']for x in v['anchors']])for role,v in row['references'].items()}))
 cost=dict(status='measured',teacher_call_count=len(calls),input_tokens=sum(c['input_tokens']for c in calls),visual_tokens=sum(c['visual_tokens']for c in calls),output_tokens=sum(c['output_tokens']for c in calls),summed_request_seconds=sum(c['end_to_end_request_seconds']for c in calls),summed_generation_seconds=sum(c['model_forward_generate_seconds']for c in calls),peak_gpu_bytes=max(c['GPU_peak_bytes']for c in calls),request_latency_excludes_shared_initial_model_load=True,interpretation='finite diagnostic cost; not end-to-end production TTA latency')
 cost['stage_launch_to_exit_seconds']={stage:read(B/(stage+'_WORKER_EXIT.json'))['time']-read(B/(stage+'_LAUNCH.json'))['time']for stage in ['FEATURE','TEACHER','STUDENT']}
 for name,val in [('all_postseal_diagnostics',d),('all_teacher_calls',dict(status='complete',count=64,calls=calls,cost=cost)),('reference_selection',dict(status='locked',records=refdiag))]:save(O/(name+'.json'),clean(val))
 groups=d['groups'];tab=['| Domain | Arm | vIoU % | Paired delta pp | Positive / negative / unchanged |','|---|---|---:|---:|---:|']
 for g in groups:tab.append(f"| {g['dataset']} | {g['arm']} | {100*g['mean_vIoU']:.6f} | {g['mean_delta_pp']:+.6f} | {g['positive']} / {g['negative']} / {g['neutral']} |")
 wins=d['window_pool_teacher'];known=[w for w in wins if w['delta_pool_pp']is not None];tw=[w for w in wins if w['delta_teacher_pp']is not None]
 def counts(rows,key):return dict(positive=sum(x[key]>1e-8 for x in rows),negative=sum(x[key]<-1e-8 for x in rows),neutral=sum(abs(x[key])<=1e-8 for x in rows))
 diag=dict(common_windows=len(known),unknown_pool=len(wins)-len(known),pool=counts(known,'delta_pool_pp'),teacher=counts(tw,'delta_teacher_pp'),teacher_known=len(tw),teacher_unknown=len(wins)-len(tw),valid_ID_pairs=sum(w['ID_invariant']is not None for w in wins),ID_invariant=sum(w['ID_invariant']is True for w in wins),base_abstain=sum(w['abstain_or_invalid']for w in wins))
 save(O/'diagnostic_summary.json',diag)
 report='''# Short event-local tracklets: fixed-optimizer spatial diagnosis

This is the second finite controlled diagnostic, on the original eight historically exposed sources (four per direction). It is not the independent 32-source-per-direction confirmation. Temporal and Spatial remain the intended dual branch. CURRENT and stopped historical queues are unchanged.

## Fixed construction and scope

Four centers per source are chosen inside the sealed QA event on the original student frame grid. Each center receives its actual physical -2 to +2 frame window (20 selected frames per source, 160 total). Each window is initialized separately. The existing Shi-Tomasi / Lucas-Kanade / RANSAC helper and parameters are unchanged. Non-Source seeds copy existing proposal geometry to each center; this is not a newly discovered whole-scene detector and does not guarantee that the object is inside that box at the new time.

152 short-window proposals were constructed. All missing boxes remain missing. 427 valid frame boxes measure coverage, not correctness. Frozen TA-STVG features were newly captured for the same 160 pixels/physical frame IDs; ROI descriptors use area overlap, per-frame L2 normalization, and equal averaging. Adjacent-window association is mutual nearest cosine over all valid candidates, with deterministic ID tie breaking. Appearance association is not proven instance identity. Independent ROI/cosine readback found 99 mutual-nearest candidate edges across 24 boundaries; 15 selected mutual pairs were all Source-to-Source, with zero supported selected non-Source positions.

R1 saw four unmarked event-context frames and five marked local frames per call. There were 32 base calls and 32 ID-rotation controls. The selected reference uses base calls only; control answers do not retroactively select a better result. Timestamps bind physical frames; encoder pacing uses mean gaps and is not an exact irregular video clock. Candidate quality is measured later at common annotated centers, not inferred from tracklet count.

Two prelocked rules are evaluated: local_R1 retains valid selected non-Source center boxes; associated_R1 retains only local_R1 boxes whose selected candidate has selected mutual-nearest support in an adjacent window. Source selection and empty references produce exact no-update F/T. Four local references (one anchor each) triggered four real Gaussian IoU OPD fits; the associated rule retained zero references. The original five declared rounds, 1792 Query/LN coordinates, configuration and stochastic key are retained. Zero-information rounds do not force an Adam update. Final full-model redecodes are required. Original-eight T is exactly reused, not retrained.

## Full-Tube results

S deltas below are versus Frozen. T+S deltas are versus the exact Temporal-only control. T itself is shown at its absolute score; its zero self-delta is not a claim of no temporal benefit. Positive/negative/unchanged count sources, not frames. Scores are postseal official metrics independently recomputed from saved full tubes.

'''+ '\n'.join(tab)+f'''

## Separate sparse geometry and teacher diagnostics

{len(known)} of 32 windows have GT-known common support for Source and every displayed candidate; {32-len(known)} remain unknown. Pool Oracle: {diag['pool']}. Teacher: {diag['teacher']}, with {diag['teacher_unknown']} unknown (including abstention). ID consistency is {diag['ID_invariant']}/{diag['valid_ID_pairs']} among pairs with two non-abstaining selections; abstain/abstain is not counted as successful identity understanding. All 64 responses parsed legally, with zero truncation; legality is not correctness.

These are sparse center IoUs. They cannot be subtracted from full-Tube vIoU or treated as complete-trajectory quality. Oracle is an offline upper bound and never enters candidate generation or online selection. Source in the pool guarantees a nonnegative maximum and alone does not prove useful improvement. Every candidate/window positive, negative, neutral and unknown is exported.

The short-window construction changes observation support and candidate presentation together. Its contrast with the old long-span experiment is a package-level diagnostic, not isolated proof that one tracking change caused any benefit. The local/associated rules share the same teacher outputs and optimizer. The associated rule's exact no-update result is an abstention/coverage result, not a learned improvement.

## Cost and audit boundary

64 real teacher calls: {cost['input_tokens']} input tokens, {cost['visual_tokens']} visual tokens, {cost['output_tokens']} output tokens; summed request latency {cost['summed_request_seconds']:.6f} seconds and generation {cost['summed_generation_seconds']:.6f} seconds. Shared model loading is excluded from these request timers. Actual stage launch-to-exit wall seconds (including loading and IO) are {cost['stage_launch_to_exit_seconds']}; these overlap internal request timers and must not be added to them. Eight frozen feature captures and eight student source captures are separate operations. These finite diagnostic costs are not production end-to-end TTA cost.

Saved source hashes, exact F/T reuse, all candidate/feature/teacher bindings, random samples, reward, likelihood, distillation weights, stored gradients and Adam updates are independently read back on CPU. This does not claim an independently repeated decoder backward. Final decoder output equality is checked by the actual worker. All eight private input contact sheets were reviewed at pixel level; raw media, query text, GT coordinates, weights and parameter caches are excluded from the public export.

Preparation/environment/index-binding failures are retained with recovery receipts. They occurred before student execution and do not alter scientific candidates or completed teacher answers. Resource protection is scoped to the current own unit. No observed OOM is not a general guarantee.

## Decision boundary

This finite screen does not freeze a robust Spatial method. No GT-selected amplitude, candidate radius, learning rate, prompt or confidence threshold is introduced. Independent >=32 sources per direction remain conditional on a fixed GT-free rule and verified exclusion from all earlier development. Such confirmation must retain F/T/S/T+S; Temporal-only is a control, not a replacement of the dual-branch objective.
'''
 (R/'docs/STVG_SHORT_EVENT_TRACKLETS_V1.md').write_text(report)
 plt.rcParams.update({'font.size':10,'pdf.fonttype':42});pdf=R/'output/pdf/stvg_short_event_tracklets_v1.pdf'
 with PdfPages(pdf)as pp:
  fig,axs=plt.subplots(1,2,figsize=(11,5))
  for ax,domain in zip(axs,['hc2','vidstg']):
   gg=[next(g for g in groups if g['dataset']==domain and g['arm']==a)for a in ['F','T','local_R1','T_local_R1','associated_R1','T_associated_R1']];ax.bar(range(len(gg)),[g['mean_vIoU']*100 for g in gg],color=['#888888','#577da7','#bb8048','#478779','#bbb1a3','#a7c8bd']);ax.set_xticks(range(len(gg)),['F','T','Local S','T+Local','Assoc. S','T+Assoc.'],rotation=25,ha='right');ax.set_ylabel('Full-Tube vIoU (%)');ax.set_title(domain+' | 4 historical sources')
  fig.suptitle('Fixed optimizer: actual full-Tube readouts');fig.text(.08,.025,'Associated reference rule abstains on all eight sources; exact F/T reuse, no fit.',fontsize=9);fig.tight_layout(rect=[0,.06,1,.95]);pp.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11,6));rr=[m for m in d['metrics']if m['arm']=='local_R1'];vals=[x['delta_vIoU_pp']for x in rr];ax.bar([alias(x['case_id'])for x in rr],vals,color=['#3a897b'if x>0 else'#b56855'if x<0 else'#aaa'for x in vals]);ax.axhline(0,c='gray',lw=.8);ax.set(ylabel='Full-Tube change from Frozen (pp)',title='Per-source results retain both gains and damage');fig.text(.08,.025,'Four actual nonempty-reference fits; other cases exactly unchanged. No GT-selected update amplitude.',fontsize=9);fig.tight_layout(rect=[0,.05,1,1]);pp.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(11,9));ax.axis('off');table=[]
  for w in wins:table.append([alias(w['case_id']),w['slot'],w['R1_candidate']if w['R1_candidate']is not None else'abstain','unknown'if w['delta_pool_pp']is None else f"{w['delta_pool_pp']:+.3f}",'unknown'if w['delta_teacher_pp']is None else f"{w['delta_teacher_pp']:+.3f}"])
  tb=ax.table(cellText=table,colLabels=['Source','Window','R1 ID','Pool delta (pp)','Teacher delta (pp)'],loc='center',cellLoc='center');tb.auto_set_font_size(False);tb.set_fontsize(8);tb.scale(1,1.12);ax.set_title('Sparse common-center IoU diagnostics (not full-Tube vIoU)',pad=18);fig.text(.08,.02,'Unknown is preserved; pool includes Source. Offline Oracle is never used for method selection.',fontsize=9);fig.tight_layout(rect=[0,.04,1,.98]);pp.savefig(fig);plt.close(fig)
 print(json.dumps(dict(status='generated_not_yet_viewed',pages=3,diagnostics=diag)))
if __name__=='__main__':main()

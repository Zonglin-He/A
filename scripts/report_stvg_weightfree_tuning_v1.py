"""Complete anonymous parameter-study and held-selection confirmation export."""
import sys,json,math,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
B=ROOT/'artifacts/stvg_weightfree_tuning_v1';E=B/'confirmation003'
def alias(c):
 if c.startswith(('HC','V')):return c
 return ('HC'if c.startswith('hc2')else'V')+f'{int(c.rsplit("_",1)[1])+1:02d}'
def main():
 out=ROOT/'results/stvg_weightfree_tuning/2026-10-15';out.mkdir(parents=True,exist_ok=True);paths=[];audit_fields=0;metric_error=0.;allrows=0
 def save(n,o):
  p=out/n;write(p,o);paths.append(p)
 phases=[('lr',B),('factors',B/'factors002'),('interactions',B/'interaction004'),('confirmation',E)]
 execution={};summaries={}
 for phase,d in phases:
  rows=read(d/'POSTSEAL_ROWS.json')['rows'];summ=read(d/'POSTSEAL_SUMMARY.json');bar=read(d/'GLOBAL_PREDICTION_BARRIER.json');sb=read(d/'STUDENT_GLOBAL_BARRIER.json')
  for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
  assert read(d/'GT_EXPOSURE.json')['seal_sha256']==sha(d/'GLOBAL_PREDICTION_BARRIER.json')
  for s in summ['summary']:
   vals=[r['metrics'][s['metric']]for r in rows if r['dataset']==s['dataset']and r['arm']==s['arm']and r['metrics'][s['metric']]is not None]
   assert len(vals)==s['count'] and ((not vals and s['mean']is None)or abs(math.fsum(vals)/len(vals)-s['mean'])<1e-14);audit_fields+=1
  correct=d/'CORRECTED_CONTRASTS002.json'
  if correct.exists():summ=dict(summ,contrasts=read(correct)['contrasts'],contrast_dispatch_corrected=True)
  lookup={(r['case_id'],r['arm']):r for r in rows}
  for s in summ['contrasts']:
   ids=sorted({r['case_id']for r in rows if r['dataset']==s['dataset']});vals=[lookup[c,s['left']]['metrics'][s['metric']]-lookup[c,s['right']]['metrics'][s['metric']]for c in ids if lookup[c,s['left']]['metrics'][s['metric']]is not None and lookup[c,s['right']]['metrics'][s['metric']]is not None]
   assert len(vals)==s['count'] and ((not vals and s['mean_delta_pp']is None)or abs(100*math.fsum(vals)/len(vals)-s['mean_delta_pp'])<1e-12);audit_fields+=1
  pub=[]
  for r in rows:
   err=r['metric_audit']['official_independent_max_abs_error'];metric_error=max(metric_error,err);pub.append(dict(case=alias(r['case_id']),dataset=r['dataset'],arm=r['arm'],metrics=r['metrics'],support_counts={k:v for k,v in r['support'].items()if k.endswith('_count')},official_independent_max_abs_error=err))
  allrows+=len(pub);save(phase+'_all_results.json',dict(status='complete',cohort='development4_per_domain'if phase!='confirmation'else'locked_confirmation16_per_domain',historically_exposed_project_sources=True,rows=pub));save(phase+'_summary.json',summ);save(phase+'_root.json',read(d/'ROOT_READBACK.json'));summaries[phase]=summ
  execution[phase]=dict(case_count=sb['count'],actual_fit_count=sum(f['actual_fit_count']for f in sb['files']),full_redecode_count=sum(f['full_redecode_count']for f in sb['files']),student_worker_wall_seconds=sb['wall_seconds'])
 save('policy_arithmetic_root.json',read(B/'ROOT_POLICY_ARITHMETIC_COMPLETE002.json'));save('input_render_root.json',read(E/'ROOT_INPUT_RENDER_READBACK.json'));save('cohort_root.json',read(E/'ROOT_COHORT_READBACK.json'))
 sel=read(B/'METHOD_SELECTION_LOCK.json');selectionaudit=read(B/'DEVELOPMENT_SELECTION_AUDIT.json');save('development_search.json',dict(status='complete',records=selectionaudit['records'],selected=selectionaudit['selected'],development_GT_used=True));save('frozen_configuration.json',sel)
 save('paired_parent_bootstrap.json',read(E/'PARENT_BOOTSTRAP.json'));save('paired_stage_cost.json',read(B/'PAIRED_COST_READBACK.json'));save('visibility_threshold_audit.json',read(B/'VISIBILITY_THRESHOLD_READBACK.json'))
 tb=read(E/'TEACHER_GLOBAL_BARRIER.json');tr=[read(ROOT/f['path'])for f in tb['files']]
 for f in tb['files']:assert sha(ROOT/f['path'])==f['sha256']
 teacher_public=[]
 for t in tr:
  ident=t['parsed']['selected_candidate_id'];binding=next((x for x in t['candidate_bindings']if x['id']==ident),None)
  teacher_public.append(dict(case=t['case_id'],dataset=t['dataset'],center_slot=t['center_slot'],parse_status=t['parsed']['status'],selected_id=ident,selected_role=binding['role']if binding else None,selected_presence=binding['presence']if binding else None,candidate_count=t['candidate_count'],truncated=t['output_truncated_at_cap'],input_tokens=t['input_tokens'],visual_tokens=t['visual_tokens'],output_tokens=t['output_tokens'],model_seconds=t['model_forward_generate_seconds'],request_seconds=t['end_to_end_request_seconds']))
 save('confirmation_teacher_all_decisions.json',dict(status='complete',records=teacher_public))
 cb=read(E/'CANDIDATE_INPUT_BARRIER.json');temp=read(E/'TEMPORAL_GLOBAL_BARRIER.json')
 execution['confirmation'].update(temporal_calls=temp['teacher_calls'],temporal_stage_wall_seconds=temp['wall_seconds'],temporal_load_seconds=temp['model_load_seconds'],candidate_stage_wall_seconds=cb['wall_seconds'],candidate_compute_seconds=cb['proposal_seconds'],candidate_windows=cb['window_count'],new_candidate_regions=read(E/'ROOT_INPUT_RENDER_READBACK.json')['mask_track_count'],spatial_R1_calls=len(tr),valid_output_count=sum(t['parsed']['status']=='valid'for t in tr),truncated_count=sum(t['output_truncated_at_cap']for t in tr),R1_generate_seconds=math.fsum(t['model_forward_generate_seconds']for t in tr),R1_request_seconds=math.fsum(t['end_to_end_request_seconds']for t in tr),input_tokens=sum(t['input_tokens']for t in tr),visual_tokens=sum(t['visual_tokens']for t in tr),output_tokens=sum(t['output_tokens']for t in tr))
 save('update_outcome_counts.json',read(E/'UPDATE_OUTCOME_COUNTS.json'))
 save('execution.json',dict(status='complete',phases=execution,old_default_results_reused=True,GT_Oracle_fits=0,new_detector_segmenter_models=0,cost_scope='Individual stage timers, different included work; not additive end-to-end latency or full Table1 ETA. Development reuses source/temporal/R1. Confirmation reuses sealed source predictions and existing canonical videos where available.'))
 ref=read(E/'REFERENCE_LOCK.json')['records'];decisions=[]
 import torch
 for r in ref:
  d=torch.load(E/'student'/(r['case_id']+'.pt'),map_location='cpu',weights_only=False);td=torch.load(E/'temporal'/(r['case_id']+'.pt'),map_location='cpu',weights_only=False)
  decisions.append(dict(case=r['case_id'],dataset=r['dataset'],anchor_count=len(r['original']['anchors']),temporal_status=td['temporal']['status']if 'status'in td['temporal']else'actual_adaptation',canonical_reused=td['canonical_reused'],spatial_receipts=d['receipts']))
 save('confirmation_decisions.json',dict(status='complete',records=decisions,unknown=cb['unknown'],GT_used_for_inference=False))
 configtable='\n'.join('| '+ds+' | '+sel['selected'][ds]+' | '+' | '.join(str(sel['configs'][ds][k])for k in ['lr','sigma','tau','steps'])+' |'for ds in ['hc2','vidstg'])
 devtable=[]
 for v in ['original','lr025','lr050','sigma050','sigma200','tau050','tau200','steps3','steps1','combo1','combo2']:
  vals=[next(r['mean_vIoU']*100 for r in selectionaudit['records']if r['dataset']==ds and r['variant']==v)for ds in ['hc2','vidstg']];devtable.append((v,*vals))
 dt='\n'.join(f'| {v} | {h:.3f} | {v2:.3f} |'for v,h,v2 in devtable)
 ct=[]
 for arm in ['F','T','original','T_original','selected','T_selected']:
  vals=[next(x['mean']*100 for x in summaries['confirmation']['summary']if x['dataset']==ds and x['arm']==arm and x['metric']=='m_vIoU')for ds in ['hc2','vidstg']];ct.append((arm,*vals))
 ctable='\n'.join(f'| {v} | {h:.3f} | {v2:.3f} |'for v,h,v2 in ct)
 boot=read(E/'PARENT_BOOTSTRAP.json')['records'];bt='\n'.join(f"| {x['dataset']} | {x['left']} − {x['right']} | {x['mean_delta_pp']:+.3f} | [{x['CI95_percentile_pp'][0]:+.3f}, {x['CI95_percentile_pp'][1]:+.3f}] | {x['negative_sources']}/16 | {x['worst_delta_pp']:+.3f} |"for x in boot)
 text=f'''# Eight-candidate spatial OPD: parameter diagnosis and larger confirmation

2026-10-15. This finite study extends the eight-source development experiment. It does not promote CURRENT or launch full Table 1. All source checkpoints remain the original cross-domain TA-STVG checkpoints; Temporal is Caption–Query Alignment with 512-parameter Adam, Spatial is episodic 1792-parameter Gaussian IoU-OPD with reference-agreement rejection. Only candidate generation is weight-free; R1 and Student are learned models.

## Actual parameter search

Four historical sources/domain are development, explicitly using post-seal GT metrics to choose hyperparameters. Two real LR alternatives, six one-factor alternatives (sigma 0.5/2, tau 0.5/2, steps 3/1), then two development-guided combinations/domain were run. Original eight-candidate outputs are exact saved reuse. This totals {sum(execution[p]['actual_fit_count']for p in ['lr','factors','interactions'])} new complete development fits and full redecodes, not endpoint parameter interpolation. R1 choices, references, noise keys, temporal predictions and supervision positions are matched. Eleven recipes/domain were considered, with adaptive combinations disclosed. Combination meanings differ by domain: HC combo1=LR0.5×/3 steps, combo2=LR0.5×/sigma2×; Vid combo1=LR0.25×/1 step, combo2=LR0.25×/sigma0.5×.

| Development T+S recipe | HC vIoU % | Vid vIoU % |
|---|---:|---:|
{dt}

Sampling sigma and step count matter; temperature changes are smaller on these four sources/domain. This does not establish that dataset heterogeneity is only a parameter issue. Domain counts are equal. All old selected references have five-of-five presence, so raising the visibility threshold 0.6 to 0.8/1.0 changes zero choices and is reported as a zero-effect metadata check. Candidate area bins, DIS cycle threshold, anchor budget, sample count32, Adam betas and carry were not additionally swept. This is a bounded search, not exhaustive optimization.

## Frozen configuration and confirmation design

| Domain | Selected recipe | LR | sigma | tau | steps |
|---|---|---:|---:|---:|---:|
{configtable}

Selection maximizes old-development mean T+S vIoU; ties within1e-12 use smaller LR, then fewer steps, then name. The complete configuration was locked before confirmation Student execution and label exposure. No reselection on confirmation. The new cohort is 16 unique parents/domain, one query/parent, selected by the prior registry's parent-hash order before development tuning results. It excludes the old development/current four parents. These32 sources have historical project exposure, so they are **not an untouched official test set**. They are new to this route's selection-confirmation split. Old-eight and new32 alias names refer to different sources and must not be joined by alias alone.

Inference does not read GT: frozen source/video hashes bind input; Temporal uses its original four candidates; four QA-centered five-frame windows generate at most eight SelectiveSearch/GrabCut/DIS regions plus Source. R1 selects IDs using the unchanged 448px contact-video protocol. Fewer than four eligible source-grid centers with physical±2 coverage produces explicit no-spatial-update fallback. Invalid/abstained/Source choices or insufficient presence provide no new reference. There is no GT candidate supplementation or Oracle fitting. All32 full Student outputs are sealed before official GT scoring.

## Confirmation full-tube results

Query-macro equals parent-macro here because each parent has one query. F=Frozen, T=Temporal-only; original/selected denote spatial-only and T_ prefixes denote the full dual branch. Original uses default five-step eight-candidate OPD, not YOLO/SAM2. No new native YOLO/SAM2 comparison was run on these32 sources.

| Arm | HC vIoU % | Vid vIoU % |
|---|---:|---:|
{ctable}

| Domain | Paired comparison | Mean Δpp | Parent-bootstrap 95% CI | Negative sources | Worst Δpp |
|---|---|---:|---:|---:|---:|
{bt}

The intervals use10,000 paired resamples within domain. All individual positive, negative, neutral and undefined values, tIoU, dense/supervised/unsupervised spatial metrics and official/independent checks are included in JSON. Undefined support remains null. Confirmation partitions use the same prelocked anchor positions for all six arms; historical development F/T and external-expert support partitions are not matched and should only be compared on overall metrics. This limited confirmation cannot substitute for full official Table1 evaluation or prove equivalence to external instance tracks.

## Interpretation of the held-selection confirmation

The selected recipes improve T+S over the default eight-candidate recipe by0.396pp HC and0.553pp Vid; versus Temporal-only the spatial increment is0.180pp and0.485pp. Both paired95% intervals cross zero. This supports a bounded parameter-control benefit but not a statistically established stable spatial improvement or equivalence to YOLO/SAM2.

Temporal-only itself is2.023pp below Frozen in HC and0.236pp below Frozen in Vid on these additional sources. The final selected dual branch is1.843pp below Frozen in HC and0.249pp above Frozen in Vid. Two HC sources have particularly large Temporal-only declines (HC06−30.571pp, HC14−31.293pp). These are confirmation aliases, not the old development aliases. They remain in the mean and are not used to change Temporal or select another spatial configuration. The larger cohort therefore exposes a separate generalization issue that old-eight development gains did not establish away.

The selected spatial increment versus Temporal has two negative sources/domain; worst−3.067pp HC and−0.941pp Vid, versus default worst−4.175pp and−5.781pp. Many sources are exactly unchanged because no new reference is available, the IoU samples are uninformative, or the proposed update is rejected. The execution/decision files distinguish these cases. A zero metric delta can also arise from zero overlap despite changed boxes; it is not automatically a no-update record. Smaller damage is valuable but is not the same as extracting correct identity from every candidate pool. The current finite result is retained without a production promotion or confirmation-driven search.

## Cost and execution

On the original same eight sources, prior native full-video tracking took2921.429s versus65.162s for cached sparse-window eight-proposal computation: a **44.834× stage-time ratio**,97.770% reduction. Work coverage and cache scope differ; this is not end-to-end acceleration. Native tracking covers full videos; the new stage covers four five-frame windows and excludes RGB export/load. Frozen R1, Temporal, Student, loading and scoring remain. Do not extrapolate this ratio to13,785 queries.

Confirmation: {temp['teacher_calls']} Temporal R1 calls, {cb['window_count']} spatial windows, {len(tr)} spatial R1 calls, {execution['confirmation']['actual_fit_count']} actual Student fits/full redecodes. Temporal worker {temp['wall_seconds']:.3f}s plus separately measured model load{temp['model_load_seconds']:.3f}s; candidate worker {cb['wall_seconds']:.3f}s with {cb['proposal_seconds']:.3f}s proposal computation; spatial R1 generation {execution['confirmation']['R1_generate_seconds']:.3f}s and request total{execution['confirmation']['R1_request_seconds']:.3f}s; Student worker{execution['confirmation']['student_worker_wall_seconds']:.3f}s. These are scoped component measurements, not a single deployment timing. Explicit no-reference/no-temporal-input fallbacks and reused controls are included, not counted as new fits.

## Verification and limitations

Full source recapture verifies checkpoint state, pixels, frame grid, native boxes/logits; actual adapted full-model outputs equal cached decoder outputs, and source state is restored. Independent CPU readback checks reference binding, float64 IoU, saved Adam/parameter rollback and final output geometry. An additional audit reconstructs Gaussian noise, sample coordinates, IoU rewards, softmax weights, weighted losses and Adam updates from saved gradients in float64; it does not independently rerun backward gradients. All confirmation canonical videos are hash-checked, selected pixels freshly decoded, mask bounding boxes recomputed and R1 inputs rendered again for exact equality. This is same-renderer replay, not a second proposal algorithm. Four private contact sheets were actually viewed, covering fixed center slot2 for all32 sources (32 of128 windows). They contain whole objects, parts, background and broad regions; this is display review, not semantic correctness certification. Private media are not published. {audit_fields} summary/contrast fields are independently reaggregated; official/independent metric maximum absolute difference {metric_error:.3g}. The early LR/factor scorer retained stale contrast names and emitted empty contrasts; original rows/means were correct. Original artifacts are preserved, and separate corrected contrast tables are published. No GPU rerun or metric value replacement was needed for this export-dispatch correction.

No CURRENT promotion, new detector/segmenter, GT Oracle fit, old-queue restart, or full Table1 run. Per-domain tuning is disclosed development selection, not label-free hyperparameter discovery. Reference agreement is not an identity guarantee. All negative confirmation results are retained without reopening selection.
'''
 report=ROOT/'docs/STVG_WEIGHTFREE_PARAMETER_CONFIRMATION_REPORT.md';report.write_text(text);paths.append(report)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(12,5));labels=['F','T','S default','T+S default','S tuned','T+S tuned']
 for j,ds in enumerate(['HC-STVG-v2','VidSTG']):
  ax=axes[j];ax.bar(range(6),[x[j+1]for x in ct],color=['gray','#5588aa','#bba077','#c98f42','#84a889','#397f58']);ax.set_xticks(range(6),labels,rotation=25,ha='right');ax.set_title(ds+' · 16 confirmation sources');ax.set_ylabel('Official vIoU (%)');ax.set_ylim(0,max(x[j+1]for x in ct)*1.2)
 fig.suptitle('Frozen development selection on 32 additional sources');fig.tight_layout();fig.savefig(out/'confirmation.png',dpi=160);plt.close(fig);paths.append(out/'confirmation.png')
 sourcepaths=set()
 for _,d in phases:
  for name in ['RUNTIME.json','STUDENT_RUNTIME.json','TEACHER_RUNTIME.json']:
   if not(d/name).exists():continue
   for p,h in read(d/name).get('pins',{}).items():
    if p.startswith(('scripts/','vg_tta/','methods/','protocols/'))and Path(p).suffix in ['.py','.md']:
     assert sha(ROOT/p)==h,p;sourcepaths.add(p)
 sourcepaths.update(str(p.relative_to(ROOT))for p in(ROOT/'scripts').glob('*stvg_weightfree*.py'))
 sourcepaths.update(['protocols/stvg_weightfree_tuning_v1.md','protocols/stvg_weightfree_parameter_addendum002.md'])
 save('source_bindings.json',dict(status='pass',sources={p:sha(ROOT/p)for p in sorted(sourcepaths)},phase_barriers={n:sha(d/'GLOBAL_PREDICTION_BARRIER.json')for n,d in phases},protected_current_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
 save('audit.json',dict(status='pass',phase_case_arm_rows=allrows,summary_contrast_fields_recomputed=audit_fields,official_independent_max_abs_error=metric_error,independent_backward_replay=False,all_positive_negative_neutral_unknown_preserved=True))
 for p in paths:
  if p.suffix=='.json':assert all(k not in p.read_text()for k in ['/home/wwww/','query_text','raw_output','physical_frame_ids','video_path','box_cxcywh']),p
 paths +=[ROOT/p for p in sorted(sourcepaths)];stage=B/'public_stage';stage.mkdir(exist_ok=False);manifest=[]
 for p in paths:
  rel=str(p.relative_to(ROOT));dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst);manifest.append(dict(path=rel,bytes=p.stat().st_size,sha256=sha(p),encoding='base64'if p.suffix=='.png'else'utf-8'))
 write(B/'PUBLIC_MANIFEST.json',manifest);write(B/'PUBLIC_SCALAR_AUDIT.json',read(out/'audit.json'));print('public',len(manifest),sum(f['bytes']for f in manifest));print(ctable)
if __name__=='__main__':main()

"""Report completed four-arm results without promoting a production method."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_temporal_fourarm_v1 import OUT,OLD
from scripts.audit_tastvg_fourarm_public_v1 import audit


def run():
    s=read(OUT/'analysis/SUMMARY.json');rows=read(OUT/'analysis/ROWS.json');excess=read(OUT/'analysis/EXCESS.json')
    write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit(OUT/'analysis'))
    allocations={stage:[read(f) for f in sorted(path.glob('*.json'))] for stage,path in [('capture',OLD/'capture_allocations'),('critic',OUT/'c2_allocations'),('adapt',OUT/'adapt_allocations')]}
    resources=dict(stage_seconds={k:sum(z['seconds'] for z in v) for k,v in allocations.items()},failures=[dict(stage=k,**z) for k,v in allocations.items() for z in v if z['status']=='failed'],
        allocation_count={k:len(v) for k,v in allocations.items()},prior_clean_capture_reused=16,prior_clean_expert_reused=16,completed_adaptation_steps=512,
        actual_new_candidate_cells=240,actual_new_expert_cells=240,includes_loading_and_cpu_decode=True,excludes_development_and_scoring=True)
    resources['total_gpu_process_seconds']=sum(resources['stage_seconds'].values());write(OUT/'RESOURCES.json',resources)
    provenance=dict(student='TA-STVG official VidSTG-source',student_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',
        expert='UniversalVTG best with PE-Core-L14-336',expert_checkpoint_sha256='ac48fa4474f65a84f5176de5dbbf4439ebe7996a3fc361fe10161e9f338a1ba8',
        PE_checkpoint_sha256='0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8',
        student_state_sha256=read(OUT/'PREDICTION_BARRIER.json')['model_state_sha256'],expert_state_sha256=read(OUT/'C2_BARRIER.json')['model_hashes'],
        model_weights_unchanged=True,parents=16,queries=16,transient_cells=240,clean_cells=16,historically_exposed_development=True,
        setting='existing seed0 GT/query-independent full-video random burst; 5 families x 1/5/10 percent',
        steps=1,motion_only=True,nominal_step=.004,step_reference='Round2 nominal 2*.02/10 times combined visual H norm',backtracking=False,GT_in_adaptation=False,production_changed=False)
    write(OUT/'PROVENANCE.json',provenance)
    p=s['transient'];c=s['clean'];metrics=['sIoU','tIoU','vIoU_corrected']
    val=lambda z:f"{z['mean']*100:.4f}"
    contrast=lambda z:f"{z['mean']*100:+.4f} [{z['ci95'][0]*100:+.4f}, {z['ci95'][1]*100:+.4f}]"
    out=['# Minimal C3-T: Frozen / Rerank / Hard / OPD','',
        'Completed on the fixed transient deployment setting. The question is whether adaptation adds value over expert reranking; no extra qualification gate or benchmark redesign was run.','',
        '16 previously exposed VidSTG parents, one query per source, 15 corrupted inputs plus clean per query. Percent values below; paired changes are percentage points. sIoU uses all valid GT-annotated observed frames; vIoU uses the corrected tube temporal union. Average the 15 corruption conditions within each parent, then bootstrap 16 parents (10,000 samples, seed 20260929).','',
        '| Arm | Transient sIoU | Transient tIoU | Transient vIoU | Clean tIoU | Clean vIoU |','|---|---:|---:|---:|---:|---:|']
    for a in ['Frozen','Rerank','Hard','OPD']:out.append('| '+a+' | '+' | '.join([val(p['arms'][a][m]) for m in metrics]+[val(c['arms'][a][m]) for m in metrics[1:]])+' |')
    out+=['','| Paired comparison | Transient tIoU change [95% CI] | Transient vIoU change [95% CI] |','|---|---:|---:|']
    for name,z in p['comparisons'].items():out.append('| '+name+' | '+' | '.join(contrast(z[m]) for m in metrics[1:])+' |')
    out+=['','## Interpretation','']
    dt=p['comparisons']['OPD - Rerank']['tIoU'];dv=p['comparisons']['OPD - Rerank']['vIoU_corrected']
    if dt['mean']<=0 and dv['mean']<=0:
        conclusion='Rerank is at least as good as OPD on both temporal and tube mean scores. Retain simple reranking for this tested recipe; the one-step OPD update has not established added value. This does not rule out other adaptation objectives or scales.'
    elif dt['mean']>0 and dv['mean']>0:
        conclusion='OPD has higher temporal and tube mean scores than Rerank in this development screen. Interpret the paired confidence intervals before claiming a reliable advantage; no production promotion or follow-on experiment is made here.'
    else:conclusion='OPD versus Rerank is mixed across temporal and tube metrics. The present screen does not establish an unqualified adaptation advantage.'
    out+=[conclusion,'',f"Loss decreased after the fixed step in Hard {p['loss_decreased']['Hard']}/240 and OPD {p['loss_decreased']['OPD']}/240 transient cells. No step was selected or rejected by the loss or GT.",'',
        '| Comparison | tIoU harms >5 pp (cells/240) | vIoU harms >5 pp (cells/240) |','|---|---:|---:|']
    for name,z in p['harms_gt5pp'].items():out.append(f"| {name} | {z['tIoU']} | {z['vIoU_corrected']} |")
    out+=['','Clean is a control for generic refinement, not a separate research line. Corrupted-minus-clean paired gain (vIoU): '+', '.join(a+' '+contrast(excess[a]['vIoU_corrected']) for a in excess)+'.','',
        '## Actual implementation','',
        'Frozen and Rerank share native boxes; Rerank chooses from at most eight unchanged TA-STVG decoder/legal-span candidates. The frozen UniversalVTG scores each student candidate by maximum confidence-weighted overlap with its proposals. The teacher interval itself is never emitted. The old clean expert outputs are reused; each corrupted expert output is cached once.','',
        'Hard: native Gaussian endpoint loss toward the critic-best student interval. OPD: mean pairwise logistic ranking loss over all strict expert preferences, using student log start/end probabilities. Both use one fixed normalized gradient step of .004 times the joint visual-H norm and edit only H_motion. H_app/H_text and all model parameters are unchanged. No backtracking, projection, protection gate, low rank, reliability network or sweep. The normal two-stage suffix reroutes dynamically and emits its native output; there is no post-update reranking.','',
        'The backbone is captured once per episode and is not rerun for gradient/update evaluation. Additional full forwards are solely exact replay and selected reinsertion checks. The existing official TTS/ASA detach Jacobian is preserved.','',
        '## Verification and limits','',
        'All 256 four-arm outputs were sealed before scoring; only the original 16 authorized GT records were retained from the historical container, solely for offline metrics. This remains a repeatedly exposed development cohort, not a fresh benchmark. The random burst and severity are unchanged even when the event is unaffected. There is one burst realization per source/severity. Spatial specialists and multiseed work remain backlog.','',
        'AUDIT.json records exact native reuse, two independent metric implementations, independent critic-score reconstruction, four full reinsertion checks, exact non-motion freeze and identical-input controls. PUBLIC_SCALAR_AUDIT.json reconstructs means/CIs from anonymous rows; CPU_TESTS.txt checks loss gradients, native hard loss and endpoint mapping. No production method was changed.','',
        f"GPU process allocation: capture {resources['stage_seconds']['capture']:.3f}s, critic {resources['stage_seconds']['critic']:.3f}s, adaptation {resources['stage_seconds']['adapt']:.3f}s; total {resources['total_gpu_process_seconds']:.3f}s ({resources['total_gpu_process_seconds']/60:.2f}min). Includes loading, decoding, original capture smoke and any failed allocations; excludes CPU development/scoring and previously published clean-cache creation. This is not a fresh-input end-to-end latency benchmark.",'',
        'The earlier C2.5/C3/C0.6 registration is preserved as superseded history. The latest user instruction cancelled C2.5 gates, C0.6 multiseed and conditional/backtracking rules before adaptation outcomes. Only the required candidate capture was reused. See protocols/tastvg_temporal_fourarm_v1.md.','',
        'Anonymous per-cell values, all per-condition summaries, paired contrasts and positive/negative examples are included in ROWS.json, SUMMARY.json, EXCESS.json and EXAMPLES.json. Media, captions, GT coordinates, checkpoints and raw feature/prediction caches remain local.']
    (OUT/'REPORT.md').write_text('\n'.join(out)+'\n')
    write(OUT/'DECISION.json',dict(status='completed',measurement='audited',conclusion=conclusion,production_changed=False,followon='none in this run',time=time.time()))
    write(OUT/'COMPLETION.json',dict(status='completed',files={str(f.relative_to(OUT)):sha(f) for f in [OUT/'REPORT.md',OUT/'AUDIT.json',OUT/'PUBLIC_SCALAR_AUDIT.json',OUT/'analysis/ROWS.json',OUT/'analysis/SUMMARY.json',OUT/'RESOURCES.json',OUT/'PROVENANCE.json',OUT/'DECISION.json']},time=time.time()))
    print(conclusion)

if __name__=='__main__':run()

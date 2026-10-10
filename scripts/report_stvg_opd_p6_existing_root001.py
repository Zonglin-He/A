"""Complete original P6 result report, without further model or GT calls."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import verify,BASE,PUB,NS,read,sha
ARMS=['Native','ActionnessProjection','TemporalHead','SpatialOPD','Offline_GT_Head']


def interval(m):
    return f"{m['parent_macro']*100:+.3f} [{m['ci95'][0]*100:+.3f}, {m['ci95'][1]*100:+.3f}]"


def run():
    rt=verify();out=PUB/'P6';state=read(out/'ROOT_MATH_STATE_DENSE_READBACK.json')
    view=read(BASE/'P6_ROOT_VISUAL_REVIEW.json');assert state['status']==view['status']=='pass'
    stats=read(out/'STATISTICS.json');rows=read(out/'ROWS.json');assert len(rows)==64
    lines=['# Fixed OPD P6: existing temporal signals, temporal head and offline supervised failure diagnostic','',
        'On the original fixed 32 historical parent sources per target, actionness projection improves vIoU relative to Native in both directions. '
        'The existing temporal-head update has a positive vIoU interval on HC2 but its VidSTG interval contains zero. '
        'On VidSTG the head is below actionness projection by 0.689 pp [−1.610, −0.020], conditional on this small historical cohort. '
        'The offline GT-supervised head gives substantially larger VidSTG gains, so weak deployment evidence and head capacity must be distinguished. '
        'No main-method change or result-driven parameter selection follows from these diagnostics.','',
        'Original attachment 029fff2e section 9 and LATER_DESIGN_LOCK fixed 32 hash-selected historical parents per target before the current P0 GT scoring. '
        'Each contributes one clean query, its complete original physical frame grid and two original offsets. Four deployment arms yield 256 outputs for 64 queries. '
        'All 256 deployment outputs globally sealed before any P6 GT read or offline GT-head fitting. Qualification outputs are not accepted as formal.','',
        '| Deployment arm | Actual implementation and state history |','|---|---|',
        '| Native | Original frozen boxes and native interval. |',
        '| ActionnessProjection | Existing original median/MAD physical-cell actionness projection with the original native prior; frozen boxes and no optimizer. |',
        '| TemporalHead | Existing 66,306-parameter two-layer temporal head; per-query source-head reset and fresh AdamW, five steps, last output with eta .25. Frozen prefix and boxes; genuine GPU native reinsertion. |',
        '| SpatialOPD | Exact complete original P0 on-policy order1 fit for the matched query and input. Its full 128-query stream predecessors and LN history remain intact. |',
        '| Offline_GT_Head | Separately supervised CPU head fit after all deployment seals, same five steps and eta .25, first legal pair maximizing physical tIoU on each offset. Not available at deployment or a fair deployable baseline. |','',
        'The temporal head retains original NLL plus margin .2, AdamW betas (.9,.999), epsilon 1e-4, no weight decay, prior .1 and original decoding. '
        'VidSTG source → HC2 uses the existing vid_to_hc1 representative lr .001 and center fraction1 without HC2 tuning; '
        'HC-STVG2 source → VidSTG uses original lr .1 and center fraction .5. '
        'Source checkpoints are VidSTG SHA256 5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83 and '
        'HC-STVG2 SHA256 47d8f15841cd57e7bbf5a10e8bf23b1054d23b753e0becbd38a07f3dd60d5036. '
        'SpatialOPD retains HC .01/.025/.05/40/LN1/16/M32 and Vid .03/.1/.25/10/LN1/8/M32, original Uniform4/Top1 frozen DINO and 1792 joint parameters.','',
        'SpatialOPD is a descriptive matched-input component reference with its original longer state history; it is not a freshly reset 32-query online stream or a history-matched temporal-versus-spatial causal contrast. '
        'Main SpatialOPD Native WHEN remains unchanged. Existing temporal alternatives are evaluated separately under the original finite P6 diagnostic.','',
        'Query macro and parent macro are equal because each original parent has one query. Intervals use 10,000 paired original-parent bootstrap draws, seed20261006. '
        'Intervals are conditional on the historical roster, source checkpoints and stated state histories, not fresh-source generalization guarantees. '
        'All ten pairwise contrasts per target and all three metrics are retained without multiplicity adjustment.','',
        '| Target | Arm | vIoU | tIoU | sIoU | vIoU−Native, pp [95% CI] | >5/>20pp harm parents |','|---|---|---:|---:|---:|---|---:|']
    for ds in stats:
        ms=stats[ds]['metrics']
        for arm in ARMS:
            d=None if arm=='Native' else ms[f'delta_{arm}_minus_Native_v']
            lines.append(f"| {ds} | {arm} | {ms[arm+'_v']['parent_macro']*100:.3f} | {ms[arm+'_t']['parent_macro']*100:.3f} | {ms[arm+'_s']['parent_macro']*100:.3f} | {interval(d) if d else 'reference'} | {str(d['harm_gt5pp_parents'])+'/'+str(d['harm_gt20pp_parents']) if d else '—'} |")
    lines+=['','Temporal arms keep all boxes fixed, so their fixed-GT-support sIoU is identical to Native. SpatialOPD retains Native time, so its tIoU is identical to Native. '
        'Offline GT supervision is not a task-score upper bound: frozen boxes, finite updates and shrink can still harm an individual query.','',
        '| Target | Complete paired contrast | vIoU, pp [95% CI] | tIoU, pp [95% CI] | sIoU, pp [95% CI] |','|---|---|---|---|---|']
    for ds in stats:
        for key,m in stats[ds]['metrics'].items():
            if key.startswith('delta_') and key.endswith('_v'):
                pre=key[:-2];lines.append(f"| {ds} | {pre[6:].replace('_minus_',' − ')} | {interval(m)} | {interval(stats[ds]['metrics'][pre+'_t'])} | {interval(stats[ds]['metrics'][pre+'_s'])} |")
    lines+=['','| Target | Head−Native positive/negative parents | Gross gain/loss, pp | Worst parent, pp | SSL/GT gradient cosine mean | Negative cosine count |','|---|---:|---:|---:|---:|---:|']
    for ds in stats:
        rr=[r for r in rows if r['dataset']==ds];m=stats[ds]['metrics']['delta_TemporalHead_minus_Native_v']
        cos=[r['signal']['gradient_cosine'] for r in rr if r['signal']['gradient_cosine'] is not None]
        lines.append(f"| {ds} | {m['positive_parents']}/{m['negative_parents']} | {m['gross_gain_pp']:.3f}/{m['gross_loss_pp']:.3f} | {m['min']*100:.3f} | {np.mean(cos):.4f} | {sum(x<0 for x in cos)}/{len(cos)} |")
    lines+=['','The gradient cosine is computed in the existing temporal-head parameter space between the actual first SSL gradient and the same-source offline GT-head gradient. '
        'It diagnoses signal alignment on this saved query; it is not a deployment selection criterion. Original source/offset support and legal-pair maximum tIoU, '
        'teacher-target tIoU, all five update steps, actual SSL losses, finite last-output tIoU, observed/unobserved spatial means and every negative case stay in ROWS.','',
        '| Target | Recorded complete temporal query s | Capture/fit/audit/reinsertion s | GPU reinsertion s | Original spatial fit s | Offline CPU GT head s | Temporal peak allocated GiB |','|---|---:|---:|---:|---:|---:|---:|']
    for ds in stats:
        rr=[r for r in rows if r['dataset']==ds]
        mean=lambda group,key:float(np.mean([r[group][key] for r in rr]))
        peak=max(r['complete_temporal_cost']['CUDA_peak_allocated'] for r in rr)/2**30
        lines.append(f"| {ds} | {mean('complete_temporal_cost','complete_query_wall_seconds'):.4f} | {mean('complete_temporal_cost','complete_capture_fit_audit_reinsertion_seconds'):.4f} | {mean('complete_temporal_cost','full_reinsertion_seconds'):.4f} | {mean('original_spatial_cost','fit_GPU_seconds'):.4f} | {mean('offline_head_cost','actual_CPU_head_seconds'):.4f} | {peak:.3f} |")
    lines+=['','Timings are actual recorded wall measurements with observation recording, frozen capture, reinsertion and existing cache conditions; they are not cold deployment benchmarks. '
        'Original SpatialOPD fitting time is retained from its exact historical fit and no spatial fit is rerun for P6. The temporal query includes its actual recorded components once. '
        'No new DINO inference is performed in P6; original frozen packed expert evidence is hash-bound and fully read back. Offline CPU supervised cost is reported separately.','',
        'Four qualification queries have eight complete original-versus-recorder GPU fits with scientific tensors and predictions bitwise equal, source process hashes unchanged, '
        'then the first two formal fits per target must match their qualified recorded fits before acceptance. '
        'Root reads all64 complete GPU head math dictionaries, all64 complete original spatial dictionaries and all64 offline supervised dictionaries; '
        'the complete two-layer head backward, native-head-output VJP, every active AdamW state, source reset, last/shrunk output and full original spatial1792 chain are independently checked. '
        'The original CPU64 formulas use explicit float32 accumulation error bounds; they do not certify the full decoder Jacobian, CUDA transcendental kernels or future numerical/OOM safety. '
        'The first offline supervised CPU fits are compared with original unrecorded CPU fits bitwise; CPU/native-logit agreement uses declared accumulation bounds and interval equality, not a CPU/GPU-bitwise claim.','',
        'Actual complete counts: `'+json.dumps(state['counts'],sort_keys=True)+'`. Official/dense comparison is made for all64queries ×5arms ×3metrics, '
        'and all original input/prediction/receipt SHA/bytes are checked again after the full root scan.','',
        'Six distinct post-hoc cases per the fixed maximum/minimum temporal effect and least gradient cosine selection are illustrative, not efficacy estimates. '
        'The actual original spatial-action/sample-to-GT diagnostic has 8,320 comparisons. Old fits without saved GPU action tensors use the explicitly labeled offline float64 sigmoid '
        'of their saved sample logits; this is not the newer same-GPU-action reward audit. Empty GT-evaluable admission supports are explicitly reported. '
        'All six report PNG/PDF pairs and six original-pixel RGB case sheets were actually viewed. '
        'A display-only revision improves bar labels and selects five RGB positions including actual GT support; original plots/cases/RGB/GT geometry/predictions/scores remain intact. '
        'Green boxes show original official spatial GT; frames with no spatial GT are labeled honestly. Private RGB/query/GT geometry is never published.','',
        'Three unsealed CPU helper problems remain preserved with source/runtime/receipts: original P0 input interpretation, an unlaunched scorer signature binding, and the display-helper module path. '
        'The final runtime revision002 predates every P6 GPU fit; original scientific fits, inputs, parameters, old payloads and original receipts are unchanged. '
        'The display-only helper changes do not affect any prediction or scientific runtime.','',
        'The anonymous public package preserves every actual score, paired contrast, negative tail, cost, signal chain, configuration, source hash and root/visual receipt. '
        'Its portable audit reproduces scalar arithmetic and exact public byte bindings; it cannot recreate private media, GT arrays, prefix tensors or fits. '
        'Scientific conclusions remain conditional: existing temporal evidence can help, head adaptation is not uniformly superior to the direct projection, '
        'and the offline VidSTG diagnostic leaves a large gap. No new temporal branch or automatic route promotion is warranted. '
        'P6 phase closure requires successful remote byte verification and archive maintenance; whole-suite closure additionally requires all original P1–P6 closing receipts and a real FINAL_COMPLETION. '
        'EATA and every historical paused queue remain paused.']
    text='\n'.join(lines)+'\n';(out/'ACTUAL_ROOT_REVIEW.md').write_text(text);(ROOT/'docs/STVG_OPD_REVISED_P6_ROOT_REVIEW.md').write_text(text)
    print(json.dumps(dict(status='written',rows=64,complete_all_pairwise_results=True,root_sha256=sha(out/'ROOT_MATH_STATE_DENSE_READBACK.json'),visual_sha256=sha(BASE/'P6_ROOT_VISUAL_REVIEW.json'))))


if __name__=='__main__':run()

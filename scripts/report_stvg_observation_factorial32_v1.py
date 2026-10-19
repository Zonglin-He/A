"""Report every signed capability result; no method/step selection."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_observation_factorial32_v1 import D,P,ARMS,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def fmt(z):
    if z['mean']is None:return 'unknown'
    return f"{z['mean']:+.4f} [{z['CI95'][0]:+.4f}, {z['CI95'][1]:+.4f}]"

def main():
    verify(True);r=read(P/'RESULTS_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
    cost=read(P/'COST_BREAKDOWN.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json')
    obs=[]
    for f in bar['files']:
        z=load(ROOT/f['path']);s=load(ROOT/'artifacts/stvg_dino_endpoint_confirm32_v1/source'/(z['alias']+'.pt'))
        obs.append(dict(alias=z['alias'],dataset=z['dataset'],new_DINO=z['compute']['new_DINO'],
            original_admitted=len(s['expert']['anchors']['single4']),
            crop_admitted=sum(o.get('probe',{}).get('accepted',False)for o in z['spatial']['observations']),
            new_estimator_support=z['spatial']['estimate']['support_frames'],
            fallback_slots=len(z['spatial']['fallback_original_pool_positions']),
            full_image_crop_count=sum(o.get('full_image_crop',False)for o in z['spatial']['observations']),
            source_relative_E_reconstruction_exact=True))
    if (P/'OBSERVATION_COUNTS_PUBLIC.json').exists():assert read(P/'OBSERVATION_COUNTS_PUBLIC.json')==dict(rows=obs)
    else:put(P/'OBSERVATION_COUNTS_PUBLIC.json',dict(rows=obs))
    lines=['# Spatial RGB re-detection / temporal dense reading / factorial joint32','',
        'Status: completed / measurement valid. Original NH16/NV16 exposed development, one query per parent. This is an observation-capability experiment; no OPD fit, Fixed/Live comparison or trained new module was run. All three requested candidates were generated and sealed before any NEW GT scoring. No branch was skipped based on another branch result.','',
        'Frozen original TA-STVG cross-domain source checkpoints and same DINO-T. Original E, original four physical observation slots, target span/parsing, source calibration and evaluation rules retained. Formal FH/FV64 is excluded. Model checkpoint provenance and hashes are in MODEL_PROVENANCE.json.','',
        'S: original E proposes a 2x RGB context, clipped integer raster, then native DINO processor re-encodes and outputs actual detected boxes. Accepted crop pools replace the original same-slot pool; rejected/unsupported slots retain it. Original E is re-estimated relative to Native Source, never added twice to E. No confidence-energy coordinate gradient or IoU reward.','',
        'T: keep all original physical IDs and add one integer midpoint per adjacent pair. Same full temporal extent, original two offsets and positional-encoding limit. Actual video pixels and frozen appearance/motion/time decoder rerun. Actual FP32 logits form the original independent-offset envelope posterior; same physical expected-tIoU finalMBR. No logit/feature interpolation. E spatial boxes stay at original nodes and use original interpolation. The denser sequence also changes index-position and motion-window timescales; its effect cannot be attributed solely to newly observed RGB evidence.','',
        'ST combines exactly S boxes and T interval without extra calls; it measures factorial joint output, NOT joint conditional adaptation or OPD. GridOnly embeds the old posterior mass exactly in the dense decision grid, then MBR; it isolates expanded decision support, not dense model length/position effects.','',
        '## Main paired changes vs E, pp [parent bootstrap95%]','',
        '| Candidate | HC full vIoU | Vid full vIoU | HC tIoU | Vid tIoU | HC unobserved actual sIoU | Vid unobserved actual sIoU |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS[1:]:
        gg=[r['groups'][ds]['paired'][arm+'_minus_E_MBR']for ds in ['hc2','vidstg']]
        lines.append('| '+arm+' | '+' | '.join(fmt(v)for v in [gg[0]['v'],gg[1]['v'],gg[0]['t'],gg[1]['t'],gg[0]['unobserved_actual_s'],gg[1]['unobserved_actual_s']])+' |')
    lines+=['','## Dense posterior versus old-mass expanded-decision control','',
        '| Contrast | HC full vIoU | Vid full vIoU | HC tIoU | Vid tIoU |','|---|---:|---:|---:|---:|']
    gg=[r['groups'][ds]['paired']['T_dense_minus_T_gridonly']for ds in ['hc2','vidstg']]
    lines.append('| T_dense - T_gridonly | '+' | '.join(fmt(v)for v in [gg[0]['v'],gg[1]['v'],gg[0]['t'],gg[1]['t']])+' |')
    lines+=['','## Absolute metrics and actual time changes','',
        '| Target | Output | full vIoU % | tIoU % | fixed-support sIoU % | unobserved actual sIoU % | changed final intervals /16 |','|---|---|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        for arm in ARMS:
            m=g['metrics'][arm];lines.append(f"| {ds} | {arm} | {100*m['v']:.4f} | {100*m['t']:.4f} | {100*m['s']:.4f} | {100*m['unobserved_actual_s']:.4f} | {g['changed_time_counts'][arm]} |")
    lines+=['','Temporal-only actual s is exact E; S time exact E; ST time exact T and spatial exact S. Unknown GT stays unknown, no-reference/zero-overlap/no-change parents remain in the main denominator. Endpoint errors, lengths, zero overlaps, all signed changes and paired intervals are in RESULTS_PUBLIC.json.','',
        '## Factorial interaction, not evidence of learned cross-branch transfer','']
    for ds,g in r['groups'].items():lines.append(f"- {ds}: V_ST-V_S-V_T+V_E = {fmt(g['interaction_v_pp'])} pp. This includes the metric's temporal/spatial coupling.")
    lines+=['','## All signed full-v changes vs E, pp','',
        '| Alias | Target | S | T | ST | GridOnly |','|---|---|---:|---:|---:|---:|']
    for z in r['rows']:lines.append('| '+z['alias']+' | '+z['dataset']+' | '+' | '.join(f"{z['changes'][a]['v']:+.4f}"for a in ARMS[1:])+' |')
    refs=read(P/'REFERENCE_SEPARATION_PUBLIC.json')
    lines+=['','## Postseal reference-versus-propagation diagnostic','',
        'This diagnosis was added after observing the main output scores; no outputs or choices were changed. Frame IoUs are first averaged within each query, then paired across parents with known support. These conditional diagnostic parents are not the main16-parent denominators. Raw top boxes are not automatically E-selected references.','',
        '| Diagnostic, IoU pp | HC | Vid |','|---|---:|---:|']
    for ep in ['raw_crop_minus_raw_full_pp','raw_crop_minus_E_pp','accepted_crop_minus_E_pp']:
        gg=[refs['groups'][ds][ep]for ds in ['hc2','vidstg']]
        lines.append('| '+ep+' | '+' | '.join(fmt(x)+f" (n={x['parents']})"for x in gg)+' |')
    lines+=['',
        'The accepted crop-top versus E comparison has a positive HC conditional signal, but raw crop-top versus the old raw full-image top does not establish an improvement. This is not a new deployable tube result and does not establish improved supervision versus the old teacher. NV016 loses26.1278pp full-v under S; of its three known planned frames, two crop-top IoUs exceed E while all three propagated S boxes degrade. Candidate selection/aggregation/propagation therefore cannot be collapsed into the claim that every crop detection is bad. NV016 stays in the main result.','',
        '## Costs, validity and execution caveats','',
        f"New DINO image forwards: {cost['new_DINO']}; new dense TA queries:32 / native-offset full forwards64. Additional postseal engineering positive control:2 original-input TA queries /4 offset forwards, {cost['original_input_engineering_GPU_seconds']:.3f}s GPU. Total new GPU lease including loading/hash verification/engineering: {cost['GPU_allocation_seconds']:.3f}s. Source/E/full-image DINO cached costs excluded; this is not first-deployment latency. 0 new networks, heads, training or student updates.",'',
        'Original unresolved target conditions remain unresolved, so <=128 DINO calls is not a claim that every query spent four. All crop acceptance, original fallback, estimator support and unchanged full-image crops are itemized in OBSERVATION_COUNTS_PUBLIC.json. FRAME_DIAGNOSTICS_PUBLIC records raw target-score top boxes, including rejected ones; these are NOT automatically the final E estimator selections and not evidence of identity probability calibration.','',
        'A static review before inference required hard stopping and narrowed GridOnly attribution. Pre-execution amendment001 preserved the original code/protocol and installed an owned systemd RuntimeMaxSec880/TimeoutStopSec5 resource boundary inside the900s GPU budget. No scientific setting or prediction was changed. Additional helper/processor/tokenizer bindings were read back AFTER worker start; they are honestly supplemental, not claimed original pre-pins. All were unmodified before the plan, historical pins matched where available, and actual DINO weights matched the declared SHA. This qualification is disclosed rather than retroactively labelled a complete original lock.','',
        'Independent audit: '+str(audit['counters'])+'. Max errors: '+str(audit['max_errors'])+'. Baseline saved metrics, actual crop geometry/NMS/target scores, exhaustive Source-relative E association, physical envelope rectangles, direct MBR utilities, all paired statistics and factorial invariants passed. Numerical CDF corrections are recorded per query.','',
        'The unexpectedly large HC dense-view regression prompted two original-input engineering checks AFTER all outputs sealed and initial scoring. NH001/NV001 using the unchanged original physical input reproduce all cached temporal logits and Source boxes bit-exactly. No candidate, configuration, output or metric was altered, and no extra task scoring occurred. These are engineering controls, not additional method trials or pre-scoring tests; cost is included above. See ORIGINAL_INPUT_ENGINEERING_CHECK.json.','',
        '## Decision scope','',
        'These are development capability results. They do not establish OPD, no matter which direct output improves. A future same-evidence/equal-budget Direct/Fixed/Live study requires a separately fixed student state, physical probability transport and actual Live observation/candidate mechanism. A positive small mean/CI does not meet the2–3pp complete-cohort goal by itself. Negative recipes are retained, not repaired by scanning crop size, frame density or prompt. CURRENT/MAIN/PAPER and historical paused queues remain unchanged; no automatic follow-up.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['vg_tta/stvg_observation_factorial32_v1.py','scripts/run_stvg_observation_factorial32_v1.py',
        'scripts/audit_stvg_observation_factorial32_v1.py','scripts/report_stvg_observation_factorial32_v1.py',
        'scripts/check_stvg_observation_original_input32_v1.py','scripts/diagnose_stvg_observation_references32_v1.py','protocols/stvg_observation_factorial32_v1.md']
    files += [str((P/f).relative_to(ROOT))for f in ['CONFIG_PUBLIC.json','MODEL_PROVENANCE.json','RESULTS_PUBLIC.json','FRAME_DIAGNOSTICS_PUBLIC.json','OBSERVATION_COUNTS_PUBLIC.json','REFERENCE_SEPARATION_PUBLIC.json','COST_BREAKDOWN.json','ROOT_INDEPENDENT_AUDIT.json','ORIGINAL_INPUT_ENGINEERING_CHECK.json','REPORT.md']]
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=files),mutable=True)
    print('report and explicit allowlist written',len(files),flush=True)

if __name__=='__main__':main()

"""Report the finite HC experiment, preserving all signed and adverse results."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_hc_anchor_content16_v1 import D,P


def main():
    res=read(P/'RESULTS_PUBLIC.json');g=res['groups']['hc2'];cost=read(P/'COST_BREAKDOWN.json')
    assert read(P/'ROOT_MECHANISM_AUDIT.json')['status']=='pass'
    provenance=read(ROOT/'results/stvg_error_state/2026-10-19/observation_factorial32/MODEL_PROVENANCE.json')
    put(P/'MODEL_PROVENANCE.json',dict(TA=provenance['TA']['hc2'],DINO_sha256=provenance['DINO_sha256'],
        new_Source_or_expert_calls=0,pretrained_weights_and_LN_frozen=True,new_network_or_head_training=False))
    def pair(name,ep):
        x=g['paired'][name][ep];return f"{x['mean']:+.4f} [{x['CI95'][0]:+.4f},{x['CI95'][1]:+.4f}]"
    lines=['# HC independent anchor content states: E / Direct / Fixed / Live',
        '', 'Completed development experiment, all16 NH parents/one query each. Frozen original Vid-source EMA TA-STVG, original E+finalMBR/four planned physical nodes and cached crop/full-image DINO evidence. No new experts, DINO/Source encoding, learned head, pretrained weight/LN updates or new data. This is not independent confirmation.',
        '', '## Intervention and supervision',
        '', 'K independent256D temporary vectors (K<=4) are added only to their physical-frame content tokens before spatial block0 temporal self-attention. All six frozen blocks decode the complete original actual trajectory. No vector is broadcast over frames; two offset decoders have no cross-offset attention. Zero state exactly restores Source and E through the fixed center-difference/log-size E map. NH013 has anchors only in offset1; its unsupported offset0 stays exactly original, not claimed supervised.',
        '', 'Teacher: admitted cached crop target-score top1, otherwise admitted original target-score top1, otherwise missing; no E joint residual selection or GT-dependent support. All16/61 references remain, including harmful/unknown/time-wrong cases. Teacher target scores are not calibrated correctness probabilities.',
        '', 'Common localization cost is5L1(cxcywh)+2(1−GIoU), mean1/K. Direct minimizes this cost. Fixed/Live form detached softmax(−cost/.05) targets on32 antithetic native-logit Gaussian samples, then the original likelihood/control-variate gradient passes through the current actual decoder mean to the temporary states. It is an output-coordinate stochastic policy, NOT sampled content states/realizable stochastic decoder trajectories. Every updated deterministic full trajectory is genuinely decoded. This state/teacher/cost combination is a route comparison, not a single-factor causal comparison against old fixed-IoU1792 results.',
        '', 'Fixed holds the initial samples/teacher feedback while fitting its current likelihood; Live resamples around each updated mean and re-evaluates the same frozen localization cost. Teacher RGB/boxes do not refresh, so no new expert calls are needed. Both arms have identical first samples, targets, gradient and update. Original HC numerical configuration is locked:40 steps,lr.01,sigma.025,tau.05,M32,Adam(.9,.999),eps1e-8; no scan, best-step selection, rollback or writeback. Direct/Fixed/Live share the same state initialization and40-update allowance.',
        '', '## Main paired differences, pp [parent bootstrap95%]',
        '', '| Contrast | complete vIoU | unobserved actual sIoU | observed actual sIoU |',
        '|---|---:|---:|---:|']
    for name in g['paired']:
        lines.append('| '+name+' | '+' | '.join(pair(name,ep) for ep in ['v','unobserved_actual_s','observed_actual_s'])+' |')
    lines+=['','All16 parents are the main denominator; missing GT is unknown. All64 outputs/full update traces sealed before new HC GT access. Temporal intervals and tIoU are unchanged; unobserved-actual supervision is not inflated by interpolated frames.',
        '', '## Absolute means and anchor fitting', '', '| Arm | full vIoU % | actual-unobserved sIoU % | actual gradient updates | initial→final teacher cost, supported-query mean |', '|---|---:|---:|---:|---:|']
    for arm in ['E_MBR','Direct','Fixed','Live']:
        if arm=='E_MBR':updates=0;fit='—'
        else:
            ds=[r['fit_diagnostics'][arm]for r in res['rows']if r['references']]
            updates=sum(x['updates']for x in ds)
            fit=f"{sum(x['anchor_loss_initial']for x in ds)/len(ds):.6f}→{sum(x['anchor_loss_final']for x in ds)/len(ds):.6f}" if ds else 'unknown'
        lines.append(f"| {arm} | {100*g['means'][arm]['v']:.6f} | {100*g['means'][arm]['unobserved_actual_s']:.6f} | {updates} | {fit} |")
    lines+=['','A lower teacher cost confirms optimization, not task correctness or cross-frame generalization. Direct−E measures interface value; Live−Fixed measures this refresh configuration; Live−Direct measures value beyond direct supervision. One HC direction can establish a result without Vid positivity, but these repeatedly exposed16 parents cannot serve as confirmation.',
        '', '## Every signed result versus E, pp', '', '| Alias | refs | Direct v | Fixed v | Live v | Direct unobserved s | Fixed unobserved s | Live unobserved s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in res['rows']:
        vals=[100*(r['values'][arm][ep]-r['values']['E_MBR'][ep]) for ep in ['v','unobserved_actual_s']for arm in ['Direct','Fixed','Live']]
        lines.append(f"| {r['alias']} | {r['references']} | "+' | '.join(f'{v:+.6f}'for v in vals)+' |')
    lines+=['','All per-parent signs, leave-one sensitivity, severe tails, state movements, update counts, observation/offset maps and interpolation contributions are retained in RESULTS_PUBLIC.json. No favorable-parent filtering or new-source selection occurred.',
        '', '## Engineering, audits and costs', '', f"Incremental GPU lease including loading, engineering probe, checks and3s conservative guard: {cost['GPU_seconds']:.3f}s. Actual deterministic trajectory decoder calls {cost['trajectory_decoder_calls']}, offset PosDecoder calls {cost['PosDecoder_calls']}, state backwards {cost['backward_calls']}. New Source/DINO/expert calls0; new network/head training0. Old RGB/feature/decoder-input/DINO acquisition costs are excluded; this is not first-deployment latency.",
        '', 'Owned unit hard cap880s+5s stop inside900s GPU cap,MemoryHigh16G/Max20G/Swap1G,CPU400%,shared nonblocking gpu.lock and resource floors. Pre-execution001 repaired progress-file mutable handling and bound CPU-imported helper dependencies before GPU start; original code preserved, scientific settings unchanged. No scientific output was altered or rerun for scoring.',
        '', 'Mechanism audit independently checks physical offset mapping, zero states, paired E output geometry, candidate coordinates/localization costs/weights, Fixed initial target reuse/Live current-mean provenance, analytic gradients and CPU Adam replay. CPU/GPU elementary float32 operations use documented numerical bounds; deterministic zero-state comparisons remain exact. The audit is not another GPU/model replication. Postseal independent official/dense and actual-node metric/parent-statistic readback is reported separately.',
        '', '## Decision scope', '', 'Report every contrast rather than turning a local fit or direct improvement into OPD evidence. The2–3pp objective is complete-cohort full-v. If only Direct improves, credit the state/supervision interface. If Live improves over E but not Direct/Fixed, OPD-specific gain remains unestablished. A finite negative result concerns this layer0/K×256/cost/budget recipe, not every frame-conditioned state. No automatic layer/step/temperature/scale scan, new expert, new cohort or confirmation is started. CURRENT/MAIN/PAPER and historical paused queues remain unchanged.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('report written',flush=True)


if __name__=='__main__':main()

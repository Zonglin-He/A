"""Report measured scale sensitivity, without selecting a new online method."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_o11_scale_readout_v1 import OUT,verify
from scripts.audit_tastvg_o11_public_v1 import audit


def run():
    p=verify();s=read(OUT/'SUMMARY.json');rr=read(OUT/'ROWS.json');ref=read(OUT/'REFERENCE.json');sb=read(OUT/'SELECTION_BARRIER.json');ab=read(OUT/'AGREEMENT_BARRIER.json');sr=read(OUT/'SCORE_RECEIPT.json')
    assert sb['time']<=ab['time']<=sr['time'];assert sb['alpha1_exact'] and sr['old_inputs_unchanged'];verification=audit(OUT);write(OUT/'PUBLIC_AUDIT.json',verification)
    resources=dict(new_GPU_seconds=0,new_model_forwards=0,new_expert_calls=0,new_training_steps=0,new_online_trajectories=0,CPU_measured_stage_seconds={k:v['seconds'] for k,v in [('select',sb),('agreement',ab),('cached_metric_readback',sr)]},timing_excludes='Python imports, development, independent public audit, reporting and publication')
    resources['CPU_measured_stages_total_seconds']=sum(resources['CPU_measured_stage_seconds'].values());write(OUT/'RESOURCES.json',resources)
    conclusion='Amplification removes selection inertia, but none of alpha8/16/32 improves mean vIoU on the fixed24-arrival trajectory. Exact Full-Rerank agreement increases modestly and non-monotonically. Scale explains why the original readout barely moved; it is not a sufficient explanation for the missing task gain.'
    decision=dict(status='completed',measurement='audited',conclusion=conclusion,new_method_selected=False,normalization_rerun=False,production_changed=False,scope='fixed saved O1 arrival states only; not an LR sweep or a new online run',statistical_limit='All amplified paired t/v confidence intervals include zero; no established mean harm or benefit and no universal representation failure claim')
    write(OUT/'DECISION.json',decision)
    provenance=dict(source_commit=p['source_O1_commit'],source_experiment='O1 Sparse-Critic Online Transfer',student_checkpoint_sha256='5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83',expert_checkpoint_sha256='ac48fa4474f65a84f5176de5dbbf4439ebe7996a3fc361fe10161e9f338a1ba8',parents=24,source_stream_parents=32,source_expert_writes=8,source_lr=.001,alphas=p['alphas'],fixed_arrival_states=True,source_data='historically exposed VidSTG development, original source-hashed transient conditions/order',metrics='reused O1 dual-implementation-verified candidate scores; no new GT file read',original_inputs_unchanged=True)
    write(OUT/'PROVENANCE.json',provenance)
    lines=['# O1.1: Residual Scale Readout Test','',conclusion,'',
        'Only CPU readback was performed. Same24 non-expert arrivals, fixed saved O1 arrival weights,768D features, native scores and candidate order. S(alpha)=native_score+alpha*(phi@arrival_w), alpha in {1,8,16,32}. No learning-rate change, retraining, expert calls, backbone forward or online trajectory rerun.','',
        '| Alpha | Changed vs Frozen | Exact Full-Rerank agreement | tIoU (%) | vIoU (%) | Delta tIoU vs alpha1 (pp) | Delta vIoU vs alpha1 (pp) |','|---|---:|---:|---:|---:|---:|---:|']
    for a,z in s.items():lines.append(f"| {a} | {z['changed_from_frozen']}/24 | {z['full_agreement']}/24 ({z['full_agreement']/24*100:.2f}%) | {z['metrics']['tIoU']['mean']*100:.4f} | {z['metrics']['vIoU_corrected']['mean']*100:.4f} | {z['delta_alpha1']['tIoU']['mean']*100:+.4f} | {z['delta_alpha1']['vIoU_corrected']['mean']*100:+.4f} |")
    lines+=['',f"Frozen: tIoU {ref['Frozen']['metrics']['tIoU']['mean']*100:.4f}%, vIoU {ref['Frozen']['metrics']['vIoU_corrected']['mean']*100:.4f}%. Full Rerank reference: tIoU {ref['Full Rerank']['metrics']['tIoU']['mean']*100:.4f}%, vIoU {ref['Full Rerank']['metrics']['vIoU_corrected']['mean']*100:.4f}%. Full Rerank uses current-arrival expert information and is not a same-budget method or oracle.",'',
        '## Paired changes and scope','',
        '| Alpha | Delta tIoU 95% interval (pp) | Delta vIoU 95% interval (pp) | vIoU gains / losses / unchanged | vIoU harms >5pp |','|---|---:|---:|---:|---:|']
    for a,z in s.items():
        ci=lambda m:'['+', '.join(f'{x*100:+.4f}' for x in z['delta_alpha1'][m]['ci95'])+']'
        win=z['positive']['vIoU_corrected'];lose=z['negative']['vIoU_corrected'];lines.append(f"| {a} | {ci('tIoU')} | {ci('vIoU_corrected')} | {win} / {lose} / {24-win-lose} | {z['harms_gt5pp']['vIoU_corrected']} |")
    lines+=['',
        'All amplified tIoU/vIoU intervals include zero. They are descriptive paired bootstraps of these24 sealed outcomes (10000 samples, seed20260929), conditional on one already-exposed stream and shared fixed states. They do not estimate robustness over online trajectories or establish that amplification is generally harmful. No scale is promoted or selected for deployment.','',
        'Alpha8/16/32 move many predictions, but top1 agreement is only5/4/3 out of24. Relative to alpha1, newly matching Full choices number4/3/3; alpha32 also loses one previous match. Most moved choices therefore do not recover Full Rerank. Small tIoU gains at8/16 do not translate into tube gains; simply giving this learned residual more weight is insufficient in this screen.','',
        '## Preserve both useful and harmful changes','',
        '| Arrival / anonymous parent | Condition | Alpha1 t/v (%) | Alpha8 t/v (%) | Alpha16 t/v (%) | Alpha32 t/v (%) |','|---|---|---:|---:|---:|---:|']
    for pos in [15,16,18]:
        r=next(x for x in rr if x['position']==pos);vals=[' / '.join(f"{r['variants'][a]['metrics'][m]*100:.4f}" for m in ['tIoU','vIoU_corrected']) for a in ['1','8','16','32']]
        lines.append(f"| {pos} / Q{r['parent']+1:02} | {r['condition']} | "+' | '.join(vals)+' |')
    lines+=['',
        'Arrival18/Q12 moves toward Full Rerank and improves both metrics. Arrival15/Q15 also matches Full Rerank after amplification but loses task accuracy: critic agreement is not ground-truth correctness. Arrival16/Q07 becomes substantially worse at alpha32 without matching the Full choice. These are descriptive readbacks; all24 cases remain in ROWS.json.','',
        '## Verification and completion','',
        'All96 selections were sealed before consuming teacher-score values. Cached teacher scores were then used solely for agreement and sealed before consuming cached GT-derived metrics. Input hashing validates bytes but does not use those values to choose scales or candidates. Alpha1 exactly reproduces the O1 scores/selections. Independent math.fsum dot products and NumPy selection agree for all24/96; original input files and production registration retain their hashes. No raw GT file was reread: task values are gathered from O1\'s already dual-implementation-verified per-candidate metrics.','',
        f"Independent public audit reconstructs {verification['selections']} selections and {verification['bootstrap_scalar_checks']} means/intervals plus all agreement and harm counts. Maximum independent residual-dot error is {sb['max_independent_dot_error']:.3g}.",'',
        'This readout multiplier is NOT equivalent to multiplying the online learning rate: a different learning rate would change subsequent gradients and states. Results rule out a simple claim that these fixed learned directions only need more voting weight to yield gains at the tested scales. They do not establish that all linear representations, score normalization, slow-state mechanisms or Slow-Fast TTA are impossible.','',
        'Only O1.1 was executed. No normalization rerun, LR grid, prototype, memory, reliability gate, spatial experiment or new deployment setting was added. The next mechanism remains undecided; production stays unchanged.','',
        f"New GPU/model/expert/update work:0. Measured selection/agreement/metric-readback CPU stages total {resources['CPU_measured_stages_total_seconds']:.6f}s, excluding Python imports, development, independent audit, reporting and publication. Raw weights/features/media stay local; anonymous scores, all four readouts and task values are exported.",'',
        'Reproduce: scripts/run_tastvg_o11_scale_readout_v1.py prepare -> select -> agreement -> score; scripts/report_tastvg_o11_scale_readout_v1.py. Public scalar readback: scripts/audit_tastvg_o11_public_v1.py <result-directory>.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');write(OUT/'COMPLETION.json',dict(status='completed',files={n:sha(OUT/n) for n in ['REPORT.md','ROWS.json','SUMMARY.json','REFERENCE.json','PUBLIC_AUDIT.json','DECISION.json','RESOURCES.json','PROVENANCE.json']},time=time.time()))
    print(conclusion);print(verification)

if __name__=='__main__':run()

"""Bounded O3 outcome and truthful scope reconciliation after user steering."""
import sys,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_tastvg_conditional_opd_o3_v1 import OUT,OLD,verify,MODES
from scripts.audit_tastvg_conditional_opd_public_v1 import audit


def run():
    p=verify();rows=read(OUT/'ROWS.json');s=read(OUT/'SUMMARY.json');diag=read(OUT/'DIAGNOSTICS.json');non=[r for r in rows if not r['expert']]
    conclusion='Both conditional students preserve all60 nonexpert native choices in this fixed first recipe; neither yields task transfer. They tie on these outputs, while saved O2 Linear Slow retains its small positive gain. Loss reduction and verified parameter updates do not establish cross-query benefit.'
    write(OUT/'PUBLIC_AUDIT.json',audit(OUT))
    resource=dict(new_GPU_seconds=0,new_backbone_forwards=0,new_expert_calls=0,new_optimizer_steps=40,expert_arrivals=20,logical_expert_calls_per_arm=20,reused_capture_cells=80,online_CPU_seconds=read(OUT/'ONLINE_BARRIER.json')['seconds'],independent_numpy_audit_seconds=read(OUT/'STATE_AUDIT.json')['seconds'],historical_cache_creation_excluded=True,states_bytes=sum(f.stat().st_size for f in (OUT/'states').glob('*.pt')))
    write(OUT/'RESOURCES.json',resource)
    write(OUT/'DECISION.json',dict(status='completed',measurement='audited',conclusion=conclusion,preferred_conditional_loss_per_user_tie_rule='reverse_kl',preference_is_implementation_choice_not_utility_validation=True,online_benefit_established=False,production_changed=False,no_hyperparameter_search=True,followon='none in this turn; S0 deferred by latest temporal-only scope',KNN='completed CPU measurements before user steering; superseded, not compared or selected against new O3'))
    provenance=read(OLD/'PROVENANCE.json');provenance.update(architecture=[768,128,1],parameters=98561,activation='ReLU',initialization='seed20260929 first layer Kaiming, zero output weight/bias',lambda_residual=1.,tau_E=1.,tau_S=1.,optimizer='SGD',lr=.001,steps_per_expert=1,replay_capacity=4,max_prior_sets_used=3,replay_weight=1.,source_occurrences=5,source_repetitions_after_first=4,source_repetitions_field_note='Legacy O2 source_repetitions=5 meant total occurrences, not additional repeats',backbone_proposal_policy='frozen TA-STVG; rankings adapt, proposal support not regenerated',global_context='No extra global vector concatenated; use final requested768D query-conditioned hidden features')
    write(OUT/'PROVENANCE.json',provenance)
    learning={}
    for a in MODES:
        er=[r['diagnostics'][a] for r in rows if r['expert']];res=[v['residual_range'] for c in p['conditions'] for v in diag[c][a]['readouts']]
        learning[a]=dict(total_loss_decreased=sum(z['after']['total']<z['before']['total'] for z in er),current_loss_decreased=sum(z['after']['current']<z['before']['current'] for z in er),replay_loss_increased=sum(z['after']['replay']>z['before']['replay']+1e-12 for z in er),median_residual_range=float(np.median(res)),maximum_residual_range=max(res),nonexpert_changed=sum(r['selected'][a]!=0 for r in non))
    write(OUT/'LEARNING.json',learning)
    lines=['# O3: Conditional Online OPD','',conclusion,'',
      '80 arrivals on exactly the saved O2 five coherent16-source streams;20 specialist arrivals,60 nonexpert cells representing12 unique nonexpert parents repeated across five conditions. Compare saved Linear Slow and two new matched MLP students with Budgeted Rerank. All expert-position outputs are identical.','',
      '| Arm | Nonexpert macro tIoU (%) | Nonexpert macro vIoU (%) | Delta t vs Budgeted (pp) | Delta v vs Budgeted (pp) | Nonexpert choices changed |','|---|---:|---:|---:|---:|---:|']
    for a in ['Budgeted Rerank','Linear Slow',*MODES]:
        z=s['macro']['nonexpert']['arms'][a];b=s['macro']['nonexpert']['arms']['Budgeted Rerank'];changed=sum(r['selected'][a]!=r['selected']['Budgeted Rerank'] for r in non)
        lines.append(f"| {a} | {z['tIoU']['mean']*100:.4f} | {z['vIoU_corrected']['mean']*100:.4f} | {(z['tIoU']['mean']-b['tIoU']['mean'])*100:+.4f} | {(z['vIoU_corrected']['mean']-b['vIoU_corrected']['mean'])*100:+.4f} | {changed}/60 |")
    lines+=['','## Per-condition primary differences','', '| Condition | Linear delta t/v (pp) | Pairwise delta t/v (pp) | Reverse-KL delta t/v (pp) |','|---|---:|---:|---:|']
    for c in p['conditions']:
        parts=[]
        for a in ['Linear Slow',*MODES]:
            z=s[c]['nonexpert']['comparisons'][a+' - Budgeted Rerank'];parts.append('/'.join(f"{z[m]['mean']*100:+.4f}" for m in ['tIoU','vIoU_corrected']))
        lines.append('| '+c+' | '+' | '.join(parts)+' |')
    lines+=['','All60 new-student versus Budgeted t/v differences are exactly zero, so their descriptive bootstrap intervals are[0,0]. This is exact identity on these sealed choices, not statistical proof of population equivalence. Reverse-KL minus Pairwise is also zero. Macro averages five conditions per source before bootstrapping12 parent clusters (10000 samples, seed20260929);80/all uses16 parents. Shared state trajectories are not rerun.','',
      '## What was actually trained','',
      'Final requested architecture768->128->1 with ReLU,98,561 parameters. Input is the original raw candidate start/end/span-mean hidden state, already conditioned by the multimodal backbone. No extra global g or score concatenation was added because that would change the explicitly requested768D input. Native score enters s=native+MLP(phi), lambda1. Initial output layer is exactly zero; identical fixed seed first-layer initialization for both losses and every reset.','',
      'One plain SGD step at LR .001 per expert,40 new updates total. Same update frequency and step count as O2. Replay uses up to4 prior expert sets with current loss + mean(past loss), weight1. Each stream has only4 expert arrivals, so at most3 prior sets are used; eviction is not exercised. Current expert output is emitted before update, and current evidence joins replay afterward. Nonexpert arrivals read neither teacher nor labels.','',
      'Pairwise uses the same strict-preference logistic loss. Reverse-KL is sum p*(log p-log q), with p=softmax(student scores) and q=softmax(expert scores), both temperatures1. These fixed choices are not calibrated or outcome-tuned. Reverse-KL still depends on teacher score scale and temperature.','',
      'Pairwise versus Reverse-KL differs only in objective. MLP versus saved Linear also differs in replay and initialization, so it is a recipe comparison, not a pure architecture attribution. Distillation uses current-input proposals generated by the frozen student backbone; it is not a test of regenerated proposal support under a fully adapted policy.','',
      '| Student | Total objective down | Current objective down | Median residual score range | Maximum residual score range |','|---|---:|---:|---:|---:|']
    for a,z in learning.items():lines.append(f"| {a} | {z['total_loss_decreased']}/20 | {z['current_loss_decreased']}/20 | {z['median_residual_range']:.8f} | {z['maximum_residual_range']:.8f} |")
    lines+=['','Both total objectives decrease20/20, current-only decreases19/20; replay can trade off the current fit. Hidden first-layer gradients are zero at the first zero-head update by construction, then nonzero in30/30 later updates across arms/streams. The implementation is learning, but the readout remains native at all60 evaluated nonexpert arrivals. Four steps per stream and these small residuals limit any claim about the conditional representation or reverse-KL itself. No scale/LR/steps rescue was run.','',
      '## Cases, decision and scope','',
      'Every new-student case is task-identical to Budgeted: zero gains, zero losses and no>5pp harms. The two original Linear positives remain visible: Q09 frame-drop (v5.1749->5.4316%) and Q16 occlusion (v10.6912->13.1584%); both conditional students retain the baseline on those cases. All candidate metrics and all80 decisions remain in ROWS.json; no successful old case is removed.','',
      'Following the user\'s tie preference, Reverse-KL is retained as the preferred CONDITIONAL IMPLEMENTATION CANDIDATE. This is not promotion, a useful-transfer claim, or evidence it is better than Pairwise. The validated current correction reference remains sparse Fast reranking. Production CURRENT is unchanged.','',
      '## Verification and resources','',
      'Independent NumPy analytic gradients reconstruct80 two-model forwards and40 SGD updates,5 reset pairs, cached replay chronology and all hashes; gradient max error6.08e-16, state8.67e-19, loss7.22e-16. Original three MLP contracts pass. Public audit rebuilds540 mean/interval statistics plus every selection, candidate metric lookup, matched control and replay/reset chain.','',
      f"New GPU/backbone/expert calls:0. Reuse80 captures and20 sparse expert results. Online CPU loop {resource['online_CPU_seconds']:.3f}s; independent analytic audit {resource['independent_numpy_audit_seconds']:.3f}s. Historical cache generation and development/reporting time excluded. State artifacts {resource['states_bytes']} bytes.",'',
      'New online predictions were sealed before reading cached O2 GT-derived candidate metrics or saved Linear outcomes. No new raw GT file was loaded; old inputs and production pins still match. Exposed development sources, one order per condition, and four updates per stream are limitations.','',
      '## Superseded work and deferred S0','',
      'Before the new steering arrived, the old KNN-memory proposal had already finished CPU measurements on the DIFFERENT O1 heterogeneous32-source stream. Its24 nonexpert delta t/v was+2.0778/+0.4862pp with intervals crossing zero;23 choices changed,9 v gains/11 losses/4 unchanged,one>5pp harm. It is archived separately as superseded, not rerun or selected against this O3.','',
      'Latest instruction is temporal only. Official Sa2VA-4B source/dependencies were prepared, but its weight download was stopped during partial transfer; no S0 model inference or spatial expansion ran. Partial downloads and isolated dependencies remain for future continuation. The user\'s preference for a real RVOS specialist is retained.','',
      'Reproduction: run_tastvg_conditional_opd_o3_v1.py prepare/online -> audit_tastvg_conditional_opd_o3_v1.py -> score_tastvg_conditional_opd_o3_v1.py -> report_tastvg_conditional_opd_o3_v1.py. Public scalar audit: audit_tastvg_conditional_opd_public_v1.py <result-directory>.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write(OUT/'COMPLETION.json',dict(status='completed',files={n:sha(OUT/n) for n in ['ROWS.json','SUMMARY.json','REPORT.md','STATE_AUDIT.json','PUBLIC_AUDIT.json','DECISION.json','RESOURCES.json','PROVENANCE.json']},time=time.time()))
    print(conclusion)

if __name__=='__main__':run()

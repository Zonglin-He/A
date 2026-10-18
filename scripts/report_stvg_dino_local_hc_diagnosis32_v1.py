"""Report finite cached diagnosis without selecting a new algorithm."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.analyze_stvg_dino_local_hc_diagnosis32_v1 import D,P,verify

def main():
    p=verify();x=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    response=read(P/'TEACHER_RESPONSE_DISPERSION.json')['summary']
    lines=['# HC local residual decline: cached teacher, aggregation and amplitude diagnosis','',
       'This analysis uses the same exposed HC16/Vid16. No new prediction, GT read/scoring, model/expert call, training, parameter search or amplitude trial. All numbers derive from saved 505103c1 feedback/IoU scalars and d9f0c700 output metrics.','',
       '## Teacher feedback','',
       '| Query-mean gated quality difference, pp | HC (16 known-support queries) | Vid (10 active known-support queries) |','|---|---:|---:|']
    for k in ['teacher_minus_input_gated_pp','teacher_minus_E_gated_pp','E_minus_input_gated_pp']:
        cells=[]
        for ds in ['hc2','vidstg']:
            v=x['groups'][ds]['quality'][k];cells.append(f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]")
        lines.append('| '+k+' | '+' | '.join(cells)+' |')
    lines+=['',
      'At fixed known nodes, teacher-minus-input = teacher-minus-E + E-minus-input exactly. HC +1.244129 = +0.018568 +1.225561 pp. Thus positive feedback relative to sampled inputs is mostly a restoration of perturbation damage in this arithmetic comparison, not established additional quality above E. This is not a causal percentage or an estimate of new deployment gain.','',
      'HC has 133 accepted feedback instances with saved GT quality: 64 improve over E, 66 worsen and 3 tie. Another42 accepted instances have no saved GT quality and stay unknown. These are repeated candidate-anchor instances, not independent observations or identity labels.','',
      '## Actual residual aggregation','',
      'For each node, r = sum(m*(T-E))/24 - sum(m*(input-E))/24, in center/log-size coordinates. Both terms are measured exactly. Dividing by24 does not cancel the admitted inputs: admission varies with sampled state and the finite admitted subset need not have zero mean. No-feedback nodes still have exactly zero residual; this is not the old no-feedback random-mean update bug.','',
      '| Finite-set property | HC | Vid |','|---|---:|---:|']
    keys=['nodes_with_feedback','accepted_total','input_norm_larger_than_teacher','applied_opposite_teacher','admission_contrast_larger_than_all24_mean','mean_coherence_query_mean','leave_one_can_reverse_direction']
    for k in keys:lines.append(f"| {k} | {x['groups']['hc2'][k]} | {x['groups']['vidstg'][k]} |")
    lines+=['',
      'Vector norms and cosines use the previously frozen source scales. HC input-displacement norm exceeds the teacher-displacement norm in27/33 supported nodes, versus3/33 in Vid. The actual HC residual points opposite the mean teacher-displacement term in8/33 such nodes (1/33 Vid). This explains why applying the average teacher correction at E is not equivalent to adopting the teacher output. These are geometrical facts, not GT directional derivatives.','',
      'For the admitted-input term, p*mean(all24 input displacement) and p*(mean(admitted)-mean(all24)) sum exactly. The latter has larger norm in30/33 HC supported nodes. This is a finite-set admission contrast; it does not establish a population bias or isolate admission causality.','',
      'HC corrections have higher mean coherence than Vid, not lower. Fixed-denominator leave-one-term analysis produces no genuine direction reversal in either domain after roundoff guards. A single term can exceed the net HC residual magnitude at one node, but this alone does not explain task loss. The evidence does not support a generic claim that averaging24 samples destroys the signal through cancellation.','',
      '## Applied amplitude','',
      '| Geometric amplitude / response | HC | Vid |','|---|---:|---:|']
    for k in ['median_accepted_per_supported_node','mean_center_relative_E_size_all64','max_center_relative_E_size','max_log_size_norm','median_teacher_input_dispersion_ratio','median_correction_on_input_slope']:
        lines.append(f"| {k} | {response['hc2'][k]:.9f} | {response['vidstg'][k]:.9f} |")
    lines+=['',
      'The center is normalized by the E box width/height. HC mean center change over all64 planned nodes is about0.006863 in these units; the maximum is0.054752. Its maximum log-size change norm is0.070414. HC changes are not globally larger than Vid. Original d9f0c700 mean image-center/log-size displacement is HC0.002837/0.014676 versus Vid0.004007/0.020298. This does not prove every HC step is safe: a small change can have the wrong direction or affect important interpolated frames.','',
      'Across the29 HC supported nodes with at least two accepted samples, the median teacher-to-input dispersion ratio is0.065841 and the correction-on-input slope is -0.986997. In these admitted same-frame states the teacher mostly returns nearby outputs while correcting input variation. This does not prove it ignores reference boxes or establish identity correctness. The corresponding Vid values are0.025655 and -0.990565, so convergence to similar outputs is not by itself a domain-specific failure; the value of those outputs relative to E differs.','',
      'This update is an external closed-form coordinate operation, not a QR/Adam/decoder-backprop step. The applied residual exactly matches the saved arithmetic. Old decoder-response failures are not evidence that this output-layer update overshoots.','',
      'No alternative amplitude, GT derivative or checkpoint was evaluated. Therefore this study cannot determine whether shrinking the same direction would rescue each case, and it does not select a learning rate. The priority supported by these data is distinguishing useful E-improvement feedback from corrections of sampled-input perturbations, not assuming all failures are overshoot.','',
      '## Per-query HC accounting','',
      '| Alias | Teacher-input gated pp | Teacher-E gated pp | Actual observed sIoU delta pp | Complete vIoU delta pp | Interpolated v contribution pp |','|---|---:|---:|---:|---:|---:|']
    for r in x['rows']:
        if r['dataset']!='hc2':continue
        q,o=r['quality'],r['outcome'];lines.append(f"| {r['alias']} | {q['teacher_minus_input_gated_pp']:+.6f} | {q['teacher_minus_E_gated_pp']:+.6f} | {o['delta_observed_s_pp']:+.6f} | {o['delta_v_pp']:+.6f} | {o['interpolation_v_contribution_pp']:+.6f} |")
    lines+=['',
      'H061: teacher repairs sampled inputs by+5.682835pp but is -0.229635pp below E in gated quality. All four supported nodes have an admitted-input norm larger than the teacher-displacement norm; actual observed sIoU loses1.438159pp and complete vIoU loses0.171277pp.','',
      'H031: teacher quality exceeds E by+0.803507pp in this accounting, yet the actual local update loses1.125214pp observed sIoU. At the second supported node, the teacher is near E (gated+0.118338pp) while the admitted-input term is much larger. Positive average teacher quality alone does not validate transporting its correction to E.','',
      'H045: the only known-GT accepted correction is one sample at the fourth node. That teacher is about11.058pp above E (accepted-only, before dividing24), but its correction applied at E points opposite the E-to-teacher mean in source-scaled coordinates (cosine about -0.637). No new alternate prediction is scored.','',
      'H032: observed sIoU changes only -0.018543pp, while complete vIoU loses0.161724pp;0.160997pp of the latter comes from official interpolation. This is the saved endpoint decomposition, not evidence that actual unobserved decoder nodes changed.','',
      'Keep positive controls: H008 gains0.056928pp complete vIoU despite its gated teacher-minus-E being -0.104320pp; H043 gains0.016504pp while that quantity is -1.503746pp. The residual transport can move differently from the teacher endpoint. These counterexamples rule out explaining all cases by a single teacher-better/worse threshold.','',
      '## Scope and decision','',
      'Saved sampled-GT quality is available for43/64 HC planned nodes; actual teacher feedback occurs at25 known and8 unknown nodes. Unknown feedback is not counted as failure. All16 HC queries remain in complete/observed endpoints; zero gated contribution means no observed accepted contribution at known nodes, not a correctness label for unknown feedback. Vid teacher-quality comparison uses10 active known queries; all16 remain in complete vIoU.','',
      'The previously measured HC decline remains -0.030264pp complete vIoU and -0.291001pp observed actual sIoU; unobserved actual nodes remain exact E. Teacher added value above E is not established. Finite admitted-input subtraction materially changes the average correction; simple direction cancellation is not the dominant evidence, and overshoot remains unisolated. Do not adjust threshold, candidate count or step size from these diagnostic GT values. No dynamic OPD, expanded cohort or method promotion.','',
      'Engineering recovery records are retained locally: JSON conversion for NumPy scalars and roundoff guards for deleting the sole residual term. All original inputs, predictions and GT scores are unchanged; no model or scientific prediction was rerun.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Anonymous chart; no raw coordinates, text, images or labels.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hc=[r for r in x['rows']if r['dataset']=='hc2'];fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
    ix=list(range(16));axes[0].bar(ix,[r['quality']['teacher_minus_input_gated_pp']for r in hc],label='Teacher minus sampled input',color='#b9c5db')
    axes[0].scatter(ix,[r['quality']['teacher_minus_E_gated_pp']for r in hc],label='Teacher minus E',color='#b14c48');axes[0].set_ylabel('Cached gated quality (pp)');axes[0].legend()
    axes[1].bar(ix,[r['outcome']['delta_v_pp']for r in hc],color='#4d758d');axes[1].set_ylabel('Complete vIoU delta vs E (pp)');axes[1].set_xticks(ix,[r['alias']for r in hc])
    for ax in axes:ax.axhline(0,color='black',lw=.6)
    fig.suptitle('HC original16: sampled-input repair is not established improvement over E');fig.tight_layout();fig.savefig(P/'HC_ACCOUNTING.svg');fig.savefig(D/'HC_ACCOUNTING.png',dpi=150);plt.close(fig)
    put(P/'CONFIGURATION.json',dict(count=32,HC=16,Vid=16,backbone='TA-STVG',same_source_checkpoints_as_d9f0c700=True,
        source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/dino_local_residual32/CONFIGURATION.json')['source_checkpoints'],
        threshold=.35,denominator=24,time='unchanged final-MBR',no_new_GT_or_scoring=True,no_new_models=True,no_amplitude_search=True,
        norm='existing source scales; applied center normalized by E width/height',old_target_exposure=True))
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=['scripts/analyze_stvg_dino_local_hc_diagnosis32_v1.py','scripts/audit_stvg_dino_local_hc_diagnosis32_v1.py','scripts/report_stvg_dino_local_hc_diagnosis32_v1.py',
        'protocols/stvg_dino_local_hc_diagnosis32_v1.md']+[str((P/n).relative_to(ROOT))for n in ['RESULTS_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','TEACHER_RESPONSE_DISPERSION.json','CONFIGURATION.json','REPORT.md','HC_ACCOUNTING.svg']]))
    print(P/'REPORT.md',flush=True)

if __name__=='__main__':main()

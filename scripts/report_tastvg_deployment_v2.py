"""Render completed C0.5/C2-T evidence with explicit population and scope."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write
OUT=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2'

def fmt(x):
    if x['mean'] is None:return 'NA'
    a,b=x['ci95'];return f"{x['mean']*100:+.3f} [{a*100:+.3f}, {b*100:+.3f}]"

def run():
    c0=read(OUT/'analysis/C05_SUMMARY.json');c2=read(OUT/'analysis/C2_SUMMARY.json');ex=read(OUT/'analysis/C2_EXCESS.json');dec0=read(OUT/'C05_DECISION.json');dec2=read(OUT/'C2_DECISION.json')
    resources={stage:sum(read(f)['seconds'] for f in (OUT/f'{stage}_allocations').glob('*.json')) for stage in ['c05','c2']}
    resources['superseded_GT_center_smoke_seconds']=10.352242434979416
    resources['total_including_superseded_seconds']=sum(resources.values());write(OUT/'RESOURCES.json',resources)
    rows=read(OUT/'analysis/C05_ROWS.json');rec=read(OUT/'analysis/C2_RECOVERY.json');p=c0['panel'];t=c2['corrupted_parent_macro'];ov=read(OUT/'analysis/C05_OVERLAP.json')
    lines=['# TA-STVG: deployment corruption and temporal critic qualification','',
      'Completed, frozen-only development experiments. No adaptation, OPD, spatial critic, S1, cross-domain tuning, or production promotion.', '',
      f"C0.5 benchmark-development gate: **{dec0['benchmark_development_gate']}**. C2-T next-OPD resource gate: **{dec2['OPD_resource_gate']}**. These are finite resource decisions, not universal mechanism claims.",'',
      '## Setting and protocol correction','',
      'The user corrected the initial GT-centered proposal before the full experiment. Three unscored black-frame smoke cells (10.352 s) from that earlier protocol are preserved and excluded. The primary protocol is now **Transient Deployment Corruption**, independent of query and GT.', '',
      'For each full physical source video, one deterministic uniform random start and ceil(1/5/10% of full-video frames) duration define a burst. Only the original observed frames falling in the burst are changed. The seed depends on source identity only; changing a query or its observed clip does not change the burst. No re-sampling to hit an event, no result-dependent severity changes. Frame drop uses black missing-frame placeholders with unchanged timestamps; freeze repeats the exact preceding original frame. Blur, 25%-area occlusion and +100 RGB exposure use fixed registered operators. These are simulated deployment faults, not an official THUMOS14-C reproduction or measured real fault frequencies.', '',
      'The original roster contains 32 previously exposed VidSTG-test parents, one query per parent, official VidSTG-source TA-STVG checkpoint (5ab12c86…), original 20–200 observed frames, two native offsets and precision. GT is used only after sealing predictions for metrics and retrospective overlap analysis. Old persistent noise/defocus/JPEG remains unchanged. No fresh confirmation claim.', '',
      '## C0.5: fixed five-family, three-duration panel','',
      '480 corrupted cells plus 32 clean references. Values are corrupted-minus-clean percentage points; brackets are paired parent-bootstrap 95% intervals. Primary averages all15 cells within each parent before bootstrapping32 parents.', '',
      '| Condition | Delta sIoU (pp) | Delta tIoU (pp) | Delta corrected vIoU (pp) | No observed hit |','|---|---:|---:|---:|---:|']
    order=['panel']+[f'{f}_{s}' for f in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure'] for s in [1,5,10]]
    for n in order:
        z=c0[n];lines.append('| '+n+' | '+' | '.join(fmt(z['delta'][m]) for m in ['sIoU','tIoU','vIoU_corrected'])+f" | {z['zero_observed_hit']}/{z['cells']} |")
    lines+=['',f"Primary qualification requires both delta-tIoU and delta-vIoU 95% upper bounds below zero: **{dec0['benchmark_development_gate']}**. Individual-condition patterns are descriptive; no worst-condition winner is promoted.",'',
      'The full-video burst may fall outside the student query clip or between observed frames. Those cases remain in the primary denominator. Physical dose and actually observed/changed frames are saved per cell. `C05_OVERLAP.json` groups retrospective GT-event overlap; these composition-varying groups are associations, not a causal estimate of overlap.', '',
      f"Retrospective >10%-GT-event-overlap group: {ov['high_gt10pct']['cells']} cells / {ov['high_gt10pct']['parents']} parents, delta-tIoU {fmt(ov['high_gt10pct']['delta']['tIoU'])} pp and delta-vIoU {fmt(ov['high_gt10pct']['delta']['vIoU_corrected'])} pp. This subset cannot replace the primary population or guide burst re-generation.", '',
      '## C2-T: fixed original candidate panel','',
      'This iteration deliberately reuses the original C1 first16 parents x clean/noise_medium/defocus_medium/jpeg_medium (64 cells), not C0.5 transient candidates. The student candidates and spatial boxes are unchanged. UniversalVTG best.pth with PE-Core-L14-336 is frozen, unifier disabled, and sees the same sampled corrupted pixels. Existing observations are mapped to physical 2fps slots. One cached expert pass per cell; no extra teacher-generated interval can become the final output.', '',
      'The registered critic score is max over pre-NMS expert proposals of confidence times temporal IoU with each student interval. This is a specific proposal-to-candidate bridge, not a calibrated native candidate likelihood. A negative result cannot establish that all expert scoring rules fail. No GT tunes the bridge; score ties keep the native-first candidate.', '',
      '| Condition | Native tIoU (%) | Expert tIoU (%) | Oracle tIoU (%) | Uniform tIoU (%) | Expert delta-tIoU (pp) | Expert delta-vIoU (pp) |','|---|---:|---:|---:|---:|---:|---:|']
    for n in ['clean','noise_medium','defocus_medium','jpeg_medium','corrupted_parent_macro']:
        z=c2[n];lines.append('| '+n+' | '+' | '.join(f"{z['arms'][a]['tIoU']['mean']*100:.3f}" for a in ['Native','Expert','Oracle','Uniform'])+' | '+fmt(z['delta']['Expert']['tIoU'])+' | '+fmt(z['delta']['Expert']['vIoU_corrected'])+' |')
    lines+=['',f"Paired excess gain (corrupted minus clean): tIoU **{fmt(ex['tIoU'])} pp**, corrected vIoU **{fmt(ex['vIoU_corrected'])} pp**. Clean/corrupt headroom must not automatically be called corruption recovery.",'',
      f"Critic minus exact uniform-candidate expectation on corrupted parent macro: tIoU {fmt(t['expert_minus_uniform']['tIoU'])} pp; vIoU {fmt(t['expert_minus_uniform']['vIoU_corrected'])} pp.", '',
      '| Pair group | Clean accuracy (%) [95% CI] | Corrupted parent-macro accuracy (%) [95% CI] | Corrupted valid pairs |','|---|---:|---:|---:|']
    for b in ['all','low','middle','high']:lines.append(f"| {b} | {fmt(c2['clean']['pair_accuracy'][b])} | {fmt(t['pair_accuracy'][b])} | {t['pair_counts'][b]} |")
    lines+=['',
      'GT ties are excluded; critic ties score0.5. Margin terciles are locked using all64 cells before scoring. Pair accuracy is averaged within cell and then within parent; CI resamples parents. High-margin accuracy is a diagnostic reliability signal, not a calibrated deployment gate.', '',
      f"Corrupted expert >5pp harm cells: tIoU {t['expert_harms_gt5pp']['tIoU']}/48, corrected vIoU {t['expert_harms_gt5pp']['vIoU_corrected']}/48. Worst changes: tIoU {t['expert_min_delta']['tIoU']*100:.3f} pp, corrected vIoU {t['expert_min_delta']['vIoU_corrected']*100:.3f} pp. Native retained {t['native_kept']}/48.", '',
      f"Among the {len(rec)} original medium-corruption cells with >5pp temporal damage, the critic restores clean-level tIoU in {sum(x['restored_clean'] for x in rec)}. This is a small descriptive subset; all positive and negative examples are retained in C2_RECOVERY.json.", '',
      '## Decision, reproducibility and resource scope','',
      f"No OPD is implemented in this round. The predeclared next-OPD gate requires positive corrupted tIoU gain CI and high-margin accuracy CI above0.5; observed gate={dec2['OPD_resource_gate']}. Benchmark gate={dec0['benchmark_development_gate']}. Preserve the separation among corruption failure, student support, and critic error.", '',
      'Three CPU semantic tests cover locality, deterministic severity, full-video query-independent bursts, and candidate-only ranking. New clean smoke is bitwise identical to old Frozen. Independent readback checks source-only burst generation, no-hit clean identities, scalar bootstrap values, critic scoring and selection, old-C1 oracle parity, and dual implementation s/t/v metrics. Model state hashes are unchanged. No backward or adaptation.', '',
      f"GPU-process wall time: C0.5 {resources['c05']:.3f}s; C2-T {resources['c2']:.3f}s; superseded smoke10.352s; total {resources['total_including_superseded_seconds']:.3f}s. Includes loading, input preprocessing, audit hashing within workers and any recorded failed allocations; excludes CPU coding/scoring/report work. Artifact cap8GiB, free-space floor8GiB, per-phase GPU-process cap3600s.", '',
      '## Source basis and implementation differences','',
      '[Zeng et al., CVPR2024](https://arxiv.org/abs/2403.20254) motivates temporal corruption; [official configuration and operators](https://github.com/Alvin-Zeng/temporal-robustness-benchmark/tree/a46eee452222fa67958c81c49496e712dedefeea/extract_corrupted_feature_code/i3d/thumos) were inspected. Their action-centered GT-conditioned generation is a diagnostic reference, not our primary deployment protocol. Our blur/occlusion/drop/freeze operators and sampling are explicitly registered adaptations.', '',
      'Private media, captions, original IDs, GT coordinates, weights, and raw feature/prediction caches are excluded from the public export. Public anonymous scalar rows and audits preserve all conditions and negative findings.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names=['frame_drop','frame_freeze','motion_blur','occlusion','exposure'];colors=['#275DAD','#DA7C30','#3B8C6E']
    for ax,m,title in zip(axes,['tIoU','vIoU_corrected'],['Temporal IoU','Video IoU']):
        for j,s in enumerate([1,5,10]):
            vv=[c0[f'{f}_{s}']['delta'][m] for f in names];means=np.array([z['mean']*100 for z in vv]);lo=np.array([z['ci95'][0]*100 for z in vv]);hi=np.array([z['ci95'][1]*100 for z in vv]);x=np.arange(5)+(j-1)*.22
            ax.errorbar(x,means,yerr=[means-lo,hi-means],fmt='o',capsize=3,label=f'{s}% duration',color=colors[j])
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(5),['Drop','Freeze','Blur','Occlusion','Exposure']);ax.set_ylabel('Change vs clean (pp), parent 95% CI');ax.set_title(title);ax.grid(axis='y',alpha=.18)
    axes[0].legend();fig.suptitle('Query/GT-independent transient bursts | 32 development parents')
    for ext in ['png','svg']:fig.savefig(OUT/f'C05_EFFECTS.{ext}',dpi=170)
    plt.close(fig)

if __name__=='__main__':run()

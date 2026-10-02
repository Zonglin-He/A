"""Report full matched evidence, unchanged future state and inference cost."""
import sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_correction_views_common_v1 import *
from scripts.diagnose_tastvg_correction_views_v1 import effect
def fmt(z):return f"{100*z['mean']:+.4f} [{100*z['ci95'][0]:+.4f}, {100*z['ci95'][1]:+.4f}]"
def run():
    assert all((BASE/f'{st}_{sp}_ROOT_AUDIT.json').exists() for st in ['round1','round2'] for sp in ['search','confirm'])
    s1=read(BASE/'ROUND1_SELECTION.json');sf=read(BASE/'FINAL_SELECTION.json');parts=[
        '# Current query correction and persistent Uniform A: two-round review\n',
        'Both rounds and matched confirmation completed. Current corrections never enter future state. '
        'The baseline is sealed best Uniform A with native temporal Fast, not Frozen TA-STVG. '
        'Each dataset has 32 historically exposed development sources and 16 source-disjoint within-batch '
        'confirmation sources, one query per source, two orders, clean and five transient 5% corruptions, '
        '25% specialist arrivals. Confirmation is also historically exposed; it does not tune the decision.\n',
        'Vid K1/lr .033761698432507946/teacher .34902548789596055; HC K8/lr .006097133675874025/teacher1. '
        'rho .05/student1/D4/1792 parameters and official same-domain checkpoints remain fixed. '
        'All persistent transitions equal saved A, including multi-step HC updates; two live A arrivals per '
        'dataset verify prediction, gradient and state bitwise. New full nine-probe suffix replays validate '
        'saved supports. Direct selection is a changed spatial readout; temporary one-step SGD is '
        'current adapt-then-predict and expires immediately. Neither is cross-query fast memory.\n',
        f"Round1 selected one rule for both datasets: **{s1['correction']}**. Final screen selection: **{sf['final']}**; "
        f"new acquisition eligible: **{sf['new_acquisition_eligible']}**. These are development decisions, "
        'not production promotion or statistical guarantees. No confirmation reselection.\n']
    parts.append('\nThe confirmation supports small positive direct-selection means but does not establish '
        'a Routed-specific gain over the matched second Uniform5 observation: pooled C minus twoUniform5 is '
        '-.000978 pp, with a confidence interval spanning zero. Temporal output is positive on Vid and negative '
        'on HC in both development and confirmation. New acquisition does not satisfy the common development '
        'eligibility rule and was excluded before confirmation. Specific temporary SGD has negative means on '
        'both confirmation datasets. The nominated CT is therefore a completed confirmation experiment, '
        'not a proven common improvement or a promoted method.\n')
    parts.append('\n## Confirmation relative to Frozen and A\n\n'
        '| Dataset | Frozen vIoU (%) | A (%) | C (%) | CT (%) | CT minus Frozen, pp [95% CI] |\n'
        '|---|---:|---:|---:|---:|---:|\n')
    for ds in DATASETS:
        z=read(PUBLIC/'round2/confirm'/ds/'SUMMARY.json')['corruption']['all']['metrics']
        ct=read(PUBLIC/'round2/confirm/CONTRASTS.json')['contrasts']['CT_minus_Frozen_v']['corruption']['all']['datasets'][ds]
        parts.append('| '+ds+' | '+' | '.join(f'{100*z[a+"_v"]["mean"]:.4f}' for a in ['Frozen','A','C','CT'])+' | '+fmt(ct)+' |\n')
    parts.append('\nVid A is below Frozen on this confirmation cohort; the nominated CT still has a negative '
        'mean versus Frozen. Reported positive current increments must not be described as an overall Frozen win. '
        'The same-cohort paired intervals and all rows are retained.\n')
    resources=[]
    for sp in ['search','confirm']:
        for st in ['round1','round2']:
            parts.append(f'\n## {st} / {sp}: corruption-all source macro incremental vIoU (pp)\n')
            parts.append('| Arm | VidSTG, mean [95% paired source CI] | HC2, mean [95% paired source CI] |\n|---|---:|---:|\n')
            summary={d:read(PUBLIC/st/sp/d/'SUMMARY.json') for d in DATASETS}
            keys=[k for k in summary['vidstg']['corruption']['all']['metrics'] if k.startswith('delta_')]
            for k in keys:parts.append('| '+k.removeprefix('delta_').removesuffix('_v')+' | '+' | '.join(fmt(summary[d]['corruption']['all']['metrics'][k]) for d in DATASETS)+' |\n')
            for ds in DATASETS:
                z=summary[ds];rows=read(PUBLIC/st/sp/ds/'ROWS.json');wr=read(PUBLIC/st/sp/ds/'WRITE_ROWS.json')
                parts.append(f'\n{ds}: Frozen/A corruption vIoU {100*z["corruption"]["all"]["metrics"]["Frozen_v"]["mean"]:.4f}/'
                    f'{100*z["corruption"]["all"]["metrics"]["A_v"]["mean"]:.4f}%. '
                    'Every nonexpert correction increment is exactly zero, because the persistent trajectory is the same A. '
                    'This establishes a current-readout increment on top of online A, not new future parameter transfer.\n')
                for arm,e in z['corruption']['all']['effects'].items():
                    if arm=='A':continue
                    g=e['gross']['metrics'];clean=z['clean']['all']['metrics']['delta_'+arm+'_v'];v=z['corruption']['all']['metrics']['delta_'+arm+'_v']
                    parts.append(f'- {arm}: gross gain/loss {100*g["gross_gain"]["mean"]:.4f}/{100*g["gross_loss"]["mean"]:.4f} pp; '
                        f'>5pp harmful arrivals {e["harm_gt5pp"]}; .3 rescue/destroy {e["correctness"]["0.3"]["rescued"]}/{e["correctness"]["0.3"]["destroyed"]}; '
                        f'clean {fmt(clean)}; two order means '+', '.join(f'{100*x:+.4f}' for x in v['order_values'])+' pp.\n')
                if st=='round1':
                    bad=sum(w['execution']['R_selected_good_update_harm'] for w in wr if w['condition']!='clean')
                    parts.append(f'\nRouted selected a better existing tube but temporary SGD harmed the output in **{bad}** corrupt expert arrivals. '
                        'This diagnosis distinguishes expert preference from parameter execution; it is not an online GT filter.\n')
                else:
                    tw=[w for w in wr if w['condition']!='clean']
                    rr=[r for r in rows if r['condition']!='clean']
                    inherited=effect(rr,'A_native','Frozen');fast=effect(rr,'A','A_native')
                    parts.append(f'\nReadout chain, same sources: Frozen to A native {fmt(inherited)} pp; '
                        f'A native to original Fast A {fmt(fast)} pp. A native includes all inherited state effects; '
                        'this is a readout decomposition, not a unique causal diagnosis of spatial or temporal learning.\n')
                    parts.append('\nTemporal actual changed-observation fraction: '
                        f'{np.mean([w["temporal"]["actual_observation_difference"] for w in tw]):.4f}. '
                        f'Old/new selected candidate changed in {sum(w["temporal"]["original_selected"]!=w["temporal"]["new_selected"] for w in tw)}/{len(tw)} expert arrivals. '
                        'Two views use the existing observed STVG frame grid, with nearest-frame sampling; they are '
                        'different image observations rather than original full-rate independent videos. Stable agreement can be wrong.\n')
                    for b in ['U','Rnew','U2']:
                        vals=[w['evidence'][b]['event_frame_precision'] for w in tw]
                        pair=[w['evidence'][b]['critic_pairwise']['pairwise_accuracy'] for w in tw if w['evidence'][b]['critic_pairwise']['pairwise_accuracy'] is not None]
                        parts.append(f'- {b}: event-frame precision {100*np.mean(vals):.3f}%; '
                            f'candidate pairwise accuracy {100*np.mean(pair):.3f}% (mean qualified arrivals); '
                            f'empty {sum(w["evidence"][b]["valid_frames"]==0 for w in tw)}/{len(tw)}.\n')
                    tm=read(PUBLIC/st/sp/'CONTRASTS.json')['temporal_diagnosis'][ds]['corruption']
                    parts.append(f'\nOld/new temporal replacements {tm["old_replaced"]}/{tm["new_replaced"]}; '
                        f'tIoU-worsening replacements versus the same A native interval '
                        f'{tm["old_wrong_replacement_t"]}/{tm["new_wrong_replacement_t"]}; '
                        f'vIoU-worsening replacements {tm["old_wrong_replacement_v"]}/{tm["new_wrong_replacement_v"]}. '
                        'These are expert-arrival counts, not error probabilities on independent videos.\n')
                    aq=read(PUBLIC/st/sp/'CONTRASTS.json')['acquisition_diagnosis'][ds]['corruption']['event_precision_new_minus_old']
                    parts.append(f'\nNew versus old Routed event-frame precision: {fmt(aq)} pp; '
                        'fixed-time tube utility remains a separate paired contrast below.\n')
                resources.append(dict(dataset=ds,stage=st,split=sp,**read(BASE/ds/st/sp/'RESOURCES.json')))
            diag=read(PUBLIC/st/sp/'CONTRASTS.json')
            parts.append('\n### Paired differences between alternatives (corrupt-all, pp)\n\n'
                '| Contrast | Vid | HC2 | Equal-dataset pooled [95% CI] |\n|---|---:|---:|---:|\n')
            for name,z in diag['contrasts'].items():
                q=z['corruption']['all']
                parts.append('| '+name+' | '+' | '.join(fmt(q['datasets'][d]) for d in DATASETS)+' | '+fmt(q['pooled'])+' |\n')
                for ds in DATASETS:
                    e=q['datasets'][ds];lo=e['leave_one_source_out_range']
                    parts.append(f'\n{name}, {ds}: leave-one-source-out mean range '
                        f'[{100*lo[0]:+.4f}, {100*lo[1]:+.4f}] pp; '
                        f'sign-changing removals {e["leave_one_source_out_sign_changes"]}/{e["sources"]}. '
                        'The source with largest influence and all source means remain in CONTRASTS.json.\n')
    for f in BASE.glob('*_RESOURCES.json'):resources.append(read(f))
    write(PUBLIC/'RESOURCES.json',dict(entries=resources,wall_includes_loading_IO=True,
        logical_Sa2VA_observations='baseline Uniform5 persistent + one separate five-frame current observation on specialist arrivals',
        logical_UniversalVTG_observations='original phase0 + shifted phase.25 on specialist arrivals',
        interpret='Actual incremental inference can be less due to input-identical historical cache reuse; this is not deployment zero cost.'))
    actor=[r for r in resources if 'suffix_replays' in r];expert=[r for r in resources if 'new_calls' in r]
    parts.append('\n## Measured incremental execution cost\n\n'
        '| Work | Suffix replays / new expert invocations | Backwards / live controls | Process wall seconds |\n|---|---:|---:|---:|\n')
    for r in resources:
        who='/'.join(str(r.get(k,'')) for k in ['dataset','stage','split']).strip('/')
        a=r.get('suffix_replays',r.get('new_calls'));b=r.get('backward_calls',r.get('control_calls'))
        parts.append(f'| {who} | {a} | {b} | {r["worker_wall_seconds"]:.3f} |\n')
    parts.append(f'\nTotal current-readout suffix replays {sum(r["suffix_replays"] for r in actor)}, '
        f'backwards {sum(r["backward_calls"] for r in actor)}, '
        f'new spatial calls {sum(r["new_calls"] for r in expert if r["stage"]!="temporal")}, '
        f'new temporal calls {sum(r["new_calls"] for r in expert if r["stage"]=="temporal")}. '
        f'Successful worker process wall sums to {sum(r["worker_wall_seconds"] for r in resources)/60:.3f} minutes. '
        'This includes loading, I/O and cache readback, excludes rejected attempts, CPU scoring, publication '
        'and the separately preserved three-setting TF32 head diagnostic. The summed wall is not a deployment '
        'per-query benchmark or pure GPU kernel time. Historical backbone/Uniform/phase0 inference is cached, '
        'so incremental new-call counts understate end-to-end deployment work.\n')
    parts.extend(['\n## Interpretation and limitations\n',
        'The two-Uniform5 comparator uses original endpoint Uniform5 for persistent learning and distinct bin-midpoint '
        'Uniform5 for the current correction, each with five frames. It matches the two observation requests, not '
        'a ten-frame merged teacher. Probes and A persistence are shared across readouts. Report both logical '
        'budget and actual unique uncached specialist invocations. All process wall times include loading/IO, '
        'not pure GPU kernels. Reference selection and outputs are never derived from GT.\n',
        'Rank-RKL remains unchanged, including nonempty flat-reward entropy effects. Geometry compatibility is '
        'not a native tube policy. Current tie policy always falls back to center, even for top ties excluding '
        'center. Specific correction additionally requires both evidence branches nonempty. A direct tube '
        'selection result does not prove a generic Uniform/event-specific Routed semantic decomposition. '
        'Min-view temporal replacement does not imply localization correctness or better acquisition.\n',
        'All anonymous arrival rows, candidate/reward diagnostics, positive and severe negative cases, '
        'order and clean controls, paired source-bootstrap intervals and resource receipts are published. '
        'Private captions, media, annotations, weights, H, parameter and gradient tensors are excluded. '
        'The HC frame-freeze donor binding failure was rejected by pixel equality before new inference. '
        'The UniversalVTG environment import and runtime revision ordering were repaired before affected '
        'new predictions. Its development cached-feature head control had a maximum Vid error .02586555 physical frames, '
        'confidence 2.77162e-5 and student candidate score 7.63083e-6, with identical winning candidate; '
        'HC development errors were zero. Every confirmation control error is separately retained in '
        'BARRIERS_AND_CONTROLS.json. Original phase0 remains the exact saved input. A <.1-frame/<1e-4/same-winner '
        'interface control is numerical, not a GT gate. Sa2VA original Uniform masks/boxes/text are bitwise '
        'reproduced before new calls. CPU diagnostic JSON serialization was fixed without changed scores. '
        'All original engineering records remain archived. CURRENT_METHOD and paused queues are unchanged.\n'])
    text='\n'.join(parts);f=ROOT/'docs/TA_CURRENT_CORRECTION_VIEWS_REVIEW.md';f.write_text(text)
    write(PUBLIC/'CONFIG.json',dict(params={d:plan(d)['params'] for d in DATASETS},search_sources=32,confirm_sources=16,
        orders=2,conditions=plan('vidstg')['conditions'],expert_fraction=.25,historically_exposed=True,
        correction=s1['correction'],final=sf['final'],persistent='Uniform A unchanged',current_lifespan='one query'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    for st in ['round1','round2']:
        fig,axes=plt.subplots(1,2,figsize=(11,3.7),sharey=True)
        names=ARMS[1:] if st=='round1' else ['C','T','CT','CT_newacquisition','twoUniform5','twoUniform5_newtime']
        for ax,ds in zip(axes,DATASETS):
            for offset,sp,color,label in [(-.13,'search','#4383ae','Development'),(.13,'confirm','#ce7648','Confirmation')]:
                m=read(PUBLIC/st/sp/ds/'SUMMARY.json')['corruption']['all']['metrics'];z=[m['delta_'+a+'_v'] for a in names]
                y=np.arange(len(names))+offset;v=np.array([r['mean']*100 for r in z]);ci=np.array([r['ci95'] for r in z])*100
                ax.errorbar(v,y,xerr=np.maximum(np.array([v-ci[:,0],ci[:,1]-v]),0),fmt='o',ms=5,color=color,capsize=3,label=label)
            ax.axvline(0,color='#555555',lw=.8);ax.set_yticks(range(len(names)),names);ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2')
            ax.set_xlabel('Incremental vIoU over A (pp)');ax.grid(axis='x',alpha=.16);ax.spines[['top','right']].set_visible(False)
        axes[0].invert_yaxis();axes[1].legend(frameon=False,loc='best');fig.tight_layout();dest=PUBLIC/'figures'/st;dest.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(dest.with_suffix('.png'),dpi=220,bbox_inches='tight');fig.savefig(dest.with_suffix('.pdf'),bbox_inches='tight');plt.close(fig)
    print('Report and figures saved')
if __name__=='__main__':run()

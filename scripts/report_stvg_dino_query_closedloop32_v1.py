"""Data-derived sparse visual-feedback QR closed-loop report."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.run_stvg_dino_query_closedloop32_v1 import D,P,CODE,verify
from vg_tta.stvg_dino_query_closedloop32_v1 import ARMS,FIT_ARMS

def ci(v):return 'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}], n={v['parents']}"
def main():
    p=verify();x=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    lines=['# E-retaining query-only conditional-feedback closed loop32','',
        'Exact exposed HC16/Vid16 from endpoint confirmation32. Same original four DINO frames, E/final-MBR, .35 and source scales. Three new student branches update QR256 only, reset every query/branch. All STVG/DINO/LN weights frozen. Two scheduled Adam steps; final step2, no GT selection, LR HC.01/Vid.03 unchanged from prior QR configuration. No historical Before replay or parameter search.','',
        'Gaussian endpoint forward KL with covariance diag(source_scale²) and fixed16 normalization is used in all three branches. It is equivalent to source-scaled endpoint squared error; it is not the original IoU-reward OPD objective, nor a direct reproduction of autoregressive GKD. Testing teacher refresh does not by itself establish a novel loss or general OPD advantage. No-feedback entries contribute zero and all-no-feedback skips Adam, including momentum-only updates.','',
        'Fixed_center repeats cached aligned T(E) four times per anchor in both rounds. Fixed_sample reuses initial-E candidate feedback, first4 then next4 of prior sealed noise. On_policy has bit-exact first update with Fixed_sample; second round uses the same next4 noises centered on the updated actual E-corrected student, with genuinely refreshed aligned DINO decoding. Sampling is four complete shared-correction trajectories, teacher supervision remains four sparse anchors. It is not independent dense reward.','',
        'Output retains fixed E correction for every actual decoder node. At QR0, every output equals E bit-exact. Both original MBR intervals and tIoU remain unchanged.','',
        '| Output | HC complete vIoU % | Vid complete vIoU % |','|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {100*x['groups']['hc2']['metrics'][a]['v']:.6f} | {100*x['groups']['vidstg']['metrics'][a]['v']:.6f} |")
    for ds,g in x['groups'].items():
        lines+=['',f'## {ds}: paired differences, pp','',
            '| Contrast | Complete vIoU [95% CI] | Unobserved actual sIoU [95% CI] | Observed actual sIoU [95% CI] |','|---|---:|---:|---:|']
        for k,v in g['paired'].items():lines.append('| '+k+' | '+' | '.join(ci(v[e])for e in ['v','unobserved_actual_s','observed_actual_s'])+' |')
        lines+=['',f"E-supported queries {g['active']}/16; all16 remain in complete means. Unknown GT support stays null.",'',
            '| Branch minus E | Observed actual v contribution | Unobserved actual v contribution | Official interpolated v contribution | Positive/negative/unchanged | Complete losses below -5pp |','|---|---:|---:|---:|---:|---:|']
        for a in FIT_ARMS:
            v=g['v_contributions'][a];t=g['paired'][a+'_minus_E_MBR']['v'];lines.append(f"| {a} | {v['observed_actual']:+.6f} | {v['unobserved_actual']:+.6f} | {v['benchmark_interpolated']:+.6f} | {t['positive']}/{t['negative']}/{t['neutral']} | {t['severe_loss_below_minus5pp']} |")
    lines+=['','## All signed query changes versus E, pp','',
        '| Alias | Domain | Center local | Fixed center KL | Fixed sample KL | On-policy KL | On-policy minus fixed sample |','|---|---|---:|---:|---:|---:|---:|']
    for r in x['rows']:
        v=r['values'];b=v['E_MBR']['v'];delta=[100*(v[a]['v']-b)for a in ARMS[1:]]+[100*(v['On_policy_KL']['v']-v['Fixed_sample_KL']['v'])];lines.append(f"| {r['alias']} | {r['dataset']} | "+' | '.join(f'{d:+.6f}'for d in delta)+' |')
    lines+=['','## Actual work','',
        f"GPU allocation including loading, current Source restoration, new teacher encodings and fits: {x['compute']['GPU_allocation_seconds']:.6f}s, prelocked cap900s. Not first-use E/DINO/source-calibration cost.",'',
        '| Work | HC | Vid |','|---|---:|---:|']
    for k in ['new_Source_captures','Source_capture_seconds','DINO_encoding_frames','new_conditional_queries','new_conditional_decoder_seconds']:
        lines.append(f"| {k} | {x['groups']['hc2']['cost'][k]} | {x['groups']['vidstg']['cost'][k]} |")
    for a in FIT_ARMS:lines.append(f"| {a} fit seconds | {x['groups']['hc2']['cost']['fit_seconds'][a]:.6f} | {x['groups']['vidstg']['cost']['fit_seconds'][a]:.6f} |")
    lines+=['','Original Source outputs and logits are exact. New frozen Source captures only restore unsaved backward interfaces, which are now retained privately. Cached center/first-round feedback is reused. Fixed-sample round2 feedback is obtained once at the unchanged initial references, using the same four-candidate layout as on-policy round2. Actual measured costs are reported. Both second-round conditions make new conditional calls with the same four-candidate layout; fixed sampling keeps initial E references, on-policy changes their center. All3 share update loss, LR, scales, number of steps and candidate identifiers; admission/no-op counts and actual update magnitudes can differ and are saved.','',
        'All32 five-arm predictions sealed before GT scoring. Paired query=parent bootstrap10000 seed20261018. This is development mechanism validation, not independent confirmation or official full results. Actual-node changes must be separated from official interpolation; four-frame feedback is not event-boundary evidence. Existing QR response/detach limitations remain disclosed. No extra rounds, new backbone, temporal adaptation or automatic expansion.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    put(P/'CONFIGURATION.json',{k:p[k]for k in ['count','per_domain','seed','lr','scale','threshold','rounds','candidates','loss','GPU_seconds_cap','read_final_step','cohort_development_exposed']})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,(ds,g)in zip(axes,x['groups'].items()):
        vv=[g['paired'][a+'_minus_E_MBR']['v']for a in FIT_ARMS];ax.bar(range(3),[v['mean']for v in vv],yerr=[[v['mean']-v['CI95'][0]for v in vv],[v['CI95'][1]-v['mean']for v in vv]],capsize=3,color=['#8b9bb1','#bc8a61','#659f95']);ax.axhline(0,color='black',lw=.6);ax.set_xticks(range(3),['Fixed center','Fixed sample','On-policy']);ax.set_title(ds+' complete vIoU vs E');ax.set_ylabel('Difference (pp)')
    fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str((P/n).relative_to(ROOT))for n in ['RESULTS_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','REPORT.md','CONFIGURATION.json','COMPARISON.svg']]))
    put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32));print(P/'REPORT.md',flush=True)
if __name__=='__main__':main()

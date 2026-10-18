"""Report frozen source path controls; no candidate selection or model calls."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.diagnose_stvg_joint_native_path_v1 import BASE,OUT,read,put

def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(BASE/'RESULTS.json');assert r['status']=='complete';checks=0;summary={};casefiles=[]
    for ds in ['hc2','vidstg']:
        mm=r['means'][ds]['sample0'];summary[ds]={}
        for scope in ['sample0','all_samples']:
            m=r['means'][ds][scope];parts={k:dict(initial_noise=m['Native_noisy_initial'][k]-m['Native_codec'][k],one_step_denoiser=m['first_clean'][k]-m['Native_noisy_initial'][k],multistep_minus_first_clean=m['final'][k]-m['first_clean'][k],final_minus_Native=m['final'][k]-m['Native_codec'][k],final_minus_noisy_initial=m['final'][k]-m['Native_noisy_initial'][k],remove_initial_noise=m['no_initial_final'][k]-m['final'][k])for k in ['v','t','s','equal_block_state_loss']}
            for k,p in parts.items():assert abs(p['initial_noise']+p['one_step_denoiser']+p['multistep_minus_first_clean']-p['final_minus_Native'])<1e-12;checks+=1
            curves=r['curves'][ds][scope];low=curves['Native_noisy'][-1];summary[ds][scope]=dict(ordered_decomposition=parts,low_noise_actual_path_input_loss_div_sigma2=low['input']['equal_block_state_loss']/low['sigma']**2,last_step_vIoU_change=low['proposal']['v']-low['input']['v'],best_step_proposal_GT_diagnostic_only=int(np.argmax([a['proposal']['v']for a in curves['Native_noisy']])),best_clean_GT_diagnostic_only=int(np.argmax([a['clean']['v']for a in curves['Native_noisy']])))
    # Compact per-source first-path and four-path-mean curves, no coordinates.
    for row in r['rows']:
        case={k:v for k,v in row.items()if k!='samples'};case['samples']=[{k:v for k,v in s.items()if k!='curves'}for s in row['samples']]
        keys=[k for k in row['samples'][0]['curves']['Native_noisy'][0]['input']if k!='duration_fraction']
        case['curve_columns']=keys;case['curve_order']=['input','clean','proposal']
        case['sample0_curves']={arm:[dict(level=st['level'],sigma=st['sigma'],metrics=[[st[w][k]for k in keys]for w in case['curve_order']])for st in path]for arm,path in row['samples'][0]['curves'].items()}
        case['mean_of_four_path_metrics_curves']={arm:[dict(level=j,sigma=row['samples'][0]['curves'][arm][j]['sigma'],metrics=[[float(np.mean([s['curves'][arm][j][w][k]for s in row['samples']]))for k in keys]for w in case['curve_order']])for j in range(16)]for arm in row['samples'][0]['curves']}
        def redact(v):
            if isinstance(v,dict):return {k:redact(a)for k,a in v.items()if k!='duration_fraction'}
            if isinstance(v,list):return [redact(a)for a in v]
            return v
        case=redact(case)
        f=OUT/'cases'/(row['alias']+'.json');f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(case,separators=(',',':'),allow_nan=False)+'\n');casefiles.append(str(f.relative_to(ROOT)))
    public={k:v for k,v in r.items()if k!='rows'};public['ordered_diagnostic_summary']=summary;public['per_source_cases']=casefiles;public['export_scope']='all4 per-source stage/probe scalars; sample0 and equal-path-metric-mean per-step curves public, all4 full scalar curves and raw states private; no coordinate averaging'
    (OUT/'RESULTS.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n')
    summary_result=dict(status='complete',summary=summary,independent_telescope_checks=checks)
    if (BASE/'REPORT_SUMMARY.json').exists():assert read(BASE/'REPORT_SUMMARY.json')==summary_result
    else:put(BASE/'REPORT_SUMMARY.json',summary_result)
    plt.rcParams.update({'font.size':10});fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    for ax,ds in zip(axs,['hc2','vidstg']):
        m=r['means'][ds]['sample0'];cc=r['curves'][ds]['sample0']
        for arm,label,color in [('Native_noisy','Native + initial noise','#d95032'),('GT_noisy','GT + same noise (diagnostic)','#32845b'),('Native_no_initial_noise','Native, no initial noise (off-kernel)','#526fba')]:
            vals=[m['Native_noisy_initial'if arm=='Native_noisy'else('GT_noisy_initial'if arm=='GT_noisy'else'Native_codec')]['v']]+[x['proposal']['v']for x in cc[arm]]
            ax.plot(range(17),np.array(vals)*100,label=label,color=color,marker='.',markersize=4)
        ax.axhline(m['Native_codec']['v']*100,color='black',ls='--',label='Native codec baseline')
        ax.set_title(ds+' / fixed first path / source-val12');ax.set_xlabel('0 = initial state; 1..16 = completed updates');ax.set_ylabel('mean vIoU (%)');ax.grid(alpha=.2)
    axs[1].legend(fontsize=8,loc='best');f=OUT/'PATH_CURVES.png';fig.savefig(f,dpi=160);plt.close(fig)
    lines=['# Why Native-start inference loses despite passing local denoising checks','','Frozen HC epoch30 / Vid epoch20, source-val12 each, same cached conditions and original four CUDA seeds. No training, observation guidance, target inference, Teacher/Student calls or method selection. Source GT controls are diagnostics; this source subset is not proven unseen to pretrained Student.','','## Ordered first-path accounting','','vIoU percent. Native codec accounts for representation readout separately from raw Source. Changes telescope but are not unique causal percentages.','','| State | HC | VidSTG |','|---|---:|---:|']
    stages=['Native_codec','Native_noisy_initial','first_clean','first_proposal','final','no_initial_first_clean','no_initial_final','GT_codec','GT_noisy_initial','GT_first_clean','GT_final']
    for stage in stages:lines.append('| '+stage+' | '+' | '.join(f"{100*r['means'][ds]['sample0'][stage]['v']:.6f}"for ds in ['hc2','vidstg'])+' |')
    lines+=['','## Ordered contributions, percentage points','','| Diagnostic change | HC | VidSTG |','|---|---:|---:|']
    for part in ['initial_noise','one_step_denoiser','multistep_minus_first_clean','final_minus_Native','final_minus_noisy_initial','remove_initial_noise']:
        lines.append('| '+part+' | '+' | '.join(f"{100*summary[ds]['sample0']['ordered_decomposition']['v'][part]:+.6f}"for ds in ['hc2','vidstg'])+' |')
    lines+=['','## Matched one-step probes','','The epsilon is matched between GT and Native at each raw sigma; it is the original CUDA warm-start epsilon, not the separate old CPU low-noise probe. These do not replace the prior qualification results.','','| Domain | sigma | GT input/output vIoU | Native input/output vIoU |','|---|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for level in ['15','8','0']:
            pp=r['paired_probes'][ds]['sample0'][level];sigma=r['curves'][ds]['sample0']['Native_noisy'][int(level)]['sigma']
            vals=[ds,f'{sigma:.6f}']+[f"{100*pp[k]['input']['v']:.6f} / {100*pp[k]['clean']['v']:.6f}"for k in ['GT','Native']];lines.append('| '+' | '.join(vals)+' |')
    measured=['','## Measured findings','','The biggest loss is introduced before the denoiser: sigma2 destroys much of the existing Native prediction. The learned denoiser then recovers only part of it. Full16-step vIoU is above the one-clean readout in both first-path and all-path means, so this is not evidence that recursive updates universally decrease vIoU. Time and raw-state MSE can still worsen; all metrics remain in RESULTS.','','The large-noise recovery problem is not specific to Native bias: even paired GT-centered initialization ends far below its original clean GT codec readout. The prior low/medium qualification did not establish accurate sigma2 reconstruction. Removing initialization noise helps, but same-schedule inference still degrades Native; dropping noise alone does not establish a valid solution.','','At the last level, actual-path raw GT state loss divided by sigma squared is reported as a descriptive residual/noise comparison, NOT a calibrated OOD probability. A small declared sigma does not mean the current state is near GT. Low-noise identity preconditioning can preserve an already wrong state.','','These ordered changes do not identify independent causal shares. The counterfactual uses the same fixed network and sigma schedule and is off the training perturbation law. Neither a new checkpoint nor a new sampler is selected.']
    lines+=measured
    lines+=['','## Interpretation boundaries','','GT+Gaussian denoising and Native+Gaussian restoration are different input populations: Native includes structured temporal/object/geometry error before random noise is added. Same sigma is not the same distance from clean GT. Local near-GT preservation cannot certify recovery from that bias. Low-noise skip preconditioning preserves incorrect states too; report actual late-path error alongside sigma.','','Removing initial noise while retaining the sigma2 schedule is an intentionally off-kernel counterfactual, not calibrated sampling and not a newly selected method. The idealized GT-centered rollout is source-GT-assisted and cannot be deployed. Full-path versus first-clean compares two fixed readouts from the same initial state, not equal-compute methods. No GT-best intermediate step is selected online.','','The implemented update is algebraically a VE Euler step; no update-sign bug was found. This diagnosis does not isolate higher-order integration error from learned vector-field error, nor establish a new sampler or authorize new training. [Official EDM Euler/Heun sampler](https://github.com/NVlabs/edm/blob/main/generate.py).','','## Evidence and cost','','See RESULTS.json for both first-path and all-path means, time/spatial/state errors, all per-source stage/probe scalars and first/mean curves. All four detailed state paths are sealed privately. Old terminal states reproduced; official and DenseTube scorers agree. Frozen replay GPU lease: '+f"{read(BASE/'GPU_COMPLETION.json')['new_reserved_seconds']:.6f}"+' seconds; cumulative with prior repair '+f"{read(BASE/'GPU_COMPLETION.json')['cumulative_with_previous_repair_seconds']:.6f}"+' seconds under original14400s cap. Cached-feature diagnostic cost, not online end-to-end latency.','','![Fixed source-path curves](PATH_CURVES.png)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

"""Describe the matched estimation-target control without claiming an OPD gain."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_dino_endpoint32_v1 import D,P,PRE,LOCAL,CODE,ARMS,verify

def ci(v):return 'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"

def main():
    p=verify();x=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    lines=['# Cached teacher-endpoint versus transported-residual update: original32','',
        'Same exposed HC16/Vid16, one query per parent. Only the estimation target changes. E, aligned teacher outputs, admission masks, original24 states, denominator24,0.35 threshold, four planned output nodes and exact final-MBR interval remain fixed. No new model, DINO, feature extraction, training or amplitude search. All32 outputs sealed before new GT scoring.','',
        'Residual_local is the exact prior Independent4_observed output. Endpoint_local adds sum(m*(z(T)-z(E)))/24 to E at each planned node, in (cx,cy,log w,log h). All other decoder nodes are bit-exact E. No admission is a no-op; teacher equals E is a no-op; teacher equals sampled input need not be a no-op for the endpoint target. This is not an automatically correct repair or an OPD result.','',
        '| Output | HC complete vIoU % | Vid complete vIoU % |','|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {100*x['groups']['hc2']['metrics'][a]['v']:.6f} | {100*x['groups']['vidstg']['metrics'][a]['v']:.6f} |")
    for ds,g in x['groups'].items():
        lines+=['',f'## {ds}: paired differences, pp','',
            '| Contrast | Complete vIoU [95% interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |',
            '|---|---:|---:|---:|---:|']
        for a,d in g['paired'].items():lines.append('| '+a+' | '+' | '.join(ci(d[k])for k in ['v','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'])+' |')
        lines+=['',f"Known observed-node quality is available for {g['known_observed_query_count']}/16 queries; unknown support stays null, while all16 remain in complete vIoU.",'',
            '| Output minus E | Observed-node v contribution pp | Unobserved actual contribution pp | Benchmark interpolated contribution pp | Positive / negative / neutral | Loss below -5pp |',
            '|---|---:|---:|---:|---:|---:|']
        for a in ARMS[1:]:
            d=g['v_effect_contributions_pp'][a];v=g['paired'][a+'_minus_E_MBR']['v']
            lines.append(f"| {a} | {d['observed_actual']['mean']:+.6f} | {d['unobserved_actual']['mean']:+.6f} | {d['benchmark_interpolated']['mean']:+.6f} | {v['positive']} / {v['negative']} / {v['neutral']} | {v['severe_loss_below_minus5pp']} /16 |")
        lines+=['','| Output | Observed image-center norm | Observed log-size norm |','|---|---:|---:|']
        for a in ARMS[1:]:
            d=g['displacement'][a]['observed'];lines.append(f"| {a} | {d['center_mean']:.9f} | {d['log_size_mean']:.9f} |")
    lines+=['','## Per-query signed complete-vIoU changes, pp','',
        '| Alias | Domain | Residual minus E | Endpoint minus E | Endpoint minus Residual |','|---|---|---:|---:|---:|']
    for r in x['rows']:
        v=r['values'];a=100*(v['Residual_local']['v']-v['E_MBR']['v']);b=100*(v['Endpoint_local']['v']-v['E_MBR']['v'])
        lines.append(f"| {r['alias']} | {r['dataset']} | {a:+.6f} | {b:+.6f} | {b-a:+.6f} |")
    lines+=['','## Limits and execution boundary','',
        'Intervals use paired parent bootstrap10000 with seed20261018; this repeatedly exposed cohort is not independent confirmation. The two targets generally produce different directions and actual amplitudes: matching coefficients and support does not isolate amplitude-independent direction effects. All no-support queries and negative cases remain.','',
        'All tIoUs are exact. Both local outputs preserve unobserved actual decoder nodes, so their quality change there is exactly zero by construction. Official interpolation changes neighboring benchmark frames; the contribution columns add exactly to complete-vIoU changes. These are not new visual observations or learned propagation.','',
        'No new cohort, center-reference comparison or dynamic OPD has run. A target-level contrast cannot establish sampling or on-policy superiority. The existing production registration and previous results are unchanged.','',
        f"Cached output construction CPU wall time: {x['compute']['CPU_output_seconds']:.6f} s; new GPU/model/DINO/Source calls0. This excludes old Source/E/feedback generation and subsequent scoring/audit, and is not first-use end-to-end latency."]
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    config={k:v for k,v in p.items()if k!='records'}
    config['source_checkpoints']=read(ROOT/'results/stvg_error_state/2026-10-18/dino_local_residual32/CONFIGURATION.json')['source_checkpoints']
    put(P/'CONFIGURATION.json',config);put(P/'CPU_PROPERTIES.json',read(D/'CPU_PROPERTIES.json'))
    put(P/'COHORT_PUBLIC.json',dict(count=32,rows=[{k:r[k]for k in ['alias','dataset']}for r in p['records']],one_query_per_parent=True,old_target_exposure=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,ds in zip(axs,['hc2','vidstg']):
        g=x['groups'][ds];v=[g['paired'][a+'_minus_E_MBR']['v']for a in ARMS[1:]];m=[v['mean']for v in v]
        err=[[s['mean']-s['CI95'][0]for s in v],[s['CI95'][1]-s['mean']for s in v]]
        ax.bar(range(2),m,yerr=err,capsize=4,color=['#6d8396','#bf784c']);ax.axhline(0,color='black',lw=.6)
        ax.set_xticks(range(2),['Residual target','Teacher endpoint']);ax.set_title(ds+' (16 exposed queries)');ax.set_ylabel('Complete vIoU difference vs E (pp)')
    fig.tight_layout();fig.savefig(P/'TARGET_COMPARISON.svg');fig.savefig(D/'TARGET_COMPARISON.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str((P/n).relative_to(ROOT))for n in ['RESULTS_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','REPORT.md','CONFIGURATION.json','CPU_PROPERTIES.json','COHORT_PUBLIC.json','TARGET_COMPARISON.svg']]))
    print(P/'REPORT.md',flush=True)

if __name__=='__main__':main()

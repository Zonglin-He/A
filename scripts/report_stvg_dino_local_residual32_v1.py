"""Render anonymous signed results and explicit interpolation/displacement scope."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_dino_local_residual32_v1 import D,P,PRE,CODE,verify,ARMS

def fmt(v):return 'unknown'if v is None else f'{v:.6f}'
def ci(v):return f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}]"if v['mean']is not None else 'unknown'

def main():
    p=verify();x=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    lines=['# Cached mask-aligned local residuals: original32','',
      'Original exposed development cohort: HC16 + Vid16, one query per parent. All methods use the exact same E and final-MBR time interval. No model/expert calls, feature extraction, training, resampling or parameter search. All 32 new outputs were sealed before new GT scoring.','',
      '| Output | HC complete vIoU % | Vid complete vIoU % |','|---|---:|---:|']
    for a in ARMS:lines.append(f"| {a} | {100*x['groups']['hc2']['metrics'][a]['v']:.6f} | {100*x['groups']['vidstg']['metrics'][a]['v']:.6f} |")
    for ds,g in x['groups'].items():
        lines+=['',f'## {ds}: paired differences, pp','',
           '| Contrast | Complete vIoU [95% paired interval] | Observed actual sIoU | Unobserved actual sIoU | Benchmark interpolated sIoU |',
           '|---|---:|---:|---:|---:|']
        for contrast,d in g['paired'].items():lines.append('| '+contrast+' | '+' | '.join(ci(d[k])for k in ['v','observed_actual_s','unobserved_actual_s','benchmark_interpolated_s'])+' |')
        lines+=['',f"Known observed-node quality: {g['known_observed_query_count']}/16 queries; known unobserved-node quality: {g['known_unobserved_query_count']}/16. Null GT support stays unknown; all16 remain in complete-tube means.",'',
            '| Output minus E | Changed observed contribution to vIoU pp | Unobserved actual contribution pp | Interpolated contribution pp | Positive / negative / neutral queries | Loss below -5pp |',
            '|---|---:|---:|---:|---:|---:|']
        for a in ARMS[1:]:
            v=g['paired'][a+'_minus_E_MBR']['v'];d=g['v_effect_contributions_pp'][a]
            lines.append(f"| {a} | {d['observed_actual']['mean']:+.6f} | {d['unobserved_actual']['mean']:+.6f} | {d['benchmark_interpolated']['mean']:+.6f} | {v['positive']} / {v['negative']} / {v['neutral']} | {v['severe_loss_below_minus5pp']} /16 |")
        lines+=['','Displacement below is the query mean of mean per-node norms: center in normalized image units, size in log units. Equal support does not imply equal update strength.','',
           '| Output | Observed center norm | Observed log-size norm | Unobserved center norm | Unobserved log-size norm |','|---|---:|---:|---:|---:|']
        for a in ARMS[1:]:
            d=g['displacement'][a];lines.append(f"| {a} | {d['observed']['center_mean']:.9f} | {d['observed']['log_size_mean']:.9f} | {d['unobserved']['center_mean']:.9f} | {d['unobserved']['log_size_mean']:.9f} |")
    lines+=['','## Interpretation boundaries','',
       'The two local outputs preserve every unobserved actual decoder node bit-for-bit. Their unobserved actual-frame sIoU is therefore exactly unchanged. Changes at benchmark-interpolated frames arise from the official xyxy interpolation of changed neighboring nodes, not new visual evidence or learned propagation. The three contribution columns add exactly to complete-vIoU differences, with the fixed temporal-overlap denominator.','',
       'Independent4 uses each accepted teacher-minus-input correction with all24 states in the denominator. It does not average different anchors. Shared4_observed uses the exact prior global-output boxes at the planned nodes; its observed-node performance must equal Shared4_global. E itself is preserved. No GT selects updates or changes threshold, support, weights or method.','',
       'All signed cases, no-ops, missing-support queries, displacement maxima/RMS and paired intervals are included in RESULTS_PUBLIC.json. These are repeatedly exposed development data; a positive mean is not independent confirmation. No dynamic OPD or deployment promotion follows this finite experiment automatically.','',
       f"Cached output-construction CPU wall time: {x['compute']['CPU_output_seconds']:.6f} s. New GPU time 0, new Source/DINO/feature calls 0. This excludes the old Source/E and mask-alignment generation cost and is not first-use online latency.",'']
    report=P/'REPORT.md';assert not report.exists();report.write_text('\n'.join(lines))
    config={k:v for k,v in p.items()if k not in ['records','created_epoch']};config['source_checkpoints']=read(ROOT/'results/stvg_error_state/2026-10-18/dino_mask_align32/CONFIGURATION.json').get('source_checkpoints',{})
    put(P/'CONFIGURATION.json',config)
    put(P/'COHORT_PUBLIC.json',dict(count=32,per_domain=16,one_query_per_parent=True,old_target_exposure=True,
         rows=[{k:r[k]for k in ['alias','dataset']}for r in p['records']],private_identifiers_excluded=True))
    # A geometry-only summary plot, containing no media or raw coordinates.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));labels=['shared/global','shared/observed','independent/observed']
    for ax,ds in zip(axes,['hc2','vidstg']):
        g=x['groups'][ds];v=[g['paired'][a+'_minus_E_MBR']['v']for a in ARMS[1:]]
        vals=[t['mean']for t in v];err=[[t['mean']-t['CI95'][0]for t in v],[t['CI95'][1]-t['mean']for t in v]]
        ax.bar(range(3),vals,yerr=err,color=['#7a8290','#399f95','#4f6cbb'],capsize=3);ax.axhline(0,color='black',lw=.7)
        ax.set_xticks(range(3),labels,rotation=20);ax.set_title(ds+' (16 exposed queries)');ax.set_ylabel('Complete vIoU change vs E (pp)')
    fig.tight_layout();fig.savefig(P/'COMPLETE_VIOU.svg');fig.savefig(D/'COMPLETE_VIOU.png',dpi=150);plt.close(fig)
    files=CODE+[str((P/name).relative_to(ROOT))for name in ['REPORT.md','RESULTS_PUBLIC.json','CONFIGURATION.json','COHORT_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','COMPLETE_VIOU.svg']]
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=files))
    print(str(report),flush=True)

if __name__=='__main__':main()

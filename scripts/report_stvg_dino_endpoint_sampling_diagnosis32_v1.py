"""Report already scored path and geometry; no new inference or GT access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put
from scripts.analyze_stvg_dino_endpoint_sampling_diagnosis32_v1 import D,P,CODE,verify,ARMS

def ci(v):
    return 'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f}, {v['CI95'][1]:+.6f}], n={v['parents']}"
def num(v):return 'unknown'if v is None else f'{v:.9f}'
def main():
    verify();x=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    lines=['# Cached diagnosis of24-state versus center feedback','',
        'Original new-parent HC16/Vid16, frozen TA-STVG Source/E, aligned content-mask, .35, four original anchors, denominator24, final-MBR. Read existing prediction traces and already sealed query metrics only. No model/weights/media/GT read, new prediction/scoring, amplitude search, sample expansion or dynamic OPD. Historical official full-result exposure remains.','',
        '## Exact accounting','',
        'With e=z(E), c=accepted center endpoint, t=mean accepted sampled endpoint, p=n/24 and center mask M: Uc=M(c-e), Up=pUc, Us=p(t-e). Us-Uc=(Up-Uc)+(Us-Up). On common-admitted nodes the terms are (p-1)(c-e) and p(t-c); sampled-only nodes add admission-support differences to the second term. Same p is not equal displacement norm. Geometry and nonlinear task-metric differences are not interchangeable.','',
        'The saved complete-score path is exact: Samples24-Center_full=(Center_p-Center_full)+(Samples24-Center_p). It is a path-specific arithmetic accounting, not a unique causal attribution without interactions.','']
    for ds,g in x['groups'].items():
        lines += [f'## {ds}: complete tube, pp','', '| Saved contrast | Mean [95% paired interval] |','|---|---:|']
        for k,v in g['existing_complete_path'].items():lines.append(f'| {k} | {ci(v)} |')
        lines += ['','| Common-admitted saved sIoU contrast | Mean [95% paired interval] |','|---|---:|']
        for k,v in g['common_saved_sIoU_path'].items():lines.append(f'| {k} | {ci(v)} |')
        lines += ['','## Admission and endpoint geometry','',f"Active {g['active']}/16; all64 planned-node support counts: {g['support_counts']}. Counts keep no-support/illegal/no-feedback nodes.",'',
            '| Admission quantity | Value |','|---|---:|']
        for k,v in g['p'].items():lines.append(f'| {k} | {num(v)} |')
        lines += ['','p=(legal_states/24)*(accepted/ legal_states). Whole-state legality is shared across four anchors: an invalid anchor can remove a sampled state from every anchor. No legal-state resampling was performed. Admission fraction is not a correctness probability.','',
            '| Common-admitted endpoint geometry, no GT | Mean |','|---|---:|']
        for k,v in g['common_endpoints'].items():lines.append(f'| {k} | {num(v)} |')
        lines += ['','| Within-admitted-node dispersion | Value |','|---|---:|']
        for k,v in g['dispersion'].items():lines.append(f'| {k} | {num(v)} |')
        lines += ['','Endpoint IoU compares sampled encoded-mean endpoint to the center teacher endpoint, not to GT. Dispersion compares accepted teacher outputs against their exact cached input references in source-scaled coordinates. Concentrated outputs do not prove correctness or identity. Raw per-state GT IoU and the unshrunk mean endpoint GT score were not cached here and remain unmeasured.','',
            '| Actual displacement, all64 planned nodes | Source-scaled4D norm | Center image norm | Log-size norm |','|---|---:|---:|---:|']
        for a,v in g['displacement'].items():lines.append(f"| {a} | {num(v['source_scaled_norm'])} | {num(v['center_norm'])} | {num(v['log_size_norm'])} |")
        lines += ['','| Actual displacement, common-admitted nodes | Source-scaled4D norm | Center image norm | Log-size norm |','|---|---:|---:|---:|']
        for a,v in g['common_displacement'].items():lines.append(f"| {a} | {num(v['source_scaled_norm'])} | {num(v['center_norm'])} | {num(v['log_size_norm'])} |")
        lines += ['','## Descriptive groups by already saved center-minus-E result','',
            '| Center effect | Queries | Mean p on center-admitted nodes | p-scaling contribution to full16 mean | Endpoint/support contribution to full16 mean |','|---|---:|---:|---:|---:|']
        for k,v in g['center_outcome_descriptive_strata'].items():lines.append(f"| {k} | {v['queries']} | {num(v['p_mean_center_nodes'])} | {v['path_contribution_to_full16_mean_pp']['p_scaling_path']:+.6f} | {v['path_contribution_to_full16_mean_pp']['endpoint_and_support_path']:+.6f} |")
        lines += ['','These query-level groups are explanatory, not node identity/accuracy labels or deployment gates. All contributions retain the original full16 denominator.','',
            f"Mechanically selected lowest/highest3 Samples24-Center_full queries: {g['case_selection']}. All signed rows below remain primary, including unchanged and unknown common-support outcomes.",'']
    lines += ['## All signed query paths, pp','',
        '| Query | Domain | Samples24 minus center | p scaling path | Endpoint/support path | Center minus E | Samples24 minus E |','|---|---|---:|---:|---:|---:|---:|']
    for r in x['rows']:
        v=r['outcome_pp'];lines.append(f"| {r['alias']} | {r['dataset']} | "+' | '.join(f'{v[k]:+.6f}'for k in ['total_gap','p_scaling_path','endpoint_and_support_path','center_gain','sample_gain'])+' |')
    lines += ['','## Scope and resource decision','',
        'Paired10000 bootstrap, seed20261018, query=parent unit. Candidate/node repetitions are not independent samples. No GT-based new method selection, no new endpoint variant or matching-norm intervention. All unobserved actual decoder nodes and MBR intervals remain the original predictions. Benchmark-interpolated effects use previously saved metrics, not new visual feedback.','',
        f"Actual CPU analysis {x['CPU_seconds']:.6f}s (excludes later audit/report/publication); GPU initialized false, new model/GT/media/weights/scoring/predictions/samples all0. Original predictions, raw traces, scores and CURRENT are hash preserved. Current results do not authorize extra rounds, threshold changes or dynamic OPD."]
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,(ds,g)in zip(axs,x['groups'].items()):
        keys=['p_scaling_path','endpoint_and_support_path','total_gap'];v=[g['existing_complete_path'][k]for k in keys]
        ax.bar(range(3),[a['mean']for a in v],color=['#b97c60','#629f99','#7e93ad'],yerr=[[a['mean']-a['CI95'][0]for a in v],[a['CI95'][1]-a['mean']for a in v]],capsize=3)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),['p scaling','Endpoint/support','Total gap']);ax.set_title(ds+' saved complete-vIoU path');ax.set_ylabel('Difference (pp)')
    fig.tight_layout();fig.savefig(P/'PATH_DECOMPOSITION.svg');fig.savefig(D/'PATH_DECOMPOSITION.png',dpi=150);plt.close(fig)
    put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+[str((P/n).relative_to(ROOT))for n in ['RESULTS_PUBLIC.json','ROOT_INDEPENDENT_AUDIT.json','REPORT.md','PATH_DECOMPOSITION.svg']]))
    put(D/'REPORT_COMPLETION.json',dict(status='complete',count=32));print(P/'REPORT.md',flush=True)
if __name__=='__main__':main()

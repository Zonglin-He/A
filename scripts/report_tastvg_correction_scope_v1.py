"""Report lifecycle and block-by-scope effects without promoting a new method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_correction_scope_common_v1 import *
from scripts.audit_tastvg_negative_evidence_v1 import summary

def fmt(x):
    return f"{x['mean']*100:+.3f} [{x['ci95'][0]*100:+.3f}, {x['ci95'][1]*100:+.3f}]"

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    assert read(BASE/'ROOT_AUDIT.json')['status']=='pass'
    cfg=read(PUB/'CONFIG.json');s=read(PUB/'SUMMARY.json');raw=read(PUB/'SCOPE_ROWS.json')
    life=read(PUB/'LIFECYCLE_ROWS.json');donors=read(PUB/'DONOR_SCOPE_ROWS.json')
    lines=['# Episodic Rank-RKL and functional correction scope','',
        'Both experiments completed on the fixed, historically exposed 32 development + 16 confirmation sources per dataset. '
        '1,152 arrivals, 288 scheduled expert writes; two orders, clean plus five 5% corruptions. '
        'The unchanged same-domain A bundles are Vid K1 and HC2 K8. No new specialist inference, loss, hyperparameter search, '
        'online GT rule or method promotion.','',
        'This experiment separates complete query-wise source resets from the transfer scope of an isolated A write. '
        'Episodic_after uses the current query’s update before its output; A_before is the original predict-before-update '
        'persistent output. A_after is a matched post-update control, not the formal online output. Primary spatial '
        'attribution fixes the source-native interval for every lifecycle arm; native and before-update Fast intervals '
        'are reported separately.','',
        '## Lifecycle: spatial attribution on a common fixed interval','',
        '| Dataset / panel | Frozen vIoU % | A_before | A_after | E_after | E_after−A_after, pp [95% CI] | E_after−A_before |',
        '|---|---:|---:|---:|---:|---|---|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            m=s['lifecycle'][ds][split]['corrupt']['metrics']
            lines.append(f"| {ds} / {split} | {m['Frozen_v']['mean']*100:.3f} | {m['A_before_v']['mean']*100:.3f} | "
                f"{m['A_after_v']['mean']*100:.3f} | {m['E_after_v']['mean']*100:.3f} | {fmt(m['E_after_minus_A_after_v'])} | "
                f"{fmt(m['E_after_minus_A_before_v'])} |")
    lines += ['', 'A_before−Frozen describes online history; A_after−A_before describes immediate current-query execution. '
        'E_after−A_after changes only the state lifecycle at matched output timing. E_after−A_before changes both lifecycle '
        'and timing and cannot uniquely attribute an episodic advantage. Episodic nonexpert outputs are source Frozen by construction.','',
        '## Transfer scope of the same sealed A write','',
        '| Dataset / panel | Scope | Donor sources | Target cells | Full write vIoU change, pp [95% CI] |',
        '|---|---|---:|---:|---|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            for scope,z in s['scope'][ds][split]['corrupt'].items():
                lines.append(f"| {ds} / {split} | {scope} | {z['sources']} | {z['target_cells']} | "
                    +(fmt(z['metrics']['full_minus_pre_v']) if z['sources'] else 'N/A')+' |')
    lines += ['', 'The same source donor’s whole K1/K8 write is evaluated under a common donor-pre state. '
        'Targets include every eligible future nonexpert query in the original order/split/corruption, frozen text-near/far, '
        'and a fixed-hash same target tested in two corruptions. Near/far roles can alias existing targets; they are not '
        'additional independent samples. Same-video other-query uses real identical-media alternative captions and their '
        'own original sampling grids. HC2 has no such alternative for these sources and is explicitly N/A.','',
        'Target effects are averaged within donor/scope, then within donor source across condition/order; paired 10,000 '
        'donor-source bootstrap is primary. Target-source bootstrap sensitivity is saved independently. Single-write transfer '
        'does not establish long-stream retention or justify an online scope router.','',
        '## Block interventions and negative tails','',
        '| Dataset / panel | Scope | Query residual | norm1 | norm3 | norm4 | Full |',
        '|---|---|---|---|---|---|---|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            for scope,z in s['scope'][ds][split]['corrupt'].items():
                if not z['sources']:continue
                lines.append(f"| {ds} / {split} | {scope} | "+' | '.join(fmt(z['metrics'][b+'_minus_pre_v']) for b in BLOCKS)+' |')
    lines += ['', 'These finite state-replacement effects are nonadditive. A sum of individual LayerNorm effects is not '
        'the Full effect; they are not a gradient decomposition. They show which saved parameter block causes useful or '
        'harmful readout changes under this intervention.','',
        '| Dataset / confirmation scope | Improved / harmed cells | >5 / >20pp harm | Correct .3 destroyed / rescued | Correct .5 destroyed / rescued |',
        '|---|---|---|---|---|']
    for ds in DATASETS:
        for scope,z in s['scope'][ds]['confirm']['corrupt'].items():
            t=z['negative_tails']['full_minus_pre']
            lines.append(f"| {ds} / {scope} | {t['gain']} / {t['harm']} | {t['severe_harm_gt5pp']} / {t['severe_harm_gt20pp']} | "
                f"{t['baseline_good_destroyed_at_03']} / {t['baseline_bad_rescued_at_03']} | "
                f"{t['baseline_good_destroyed_at_05']} / {t['baseline_bad_rescued_at_05']} |")
    lines += ['', 'Gross gains/losses, clean controls, expert/nonexpert lifecycle slices, every order, native-free and '
        'source-fixed interval controls are included in SUMMARY.json and all anonymous rows. Cell-level tails are counts, '
        'not independent source counts. Each scope’s positive-gain concentration is disclosed below.','']
    for ds in DATASETS:
        for scope,z in s['scope'][ds]['confirm']['corrupt'].items():
            if not z['sources']:continue
            a=np.array(list(z['metrics']['full_minus_pre_v']['source_values'].values()));positive=np.maximum(a,0)
            share=float(positive.max()/positive.sum()) if positive.sum()>0 else None
            lines.append(f"- {ds} / {scope}: {int((a>1e-12).sum())}/{len(a)} positive donor sources; "
                +(f"largest donor share of positive source gain {share*100:.2f}%." if share is not None else 'no positive source gain.'))
    lines += ['', '## Matched cross-corruption intervention','',
        '| Dataset / panel | Full cross-corruption−same-corruption transfer, pp [95% CI] |',
        '|---|---|']
    for ds in DATASETS:
        for split in ['search','confirm']:
            z=s['cross_corruption'][ds][split]['corrupt']['metrics']['full_cross_minus_same_v']
            lines.append(f'| {ds} / {split} | {fmt(z)} |')
    lines += ['', 'A donor write is applied to the exact same target caption/media in donor corruption and the next '
        'prelocked corruption. This controls target identity; it does not prove natural temporal continuity between '
        'adjacent independent videos. No text-similarity threshold, learned router or memory is derived from GT.','',
        '## Actual execution, limitations and decision scope','',
        f"Actual unique target cells: {cfg['scope_target_cells']}; donor/scope aggregate rows: {cfg['donor_balanced_scope_rows']}. "
        f"Vid alternative-query frozen captures: {cfg['new_alternative_query_inputs']}; new specialist calls: 0. "
        'RESOURCES.json records suffix replays and worker wall time, not pure GPU kernel latency.','',
        'All new episode and matrix predictions seal globally before diagnostic labels. Root audit checks full source '
        'resets, refreshed K-step candidates, original donor state parity, exact block masks, frozen text target selection, '
        'Rank-RKL/SGD arithmetic and every dense readout. Public scalar audit independently reconstructs paired source '
        'bootstrap, tails and donor balancing. Full network Jacobians are not recomputed on CPU.','',
        'The results qualify local execution and isolated transfer scopes. They do not automatically authorize a new '
        'state router, a universal episodic replacement, dataset-specific rules or deployment changes.']
    lines += ['', 'The confirmation writer omitted direct post-update predictions. A_after there uses the sealed '
        'matched rank_native replay’s final-step raw output; original pre/post parameter states, pre-output, and '
        'search post-output boxes/intervals are independently identical. The replay’s top-level Fast-fixed interval '
        'is explicitly excluded from the native-free control. Nonexpert no-op uses the unchanged slow output. '
        'Schema/type recovery preserved the failed CPU attempts and changed no GPU inference or scientific setting.','',
        'On both confirmation panels, complete episodic reset did not establish an advantage over the matched '
        'persistent A_after control: both paired intervals cross zero. The isolated Full write improves Vid self '
        'readout, but confirmation transfer to other videos under the same corruption is inconclusive on both datasets. '
        'The Vid text-far contrast is negative in this small diagnostic panel; it is not a validated deployment router '
        'or a universal rule about semantic distance. Parameter-block effects are exploratory matched interventions, '
        'and many scope contrasts have not been adjusted for multiple testing.']
    report=ROOT/'docs/TA_CORRECTION_SCOPE_REVIEW.md';report.write_text('\n'.join(lines)+'\n')
    write(PUB/'DECISION.json',dict(production_promoted=False,new_router_launched=False,
        isolated_transfer_not_persistent=True,complete_full_source_reset=True,blocks_nonadditive=True))
    selected={ds:{scope:z for scope,z in s['scope'][ds]['confirm']['corrupt'].items()} for ds in DATASETS}
    scope_labels=['self','same_video_other_query','different_video_same_corruption','semantic_near','semantic_far','different_corruption']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        zs=[selected[ds][scope] for scope in scope_labels];eligible=[i for i,z in enumerate(zs) if z['sources']]
        v=np.array([zs[i]['metrics']['full_minus_pre_v']['mean']*100 for i in eligible])
        ci=np.array([zs[i]['metrics']['full_minus_pre_v']['ci95'] for i in eligible])*100
        ax.errorbar(v,eligible,xerr=np.stack([np.maximum(v-ci[:,0],0),np.maximum(ci[:,1]-v,0)]),fmt='o',color='#4385b8',capsize=3)
        ax.axvline(0,color='#444',lw=.8);ax.set_yticks(range(6),['Self','Same video, other query','Other video, same corruption','Text near','Text far','Matched other corruption'])
        if ds=='hc2':ax.text(.95,1,'N/A',transform=ax.get_yaxis_transform(),ha='right',color='#777')
        ax.set_title(ds+' confirmation');ax.invert_yaxis();ax.set_xlabel('Isolated Full-write vIoU (pp)');ax.grid(axis='x',alpha=.15)
    for ext in ['png','pdf']:fig.savefig(PUB/f'correction_scope.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        z=s['lifecycle'][ds]['confirm']['corrupt']['metrics'];arms=['Frozen','A_before','A_after','E_after']
        v=[z[a+'_v']['mean']*100 for a in arms]
        ax.bar(range(4),v,color=['#8c929a','#4385b8','#789cc0','#d65d58']);ax.set_xticks(range(4),['Frozen','A before','A after','Episodic after'],rotation=12)
        ax.set_title(ds+' confirmation');ax.set_ylabel('Common-time vIoU (%)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(PUB/f'lifecycle.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
    for ax,ds in zip(axes,DATASETS):
        scopes=[scope for scope in scope_labels if selected[ds][scope]['sources']]
        values=np.array([[selected[ds][scope]['metrics'][b+'_minus_pre_v']['mean']*100 for b in BLOCKS] for scope in scopes])
        lim=max(1,float(abs(values).max()));im=ax.imshow(values,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
        ax.set_xticks(range(5),['Query','norm1','norm3','norm4','Full']);ax.set_yticks(range(len(scopes)),[scope.replace('_',' ') for scope in scopes],fontsize=8)
        ax.set_title(ds+' confirmation')
        for i in range(len(scopes)):
            for j in range(5):ax.text(j,i,f'{values[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(values[i,j])>.65*lim else '#222')
        fig.colorbar(im,ax=ax,label='Isolated vIoU change (pp)',shrink=.8)
    for ext in ['png','pdf']:fig.savefig(PUB/f'block_scope.{ext}',dpi=220)
    plt.close(fig)
    write(PUB/'FIGURE_MANIFEST.json',dict(files={f.name:sha(f) for f in PUB.glob('*.*') if f.suffix in ['.png','.pdf']}))
    print('REPORT_AND_FIGURES',str(report),flush=True)

if __name__=='__main__':run()

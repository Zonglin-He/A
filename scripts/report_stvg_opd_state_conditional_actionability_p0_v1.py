"""Complete anonymous P9-P0 tables and four actual scientific PDF pages."""
import os,sys,csv,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_conditional_actionability_p0_v1 import BASE,OUT,read,write,sha,rows,verify,EPS
def run():
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass';rr=rows(OUT/'ROWS.jsonl.gz');s=read(OUT/'SUMMARY.json');bar=read(BASE/'GLOBAL_PAIRED_READOUT_BARRIER.json')
    def flip(r):
        a=r['order1_common_U_delta'];b=r['order2_common_U_delta'];return a is not None and b is not None and abs(a)>EPS and abs(b)>EPS and a*b<0
    table=[]
    for r in rr:
        a=r['order1_common_U_delta'];b=r['order2_common_U_delta'];f=flip(r)
        table.append(dict(**r,common_U_sign_flip=f,common_U_flip_both_over1pp=f and min(abs(a),abs(b))>=.01,common_U_flip_both_over5pp=f and min(abs(a),abs(b))>=.05))
    for name,q in [('ALL_PARENT_STATE_PAIRS.csv',table),('ALL_SIGN_REVERSALS.csv',[r for r in table if r['common_U_sign_flip']])]:
        with (OUT/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(q)
    report=['# P9-P0: same expert evidence across two inherited student states','',
        'Complete historical clean P7 roster:128 parents per target,2 original online orders,Omega0/original andOmega1/projected observations.512 parent-scope pairs represent1024 saved fits and256 independent parent units. Before states within an order are the same original Full state; P7-B did not write its new C state into a separate online stream. No new GPU/model/DINO/optimizer/raw-GT scoring or manuscript.','',
        f'All256 paired caches sealed before reusing the existing independently verified P9 GT-derived labels. Checkpoint/query/pixel/physical frames/expert/admission/actual observed positions/configuration match across orders. Both query residuals are exactly zero; only the saved1536 LN coordinates differ within the registered1792 parameter interface.917504 initial paired coordinates are source-bound. All{bar["random_rounds"]} actual sampled rounds/{bar["random_sample_coordinate_checks"]} candidate sample coordinates replay exactly from original CPU FP32 noise and the respective student mean. Of1024 saved episodes,{bar["actual_sampled_saved_updates"]} actually sampled;{bar["empty_saved_updates_no_sampler_or_optimizer"]} original empty episodes have no sampler/optimizer and remain in the cohort. Fixed noise does not imply equal candidate actions.','',
        'Primary common-U uses annotated GT-event genuine decoder frames excluding the union of the two allocations. Own-U excludes only the corresponding actual visited positions. Query annotation identity, physical support and exclusions match between orders. Missing/empty GT support is unknown, not zero. The labels are unchanged P9 offline labels; no raw GT was reopened.','',
        'Sign tolerance1e-12; meaningful descriptive tiers require BOTH opposite effects to be at least1pp or5pp in magnitude. Order differences of5/20pp and any-order harms below-5/-20pp are separate endpoints. All128, previous32, added96 and individual order means/transitions are fully retained.10000 parent bootstrap draws, seed20261013; exploratory unadjusted intervals, undefined constant resamples retained. No threshold trained or selected from these results.','',
        '|Target|Scope|Subset|Parents|Both nonneutral|Flips|Both >=1pp|Both >=5pp|Order difference >=5pp/>=20pp|Any-order harm <-5pp/<-20pp|Pearson LN RMS vs absolute effect difference [CI]|Spearman [CI]|',
        '|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|']
    def fmt(z):
        if z is None or z['value'] is None:return 'undefined'
        return f'{z["value"]:.4f} ['+', '.join(f'{v:.4f}' for v in z['ci95'])+']' if z['ci95'] is not None else f'{z["value"]:.4f} [undefined CI]'
    for ds in ['hc2','vidstg']:
        for scope in ['Omega0','Omega1']:
            for sub in ['all128','previous_P6_32','added96']:
                block=s[ds][scope][sub];o=block['outcomes']['common_U_delta'];a=block['associations']['common_U_delta']['LN_RMS_vs_absolute_effect_difference']
                report.append(f'|{ds}|{scope}|{sub}|{block["parents"]}|{o["both_nonneutral_pairs"]}|{o["sign_flip_parents"]}|{o["flip_both_over1pp"]}|{o["flip_both_over5pp"]}|{o["difference_over5pp"]}/{o["difference_over20pp"]}|{o["any_order_harm5pp"]}/{o["any_order_harm20pp"]}|{fmt(a["Pearson"])}|{fmt(a["Spearman"])}|')
    report.extend(['','This natural order/state comparison establishes matched evidence and common random numbers, but does not establish causal attribution to LN inheritance. Historical order, state composition and model sensitivity remain part of the observational contrast. State-vector distance is not automatically an unlabeled predictor of useful adaptation. Equal-state episodes, empty evidence, neutrals and all negative results are retained.','',
        'Wait for the already running5430-fit C Expanded-State experiment and its independent root before deciding on a controlled source-state versus inherited-state intervention. No new gate, scoring model, temporal gate, Query strategy or GPU intervention is authorized by this result alone. Production and original hyperparameters remain unchanged.','',
        'The first CPU reader mistakenly assumed empty expert episodes had rollout fields. Original no-information rounds and69 accepted parent readouts were preserved; a separately pinned dispatch validates original empty no-sampling/no-optimizer records. Science, saved predictions and the P9 labels were unchanged. Full original GPU-fit mathematics/root are reused, not newly replayed through a model Jacobian.'])
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.titlesize':11,'axes.labelsize':10})
    folder=ROOT/'output/pdf/stvg_opd_state_conditional_actionability';folder.mkdir(parents=True,exist_ok=True)
    p1=folder/'cross_order_update_reversals.pdf';p2=folder/'inherited_state_and_update_consequence.pdf'
    panels=[(d,k) for d in ['hc2','vidstg'] for k in ['Omega0','Omega1']]
    with PdfPages(p1) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,(ds,scope) in zip(axes.ravel(),panels):
            q=[r for r in rr if r['dataset']==ds and r['scope']==scope];valid=[r for r in q if r['order1_common_U_delta'] is not None and r['order2_common_U_delta'] is not None]
            for old,color in [(True,'#888888'),(False,'#2a777f')]:
                t=[r for r in valid if r['previous_P6']==old];ax.scatter([100*r['order1_common_U_delta'] for r in t],[100*r['order2_common_U_delta'] for r in t],s=24,c=color,alpha=.75)
            lim=max(10,max(max(abs(r['order1_common_U_delta']),abs(r['order2_common_U_delta']))*105 for r in valid)) if valid else 10
            ax.plot([-lim,lim],[-lim,lim],color='.6',ls=':',lw=1);ax.axhline(0,color='.5');ax.axvline(0,color='.5');ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_xlabel('Order1 common-U update effect (pp)');ax.set_ylabel('Order2 common-U update effect (pp)');ax.grid(alpha=.2)
            o=s[ds][scope]['all128']['outcomes']['common_U_delta'];ax.set_title(f'{ds} / {scope} / flips {o["sign_flip_parents"]}/{o["both_nonneutral_pairs"]} nonneutral')
        fig.suptitle('Same evidence and standardized noise: previous32 (gray), added96 (teal)',fontsize=12);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(10,8),constrained_layout=True);names=['positive','negative','neutral','unknown']
        for ax,(ds,scope) in zip(axes.ravel(),panels):
            o=s[ds][scope]['all128']['outcomes']['common_U_delta'];mat=np.array([[o['transition_matrix'].get(a+'_'+b,0) for b in names] for a in names]);ax.imshow(mat,cmap='Blues',vmin=0,vmax=128)
            for i in range(4):
                for j in range(4):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>64 else 'black')
            ax.set_xticks(range(4),names);ax.set_yticks(range(4),names);ax.set_xlabel('Order2 consequence');ax.set_ylabel('Order1 consequence');ax.set_title(f'{ds} / {scope} / all128 parents')
        fig.suptitle('Complete cross-order transitions; empty experts/neutral/unknown are retained',fontsize=12);pdf.savefig(fig);plt.close(fig)
    with PdfPages(p2) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for ax,(ds,scope) in zip(axes.ravel(),panels):
            q=[r for r in rr if r['dataset']==ds and r['scope']==scope and r['absolute_common_U_delta_difference'] is not None]
            for old,color in [(True,'#888888'),(False,'#2a777f')]:
                t=[r for r in q if r['previous_P6']==old];ax.scatter([r['LN_difference_RMS'] for r in t],[r['absolute_common_U_delta_difference']*100 for r in t],s=24,c=color,alpha=.75)
            ax.set_xlabel('Inherited LN cross-order RMS difference');ax.set_ylabel('Absolute cross-order effect difference (pp)');ax.set_title(ds+' / '+scope+f' / {len(q)}/128 defined pairs');ax.grid(alpha=.2)
        fig.suptitle('State distance versus update variability: previous32 (gray), added96 (teal)',fontsize=12);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);labels=[]
        for ax,(ds,scope) in zip(axes.ravel(),panels):
            labels=[];j=0
            for sub,color in [('all128','#2a777f'),('added96','#c08339'),('previous_P6_32','#888888')]:
                a=s[ds][scope][sub]['associations']['common_U_delta']['LN_RMS_vs_absolute_effect_difference']
                for method in ['Pearson','Spearman']:
                    v=a[method];labels.append(sub+' '+method)
                    if v is None or v['value'] is None:ax.text(-.95,j,'undefined',fontsize=9)
                    else:
                        ax.plot(v['value'],j,'o',color=color)
                        if v['ci95'] is not None:lo,hi=v['ci95'];ax.hlines(j,lo,hi,color=color,lw=1.4);ax.plot([lo,hi],[j,j],'|',color=color)
                    j+=1
            ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(-1,1);ax.axvline(0,color='.5',ls='--');ax.set_xlabel('Correlation (95% parent bootstrap CI)');ax.set_title(ds+' / '+scope);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Observational associations; intervals do not establish causal LN inheritance',fontsize=12);pdf.savefig(fig);plt.close(fig)
    write(BASE/'FIGURE_EXPORT.json',dict(status='pending_actual_all_four_pages_pixel_view_publication_archive',PDFs=2,pages=4,files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [p1,p2]},CPU_only=True,GT_read=True,time=time.time()))
    print('P9_P0_ACTUAL_FULL_TABLES_AND_FOUR_PDF_PAGES',len(table),sum(r['common_U_sign_flip'] for r in table))
if __name__=='__main__':run()

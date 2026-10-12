"""Report complete P8 scalar results and scientific support-factor figures."""
import csv,os,sys,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_support_factorization_v1 import BASE,OUT,read,write,sha,rows,verify
def run():
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass'
    summary=read(OUT/'SUMMARY.json');rr=rows(OUT/'ROWS.jsonl.gz')
    contrasts=['temporal_Omega0_v','temporal_Omega1_v','observation_I0_v','observation_I1_v','interaction_v']
    report=['# P8: prediction support x adaptation support, sealed-cache2x2','',
        'All512 original clean cells/256 historically exposed parents/two orders are fixed. A=(I0,Omega0), B=(I1,Omega0), E=(I0,Omega1), C=(I1,Omega1). E uses exactly the saved C boxes and changes only output time. D is the unchanged projected Direct reference. Zero new GPU fits, model calls, DINO calls or optimization. All2048 logical readouts were sealed before this offline GT score.','',
        'Parent statistics first average original orders, then equal-weight parents;10000 paired bootstrap, seed20261012. Old32/added96/all128 are separate. Intervals are conditional on saved histories, unadjusted for multiple reported contrasts. CI crossing zero is not an equivalence/weak-interaction conclusion; no equivalence margin was fitted.','',
        '| Target | Subset | Contrast | Parents | Mean pp |95% CI pp|>5pp harm|>20pp harm|',
        '|---|---|---|---:|---:|---|---:|---:|']
    for ds in ['hc2','vidstg']:
        for subset in ['all128','previous_P6_32','added96']:
            s=summary[ds][subset]['both']['statistics']
            for name in contrasts:
                v=s[name];report.append(f"|{ds}|{subset}|{name}|{v['parents']}|{100*v['mean']:+.4f}|[{100*v['ci95'][0]:+.4f},{100*v['ci95'][1]:+.4f}]|{v['harm_5pp']}|{v['harm_20pp']}|")
    report+=['','|Target|Subset|A mvIoU %|B mvIoU %|E mvIoU %|C mvIoU %|D reference mvIoU %|','|---|---|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        for subset in ['all128','previous_P6_32','added96']:
            s=summary[ds][subset]['both']['statistics'];report.append('|'+ds+'|'+subset+'|'+ '|'.join(f"{100*s[a+'_v']['mean']:.4f}" for a in ['A','B','E','C','D_reference'])+'|')
    report+=['','The interaction is exactly (C-E)-(B-A)=(C-B)-(E-A). Its framewise identity uses the official vIoU envelope denominator, distinct from tIoU temporal union: sum_f[(event and I1)/den1-(event and I0)/den0]*(IoU_Omega1-IoU_Omega0). Gained/lost/common-event contributions sum exactly to each cell interaction. This is a metric decomposition, not identification of expert quality or one optimizer variable as a cause.','',
        'Within A/B and E/C, the entire spatial trajectory and fixed-GT-support sIoU are identical. Interaction tIoU and sIoU are exactly zero; recall interactions can differ because thresholds are nonlinear. Genuine unobserved decoder event frames exclude both allocations\' visited-frame union; interpolation-only and unsupported-event frames are separate. Undefined partitions retain their valid/undefined counts.','',
        'All old P7 source/math/state roots are bound and reused; this root independently rereads sealed coordinates and sources, dense metrics, support masks, every decomposition, all anonymous rows and all parent/bootstrap statistics. It does not rerun a model or independently replay the entire decoder Jacobian.','',
        'Severe negative parents for every main contrast are preserved in SEVERE_NEGATIVE_PARENTS.csv. Full per-order/old32/new96/all128 vectors, R@0.3/.5, support change, reference quality and all zero/undefined cases remain in ROWS/SUMMARY. No paper writing, gate, reward/temperature/lr sweep, roster change, P7-B retry or production promotion was performed.']
    target=OUT/'REPORT.md';assert not target.exists();target.write_text('\n'.join(report)+'\n')
    negatives=[]
    for ds in ['hc2','vidstg']:
        chosen=[r for r in rr if r['dataset']==ds];ids=list(dict.fromkeys(r['parent_id'] for r in chosen))
        for parent in ids:
            group=[r for r in chosen if r['parent_id']==parent];assert len(group)==2
            for field in contrasts:
                value=np.mean([r[field] for r in group])
                if value<-.05:negatives.append(dict(dataset=ds,parent_id=parent,previous_P6=group[0]['previous_P6'],contrast=field,paired_value=float(value),order1_value=next(r[field] for r in group if r['order']=='order1'),order2_value=next(r[field] for r in group if r['order']=='order2'),harm_over20pp=bool(value<-.2)))
    path=OUT/'SEVERE_NEGATIVE_PARENTS.csv';assert not path.exists()
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','parent_id','previous_P6','contrast','paired_value','order1_value','order2_value','harm_over20pp']);w.writeheader();w.writerows(negatives)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=ROOT/'output/pdf/stvg_opd_support_factorization';dest.mkdir(parents=True,exist_ok=True)
    p1=dest/'support_factorization.pdf';p2=dest/'support_decomposition_and_tails.pdf';assert not p1.exists() and not p2.exists()
    def interval(ax,v,x,col,label=None):
        m=100*v['mean'];lo,hi=np.array(v['ci95'])*100
        ax.errorbar(m,x,xerr=[[m-lo],[hi-m]],fmt='o',color=col,capsize=3,label=label)
    with PdfPages(p1) as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            s=summary[ds]['all128']['both']['statistics'];names=['A: I0 / Omega0','B: I1 / Omega0','E: I0 / Omega1','C: I1 / Omega1'];x=np.arange(4);m=np.array([s[a+'_v']['mean'] for a in ['A','B','E','C']])*100;cis=np.array([s[a+'_v']['ci95'] for a in ['A','B','E','C']])*100
            ax.bar(x,m,color=['#68778c','#2c7d82','#c88a40','#547653']);ax.errorbar(x,m,yerr=[m-cis[:,0],cis[:,1]-m],fmt='none',ecolor='.25',capsize=4);ax.set_xticks(x,names,rotation=15,ha='right');ax.set_ylabel('mvIoU (%)');ax.set_title(ds+' / all128 paired parents');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.suptitle('Prediction support and adaptation support: complete cached2x2',fontsize=15);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,7),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            labels=[]
            for i,subset in enumerate(['previous_P6_32','added96','all128']):
                s=summary[ds][subset]['both']['statistics']
                for j,(field,col) in enumerate(zip(contrasts,['#64758b','#2a7d7b','#bd853b','#77964f','#9b465c'])):
                    interval(ax,s[field],i*6+j,col);labels.append(subset+' / '+field.replace('_v',''))
            ax.axvline(0,color='.5',lw=1);ax.set_yticks([i*6+j for i in range(3) for j in range(5)],labels)
            ax.invert_yaxis();ax.set_xlabel('Paired parent effect (pp),95% CI');ax.set_title(ds);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Effects and interaction: old32 / added96 / all128',fontsize=15);pdf.savefig(fig);plt.close(fig)
    with PdfPages(p2) as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
        fields=['gained_event_interaction_contribution','lost_event_interaction_contribution','common_event_interaction_contribution','interaction_v']
        for ax,ds in zip(axes,['hc2','vidstg']):
            for i,subset in enumerate(['previous_P6_32','added96','all128']):
                for j,field in enumerate(fields):interval(ax,summary[ds][subset]['both']['statistics'][field],i*5+j,['#2d7f78','#c0873b','#738299','#9e4b60'][j])
            ax.set_yticks([i*5+j for i in range(3) for j in range(4)],[sub+' / '+label for sub in ['old32','added96','all128'] for label in ['gained','lost','common','total']]);ax.invert_yaxis();ax.axvline(0,color='.5');ax.set_xlabel('Exact interaction contribution (pp),95% CI');ax.set_title(ds);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Gained, lost and common-event weights sum to the interaction',fontsize=14);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for j,ds in enumerate(['hc2','vidstg']):
            chosen=[r for r in rr if r['dataset']==ds];parents=list(dict.fromkeys(r['parent_id'] for r in chosen))
            vals={field:np.array([np.mean([r[field] for r in chosen if r['parent_id']==p]) for p in parents])*100 for field in ['interaction_v','observation_I1_v','original_delta_event_coverage']}
            added=np.array([not next(r['previous_P6'] for r in chosen if r['parent_id']==p) for p in parents])
            ax=axes[0,j];ax.scatter(vals['original_delta_event_coverage'][~added],vals['interaction_v'][~added],color='#7d8798',s=25,label='old32');ax.scatter(vals['original_delta_event_coverage'][added],vals['interaction_v'][added],color='#2e7e81',s=25,label='added96');ax.axhline(0,color='.5');ax.axvline(0,color='.5');ax.set_xlabel('GT-event coverage change (pp)');ax.set_ylabel('Interaction (pp)');ax.set_title(ds);ax.legend();ax.grid(alpha=.2)
            ax=axes[1,j];order=np.argsort(vals['observation_I1_v']);ax.scatter(np.arange(len(order)),vals['observation_I1_v'][order],c=np.where(added[order],'#2e7e81','#7d8798'),s=18);ax.axhline(0,color='.5');ax.axhline(-5,color='#bd873b',ls='--',label='-5 pp');ax.axhline(-20,color='#a24a5f',ls='--',label='-20 pp');ax.set_xlabel('All128 parents sorted by C-B');ax.set_ylabel('Observation allocation effect C-B (pp)');ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Complete parent effects and severe negative tails, no selected positive-only panel',fontsize=14);pdf.savefig(fig);plt.close(fig)
    write(BASE/'FIGURE_EXPORT.json',dict(status='pending_actual_pixel_view_publication_archive',PDFs=2,pages=4,files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in [p1,p2]},CPU_only=True,GT_read=True,time=time.time()))
    print('P8_REPORT_AND_FOUR_ACTUAL_PDF_PAGES',len(negatives))
if __name__=='__main__':run()

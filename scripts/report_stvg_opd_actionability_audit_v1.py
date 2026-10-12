"""Full positive-negative P9 tables and scientific figures; no manuscript."""
import os,sys,csv,time
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,OUT,SIGNALS,read,write,sha,rows,verify,parent_records,EPS

def run():
    verify();assert read(BASE/'ROOT_READBACK.json')['status']=='pass';rr=rows(OUT/'ROWS.jsonl.gz');s=read(OUT/'SUMMARY.json');decision=read(BASE/'CANDIDATE_DECISION.json')
    allparents=[];negative=[];repeat={}
    for ds in ['hc2','vidstg']:
        repeat[ds]={}
        for scope in ['Omega0','Omega1']:
            chosen=[r for r in rr if r['dataset']==ds and r['scope']==scope];counts=dict(mean_quality_correct_harmful=0,fully_labeled_correct_harmful=0,mean_quality_correct_both_orders_harmful=0,fully_labeled_correct_both_orders_harmful=0,opposite_order_effect_signs=0)
            for p in parent_records(chosen):
                group=[r for r in chosen if r['parent_id']==p['parent_id']];assert len(group)==2
                q=p['teacher_quality'];d=p['common_U_delta'];valid=[r for r in group if r['common_U_delta'] is not None]
                both_harm=len(valid)==2 and all(r['common_U_delta']< -EPS for r in valid)
                sign_flip=len(valid)==2 and all(abs(r['common_U_delta'])>EPS for r in valid) and valid[0]['common_U_delta']*valid[1]['common_U_delta']<0
                counts['opposite_order_effect_signs']+=sign_flip
                x=dict(dataset=ds,scope=scope,previous_P6=group[0]['previous_P6'],**p,
                    order1_common_U_delta=next(r['common_U_delta'] for r in group if r['order']=='order1'),order2_common_U_delta=next(r['common_U_delta'] for r in group if r['order']=='order2'),both_orders_harmful=both_harm,opposite_order_effect_signs=sign_flip)
                allparents.append(x)
                if q is not None and q>=.5 and d is not None and d< -EPS:
                    counts['mean_quality_correct_harmful']+=1;counts['fully_labeled_correct_harmful']+=p['teacher_fully_labeled_correct']==1
                    counts['mean_quality_correct_both_orders_harmful']+=both_harm;counts['fully_labeled_correct_both_orders_harmful']+=both_harm and p['teacher_fully_labeled_correct']==1
                    negative.append(x)
            repeat[ds][scope]=counts
    for name,data in [('ALL_PARENT_UPDATES.csv',allparents),('CORRECT_TEACHER_HARMFUL_UPDATES.csv',negative)]:
        p=OUT/name;assert not p.exists()
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(allparents[0]));w.writeheader();w.writerows(data)
    write(OUT/'REPEATED_ORDER_DIAGNOSTICS.json',dict(status='complete_offline_descriptive_diagnostic',counts=repeat,independent_parent_units=256,parent_scope_units=512,new_GPU_fits=0,new_DINO_calls=0))
    report=['# P9: teacher geometry versus update actionability','',
        'One finite CPU audit of the exact P7 roster:128 historical nondevelopment parents per target,2 original orders,512 cells and1024 cached Omega0/Omega1 updates.512 complete new readouts and every test-visible score were sealed before offline GT labels. Zero new model forward/GPU fit/DINO/optimizer steps. Shared Before boxes/parameter states and original source/pixel/frame/expert bindings were checked. Original complete fit mathematics and GPU repeat qualification are reused; no new model-Jacobian replay.','',
        'Primary consequence is spatial IoU after-minus-before on GT-event real decoder frames excluding both observation sets. Requested own-U excludes only the training allocation and is also reported in full. Interpolated/unsupported/missing GT frames never enter these labels. Held-out admitted anchors are chosen strictly from other allocation minus actual training-observed positions, without GT filtering. Both sets come from the same frozen DINO and cannot certify referent identity or independent-teacher correctness.','',
        'Parent means average two orders first. Single orders, previous32, added96 and all128 are separately saved. Binary AUROC and tie-group average precision (AP) condition on positive/negative nonneutral consequences and defined scores; unknown/neutral cases remain in all-parent tables.10000 paired parent bootstrap, seed20261012; confidence intervals are exploratory and unadjusted for multiple diagnostics. Undefined classless resamples are counted, not imputed. Matched AUROC differences use the identical eligible intersection. Teacher GT quality is offline oracle information, unavailable in deployment.','',
        '|Target|Training scope|Subset|All parents|Positive|Negative|Neutral|Undefined U|Held-out eligible +/-|AUROC [95% CI]|AP [95% CI]|Eligible positive prevalence|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.4f}'
    def ci(v):return 'undefined' if v is None else '['+', '.join(fmt(x) for x in v)+']'
    for ds in ['hc2','vidstg']:
        for scope in ['Omega0','Omega1']:
            for sub in ['all128','previous_P6_32','added96']:
                b=s[ds][scope][sub]['both'];o=b['outcomes']['common_U_delta'];v=b['rank']['common_U_delta']['holdout_agreement_gain']
                report.append(f"|{ds}|{scope}|{sub}|{b['parents']}|{o['positive']}|{o['negative']}|{o['neutral']}|{o['undefined_parents']}|{v['eligible_parents']} ({v['positive_parents']}/{v['negative_parents']})|{fmt(v['AUROC'])} {ci(v['AUROC_bootstrap']['ci95'] if v['AUROC_bootstrap'] else None)}|{fmt(v['AP'])} {ci(v['AP_bootstrap']['ci95'] if v['AP_bootstrap'] else None)}|{fmt(v['positive_prevalence'])}|")
    report+=['','|Target|Scope|Mean labeled Q>=0.5, harmful parent|Fully-labeled all anchors>=0.5, harmful parent|Former, harmful in both orders|Strict, harmful in both orders|Opposite order signs|','|---|---|---:|---:|---:|---:|---:|']
    for ds,scopes in repeat.items():
        for scope,c in scopes.items():report.append('|'+ds+'|'+scope+'|'+ '|'.join(str(c[k]) for k in ['mean_quality_correct_harmful','fully_labeled_correct_harmful','mean_quality_correct_both_orders_harmful','fully_labeled_correct_both_orders_harmful','opposite_order_effect_signs'])+'|')
    report+=['','Mean labeled Q>=0.5 is a prespecified descriptive geometric criterion, not an online admission threshold. Partial annotation does not label every other teacher anchor correct. Strict counts require every admitted anchor to have an event annotation and IoU>=0.5; empty evidence is not correct by vacuous truth. These are parent/scope diagnostics, not a universal independent-video failure rate. Accurate geometry co-occurring with harm does not isolate referent confusion, an optimizer variable or a Jacobian as its cause.','',
        '|Target|Scope|Subset|Held-out minus baseline on identical parents|Eligible parents|AUROC difference [95% CI]|','|---|---|---|---|---:|---|']
    for ds in ['hc2','vidstg']:
        for scope in ['Omega0','Omega1']:
            for sub in ['all128','added96']:
                for field,v in s[ds][scope][sub]['both']['matched']['common_U_delta'].items():report.append(f"|{ds}|{scope}|{sub}|{field}|{v['eligible_parents']}|{fmt(v['AUROC_difference'])} {ci(v['ci95'])}|")
    report+=['',f"Predeclared information criterion supports considering one bounded counterfactual selection test: **{decision['heldout_signal_supports_bounded_counterfactual_selection_test']}**. No learned gate, threshold search, dynamic alpha, additional teacher, query-space strategy or production change was implemented. Failure of this conjunction does not imply every test-visible signal is information-free or every future actionability mechanism impossible.",
        '', 'After-the-update held-out agreement requires a candidate update and additional cached teacher observations; this audit is not a zero-cost pre-update predictor. Its zero new DINO count means existing evidence reuse, not zero deployment expert cost. Any future controlled selection test needs an explicit matched budget and state lifecycle, and cannot use these GT labels online.',
        '', 'All1024 anonymous rows,512 parent/scope rows, both outcomes, six fixed test-visible scores, oracle geometry, full36 target/scope/subset/order blocks, positive prevalence, neutral/undefined coverage and every negative case are retained. No target-GT polarity flip, fitted threshold, selected positive-only subset or paper manuscript. C expanded-state GPU execution continues with its original scientific runtime; cancelled formal baselines are not restored.']
    path=OUT/'REPORT.md';assert not path.exists();path.write_text('\n'.join(report)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    dest=ROOT/'output/pdf/stvg_opd_actionability_audit';dest.mkdir(parents=True,exist_ok=True)
    p1=dest/'teacher_quality_and_update_consequences.pdf';p2=dest/'heldout_evidence_predictability.pdf';assert not p1.exists() and not p2.exists()
    with PdfPages(p1) as pdf:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(ds,scope) in zip(axes.ravel(),[(d,k) for d in ['hc2','vidstg'] for k in ['Omega0','Omega1']]):
            pp=[r for r in allparents if r['dataset']==ds and r['scope']==scope];mat=np.zeros((3,4),int)
            for r in pp:
                q=r['teacher_quality'];i=2 if q is None else 0 if q>=.5 else 1;d=r['common_U_delta'];j=3 if d is None else 0 if d>EPS else 1 if d< -EPS else 2;mat[i,j]+=1
            ax.imshow(mat,cmap='Blues',vmin=0,vmax=max(1,int(mat.max())))
            for i in range(3):
                for j in range(4):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>.6*mat.max() else '#172435',fontsize=13)
            ax.set_xticks(range(4),['benefit','harm','neutral','undefined U']);ax.set_yticks(range(3),['mean labeled Q >= .5','mean labeled Q < .5','teacher unknown']);ax.set_title(ds+' / '+scope+' /128 paired parents')
        fig.suptitle('Teacher geometry and update consequence: all parents retained',fontsize=15);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(ds,scope) in zip(axes.ravel(),[(d,k) for d in ['hc2','vidstg'] for k in ['Omega0','Omega1']]):
            pp=[r for r in allparents if r['dataset']==ds and r['scope']==scope]
            for old,col in [(True,'#8190a2'),(False,'#27837c')]:
                valid=[r for r in pp if r['previous_P6']==old and r['teacher_quality'] is not None and r['common_U_delta'] is not None]
                ax.scatter([r['teacher_quality'] for r in valid],[r['common_U_delta']*100 for r in valid],s=25,color=col,alpha=.8,label='old32' if old else 'added96')
            ax.axhline(0,color='.5');ax.axvline(.5,color='.5',ls='--');ax.set_xlim(0,1);ax.set_xlabel('Mean labeled teacher IoU (offline GT)');ax.set_ylabel('Common-U spatial update effect (pp)');ax.grid(alpha=.2)
            valid=sum(r['teacher_quality'] is not None and r['common_U_delta'] is not None for r in pp);ax.set_title(ds+' / '+scope+f' / {valid}/128 labeled parents')
        fig.suptitle('Geometry and consequences: old32 (gray), added96 (teal); unknown retained in rows',fontsize=13);pdf.savefig(fig);plt.close(fig)
    labels=['held-out gain','detector score','Before train IoU','training gain','teacher geometry','small output move','GT teacher quality (oracle)']
    colors={'all128':'#2a777f','added96':'#c08339'}
    def error(ax,value,interval,y,col):
        if value is None or interval is None:ax.text(.03,y,'undefined',color=col,fontsize=8);return
        lo,hi=interval
        ax.hlines(y,lo,hi,color=col,lw=1.4);ax.plot([lo,hi],[y,y],'|',color=col,ms=6);ax.plot(value,y,'o',color=col,ms=5)
    with PdfPages(p2) as pdf:
        for metric in ['AUROC','AP']:
            fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
            for ax,(ds,scope) in zip(axes.ravel(),[(d,k) for d in ['hc2','vidstg'] for k in ['Omega0','Omega1']]):
                for sub,offset in [('all128',-.12),('added96',.12)]:
                    block=s[ds][scope][sub]['both']['rank']['common_U_delta']
                    for j,f in enumerate(SIGNALS+['teacher_quality']):
                        v=block[f];boot=v[metric+'_bootstrap'];error(ax,v[metric],None if boot is None else boot['ci95'],j+offset,colors[sub])
                        if metric=='AP' and v['positive_prevalence'] is not None:ax.plot(v['positive_prevalence'],j+offset,'x',color='.3',ms=5)
                if metric=='AUROC':ax.axvline(.5,color='.55',ls='--')
                ax.set_xlim(0,1);ax.set_yticks(range(7),labels);ax.invert_yaxis();ax.set_xlabel(metric+' (95% parent bootstrap CI)');ax.set_title(ds+' / '+scope);ax.grid(axis='x',alpha=.2)
            fig.suptitle(metric+': all128 (teal), added96 (orange)'+(' / x = eligible positive prevalence' if metric=='AP' else ' / fixed score polarity'),fontsize=13);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(ds,scope) in zip(axes.ravel(),[(d,k) for d in ['hc2','vidstg'] for k in ['Omega0','Omega1']]):
            for sub,offset in [('all128',-.12),('added96',.12)]:
                block=s[ds][scope][sub]['both']['matched']['common_U_delta']
                for j,f in enumerate(SIGNALS[1:]):v=block[f];error(ax,v['AUROC_difference'],v['ci95'],j+offset,colors[sub])
            ax.axvline(0,color='.55',ls='--');ax.set_yticks(range(5),labels[1:6]);ax.invert_yaxis();ax.set_xlabel('Held-out AUROC minus baseline on same parents');ax.set_title(ds+' / '+scope);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Matched AUROC contrasts: all128 (teal), added96 (orange); identical parents',fontsize=13);pdf.savefig(fig);plt.close(fig)
    write(BASE/'FIGURE_EXPORT.json',dict(status='pending_actual_pixel_view_publication_archive',PDFs=2,pages=5,files={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [p1,p2]},CPU_only=True,GT_read=True,time=time.time()))
    print('P9_ACTUAL_FULL_REPORT_AND_FIVE_PDF_PAGES',len(allparents),len(negative))
if __name__=='__main__':run()

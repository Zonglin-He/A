"""Post-seal budget figures and complete tails; no inference or metric scoring.

Figures are standalone scientific PDF/PNG/SVG exports. They cannot be generated
before the corresponding full-stage score receipt and seal are present.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
R=Path(__file__).resolve().parents[2];sys.path[:0]=[str(R),str(R/'tools')]
from full import RUN,read,sha,atomic
from full_score import cluster
D=R/'outputs/paper_review_suite_v1';P=R/'results/paper_review_suite_v1'
CONFIGS=[(f'{s}{k}',s,k)for k in [2,4,8,16,32]for s in ['Hybrid','Uniform']]+[('Random8','Random',8),('SourceOnly8','SourceOnly',8),('DenseNative','Dense',None)]


def absolute_CI(values):
    v=np.asarray(values,np.float64);assert len(v)>0 and np.isfinite(v).all()
    # One query/parent, so equal-query parent bootstrap is the ordinary mean
    # bootstrap. Fixed seed; no choice of interval after seeing a curve.
    counts=np.random.default_rng(20261024).multinomial(len(v),np.ones(len(v))/len(v),size=10000)
    return np.quantile(counts@v/len(v),[.025,.975]).tolist()


def main(stage,prefix):
    score=read(D/(prefix+'_'+stage+'_score_receipt.json'));seal=read(D/(prefix+'_'+stage+'_sealed.json'))
    assert score['seal_sha256']==sha(D/(prefix+'_'+stage+'_sealed.json'))and not seal['GT_read']
    selection=read(D/'selection.json');plots=[];data={}
    for ds,groups in selection.items():
        pub=P/prefix/ds/stage;report=read(pub/'report.json');old={x['id']:x for x in read(RUN/ds/'metrics.json')['private_rows']}
        signed=list(csv.DictReader((pub/'signed.tsv').open(),delimiter='\t'));lookup={(x['query'],x['configuration']):x for x in signed}
        summary={};all_signed=[];plot_rows=[];dist=[]
        for name,schedule,k in CONFIGS:
            if stage=='confirm'and name not in read(D/(prefix+'_qualification.json'))['eligible']:
                summary[name]=dict(status='not_advanced_by_fixed_implementation_or_cost_rule',planned_queries=0,confirmation_cohort_size=len(groups[stage]),queries=0)
                continue
            rows=[];costs=[];ratios=[];missing=[]
            for i,row in enumerate(groups[stage],1):
                key=row['id'];anon=f'Q{i:03d}';entry=seal['files'][ds][key].get(name)
                if entry is None:missing.append(anon);continue
                assert sha(D/entry['path'])==entry['sha256'];x=lookup[(anon,name)]
                vals={a:dict(old[key]['values'][a])for a in ['Source','Full']};vals[name]=dict(t=float(x['t']),v=float(x['v']))
                rows.append(dict(parent=old[key]['parent'],values=vals,duration_group=old[key]['duration_group']))
                cost=float(x['actual_GPU_forward_seconds']);assert cost==entry['expert_GPU_forward_seconds']
                assert int(x['actual_K'])==entry['actual_K']and int(x['T'])==entry['T'];costs.append(cost);ratios.append(entry['K_over_T'])
                dt=100*(vals[name]['t']-vals['Source']['t']);dv=100*(vals[name]['v']-vals['Source']['v']);all_signed.append([anon,name,old[key]['duration_group'],vals['Source']['t'],vals['Source']['v'],vals[name]['t'],vals[name]['v'],dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,entry['actual_K'],entry['T'],cost,entry['cache_hit']])
                dist.append([anon,name,entry['actual_K'],entry['T'],entry['K_over_T'],entry['new_expert_forward'],entry['cache_hit']])
            if not rows:
                summary[name]=dict(status='no_predictions',requested=len(groups[stage]),queries=0,missing=missing);continue
            v=np.array([x['values'][name]['v']for x in rows])*100;t=np.array([x['values'][name]['t']for x in rows])*100
            delta=np.array([[x['values'][name][f]-x['values']['Source'][f]for f in ['t','v']]for x in rows])*100
            status='complete'if not missing else'partial_not_full_cohort'
            summary[name]=dict(status=status,requested=len(groups[stage]),queries=len(rows),missing=missing,mean_v_percent=float(v.mean()),mean_t_percent=float(t.mean()),v_CI95_percent=absolute_CI(v),t_CI95_percent=absolute_CI(t),expert_GPU_forward_seconds=float(np.mean(costs)),severe_t=int((delta[:,0]< -5).sum()),severe_v=int((delta[:,1]< -5).sum()),severe_union_once=int((delta< -5).any(1).sum()),groups={gg:dict(queries=sum(x['duration_group']==gg for x in rows),paired_Source=cluster([x for x in rows if x['duration_group']==gg],name,'Source',fields=[('t',100),('v',100)]))for gg in ['short_le5s','long_gt5s']if any(x['duration_group']==gg for x in rows)})
            assert abs(summary[name]['mean_v_percent']-100*report['configs'][name]['means'][name]['v'])<1e-10
            if status=='complete':plot_rows.append(dict(name=name,schedule=schedule,K=k,**summary[name],ratios=ratios))
        baseline={f:100*float(np.mean([old[x['id']]['values']['Source'][f]for x in groups[stage]]))for f in ['t','v']}
        data[ds]=dict(summary=summary,plot_rows=plot_rows,Source=baseline,requested_queries=len(groups[stage]),missing_curves={n:x['status']for n,x in summary.items()if x['status']!='complete'})
        atomic(pub/'tails_timegroups_and_plot_data.json',data[ds])
        for filename,header,values in [
            ('all_signed_severe_timegroups.tsv',['query','configuration','duration_group','Source_t','Source_v','t','v','delta_t_pp','delta_v_pp','severe_t_gt5pp','severe_v_gt5pp','severe_union_once','actual_K','T','actual_expert_GPU_forward_seconds','cache_hit'],all_signed),
            ('actual_observation_ratios.tsv',['query','configuration','actual_K','T','K_over_T','new_expert_forward','cache_hit'],dist)]:
            with(pub/filename).open('w')as out:w=csv.writer(out,delimiter='\t',lineterminator='\n');w.writerow(header);w.writerows(values)
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    colors={'Hybrid':'#2864A0','Uniform':'#B27D2B','Random':'#6A6592','SourceOnly':'#567D69','Dense':'#555555'}
    markers={'Hybrid':'o','Uniform':'s','Random':'^','SourceOnly':'D','Dense':'X'}
    target=P/prefix/'figures';target.mkdir(parents=True,exist_ok=True)
    private_exports=D/(prefix+'_figure_exports');private_exports.mkdir(parents=True,exist_ok=True)
    def finish(fig,name):
        fig.tight_layout(rect=[0,.12,1,.94])
        for suffix in ['pdf','png','svg']:
            f=(target if suffix=='svg'else private_exports)/(stage+'_'+name+'.'+suffix);fig.savefig(f,dpi=220,bbox_inches='tight');plots.append(dict(path=str(f.relative_to(R)),sha256=sha(f),public=suffix=='svg'))
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4.3))
    for ax,(ds,x)in zip(axes,data.items()):
        ax.scatter([0],[x['Source']['v']],c='#333333',marker='P',s=55,label='Source (no expert)')
        for schedule in colors:
            rr=[z for z in x['plot_rows']if z['schedule']==schedule]
            if not rr:continue
            ax.scatter([z['expert_GPU_forward_seconds']for z in rr],[z['mean_v_percent']for z in rr],c=colors[schedule],marker=markers[schedule],s=42,label=schedule)
            for z in rr:
                ci=z['v_CI95_percent'];ax.vlines(z['expert_GPU_forward_seconds'],ci[0],ci[1],color=colors[schedule],alpha=.5,lw=1)
                ax.annotate('Dense'if z['K']is None else str(z['K']),(z['expert_GPU_forward_seconds'],z['mean_v_percent']),xytext=(4,5),textcoords='offset points',fontsize=8)
        ax.set(title=f'{ds.upper()} · {stage} · n={x["requested_queries"]}',xlabel='Additional expert GPU forward (s/query)',ylabel='Mean vIoU (%)');ax.grid(axis='y',color='#e5e5e5',lw=.6);ax.set_xlim(left=-.03)
    axes[1].legend(frameon=False,fontsize=8,loc='best');fig.suptitle('Native joint observation budgets: cost and grounding quality',fontsize=12)
    fig.text(.06,.025,'95% parent-bootstrap CIs; per-domain y ranges. Only complete-cohort configurations plotted; all failures remain in tables.\nCost includes the same expert’s original measured forward, even on cache hits; Source/load/CPU costs are separate.',fontsize=8)
    finish(fig,'expert_cost_viou')
    fig,axes=plt.subplots(1,2,figsize=(9,4.1))
    for ax,(ds,x)in zip(axes,data.items()):
        for schedule in ['Hybrid','Uniform']:
            rr=sorted((z for z in x['plot_rows']if z['schedule']==schedule),key=lambda z:z['K'])
            if not rr:continue
            ax.plot([z['K']for z in rr],[z['mean_t_percent']for z in rr],color=colors[schedule],marker=markers[schedule],ls='-'if schedule=='Hybrid'else'--',label=schedule)
            for z in rr:ax.vlines(z['K'],*z['t_CI95_percent'],color=colors[schedule],alpha=.5,lw=1)
        ax.axhline(x['Source']['t'],color='#555555',ls=':',lw=1,label='Source')
        ax.set(title=f'{ds.upper()} · {stage} · n={x["requested_queries"]}',xlabel='Requested observations K (capped at T)',ylabel='Mean tIoU (%)',xticks=[2,4,8,16,32]);ax.grid(axis='y',color='#e5e5e5',lw=.6)
    axes[1].legend(frameon=False,fontsize=8);fig.suptitle('Native joint observation budgets and temporal quality',fontsize=12);fig.text(.06,.025,'95% parent-bootstrap CIs; per-domain y ranges. Fixed parameters, no accuracy-based selection. Actual K/T is exported separately.',fontsize=8)
    finish(fig,'K_tiou')
    fig,axes=plt.subplots(1,2,figsize=(9,4.1))
    for ax,(ds,x)in zip(axes,data.items()):
        rr=sorted((z for z in x['plot_rows']if z['schedule']=='Hybrid'),key=lambda z:z['K'])
        if rr:ax.boxplot([np.asarray(z['ratios'])*100 for z in rr],tick_labels=[str(z['K'])for z in rr],showfliers=True,patch_artist=False,medianprops={'color':colors['Hybrid']})
        ax.set(title=f'{ds.upper()} · {stage} · n={x["requested_queries"]}',xlabel='Requested hybrid observations K',ylabel='Observed sampled frames K/T (%)',ylim=(0,105));ax.grid(axis='y',color='#e5e5e5',lw=.6)
    fig.suptitle('Actual observation fractions on each sampled video',fontsize=12);fig.text(.06,.025,'Boxes: median and interquartile range; whiskers 1.5×IQR; all outliers retained. These are distributions, not CIs.',fontsize=8)
    finish(fig,'actual_K_over_T')
    atomic(D/(prefix+'_'+stage+'_figures_receipt.json'),dict(status='rendered_requires_view_image_QA',stage=stage,plots=plots,code_sha256=sha(Path(__file__)),score_receipt_sha256=sha(D/(prefix+'_'+stage+'_score_receipt.json')),all_incomplete_configurations_retained=True,new_predictions=0,new_GT_scoring=0,Fig1_unchanged=True))
    print(json.dumps(dict(stage=stage,figures=len(plots),status='rendered_requires_view_image_QA')))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['screen','confirm'],required=True);ap.add_argument('--prefix',choices=['budget','budget_v2'],default='budget_v2');args=ap.parse_args();main(args.stage,args.prefix)

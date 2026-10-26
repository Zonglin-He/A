"""Read-only signed-tail/cost/figure supplement to complete offline audits."""
from review_common import *
import argparse,csv


def run(ds):
    import numpy as np
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=D/'offline_current'/ds;folder=P/'offline_current'/ds
    audit=read(dest/'audit_receipt.json');root=read(dest/'root_export_receipt.json')
    assert audit['status']=='independent_full_offline_rederive_exact'
    assert root['audit_sha256']==sha(dest/'audit_receipt.json')
    assert audit['derived_sha256']==sha(dest/'derived.json')
    assert root['report_sha256']==sha(folder/'report.json')
    rows=read(dest/'derived.json')['rows'];report=read(folder/'report.json');assert len(rows)==COUNTS[ds]
    pairs=[('Ours','Source'),('Ours','SharedCorrection'),('Ours','FinalIndependent')]
    tails={}
    with(folder/'quality_all_signed_tails.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(['query','quality_group','comparison','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union_once'])
        for group in report['quality_groups']:
            tails[group]={}
            for a,b in pairs:
                rr=[(i,x) for i,x in enumerate(rows,1) if x['quality_group']==group]
                dt=np.asarray([100*(x['main'][a]['t']-x['main'][b]['t']) for _,x in rr])
                dv=np.asarray([100*(x['main'][a]['v']-x['main'][b]['v']) for _,x in rr])
                v=dict(queries=len(rr),severe_t=int((dt< -5).sum()),severe_v=int((dv< -5).sum()),
                       severe_union_once=int(((dt< -5)|(dv< -5)).sum()),
                       rescue_union_once=int(((dt>5)|(dv>5)).sum()))
                tails[group][a+'-'+b]=v
                for (i,x),t,s in zip(rr,dt,dv):
                    w.writerow([f'Q{i:05d}',group,a+'-'+b,t,s,t< -5,s< -5,(t< -5)or(s< -5),(t>5)or(s>5)])
    atomic(folder/'quality_paired_tails.json',tails)
    with(folder/'distance_all_signed_tails.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n')
        w.writerow(['query','distance_bin','annotated_unobserved_frames','comparison','delta_fixed_support_sIoU_pp','severe_over5pp','rescue_over5pp'])
        for i,x in enumerate(rows,1):
            for bi,frames in enumerate(x['distance']):
                if not frames:continue
                for a,b in pairs+[('Ours','Interpolation')]:
                    delta=100*float(np.mean([y['values'][a]-y['values'][b] for y in frames]))
                    w.writerow([f'Q{i:05d}',bi,len(frames),a+'-'+b,delta,delta< -5,delta>5])
    fig,axes=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
    names=list(report['quality_groups']);xs=np.arange(len(names));means=[];lo=[];hi=[]
    for name in names:
        p=report['quality_groups'][name]['paired'].get('Ours-Source',{}).get('v')
        means.append(p['mean'] if p else np.nan)
        lo.append(p['mean']-p['CI95'][0] if p else 0)
        hi.append(p['CI95'][1]-p['mean'] if p else 0)
    ax=axes[0];ax.axhline(0,color='gray',lw=1);ax.errorbar(xs,means,yerr=[lo,hi],fmt='o',capsize=4,color='#2268a8')
    ax.set_xticks(xs,['[0,.25)','[.25,.5)','[.5,.75)','[.75,1]','No GT\nanchor'])
    ax.set_xlabel('Mean expert-anchor IoU (annotated anchors only)')
    ax.set_ylabel('Current minus native Source m_vIoU (pp)')
    ax.set_title('Offline quality groups: whole-system outcomes')
    for i,name in enumerate(names):ax.annotate('n='+str(report['quality_groups'][name]['queries']),(xs[i],means[i]),xytext=(0,12),textcoords='offset points',ha='center',fontsize=8)
    ax=axes[1];ax.axhline(0,color='gray',lw=1)
    for key,label,color,shift in [('Ours-Source','Current − Source','#2268a8',-.12),('Ours-SharedCorrection','Current − shared-corrected boxes','#d85930',0),('Ours-Interpolation','Current − interpolation','#8554b0',.12)]:
        ii=[];yy=[];ll=[];hh=[]
        for bi,c in enumerate(report['distance_curve']):
            if not c['queries']:continue
            p=c['paired'][key];ii.append(bi+shift);yy.append(p['mean']);ll.append(p['mean']-p['CI95'][0]);hh.append(p['CI95'][1]-p['mean'])
        ax.errorbar(ii,yy,yerr=[ll,hh],fmt='o-',capsize=3,color=color,label=label,lw=1)
    labels=['[0,.5)','[.5,1)','[1,2)','[2,4)','[4,∞)']
    ax.set_xticks(range(5),[s+'\n'+str(c['annotated_unobserved_frames'])+' frames' for s,c in zip(labels,report['distance_curve'])])
    ax.set_xlabel('Nearest physical expert observation distance (seconds)')
    ax.set_ylabel('Fixed-support unobserved sIoU difference (pp)')
    ax.set_title('Same annotated frame support; query-equal bin means')
    ax.legend(fontsize=8)
    fig.suptitle(ds+': descriptive exposed-target analysis; pointwise parent-cluster 95% CI',fontsize=12)
    png=dest/'quality_distance_v1.png';svg=dest/'quality_distance_v1.svg'
    fig.savefig(png,dpi=150);fig.savefig(svg);plt.close(fig)
    costs=dict(derive_worker=read(dest/'sealed.json')['worker'],
               derive_elapsed_wall_seconds=read(dest/'sealed.json')['seconds'],
               audit_worker=audit['worker'],supplement_worker=identity(),
               audit_standalone_elapsed_clock_saved=False,
               missing_cost_rule='Do not invent zero for audit/export clocks that were not separately saved.',
               inherited_Source_expert_acquisition_nonzero=True,new_Source=0,new_expert=0,new_model=0,new_prediction=0,
               cost_scope='Offline annotation-supported IoU / cached-array analysis; elapsed wall is not process CPU or uncached deployment. Original producer/teacher costs remain nonzero and nested; no added shared TTA Source costs.')
    atomic(folder/'offline_cost_segments.json',costs)
    receipt=dict(status='supplement_exported_pending_root_PNG_view',dataset=ds,queries=len(rows),
                 code_sha256=sha(Path(__file__)),audit_sha256=sha(dest/'audit_receipt.json'),
                 root_export_sha256=sha(dest/'root_export_receipt.json'),
                 quality_tails_sha256=sha(folder/'quality_paired_tails.json'),
                 PNG_sha256=sha(png),SVG_sha256=sha(svg),
                 quality_main_t_v_severe_union_once=True,distance_same_support_signed=True,
                 pointwise_parent_CI=True,offline_GT_not_online_gate=True)
    atomic(dest/'supplement_receipt.json',receipt);atomic(folder/'supplement_receipt.json',receipt)
    print(json.dumps(dict(dataset=ds,status=receipt['status'],queries=len(rows))))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dataset',choices=COUNTS,required=True)
    run(a.parse_args().dataset)

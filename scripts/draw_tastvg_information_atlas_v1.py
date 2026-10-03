"""Render frozen atlas statistics and position-removed endpoint diagnostics."""
import os
os.environ['MPLBACKEND']='Agg'
os.environ['OPENBLAS_NUM_THREADS']='2'
import sys,json,collections
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/tastvg_temporal_information_atlas/2026-10-03'
GROUPS=['source_validation','target_clean','target_corrupt']
TASKS=['precision','recall','tiou','delta']
VIEWS=['Endpoint','Inside','Context','Contrast','Full','Geometry','Full-shuffle']

def read(p):return json.loads(Path(p).read_text())
def write(p,z):Path(p).write_text(json.dumps(z,indent=2,allow_nan=False)+'\n')
def scalar(summary,group,key,field='r2'):return summary[group]['metrics'][key]['metrics'][field]['mean']
def save(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=220,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

def endpoint_summary(rows):
    parts={'source_validation':[r for r in rows if r['domain']=='source']}
    for mode in ['clean','corrupt']:
        group=[r for r in rows if r['domain']=='target' and (r['condition']=='clean')==(mode=='clean')]
        parts['target_'+mode]=group
        for panel in ['search','confirm']:parts['target_'+mode+'_'+panel]=[r for r in group if r['panel']==panel]
    result={}
    for label,rr in parts.items():
        z={}
        for key in rr[0]['position_removed_endpoints']:
            grouped=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
            for r in rr:
                item=r['position_removed_endpoints'][key]
                grouped[r['source_index']][r['order']][r['condition']].append([item['endpoint_abs_error'],item['inferred_endpoint_std']])
            values=[]
            for source,orders in sorted(grouped.items()):
                values.append(np.mean([np.mean([np.mean(v,axis=0) for c,v in conditions.items()],axis=0)
                    for o,conditions in orders.items()],axis=0))
            a=np.array(values);w=np.random.default_rng(20261003).multinomial(len(a),np.full(len(a),1/len(a)),size=10000)/len(a)
            b=w@a
            z[key]=dict(sources=len(a),endpoint_abs_error=dict(mean=float(a[:,0].mean()),ci95=np.quantile(b[:,0],[.025,.975]).tolist()),
                inferred_endpoint_std=dict(mean=float(a[:,1].mean()),ci95=np.quantile(b[:,1],[.025,.975]).tolist()))
        result[label]=z
    return result

def title(group,n):
    return {'source_validation':'Source validation','target_clean':'Target clean','target_corrupt':'Target corrupt'}[group]+f'  ·  {n} sources'

def heat(ax,data,xlabels,ylabels,heading):
    im=ax.imshow(data,cmap='RdBu',norm=TwoSlopeNorm(vmin=-.5,vcenter=0.,vmax=1.),aspect='auto')
    ax.set_xticks(range(len(xlabels)),xlabels,fontsize=10);ax.set_yticks(range(len(ylabels)),ylabels,fontsize=10)
    ax.set_title(heading,fontsize=11,pad=14,fontweight='medium')
    ax.tick_params(length=0,pad=7)
    for sp in ax.spines.values():sp.set_visible(False)
    for (i,j),v in np.ndenumerate(data):
        if np.isfinite(v):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=10,
            color='white' if v>.74 or v<-.33 else '#17212b')
        else:ax.text(j,i,'n/a',ha='center',va='center',fontsize=10)
    ax.set_xticks(np.arange(-.5,len(xlabels),1),minor=True);ax.set_yticks(np.arange(-.5,len(ylabels),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1.8);ax.tick_params(which='minor',length=0)
    return im

def run(folder):
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'axes.titlesize':11})
    out=folder/'figures';out.mkdir(exist_ok=True);figure_data={}
    for ds,label in [('vidstg','VidSTG'),('hc2','HC-STVG-v2')]:
        s=read(folder/ds/'SUMMARY.json');rows=read(folder/ds/'ROWS.json')
        write(folder/ds/'ENDPOINT_DIAGNOSTICS.json',endpoint_summary(rows))
        for family in ['candidate','frame']:
            fig,axes=plt.subplots(1,3,figsize=(14.2,4.6 if family=='candidate' else 3.35),layout='constrained')
            matrices=[]
            for ax,group in zip(axes,GROUPS):
                if family=='candidate':
                    matrix=np.array([[scalar(s,group,f'candidate/{"Full" if v=="Full-shuffle" else v}/{t}/{"shuffle" if v=="Full-shuffle" else "real"}') for t in TASKS] for v in VIEWS])
                    cols=['Precision','Recall','tIoU','Δ tIoU'];labels=['Endpoint (512)','Inside (256)','Context (512)','Contrast (512)','Full (1792)','Geometry (3)','Full · shuffled']
                else:
                    tasks=['position','event','start_distance','end_distance','phase'];labels=['Hidden (256)','Position (1)','Hidden · shuffled']
                    matrix=np.array([[scalar(s,group,f'frame/{v}/{t}/{c}','auc' if t=='event' else 'r2')
                        for t in tasks] for v,c in [('Hidden','real'),('Geometry','real'),('Hidden','shuffle')]])
                    cols=['Position\nR²','Event\nAUROC','Start dist.\nR²','End dist.\nR²','Phase\nR²']
                im=heat(ax,matrix,cols,labels,title(group,s[group]['coverage']['sources']));matrices.append(matrix.tolist())
            cb=fig.colorbar(im,ax=axes,shrink=.79,pad=.025,ticks=[-.5,0,.5,1.]);cb.ax.set_yticklabels(['≤ −0.5','0','0.5','1.0'])
            cb.set_label('R²' if family=='candidate' else 'R² / AUROC',fontsize=10)
            fig.suptitle(f'{label} · '+('Candidate information' if family=='candidate' else 'Frame information'),fontsize=14,fontweight='medium')
            save(fig,out/f'{ds}_{family}_atlas')
            figure_data[f'{ds}/{family}']=dict(groups=GROUPS,rows=labels,columns=cols,values=matrices,
                color_clips_at=-.5,values_not_clipped=True)
        # Paired controls are plotted separately; the figure does not select a view.
        figure_data[f'{ds}/endpoints']='ENDPOINT_DIAGNOSTICS.json; observed-clip units, lower error is better'
    fig,axes=plt.subplots(1,2,figsize=(12,3.5),layout='constrained')
    pairs={}
    for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        z=read(folder/ds/'SUMMARY.json')['target_corrupt'];xx=np.arange(4);pairs[ds]={}
        for tag,offset,color,name in [('over_geometry',-.09,'#156b99','Full − Geometry'),('over_shuffle',.09,'#d1822b','Full − Shuffled')]:
            rr=[z['paired_differences'][f'candidate/Full/{t}/real/{tag}'] for t in TASKS]
            m=np.array([r['mean'] for r in rr]);ci=np.array([r['ci95'] for r in rr])
            ax.errorbar(xx+offset,m,yerr=np.vstack([m-ci[:,0],ci[:,1]-m]),fmt='o',markersize=5,color=color,capsize=3,lw=1.4,label=name)
            pairs[ds][tag]=rr
        ax.axhline(0,color='#9099a2',ls='--',lw=1);ax.set_xticks(xx,['Precision','Recall','tIoU','Δ tIoU']);ax.set_title(label)
        ax.set_ylabel('Paired difference in R²');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.13)
    axes[0].legend(frameon=False,fontsize=9,loc='upper left');save(fig,out/'candidate_paired_controls')
    figure_data['paired_controls']=pairs;write(folder/'FIGURE_DATA.json',figure_data)
    print('ATLAS_FIGURES',10,'PNG/PDF files; endpoint diagnostics saved')

if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT)

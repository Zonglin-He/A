"""Export source-backed query intervention figures, with paired uncertainty."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/tastvg_query_swap_specificity/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def save(fig,p):
    fig.savefig(p.with_suffix('.png'),dpi=230,bbox_inches='tight',facecolor='white')
    fig.savefig(p.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def intervals(ax,x,values,color,label):
    means=np.array([r['mean'] for r in values]);ci=np.array([r['ci95'] for r in values])
    # Draw bounds directly: percentile intervals need not contain the estimate.
    ax.vlines(x,ci[:,0],ci[:,1],color=color,lw=1.5)
    ax.plot(x,means,'o',color=color,ms=5,label=label)
    ax.hlines(ci[:,0],x-.035,x+.035,color=color,lw=1.2)
    ax.hlines(ci[:,1],x-.035,x+.035,color=color,lw=1.2)
def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
def run(folder):
    folder=Path(folder);out=folder/'figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':10})
    summaries={d:read(folder/d/'SUMMARY.json') for d in ['vidstg','hc2']};data={}
    fig,axes=plt.subplots(1,2,figsize=(10.7,3.25),layout='constrained')
    for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        s=summaries[ds]['target_corrupt'];xx=np.arange(3);data[ds]={}
        for arm,offset,color,label in [('true',-.09,'#177393','Original query'),('swap',.09,'#d4833a','Swapped query')]:
            v=[s['metrics'][f'{arm}/candidate/Full/{t}/real']['metrics']['r2'] for t in ['precision','recall','tiou']]
            intervals(ax,xx+offset,v,color,label);data[ds][arm]=v
        ax.axhline(0,color='#a7adb3',ls='--',lw=1);style(ax)
        ax.set_xticks(xx,['Precision','Recall','tIoU']);ax.set_title(title,pad=10);ax.set_ylabel('Source-balanced R²')
    axes[0].legend(frameon=False,fontsize=9,loc='lower right');save(fig,out/'candidate_true_swap')
    fig,axes=plt.subplots(1,3,figsize=(12,3.2),layout='constrained');gaps={}
    tasks=['precision','recall','tiou'];xx=np.arange(3)
    for ax,ds,title in zip(axes[:2],['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        s=summaries[ds];gaps[ds]={}
        for mode,offset,color,label in [('corrupt',-.075,'#177393','Corrupted'),('clean',.075,'#7b8593','Clean')]:
            v=[s['target_'+mode]['paired_true_minus_swap'][f'candidate/Full/{t}/real']['r2'] for t in tasks]
            intervals(ax,xx+offset,v,color,label);gaps[ds][mode]=v
        ax.axhline(0,color='#9ba4ad',ls='--',lw=1);ax.set_xticks(xx,['Precision','Recall','tIoU'])
        ax.set_title(title,pad=10);ax.set_ylabel('Paired original − swapped R²');style(ax)
    ax=axes[2];xx=np.arange(2)
    for mode,offset,color,label in [('corrupt',-.075,'#177393','Corrupted'),('clean',.075,'#7b8593','Clean')]:
        v=[summaries[ds]['target_'+mode]['paired_true_minus_swap']['frame/Hidden/event/real']['auc'] for ds in ['vidstg','hc2']]
        intervals(ax,xx+offset,v,color,label);gaps['event_'+mode]=v
    ax.axhline(0,color='#9ba4ad',ls='--',lw=1);ax.set_xticks(xx,['VidSTG','HC-STVG-v2']);style(ax)
    ax.set_title('Frame event membership',pad=10);ax.set_ylabel('Paired original − swapped AUROC')
    axes[0].legend(frameon=False,fontsize=9);save(fig,out/'paired_query_gap')
    fig,axes=plt.subplots(1,2,figsize=(11.1,4.1),layout='constrained');blocks={}
    views=['Endpoint','Inside','Context','Contrast','Full','Geometry']
    for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        s=summaries[ds]['target_corrupt'];a=np.array([[s['paired_true_minus_swap'][f'candidate/{v}/{t}/real']['r2']['mean'] for t in tasks] for v in views]);blocks[ds]=a.tolist()
        limit=max(.01,float(np.max(abs(a))));im=ax.imshow(a,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
        ax.set_xticks(range(3),['Precision','Recall','tIoU']);ax.set_yticks(range(6),views);ax.set_title(title,pad=12)
        for (i,j),x in np.ndenumerate(a):ax.text(j,i,f'{x:+.3f}',ha='center',va='center',fontsize=10,color='white' if abs(x)>.7*limit else '#182530')
        ax.tick_params(length=0,pad=7);ax.spines[:].set_visible(False)
        fig.colorbar(im,ax=ax,shrink=.85,pad=.035,label='Original − swapped R²')
    save(fig,out/'block_query_gaps')
    (folder/'FIGURE_DATA.json').write_text(json.dumps(dict(candidate=data,paired=gaps,blocks=blocks,
        uncertainty='10,000 paired source-bootstrap, fixed donor mapping',units='R² and AUROC, not vIoU gains'),indent=2,allow_nan=False)+'\n')
    print('QUERY_SWAP_FIGURES 6 PNG/PDF files')
if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT)

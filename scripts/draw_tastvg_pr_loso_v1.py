"""Publication vector plots from sealed anonymous source-held-out summaries."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'results/tastvg_pr_loso/2026-10-03'
DEST=OUT/'figures';DEST.mkdir(parents=True,exist_ok=True)
DATA={ds:json.loads((OUT/ds/'SUMMARY.json').read_text()) for ds in ['vidstg','hc2']}
ROLES=['P_A','R_A','P_W','R_W'];LABELS=[r'$P_A$',r'$R_A$',r'$P_W$',r'$R_W$']
variants=['all/in_sample','all/nmax','role/in_sample','role/nmax']
COLORS=['#90A7C4','#294B71','#8CC8BE','#217E72'];NAMES=['M-all · in-sample','M-all · source-LOSO','M-role · in-sample','M-role · source-LOSO']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
used={};written=[]
def save(fig,name):
    for ext in ['png','pdf']:
        path=DEST/f'{name}.{ext}';assert not path.exists();fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white');written.append(path.name)
    plt.close(fig)
def err(vals):
    m=np.array([v['mean'] for v in vals]);c=np.array([v['ci95'] for v in vals]);return m,np.maximum(0,np.array([m-c[:,0],c[:,1]-m])),c

fig,axes=plt.subplots(1,2,figsize=(11.6,3.7),layout='constrained')
for ax,ds in zip(axes,DATA):
    z=DATA[ds]['corrupt'];upper=[]
    for i,v in enumerate(variants):
        vals=[z['regression'][v][r]['metrics']['mae'] for r in ROLES];m,e,c=err(vals);upper.extend(c[:,1]);x=np.arange(4)+(i-1.5)*.18
        ax.bar(x,m,.17,yerr=e,color=COLORS[i],label=NAMES[i],capsize=2.5,error_kw={'elinewidth':.8},zorder=3);used[ds+'/'+v+'/mae']=vals
    ax.set_xticks(range(4),LABELS);ax.set_ylim(0,max(upper)*1.16);ax.set_ylabel('Raw P/R MAE');ax.set_title(('VidSTG' if ds=='vidstg' else 'HC-STVG-v2')+' · same search sources')
    ax.grid(axis='y',alpha=.15,zorder=0)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='outside lower center',frameon=False);save(fig,'source_holdout_generalization')

fig,axes=plt.subplots(2,4,figsize=(12.2,5.6),layout='constrained')
for i,ds in enumerate(DATA):
    xs=[4,8,12,15 if ds=='vidstg' else 13];z=DATA[ds]['corrupt']
    for j,r in enumerate(ROLES):
        ax=axes[i,j];bounds=[]
        for pop,col in [('all',COLORS[1]),('role',COLORS[3])]:
            vals=[z['regression'][pop+'/'+l][r]['metrics']['mae'] for l in ['n4','n8','n12','nmax']];m,e,c=err(vals);bounds.extend(c[:,1])
            ax.plot(xs,m,'o-',lw=1.7,ms=4,color=col,label='M-'+pop);ax.fill_between(xs,c[:,0],c[:,1],color=col,alpha=.11)
            used[ds+'/'+pop+'/'+r+'/learning_curve']={'sources':xs,'metrics':vals}
        ax.set_xticks(xs);ax.set_xlim(3.5,xs[-1]+.5);ax.set_ylim(0,max(bounds)*1.08)
        ax.set_title(('VidSTG' if ds=='vidstg' else 'HC-STVG-v2')+' · '+LABELS[j]);ax.grid(alpha=.15)
        if j==0:ax.set_ylabel('Held-source MAE')
        if i==1:ax.set_xlabel('Training sources per fold')
axes[0,0].legend(frameon=False);save(fig,'source_count_learning_curve')

fig,axes=plt.subplots(1,2,figsize=(10.8,3.5),layout='constrained')
for ax,ds in zip(axes,DATA):
    z=DATA[ds]['corrupt']
    for i,v in enumerate(variants):
        vals=[z['decision'][v]['metrics'][m] for m in ['auc','balanced_accuracy']];y,e,c=err(vals)
        ax.bar(np.arange(2)+(i-1.5)*.18,y,.17,yerr=e,color=COLORS[i],label=NAMES[i],capsize=2.5,error_kw={'elinewidth':.8},zorder=3);used[ds+'/'+v+'/decision']=vals
    ax.set_xticks(range(2),['AUROC','Balanced accuracy']);ax.set_ylim(0,1.12);ax.axhline(.5,color='#949BA3',ls='--',lw=.8)
    ax.set_ylabel('Fixed analytic correction decision');ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2');ax.grid(axis='y',alpha=.15,zorder=0)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,ncol=4,loc='outside lower center',frameon=False);save(fig,'source_holdout_decision')

binding=dict(data=used,files={n:hashlib.sha256((DEST/n).read_bytes()).hexdigest() for n in written},
    source_summaries={ds:hashlib.sha256((OUT/ds/'SUMMARY.json').read_bytes()).hexdigest() for ds in DATA},
    intervals='conditional fixed-prediction whole-source10000 bootstrap; shaded pointwise intervals, no refits')
p=OUT/'FIGURE_BINDING.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n')
print('LOSO_FIGURES_WRITTEN',len(written))

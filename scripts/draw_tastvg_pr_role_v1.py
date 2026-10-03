"""Source-backed population comparison figures; no model/data inference."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'results/tastvg_pr_role/2026-10-03'
DEST=OUT/'figures';DEST.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','axes.labelcolor':'#283442','text.color':'#283442'})
DATA={ds:json.loads((OUT/ds/'SUMMARY.json').read_text()) for ds in ['vidstg','hc2']}
COLORS={'all':'#405E86','role':'#33A595'}
ROLES=['P_A','R_A','P_W','R_W'];LABELS=[r'$P_A$',r'$R_A$',r'$P_W$',r'$R_W$']
used={};written=[]
def save(fig,name):
    for ext in ['png','pdf']:
        p=DEST/(name+'.'+ext);assert not p.exists()
        fig.savefig(p,dpi=230,bbox_inches='tight');written.append(p.name)
    plt.close(fig)
def bar(ax,series,labels,ylim=None):
    xs=np.arange(len(labels));upper=[]
    for j,fn in enumerate(['all','role']):
        packs=series[fn];means=np.array([p['mean'] for p in packs]);bounds=np.array([p['ci95'] for p in packs])
        loc=xs+(j-.5)*.34
        ax.bar(loc,means,.31,color=COLORS[fn],label='M-all' if fn=='all' else 'M-role',zorder=3)
        ax.errorbar(loc,means,yerr=np.maximum(0,np.array([means-bounds[:,0],bounds[:,1]-means])),
                    fmt='none',ecolor='#374151',capsize=3,lw=.8,zorder=4)
        upper.extend(bounds[:,1]);ax.set_xticks(xs,labels)
        for x,y in zip(loc,means):ax.text(x,y+.014,f'{y:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_ylim(ylim if ylim else (0,max(upper)*1.17+.025));ax.grid(axis='y',alpha=.18,zorder=0)

fig,axes=plt.subplots(2,2,figsize=(10.4,6.0),layout='constrained')
for i,ds in enumerate(['vidstg','hc2']):
    for j,panel in enumerate(['search_corrupt','confirm_corrupt']):
        z=DATA[ds][panel];series={fn:[z['regression'][fn][r]['metrics']['mae'] for r in ROLES] for fn in ['all','role']}
        bar(axes[i,j],series,LABELS);axes[i,j].set_ylabel('Raw P/R MAE (lower is better)')
        axes[i,j].set_title(('VidSTG' if ds=='vidstg' else 'HC-STVG-v2')+' · '+('search (in-sample)' if j==0 else 'confirmation'))
        used[ds+'/'+panel+'/MAE']=series
axes[0,0].legend(frameon=False,ncol=2,loc='upper left')
save(fig,'role_fit_generalization')

fig,axes=plt.subplots(1,2,figsize=(9.6,3.2),layout='constrained')
for ax,ds in zip(axes,['vidstg','hc2']):
    z=DATA[ds]['confirm_corrupt'];series={fn:[z['decision'][fn]['metrics'][k] for k in ['auc','balanced_accuracy']] for fn in ['all','role']}
    bar(ax,series,['AUROC','Balanced accuracy'],(0,1.1));ax.axhline(.5,color='#9AA2AB',ls='--',lw=.8)
    ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2');ax.set_ylabel('Fixed analytic decision')
    used[ds+'/confirm_corrupt/decision']=series
axes[0].legend(frameon=False,ncol=2,loc='upper left');save(fig,'role_confirmation_decision')

fig,axes=plt.subplots(1,2,figsize=(9.6,3.2),layout='constrained')
for ax,ds in zip(axes,['vidstg','hc2']):
    vals=[DATA[ds]['confirm_corrupt']['paired_role_minus_all_regression'][r]['mae'] for r in ROLES]
    mean=np.array([p['mean'] for p in vals]);ci=np.array([p['ci95'] for p in vals]);ys=np.arange(4)
    ax.errorbar(mean,ys,xerr=np.maximum(0,np.array([mean-ci[:,0],ci[:,1]-mean])),fmt='o',color=COLORS['role'],ecolor=COLORS['all'],capsize=4)
    ax.axvline(0,color='#9AA2AB',ls='--',lw=.8);ax.set_yticks(ys,LABELS);ax.invert_yaxis()
    ax.set_xlim(min(ci[:,0])-.018,max(ci[:,1])+.018)
    ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2');ax.set_xlabel('MAE change: M-role − M-all');ax.grid(axis='x',alpha=.18)
    used[ds+'/confirm_corrupt/paired_MAE']=vals
save(fig,'role_paired_confirmation_mae')
binding=dict(data=used,files={n:hashlib.sha256((DEST/n).read_bytes()).hexdigest() for n in written},
    source_summaries={ds:hashlib.sha256((OUT/ds/'SUMMARY.json').read_bytes()).hexdigest() for ds in DATA},
    intervals='paired whole-source bootstrap 10000 for differences; pointwise whole-source intervals for bars')
p=OUT/'FIGURE_BINDING.json';assert not p.exists();p.write_text(json.dumps(binding,indent=2)+'\n')
print('ROLE_FIGURES_WRITTEN',len(written))

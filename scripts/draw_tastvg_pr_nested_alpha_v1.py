"""Publication figures/CSV from sealed anonymous scalars; no private data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MPLBACKEND']='Agg'
import sys,json,csv,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'results/tastvg_pr_nested_alpha/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.titleweight':'bold'})
roles=['P_A','R_A','P_W','R_W'];pops=['all','role'];dsnames=['vidstg','hc2'];names=['VidSTG','HC-STVG-v2']
colors={'fixed':'#6F7F91','nested':'#1D9388'}
files=[];plotdir=OUT/'figures';plotdir.mkdir(exist_ok=True)
for metric,title in [('mae','Held-source absolute error'),('r2','Held-source coefficient of determination')]:
    fig,axes=plt.subplots(2,2,figsize=(11.4,6.3),sharex=True,constrained_layout=True)
    for col,ds in enumerate(dsnames):
        s=read(OUT/ds/'SUMMARY.json')['corrupt']
        for row,pop in enumerate(pops):
            ax=axes[row,col]
            for regime,offset in [('fixed',-.1),('nested',.1)]:
                z=[s['regression'][pop+'/'+regime][k]['metrics'][metric] for k in roles]
                y=np.array([v['mean'] for v in z]);ci=np.array([v['ci95'] for v in z])
                ax.errorbar(np.arange(4)+offset,y,yerr=np.stack([y-ci[:,0],ci[:,1]-y]),fmt='o',
                    markersize=5,color=colors[regime],elinewidth=1.3,capsize=3,label=regime.capitalize()+' alpha')
            ax.set_title(names[col]+' · M-'+pop);ax.set_xticks(range(4),['$P_A$','$R_A$','$P_W$','$R_W$'])
            ax.set_ylabel('MAE' if metric=='mae' else '$R^2$');ax.grid(axis='y',alpha=.18)
            if metric=='r2':ax.axhline(0,color='#b4b9bd',lw=1,ls='--')
            if row==0 and col==0:ax.legend(frameon=False,loc='best',fontsize=9)
    for ext in ['png','pdf']:
        p=plotdir/f'{metric}_fixed_vs_nested.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight');files.append(p)
    plt.close(fig)
dist=read(OUT/'ALPHA_DISTRIBUTION.json');grid=read(OUT/'CONFIG.json')['alpha_grid']
fig,axes=plt.subplots(2,2,figsize=(10.7,5.7),sharex=True,constrained_layout=True)
for col,ds in enumerate(dsnames):
    for row,n in enumerate(['P','R']):
        ax=axes[row,col]
        for pop,offset,color in [('all',-.17,'#488FB3'),('role',.17,'#E29A49')]:
            y=[dist[ds][pop][n][str(a)] for a in grid]
            ax.bar(np.arange(7)+offset,y,width=.33,color=color,label='M-'+pop)
        ax.set_title(names[col]+' · '+n+' head');ax.set_ylabel('Outer source folds');ax.set_ylim(0,17)
        ax.set_xticks(range(7),['$10^{-3}$','$10^{-2}$','$10^{-1}$','$1$','$10$','$100$','$1000$'])
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        if row==0 and col==0:ax.legend(frameon=False)
        if row==1:ax.set_xlabel('Inner-selected ridge alpha')
for ext in ['png','pdf']:
    p=plotdir/f'alpha_frequency.{ext}';fig.savefig(p,dpi=220,bbox_inches='tight');files.append(p)
plt.close(fig)
def csvwrite(name,records):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
recs=[]
for ds in dsnames:
    s=read(OUT/ds/'SUMMARY.json')
    for panel in ['all','corrupt','clean']:
        for pop in pops:
            for regime in ['fixed','nested']:
                for role in roles:
                    for m,z in s[panel]['regression'][pop+'/'+regime][role]['metrics'].items():
                        recs.append(dict(dataset=ds,panel=panel,population=pop,regime=regime,role=role,metric=m,
                            mean=z['mean'],ci_low=z['ci95'][0],ci_high=z['ci95'][1],undefined=z['bootstrap_undefined']))
csvwrite('REGRESSION.csv',recs)
recs=[]
for m in read(OUT/'FIT_SUMMARY.json'):
    for n,z in m['heads'].items():recs.append(dict(dataset=m['dataset'],held_source=m['held_source'],population=m['population'],head=n,
        alpha=z['alpha'],inner_MAE=z['inner_source_MAE'][grid.index(z['alpha'])],training_MAE=z['training_MAE'],training_MSE=z['training_MSE']))
csvwrite('ALPHA_SELECTION.csv',recs)
recs=[]
for c in read(OUT/'INNER_CV.json'):
    for n,cc in c['heads'].items():
        for z in cc:
            for e in z['scores']:recs.append(dict(dataset=c['dataset'],outer_source=c['held_source'],population=c['population'],head=n,
                inner_source=z['inner_source'],alpha=e['alpha'],A_MAE=e['A'],W_MAE=e['W'],role_MAE=e['mean'],clean_MAE=e['clean'],corrupt_MAE=e['corrupt']))
csvwrite('INNER_SOURCE_MAE.csv',recs)
receipt=dict(status='rendered_pending_visual_review',files={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    source_summaries={ds:hashlib.sha256((OUT/ds/'SUMMARY.json').read_bytes()).hexdigest() for ds in dsnames},
    metric='source-macro corrupt expert OOF; paired source bootstrap10000; no refits',figure_count=3)
p=OUT/'FIGURES.json';assert not p.exists();p.write_text(json.dumps(receipt,indent=2)+'\n')
print('Rendered 3 PNG/PDF figure pairs and regression/alpha/inner-CV CSV tables.')

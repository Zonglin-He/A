"""Publication plots from sealed anonymous statistics; no new scoring."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_anchor_quality/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'axes.titleweight':'bold','savefig.facecolor':'white'})
COLORS={'L32':'#9AA5B1','Pair-Norm':'#9569AD','M0':'#4276B2','M1':'#D57E3C'}
figdir=OUT/'figures';figdir.mkdir(exist_ok=True)
data={ds:dict(summary=read(OUT/'confirm'/ds/'SUMMARY.json'),strata=read(OUT/'confirm'/ds/'ANCHOR_STRATA.json'),
                    LOSO=read(OUT/ds/'LOSO_SUMMARY.json')) for ds in ['vidstg','hc2']}
(OUT/'FIGURE_DATA.json').write_text(json.dumps(data,indent=2)+'\n')
exports=[]
def save(fig,name):
    fig.tight_layout(pad=1.6)
    for ext in ['png','pdf']:
        p=figdir/(name+'.'+ext);fig.savefig(p,dpi=240,bbox_inches='tight');exports.append(dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size))
    plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10.6,3.8),sharey=True)
for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
    arm=['L32','Pair-Norm','M0','M1'];s=data[ds]['summary']['corruption']['all']
    for x,a in enumerate(arm):
        m=s[a]['metrics']['dv'];mu=m['mean']*100;lo,hi=np.array(m['ci95'])*100
        ax.errorbar(x,mu,yerr=[[mu-lo],[hi-mu]],fmt='o',ms=7,capsize=5,color=COLORS[a],elinewidth=1.8)
        ax.annotate(f'{mu:+.3f}',(x,mu),xytext=(0,12 if mu>=0 else -19),textcoords='offset points',ha='center',fontsize=9,color=COLORS[a],
                    bbox=dict(facecolor='white',edgecolor='none',pad=1.2,alpha=.95))
    ax.axhline(0,color='#303030',lw=.8,ls='--');ax.set_xticks(range(4),arm);ax.set_xlim(-.5,3.5);ax.set_ylim(-1.9,.95)
    ax.set_title(label);ax.grid(axis='y',alpha=.16)
axes[0].set_ylabel('Complete corruption flow ΔvIoU (pp)')
save(fig,'confirmation_readout')

fig,axes=plt.subplots(1,2,figsize=(10.6,3.8),sharey=True)
for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
    qs=data[ds]['strata']['quartiles']
    for a,ls in [('L32','--'),('M1','-')]:
        vals=[q['arms'][a]['cell_means']['dt']*100 for q in qs]
        ax.plot(range(1,5),vals,marker='o',ls=ls,color=COLORS[a],label=a,lw=2,ms=6)
    ax.axhline(0,color='#303030',lw=.8);ax.set_xticks(range(1,5),['Q1 low','Q2','Q3','Q4 high'])
    ax.set_xlabel('GT anchor-quality quartile (post hoc only)');ax.set_title(label+' · 40 expert cells');ax.grid(axis='y',alpha=.16);ax.legend(frameon=False)
axes[0].set_ylabel('Cell-mean temporal correction ΔtIoU (pp)')
save(fig,'GT_anchor_strata')

fig,axes=plt.subplots(1,2,figsize=(10.6,3.8),sharey=True)
for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG · 31 source folds','HC-STVG-v2 · 16 source folds']):
    s=data[ds]['LOSO']['all_pseudo_anchors']['metrics']['MSE_improvement'];v=np.array(list(s['source_values'].values()))
    ax.scatter(np.arange(len(v)),v,color=np.where(v>0,COLORS['M0'],COLORS['M1']),s=24,zorder=3)
    ax.axhline(0,color='#303030',lw=.8);ax.axhline(s['mean'],color='#333333',lw=1,ls='--')
    ax.axhspan(*s['ci95'],color='#64748B',alpha=.12);ax.set_title(label);ax.set_xlabel('Held-out source (fixed source order)')
    ax.grid(axis='y',alpha=.16)
axes[0].set_ylabel('LOSO error reduction: M0 MSE − M1 MSE')
save(fig,'source_LOSO')
(OUT/'FIGURES.json').write_text(json.dumps(dict(figures=exports,data_sha256=sha(OUT/'FIGURE_DATA.json'),
    legend='readout CI paired10000 source-bootstrap; quartile lines descriptive cell means; LOSO points independent source folds'),indent=2)+'\n')
print(json.dumps(dict(status='rendered_pending_visual_review',figures=len(exports)//2)))

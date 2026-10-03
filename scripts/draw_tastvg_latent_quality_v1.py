"""Export source validation and target confirmation, with no target selection."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read(p):return json.loads(Path(p).read_text())

def draw(folder):
    folder=Path(folder);dest=folder/'figures';dest.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220})
    fit=read(folder/'SOURCE_FIT_SUMMARY.json');fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        for signal,color,name in [('L','#3575a5','Temporal latent (1792D)'),('G','#ce8b43','Geometry only (3D)')]:
            rr=fit[ds][signal];x=[r['alpha'] for r in rr['path']];y=[100*r['validation_top1_tIoU'] for r in rr['path']]
            ax.plot(x,y,marker='o',ms=4,lw=1.5,color=color,label=name)
            i=rr['selected_index'];ax.scatter([x[i]],[y[i]],s=75,marker='*',color=color,zorder=5)
        ax.axhline(100*fit[ds]['L']['validation_native'],color='.45',lw=1,ls='--',label='Native candidate')
        ax.axhline(100*fit[ds]['L']['validation_oracle'],color='.65',lw=1,ls=':',label='32-candidate oracle')
        ax.set_xscale('log');ax.set_xlabel('Ridge strength (source validation selects *)')
        ax.set_ylabel('Source validation top-1 tIoU (%)');ax.set_title(label);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8,loc='best');fig.savefig(dest/'source_validation.png');fig.savefig(dest/'source_validation.pdf');plt.close(fig)
    names=['L8','L32','G32','N32','U32','S32'];labels=['Latent / Old8','Latent / Expanded32','Geometry / Expanded32',
        'Native logits / Expanded32','UVTG boundaries / Expanded32','Two-view boundaries / Expanded32']
    colors=['#5a93bc','#3575a5','#ce8b43','#808b94','#9e777b','#9775a6']
    fig,axes=plt.subplots(1,2,figsize=(11.4,3.8),layout='constrained')
    for ax,ds,label in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
        summary=read(folder/'confirm'/ds/'SUMMARY.json')['corruption']['all']['metrics']
        legacy=read(folder.parents[1]/'tastvg_large_correction_evidence'/'2026-10-03'/'confirm'/ds/'SUMMARY.json')['corruption']['all']['metrics']
        for i,name in enumerate(names):
            z=summary[name+'_gain'] if name.startswith(('L','G')) else legacy[name+'_gain']
            val=100*z['mean'];lo,hi=np.array(z['ci95'])*100
            ax.errorbar(val,i,xerr=[[max(0,val-lo)],[max(0,hi-val)]],fmt='o',color=colors[i],capsize=3,lw=1.6,ms=6)
        ax.axvline(0,color='.35',ls='--',lw=1);ax.set_yticks(range(len(names)),labels);ax.invert_yaxis()
        ax.set_xlabel('Corruption full-flow vIoU change vs A8 (pp)')
        ax.set_title(label+' — confirmation');ax.grid(axis='x',alpha=.15)
    fig.savefig(dest/'target_confirmation.png');fig.savefig(dest/'target_confirmation.pdf');plt.close(fig)
    return [str(p) for p in sorted(dest.glob('*'))]

if __name__=='__main__':print('\n'.join(draw(sys.argv[1])))

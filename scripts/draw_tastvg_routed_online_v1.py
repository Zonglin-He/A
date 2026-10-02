"""Reproducible measured-result figure, with source bootstrap intervals."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(folder):
    p=Path(folder);read=lambda f:json.loads(f.read_text())
    datasets=['hc2','vidstg'];names=['HC-STVG-v2','VidSTG'];online=[read(p/d/'ONLINE_SUMMARY.json') for d in datasets];tokens=[read(p/d/'TOKEN_SUMMARY.json') for d in datasets]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,'axes.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12.3,3.0),gridspec_kw={'width_ratios':[1,1.06,1]})
    colors=['#397A98','#D47C5B']
    def errors(ax,y,m,color,label=None):
        center=m['mean']*100;lo,hi=np.array(m['ci95'])*100
        ax.errorbar(center,y,xerr=np.array([[center-lo],[hi-center]]),fmt='o',markersize=6,color=color,capsize=3,lw=1.7,label=label)
    ax=axes[0]
    for i,d in enumerate(datasets):errors(ax,1-i,online[i]['corruption']['future_nonexpert']['metrics']['delta_R_A_m_vIoU'],colors[i])
    ax.axvline(0,color='#9BA3AA',lw=1,ls='--');ax.set_yticks([1,0],names);ax.set_ylim(-.7,1.7);ax.set_xlim(-2.85,.92);ax.set_xlabel('Future nonexpert R − A  (vIoU pp)');ax.set_title('(a) Online transfer',loc='left',weight='bold')
    ax=axes[1]
    for i,d in enumerate(datasets):
        for arm,dy,col in [('S',.15,'#397A98'),('T',-.15,'#D47C5B')]:errors(ax,1-i+dy,tokens[i]['corruption']['metrics'][arm+'_pairwise'],col,'Routed Sa2VA (S)' if arm=='S' and i==0 else 'Native binding (T)' if i==0 else None)
    ax.axvline(50,color='#9BA3AA',lw=1,ls='--');ax.set_yticks([1,0],names);ax.set_ylim(-.8,1.8);ax.set_xlim(20,100);ax.set_xlabel('Fixed-candidate pairwise accuracy (%)');ax.set_title('(b) Token qualification',loc='left',weight='bold');ax.legend(loc='lower left',frameon=False,fontsize=8.5,handlelength=1.6,bbox_to_anchor=(0,-.06))
    ax=axes[2]
    for i,d in enumerate(datasets):errors(ax,1-i,tokens[i]['corruption']['metrics']['delta_T_S_v'],colors[i])
    ax.axvline(0,color='#9BA3AA',lw=1,ls='--');ax.set_yticks([1,0],names);ax.set_ylim(-.7,1.7);ax.set_xlim(-1.65,.3);ax.set_xlabel('Fixed-candidate T − S  (vIoU pp)');ax.set_title('(c) Token selection utility',loc='left',weight='bold')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.spines[['left','bottom']].set_color('#B4BBC1');ax.grid(axis='x',color='#E6EAEE',lw=.65);ax.set_axisbelow(True);ax.tick_params(length=3)
    fig.subplots_adjust(left=.08,right=.985,bottom=.25,top=.86,wspace=.55)
    for suffix in ['png','pdf','svg']:fig.savefig(p/f'ROUTED_ONLINE_TOKEN.{suffix}',dpi=220,facecolor='white')
    plt.close(fig)
    print('Saved PNG/PDF/SVG from measured summaries; no generated data')
if __name__=='__main__':run(Path(sys.argv[1]))

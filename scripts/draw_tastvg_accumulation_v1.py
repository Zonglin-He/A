"""Publication exports from anonymous scalar records only."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(directory):
    directory = Path(directory)
    summary = json.loads((directory/'SUMMARY.json').read_text())
    geometry = json.loads((directory/'WRITE_GEOMETRY.json').read_text())
    colors = {'A':'#3675AD', 'R':'#D17A35'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(12.4,3.35),gridspec_kw={'width_ratios':[1.15,1.2,1.]})
    corruption = summary['corruption']
    for i, arm in enumerate(['A','R']):
        for offset, field, marker, face in [(-.10,'delta_all_source_v','o',colors[arm]),
                                           (.10,'delta_last_source_v','s','white')]:
            value=corruption[arm]['metrics'][field];mean=100*value['mean'];lo,hi=100*np.asarray(value['ci95'])
            axes[0].errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt=marker,
                             color=colors[arm],markerfacecolor=face,markersize=7,
                             capsize=3,linewidth=1.5)
        for offset,field,marker,face in [(-.10,'delta_last_all_v','o',colors[arm]),
                                        (.10,'delta_last_all_fixed_v','s','white')]:
            value=corruption[arm]['metrics'][field];mean=100*value['mean'];lo,hi=100*np.asarray(value['ci95'])
            axes[1].errorbar(i+offset,mean,yerr=[[mean-lo],[hi-mean]],fmt=marker,
                             color=colors[arm],markerfacecolor=face,markersize=7,
                             capsize=3,linewidth=1.5)
        values=np.array([row['prefix_cancellation'] for row in geometry if row['arm']==arm and row['condition']!='clean'],dtype=float)
        axes[2].plot(np.arange(1,9),np.nanmean(values,axis=0),marker='o',markersize=4,
                      color=colors[arm],label='Uniform A' if arm=='A' else 'Routed R',linewidth=1.6)
        axes[2].fill_between(np.arange(1,9),np.nanmin(values,axis=0),np.nanmax(values,axis=0),
                             color=colors[arm],alpha=.10,linewidth=0)
    for ax in axes[:2]:
        ax.axhline(0,color='#8B8B8B',linewidth=.8,linestyle='--')
        ax.set_xticks([0,1],['Uniform A','Routed R']);ax.set_xlim(-.55,1.55)
        ax.grid(axis='y',color='#E8E8E8',linewidth=.5)
    axes[0].set_title('(a) Transfer relative to Source',loc='left',fontweight='bold',fontsize=10)
    axes[0].set_ylabel('Dense vIoU gain (pp)')
    axes[0].plot([],[],marker='o',color='#606060',linestyle='',label='All historical writes')
    axes[0].plot([],[],marker='s',color='#606060',markerfacecolor='white',linestyle='',label='Latest historical write')
    axes[0].legend(loc='best',frameon=False,fontsize=8)
    axes[1].set_title('(b) Latest minus All',loc='left',fontweight='bold',fontsize=10)
    axes[1].set_ylabel('Paired dense vIoU difference (pp)')
    axes[1].plot([],[],marker='o',color='#606060',linestyle='',label='Native temporal interval')
    axes[1].plot([],[],marker='s',color='#606060',markerfacecolor='white',linestyle='',label='Source temporal interval')
    axes[1].legend(loc='best',frameon=False,fontsize=8)
    axes[2].set_title('(c) Saved-write cancellation',loc='left',fontweight='bold',fontsize=10)
    axes[2].set_xlabel('Number of historical expert arrivals')
    axes[2].set_ylabel(r'$\|\sum_i\delta_i\|\,/\,\sum_i\|\delta_i\|$')
    axes[2].set_xticks([1,2,4,6,8]);axes[2].set_ylim(0,1.05)
    axes[2].grid(axis='y',color='#E8E8E8',linewidth=.5)
    axes[2].legend(frameon=False,fontsize=8)
    fig.tight_layout(w_pad=1.8)
    for extension in ['png','pdf','svg']:
        fig.savefig(directory/('accumulation_audit.'+extension),dpi=250,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    draw(sys.argv[1])

"""Publication-resolution plots of the fixed diagnostic; no score selection."""
import os
os.environ['MPLBACKEND']='Agg'
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_structured_separability/2026-10-03'
def read(p):return json.loads(Path(p).read_text())
def main():
    assert read(OUT/'ROOT_AUDIT.json')['status']=='pass'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.titlesize':11,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    source={ds:read(OUT/ds/'SUMMARY.json') for ds in ['vidstg','hc2']};data={};files=[]
    fig,axes=plt.subplots(2,2,figsize=(10.2,5.9),sharex=True,sharey=True)
    labels=[r'$-\widehat P_A$',r'$-\widehat R_A$',r'$\Delta\widehat P$',r'$\Delta\widehat R$']
    fields=['P_A','R_A','delta_P','delta_R']
    colors={'Inside_Endpoint':'#214e68','Geometry':'#929da7','Shuffle_Inside_Endpoint':'#ca8a5b'}
    names={'Inside_Endpoint':'Frozen latent P/R','Geometry':'Geometry control','Shuffle_Inside_Endpoint':'Shuffled-label control'}
    for i,ds in enumerate(['vidstg','hc2']):
        for j,panel in enumerate(['search_corrupt','confirm_corrupt']):
            ax=axes[i,j];summ=source[ds][panel];data[ds+'/'+panel]={}
            for m,(recipe,color) in enumerate(colors.items()):
                rr=summ['recipes'][recipe]['discrimination']['t'];values=[]
                for k,f in enumerate(fields):
                    z=rr[f]['auc'];x=z['mean'];lo,hi=z['ci95'];y=3-k+(1-m)*.16
                    ax.errorbar(x,y,xerr=np.array([[max(0,x-lo)],[max(0,hi-x)]]),fmt=['o','s','^'][m],
                        color=color,markersize=5 if m==0 else 4,elinewidth=1.3 if m==0 else .9,capsize=2,
                        label=names[recipe] if k==0 else None)
                    values.append(dict(field=f,mean=x,ci95=[lo,hi]))
                data[ds+'/'+panel][recipe]=values
            ax.axvline(.5,color='#aeb6bd',lw=1,ls=(0,(3,3)),zorder=0)
            ax.set_xlim(-.03,1.03);ax.set_ylim(-.48,3.48);ax.set_yticks(range(4),labels[::-1]);ax.grid(axis='x',color='#e5e9ec',lw=.6)
            c=summ['coverage'];dataset='VidSTG' if ds=='vidstg' else 'HC-STVG-v2'
            ax.set_title(f'{dataset} · '+('Development' if j==0 else 'Confirmation')+f' ({c["sources"]} sources)',loc='left',pad=10)
            if i==1:ax.set_xlabel('Source-balanced AUROC for helpful vs. harmful correction')
    handles,leg=axes[0,0].get_legend_handles_labels();fig.legend(handles,leg,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,1.01))
    fig.subplots_adjust(top=.9,bottom=.1,left=.1,right=.98,hspace=.37,wspace=.2)
    for ext in ['png','pdf']:
        f=folder/('structured_discrimination.'+ext);fig.savefig(f,dpi=220,bbox_inches='tight');files.append(str(f.relative_to(OUT)))
    plt.close(fig)
    rows=read(OUT/'ROWS.json');fig,axes=plt.subplots(1,2,figsize=(9.4,3.7));planes={}
    for ax,ds in zip(axes,['vidstg','hc2']):
        rr=[r for r in rows if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['eligible']]
        planes[ds]=[dict(cell_key=r['cell_key'],source_id=r['source_id'],delta_P=r['readouts']['Inside_Endpoint']['delta_P'],
            delta_R=r['readouts']['Inside_Endpoint']['delta_R'],label=r['label_t'],delta_t=r['delta_t']) for r in rr]
        for label,color,marker in [('helpful','#258579','o'),('harmful','#c55d39','X')]:
            pts=[r for r in planes[ds] if r['label']==label]
            ax.scatter([r['delta_P'] for r in pts],[r['delta_R'] for r in pts],c=color,marker=marker,s=43,
                linewidths=.4,edgecolors='white',alpha=.85,label=f'{label.capitalize()} ({len(pts)})')
        ax.axhline(0,color='#c1c8cd',lw=.9);ax.axvline(0,color='#c1c8cd',lw=.9)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel(r'$\Delta\widehat P$ (Inside probe)');ax.set_ylabel(r'$\Delta\widehat R$ (Endpoint probe)')
        ax.set_title('VidSTG confirmation' if ds=='vidstg' else 'HC-STVG-v2 confirmation',loc='left',pad=10)
        ax.legend(frameon=False,fontsize=9,loc='best');ax.margins(.17)
    fig.tight_layout(w_pad=3)
    for ext in ['png','pdf']:
        f=folder/('confirmation_tradeoffs.'+ext);fig.savefig(f,dpi=220,bbox_inches='tight');files.append(str(f.relative_to(OUT)))
    plt.close(fig)
    (OUT/'FIGURE_DATA.json').write_text(json.dumps(dict(AUC=data,planes=planes),indent=2)+'\n')
    (OUT/'FIGURES.json').write_text(json.dumps(dict(files={p:hashlib.sha256((OUT/p).read_bytes()).hexdigest() for p in files},
        notes='Diagnostic discrimination; historical exposed panels; no online selection/gain; CIs are source-cluster exploratory intervals'),indent=2)+'\n')
if __name__=='__main__':main()

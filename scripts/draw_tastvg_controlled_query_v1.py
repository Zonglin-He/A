"""Source-backed controlled-query figures; no inference or selection."""
import os
os.environ['MPLBACKEND']='Agg'
import sys,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def style(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
def draw(ax,x,values,color,label):
    means=np.array([np.nan if v['mean'] is None else v['mean'] for v in values])
    bounds=np.array([[np.nan,np.nan] if v['ci95'] is None else v['ci95'] for v in values])
    ax.vlines(x,bounds[:,0],bounds[:,1],color=color,lw=1.3)
    ax.hlines(bounds[:,0],x-.025,x+.025,color=color,lw=1.2);ax.hlines(bounds[:,1],x-.025,x+.025,color=color,lw=1.2)
    ax.plot(x,means,'o',color=color,ms=4.5,label=label)
def export(fig,path):
    fig.savefig(path.with_suffix('.png'),dpi=230,bbox_inches='tight',facecolor='white')
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def run(folder):
    folder=Path(folder);out=folder/'figures';out.mkdir(exist_ok=True)
    summaries={d:read(folder/d/'SUMMARY.json') for d in ['vidstg','hc2']};data={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    tasks=['precision','recall','tiou'];x=np.arange(3)
    fig,axes=plt.subplots(2,2,figsize=(10.1,6.9),layout='constrained')
    for j,ds in enumerate(['vidstg','hc2']):
        data[ds]={};title='VidSTG' if ds=='vidstg' else 'HC-STVG-v2'
        for k,view in enumerate(['Full','Contrast']):
            ax=axes[j,k]
            for cohort,offset,color,label in [('common_strict',-.065,'#18778b','Text-signature tiers 0–2'),('common',.065,'#9a866d','All paired tiers')]:
                s=summaries[ds]['corrupt/all/'+cohort]
                vals=[s['specificity_subject_minus_event'][f'candidate/{view}/{t}/real']['r2'] for t in tasks]
                draw(ax,x+offset,vals,color,f'{label} (n={s["coverage"]["sources"]})')
                data[ds][view+'/'+cohort]=vals
            ax.axhline(0,color='#aab0b5',lw=1,ls='--');style(ax)
            ax.set_title(title+' · '+view,pad=42);ax.set_xticks(x,['Precision','Recall','tIoU'])
            ax.set_ylabel('R²(subject swap) − R²(event swap)')
            ax.legend(frameon=False,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.015))
    export(fig,out/'action_subject_specificity')
    fig,axes=plt.subplots(2,2,figsize=(10.2,6.3),layout='constrained')
    arms=[('true','Original','#18778b'),('event','Event swap','#bf773c'),('subject','Subject swap','#72568c'),('generic','Generic swap','#9ba0a5')]
    for j,ds in enumerate(['vidstg','hc2']):
        s=summaries[ds]['corrupt/all/common_strict']
        for k,view in enumerate(['Full','Contrast']):
            ax=axes[j,k]
            for i,(arm,label,color) in enumerate(arms):
                vals=[s['metrics'][f'{arm}/candidate/{view}/{t}/real']['metrics']['r2'] for t in tasks]
                draw(ax,x+(i-1.5)*.11,vals,color,label);data[ds][view+'/'+arm+'/absolute']=vals
            ax.axhline(0,color='#aab0b5',lw=1,ls='--');style(ax);ax.set_xticks(x,['Precision','Recall','tIoU'])
            title='VidSTG' if ds=='vidstg' else 'HC-STVG-v2'
            ax.set_title(f'{title} · {view} · {s["coverage"]["sources"]} paired sources',pad=8)
            ax.set_ylabel('Source-balanced pooled R²')
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2);export(fig,out/'strict_readout_quality')
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8),layout='constrained');coverage={}
    cohorts=['all','event_available','subject_available','common','common_strict','same_video_event','same_video_subject']
    labels=['Original panel','Event donor available','Subject donor available','Both donors','Both text tiers ≤2','Same-video event','Same-video subject']
    for ax,ds in zip(axes,['vidstg','hc2']):
        cc=[summaries[ds]['corrupt/all/'+c]['coverage'] for c in cohorts];coverage[ds]=dict(zip(cohorts,cc))
        values=[c['sources'] for c in cc];y=np.arange(len(values));ax.barh(y,values,color=['#aeb8bc']*4+['#18778b','#d1ab79','#d1ab79'],height=.61)
        for i,v in enumerate(values):ax.text(v+.25,i,str(v),va='center',fontsize=9)
        ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,max(values)+3)
        ax.set_xlabel('Independent recipient sources');ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2',pad=9)
        ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
    export(fig,out/'natural_pair_coverage')
    (folder/'FIGURE_DATA.json').write_text(json.dumps(dict(readouts=data,coverage=coverage,
        uncertainty='10,000 paired recipient-source bootstrap, fixed donor mapping; pointwise unadjusted',
        caution='Text tiers do not certify complete event equivalence. Confirmation strict cohort has one source per dataset.',
        units='R², not vIoU gains'),indent=2,allow_nan=False)+'\n')
    print('CONTROLLED_QUERY_FIGURES 6 PNG/PDF exports')
if __name__=='__main__':run(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'results/tastvg_controlled_query/2026-10-03')

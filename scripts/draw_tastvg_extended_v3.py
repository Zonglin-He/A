"""Full fixed-coordinate sensitivity, paired intervals and measured work."""
import sys,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(folder):
    b=Path(folder);read=lambda p:json.loads(p.read_text());trials=read(b/'TRIALS.json');design=read(b/'SEARCH_DESIGN.json');anchor=design['anchor'];paired=read(b/'PAIRED_VS_ANCHOR.json')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,4,figsize=(15.2,7))
    coords=[('rho','Probe radius'),('steps','Update steps'),('student_temperature','Student temperature'),('direction_count','Directions')]
    for col,(key,label) in enumerate(coords):
        tt=[t for t in trials if t['stage']=='screen' and (t['tag']=='screen_anchor' or t['tag'].startswith('screen_'+key+'_'))];tt=sorted(tt,key=lambda t:t['params'][key]);valid=[t for t in tt if t['state']=='COMPLETE']
        for sub,color,marker,name in [('all','#315b8a','o','All arrivals'),('nonexpert','#b77932','s','Nonexpert arrivals')]:
            m=[paired[t['tag']]['corruption'][sub]['metrics']['delta_vs_anchor'] for t in valid];x=[t['params'][key] for t in valid];y=[100*v['mean'] for v in m]
            axes[0,col].plot(x,y,color=color,marker=marker,ms=4,label=name)
            if sub=='all':axes[0,col].fill_between(x,[100*v['ci95'][0] for v in m],[100*v['ci95'][1] for v in m],color=color,alpha=.13,label='Paired 95% CI')
        work=[read(b/t['result_dir']/'AUDIT.json') for t in valid]
        axes[1,col].plot(x,[w['compute']['native_replays'] for w in work],color='#487c62',marker='o',ms=4,label='Native replays')
        for ax in axes[:,col]:
            if key in ['rho','student_temperature']:ax.set_xscale('log')
            ax.axvline(anchor[key],color='#777',ls=':',lw=1);ax.set_xlabel(label);ax.grid(axis='y',alpha=.2)
        axes[0,col].axhline(0,color='#888',lw=.8);axes[0,col].set_title('('+chr(97+col)+') '+label,loc='left');axes[0,col].set_ylabel('Corruption ΔvIoU vs. anchor (pp)')
        axes[1,col].set_ylabel('Replays per 384-arrival trial')
        failed=[t for t in tt if t['state']=='FAIL']
        if failed:axes[0,col].text(.02,.03,'Numerical failures: '+', '.join(str(t['params'][key]) for t in failed),transform=axes[0,col].transAxes,fontsize=8)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ['svg','pdf','png']:fig.savefig(b/f'sensitivity.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    conf=read(b/'CONFIRMATION_SUMMARY.json');fig,ax=plt.subplots(figsize=(5.8,4.2))
    for j,name in enumerate(['anchor','selected']):
        for off,sub,color in [(-.07,'all','#315b8a'),(.07,'nonexpert','#b77932')]:
            if 'summary' not in conf[name]:continue
            m=conf[name]['summary']['corruption'][sub]['metrics']['delta_m_vIoU'];y=100*m['mean'];lo,hi=[100*v for v in m['ci95']];ax.errorbar(j+off,y,yerr=[[y-lo],[hi-y]],fmt='o',color=color,capsize=4,label=sub if j==0 else None)
    ax.axhline(0,color='#888',lw=.8);ax.set_xticks([0,1],['V2 anchor','V3 selected']);ax.set_xlim(-.5,1.5);ax.set_ylabel('Corruption ΔvIoU vs. Frozen (pp)');ax.set_title('Separate 16-source confirmation');ax.legend(frameon=False);fig.tight_layout()
    for ext in ['svg','pdf','png']:fig.savefig(b/f'confirmation.{ext}',dpi=170,facecolor='white')
    plt.close(fig)
    # Focused coordinates are separate from the original one-factor screen.
    for stage in sorted({t['stage'] for t in trials if t['stage']!='screen'}):
        grid=read(b/f'{stage}_GRID.json');key=grid['parameter'];tt=sorted([t for t in trials if t['stage']==stage and t['state']=='COMPLETE'],key=lambda t:t['params'][key])
        fig,axs=plt.subplots(1,2,figsize=(9.8,4.1));xs=[t['params'][key] for t in tt]
        for sub,color,label in [('all','#315b8a','All arrivals'),('nonexpert','#b77932','Nonexpert arrivals')]:
            mm=[paired[t['tag']]['corruption'][sub]['metrics']['delta_vs_anchor'] for t in tt];axs[0].plot(xs,[100*m['mean'] for m in mm],marker='o',color=color,label=label)
            if sub=='all':axs[0].fill_between(xs,[100*m['ci95'][0] for m in mm],[100*m['ci95'][1] for m in mm],alpha=.12,color=color)
        selected=read(b/f'{stage}_SELECTION.json')['params'][key]
        axs[0].axvline(selected,ls=':',color='#555',label='Sealed selection');axs[0].axhline(0,color='#888',lw=.8);axs[0].set_ylabel('Corruption ΔvIoU vs. anchor (pp)');axs[0].legend(frameon=False,fontsize=8)
        work=[read(b/t['result_dir']/'AUDIT.json') for t in tt];axs[1].plot(xs,[a['compute']['native_replays'] for a in work],marker='o',color='#487c62');axs[1].set_ylabel('Replays per 384-arrival trial')
        for ax in axs:
            ax.set_xlabel(key);ax.grid(axis='y',alpha=.2)
            if key in ['rho','student_temperature']:ax.set_xscale('log')
        axs[0].set_title('Focused coordinate: '+key,loc='left');axs[1].set_title('Exact downstream work',loc='left');fig.tight_layout()
        for ext in ['svg','pdf','png']:fig.savefig(b/f'{stage}.{ext}',dpi=170,facecolor='white')
        plt.close(fig)
    costs=[]
    for t in trials:
        if t['state']=='COMPLETE':
            a=read(b/t['result_dir']/'AUDIT.json');costs.append(dict(tag=t['tag'],stage=t['stage'],reused_from=t.get('reused_from',''),**t['params'],**a['compute'],worker_wall_seconds=a['worker_wall_seconds']))
    with (b/'COSTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(costs[0]),lineterminator='\n');w.writeheader();w.writerows(costs)

if __name__=='__main__':run(sys.argv[1])

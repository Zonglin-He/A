"""Display-only correction: put scalar labels above bootstrap CI whiskers."""
import json,time,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent

def run():
    summary=json.loads((BASE/'SUMMARY.json').read_text())
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    colors={'Source Only':'#888888','DINO-Refine':'#4c78a8','TENT-STVG':'#f58518','SAR-STVG':'#54a24b','EATA-STVG':'#e45756','Target-trained reference':'#b279a2'}
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for ds,ax in zip(['vidstg','hc2'],axes):
        arms=summary['datasets'][ds];names=list(arms);x=np.arange(len(names));v=np.array([arms[n]['metrics']['After_v']['source_macro']*100 for n in names]);ci=np.array([arms[n]['metrics']['After_v']['ci95_source'] for n in names])*100
        ax.bar(x,v,color=[colors[n] for n in names]);ax.errorbar(x,v,yerr=np.maximum(0,np.vstack([v-ci[:,0],ci[:,1]-v])),fmt='none',color='black',capsize=3)
        ax.set_xticks(x,names,rotation=28,ha='right');ax.set_ylabel('Parent-source macro vIoU (%)');ax.set_title('HC source → VidSTG test' if ds=='vidstg' else 'Vid source → HC2 validation')
        for i,y in enumerate(v):ax.text(i,ci[i,1]+.3,f'{y:.2f}',ha='center',fontsize=9)
        ax.set_ylim(0,float(ci[:,1].max())+1.5)
    fig.suptitle('Sealed baseline scores · 95% parent-source bootstrap CI · OPD full-roster scores unavailable')
    fig.savefig(BASE/'baseline_scores.png');fig.savefig(BASE/'baseline_scores.pdf');plt.close(fig)

if __name__=='__main__':run()

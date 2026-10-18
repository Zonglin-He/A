"""Static scientific figures from sealed aggregate statistics only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_evidence_strata_v1 import P,read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
x=read(P/'FULL_STRATA.json');t=read(P/'TEMPORAL64.json');fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,ds in enumerate(['hc2','vidstg']):
 s=x['groups'][ds]['excluding_tuning_parents']['strata']['speed_sizes_sec'];labels=list(s);p=np.arange(len(labels))
 for i,(mode,label)in enumerate([('query_macro','Query macro'),('parent_macro','Parent macro')]):axs[0,col].bar(p+(i-.5)*.35,[100*s[k][mode]['delta_current_v']for k in labels],width=.35,label=label)
 axs[0,col].axhline(0,color='black',lw=.7);axs[0,col].set(xticks=p,xticklabels=[k+'\nn='+str(s[k]['queries'])for k in labels],ylabel='Current-query vIoU delta (pp)',title=ds+' full clean, tuning parents excluded',xlabel='Common target-size-normalized speed quartiles');axs[0,col].legend()
 ss=t['groups'][ds]['After']['fused']['strata']['event_length_bin'];labels=list(ss);p=np.arange(len(labels))
 for a,l in [('projection','Projection'),('native_MBR','Native MBR')]:axs[1,col].plot(p,[100*ss[k][a]['query_macro']['delta_v_vs_Native']for k in labels],marker='o',label=l)
 axs[1,col].axhline(0,color='black',lw=.7);axs[1,col].set(xticks=p,xticklabels=[k+'\nn='+str(ss[k]['Native']['queries'])for k in labels],ylabel='vIoU delta vs Native (pp)',title=ds+' exact64 subset, fixed online After boxes',xlabel='Common event/input-duration bins');axs[1,col].legend()
fig.suptitle('Frozen-cache diagnosis: full online effects and fixed temporal readouts')
fig.savefig(P/'STRATA.svg');fig.savefig(P/'STRATA.png',dpi=140)

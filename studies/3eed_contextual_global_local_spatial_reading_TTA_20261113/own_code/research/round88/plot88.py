from core88 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=read(H88/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']['balanced_mIoU_percent'];rr=read(H88/'COMPONENT_EFFECT_ROWS.json');fig,axs=plt.subplots(1,2,figsize=(13,5));arms=['frozen_bi','full_direct','full_frozen3','full_tta1','full_tta3','R59'];labels=['Frozen bidirectional','New reading / no update','Same 3 reads / no update','New reading + 1 step','New reading + 3 steps','R59'];ax=axs[0];ax.barh(labels,[s[a] for a in arms],color=['#777777','#4c78a8','#72b7b2','#f58518','#e45756','#b279a2']);ax.invert_yaxis();ax.set_xlabel('DEV336 balanced mIoU (%)')
for i,a in enumerate(arms):ax.text(s[a]+.02,i,f'{s[a]:.3f}',va='center',fontsize=8)
ax.set_xlim(min(s[a] for a in arms)-.5,max(s[a] for a in arms)+.65);ax.grid(axis='x',alpha=.2);ax=axs[1];keys=['no_relation','no_local','physical_only1','no_rank1','no_correspondence1'];lab=['Relation-guided reading','Spatial local reading','Added supervision vs physical','Partial spatial preference','Region correspondence'];vals=[next(r['component_net_pp'] for r in rr if r['component_removed']==k and r['readout']=='tta1') for k in keys];ax.barh(lab,vals,color=['#f58518' if v>=0 else '#4c78a8' for v in vals]);ax.invert_yaxis();ax.axvline(0,c='black',lw=.8);ax.set_xlabel('Full minus ablated mIoU (percentage points)');ax.grid(axis='x',alpha=.2)
for i,v in enumerate(vals):ax.text(v/2 if v<0 else -.002,i,f'{v:+.3f}',ha='center' if v<0 else 'right',va='center',fontsize=8,color='white' if v<0 else 'black')
fig.suptitle('R88 / structure, same-information readout and parameter update');fig.tight_layout();fig.savefig(H88/'RESULTS.png',dpi=150);plt.close(fig)

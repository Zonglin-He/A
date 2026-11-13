from target_core89 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=read(H89/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336']['balanced_mIoU_percent'];rr=read(H89/'COMPONENT_EFFECT_ROWS.json');fig,axs=plt.subplots(1,2,figsize=(14,6));arms=['frozen_bi','continue_frozen','query_frozen','candidate_frozen','candidate_mean3','candidate_tta1','candidate_tta3','R59'];labels=['Frozen R76','Matching continuation','Query-only prepared head','Candidate-conditioned frozen','Same 3 reads / no update','Candidate-conditioned +1 step','Candidate-conditioned +3 steps','R59'];ax=axs[0];ax.barh(labels,[s[a] for a in arms],color=['#777777','#999999','#72b7b2','#4c78a8','#54a24b','#f58518','#e45756','#b279a2']);ax.invert_yaxis();ax.set_xlabel('DEV336 balanced mIoU (%)')
for i,a in enumerate(arms):ax.text(s[a]+.01,i,f'{s[a]:.3f}',va='center',fontsize=8)
ax.set_xlim(min(s[a] for a in arms)-.4,max(s[a] for a in arms)+.7);ax.grid(axis='x',alpha=.2);ax=axs[1];names=['no_ref','shared_mix','wrong_space','role_swap','local_correspondence','correspondence_plus_rank'];vals=[next(r['net_pp'] for r in rr if r['component']==k and r['platform']=='balanced' and r['arm']=='candidate_tta1') for k in names];ax.barh(names,vals,color=['#f58518' if v>=0 else '#4c78a8' for v in vals]);ax.invert_yaxis();ax.axvline(0,c='black',lw=.8);ax.set_xlabel('Full minus control mIoU (percentage points)');ax.grid(axis='x',alpha=.2)
for i,v in enumerate(vals):ax.text(v,i,f' {v:+.3f}',va='center',fontsize=8)
fig.suptitle('R89 / source preparation, candidate-conditioned evidence and TTA');fig.tight_layout();fig.savefig(H89/'RESULTS.png',dpi=150);plt.close(fig)

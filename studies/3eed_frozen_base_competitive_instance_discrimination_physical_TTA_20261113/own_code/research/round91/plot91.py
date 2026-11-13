from targets91 import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
su=read(H91/'score_dev336_attempt001/TASK_SUMMARY.json')['ALL336'];s=su['balanced_mIoU_percent'];arms=['R76_frozen','R59','R80_raw','R82_bt','R76_full_ce_frozen','R76_query_disc_frozen','R76_full_disc_frozen','R76_full_disc_feature3','R76_full_disc_physical_raw','R76_full_disc_physical_bt','R76_full_disc_conditional_bt'];labels=['Frozen R76','R59','Original R80 raw','Original R82 backtracking','Spatial CE / frozen','Query competition / frozen','Spatial competition / frozen','Same-view feature mean','Spatial + physical raw','Spatial + physical backtracking','Spatial + conditional backtracking'];fig,ax=plt.subplots(figsize=(11,7));v=[s[a] for a in arms];ax.barh(labels,v,color=['#888888']*4+['#72b7b2','#999999','#4c78a8','#54a24b','#f58518','#e45756','#b279a2']);ax.invert_yaxis();ax.set_xlim(min(v)-.35,max(v)+.5);ax.set_xlabel('DEV336 equal-platform mIoU (%)');ax.grid(axis='x',alpha=.2)
for i,x in enumerate(v):ax.text(x+.01,i,f'{x:.3f}',va='center',fontsize=9)
ax.set_title('R91 / frozen-base competitive instance discrimination and physical TTA');fig.tight_layout();fig.savefig(H91/'RESULTS.png',dpi=160);plt.close(fig)

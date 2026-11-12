"""Aggregate-only scientific ablation chart, fixed denominators, no invented CI."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;rows=json.loads((H/'COMPONENT_EFFECT_ROWS.json').read_text());modes=list(dict.fromkeys(r['component_removed'] for r in rows if r['component_removed']!='full'));names={'hard_alias':'Soft vs hard alias','no_alias':'Soft alias vs none','independent':'Joint vs fixed anchors','no_subject':'Subject phrase prior','no_relation':'Spatial relation evidence','no_RGB':'RGB attribute evidence','no_null':'Unobserved-reference NULL','no_exclusivity':'Soft role exclusivity'};y=np.arange(len(modes));fig,ax=plt.subplots(figsize=(11,6));
for offset,readout,col in [(-.18,'direct','#2575bb'),(.18,'tta1','#d88116')]:
 vals=[next(r['component_net_pp'] for r in rows if r['component_removed']==m and r['readout']==readout) for m in modes];ax.barh(y+offset,vals,height=.32,label='Direct readout' if readout=='direct' else 'One-step TTA',color=col)
 for yy,v in zip(y+offset,vals):ax.annotate(f'{v:+.3f}',(v,yy),xytext=(4 if v>=0 else -4,0),textcoords='offset points',ha='left' if v>=0 else 'right',va='center',fontsize=8)
ax.axvline(0,color='black',lw=.8);ax.set_yticks(y,[names[m] for m in modes]);ax.invert_yaxis();ax.set_xlabel('Full minus ablated balanced mIoU (percentage points)');ax.set_title('R87 / DEV336 component effects\nPositive: retaining the component helps; negative: it hurts');ax.legend();ax.grid(axis='x',alpha=.2);ax.margins(x=.2);fig.tight_layout();fig.savefig(H/'COMPONENT_EFFECTS.png',dpi=160);plt.close(fig)

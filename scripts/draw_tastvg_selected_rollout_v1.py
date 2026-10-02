"""Quantitative figures from audited anonymous A/G evidence only."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(Path(p).read_text())
def save(fig,base,name):
 for ext in ['png','pdf','svg']:fig.savefig(base/f'{name}.{ext}',dpi=220)
 plt.close(fig)
def run(base):
 base=Path(base);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 colors=['#606A7A','#277FA4'];names=['A: Rank-RKL','G: Selected rollout'];fig,axes=plt.subplots(1,2,figsize=(8.4,2.6),constrained_layout=True)
 for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
  for i,arm in enumerate(['A','G']):
   m=read(base/ds/arm/'SUMMARY.json')['corruption']['nonexpert']['metrics']['delta_m_vIoU'];v=100*m['mean'];lo,hi=100*np.array(m['ci95']);ax.errorbar(v,1-i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=colors[i],capsize=3,markersize=6,lw=1.4)
  ax.axvline(0,color='#AAB1B9',lw=.8,ls='--');ax.set_yticks([1,0],names);ax.set_xlabel('Future nonexpert ΔvIoU vs Frozen (pp)');ax.set_title(title,fontweight='bold');ax.set_ylim(-.5,1.5);ax.grid(axis='x',color='#ECEEF1',lw=.6)
 save(fig,base,'future_comparison')
 fig,axes=plt.subplots(1,2,figsize=(8.4,2.7),constrained_layout=True)
 for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
  for i,arm in enumerate(['A','G']):
   r=read(base/ds/arm/'SPATIAL_STEP_ROWS.json');v=np.sort([100*x['delta_update'] for x in r if x['condition']!='clean' and x['rewards'] is not None]);ax.plot(v,(np.arange(len(v))+1)/len(v),color=colors[i],label=names[i],lw=1.5)
  ax.axvline(0,color='#AAB1B9',lw=.8,ls='--');ax.set_title(title,fontweight='bold');ax.set_xlabel('Fixed-time spatial ΔvIoU per step (pp)');ax.set_ylabel('Empirical cumulative fraction');ax.set_ylim(0,1);ax.grid(color='#ECEEF1',lw=.6);ax.legend(frameon=False,fontsize=8,loc='lower right')
 save(fig,base,'local_update_diagnosis')
 fig,axes=plt.subplots(1,2,figsize=(8.4,2.7),constrained_layout=True)
 for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
  for i,arm in enumerate(['A','G']):
   rows=[x for x in read(base/ds/arm/'SPATIAL_STEP_ROWS.json') if x['condition']!='clean' and x['rewards'] is not None]
   v=np.sort([x['output_space_cosine'] for x in rows if x['output_space_cosine'] is not None]);ax.step(v,(np.arange(len(v))+1)/len(v),where='post',color=colors[i],label=f'{names[i]} (defined {len(v)}/{len(rows)})',lw=1.5)
  ax.axvline(0,color='#AAB1B9',lw=.8,ls='--');ax.set_title(title,fontweight='bold');ax.set_xlabel('cos(post − central, selected − central)');ax.set_ylabel('Empirical cumulative fraction');ax.set_xlim(-1.02,1.02);ax.set_ylim(0,1);ax.grid(color='#ECEEF1',lw=.6);ax.legend(frameon=False,fontsize=7.5,loc='lower right')
 save(fig,base,'functional_direction')
 print('Saved nine quantitative figure files')
if __name__=='__main__':run(sys.argv[1])

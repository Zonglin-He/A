"""Publication artifacts from audited anonymous P0 results; no model execution."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def read(p):return json.loads(Path(p).read_text())
def run(base):
 base=Path(base);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 colors=['#606A7A','#277FA4','#C57935'];names=['A: RKL','E: Rank direction','F: Top direction'];fig,axes=plt.subplots(1,2,figsize=(8.4,2.7),constrained_layout=True)
 for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
  for i,arm in enumerate(['A','E','F']):
   m=read(base/ds/arm/'SUMMARY.json')['corruption']['nonexpert']['metrics']['delta_m_vIoU'];v=100*m['mean'];lo,hi=100*np.array(m['ci95']);ax.errorbar(v,2-i,xerr=[[max(0,v-lo)],[max(0,hi-v)]],fmt='o',color=colors[i],capsize=3,markersize=6,lw=1.4)
  ax.axvline(0,color='#AAB1B9',lw=.8,ls='--');ax.set_yticks([2,1,0],names);ax.set_xlabel('Future nonexpert ΔvIoU vs Frozen (pp)');ax.set_title(title,fontweight='bold');ax.set_ylim(-.5,2.5);ax.grid(axis='x',color='#ECEEF1',lw=.6)
 for ext in ['png','pdf','svg']:fig.savefig(base/f'future_comparison.{ext}',dpi=220)
 plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(8.4,2.7),constrained_layout=True)
 for ax,ds,title in zip(axes,['vidstg','hc2'],['VidSTG','HC-STVG-v2']):
  for i,arm in enumerate(['A','E','F']):
   r=read(base/ds/arm/'SPATIAL_STEP_ROWS.json');v=np.sort([100*x['delta_update'] for x in r if x['condition']!='clean' and x['rewards'] is not None]);ax.plot(v,(np.arange(len(v))+1)/len(v),color=colors[i],label=names[i],lw=1.5)
  ax.axvline(0,color='#AAB1B9',lw=.8,ls='--');ax.set_title(title,fontweight='bold');ax.set_xlabel('Fixed-time spatial ΔvIoU per step (pp)');ax.set_ylabel('Empirical cumulative fraction');ax.set_ylim(0,1);ax.grid(color='#ECEEF1',lw=.6);ax.legend(frameon=False,fontsize=8,loc='lower right')
 for ext in ['png','pdf','svg']:fig.savefig(base/f'local_update_diagnosis.{ext}',dpi=220)
 plt.close(fig)
 print('Saved six figure files from audited anonymous P0 measurements')
if __name__=='__main__':run(sys.argv[1])

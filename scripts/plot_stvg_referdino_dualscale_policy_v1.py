"""Standalone SVG summary of all development policies and frozen confirmation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read
from scripts.run_stvg_referdino_dualscale_hc128_v1 import P,D

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 dev=read(P/'RESULTS_PUBLIC.json');con=read(P/'confirmation/RESULTS_PUBLIC.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'and read(P/'confirmation/ROOT_INDEPENDENT_AUDIT.json')['status']=='pass';selected=read(D/'SELECTION_LOCK.json')['selected_policy'];colors=['#52616b','#007f8b','#9b5b00','#8254a8'];names=['Old','N005F025','N005F050','N010F050'];labels=['Old logit Gaussian','Near .05 / far .25','Near .05 / far .50','Near .10 / far .50'];fig,axes=plt.subplots(2,2,figsize=(13.5,8.7),layout='constrained')
 def point(ax,y,s,color):
  ax.hlines(y,*s['CI95'],color=color,lw=2);ax.plot(s['mean'],y,'o',color=color,ms=6);ax.text(s['mean'],y+.12,f"{s['mean']:+.3f}",fontsize=8,color=color,ha='center')
 for col,ep in enumerate(['v','unobserved_actual_s']):
  ax=axes[0,col]
  for y,(name,label,color)in enumerate(zip(names,labels,colors)):
   st=dev['primary']['paired'][name+'_Live_seedmean_minus_E_MBR'][ep];point(ax,y,st,color)
  direct=dev['primary']['paired']['Direct_minus_E_MBR'][ep]['mean'];ax.axvline(direct,color='#333333',ls='--',lw=1,label='Direct mean');ax.set_yticks(range(4),[l+(' [selected]'if n==selected else'')for n,l in zip(names,labels)]);ax.set_ylim(3.5,-.5);ax.set_title('HC128 exposed development\n'+('full vIoU'if col==0 else'actual sIoU outside 8'));ax.legend(loc='best',fontsize=8)
  ax=axes[1,col];pairs=[('E','Selected_Live_seedmean_minus_E_MBR'),('Direct','Selected_Live_seedmean_minus_Direct'),('fresh Fixed','Selected_Live_seedmean_minus_Selected_Fixed_seedmean'),('old Live','Selected_Live_seedmean_minus_Old_Live_seedmean')]
  for y,(label,key)in enumerate(pairs):point(ax,y,con['primary']['paired'][key][ep],'#007f8b')
  ax.set_yticks(range(4),['selected Live - '+s for s,k in pairs]);ax.set_ylim(3.5,-.5);ax.set_title('Frozen HC32 confirmation\n'+('full vIoU'if col==0 else'actual sIoU outside 8'))
 for ax in axes.flat:ax.axvline(0,color='#777777',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_xlabel('Delta (pp); 95% parent-bootstrap CI');ax.spines[['top','right']].set_visible(False)
 fig.suptitle('Policy selection and frozen confirmation | seeds 41 / 42 / 43',fontsize=14);fig.savefig(P/'POLICY_COMPARISON.svg',metadata={'Date':None,'Description':'All declared policies, no seed winner. Metrics average seeds within parent; not a box ensemble. Confirmation roster has historical project exposure.'});fig.savefig(D/'POLICY_COMPARISON_QA.png',dpi=130);plt.close(fig);print('SVG saved',flush=True)
if __name__=='__main__':main()

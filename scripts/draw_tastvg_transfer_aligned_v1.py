"""Standalone measured plots; never image-generated numerical bars."""
import sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.decota_matrix_common_v1 import read
def run(b):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 colors={'A':'#72869a','R':'#188c80','S':'#d49a52','T':'#6d72b6'};fig,ax=plt.subplots(1,3,figsize=(12.4,3.15),gridspec_kw={'width_ratios':[1.35,1,1]},constrained_layout=True)
 s=read(b/'hc2/TRANSFER_SUMMARY.json')['corruption'];roles=['self','next','near','far'];names=['Same query','Next query','Nearest text','Farthest text']
 for arm,offset in [('A',-.13),('R',.13)]:
  m=[s[k][arm]['metrics']['delta_v'] for k in roles];y=np.array([x['mean']*100 for x in m]);ci=np.array([x['ci95'] for x in m])*100;ax[0].errorbar(np.arange(4)+offset,y,yerr=np.maximum(np.array([y-ci[:,0],ci[:,1]-y]),0),fmt='o',ms=5,capsize=2,color=colors[arm],label=arm+' '+('Uniform' if arm=='A' else 'Routed'))
 ax[0].axhline(0,c='#bbbbbb',lw=.8);ax[0].set_xticks(range(4),names,rotation=13);ax[0].set_ylabel('One-write change in vIoU (pp)');ax[0].set_title('(a) HC: saved-write transfer',loc='left',weight='bold');ax[0].legend(frameon=False,fontsize=8)
 for i,ds in enumerate(['hc2','vidstg']):
  x=read(b/ds/'TOKEN_SUMMARY.json')['corruption']['matched_available_S_T']
  for arm,offset in [('S',-.14),('T',.14)]:
   m=x[arm+'_ranking']['metrics'][arm+'_pairwise'];mean=m['mean']*100;ci=np.array(m['ci95'])*100;ax[1].errorbar(i+offset,mean,yerr=[[mean-ci[0]],[ci[1]-mean]],fmt='o',color=colors[arm],ms=5,capsize=2,label=('Sa2VA' if arm=='S' else 'CLIP token') if i==0 else None)
 ax[1].axhline(50,c='#bbbbbb',lw=.8,ls='--');ax[1].set_ylim(25,100);ax[1].set_xticks([0,1],['HC\n10 sources','Vid\n3 strict-pair sources']);ax[1].set_ylabel('Candidate pairwise accuracy (%)');ax[1].set_title('(b) Same available candidates',loc='left',weight='bold');ax[1].legend(frameon=False,fontsize=8)
 for i,ds in enumerate(['hc2','vidstg']):
  m=read(b/ds/'TOKEN_SUMMARY.json')['corruption']['metrics']['T_gain'];y=100*m['mean'];ci=np.array(m['ci95'])*100;ax[2].errorbar(i,y,yerr=[[y-ci[0]],[ci[1]-y]],fmt='o',color=colors['T'],capsize=2,ms=5)
 ax[2].axhline(0,c='#bbbbbb',lw=.8);ax[2].set_xticks([0,1],['HC','Vid']);ax[2].set_ylabel('Token top1 − native vIoU (pp)');ax[2].set_title('(c) Full panel; native fallback',loc='left',weight='bold')
 for a in ax:a.grid(axis='y',color='#eeeeee',lw=.5);a.set_axisbelow(True)
 for ext in ['png','pdf','svg']:fig.savefig(b/('TRANSFER_ALIGNED.'+ext),dpi=220)
 plt.close(fig)
 rows=[r for r in read(b/'hc2/TRANSFER_ROWS.json') if r['condition']!='clean'];donors=sorted({r['donor_source_id'] for r in rows});targets=sorted({r['target_source_id'] for r in rows if r['role']!='self'});matrices=[]
 for arm in ['A','R']:
  unique={(r['condition'],r['order'],r['donor_arrival'],r['target_arrival']):r for r in rows if r['arm']==arm and r['role']!='self'};groups=collections.defaultdict(list)
  for r in unique.values():groups[r['donor_source_id'],r['target_source_id']].append(r['delta_common_v']*100)
  m=np.full((len(donors),len(targets)),np.nan)
  for (d,t),v in groups.items():m[donors.index(d),targets.index(t)]=np.mean(v)
  matrices.append(m)
 vmax=max(np.nanmax(np.abs(m)) for m in matrices);cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#e5e5e5');fig,aa=plt.subplots(1,2,figsize=(12,4.1),constrained_layout=True)
 for a,m,arm in zip(aa,matrices,['A Uniform','R Routed']):
  im=a.imshow(m,cmap=cmap,vmin=-vmax,vmax=vmax,aspect='auto');a.set_title(arm,loc='left',weight='bold');a.set_xticks(range(len(targets)),targets,fontsize=7);a.set_yticks(range(len(donors)),donors,fontsize=7);a.set_xlabel('Selected future target source');a.set_ylabel('Historical donor source')
 fig.colorbar(im,ax=aa,label='Single-write vIoU effect (pp)',shrink=.82);fig.suptitle('Selected pairs only; grey = not evaluated; role aliases counted once',fontsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(b/('SELECTED_TRANSFER_MATRIX.'+ext),dpi=220)
 plt.close(fig)
if __name__=='__main__':run(Path(sys.argv[1]))

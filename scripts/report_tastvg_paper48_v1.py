"""CPU paper tables, severity/budget/stability plots and deterministic case index."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_paper48_common_v1 import BASE,read,write,sha,verify

def run():
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 verify();panels=['P0','P1','P2','P3_b0','P3_b25','P3_b100','P4']
 for name in panels:assert read(BASE/name/'COMPLETION.json')['status']=='completed' and read(BASE/name/'AUDIT.json')['status']=='pass'
 p1=read(BASE/'P1/SUMMARY.json');p2=read(BASE/'P2/SUMMARY.json');families=['frame_drop','frame_freeze','motion_blur','occlusion','exposure'];plt.rcParams.update({'font.size':10})
 fig,axes=plt.subplots(1,5,figsize=(15,3.3),sharey=True)
 for ax,f in zip(axes,families):
  for arm in ['Frozen','Ours']:
   y=[p2[f'{f}_{s}']['all']['metrics'][arm+'_m_vIoU']['mean']*100 for s in [1,5,10]];ax.plot([1,5,10],y,'o-',label=arm)
  ax.set(title=f.replace('_',' '),xlabel='Corrupted frames (%)');ax.grid(alpha=.2)
 axes[0].set_ylabel('Dense m_vIoU (%)');axes[-1].legend();fig.tight_layout();fig.savefig(BASE/'severity.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(5,3.5));ys=[]
 for b in [0,25,100]:ys.append(read(BASE/f'P3_b{b}/SUMMARY.json')['corruption']['all']['metrics']['Ours_m_vIoU']['mean']*100)
 ax.plot([0,25,100],ys,'o-');ax.set(xlabel='Expert availability (%)',ylabel='Dense m_vIoU (%)');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(BASE/'budget.png',dpi=180);plt.close(fig)
 q=read(BASE/'P1/QUARTILES.json');fig,axes=plt.subplots(1,2,figsize=(9,3.5));axes[0].plot(range(1,5),[q[str(k)]['metrics']['delta_m_vIoU']['mean']*100 for k in range(4)],'o-');axes[0].axhline(0,color='gray',lw=.7);axes[0].set(xlabel='Stream quartile',ylabel='Ours − Frozen m_vIoU (pp)');axes[1].plot(range(1,5),[q[str(k)]['metrics']['drift']['mean'] for k in range(4)],'o-');axes[1].set(xlabel='Stream quartile',ylabel='Parameter displacement');fig.tight_layout();fig.savefig(BASE/'online_stability.png',dpi=180);plt.close(fig)
 rows=read(BASE/'P1/ROWS.json');plan=read(BASE/'P4_PLAN.json');selected=set(plan['parents'][:6]);cases=[{k:r[k] for k in ['parent','order','condition','arrival','expert_scheduled','Frozen_m_vIoU','Ours_m_vIoU','delta_m_vIoU']} for r in rows if r['parent'] in selected and r['order']=='order1' and r['condition'] in ['clean','frame_drop_5']];write(BASE/'FIXED_CASE_INDEX.json',dict(selection='first6 from prespecified source hash; no outcome sorting',cases=cases,visualizations='indices only; private media not exported'))
 future=p1['corruption']['nonexpert']['metrics']['delta_m_vIoU'];excess=read(BASE/'P1/CORRUPTION_EXCESS.json')['metrics']['delta_excess'];lines=['# Paper48 mandatory evidence','', 'Frozen TA-STVG method; hash-selected one query per source. Original451k B1 stopped for compute budget and never partially scored. No external baselines, Pairwise, mixed stream or new method was run.','',f"P1 future nonexpert dense ΔvIoU: {future['mean']*100:+.6f} pp; source-bootstrap95% CI [{future['ci95'][0]*100:+.6f}, {future['ci95'][1]*100:+.6f}].",f"Corruption−clean paired gain: {excess['mean']*100:+.6f} pp. Corruption-specific recovery is not asserted solely from a positive corrupt gain.",'','Tables: P1/REPORT.md (main), P0/REPORT.md (five-order development ablation), P4/SUMMARY.json (real uncached cold-start latency with all loading costs). P2 and P3 are prespecified subsets, not independent cohorts. P0 retains historical sampled scoring and is not merged into official dense main metrics.','', 'P5 is optional and needs a separate completion/skipping receipt. Required results remain subject to root review and verified public synchronization.','']
 (BASE/'REPORT.md').write_text('\n'.join(lines));write(BASE/'MANDATORY_COMPLETION.json',dict(status='completed_pending_root_review',phases=panels,completion_hashes={n:sha(BASE/n/'COMPLETION.json') for n in panels},P5='optional_pending_root_decision',publication='pending',time=time.time()))
if __name__=='__main__':run()

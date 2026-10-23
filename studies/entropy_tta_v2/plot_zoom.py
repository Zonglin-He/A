"""Display-only Fig1 zoom. Boundary triangles retain all off-scale cases."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];P=R/'results/entropy_tta_v2';D=R/'outputs/entropy_tta_v2/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
proof={}
for ds in ['hc2','vidstg']:
 f=P/ds/'paired.csv';before=hashlib.sha256(f.read_bytes()).hexdigest();rows=list(csv.DictReader(f.open()));report=json.loads((P/ds/'report.json').read_text())
 x=np.array([float(r['delta_surrogate'])for r in rows]);y=np.array([float(r['delta_vIoU_pp'])for r in rows]);lo=y < -5;hi=y > 5;inside=~(lo|hi)
 fig,ax=plt.subplots(figsize=(7.2,6));xs=float(np.ceil(max(abs(x).max(),.01)*10)/10)
 ax.set_xlim(-xs,xs);ax.set_ylim(-5,5);ax.set_yticks(np.arange(-4,5,1));ax.set_xticks(np.linspace(-xs,xs,5))
 ax.spines['left'].set_position('zero');ax.spines['bottom'].set_position('zero')
 for side in ['left','bottom']:ax.spines[side].set_color('#96999C');ax.spines[side].set_linewidth(1.1)
 ax.tick_params(axis='both',labelsize=8,colors='#777B7E',length=3,pad=4)
 ax.set_xticklabels([f'{v:.1f}' if abs(v)>1e-10 else '0' for v in ax.get_xticks()])
 ax.set_yticklabels([str(int(v)) if v!=0 else '' for v in ax.get_yticks()])
 ax.axhspan(-5,0,xmin=max(0,(0-ax.get_xlim()[0])/(ax.get_xlim()[1]-ax.get_xlim()[0])),xmax=1,facecolor='#F6E7E3',zorder=0)
 ax.plot(1,0,'>',transform=ax.get_yaxis_transform(),color='#96999C',clip_on=False,markersize=7)
 ax.plot(0,1,'^',transform=ax.get_xaxis_transform(),color='#96999C',clip_on=False,markersize=7)
 ax.scatter(x[inside],y[inside],s=23,c='#73777A',alpha=.65,linewidths=0,zorder=3)
 ax.scatter(x[lo],np.full(lo.sum(),-5),s=45,marker='v',c='#73777A',linewidths=0,clip_on=False,zorder=4)
 ax.scatter(x[hi],np.full(hi.sum(),5),s=45,marker='^',c='#73777A',linewidths=0,clip_on=False,zorder=4)
 name='HC-STVG2.0'if ds=='hc2'else'VidSTG'
 ax.set_title('Does Optimizing a TTA Surrogate Improve STVG?',fontsize=12,pad=46)
 ax.text(.5,1.095,f'{name} · 128 queries · Entropy minimization, temporal LN · 5 steps',ha='center',transform=ax.transAxes,fontsize=8.3,color='#555555')
 fig.text(.52,.19,r'Surrogate loss reduction $\mathcal{L}_{\rm before}-\mathcal{L}_{\rm after}$ (nats)',ha='center',fontsize=10)
 ax.text(0,5.35,r'$\Delta$ vIoU (pp) ↑',ha='center',va='bottom',fontsize=10,color='#414549')
 ax.text(.70*xs,4.4,'Aligned update',ha='center',fontsize=10,color='#54856F')
 ax.text(.70*xs,-4.2,'Misleading update',ha='center',fontsize=10,color='#AA5B46')
 for side in ['top','right']:ax.spines[side].set_visible(False)
 k,n=report['grounding_degrades_given_surrogate_improves'],report['surrogate_improves']
 fig.text(.10,.11,f'Grounding degrades among surrogate-improving queries: {k}/{n} = {100*k/n:.1f}%',fontsize=9,color='#925644')
 fig.text(.10,.055,f'Display window: ±5 pp. Boundary triangles: {int(lo.sum())} below −5; {int(hi.sum())} above +5.',fontsize=8,color='#64696D')
 fig.subplots_adjust(left=.08,right=.94,bottom=.26,top=.76)
 for ext in ['svg','pdf','png']:
  dest=(P/ds/'fig1_zoom5.svg')if ext=='svg'else D/f'fig1_{ds}_zoom5.{ext}';fig.savefig(dest,dpi=300,facecolor='white')
  if ext=='svg':dest.write_text('\n'.join(s.rstrip()for s in dest.read_text().splitlines())+'\n')
 plt.close(fig)
 assert before==hashlib.sha256(f.read_bytes()).hexdigest()
 proof[ds]=dict(queries=len(rows),ordinary_points=int(inside.sum()),below_window=int(lo.sum()),above_window=int(hi.sum()),data_sha256_unchanged=before,conditional_fraction_unchanged=k/n,full_range_original_retained=True,x_display_range=[-xs,xs],all_x_values_visible=bool((abs(x)<=xs).all()),zero_axes_centered=True)
(P/'zoom_display.json').write_text(json.dumps(dict(display_range_pp=[-5,5],data_and_results_unchanged=True,boundary_markers='Triangles mark off-scale observations at display boundaries, not actual y values.',datasets=proof),indent=2)+'\n')
print(json.dumps(proof))

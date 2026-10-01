"""Publication figures from sealed, audited scalar summaries; no model or GT I/O."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'tastvg':'TA-STVG','tubedetr':'TubeDETR','ptd':'PTD\nQwen3-VL-4B'}
COLORS={'temporal':'#287DA5','spatial':'#BF7D23'}

def panel(ax,summary,group,raw=False,annotate=True):
 for j,(key,label,marker,offset) in enumerate([('temporal','Temporal','o',-.12),('spatial','Spatial','^',.12)]):
  for i,model in enumerate(NAMES):
   s=summary[model][group];v=s['raw_'+key+'_gain_pp_with_ci'] if raw else s['paired'][key]
   m=v['mean'];lo,hi=v['ci95']
   ax.errorbar(i+offset,m,yerr=np.array([[m-lo],[hi-m]]),fmt=marker,ms=8.2,capsize=4,elinewidth=1.55,color=COLORS[key],markeredgecolor='white',markeredgewidth=.7,label=label if i==0 else None,zorder=3)
   if annotate:ax.annotate(f'{m:.2f}',(i+offset,hi),xytext=(-4 if j==0 else 4,8),textcoords='offset points',ha='right' if j==0 else 'left',fontsize=10,color=COLORS[key])
 ax.set_xticks(range(3),list(NAMES.values()));ax.set_xlim(-.48,2.48)
 ax.set_ylim(bottom=-.8)
 ax.set_ylabel('Oracle gain (pp)' if raw else 'Recoverable native-support error (%)',labelpad=10)
 ax.spines[['top','right']].set_visible(False)
 ax.spines[['bottom','left']].set_color('#A7ADB3');ax.tick_params(axis='both',length=0,pad=9)
 ax.yaxis.grid(True,color='#E7EAED',linewidth=.7);ax.set_axisbelow(True)
 ax.margins(y=.2)

def run(folder,output=None,main_only=False):
 folder=Path(folder);summary=json.loads((folder/'SUMMARY.json').read_text());assert json.loads((folder/'SCALAR_AUDIT.json').read_text())['status']=='pass'
 output=Path(output) if output else folder;output.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'xtick.labelsize':11,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.color':'#253340','axes.labelcolor':'#253340'})
 fig,ax=plt.subplots(figsize=(8.2,3.8));fig.subplots_adjust(left=.105,right=.98,bottom=.18,top=.88)
 panel(ax,summary,'corruption')
 ax.legend(loc='upper left',bbox_to_anchor=(0,1.16),frameon=False,ncol=2,handletextpad=.5,columnspacing=1.8)
 # Explicit scale derived from every upper confidence bound, retaining zero.
 ymax=max(summary[m]['corruption']['paired'][k]['ci95'][1] for m in NAMES for k in COLORS)
 ax.set_ylim(-.8,max(5,ymax*1.28))
 for ext in ['pdf','svg','png']:fig.savefig(output/('fig1_native_support.'+ext),dpi=300,facecolor='white')
 plt.close(fig)
 if main_only:
  (output/'PRESENTATION_MANIFEST.json').write_text(json.dumps({'source_summary_sha256':hashlib.sha256((folder/'SUMMARY.json').read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('fig1_native_support.*')},'change':'Removed main title, subtitle and both footer notes; data, intervals, labels and encodings unchanged.','status':'rendered_pending_visual_review'},indent=2)+'\n')
  return
 fig,axes=plt.subplots(2,2,figsize=(10.5,7.5));fig.subplots_adjust(left=.085,right=.975,bottom=.15,top=.91,hspace=.68,wspace=.32)
 for ax,group,raw,title in [(axes[0,0],'corruption',False,'(a) Corruption: normalized'),(axes[0,1],'clean',False,'(b) Clean: normalized'),(axes[1,0],'corruption',True,'(c) Corruption: raw gain'),(axes[1,1],'clean',True,'(d) Clean: raw gain')]:
  panel(ax,summary,group,raw,False);ax.set_title(title,loc='left',fontweight='bold',pad=15)
 for row in axes:
  high=max(a.get_ylim()[1] for a in row)
  for a in row:a.set_ylim(-.8,high*1.12)
 axes[0,0].legend(frameon=False,ncol=2,loc='upper left')
 fig.text(.085,.045,'Matched six-candidate budgets do not imply identical search spaces. Missing spatial support scores zero.\nClean controls are separate observations; raw tIoU / sIoU gains are not a common normalized scale.',fontsize=9,color='#606D79')
 for ext in ['pdf','svg','png']:fig.savefig(folder/('native_support_controls.'+ext),dpi=240,facecolor='white')
 plt.close(fig)
 (folder/'FIGURE_MANIFEST.json').write_text(json.dumps({'source_summary_sha256':hashlib.sha256((folder/'SUMMARY.json').read_bytes()).hexdigest(),'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.png')},'main_chart':'paired normalized corruption means and 95% source intervals','status':'rendered_pending_visual_review'},indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--output');p.add_argument('--main-only',action='store_true');args=p.parse_args();run(args.folder,args.output,args.main_only)

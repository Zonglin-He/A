"""Private caption-aware layout, same frozen illustrative cases/prediction arrays."""
from common import *
import textwrap

def run():
 import numpy as np
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from stvg.decode import decode
 validate('qualitative');q=D/'qualitative';parent=read(q/'render_receipt.json');assert sha(q/'current_MAP_qualitative.png')==parent['PNG_sha256'];cases=read(q/'selection.json');fig=plt.figure(figsize=(18,14),layout='constrained');grid=fig.add_gridspec(9,6,height_ratios=[.45,3,1,.45,3,1,.45,3,1])
 for ci,c in enumerate(cases):
  ds=c['dataset'];row=next(x for x in records(ds)if x['id']==c['id']);frames,ids=decode(row['metadata'],ds);path,h=current_MAP(ds,c['id']);assert h==c['prediction_sha256']
  with np.load(path)as z:src=z['Source'].copy();full=z['FullMAP'].copy();pos=z['positions8'].tolist();iv0=z['Source_interval'].tolist();iv1=z['FullMAP_interval'].tolist()
  header=fig.add_subplot(grid[ci*3,:]);header.axis('off');caption=row['metadata']['caption'];header.text(0,.7,f"Case{ci+1} ({ds}) — Query: "+'\n'.join(textwrap.wrap(caption,160)),fontsize=12,va='center',transform=header.transAxes)
  for col,t in enumerate(c['shown_positions']):
   ax=fig.add_subplot(grid[ci*3+1,col]);ax.imshow(frames[t]);H,W=frames[t].shape[:2]
   for b,color,label in [(src[t],'deepskyblue','Native Source'),(full[t],'orangered','FullMAP')]:
    x,y,w,hh=b;ax.add_patch(Rectangle(((x-w/2)*W,(y-hh/2)*H),w*W,hh*H,fill=False,edgecolor=color,lw=2,label=label))
   ax.set_title(f"Frame {ids[t]} / "+('Expert observed'if t in pos else'Unobserved'),fontsize=10);ax.axis('off')
   if col==0:ax.legend(fontsize=8,loc='lower left')
  ax=fig.add_subplot(grid[ci*3+2,:]);fps=row['metadata']['fps'];origin=ids[0];cv=lambda x:(np.asarray(x)-origin)/fps;ax.hlines(0,*cv([ids[0],ids[-1]+1]),color='gray',lw=1);ax.scatter(cv([ids[t]for t in pos]),np.zeros(8),color='purple',marker='v',s=40,label='Eight expert observations')
  for iv,y,color,label in [(iv0,.3,'deepskyblue','Native Source event'),(iv1,.7,'orangered','Final MAP event')]:ax.broken_barh([(cv([iv[0]])[0],(iv[1]-iv[0])/fps)],(y,.25),facecolors=color,label=label)
  ax.set_yticks([]);ax.set_xlabel('Seconds from first sampled physical frame');ax.set_ylim(-.15,1.1);ax.set_title(f"Post-hoc {c['selection']}; Δt={100*c['delta']['t']:+.2f}pp, Δv={100*c['delta']['v']:+.2f}pp; boxes are predictions, no GT overlay",fontsize=10)
  if ci==0:ax.legend(loc='upper right',ncol=3,fontsize=8)
 fig.suptitle('Eight expert observations: trajectory changes on unobserved frames and event boundaries',fontsize=15);png=q/'current_MAP_qualitative_v2.png';pdf=q/'current_MAP_qualitative_v2.pdf';fig.savefig(png,dpi=150);fig.savefig(pdf);plt.close(fig);rec=dict(status='rendered_private_pending_root_visual_inspection',same3_frozen_cases=True,same_prediction_arrays=True,actual_query_captions_private=True,parent_render_receipt_sha256=sha(q/'render_receipt.json'),parent_PNG_sha256=parent['PNG_sha256'],PNG_sha256=sha(png),PDF_sha256=sha(pdf),renderer_code_sha256=sha(Path(__file__)),new_Source=0,new_expert=0,new_GPU=0,new_GT_scoring=0);atomic(q/'render_v2_receipt.json',rec);atomic(P/'qualitative/render_v2_receipt.json',rec)
if __name__=='__main__':run()

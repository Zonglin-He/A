"""Private actual-pixel engineering panels; no semantic labels or public media."""
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_soft_correspondence32_v1 import D,OLD
from scripts.run_decota_paper_main_v1 import frames_for,read_row
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def box(ax,b,w,h,color):
 x,y,bw,bh=np.asarray(b)*np.array([w,h,w,h]);ax.add_patch(Rectangle((x-bw/2,y-bh/2),bw,bh,fill=False,edgecolor=color,lw=1.5))
def run():
 assert read(D/'GLOBAL_FEEDBACK_BARRIER.json')['count']==32;cache=collections.OrderedDict();panels=[]
 for alias in ['H001','V001']:
  z=load(D/'predictions'/(alias+'.pt'));row=read_row(z['dataset'],z['query_ordinal']);frames,ids,_=frames_for(z['dataset'],row,cache);assert ids==z['frame_ids'];old=load(OLD/'predictions'/(alias+'.pt'));planned=z['planned_positions'];h,w=frames[0].shape[:2];target=max([j for j in range(len(ids))if j not in planned],key=lambda j:min(abs(ids[j]-ids[a])for a in planned));soft=z['soft_maps_per_anchor'].numpy();fig,axs=plt.subplots(3,4,figsize=(13,8),layout='constrained');fig.suptitle(alias+' engineering view: four anchors / one unobserved frame; no GT overlay')
  for j,pos in enumerate(planned):
   ax=axs[0,j];ax.imshow(frames[pos]);ax.set_title('DINO slot '+str(j));matches=[a for a in old['expert']['anchors']['single4']if a['position']==pos]
   if matches:box(ax,matches[0]['box'],w,h,'orange')
  for j in range(4):
   ax=axs[1,j];ax.imshow(frames[target]);ax.set_title('Unobserved: anchor '+str(j)+' map')
   if j<len(soft):ax.imshow(soft[j,target],extent=[0,w,h,0],alpha=.55,cmap='magma',vmin=0,vmax=1)
   box(ax,z['central'][target],w,h,'red')
  for j,label in enumerate(['Before','real mean','shuffled mean','geometry interpolation']):
   ax=axs[2,j];ax.imshow(frames[target]);ax.set_title(label);box(ax,z['central'][target],w,h,'red')
   if j in [1,2]and len(soft):ax.imshow(soft[:,target if j==1 else z['permutation'][target]].mean(0),extent=[0,w,h,0],alpha=.55,cmap='magma',vmin=0,vmax=1)
   if j==3 and len(z['admitted_positions']):
    ap=np.asarray(z['admitted_positions']);a=np.asarray([a['box']for a in old['expert']['anchors']['single4']]);order=np.argsort(ap);b=[np.interp(ids[target],np.asarray(ids)[ap[order]],a[order,c])for c in range(4)];box(ax,b,w,h,'orange')
  for ax in axs.flat:ax.axis('off')
  dest=D/'private_pixels'/(alias+'.png');dest.parent.mkdir(exist_ok=True);fig.savefig(dest,dpi=130);plt.close(fig);panels.append(dict(alias=alias,path=str(dest.relative_to(ROOT)),sha256=sha(dest),actual_RGB_cells=12,GT_overlay=False,new_semantic_identity_labels=0))
 put(D/'PIXEL_PANEL_MANIFEST.json',dict(status='rendered_pending_actual_view',count=2,panels=panels))
if __name__=='__main__':run()

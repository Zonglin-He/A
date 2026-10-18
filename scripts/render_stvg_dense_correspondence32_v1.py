"""Private actual RGB Source/Before/After panels, after CPU complete only."""
from pathlib import Path
import sys,os,collections,hashlib
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=ROOT/'artifacts/stvg_dense_correspondence32_v1';ARMS=['A_sparse','B_raw','C_rank','D_interpolation']

def run():
 assert read(D/'CPU_COMPLETION.json')['status']=='complete';assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.run_decota_paper_main_v1 import frames_for,read_row
 from scripts.stvg_opd_legacy_input_bridge003 import normalized
 import torch
 torch.set_num_threads(2)
 p=read(D/'PLAN_LOCK.json');cache=collections.OrderedDict();panels=[]
 for alias in ['H001','V001']:
  r=next(r for r in p['records']if r['alias']==alias);z=load(D/'predictions'/(alias+'.pt'));row=read_row(z['dataset'],z['query_ordinal']);frames,ids,_=frames_for(z['dataset'],row,cache);assert ids==z['frame_ids'];inp=normalized(torch.load(ROOT/r['input'],map_location='cpu',weights_only=False));assert hashlib.sha256(frames.tobytes()).hexdigest()==inp['pixel_sha256'];positions=z['planned_positions'];assert len(positions)==4
  for(_,pos),ob in z['expert']['observations'].items():assert hashlib.sha256(frames[pos].tobytes()).hexdigest()==ob['receipt']['rgb_sha256']
  def draw(ax,b,color,lw=1.8,style='-'):
   hh,ww=frames[0].shape[:2];cx,cy,bw,bh=np.asarray(b)*np.array([ww,hh,ww,hh]);ax.add_patch(Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,edgecolor=color,lw=lw,linestyle=style))
  for state in ['Source','Before','After']:
   rows=['Source']if state=='Source'else ARMS;fig,axs=plt.subplots(len(rows),4,figsize=(15,3*len(rows)),squeeze=False,layout='constrained');fig.suptitle(alias+' | '+state+' at original four DINO slots | no GT / identity labels')
   for ai,arm in enumerate(rows):
    boxes=z['Source_boxes']if state=='Source'else z['arms'][arm][state.lower()if state=='Before'else'final']
    for j,pos in enumerate(positions):
     ax=axs[ai,j];ax.imshow(frames[pos]);draw(ax,boxes[pos].numpy(),'#00ffff');matches=[v for v in z['expert']['anchors']['single4']if v['position']==pos]
     if matches:draw(ax,matches[0]['box'],'#ff00ff',1.2,'--')
     ax.set_title(arm+' | slot'+str(j)+' | cyan Student / magenta DINO',fontsize=9);ax.axis('off')
   path=D/'private_pixels'/(alias+'_'+state+'.png');path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=130);plt.close(fig);panels.append(dict(alias=alias,state=state,path=str(path.relative_to(ROOT)),sha256=sha(path),actual_RGB_cells=4*len(rows),GT_overlay=False,new_identity_semantic_labels=0,source_and_DINO_pixel_hashes_exact=True))
 put(D/'PIXEL_PANEL_MANIFEST.json',dict(status='rendered_pending_actual_view',count=6,panels=panels,actual_RGB_cells=sum(p['actual_RGB_cells']for p in panels),no_public_media=True))
 print('RENDERED6_PRIVATE72CELLS',flush=True)
if __name__=='__main__':run()

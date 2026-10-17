"""Private fixed-order actual anchor views; no automatic semantic labels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,PAPER,read,sha,put
def main():
 import numpy as np,torch
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete'
 rr=read(BASE/'ROSTER_LOCK.json')['records'];jobs=read(BASE/'INPUT_JOB_LOCK.json')['records'];chosen=[next(r for r in rr if r['dataset']==ds) for ds in ['hc2','vidstg']]
 refs={a:{r['case_id']:{x['position']:x for x in r['anchors']} for r in read(BASE/(a+'_REFERENCE_LOCK.json'))['records']} for a in ['DIRECT','HEADS','DINO']}
 refs['Current_WF']={r['case_id']:{x['position']:x for x in r['anchors']} for r in read(PAPER/'arms/B/trials/G32/REFERENCE_LOCK.json')['records']}
 bm={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']};fig,axs=plt.subplots(2,4,figsize=(18,8));bindings=[]
 for row,rec in enumerate(chosen):
  cid=rec['probe_alias'];jj=[j for j in jobs if j['case_id']==cid];j=jj[0];assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];pos=j['position'];preds={};f=bm[cid];assert sha(ROOT/f['path'])==f['sha256'];preds['Current_WF']=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)['arms']['TS_M2']
  for arm in ['DIRECT','HEADS','DINO']:
   m={r['case_id']:r for r in read(BASE/'arms'/arm/'trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']};f=m[cid];assert sha(ROOT/f['path'])==f['sha256'];preds[arm]=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False)['arms']['TS_M2']
  for col,arm in enumerate(['Current_WF','DIRECT','HEADS','DINO']):
   ax=axs[row,col];ax.imshow(rgb);z=preds[arm]['boxes'][pos];x,y,u,v=map(float,z);ax.add_patch(Rectangle((x*W,y*H),(u-x)*W,(v-y)*H,fill=False,color='cyan',lw=1.6,label='actual Student'))
   ref=refs[arm][cid].get(pos)
   if ref:
    x,y,w,h=ref['box_cxcywh'];ax.add_patch(Rectangle(((x-w/2)*W,(y-h/2)*H),w*W,h*H,fill=False,color='yellow',lw=1.6,label='sparse reference'))
   ax.set_title(rec['dataset']+' fixed first source / '+arm+'\nreference '+('available' if ref else 'absent'));ax.axis('off');ax.legend(fontsize=6)
  bindings.append(dict(case_id=cid,job_id=j['job_id'],input_sha256=j['image']['sha256'],position=pos,query_retained_private=True))
 path=BASE/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(path,dpi=130);plt.close(fig);put(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,records=bindings,path=str(path.relative_to(ROOT)),sha256=sha(path),GT_read=False));print(path)
if __name__=='__main__':main()

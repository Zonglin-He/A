"""Private deterministic input/output engineering views, no semantic labels."""
from pathlib import Path
import collections,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_vid_lr128_v1 import BASE,read,sha,put
def main():
 import torch
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts.run_decota_paper_main_v1 import frames_for
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete';maps={'LR_CURRENT':{r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']}}
 for arm in ('LR_HALF','LR_QUARTER'):maps[arm]={r['case_id']:r for r in read(BASE/'arms'/arm/'trials/G32/GLOBAL_PREDICTION_BARRIER.json')['files']}
 fig,axs=plt.subplots(2,2,figsize=(12,8));bindings=[]
 for row,rec in enumerate(read(BASE/'ROSTER_LOCK.json')['records'][:2]):
  cid=rec['probe_alias'];case,_=load_case(rec);frames,ids,_=frames_for('vidstg',dict(case,frame_ids=list(map(int,case['frame_ids']))),collections.OrderedDict());preds={}
  for a,mm in maps.items():
   r=mm[cid];assert sha(ROOT/r['path'])==r['sha256'];d=torch.load(ROOT/r['path'],map_location='cpu',weights_only=False);preds[a]=d['arms']['TS_M2']['boxes']
  for col,pos in enumerate([len(ids)//3,2*len(ids)//3]):
   ax=axs[row,col];ax.imshow(frames[pos]);H,W=frames[pos].shape[:2]
   for (a,boxes),color in zip(preds.items(),['cyan','yellow','magenta']):
    x,y,z,w=boxes[pos];ax.add_patch(Rectangle((x*W,y*H),(z-x)*W,(w-y)*H,fill=False,color=color,lw=1.3,label=a))
   ax.legend(fontsize=7);ax.set_title(f'fixed development source {row+1}, native position {pos}');ax.axis('off')
  bindings.append(dict(case_id=cid,source_case_sha256=case['source_case_sha256'],Student_sha256={a:mm[cid]['sha256'] for a,mm in maps.items()}))
 p=BASE/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig);put(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,records=bindings,path=str(p.relative_to(ROOT)),sha256=sha(p),GT_read=False));print(p)
if __name__=='__main__':main()

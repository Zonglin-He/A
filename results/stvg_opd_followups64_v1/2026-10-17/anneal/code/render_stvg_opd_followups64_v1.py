"""Private real-pixel engineering panel, no GT or Teacher request."""
from pathlib import Path
import collections,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_followups64_v1 import BASE,read,sha,put

def main(family):
 import torch,numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts.run_decota_paper_main_v1 import frames_for
 rr=read(BASE/'ROSTER_LOCK.json')['records'];bar=read(BASE/family/'STUDENT_GLOBAL_BARRIER.json');files={f['case_id']:f for f in bar['files']};fig,axs=plt.subplots(2,2,figsize=(12,8));evidence=[]
 for row,ds in enumerate(('hc2','vidstg')):
  rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];case,_=load_case(rec);entry=files[cid];assert sha(ROOT/entry['path'])==entry['sha256'];data=torch.load(ROOT/entry['path'],map_location='cpu',weights_only=False);frames,ids,_=frames_for(ds,dict(case,frame_ids=list(map(int,case['frame_ids']))),collections.OrderedDict());pp=data['original_anchor_positions'];positions=[pp[0] if pp else 0,len(ids)//2];names=['P0_F','coarse_to_fine','constant','reverse'] if family=='anneal' else ['P0_F','P0_O','P0_R','P0_R_to_S'];colors=['white','cyan','yellow','magenta']
  for col,pos in enumerate(positions):
   ax=axs[row,col];frame=frames[pos];ax.imshow(frame);H,W=frame.shape[:2]
   for name,color in zip(names,colors):
    x1,y1,x2,y2=data['arms'][name]['boxes'][pos];ax.add_patch(Rectangle((x1*W,y1*H),(x2-x1)*W,(y2-y1)*H,fill=False,color=color,lw=1.4,label=name))
   ax.set_title(f'{ds} / fixed source {row+1} / native position {pos}');ax.axis('off');ax.legend(fontsize=7)
  evidence.append(dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256'],student_sha256=entry['sha256'],positions=positions,GT_read=False))
 p=BASE/family/'private_actual_pixel_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig);put(BASE/family/'ROOT_INPUT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,records=evidence,panel_sha256=sha(p),GT_read=False));print(p)
if __name__=='__main__':main(sys.argv[1])

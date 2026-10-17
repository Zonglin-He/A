"""Deterministic private actual-frame views; no semantic outcome labels."""
from pathlib import Path
import collections,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_dino_pool_r1_32_v3 import BASE,read,sha,put
def main():
 import torch
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.stvg_reference_recovery_common_v1 import load_case
 from scripts.run_decota_paper_main_v1 import frames_for
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete';rr=read(BASE/'ROSTER_LOCK.json')['records'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};chosen=[next(r for r in rr if r['dataset']==ds and refs[r['case_id']]['common_wf_anchors']) for ds in ['hc2','vidstg']];files={r['case_id']:r for r in read(BASE/'STUDENT_GLOBAL_BARRIER.json')['files']};fig,axs=plt.subplots(2,2,figsize=(12,8));bindings=[]
 for row,rec in enumerate(chosen):
  cid=rec['case_id'];case,_=load_case(rec);frames,ids,_=frames_for(rec['dataset'],dict(case,frame_ids=list(map(int,case['frame_ids']))),collections.OrderedDict());f=files[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);anchor=refs[cid]['common_wf_anchors'][0]['position']
  for col,pos in enumerate([anchor,next(i for i in range(len(ids)) if i!=anchor)]):
   ax=axs[row,col];ax.imshow(frames[pos]);H,W=frames[pos].shape[:2]
   for a,color in [('F','cyan'),('S_Top1_ORIGINAL','yellow'),('S_R1_ORIGINAL','magenta')]:
    x,y,z,w=d['arms'][a]['boxes'][pos];ax.add_patch(Rectangle((x*W,y*H),(z-x)*W,(w-y)*H,fill=False,color=color,lw=1.3,label=a))
   ax.legend(fontsize=6);ax.set_title(f"{rec['dataset']} fixed first supported source, position {pos}");ax.axis('off')
  bindings.append(dict(case_id=cid,source_case_sha256=case['source_case_sha256'],Student_sha256=f['sha256']))
 p=BASE/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig);put(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,records=bindings,path=str(p.relative_to(ROOT)),sha256=sha(p),GT_read=False));print(p)
if __name__=='__main__':main()

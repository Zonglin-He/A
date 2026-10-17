"""Private actual-pixel engineering views; no Teacher, GT or semantic labels."""
from pathlib import Path
import collections,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,sha,write
from scripts.report_stvg_paper_frozen_v1 import BASES

def main(kind):
 import torch,numpy as np
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts.run_decota_paper_main_v1 import frames_for
 base=BASES[kind];b=base/'arms/B/trials/G32' if kind=='paper' else base;bar=read(b/'GLOBAL_PREDICTION_BARRIER.json');files={f['case_id']:f for f in bar['files']};rr=read(base/'ROSTER_LOCK.json')['records'];fig,axs=plt.subplots(2,2,figsize=(12,8));bindings=[]
 for row,ds in enumerate(('hc2','vidstg')):
  rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];case,_=load_case(rec);f=files[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);frames,ids,_=frames_for(ds,dict(case,frame_ids=list(map(int,case['frame_ids']))),collections.OrderedDict());positions=[len(ids)//3,2*len(ids)//3];names=['F','TS_M2'] if kind=='paper' else ['Source_T','M0','M1','M2'];colors=['white','cyan','yellow','magenta']
  for col,pos in enumerate(positions):
   ax=axs[row,col];ax.imshow(frames[pos]);H,W=frames[pos].shape[:2]
   for a,color in zip(names,colors):
    x1,y1,x2,y2=d['arms'][a]['boxes'][pos];ax.add_patch(Rectangle((x1*W,y1*H),(x2-x1)*W,(y2-y1)*H,fill=False,color=color,lw=1.2,label=a))
   ax.legend(fontsize=7);ax.set_title(f'{ds} fixed engineering source / native position {pos}');ax.axis('off')
  bindings.append(dict(case_id=cid,dataset=ds,source_case_sha256=case['source_case_sha256'],student_sha256=f['sha256'],positions=positions,GT_read=False))
 p=base/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig)
 paths=[str(p.relative_to(ROOT))]
 if kind=='paper':
  fig,axs=plt.subplots(2,2,figsize=(12,7));jobs=read(base/'TEACHER_JOB_LOCK.json')['records']
  for row,ds in enumerate(('hc2','vidstg')):
   j=next(j for j in jobs if j['dataset']==ds)
   for col,v in enumerate(j['videos']):
    p=ROOT/v['path'];assert sha(p)==v['sha256'];a=np.load(p,allow_pickle=False);pixels=np.concatenate([x.transpose(1,2,0) for x in a],axis=1);axs[row,col].imshow(pixels);axs[row,col].set_title(ds+(' context4' if col==0 else ' local candidate5'));axs[row,col].axis('off')
  p=base/'private_actual_WHO_inputs.png';fig.tight_layout();fig.savefig(p,dpi=140);plt.close(fig);paths.append(str(p.relative_to(ROOT)))
 write(base/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,records=bindings,panels=[dict(path=p,sha256=sha(ROOT/p)) for p in paths],GT_read=False));print('\n'.join(str(ROOT/p) for p in paths))
if __name__=='__main__':main(sys.argv[1])

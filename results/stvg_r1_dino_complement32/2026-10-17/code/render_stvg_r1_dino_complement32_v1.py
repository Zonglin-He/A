"""Fixed first source/domain first anchor, actual private reference/output pixels."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_dino_complement32_v1 import BASE,STAGE1,read,sha,put
def main():
 import numpy as np,torch
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete';rr=read(BASE/'ROSTER_LOCK.json')['records'];jobs=read(BASE/'TEACHER_JOB_LOCK.json')['records'];refs={r['case_id']:{a['position']:a for a in r['anchors']} for r in read(BASE/'D1_REFERENCE_LOCK.json')['records']};bm={r['case_id']:r for r in read(BASE/'BASELINE_BARRIER.json')['files']};fig,axs=plt.subplots(2,4,figsize=(18,8));records=[]
 for row,ds in enumerate(['hc2','vidstg']):
  rec=next(r for r in rr if r['dataset']==ds);cid=rec['probe_alias'];j=next(j for j in jobs if j['case_id']==cid);pos=j['position'];assert sha(ROOT/j['image']['path'])==j['image']['sha256'];rgb=np.load(ROOT/j['image']['path'],allow_pickle=False);H,W=rgb.shape[:2];f=bm[cid];assert sha(ROOT/f['path'])==f['sha256'];old=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds={'Current_WF':old['arms']['TS_M2']}
  for a,p in [('D0',STAGE1/'arms/DINO/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),('D1',BASE/'arms/D1/trials/G32/GLOBAL_PREDICTION_BARRIER.json'),('D2',BASE/'D2_GLOBAL_PREDICTION_BARRIER.json')]:
   m={f['case_id']:f for f in read(p)['files']};f=m[cid];assert sha(ROOT/f['path'])==f['sha256'];d=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);preds[a]=d['arm'] if a=='D2' else d['arms']['TS_M2']
  for col,a in enumerate(['Current_WF','D0','D1','D2']):
   ax=axs[row,col];ax.imshow(rgb);x,y,u,v=map(float,preds[a]['boxes'][pos]);ax.add_patch(Rectangle((x*W,y*H),(u-x)*W,(v-y)*H,fill=False,color='cyan',lw=1.5,label='actual output'));ref=refs[cid].get(pos)
   if a in ['D1','D2'] and ref:
    x,y,w,h=ref['box_cxcywh'];ax.add_patch(Rectangle(((x-w/2)*W,(y-h/2)*H),w*W,h*H,fill=False,color='yellow',lw=1.4,label='same D1 reference'))
   ax.set_title(ds+' fixed first source / '+a);ax.axis('off');ax.legend(fontsize=6)
  records.append(dict(case_id=cid,job_id=j['job_id'],input_sha256=j['image']['sha256']))
 p=BASE/'private_actual_output_panel.png';fig.tight_layout();fig.savefig(p,dpi=130);plt.close(fig);put(BASE/'ROOT_PIXEL_VIEW_BINDING.json',dict(status='prepared_pending_actual_view',count=2,path=str(p.relative_to(ROOT)),sha256=sha(p),records=records,GT_read=False));print(p)
if __name__=='__main__':main()

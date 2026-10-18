"""Cached selected source-validation teacher discrimination; no inference/training."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dfine_source_capture_v1 import D,P
from scripts.run_stvg_dfine_refinement32_v1 import verify
from vg_tta.stvg_cosine_candidate_headroom32_v1 import xy,iou,summarize_frame
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
from scripts.score_stvg_joint_robust_correction32_v1 import paired

def main():
 torch.set_num_threads(4);p=verify();sources=read(P/'SOURCE_TRAINING.json')['rows'];rows=[]
 for m in sources:
  ds=m['source_domain'];val=load(D/'source_validation'/ds/f"epoch{m['selected']['epoch']:02}.pt");records={r['alias']:r for r in p['source']if r['dataset']==ds};
  for o in val['packets']:
   a=o['alias'];r=records[a];truth=load(D/'source_supervision'/(a+'.pt'));x=load(D/'source_features'/(a+'.pt'));fits=o['paths']['Fixed_distill'];b=torch.cat([s['candidates']for s in fits],1);rew=torch.cat([s['rewards']for s in fits],1);prior=torch.cat([s['prior']for s in fits],1);prob=(prior+rew/.1).softmax(-1);valid=(cell_overlap(b,*x['valid'].shape[-2:])*x['valid'][:,None]).sum((-1,-2))>1e-12;valid[:,0]=True;xx=xy(b.numpy(),dict(input=r['input']),ds);after=xy(o['outputs']['Requery_OPD'],dict(input=r['input']),ds);one=xy(o['paths']['Requery_OPD'][0]['boxes'].numpy(),dict(input=r['input']),ds);frame=[]
   for t,fid in enumerate(x['frame_ids']):
    if t in x['planned_positions']or int(fid)not in truth['gt']:continue
    gt=np.array(truth['gt'][int(fid)]);quality=iou(xx[t],gt);v=summarize_frame(quality,rew[t].numpy(),prob[t].numpy(),valid[t].numpy(),quality[0],o['active'],float(iou(after[t],gt)),float(iou(one[t],gt)));frame.append(v)
   keys=['raw_auc','posterior_auc','strong_pool_raw_top_good','strong_pool_posterior_top_good','strong_pool_no_good_outranks_E_raw','strong_pool_no_good_outranks_E_posterior','best_gain_pp']
   rows.append(dict(alias=a,dataset=ds,source_validation_only=True,active=o['active'],known_unobserved_nodes=len(frame),means={k:float(np.mean([f[k]for f in frame if f[k]is not None]))if [f[k]for f in frame if f[k]is not None]else None for k in keys}))
 groups={ds:dict(count=16,active=sum(r['active']for r in rows if r['dataset']==ds),statistics={k:paired([r['means'][k]for r in rows if r['dataset']==ds])for k in rows[0]['means']})for ds in ['hc2','vidstg']};put(P/'SOURCE_FEEDBACK_RANKING_PUBLIC.json',dict(status='complete',count=32,groups=groups,rows=rows,no_new_model_or_candidate_or_update=True,source_validation_only=True,target_status=read(P/'TARGET_STATUS.json')['status']))
 a=read(D/'PUBLIC_FILES_FINAL.json');a['files']+=['scripts/analyze_stvg_dfine_source_feedback_v1.py',str((P/'SOURCE_FEEDBACK_RANKING_PUBLIC.json').relative_to(ROOT))];put(D/'PUBLIC_FILES_FINAL.json',a,True)
 print({ds:{k:v['mean']for k,v in g['statistics'].items()}for ds,g in groups.items()},flush=True)
if __name__=='__main__':main()

"""Independent postseal GT arithmetic for local/shared4/unobserved accounting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_dino_mask_align32_v1 import D,P,verify


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};private=load(D/'SCORED_FEEDBACK_PRIVATE.pt');public={x['alias']:x for x in read(P/'RESULTS_PUBLIC.json')['rows']};count=0;error=0.
    for rec,prior in zip(p['records'],private):
        z=load(D/'predictions'/(rec['alias']+'.pt'));e=load(ROOT/rec['prior_prediction']);source=np.asarray(e['outputs']['Source_no_correction'],float);ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];g=gt[ds][0][q];factor=np.array([row['input']['width'],row['input']['height']]*2)
        def tube(action):
            delta=np.asarray(action)*z['scale'];return np.c_[source[:,:2]+delta[:2],np.exp(np.log(source[:,2:])+delta[2:])]
        def iou(box,truth):
            box=np.asarray(box,float);xy=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]*factor
            if ds=='hc2':xy=np.maximum(xy,0)
            inter=np.maximum(np.minimum(xy[2:],truth[2:])-np.maximum(xy[:2],truth[:2]),0).prod();union=np.maximum(xy[2:]-xy[:2],0).prod()+np.maximum(np.asarray(truth[2:])-truth[:2],0).prod()-inter
            return float(inter/union)if union>0 else 0.
        for rr in prior['rows']:
            m,k,pos=rr['mode'],rr['sample'],rr['position'];truth=g[int(z['frame_ids'][pos])];v=z['matched'][m]['details'][k]['anchors'].get(pos)
            expected=dict(input=iou(tube(z['actions'][k])[pos],truth),E=iou(z['outputs']['E_MBR'][pos],truth),shared=iou(tube(z['matched'][m]['targets'][k])[pos],truth),teacher=None if v is None else iou(v['box'],truth))
            for key,x in expected.items():
                if x is None:assert rr[key]is None
                else:error=max(error,abs(x-rr[key]));count+=1
        for arm,boxes in z['outputs'].items():
            for observed,key in [(False,'unobserved_actual_s'),(True,'observed_actual_s')]:
                values=[iou(boxes[j],g[int(fid)])for j,fid in enumerate(z['frame_ids'])if int(fid)in g and (j in z['planned_positions'])==observed]
                actual=float(np.mean(values))if values else None;v=public[rec['alias']]['values'][arm][key]
                if actual is None:assert v is None
                else:error=max(error,abs(actual-v));count+=1
    assert error<1e-12,error
    put(P/'ROOT_ACCOUNTING_AUDIT.json',dict(status='pass',count=32,independent_local_shared_and_actual_frame_IoU_checks=count,max_error=error,GT_only_after_fullseal=True,no_new_model_or_updates=True))
    print(dict(status='pass',checks=count,error=error))


if __name__=='__main__':main()

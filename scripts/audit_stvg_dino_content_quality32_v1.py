"""Independent GT geometry audit after complete32 barrier; no model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_content_match32_v1 import D,P,verify


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    p=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
    private=load(D/'SCORED_ANCHOR_ROWS_PRIVATE.pt');count=0;maxerr=0.
    for r,pr in zip(p['records'],private):
        z=load(D/'predictions'/(r['alias']+'.pt'));e=load(ROOT/r['prior_prediction']);b=np.asarray(e['outputs']['Source_no_correction'],float);ds=r['dataset'];row=plans[ds][r['query_ordinal']];truth=gt[ds][0][r['query_ordinal']]
        factor=np.array([row['input']['width'],row['input']['height']]*2)
        def iou(box,g):
            box=np.asarray(box,np.float64);xy=np.r_[box[:2]-box[2:]/2,box[:2]+box[2:]/2]*factor
            if ds=='hc2':xy=np.maximum(xy,0)
            inter=np.maximum(np.minimum(xy[2:],g[2:])-np.maximum(xy[:2],g[:2]),0).prod();union=np.maximum(xy[2:]-xy[:2],0).prod()+np.maximum(np.array(g[2:])-g[:2],0).prod()-inter
            return inter/union if union>0 else 0.
        for record in pr['records']:
            pos=record['position'];fid=int(z['frame_ids'][pos]);k=record['sample']
            assert record['known_GT']==(fid in truth)
            if fid not in truth:continue
            s=z['actions'][k]*np.asarray(p['scale'][ds]);inp=np.r_[b[pos,:2]+s[:2],np.exp(np.log(b[pos,2:])+s[2:])]
            expected=iou(inp,truth[fid]);maxerr=max(maxerr,abs(expected-record['input_iou']));count+=1
            for mode in ['nearest','target_high']:
                a=z['matched'][mode]['details'][k]['anchors'].get(pos)
                raw=iou(a['box'],truth[fid])if a else expected
                maxerr=max(maxerr,abs(raw-record['arms'][mode]['raw_iou']));count+=1
    assert maxerr<1e-10,maxerr
    put(P/'ROOT_GEOMETRY_AUDIT.json',dict(status='pass',count=32,independent_anchor_IoU_values=count,max_error=maxerr,GT_only_after_global_seal=True,
        independent_numpy_intersection_union=True,raw_boxes_and_GT_not_exported=True))
    print(dict(status='pass',values=count,max_error=maxerr))


if __name__=='__main__':main()

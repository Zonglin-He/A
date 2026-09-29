"""Post-seal collateral metrics promised in protocol, without new model inference."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.special import logsumexp
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_causal_round2_v1 import OUT,BASE,ARMS,verify
from scripts.score_tastvg_causal_round2_v1 import stats
from scripts.score_tastvg_evidence_vulnerability_v1 import arr


def box_iou(a,b):
    a=arr(a);b=arr(b);aa=np.c_[a[:,:2]-a[:,2:]/2,a[:,:2]+a[:,2:]/2];bb=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]
    inter=np.maximum(0,np.minimum(aa[:,2:],bb[:,2:])-np.maximum(aa[:,:2],bb[:,:2])).prod(1)
    return inter/np.maximum(a[:,2:].prod(1)+b[:,2:].prod(1)-inter,1e-30)


def run():
    torch.set_num_threads(4);p=verify();assert (OUT/'analysis/ORACLE_AUDIT.json').exists();gt=read(OUT/'GT_SUBSET.json');rows=[]
    for row in p['rows']:
        key=row['key'];stem=key.replace(':','_');b=load(BASE/'capture'/f'{stem}.pt');base=b['prediction'];valid=np.array(gt[key]['valid'],bool);inside=np.zeros(len(valid),bool);l,r=base['indices'];inside[l:r+1]=True
        entry=dict(key=key,cohort=row['cohort'],baseline_interval_GT_support_fraction=float((valid&inside).sum()/valid.sum()),arms={})
        for arm in ARMS:
            a=load(OUT/'arms'/f'{stem}_{arm}.pt');now=a['prediction'];iou=box_iou(base['boxes'],now['boxes']);kl=[]
            for x,y in zip(base['logits'],now['logits']):
                x=arr(x);y=arr(y);lp=x-logsumexp(x,axis=1,keepdims=True);lq=y-logsumexp(y,axis=1,keepdims=True);kl.append(float((np.exp(lp)*(lp-lq)).sum(1).mean()))
            included=np.zeros(len(valid),bool);u,v=now['indices'];included[u:v+1]=True
            entry['arms'][arm]=dict(endpoint_KL_base_to_new=float(np.mean(kl)),all_frame_box_iou=float(iou.mean()),GT_frame_box_self_iou=float(iou[valid].mean()),outside_baseline_interval_box_iou=float(iou[~inside].mean()) if (~inside).any() else None,newly_lost_GT_supported_frames=int((valid&inside&~included).sum()),newly_gained_GT_supported_frames=int((valid&~inside&included).sum()))
        rows.append(entry)
    summary={}
    for cohort in ('hcstvg1_test','vidstg_test','all'):
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];s=dict(baseline_protected_interval_GT_coverage=stats([r['baseline_interval_GT_support_fraction'] for r in rr]),arms={})
        for arm in ARMS:s['arms'][arm]={m:stats([r['arms'][arm][m] for r in rr]) for m in rows[0]['arms'][arm]}
        summary[cohort]=s
    write(OUT/'analysis/COLLATERAL_ROWS.json',rows);write(OUT/'analysis/COLLATERAL_SUMMARY.json',summary)
    print('Collateral readback64 x6, no inference')
if __name__=='__main__':run()

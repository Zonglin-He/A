"""Postseal teacher/association separation; no new predictions or selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_observation_factorial32_v1 import D,P,PRE,verify

def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.score_stvg_dino_mask_align32_v1 import iou_gt
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    from scripts.run_stvg_dino_local_residual32_v1 import mean
    plan=verify();assert read(D/'GLOBAL_PREDICTION_BARRIER.json')['count']==32;assert not torch.cuda.is_initialized()
    gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in plan['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    frames=[];queries=[]
    for r in plan['records']:
        ds=r['dataset'];q=r['query_ordinal'];a=r['alias'];row=read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows'][q]
        z=load(D/'predictions'/(a+'.pt'));s=load(PRE/'source'/(a+'.pt'));e=load(PRE/'E'/(a+'.pt'));truth=gt[ds][0][q]
        original={p['position']:s['expert']['observations'][('original',p['position'])]['probe']for p in s['expert']['anchors']['single4']}
        pool=dict(original);crop={ob['position']:ob for ob in z['spatial']['observations']}
        for pos,ob in crop.items():
            if ob.get('probe',{}).get('accepted',False):pool[pos]=ob['probe']
        selected=dict(zip(sorted(pool),z['spatial']['estimate']['selected']));oldselected=dict(zip(original,e['estimate']['selected']))
        rows=[]
        for pos in e['planned_positions']:
            fid=int(z['frame_ids'][pos]);ob=crop[pos];rawold=s['expert']['observations'].get(('original',pos),{}).get('probe',{})
            cp=ob.get('probe',{});known=fid in truth
            def score(b):return iou_gt(np.asarray(b,float),truth[fid],row,ds)if known and b is not None else None
            def picked(pr):
                return pr['boxes'][int(np.argmax(pr['target_scores']))]if len(pr.get('target_scores',[]))else None
            newchoice=pool[pos]['boxes'][selected[pos]]if pos in selected and selected[pos]>=0 else None
            oldchoice=original[pos]['boxes'][oldselected[pos]]if pos in oldselected and oldselected[pos]>=0 else None
            oldtop=score(picked(rawold));newtop=score(picked(cp));baseline=score(z['outputs']['E_MBR'][pos])
            f=dict(alias=a,dataset=ds,position=pos,GT_known=known,called=ob['called'],crop_admitted=cp.get('accepted',False),
                original_admitted=rawold.get('accepted',False),new_E_evidence_origin='crop'if cp.get('accepted',False)else'original_fallback'if pos in pool else'none',
                original_raw_top_IoU=oldtop,crop_raw_top_IoU=newtop,E_IoU=baseline,
                original_E_selected_reference_IoU=score(oldchoice),new_E_selected_reference_IoU=score(newchoice),
                propagated_S_IoU=score(z['outputs']['S_crop'][pos]),
                raw_crop_minus_raw_full_pp=100*(newtop-oldtop)if newtop is not None and oldtop is not None else None,
                raw_crop_minus_E_pp=100*(newtop-baseline)if newtop is not None and baseline is not None else None)
            rows.append(f);frames.append(f)
        queries.append(dict(alias=a,dataset=ds,
            raw_crop_minus_raw_full_pp=mean([f['raw_crop_minus_raw_full_pp']for f in rows]),
            raw_crop_minus_E_pp=mean([f['raw_crop_minus_E_pp']for f in rows]),
            accepted_crop_minus_E_pp=mean([f['raw_crop_minus_E_pp']for f in rows if f['crop_admitted']]),
            common_known_frames=sum(f['raw_crop_minus_raw_full_pp']is not None for f in rows)))
    groups={ds:{ep:paired([q[ep]for q in queries if q['dataset']==ds])for ep in ['raw_crop_minus_raw_full_pp','raw_crop_minus_E_pp','accepted_crop_minus_E_pp']}for ds in gt}
    put(P/'REFERENCE_SEPARATION_PUBLIC.json',dict(status='postseal_diagnostic',groups=groups,queries=queries,frames=frames,
        raw_top_not_estimator_selection=True,unknown_remains_unknown=True,not_main_denominator=True,no_prediction_changed=True))
    print(groups,flush=True)

if __name__=='__main__':main()

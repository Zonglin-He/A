"""Independent readback of all saved student actions, conditional targets and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,load,put,sha
from scripts.run_stvg_dino_conditional_opd32_v1 import D,P,ARMS,verify


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,box_iou
    from scripts.score_stvg_joint_robust_correction32_v1 import paired
    p=verify(True);r=read(P/'RESULTS_PUBLIC.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert len(bar['files'])==32
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    gtall={d:truths(d,dict(parents=[x['query_ordinal']for x in p['records']if x['dataset']==d]))for d in ['hc2','vidstg']}
    plans={d:read(ROOT/f'artifacts/decota_paper_experiments_v1/{d}/PLAN.json')['rows']for d in gtall}
    rows={x['alias']:x for x in r['rows']};counts=dict(actions=0,conditional_anchor_boxes=0,distill_rounds=0,trajectory_arrays=0,metric_scalars=0,pair_bootstraps=0)
    qualification={};logical=unique=full=0
    def chart(b):b=np.asarray(b,np.float64);return np.concatenate([b[:,:2],np.log(b[:,2:])],1)
    def tube(z):return np.concatenate([z[:,:2],np.exp(z[:,2:])],1)
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));old=load(ROOT/rec['prior_prediction']);b=np.asarray(old['outputs']['Source_no_correction']);scale=np.array(p['scale'][rec['dataset']]);init=np.array(old['estimate']['delta'])/scale
        assert np.array_equal(z['outputs']['Source_MBR'],b)and np.array_equal(z['outputs']['E_MBR'],old['outputs']['E_DINO_support'])and z['interval']==old['interval']
        eps=np.random.default_rng(p['seed']+int(hashlib.sha256(rec['alias'].encode()).hexdigest()[:8],16)).standard_normal((24,4))
        for arm in ARMS[2:]:
            mu=init.copy();fit=z['fits'][arm]
            for j,step in enumerate(fit['rounds']):
                acts=np.repeat(init[None],24,0)if arm=='One_correction'else init+eps if arm=='Fixed_sample_distill'else mu+eps[j*8:j*8+8]
                assert np.array_equal(acts,step['actions']);targets=[]
                for action,detail in zip(acts,step['details']):
                    ref=tube(chart(b)+action*scale);valid=np.isfinite(ref[z['planned_positions']]).all()and ((ref[z['planned_positions']]>0)&(ref[z['planned_positions']]<1)).all()
                    assert bool(valid)==detail['legal'];ob=[]
                    for pos,v in detail['anchors'].items():
                        assert v['accepted']==(v['score']>=.35)and 0<=v['slot']<900
                        if v['accepted']:ob.append(chart(np.array([v['box']]))[0]-chart(b)[pos])
                        counts['conditional_anchor_boxes']+=1
                    target=np.mean(ob,0)/scale if ob else action
                    assert np.max(abs(target-detail['teacher_action']))<1e-12;targets.append(target);counts['actions']+=1
                targets=np.array(targets);assert np.max(abs(targets-step['targets']))<1e-12
                new=targets.mean(0);assert np.max(abs(new-step['student_mean_after']))<1e-12
                assert abs(np.square(targets-mu).sum(-1).mean()/2-step['loss_before'])<1e-10
                assert abs(np.square(targets-new).sum(-1).mean()/2-step['loss_after'])<1e-10
                mu=new;counts['distill_rounds']+=1
            output=tube(chart(b)+mu*scale)if z['active']else z['outputs']['E_MBR']
            assert np.max(abs(output-z['outputs'][arm]))<1e-12;counts['trajectory_arrays']+=1
        if z['qualifications']:
            assert all(q['status']=='pass'and q['native_replay_max_error']==0 and q['cloned_native_box_max_error']<1e-4 and q['native_context_max_error']<1e-4 for q in z['qualifications'])
            qualification[rec['dataset']]=z['qualifications']
        ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=gtall[ds][0][q],gtall[ds][1][q];ids=np.array(z['frame_ids']);un=np.array([int(t)in gt for t in ids])&~np.isin(np.arange(len(ids)),z['planned_positions'])
        for arm,boxes in z['outputs'].items():
            v=official(boxes,row,gt,span,z['interval'],ds)
            xy=np.c_[boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            v['unobserved_actual_s']=float(box_iou(xy[un],np.array([gt[int(t)]for t in ids[un]])).mean())if un.any()else None
            for ep,x in v.items():
                y=rows[rec['alias']]['values'][arm][ep];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12;counts['metric_scalars']+=1
        logical+=z['compute']['logical_candidate_queries'];unique+=z['compute']['unique_candidate_queries'];full+=z['compute']['full_image_text_encodings']
    assert set(qualification)=={'hc2','vidstg'}
    for ds,g in r['groups'].items():
        rr=[x for x in r['rows']if x['dataset']==ds]
        for pair,ends in g['paired'].items():
            a,b=pair.split('_minus_')
            for ep,v in ends.items():
                got=paired([None if x['values'][a][ep]is None or x['values'][b][ep]is None else 100*(x['values'][a][ep]-x['values'][b][ep])for x in rr]);assert got==v;counts['pair_bootstraps']+=1
        for a in ARMS:
            for ep,val in g['metrics'][a].items():
                vv=[x['values'][a][ep]for x in rr if x['values'][a][ep]is not None];assert abs(float(np.mean(vv))-val)<1e-14
    put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=counts,both_domain_decoder_interface_qualification=qualification,
        logical_teacher_reference_queries=logical,unique_teacher_reference_queries=unique,new_DINO_encoded_frames=full,new_Source_calls=0,
        all_time_intervals_exact=True,no_GT_candidate_or_update_selection=True,all_pretrained_weights_frozen_exact=True,
        no_new_identity_semantic_labels=True,all_signed_unknowns_retained=True,GT_metrics_recomputed=True))
    print(counts)


if __name__=='__main__':main()

"""Seal both complete32 arms before GT; account local/shared4/actual update."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,argparse,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_dino_mask_align32_v1 import D,P,PRE,ARMS,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired
from vg_tta.stvg_error_state_correction_v1 import encode,decode


def mean(v):
    v=[float(x)for x in v if x is not None];return float(np.mean(v))if v else None


def iou_gt(box,truth,row,ds):
    from vg_tta.tastvg_oracle_event5_v1 import box_iou
    b=np.asarray(box,float);xy=np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
    if ds=='hc2':xy=np.maximum(xy,0)
    return float(box_iou(xy[None],np.asarray(truth,float)[None])[0])


def full_metrics(z,row,gt,span,ds):
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube
    ids=np.array(z['frame_ids']);known=np.array([int(fid)in gt for fid in ids]);observed=np.isin(np.arange(len(ids)),z['planned_positions']);values={}
    for arm,b in z['outputs'].items():
        v=official(b,row,gt,span,z['interval'],ds);other=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(v[k]-other[k])for k in ['v','t','s'])<2e-10
        si=np.array([iou_gt(b[i],gt[int(ids[i])],row,ds)if known[i]else np.nan for i in range(len(ids))])
        values[arm]=dict(**v,unobserved_actual_s=mean(si[known&~observed]),observed_actual_s=mean(si[known&observed]),known_unobserved_frames=int((known&~observed).sum()))
    assert len(set(x['t']for x in values.values()))==1;return values


def summarize(rows,arms,pairs):
    groups={}
    for ds in ['hc2','vidstg']:
        rr=[x for x in rows if x['dataset']==ds]
        groups[ds]=dict(queries=16,active=sum(x['active']for x in rr),metrics={a:{k:mean([x['values'][a][k]for x in rr])for k in ['v','t','s','unobserved_actual_s','observed_actual_s']}for a in arms},
            paired={a+'_minus_'+b:{k:paired([None if x['values'][a][k]is None or x['values'][b][k]is None else 100*(x['values'][a][k]-x['values'][b][k])for x in rr])for k in ['v','unobserved_actual_s','observed_actual_s']}for a,b in pairs})
    return groups


def score(dynamic=False):
    from scripts.score_stvg_opd_p1_v1 import truths
    import torch
    p=verify();folder=D/'dynamic'if dynamic else D;bar=read(folder/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    allgt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};torch.set_num_threads(4)
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in allgt};rows=[];private=[];start=time.perf_counter()
    baseline={x['alias']:x for x in read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional_delta32/RESULTS_PUBLIC.json')['rows']}
    for rec in p['records']:
        alias=rec['alias'];ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=allgt[ds][0][q],allgt[ds][1][q]
        z=load(folder/'predictions'/(alias+'.pt'));ids=z['frame_ids'];e=load(ROOT/rec['prior_prediction']);source=np.asarray(e['outputs']['Source_no_correction'],float)
        if dynamic:
            fixed=load(D/'predictions'/(alias+'.pt'));z['outputs']['B_aligned_once']=fixed['outputs']['B_aligned_once']
        values=full_metrics(z,row,gt,span,ds)
        for a in ['Source_MBR','E_MBR']:
            for k in ['v','t','s','unobserved_actual_s']:
                x,y=values[a][k],baseline[alias]['values'][a][k];assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12
        out=dict(alias=alias,dataset=ds,active=z['active'],values=values,compute=z['compute'])
        if not dynamic:
            out['feedback']={};out['accounting']={};out['layer_scores']={};details=[]
            for m in ['A_unaligned','B_aligned']:
                res=z['matched'].get(m,dict(details=[],targets=np.empty((0,4))));dd=res['details'];c=collections.Counter();local=[];layer=[[]for _ in range(6)]
                for k,d in enumerate(dd):
                    sampled=decode(encode(source)+z['actions'][k]*z['scale']);shared=decode(encode(source)+res['targets'][k]*z['scale'])
                    c['candidates']+=1;c['legal_candidates']+=d['legal'];c['candidates_with_feedback']+=d['accepted_anchors']>0
                    for pos in z['planned_positions']:
                        v=d['anchors'].get(pos);known=int(ids[pos])in gt;c['planned_refs']+=1;c['known_GT_refs']+=known;c['returned_refs']+=v is not None;c['accepted_refs']+=bool(v and v['accepted']);c['known_accepted_refs']+=bool(known and v and v['accepted'])
                        if not known:continue
                        gi=gt[int(ids[pos])];ini=iou_gt(sampled[pos],gi,row,ds);ee=iou_gt(z['outputs']['E_MBR'][pos],gi,row,ds);ss=iou_gt(shared[pos],gi,row,ds);raw=None if v is None else iou_gt(v['box'],gi,row,ds);accepted=bool(v and v['accepted'])
                        t=dict(mode=m,sample=k,position=pos,accepted=accepted,input=ini,E=ee,teacher=raw,shared=ss)
                        local.append(t);details.append(t)
                for pos,traces in z['traces'].items():
                    tr=traces[m]
                    for l in range(6):layer[l].extend(tr['scores'][l,traces['inverse']].tolist())
                accepted=[x for x in local if x['accepted']]
                def gated(a,b):return 100*mean([x[a]-x[b]if x['accepted']else 0. for x in local])if local else 0.
                out['feedback'][m]=dict(**dict(c),query_with_feedback=any(x['accepted_anchors']for x in dd))
                out['accounting'][m]=dict(gated_teacher_minus_sample_pp=gated('teacher','input'),gated_teacher_minus_E_pp=gated('teacher','E'),
                    gated_shared_minus_E_pp=gated('shared','E'),gated_compression_minus_local_pp=gated('shared','teacher'),
                    accepted_only_teacher_minus_sample_pp=None if not accepted else 100*mean([x['teacher']-x['input']for x in accepted]),
                    accepted_only_teacher_minus_E_pp=None if not accepted else 100*mean([x['teacher']-x['E']for x in accepted]),
                    accepted_only_shared_minus_teacher_pp=None if not accepted else 100*mean([x['shared']-x['teacher']for x in accepted]),
                    known_GT_samples=len(local),accepted_known_GT_samples=len(accepted))
                out['layer_scores'][m]=[mean(v)for v in layer]
            private.append(dict(alias=alias,dataset=ds,rows=details))
        rows.append(out)
    if dynamic:
        arms=['Source_MBR','E_MBR','B_aligned_once','B_aligned_dynamic'];pairs=[('B_aligned_dynamic',b)for b in ['E_MBR','B_aligned_once']]
        groups=summarize(rows,arms,pairs);decision=dict(status='bounded_dynamic_complete_no_automatic_expansion',matching_cumulative_denominator=24)
        put(P/'DYNAMIC_RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,decision=decision,all_outputs_before_new_GT=True))
    else:
        groups=summarize(rows,ARMS,[('A_unaligned_once','E_MBR'),('B_aligned_once','E_MBR'),('B_aligned_once','A_unaligned_once')]);passed=True
        for ds,g in groups.items():
            rr=[r for r in rows if r['dataset']==ds];g['feedback']={};g['accounting']={};g['layers']={}
            for m in ['A_unaligned','B_aligned']:
                c=collections.Counter()
                for r in rr:c.update({k:v for k,v in r['feedback'][m].items()if k!='query_with_feedback'})
                g['feedback'][m]=dict(**dict(c),queries_with_feedback=sum(r['feedback'][m]['query_with_feedback']for r in rr))
                keys=next(iter(rr))['accounting'][m];g['accounting'][m]={k:paired([r['accounting'][m][k]for r in rr])for k in keys if k not in ['known_GT_samples','accepted_known_GT_samples']}
                g['layers'][m]=[mean([r['layer_scores'][m][l]for r in rr])for l in range(6)]
            ok=g['feedback']['B_aligned'].get('accepted_refs',0)>g['feedback']['A_unaligned'].get('accepted_refs',0)and g['paired']['B_aligned_once_minus_E_MBR']['v']['mean']>0
            g['pass_for_dynamic']=bool(ok);passed &=ok
        decision=dict(status='qualified_for_matched_weight_dynamic3x8'if passed else 'stop_after_mask_pair_and_once_update',pass_for_dynamic=bool(passed),
            rule='both domains B coverage>A and B once complete vIoU>E',no_threshold_or_layer_selection=True,no_automatic_expansion=True)
        save(D/'SCORED_FEEDBACK_PRIVATE.pt',private);put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,decision=decision,
            old_exposed_development=True,all_outputs_before_new_GT=True,GT_not_used_for_admission_update=True,query_level_bootstrap=True))
        put(P/'DECISION.json',decision)
    put(folder/'CPU_COMPLETION.json',dict(status='complete',count=32,score_seconds=time.perf_counter()-start,GT_only_after_all32_seal=True))
    print(dict(groups={d:dict(metrics=g['metrics'],feedback=g.get('feedback'),layers=g.get('layers'))for d,g in groups.items()},decision=decision))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--dynamic',action='store_true');score(a.parse_args().dynamic)

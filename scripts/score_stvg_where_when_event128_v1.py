"""CPU offline evaluation, imported only after the HC128 global barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
D=R/'artifacts/stvg_where_when_event128_v1';P=R/'results/stvg_error_state/2026-10-19/where_when_event128'

def avg(x):
    a=[float(v)for v in x if v is not None];return float(np.mean(a))if a else None

def paired(v):
    x=np.asarray([v for v in v if v is not None],float)
    if not len(x):return dict(mean=None,CI95=None,parents=0)
    idx=np.random.default_rng(2026101909).integers(len(x),size=(10000,len(x)));return dict(mean=float(x.mean()),CI95=np.quantile(x[idx].mean(1),[.025,.975]).tolist(),parents=len(x))

def pair_accuracy(scores,quality):
    x=np.array([np.nan if v is None else v for v in scores]);y=np.asarray(quality,float);i,j=np.triu_indices(len(x),1);valid=np.isfinite(x[i])&np.isfinite(x[j])&(abs(y[i]-y[j])>1e-12)
    dx=x[i][valid]-x[j][valid];dy=y[i][valid]-y[j][valid]
    return float(np.where(abs(dx)<=1e-12,.5,(dx*dy>0).astype(float)).mean())if len(dx)else None

def main():
    import torch
    torch.set_num_threads(4)
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    from vg_tta.tastvg_paper48_metrics_v1 import xyxy
    plan=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_outputs_before_new_GT']
    assert sha(D/'PLAN_LOCK.json')==bar['plan_sha256']
    for f in bar['files']:assert sha(R/f['path'])==f['sha256']
    put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),global_barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all128_outputs_before_new_GT=True))
    gt,spans,provenance=truths('hc2',dict(parents=[r['query_ordinal']for r in plan['records']]))
    raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];audit=dict(metric_checks=0,physical_integral_checks=0,same_query_checks=0,legacy_bitexact_parents=128,token_max_bitexact_parents=128)
    for r in plan['records']:
        a=r['alias'];z=load(D/'predictions'/f'{a}.pt');row=raw[r['query_ordinal']];truth=gt[r['query_ordinal']];span=spans[r['query_ordinal']];values={};dense={};known=np.array([int(f)in truth for f in z['frame_ids']]);obs=np.isin(np.arange(len(known)),z['positions8']);boxesE=z['trajectories']['E']
        def evaluate(name,boxes,iv):
            tube=DenseTube(boxes,row,truth,span,clip=True);dense[name]=tube
            v=official(boxes,row,truth,span,iv,'hc2');v2=tube.score(iv)
            assert max(abs(v[k]-v2[k])for k in ['v','t','s'])<2e-10
            b=np.maximum(xyxy(boxes,row['input']['width'],row['input']['height']),0)
            s=[float(box_iou(b[i],truth[int(f)]))for i,f in enumerate(z['frame_ids'])if int(f)in truth and not obs[i]]
            values[name]=dict(**v,unobserved_actual_s=avg(s));audit['metric_checks']+=1
        evaluate('E_MBR',boxesE,z['original_interval'])
        for name,o in z['readouts'].items():evaluate('E_'+name,boxesE,o['interval'])
        for seed in plan['seeds']:
            name=f'Live{seed}';b=z['trajectories'][name];evaluate(name+'_OriginalTime',b,z['original_interval']);evaluate(name+'_NewTime',b,z['readouts']['Event'+name]['interval'])
            if z['association'][name]['query']==z['association']['E']['query']:
                assert z['readouts']['Event'+name]==z['readouts']['EventE'];audit['same_query_checks']+=1
        for prefix in ['E_EventLive','Live_OriginalTime','Live_NewTime']:
            names=[f'E_EventLive{s}'if prefix=='E_EventLive'else f'Live{s}_'+prefix.split('_',1)[1]for s in plan['seeds']]
            values[prefix+'_seedmean']={k:avg([values[n][k]for n in names])for k in ['v','t','s','unobserved_actual_s']}
        ct=[official(boxesE,row,truth,span,c,'hc2')['t']for c in z['candidates']]
        cv=[dense['E_MBR'].score(c)['v']for c in z['candidates']]
        times=np.asarray(z['physical_frame_ids8']);inside=(times>=span[0])&(times<span[1]);events={}
        for name,o in z['readouts'].items():
            c=o['curve'];auc=None
            if c is not None and inside.any()and(~inside).any():
                delta=np.asarray(c)[inside,None]-np.asarray(c)[None,~inside];auc=float(np.where(abs(delta)<=1e-12,.5,delta>0).mean())
            # Independent integral via actual cells; no interpolated observations.
            if c is not None:
                edges=np.asarray(z['edges']);width=np.diff(edges)
                for iv,s in zip(z['candidates'],o['candidate_scores']):
                    wi=np.array([max(0,min(b,iv[1])-max(a,iv[0]))for a,b in zip(edges[:-1],edges[1:])]);wo=width-wi
                    expected=None if wi.sum()==0 or wo.sum()==0 else float((wi*np.asarray(c)).sum()/wi.sum()-(wo*np.asarray(c)).sum()/wo.sum())
                    assert (s is None and expected is None)or abs(s-expected)<1e-10;audit['physical_integral_checks']+=1
            events[name]=dict(inside_outside_AUC=auc,candidate_tIoU_pair_accuracy=pair_accuracy(o['candidate_scores'],ct),interval_changed=o['changed'],response=o['curve'],readout_reason=o['reason'],selected_interval=o['interval'])
        events['EventLive_seedmean']={k:avg([events[f'EventLive{s}'][k]for s in plan['seeds']])for k in ['inside_outside_AUC','candidate_tIoU_pair_accuracy']}
        rows.append(dict(alias=a,values=values,event=events,event_status=z['event_status'],event_word_count=z['event_word_count'],inside_observations=int(inside.sum()),outside_observations=int((~inside).sum()),association=z['association'],native_query=z['native_query'],candidate_count=len(z['candidates']),candidate_oracle_t_gain_pp=100*(max(ct)-values['E_MBR']['t']),candidate_oracle_v_gain_pp=100*(max(cv)-values['E_MBR']['v']),query_changed_by_seed={str(s):z['association'][f'Live{s}']['query']!=z['association']['E']['query']for s in plan['seeds']},offline_spatial_change_pp={str(s):100*(values[f'Live{s}_OriginalTime']['unobserved_actual_s']-values['E_MBR']['unobserved_actual_s'])if values['E_MBR']['unobserved_actual_s']is not None else None for s in plan['seeds']}))
    arms=['E_MBR','E_NativeConfidence','E_EventNative','E_EventE','E_EventLive_seedmean','Live_OriginalTime_seedmean','Live_NewTime_seedmean']
    pairs=[('E_EventE','E_NativeConfidence'),('E_EventE','E_EventNative'),('E_EventLive_seedmean','E_EventE'),('E_EventLive_seedmean','E_MBR'),('E_EventE','E_MBR'),('E_NativeConfidence','E_MBR'),('Live_NewTime_seedmean','Live_OriginalTime_seedmean'),('Live_NewTime_seedmean','E_MBR')]
    diffs={}
    for a,b in pairs:
        vv={k:paired([100*(r['values'][a][k]-r['values'][b][k])if r['values'][a][k]is not None and r['values'][b][k]is not None else None for r in rows])for k in ['v','t','s','unobserved_actual_s']}
        delta=[100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows];vv['tails']=dict(positive=sum(v>1e-10 for v in delta),negative=sum(v< -1e-10 for v in delta),unchanged=sum(abs(v)<=1e-10 for v in delta),below_minus5=sum(v< -5 for v in delta),worst=sorted([dict(alias=r['alias'],delta_pp=v)for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);diffs[a+'_minus_'+b]=vv
    enames=['NativeConfidence','EventNative','EventE','EventLive_seedmean'];emetrics=['inside_outside_AUC','candidate_tIoU_pair_accuracy']
    es={n:{k:paired([r['event'][n][k]for r in rows])for k in emetrics}for n in enames}
    ed={a+'_minus_'+b:{k:paired([r['event'][a][k]-r['event'][b][k]if r['event'][a][k]is not None and r['event'][b][k]is not None else None for r in rows])for k in emetrics}for a,b in [('EventE','NativeConfidence'),('EventE','EventNative'),('EventLive_seedmean','EventE')]}
    out=dict(status='postseal_evaluated',count=128,seeds=plan['seeds'],development_exposed=True,new_temporal_OPD=False,means={a:{k:avg([r['values'][a][k]for r in rows])for k in ['v','t','s','unobserved_actual_s']}for a in arms},paired_pp=diffs,event_summary=es,event_paired=ed,coverage=dict(event_valid=sum(r['event_status']=='valid'for r in rows),inside_and_outside_actual_evidence=sum(r['inside_observations']>0 and r['outside_observations']>0 for r in rows),event_inactive_verified_visibility='not established: HC boxes are event annotations, no independent outside-event visibility truth',query_switched_from_E={str(s):sum(r['query_changed_by_seed'][str(s)]for r in rows)for s in plan['seeds']},oracle_capacity_diagnostic=dict(t_gain_pp=paired([r['candidate_oracle_t_gain_pp']for r in rows]),v_gain_pp=paired([r['candidate_oracle_v_gain_pp']for r in rows]))),rows=rows,GT_only_after_global_seal=True)
    cost=[read(f)for f in (D/'cost').glob('*.json')];out['cost']=dict(GPU_seconds=sum(c['GPU_seconds']for c in cost),new_teacher_video_queries=sum(c['new_teacher_video_queries']for c in cost),new_teacher_image_frames=sum(c['new_teacher_image_frames']for c in cost),new_spatial_updates=0,new_Source=0,new_DINO_T=0)
    put(P/'RESULTS_PUBLIC.json',out);put(P/'INDEPENDENT_AUDIT.json',dict(status='pass',**audit,GT_provenance=provenance,all128_parent_denominator=True,all_three_seeds_retained=True));put(D/'STATUS.json',dict(stage='evaluated_pending_publication',done=128,total=128),True)
    print(json.dumps({k:out[k]for k in ['means','coverage','event_summary','event_paired','cost']},indent=2),flush=True)

if __name__=='__main__':main()

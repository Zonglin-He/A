"""Independent CPU cohort, state/SGD/rank/dense audit and public scalar audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,time,collections,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *
from scripts.tastvg_cpu_handoff_v1 import verify_cpu
from scripts.audit_tastvg_negative_evidence_v1 import Checker,shstate,metric,geometry,ranks,summary,corners,iou

ARMS=('F','T','S','A','U')
CONTRASTS=(('A','F'),('A','T'),('S','F'),('T','F'),('A','S'))

def select(rows,name):
    if name in ['all','expert','nonexpert']:
        return [r for r in rows if name=='all' or r['expert_scheduled']==(name=='expert')]
    if name in ['small_object','larger_object']:
        return [r for r in rows if r['small_object']==(name=='small_object')]
    return [r for r in rows if r['short_event']==(name=='short_event')]

def audit_tails(rows,a,b):
    if not rows:return dict(cells=0)
    d=np.array([r[a+'_v']-r[b+'_v'] for r in rows]);z=dict(cells=len(rows),
        improved=int((d>1e-12).sum()),degraded=int((d< -1e-12).sum()),unchanged=int((abs(d)<=1e-12).sum()),
        gross_gain_pp=float(d.clip(min=0).mean()*100),gross_loss_pp=float((-d).clip(min=0).mean()*100),
        severe_harm_gt5pp=int((d<-.05).sum()),severe_harm_gt20pp=int((d<-.2).sum()))
    for t in [.3,.5]:
        before=np.array([r[b+'_v']>t for r in rows]);after=np.array([r[a+'_v']>t for r in rows]);n=int(before.sum())
        z[str(t)]=dict(correct_before=n,correct_to_wrong=int((before&~after).sum()),
            wrong_to_correct=int((~before&after).sum()),conditional_correct_damage=float((before&~after).sum()/n) if n else None)
    return z

def public(directory=PUB):
    directory=Path(directory);ck=Checker();tick=time.monotonic()
    for direction in DIRECTIONS:
        d=directory/direction;cfg=read(d/'CONFIG.json');rows=read(d/'ROWS.json');steps=read(d/'STEP_ROWS.json')
        s=read(d/'SUMMARY.json');diag=read(d/'PIPELINE_DIAGNOSIS.json');bar=read(d/'PREDICTION_BARRIER.json')
        assert bar['GT_read'] is False and bar['time']<bar['score_start_time']
        assert cfg['params']==BUNDLES[cfg['source_dataset']] and cfg['parameters']==1792
        assert cfg['condition']=='clean' and cfg['availability']==25 and cfg['predict_before_write']
        assert cfg['historically_exposed'] and not cfg['target_hyperparameter_tuning'] and not cfg['production_promoted']
        assert not cfg['GT_used_for_selection'] and not cfg['GT_used_for_updates']
        assert len(rows)==cfg['arrivals']==3*cfg['queries'] and cfg['queries']==cfg['sources']
        assert len({(r['order'],r['arrival']) for r in rows})==len(rows)
        assert sum(r['expert_scheduled'] for r in rows)==cfg['scheduled']
        for r in rows:
            assert r['expert_scheduled']==(r['arrival']%4==0)
            for a,b in CONTRASTS:
                for k in ('v','t','s','at03','at05'):
                    ck.close(r[a+'_minus_'+b+'_'+k],r[a+'_'+k]-r[b+'_'+k],'differences')
            for a in ARMS:
                ck.close([r[a+'_at03'],r[a+'_at05']],[float(r[a+'_v']>.3),float(r[a+'_v']>.5)],'thresholds')
            ck.close(r['A_minus_F_v'],r['delta_boxes']+r['delta_native_interval']+r['delta_fast'],'path_sum')
            ck.close(r['headroom_GT_time'],r['GT_time_v']-r['S_v'],'ideal_time')
            ck.close(r['headroom_GT_space'],r['S_t']-r['S_v'],'ideal_space')
            ck.close(r['headroom_A_GT_time'],r['GT_time_v']-r['A_v'],'final_ideal_time')
            ck.close(r['headroom_A_GT_space'],r['A_t']-r['A_v'],'final_ideal_space')
            if not r['expert_scheduled']:
                for k in ('v','t','s'):ck.close([r['T_'+k],r['A_'+k]],[r['F_'+k],r['S_'+k]],'nonexpert_controls')
                assert r['pre_state_sha256']==r['post_state_sha256']
            else:
                ck.close(r['temporal_oracle_v'],max(r['temporal_candidate_v']),'temporal_oracle')
                ck.close(r['temporal_regret'],r['temporal_oracle_v']-r['A_v'],'regret')
                ck.close(r['temporal_candidate_v'][r['temporal_selected']],r['A_v'],'selected_v')
                assert r['temporal_selected']==int(np.argmax(r['temporal_scores']))
        for category in ['panels','characteristic_slices']:
            for name,z in s[category].items():
                rr=select(rows,name);fields=list(z['metrics']);calc=summary(rr,fields)
                for k,v in calc.items():ck.tree(v,z[k],'source_bootstrap')
                ck.tree({a+'_minus_'+b:audit_tails(rr,a,b) for a,b in CONTRASTS},z['tails'],'tails')
                for order,ov in z['orders'].items():
                    ck.tree(summary([r for r in rr if r['order']==order],fields),ov,'order_bootstrap')
                for field in fields:
                    vals=[ov['metrics'][field]['mean'] for ov in z['orders'].values() if ov['sources']]
                    sd=float(np.std(vals,ddof=1)) if len(vals)>1 else None
                    ck.tree(sd,z['order_sample_SD'][field],'order_SD')
        for name,z in s['recency'].items():
            ck.tree(summary([r for r in rows if r['recency']==name],['A_minus_T_v','S_minus_F_v']),z,'recency')
        # Independently reconstruct the paired ratio draws, retaining invalid denominators.
        groups=collections.defaultdict(list)
        for r in rows:groups[r['source_id']].append([r['A_v']-r['F_v'],r['U_v']-r['F_v']])
        x=np.array([np.mean(groups[k],0) for k in sorted(groups)]);mu=x.mean(0);rng=np.random.default_rng(20261004)
        bs=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
        eligible=bs[:,1]>1e-8;rat=bs[eligible,0]/bs[eligible,1];z=s['gap_recovery']
        ck.close([z['numerator'],z['denominator']],mu,'gap_ratio')
        ck.close(z['denominator_ci95'],np.percentile(bs[:,1],[2.5,97.5]),'gap_denominator_ci')
        ck.tree(float(mu[0]/mu[1]) if mu[1]>1e-8 else None,z['ratio'],'gap_ratio')
        ck.tree(np.percentile(rat,[2.5,97.5]).tolist() if len(rat) else None,
            z['ratio_ci95_conditional_on_positive_denominator'],'gap_ratio_ci')
        assert int(eligible.sum())==z['positive_denominator_bootstrap_draws']
        assert z['stable_positive_denominator']==bool(np.percentile(bs[:,1],2.5)>1e-8)
        for key,z in diag['path_transitions'].items():
            a,b=key.split('_minus_');ck.tree(audit_tails(rows,a,b),z,'path_tails')
        for field,z in diag['paired_source_effects'].items():ck.tree(summary(rows,[field]),z,'diagnostic_bootstrap')
        for at,z in diag['spatial'].items():
            ss=[r for r in steps if r['step']==int(at)]
            expected=dict(steps=len(ss),empty_evidence=sum(r['valid_frames']==0 for r in ss),
                no_event_reference=sum(r['valid_frames']>0 and r['valid_event_frames']==0 for r in ss),
                flat_rewards=sum(r['flat_rewards'] for r in ss),
                loss_down_GT_time_harm=sum(r['loss_decreased'] and r['post_GT_v']<r['pre_GT_v']-1e-12 for r in ss),
                loss_down_A_time_harm=sum(r['loss_decreased'] and r['post_A_v']<r['pre_A_v']-1e-12 for r in ss),
                severe_GT_time_harm=sum(r['post_GT_v']<r['pre_GT_v']-.05 for r in ss),
                good_top_but_harm=sum(r['top_best_GT_v']>r['pre_GT_v']+1e-12 and r['post_GT_v']<r['pre_GT_v']-1e-12 for r in ss),
                unique_good_top_but_harm=sum(len(r['top_indices'])==1 and r['top_best_GT_v']>r['pre_GT_v']+1e-12 and r['post_GT_v']<r['pre_GT_v']-1e-12 for r in ss),
                mean_post_minus_pre_GT_v=float(np.mean([r['post_GT_v']-r['pre_GT_v'] for r in ss])))
            expected['correct_support']={str(t):dict(present=sum(r['oracle_GT_v']>t for r in ss),
                absent=sum(r['oracle_GT_v']<=t for r in ss),
                top_group_misses=sum(r['oracle_GT_v']>t and r['top_best_GT_v']<=t for r in ss),
                correct_update_destroyed=sum(r['pre_GT_v']>t and r['post_GT_v']<=t for r in ss)) for t in [.3,.5]}
            for field in ['observed_GT_sIoU_delta','unobserved_GT_sIoU_delta']:
                expected[field]=summary([r for r in ss if r[field] is not None],[field])
            ck.tree(expected,z,'step_counts')
        ex=[r for r in rows if r['expert_scheduled']]
        for threshold,z in diag['temporal'].items():
            t=float(threshold)
            ck.tree(dict(scheduled=len(ex),good_candidate_present=sum(r['temporal_oracle_v']>t for r in ex),
                good_candidate_missed=sum(r['temporal_oracle_v']>t and r['A_v']<=t for r in ex),
                no_good_candidate=sum(r['temporal_oracle_v']<=t for r in ex),
                native_correct_destroyed=sum(r['S_v']>t and r['A_v']<=t for r in ex),
                native_wrong_rescued=sum(r['S_v']<=t and r['A_v']>t for r in ex)),z,'temporal_correct_counts')
        for threshold,z in diag['temporal_tIoU'].items():
            t=float(threshold)
            ck.tree(dict(scheduled=len(ex),good_candidate_present=sum(r['temporal_oracle_t']>t for r in ex),
                good_candidate_missed=sum(r['temporal_oracle_t']>t and r['A_t']<=t for r in ex),
                no_good_candidate=sum(r['temporal_oracle_t']<=t for r in ex),
                native_correct_destroyed=sum(r['S_t']>t and r['A_t']<=t for r in ex),
                native_wrong_rescued=sum(r['S_t']<=t and r['A_t']>t for r in ex)),z,'temporal_only_correct_counts')
        ck.tree(dict(scheduled=len(ex),empty=sum(r['valid_expert_frames']==0 for r in ex),
            nonempty_without_event=sum(r['valid_expert_frames']>0 and r['valid_event_frames']==0 for r in ex),
            measures={f:summary([r for r in ex if r[f] is not None],[f]) for f in
                ['expert_event_frame_precision','expert_event_GT_IoU']}),diag['evidence'],'evidence_source_aggregation')
        ck.tree(summary(ex,['net_update_A_v','net_update_GT_v']),diag['arrival_update'],'net_arrival_update')
    if (directory/'DECISION.json').exists():
        decision=read(directory/'DECISION.json')
        valid=all(read(directory/d/'SUMMARY.json')['panels']['all']['metrics']['A_minus_F_v']['ci95'][0]>0 for d in DIRECTIONS)
        persistence=any(read(directory/d/'SUMMARY.json')['panels']['all']['metrics']['A_minus_T_v']['ci95'][0]>0 for d in DIRECTIONS)
        assert decision['GO']==(valid and persistence) and not decision['production_promoted']
    return dict(status='pass',checks=dict(ck.count),scalar_checks=sum(ck.count.values()),
        max_errors=dict(ck.errors),CPU_wall_seconds=time.monotonic()-tick,time=time.time())

def root():
    import torch
    verify_cpu(BASE)
    torch.set_num_threads(2);bar=seal();ck=Checker();tick=time.monotonic()
    from scripts.score_tastvg_cross_domain_v1 import labels
    for direction in DIRECTIONS:
        p=verify(direction);out=BASE/direction;target=p['target_dataset'];cfg=p['params']
        assert p['source_dataset']!=target
        assert sha(ROOT/p['source_checkpoint'])==p['source_checkpoint_sha256']
        assert sha(ROOT/p['target_checkpoint'])==p['target_checkpoint_sha256']
        parent=read(p['parent_lock']);assert sha(p['parent_lock'])==p['parent_lock_sha256']
        queries=parent['groups']['vid_to_hc' if target=='hc2' else 'hc_to_vid']['queries']
        grouped=collections.defaultdict(list)
        for q in queries:grouped[q['source']].append(q)
        qhash=lambda q:hashlib.sha256((str(q['index'])+'|'+q['caption']+'|'+str(q['original_video_id'])).encode()).hexdigest()
        picked=[min(grouped[k],key=qhash) for k in sorted(grouped)]
        assert len(picked)==p['sources']==len({r['input']['video_sha256'] for r in p['rows']})
        for q,r in zip(picked,p['rows']):
            assert q['index']==r['parent_index'] and qhash(q)==r['query_hash']
            assert q['caption'].lower()==r['input']['caption'] and q['frame_ids']==r['frame_ids']
            assert q['video_sha256']==r['input']['video_sha256'];ck.count['cohort_query_bindings']+=1
        for k,seed in enumerate(p['order_seeds']):
            expected=sorted(range(p['sources']),key=lambda i:hashlib.sha256((str(seed)+'|'+p['rows'][i]['source']).encode()).hexdigest())
            assert expected==p['orders'][f'order{k}'];ck.count['cohort_orders']+=1
        assert read(out/'GT_EXPOSURE.json')['time']>bar['time']
        source=checked(out/'PARAMETER_SUPPORT.pt');assert sum(v.numel() for v in source['center'].values())==1792
        ck.close(source['basis']@source['basis'].T,np.eye(4),'basis_orthonormal',1e-12)
        assert source['spec']['candidates']==9 and source['spec']['rho']==.05
        assert shstate(source['center'])==read(out/'SUPPORT.json')['center_sha256']
        gt,spans=labels(direction);rows={(r['order'],r['arrival']):r for r in read(PUB/direction/'ROWS.json')}
        sr={(r['order'],r['arrival'],r['step']):r for r in read(PUB/direction/'STEP_ROWS.json')}
        par=read(out/'ARM_TRAJECTORY_PARITY.json');assert par['status']=='pass' and len(par['controls'])==2
        def equal(a,b,kind):
            assert set(a)==set(b)
            for n,v in a.items():ck.close(v,b[n],kind,0)
        for order,seq in p['orders'].items():
            state=source['center'];lastwrite=None
            for at,parent in enumerate(seq):
                f=out/'online'/order/f'{at:05}.pt.gz';assert sha(f)==read(f.with_suffix('.json'))['sha256'];x=loadz(f)
                equal(x['pre_state'],state,'persistent_pre');assert x['pre_sha']==shstate(state)
                row=p['rows'][parent];g=gt[parent];span=spans[parent];z=rows[order,at]
                assert x['nearest_write_distance']==(None if lastwrite is None else at-lastwrite)
                assert z['nearest_write_distance']==x['nearest_write_distance'];assert x['GT_read'] is False
                cp=checked(out/'capture'/f'{parent:05}.pt');rp=checked(out/'target_reference'/f'{parent:05}.pt')
                assert cp['checkpoint_state_sha256']==MODEL_SHA[p['source_dataset']]
                assert rp['model_state_sha256']==MODEL_SHA[target] and rp['pixel_sha256']==cp['pixel_sha256']==x['pixel_sha256']
                for a,pred in dict(F=x['source_native'],T=x['fast_only'],S=x['slow'],A=x['final'],U=rp['prediction']).items():
                    values=metric(pred,row,g,span,target)
                    for k,v in values.items():ck.close(v,z[a+'_'+k],'independent_dense',1e-10)
                ck.close(x['final']['boxes'],x['slow']['boxes'],'output_sealed_before_write',0)
                trace=x['update_steps'];assert bool(trace)==(at%4==0) and len(trace)<=cfg['steps']
                if trace:
                    ef=read(out/'experts/spatial/clean'/f'{parent:05}.json')
                    assert sha(out/'experts'/ef['cache'])==ef['cache_sha256']
                    se=load(out/'experts'/ef['cache']);valid=np.asarray(se['valid'],bool)
                for j,st in enumerate(trace):
                    equal(state,st['pre_state'],'inner_pre');assert shstate(state)==st['pre_state_sha256']
                    cs=torch.stack([c['prediction']['boxes'] for c in st['candidates']]);assert len(cs)==9
                    ck.close(cs[0],st['prediction']['boxes'],'refreshed_central',0)
                    u=st['update'];rew=st['rewards']
                    if valid.any():
                        ev=corners(np.asarray(se['boxes'])[valid])
                        expected=[float(iou(corners(c.numpy()[valid]),ev).mean()) for c in cs]
                        ck.close(expected,rew,'expert_reward',1e-12)
                    else:assert rew is None
                    if u is not None:
                        assert u['lr']==cfg['lr'] and u['teacher_temperature']==cfg['teacher_temperature']
                        assert u['student_temperature']==1. and u['coefficients']==([5.,3.] if p['source_dataset']=='vidstg' else [5.,4.])
                        d=geometry(st['prediction']['boxes'],cs,u['coefficients']);lp=(-d).log_softmax(0)
                        lq=(-torch.as_tensor(ranks(np.array(rew)),dtype=d.dtype)/cfg['teacher_temperature']).log_softmax(0)
                        ck.close(d,u['distances'],'geometry',5e-6,4e-6)
                        ck.close(lp.exp(),u['p'],'student_distribution',3e-6)
                        ck.close(lq.exp(),u['q'],'rank_teacher',3e-6)
                        ck.close(float((lp.exp()*(lp-lq)).sum()),u['loss_before'],'RKL',1e-5,5e-6)
                        da=geometry(st['post_prediction']['boxes'],cs,u['coefficients']);la=(-da).log_softmax(0)
                        ck.close(float((la.exp()*(la-lq)).sum()),u['loss_after'],'post_RKL',1e-5,5e-6)
                        for n,v in state.items():
                            ck.close(v.clone().add_(u['gradients'][n],alpha=-u['update_scale']),st['post_state'][n],'SGD',3e-7,3e-6)
                        assert u['update_scale'] in [0.,cfg['lr']]
                    else:
                        assert rew is None;equal(state,st['post_state'],'empty_noop')
                    strow=sr[order,at,j]
                    for name,pred in [('pre',st['prediction']),('post',st['post_prediction'])]:
                        for suffix,interval in [('A',x['final']['physical_interval']),('GT',span)]:
                            ck.close(metric(pred,row,g,span,target,interval)['v'],strow[name+'_'+suffix+'_v'],'step_dense',1e-10)
                    for k,c in enumerate(st['candidates']):
                        ck.close(metric(c['prediction'],row,g,span,target,span)['v'],strow['candidate_GT_v'][k],'candidate_dense',1e-10)
                    frames=np.array(sorted(g));ids=np.array(row['frame_ids']);truth=np.array([g[int(f)] for f in frames])
                    q=[]
                    for pred in [st['prediction'],st['post_prediction']]:
                        boxes=corners(pred['boxes'])*np.array([row['input']['width'],row['input']['height']]*2)
                        if target=='hc2':boxes=np.maximum(boxes,0)
                        dense=np.stack([np.interp(frames,ids,boxes[:,k]) for k in range(4)],-1)
                        v=iou(dense,truth);v[(frames<ids[0])|(frames>ids[-1])]=0;q.append(v)
                    event=(frames>=span[0])&(frames<span[1]);observed=np.isin(frames,ids[np.flatnonzero(valid)])
                    for field,mask in [('observed_GT_sIoU_delta',event&observed),('unobserved_GT_sIoU_delta',event&~observed)]:
                        ck.tree(float((q[1]-q[0])[mask].mean()) if mask.any() else None,strow[field],'observed_unobserved_dense')
                    state=st['post_state'];assert shstate(state)==st['post_state_sha256']
                equal(state,x['post_state'],'persistent_post');assert shstate(state)==x['post_sha']
                if x['updated']:lastwrite=at
                if x['expert_scheduled']:
                    for td,pre,selected in [(x['temporal'],x['slow'],x['final']),
                                           (cp['zero_temporal'],cp['prediction'],x['fast_only'])]:
                        ef=read(out/'experts/temporal/clean'/f'{parent:05}.json');e=load(out/'experts'/ef['cache'])
                        scores=[]
                        for c in td['candidates']:
                            a,b=c['physical_interval'];v=[]
                            for (g0,h0),w in zip(e['proposals'],e['proposal_confidence']):
                                inter=max(0,min(b,h0)-max(a,g0));v.append(w*inter/max(b-a+h0-g0-inter,1e-12))
                            scores.append(max(v,default=0.))
                        ck.close(scores,td['scores'],'Fast_critic',1e-12)
                        assert td['selected']==int(np.argmax(scores))
                        assert selected['indices']==td['candidates'][td['selected']]['indices']
                if x['reinsertion']:assert x['reinsertion']['full_pipeline_exact']
            assert len(seq)==len(set(seq))==p['queries']
    pub=public();ck.count['public_scalar_checks']+=pub['scalar_checks']
    return dict(status='pass',checks=dict(ck.count),scalar_checks=sum(ck.count.values()),max_errors=dict(ck.errors),
        CPU_wall_seconds=time.monotonic()-tick,time=time.time(),GPU_initialized=torch.cuda.is_initialized(),
        limitation='Checks loss arithmetic, saved gradients/SGD, bitwise live arm controls and state chain; does not recompute full model Jacobian on CPU.')

if __name__=='__main__':
    mode=sys.argv[1];r=root() if mode=='root' else public(Path(sys.argv[2]) if len(sys.argv)>2 else PUB)
    if not (mode=='public' and len(sys.argv)>2):
        write(BASE/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json'),r)
        write(PUB/('ROOT_AUDIT.json' if mode=='root' else 'PUBLIC_AUDIT.json'),r)
    print(r,flush=True)

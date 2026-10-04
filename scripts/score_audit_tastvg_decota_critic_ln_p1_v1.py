"""Independent CPU audit of actual LN chains, Adam and dense paired metrics."""
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_ln_common_v1 import *
from scripts.score_audit_tastvg_decota_critic_p0_v1 import energy_numpy
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import check_state,flat
import numpy as np
import torch
READOUTS=['frozen','episodic','before','online']
DIFFS={'inherited':('before','frozen'),'online_vs_episodic':('online','episodic'),
       'online_vs_frozen':('online','frozen'),'current_correction':('online','before')}
FIELDS=[f'{a}_{k}' for a in READOUTS for k in ['v','t','s']]
FIELDS += [f'{a}_{k}' for a in DIFFS for k in ['v','t','s']]
FIELDS += [f'{a}_gross_{s}_v' for a in DIFFS for s in ['gain','loss']]


def source_summary(rows):
    grouped=collections.defaultdict(list)
    for r in rows:grouped[r['source_id']].append([r[k] for k in FIELDS])
    ids=sorted(grouped);assert ids
    x=np.array([np.mean(grouped[i],0) for i in ids]);rng=np.random.default_rng(20261004)
    boot=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0);tails={}
    for name,(a,b) in DIFFS.items():
        g=np.maximum(x[:,FIELDS.index(name+'_v')],0)
        tails[name]=dict(harm_gt5pp=sum(r[name+'_v']<-.05 for r in rows),
            harm_gt20pp=sum(r[name+'_v']<-.20 for r in rows),gain_gt5pp=sum(r[name+'_v']>.05 for r in rows),
            positive_sources=int((g>0).sum()),largest_source_share_positive_gain=float(g.max()/g.sum()) if g.sum()>0 else None,
            correctness={str(t):dict(baseline_correct=sum(r[b+'_v']>=t for r in rows),
                arm_correct=sum(r[a+'_v']>=t for r in rows),
                destroyed=sum(r[b+'_v']>=t and r[a+'_v']<t for r in rows),
                recovered=sum(r[b+'_v']<t and r[a+'_v']>=t for r in rows)) for t in [.3,.5]})
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
        metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,x[:,j])}) for j,k in enumerate(FIELDS)},tails=tails)


def make_summary(rows,conditions):
    out={}
    for ds in DATASETS:
        out[ds]={}
        for split in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==split];corrupt=[r for r in rr if r['condition']!='clean']
            out[ds][split]=dict(corruption=source_summary(corrupt),clean=source_summary([r for r in rr if r['condition']=='clean']),
                conditions={c:source_summary([r for r in rr if r['condition']==c]) for c in conditions[ds]},
                orders={o:source_summary([r for r in corrupt if r['order']==o]) for o in ['order1','order2']},
                first=source_summary([r for r in corrupt if r['arrival']==0]),
                later=source_summary([r for r in corrupt if r['arrival']>0]))
    return out


def run():
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    from vg_tta.spatial_online_state_v1 import QUERY
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts import tastvg_decota_c1_common_v1 as c1
    torch.set_num_threads(4);barrier=verify_seal();tick=time.time();checks=0;rows=[];diag=[];cost={};conditions={};source={}
    write(BASE/'GT_EXPOSURE.json',dict(time=tick,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        all_predictions_precede_GT=barrier['time']<tick,use='offline scoring only',GT_optimization=False))
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');conditions[ds]=p['conditions']
        for split,sp in p['splits'].items():
            labels=read(POOL/ds/f'GT_LABELS_{split}.json')
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    prev=None;prevhash=None
                    for at,parent in enumerate(seq):
                        path=BASE/ds/'online'/split/cond/order/f'{at:05}.pt';x=checked(path);row=p['rows'][parent]
                        assert (x['dataset'],x['split'],x['condition'],x['order'],x['arrival'],x['parent'])==(ds,split,cond,order,at,parent)
                        assert x['query_reset'] and x['Adam_reset'] and x['LN_writeback_fraction']==1/16 and x['temporal_updates']==0 and not x['GT_read']
                        assert x['previous_payload_sha256']==prevhash
                        if ds not in source:source[ds]=x['source_state']
                        checks+=check_state(source[ds],x['source_state']);expected={n:v.clone() for n,v in (prev if prev is not None else source[ds]).items()}
                        expected[QUERY]=source[ds][QUERY].clone();assert torch.count_nonzero(expected[QUERY])==0
                        checks+=check_state(expected,x['initial'])
                        assert sha(ROOT/x['evidence_path'])==x['evidence_sha256'] and sha(ROOT/x['P0_path'])==x['P0_sha256']
                        ex=c1.checked(ROOT/x['evidence_path'])['expert'];prior=p0.checked(ROOT/x['P0_path'])
                        checks+=check_state(prior['source_state'],x['source_state'])
                        assert torch.equal(prior['native']['boxes'],x['native']['boxes'])
                        z=x['fit'];names=list(z['initial']);checks+=check_state(z['initial'],x['initial'])
                        assert torch.equal(z['path'][0]['boxes'],x['before']) and not z['GT_used'] and z['parameter_count']==1792
                        nonempty=bool(z['frame_metadata']);assert len(z['path'])==(11 if nonempty else 1)
                        assert z['gradient_calls']==(10 if nonempty else 0) and z['skipped']==(not nonempty)
                        selected=min(range(len(z['path'])),key=lambda j:z['path'][j]['loss'])
                        assert z['selected_step']==selected and torch.equal(z['final'],z['path'][selected]['boxes'])
                        checks+=check_state(z['state'],z['path'][selected]['state'])
                        if at==0:
                            assert torch.equal(x['before'],x['native']['boxes'])
                            zz=prior['fits']['critic'];assert zz['selected_step']==selected
                            for a,b in zip(z['path'],zz['path']):
                                assert a['loss']==b['loss'] and torch.equal(a['boxes'],b['boxes'])
                                checks+=check_state(a['state'],b['state'])
                                if 'update' in a:assert torch.equal(a['update']['gradient'],b['update']['gradient'])
                        committed={}
                        for n in expected:
                            committed[n]=torch.zeros_like(expected[n]) if n==QUERY else expected[n]+(z['state'][n]-expected[n])*(1/16)
                        checks+=check_state(committed,x['committed']);prev=x['committed'];prevhash=sha(path)
                        truth={int(k):v for k,v in labels[str(parent)]['truth'].items()};span=labels[str(parent)]['span'];iv=x['native']['physical_interval']
                        metrics={}
                        for name,boxes in [('frozen',x['native']['boxes']),('episodic',prior['fits']['critic']['final']),('before',x['before']),('online',z['final'])]:
                            fast=DenseTube(boxes.numpy(),row,truth,span,clip=ds=='hc2').score(iv)
                            ref=official(boxes.numpy(),row,truth,span,iv,ds)
                            assert max(abs(fast[k]-ref[k]) for k in fast)<2e-12
                            metrics.update({f'{name}_{k}':v for k,v in ref.items()});checks+=3
                        for name in READOUTS:assert metrics[name+'_t']==metrics['frozen_t']
                        m=np.zeros(1792);v=np.zeros(1792);err=0;pathv=[];gn=[];un=[]
                        for j,h in enumerate(z['path']):
                            actual=energy_numpy(h['boxes'].numpy(),ex);assert abs(actual-h['loss'])<2e-6;checks+=1
                            pathv.append(DenseTube(h['boxes'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)['v'])
                            for vv in h['state'].values():assert torch.isfinite(vv).all();checks+=vv.numel()
                            if 'update' not in h:continue
                            u=h['update'];g=u['gradient'].numpy().astype(np.float64);assert g.shape==(1792,) and np.isfinite(g).all()
                            m=.9*m+.1*g;v=.999*v+.001*g*g
                            expect=-.03*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
                            error=float(np.max(np.abs(expect-u['raw'].numpy())));err=max(err,error);assert error<2e-6
                            before=flat(h['state'],names);after=flat(z['path'][j+1]['state'],names)
                            assert torch.equal(after-before,u['raw']) and torch.allclose(after,before+u['raw'],atol=1e-7,rtol=0)
                            assert all(s==j+1 for s in u['adam_steps'].values());checks+=1792*4
                            gn.append(float(np.linalg.norm(g)));un.append(float(u['raw'].norm()))
                        rec=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,**metrics,
                            payload_sha256=prevhash,initial_state_sha256=state_hash(x['initial']),committed_state_sha256=state_hash(x['committed']))
                        for name,(a,b) in DIFFS.items():
                            for k in ['v','t','s']:rec[f'{name}_{k}']=metrics[f'{a}_{k}']-metrics[f'{b}_{k}']
                            rec[name+'_gross_gain_v']=max(rec[name+'_v'],0);rec[name+'_gross_loss_v']=max(-rec[name+'_v'],0)
                        rows.append(rec)
                        diag.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,
                            initial_state_sha256=rec['initial_state_sha256'],committed_state_sha256=rec['committed_state_sha256'],
                            previous_payload_sha256=x['previous_payload_sha256'],payload_sha256=prevhash,
                            objective_losses=[h['loss'] for h in z['path']],posthoc_GT_vIoU_path=pathv,selected_step=selected,
                            selected_by_own_objective=True,GT_selected_step=False,query_reset=True,Adam_reset=True,LN_writeback_fraction=1/16,
                            gradient_norms=gn,update_norms=un,max_Adam_arithmetic_error=err,skipped=z['skipped'],
                            gradient_calls=z['gradient_calls'],fit_seconds=x['fit_seconds'],
                            inherited_LN_displacement=float((flat(x['initial'],names)-flat(source[ds],names)).norm()),
                            committed_LN_displacement=float((flat(x['committed'],names)-flat(source[ds],names)).norm()),
                            proxy_improved_GT_harmed=z['path'][selected]['loss']<z['path'][0]['loss'] and pathv[selected]<pathv[0]-1e-12))
                    print('P1_CPU_CHAIN',ds,split,cond,order,len(rows),'checks',checks,flush=True)
        b=read(BASE/ds/'PREDICTION_BARRIER.json');dd=[d for d in diag if d['dataset']==ds]
        assert b['backwards']==sum(d['gradient_calls'] for d in dd)
        cost[ds]=dict(arrivals=576,new_DINO_calls=0,new_full_backbone_forwards=0,temporal_updates=0,
            spatial_backward_calls=b['backwards'],optimizer_and_before_replay_evaluations=sum(len(d['objective_losses'])+1 for d in dd),
            cached_PosDecoder_offset_forward_calls=sum(8+2*len(d['objective_losses']) for d in dd),
            cached_PosDecoder_count_includes_native_two_pass_capture_and_parity=True,
            worker_seconds=b['seconds'],fit_seconds=sum(d['fit_seconds'] for d in dd),peak_memory_bytes=b['peak_memory_bytes'],
            worker_wall_not_GPU_kernel=True,episodic_control_reused=True)
    assert len(rows)==len(diag)==1152
    write(PUB/'ROWS.json',rows);write(PUB/'UPDATE_DIAGNOSTICS.json',diag)
    summary=make_summary(rows,conditions);write(PUB/'SUMMARY.json',summary)
    write(PUB/'COST.json',dict(datasets=cost,smoke=read(BASE/'SMOKE_ROOT_ACCEPTANCE.json')))
    write(PUB/'PROTOCOL_BINDING.json',dict(version='tastvg_decota_critic_ln_p1_v1',runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),P0_barrier_sha256=sha(p0.BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        parameters=1792,LN_writeback_fraction=1/16,query_reset=True,Adam_reset=True,lr=.03,steps=10,temperatures=[1,1],
        source_inputs=576,online_arrivals=1152,expert_every_query=True,observations_per_query=4,native_time_fixed=True,
        Before_is_not_25percent_nonexpert_stream=True,historical_exposure=True,GT_used_for_optimization=False,
        datasets={ds:dict(checkpoint=read(BASE/ds/'PLAN.json')['configuration']['checkpoint'],
            checkpoint_sha256=read(BASE/ds/'PLAN.json')['configuration']['checkpoint_sha256'],search_sources=32,confirm_sources=16,
            conditions=conditions[ds]) for ds in DATASETS}))
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,arrivals=1152,global_seal_before_GT=True,
        exact_LN_one_sixteenth_arithmetic=True,query_Adam_reset=True,all_predecessor_chains_verified=True,
        source_start_P0_all_steps_bitwise=True,independent_numpy_energy_Adam=True,official_dense_agreement=True,seconds=time.time()-tick))
    assessment={ds:{name:dict(mean=summary[ds]['confirm']['corruption']['metrics'][name+'_v']['mean'],
        positive_CI=summary[ds]['confirm']['corruption']['metrics'][name+'_v']['ci95'][0]>0) for name in ['inherited','online_vs_episodic','online_vs_frozen']} for ds in DATASETS}
    write(PUB/'DECISION.json',dict(scope='fixed matched small-panel LN1/16 research',datasets=assessment,
        method_promoted=False,new_scientific_variables_selected=False,future_formal_nonexpert_gain_established=False))
    result=public_check(PUB);write(PUB/'PUBLIC_AUDIT.json',result)
    print('P1_CPU_PASS',checks,result['checks'],flush=True)


def public_check(directory):
    p=Path(directory);rows=read(p/'ROWS.json');diag=read(p/'UPDATE_DIAGNOSTICS.json');s=read(p/'SUMMARY.json');checks=0
    assert len(rows)==len(diag)==1152
    assert len({(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']) for r in rows})==1152
    for r in rows:
        for a in READOUTS:
            assert all(0<=r[a+'_'+k]<=1 for k in ['v','t','s']) and r[a+'_t']==r['frozen_t'];checks+=4
        for name,(a,b) in DIFFS.items():
            for k in ['v','t','s']:assert abs(r[name+'_'+k]-(r[a+'_'+k]-r[b+'_'+k]))<1e-14;checks+=1
            assert r[name+'_gross_gain_v']==max(r[name+'_v'],0) and r[name+'_gross_loss_v']==max(-r[name+'_v'],0);checks+=2
        if r['arrival']==0:assert r['inherited_v']==0
    groups=collections.defaultdict(list)
    for d in diag:
        assert d['selected_step']==min(range(len(d['objective_losses'])),key=lambda j:d['objective_losses'][j])
        assert d['query_reset'] and d['Adam_reset'] and d['LN_writeback_fraction']==1/16 and not d['GT_selected_step'];checks+=5
        groups[(d['dataset'],d['split'],d['condition'],d['order'])].append(d)
    assert len(groups)==48
    for chain in groups.values():
        chain=sorted(chain,key=lambda d:d['arrival']);assert chain[0]['previous_payload_sha256'] is None
        for a,b in zip(chain,chain[1:]):
            assert b['previous_payload_sha256']==a['payload_sha256'] and b['initial_state_sha256']==a['committed_state_sha256'];checks+=2
    conditions={ds:list(s[ds]['search']['conditions']) for ds in DATASETS}
    assert make_summary(rows,conditions)==s;checks+=len(FIELDS)*3*12*4
    return dict(status='pass',checks=checks,arrivals=1152,actual_order_trajectories=True,
        independent_summary_regeneration=True,all_public_predecessor_chains=True,time=time.time())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['run','public']);p.add_argument('directory',nargs='?');a=p.parse_args()
    if a.stage=='run':run()
    else:print(public_check(a.directory or PUB))

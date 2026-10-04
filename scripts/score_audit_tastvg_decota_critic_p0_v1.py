"""Sealed CPU audit: independent losses/Adam, official dense scores, paired sources."""
import sys, time, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.tastvg_decota_critic_common_v1 import *
from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import loss_numpy, check_state, flat
import numpy as np
import torch

FIELDS = [f'{arm}_{k}' for arm in ['frozen', 'direct', 'critic'] for k in ['v','t','s']]
FIELDS += [f'{arm}_delta_{k}' for arm in ARMS for k in ['v','t','s']]
FIELDS += ['critic_minus_direct_v']
FIELDS += [f'{arm}_gross_{sign}_v' for arm in ARMS for sign in ['gain','loss']]


def energy_numpy(boxes, expert):
    terms = []
    for (_, pos), observation in sorted(expert['observations'].items()):
        q = observation['probe']
        b = np.asarray(q['boxes'], dtype=np.float64).reshape(-1,4)
        scores = np.asarray(q['target_scores'], dtype=np.float64)
        ok = (b[:,2:] > 0).all(1); b, scores = b[ok], scores[ok]
        if not len(scores): continue
        pred = np.asarray(boxes[pos], dtype=np.float64)
        p = np.r_[pred[:2]-.5*pred[2:], pred[:2]+.5*pred[2:]]
        e = np.c_[b[:,:2]-.5*b[:,2:], b[:,:2]+.5*b[:,2:]]
        inter = np.maximum(np.minimum(p[2:], e[:,2:])-np.maximum(p[:2],e[:,:2]),0).prod(-1)
        union = np.maximum(np.maximum(p[2:]-p[:2],0).prod() + np.maximum(e[:,2:]-e[:,:2],0).prod(-1)-inter,1e-7)
        w = np.exp(scores-scores.max()); w /= w.sum()
        x = np.log(w) + inter/union
        terms.append(-(x.max()+np.log(np.exp(x-x.max()).sum())))
    return float(np.mean(terms)) if terms else 0.


def source_summary(rows):
    grouped = collections.defaultdict(list)
    for r in rows: grouped[r['source_id']].append([r[k] for k in FIELDS])
    ids = sorted(grouped); assert ids
    x = np.array([np.mean(grouped[i],0) for i in ids])
    rng = np.random.default_rng(20261004)
    boot = np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci = np.quantile(boot,[.025,.975],axis=0)
    tails = {}
    for arm in ARMS:
        source_gains = np.maximum(x[:,FIELDS.index(f'{arm}_delta_v')],0)
        tails[arm] = dict(harm_gt5pp=sum(r[f'{arm}_delta_v']<-.05 for r in rows),
            harm_gt20pp=sum(r[f'{arm}_delta_v']<-.20 for r in rows),
            gain_gt5pp=sum(r[f'{arm}_delta_v']>.05 for r in rows),
            positive_sources=int((source_gains>0).sum()),
            largest_source_share_positive_gain=float(source_gains.max()/source_gains.sum()) if source_gains.sum()>0 else None,
            correctness={str(t):dict(frozen_correct=sum(r['frozen_v']>=t for r in rows),
                arm_correct=sum(r[f'{arm}_v']>=t for r in rows),
                destroyed=sum(r['frozen_v']>=t and r[f'{arm}_v']<t for r in rows),
                recovered=sum(r['frozen_v']<t and r[f'{arm}_v']>=t for r in rows)) for t in [.3,.5]})
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
        metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,x[:,j])}) for j,k in enumerate(FIELDS)}, tails=tails)


def make_summary(rows, conditions):
    out = {}
    for ds in DATASETS:
        out[ds] = {}
        for split in ['search','confirm']:
            rr = [r for r in rows if r['dataset']==ds and r['split']==split]
            out[ds][split] = dict(corruption=source_summary([r for r in rr if r['condition']!='clean']),
                clean=source_summary([r for r in rr if r['condition']=='clean']),
                conditions={c:source_summary([r for r in rr if r['condition']==c]) for c in conditions[ds]},
                orders={o:source_summary([r for r in rr if r['condition']!='clean' and r['order']==o]) for o in ['order1','order2']})
    return out


def run():
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube, official
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_online_state_v1 import QUERY
    from scripts import tastvg_decota_c1_common_v1 as c1
    torch.set_num_threads(4); barrier = verify_seal(); tick = time.time()
    assert not (BASE/'GT_EXPOSURE.json').exists(), 'Scoring cannot silently overwrite a previous audit'
    write(BASE/'GT_EXPOSURE.json',dict(time=tick, global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        global_prediction_seal_precedes_GT=barrier['time']<tick, use='offline evaluation only', configurations_selected_with_GT=False))
    checks = 0; unique_rows = {}; diagnostics = []; source_states = {}; costs = {}; conditions = {}
    for ds in DATASETS:
        plan = read(BASE/ds/'PLAN.json'); conditions[ds] = plan['conditions']
        labels = {split:read(POOL/ds/f'GT_LABELS_{split}.json') for split in plan['splits']}
        split_for = {parent:split for split,sp in plan['splits'].items() for parent in sp['orders']['order1']}
        assert len(split_for)==48
        for row in plan['rows']:
            parent = row['ordinal']; split = split_for[parent]
            truth = {int(k):v for k,v in labels[split][str(parent)]['truth'].items()}
            span = labels[split][str(parent)]['span']
            for cond in plan['conditions']:
                path = BASE/ds/'predictions'/cond/f'{parent:05}.pt'; x = checked(path)
                assert x['dataset']==ds and x['parent']==parent and x['condition']==cond
                assert x['episodic_reset'] and x['LN_writeback_fraction']==0 and x['temporal_updates']==0 and not x['GT_read']
                assert sha(ROOT/x['evidence_path'])==x['evidence_sha256']
                expert = c1.checked(ROOT/x['evidence_path'])['expert']
                assert expert['anchors']['single4']==x['anchors']
                if ds not in source_states: source_states[ds]=x['source_state']
                checks += check_state(source_states[ds],x['source_state'])
                assert torch.count_nonzero(x['source_state'][QUERY])==0
                iv = x['native']['physical_interval']; metrics = {}
                for name,boxes in [('frozen',x['native']['boxes'])]+[(a,x['fits'][a]['final']) for a in ARMS]:
                    fast = DenseTube(boxes.numpy(),row,truth,span,clip=ds=='hc2').score(iv)
                    reference = official(boxes.numpy(),row,truth,span,iv,ds)
                    assert max(abs(fast[k]-reference[k]) for k in fast)<2e-12,(ds,name,fast,reference)
                    metrics.update({f'{name}_{k}':val for k,val in reference.items()}); checks += 3
                for arm in ARMS:
                    z = x['fits'][arm]; names = list(z['initial'])
                    checks += check_state(z['initial'],x['source_state'])
                    assert torch.equal(z['path'][0]['boxes'],x['native']['boxes'])
                    assert z['parameter_count']==1792 and not z['GT_used']
                    nonempty = bool(x['anchors']) if arm=='direct' else bool(z['frame_metadata'])
                    assert len(z['path'])==(11 if nonempty else 1)
                    assert z['gradient_calls']==(10 if nonempty else 0) and z['skipped']==(not nonempty)
                    selected = min(range(len(z['path'])),key=lambda j:z['path'][j]['loss'])
                    assert selected==z['selected_step'] and torch.equal(z['final'],z['path'][selected]['boxes'])
                    checks += check_state(z['state'],z['path'][selected]['state'])
                    m=np.zeros(1792); v=np.zeros(1792); maxadam=0.; losses=[]; path_v=[]; gn=[]; displacements=[]
                    for j,h in enumerate(z['path']):
                        actual = loss_numpy(h['boxes'].numpy(),x['anchors']) if arm=='direct' else energy_numpy(h['boxes'].numpy(),expert)
                        assert abs(actual-h['loss'])<(2e-5 if arm=='direct' else 2e-6),(arm,actual,h['loss']); checks += 1
                        losses.append(h['loss']); path_v.append(DenseTube(h['boxes'].numpy(),row,truth,span,clip=ds=='hc2').score(iv)['v'])
                        for vv in h['state'].values(): assert torch.isfinite(vv).all(); checks += vv.numel()
                        if 'update' not in h: continue
                        u=h['update']; g=u['gradient'].numpy().astype(np.float64)
                        assert g.shape==(1792,) and np.isfinite(g).all()
                        m=.9*m+.1*g; v=.999*v+.001*g*g
                        expect=-.03*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
                        error=float(np.max(np.abs(expect-u['raw'].numpy()))); maxadam=max(maxadam,error)
                        assert error<2e-6,error
                        before=flat(h['state'],names); after=flat(z['path'][j+1]['state'],names)
                        assert torch.equal(after-before,u['raw'])
                        if arm=='direct':
                            assert torch.equal(after,before+u['direction'])
                            assert torch.equal(u['raw'],u['direction']) and torch.equal(u['applied'],u['raw'])
                        else:
                            assert torch.allclose(after,before+u['raw'],atol=1e-7,rtol=0)
                        assert all(s==j+1 for s in u['adam_steps'].values()); checks+=1792*4
                        gn.append(float(np.linalg.norm(g))); displacements.append(float(u['raw'].norm()))
                    assert metrics[f'{arm}_t']==metrics['frozen_t']
                    grad0=gn[0] if gn else 0.; frame0=z['path'][0].get('evidence_diagnostics',[])
                    diagnostics.append(dict(dataset=ds,split=split,condition=cond,source_id=parent,arm=arm,
                        objective_losses=losses,posthoc_GT_vIoU_path=path_v,selected_step=selected,
                        selected_by_own_objective=True,GT_selected_step=False,gradient_norms=gn,
                        update_norms=displacements,max_Adam_arithmetic_error=maxadam,
                        initial_gradient_zero=bool(grad0==0),selected_step_zero=selected==0,
                        initial_all_evidence_disjoint=bool(frame0 and all(q['all_zero_overlap'] for q in frame0)),
                        initial_disjoint_frames=sum(q['all_zero_overlap'] for q in frame0),
                        selected_disjoint_frames=sum(q['all_zero_overlap'] for q in z['path'][selected].get('evidence_diagnostics',[])),
                        frame_metadata=z.get('frame_metadata',[]),
                        observed_frames=len(expert['observations']),admitted_frames=len(x['anchors']),
                        retained_critic_frames=len(z.get('frame_metadata',[])),
                        objective_improved=losses[selected]<losses[0],
                        GT_harmed_despite_proxy_improved=losses[selected]<losses[0] and path_v[selected]<path_v[0]-1e-12,
                        final_parameter_displacement=float((flat(z['state'],names)-flat(z['initial'],names)).norm()),
                        source_state_sha256=state_hash(z['initial']),selected_state_sha256=state_hash(z['state']),
                        query_reset=True,Adam_reset=True,LN_writeback_fraction=0,temporal_updates=0,
                        fit_seconds=x['fit_seconds'][arm],skipped=z['skipped'],gradient_calls=z['gradient_calls']))
                rec=dict(dataset=ds,split=split,condition=cond,source_id=parent,**metrics,
                    critic_minus_direct_v=metrics['critic_v']-metrics['direct_v'],
                    receipt_sha256=sha(path.with_suffix('.json')),source_state_sha256=state_hash(x['source_state']))
                for arm in ARMS:
                    for k in ['v','t','s']: rec[f'{arm}_delta_{k}']=metrics[f'{arm}_{k}']-metrics[f'frozen_{k}']
                    rec[f'{arm}_gross_gain_v']=max(rec[f'{arm}_delta_v'],0)
                    rec[f'{arm}_gross_loss_v']=max(-rec[f'{arm}_delta_v'],0)
                unique_rows[(ds,parent,cond)]=rec
            print('CPU_AUDIT',ds,parent+1,48,'checks',checks,flush=True)
        b=read(BASE/ds/'PREDICTION_BARRIER.json')
        dd=[d for d in diagnostics if d['dataset']==ds]
        costs[ds]=dict(unique_inputs=288,logical_arrivals=576,arms=2,new_DINO_forwards=0,
            formal_full_backbone_forwards=0,spatial_backward_calls=b['backwards'],
            temporal_backward_calls=0,spatial_replay_evaluations=sum(len(d['objective_losses']) for d in dd),
            worker_seconds=b['seconds'],fit_seconds_by_arm={a:sum(d['fit_seconds'] for d in dd if d['arm']==a) for a in ARMS},
            peak_memory_bytes=b['peak_memory_bytes'],worker_wall_includes_IO=True,not_pure_GPU_kernel_time=True)
    assert len(unique_rows)==576 and len(diagnostics)==1152
    rows=[]
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json')
        for split,sp in p['splits'].items():
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    for at,parent in enumerate(seq):
                        rec=unique_rows[(ds,parent,cond)]; assert rec['split']==split
                        rows.append(dict(**rec,order=order,arrival=at))
    assert len(rows)==1152
    write(PUB/'ROWS.json',rows); write(PUB/'UPDATE_DIAGNOSTICS.json',diagnostics)
    summary=make_summary(rows,conditions);write(PUB/'SUMMARY.json',summary)
    write(PUB/'COST.json',dict(datasets=costs,smoke=read(BASE/'SMOKE_ROOT_ACCEPTANCE.json'),
        formal_unique_paired_inputs=576,logical_arrivals_per_arm=1152))
    write(PUB/'PROTOCOL_BINDING.json',dict(version='tastvg_decota_critic_p0_v1',
        datasets={ds:dict(search_sources=32,confirm_sources=16,unique_inputs=288,logical_arrivals=576,
            checkpoint=read(BASE/ds/'PLAN.json')['configuration']['checkpoint'],
            checkpoint_sha256=read(BASE/ds/'PLAN.json')['configuration']['checkpoint_sha256'],
            conditions=conditions[ds],orders=read(BASE/ds/'PLAN.json')['splits']) for ds in DATASETS},
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        historical_exposure=True,parameters=1792,lr=.03,Adam_steps=10,proposal_temperature=1,reward_temperature=1,
        LN_writeback_fraction=0,temporal_updates=0,current_interval='same source native I0 for both arms',
        direct_admission='unchanged original top-one accepted',critic_admission='all nonempty valid cached supports',
        support_statistics=read(BASE/'RUNTIME_LOCK.json')['supports'],source_reset_each_input=True,
        selection='earliest minimum own objective at steps0..10',GT_optimization=False,
        expert_schedule='four native-interval observations on every query; cached original evidence'))
    assessment={}
    for ds in DATASETS:
        m=summary[ds]['confirm']['corruption']['metrics']
        estimate=m['critic_delta_v']; difference=m['critic_minus_direct_v']
        assessment[ds]=dict(critic_positive_CI=estimate['ci95'][0]>0,
            critic_reversal=estimate['mean']<=0,beats_direct_CI=difference['ci95'][0]>0,
            scoped_conclusion='positive_current_panel' if estimate['ci95'][0]>0 else 'NO_GO_fixed_critic' if estimate['mean']<=0 else 'inconclusive')
    write(PUB/'DECISION.json',dict(scope='P0 episodic current-query correction only',
        datasets=assessment,method_promoted=False,automatic_P1_started=False,
        future_online_gain_established=False,objective_only_causal_attribution=False,
        proxies_are_DINO_coordinate_evidence=True,new_scientific_settings_selected=False))
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,unique_inputs=576,
        logical_arrivals=1152,unique_arm_paths=1152,global_seal_verified=True,GT_after_global_seal=True,
        independent_numpy_Direct_and_energy=True,independent_numpy_Adam=True,
        source_reset_every_input=True,unchanged_temporal_interval=True,
        official_dense_and_vectorized_metrics_identical=True,seconds=time.time()-tick,
        GT_used_for_optimization=False))
    result=public_check(PUB);write(PUB/'PUBLIC_AUDIT.json',result)
    status(BASE/'STATUS.json',dict(status='CPU_scored_audited_pending_visual_publication',unique_inputs=576,logical_arrivals=1152,time=time.time()))
    print('CPU_AUDIT_PASS',checks,result['checks'],flush=True)


def public_check(directory):
    p=Path(directory);rows=read(p/'ROWS.json');d=read(p/'UPDATE_DIAGNOSTICS.json');summary=read(p/'SUMMARY.json');checks=0
    assert len(rows)==len(d)==1152
    keys={(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']) for r in rows};assert len(keys)==1152
    seen={}
    for r in rows:
        assert all(0<=r[f'{a}_{k}']<=1 for a in ['frozen']+ARMS for k in ['v','t','s'])
        for a in ARMS:
            assert r[f'{a}_t']==r['frozen_t']
            for k in ['v','t','s']: assert abs(r[f'{a}_delta_{k}']-(r[f'{a}_{k}']-r[f'frozen_{k}']))<1e-14;checks+=1
        assert abs(r['critic_minus_direct_v']-(r['critic_v']-r['direct_v']))<1e-14
        key=(r['dataset'],r['source_id'],r['condition'])
        vector=[r[k] for k in FIELDS]
        if key in seen: assert vector==seen[key]
        seen[key]=vector;checks+=len(FIELDS)
    assert len(seen)==576
    for h in d:
        assert h['selected_step']==min(range(len(h['objective_losses'])),key=lambda j:h['objective_losses'][j])
        assert h['query_reset'] and h['Adam_reset'] and h['LN_writeback_fraction']==0 and h['temporal_updates']==0
        assert h['selected_by_own_objective'] and not h['GT_selected_step'];checks+=7
    conditions={ds:list(summary[ds]['search']['conditions']) for ds in DATASETS}
    assert make_summary(rows,conditions)==summary;checks+=sum(len(FIELDS)*3*10 for _ in DATASETS for split in ['search','confirm'])
    return dict(status='pass',checks=checks,logical_arrivals=1152,unique_inputs=576,
        exact_order_invariance=True,independent_summary_regeneration=True,time=time.time())


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','public']);parser.add_argument('directory',nargs='?')
    args=parser.parse_args()
    if args.stage=='run':run()
    else:print(public_check(args.directory or PUB))

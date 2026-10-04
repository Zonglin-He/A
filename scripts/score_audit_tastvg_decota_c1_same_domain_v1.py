"""CPU-only independent arithmetic, state recurrence and official dense scoring."""
import sys, time, math, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_c1_common_v1 import *
import numpy as np
import torch


def loss_numpy(boxes, anchors):
    if not anchors: return 0.
    p=np.asarray(boxes,dtype=np.float64)[[a['position'] for a in anchors]]
    t=np.array([a['box'] for a in anchors],dtype=np.float32).astype(np.float64)
    a=np.c_[p[:,:2]-.5*p[:,2:],p[:,:2]+.5*p[:,2:]]
    b=np.c_[t[:,:2]-.5*t[:,2:],t[:,:2]+.5*t[:,2:]]
    inter=np.maximum(np.minimum(a[:,2:],b[:,2:])-np.maximum(a[:,:2],b[:,:2]),0).prod(-1)
    union=np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)-inter,1e-7)
    enclosing=np.maximum(np.maximum(a[:,2:],b[:,2:])-np.minimum(a[:,:2],b[:,:2]),0).prod(-1).clip(1e-7)
    giou=inter/union-(enclosing-union)/enclosing
    return float((5*np.abs(p-t).sum(-1)+2*(1-giou)).sum()/4)


def logsum(x):
    m=x.max();return m+np.log(np.exp(x-m).sum())


def time_loss(logits, evidence, margin):
    nll=[];hinge=[]
    for z,e in zip(logits,evidence['offsets']):
        z=z.numpy().reshape(-1,2).astype(np.float64);ij=e['ij'].numpy();at=e['target']
        score=z[ij[0],0]+z[ij[1],1];lp=score-logsum(score)
        nll.append(-lp[at]);others=np.delete(lp,at)
        hinge.append(max(margin-(lp[at]-others.max()),0) if len(others) else 0)
    return float(np.mean(nll)+np.mean(hinge))


def check_state(a,b):
    assert set(a)==set(b)
    for n in a: assert torch.equal(a[n],b[n]),n
    return sum(v.numel() for v in a.values())


def flat(state, names):
    return torch.cat([state[n].flatten() for n in names])


def source_summary(rows):
    """Paired sources are the bootstrap units, never arrivals or observations."""
    fields=['frozen_v','frozen_t','frozen_s','decota_v','decota_t','decota_s',
        'delta_v','delta_t','delta_s','inherited_delta_v','spatial_delta_v',
        'temporal_delta_v','current_spatial_delta_v','gross_gain_v','gross_loss_v']
    grouped=collections.defaultdict(list)
    for r in rows: grouped[r['source_id']].append([r[k] for k in fields])
    ids=sorted(grouped)
    x=np.array([np.mean(grouped[i],0) for i in ids])
    rng=np.random.default_rng(20261004)
    boot=np.concatenate([x[rng.integers(len(x),size=(100,len(x)))].mean(1) for _ in range(100)])
    ci=np.quantile(boot,[.025,.975],axis=0)
    return dict(cells=len(rows),sources=len(ids),bootstrap_draws=10000,seed=20261004,
        metrics={k:dict(mean=float(x[:,j].mean()),ci95=ci[:,j].tolist(),
            source_values={str(i):float(v) for i,v in zip(ids,x[:,j])}) for j,k in enumerate(fields)},
        harm_gt5pp=sum(r['delta_v']<-.05 for r in rows),harm_gt20pp=sum(r['delta_v']<-.20 for r in rows),
        gain_gt5pp=sum(r['delta_v']>.05 for r in rows),
        correctness={str(t):dict(frozen_correct=sum(r['frozen_v']>=t for r in rows),
            decota_correct=sum(r['decota_v']>=t for r in rows),
            destroyed=sum(r['frozen_v']>=t and r['decota_v']<t for r in rows),
            recovered=sum(r['frozen_v']<t and r['decota_v']>=t for r in rows)) for t in [.3,.5]})


def run():
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube, official, physical
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.spatial_consolidation_v1 import consolidate
    from vg_tta.spatial_online_state_v1 import arrival, QUERY
    from methods.decota_final_simplified_v1.objectives import prediction
    torch.set_num_threads(4);b=verify_seal();tick=time.time();checks=0;rows=[];updates=[];cost={};timechecked=set()
    for ds in DATASETS:
        p=read(BASE/ds/'PLAN.json');cfg=p['configuration'];cost[ds]=dict(evidence=read(BASE/ds/'EVIDENCE_BARRIER.json'),online=read(BASE/ds/'PREDICTION_BARRIER.json'))
        for split,sp in p['splits'].items():
            labels=read(POOL/ds/f'GT_LABELS_{split}.json')
            for cond in p['conditions']:
                for order,seq in sp['orders'].items():
                    prev=prevsha=None
                    for at,parent in enumerate(seq):
                        f=BASE/ds/'online'/split/cond/order/f'{at:05}.pt';x=checked(f);r=p['rows'][parent];z=x['fit'];ex=checked(BASE/x['evidence_path'])
                        assert x['dataset']==ds and x['split']==split and x['condition']==cond and x['order']==order and x['arrival']==at and x['parent']==parent
                        assert x['previous_sha256']==prevsha and x['GT_read'] is False
                        expected=arrival(x['source_state'],prev,'O-split');checks+=check_state(expected,x['initial'])
                        assert torch.count_nonzero(x['initial'][QUERY])==0
                        checks+=check_state(z['initial'],x['initial'])
                        assert torch.equal(z['path'][0]['boxes'],x['before'])
                        assert z['parameter_count']==1792 and len(z['path'])==(1 if not x['anchors'] else 11)
                        selected=min(range(len(z['path'])),key=lambda k:z['path'][k]['loss'])
                        assert selected==z['selected_step'] and torch.equal(z['final'],z['path'][selected]['boxes'])
                        checks+=check_state(z['state'],z['path'][selected]['state'])
                        assert z['gradient_calls']==(10 if x['anchors'] else 0)
                        names=list(z['initial']);m=np.zeros(1792);v=np.zeros(1792);maxadam=0.
                        for j,h in enumerate(z['path']):
                            actual=loss_numpy(h['boxes'].numpy(),x['anchors']);assert abs(actual-h['loss'])<2e-5,(actual,h['loss']);checks+=1
                            for vv in h['state'].values():assert torch.isfinite(vv).all();checks+=vv.numel()
                            if 'update' not in h: continue
                            u=h['update'];g=u['gradient'].numpy().astype(np.float64);assert g.shape==(1792,) and np.isfinite(g).all()
                            m=.9*m+.1*g;v=.999*v+.001*g*g
                            expect=-.03*(m/(1-.9**(j+1)))/(np.sqrt(v/(1-.999**(j+1)))+1e-8)
                            # CUDA fp32 Adam and cancellation in raw displacement.
                            error=np.max(np.abs(expect-u['raw'].numpy()));maxadam=max(maxadam,float(error));assert error<2e-6,error
                            before=flat(h['state'],names);after=flat(z['path'][j+1]['state'],names)
                            assert torch.equal(after,before+u['direction'])
                            assert torch.equal(after-before,u['applied']) and torch.equal(u['raw'],u['direction'])
                            assert all(s==j+1 for s in u['adam_steps'].values());checks+=1792*4
                        cons=consolidate(x['initial'],z['state'],1/16);checks+=check_state(cons,x['committed'])
                        assert torch.count_nonzero(cons[QUERY])==0
                        tf=BASE/x['temporal_path'];assert sha(tf)==x['temporal_sha256'];tt=checked(tf)
                        if str(tf) not in timechecked:
                            assert tt['selected_step']==5 and tt['backwards']==5 and len(tt['path'])==6 and tt['failure'] is None
                            checks+=check_state(tt['initial_state'],tt['path'][0]['state'])
                            checks+=check_state(tt['state'],tt['path'][-1]['state'])
                            for h,l in zip(tt['path'],tt['losses']):
                                assert abs(time_loss(h['logits'],tt['evidence'],.2)-l)<1e-8
                                checks+=1
                            for n,src in tt['initial_state'].items():
                                shrunk=src+.25*(tt['state'][n]-src)
                                assert torch.equal(shrunk,tt['shrunk_state'][n]),n;checks+=src.numel()
                            timechecked.add(str(tf))
                        native=x['native'];final=x['final'];ids=r['frame_ids']
                        reconstructed=prediction(tt['shrunk']['logits'],z['final'],[
                            dict(frame_ids=ids[::2]),dict(frame_ids=ids[1::2])],ids)
                        assert reconstructed['physical_interval']==final['physical_interval']
                        truth={int(k):vv for k,vv in labels[str(parent)]['truth'].items()};span=labels[str(parent)]['span']
                        cases={'frozen':(native['boxes'],native['physical_interval']),
                            'inherited':(x['before'],native['physical_interval']),
                            'spatial':(z['final'],native['physical_interval']),
                            'temporal':(native['boxes'],final['physical_interval']),
                            'decota':(z['final'],final['physical_interval'])}
                        metrics={}
                        for name,(boxes,iv) in cases.items():
                            fast=DenseTube(boxes.numpy(),r,truth,span,clip=ds=='hc2').score(iv)
                            reference=official(boxes.numpy(),r,truth,span,iv,ds)
                            assert max(abs(fast[k]-reference[k]) for k in fast)<2e-12,(ds,name,fast,reference);checks+=3
                            metrics.update({f'{name}_{k}':val for k,val in reference.items()})
                        dv=metrics['decota_v']-metrics['frozen_v']
                        rec=dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,
                            **metrics,delta_v=dv,delta_t=metrics['decota_t']-metrics['frozen_t'],delta_s=metrics['decota_s']-metrics['frozen_s'],
                            inherited_delta_v=metrics['inherited_v']-metrics['frozen_v'],
                            spatial_delta_v=metrics['spatial_v']-metrics['frozen_v'],
                            temporal_delta_v=metrics['temporal_v']-metrics['frozen_v'],
                            current_spatial_delta_v=metrics['spatial_v']-metrics['inherited_v'],
                            gross_gain_v=max(dv,0),gross_loss_v=max(-dv,0),anchors=len(x['anchors']),
                            planned_frames=len(ex['expert']['positions4']),DINO_forwards=ex['expert']['new_DINO'],
                            selected_step=selected,skipped=z['skipped'],fit_seconds=x['fit_seconds'],
                            initial_state_sha256=state_hash(x['initial']),selected_state_sha256=state_hash(z['state']),
                            committed_state_sha256=state_hash(cons),source_state_sha256=state_hash(x['source_state']),
                            prediction_receipt_sha256=sha(f.with_suffix('.json')),temporal_receipt_sha256=sha(tf.with_suffix('.json')))
                        rows.append(rec)
                        updates.append(dict(dataset=ds,split=split,condition=cond,order=order,arrival=at,source_id=parent,
                            spatial_reference_losses=[h['loss'] for h in z['path']],selected_step=selected,
                            temporal_losses=tt['losses'],temporal_shrink=.25,max_Adam_arithmetic_error=maxadam,
                            query_reset=True,Adam_reset=True,LN_writeback_fraction=1/16,
                            inherited_LN_norm=float(torch.cat([(x['initial'][n]-x['source_state'][n]).flatten() for n in names if n!=QUERY]).norm()),
                            committed_LN_norm=float(torch.cat([(cons[n]-x['source_state'][n]).flatten() for n in names if n!=QUERY]).norm()),
                            temporary_query_norm=float(z['state'][QUERY].norm())))
                        prev=cons;prevsha=sha(f)
        print('AUDIT_SCORE',ds,'done',len(rows),'checks',checks,flush=True)
    assert len(rows)==1152 and len(timechecked)==576
    write(PUB/'ROWS.json',rows);write(PUB/'UPDATE_DIAGNOSTICS.json',updates)
    summary={}
    for ds in DATASETS:
        summary[ds]={}
        for split in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==split]
            summary[ds][split]=dict(corruption=source_summary([r for r in rr if r['condition']!='clean']),
                clean=source_summary([r for r in rr if r['condition']=='clean']),
                conditions={c:source_summary([r for r in rr if r['condition']==c]) for c in read(BASE/ds/'PLAN.json')['conditions']},
                orders={o:source_summary([r for r in rr if r['condition']!='clean' and r['order']==o]) for o in ['order1','order2']})
    write(PUB/'SUMMARY.json',summary)
    costs={ds:dict(unique_inputs=288,logical_arrivals=576,
        new_DINO_forward_count=cost[ds]['evidence']['new_DINO'],
        logical_observation_requests=sum(r['DINO_forwards'] for r in rows if r['dataset']==ds),
        accepted_reference_count=sum(r['anchors'] for r in rows if r['dataset']==ds),
        empty_reference_arrivals=sum(r['anchors']==0 for r in rows if r['dataset']==ds),
        spatial_backwards=cost[ds]['online']['spatial_backwards'],temporal_backwards=288*5,
        temporal_result_reuse_count=288,evidence_worker_seconds=cost[ds]['evidence']['seconds'],
        online_worker_seconds=cost[ds]['online']['seconds'],
        measured_spatial_fit_seconds=sum(r['fit_seconds'] for r in rows if r['dataset']==ds),
        peak_memory_bytes=cost[ds]['online']['peak_memory_bytes'],
        worker_time_includes_loading_IO=True,not_pure_GPU_kernel_time=True) for ds in DATASETS}
    write(PUB/'COST.json',costs)
    public_plan=dict(datasets={ds:dict(rows=48,search_sources=32,confirm_sources=16,arrivals=576,
        checkpoint=read(BASE/ds/'PLAN.json')['configuration']['checkpoint'],
        checkpoint_sha256=read(BASE/ds/'PLAN.json')['configuration']['checkpoint_sha256'],
        configuration=read(BASE/ds/'PLAN.json')['configuration'],
        conditions=read(BASE/ds/'PLAN.json')['conditions'],
        orders=read(BASE/ds/'PLAN.json')['splits']) for ds in DATASETS},
        historical_exposure=True,expert_schedule='every query up to four native-interval frames',
        spatial_LN_writeback=1/16,temporal_episodic=True,method_selected_after_GT=False,
        worker_GT_read=False,score_GT_read=True,root_GT_schema_inspected_before_launch=True,
        root_schema_inspection_used_for_configuration=False,
        global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'))
    write(PUB/'PROTOCOL_BINDING.json',public_plan)
    write(PUB/'ROOT_AUDIT.json',dict(status='pass',checks=checks,arrivals=1152,temporal_unique_inputs=576,
        global_seal_verified=True,independent_numpy_spatial_loss=True,independent_numpy_Adam=True,
        independently_checked_temporal_NLL_hinge=True,exact_1_16_recurrence=True,
        official_dense_and_vectorized_metrics_identical=True,
        seconds=time.time()-tick,GT_used_for_optimization=False))
    status(BASE/'STATUS.json',dict(status='CPU_scored_audited_pending_visual_publication',arrivals=1152,GT_read=True,time=time.time()))


def public_check(directory):
    p=Path(directory);rows=read(p/'ROWS.json');s=read(p/'SUMMARY.json');u=read(p/'UPDATE_DIAGNOSTICS.json');checks=0
    assert len(rows)==len(u)==1152
    lookup={(r['dataset'],r['split'],r['condition'],r['order'],r['arrival']):r for r in rows}
    assert len(lookup)==1152
    for r in rows:
        assert abs(r['delta_v']-(r['decota_v']-r['frozen_v']))<1e-14
        assert abs(r['delta_t']-(r['decota_t']-r['frozen_t']))<1e-14
        assert abs(r['delta_s']-(r['decota_s']-r['frozen_s']))<1e-14;checks+=3
    for h in u:
        assert h['selected_step']==min(range(len(h['spatial_reference_losses'])),key=lambda j:h['spatial_reference_losses'][j])
        assert h['query_reset'] and h['Adam_reset'] and h['LN_writeback_fraction']==1/16 and h['temporal_shrink']==.25;checks+=5
    for ds in DATASETS:
        for split in ['search','confirm']:
            rr=[r for r in rows if r['dataset']==ds and r['split']==split]
            scopes={'corruption':[r for r in rr if r['condition']!='clean'],'clean':[r for r in rr if r['condition']=='clean']}
            scopes.update({c:[r for r in rr if r['condition']==c] for c in s[ds][split]['conditions']})
            for scope, subset in scopes.items():
                expected=source_summary(subset);actual=s[ds][split][scope] if scope in ['corruption','clean'] else s[ds][split]['conditions'][scope]
                assert expected==actual,(ds,split,scope);checks+=len(expected['metrics'])*3
    return dict(status='pass',checks=checks,arrivals=1152,independent_summary_regeneration=True,time=time.time())


if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['run','public']);a.add_argument('directory',nargs='?');args=a.parse_args()
    if args.stage=='run':run()
    else: print(public_check(args.directory or PUB))

"""P6 all-deployment postseal official/dense, complete state and offline head root.

No GT call or label import is made until every original deployment arm has sealed.
The supervised CPU oracle is the existing five-step head recipe, kept separate.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections
import copy
import itertools
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *
from scripts.run_stvg_opd_p6_existing_temporal001 import equal,recorded_fit
OUT=PUB/'P6'


def statistic(rows,fields):
    """Paired parent vectors, identical seed and indices for every comparison."""
    groups=collections.defaultdict(list)
    for r in rows:groups[r['source_id']].append(r)
    parents=sorted(groups)
    a=np.asarray([[np.mean([r[f] for r in groups[p]]) for f in fields] for p in parents])
    assert len(parents)==32 and len(rows)==32 and np.isfinite(a).all()
    rng=np.random.default_rng(20261006);samples=[]
    for _ in range(100):samples.append(a[rng.integers(0,len(a),(100,len(a)))].mean(1))
    ci=np.quantile(np.concatenate(samples),[.025,.975],axis=0)
    result={}
    for j,f in enumerate(fields):
        v=a[:,j];delta=f.startswith('delta_')
        result[f]=dict(parent_macro=float(v.mean()),query_macro=float(np.mean([r[f] for r in rows])),ci95=ci[:,j].tolist(),
            all_parent_values=v.tolist(),min=float(v.min()),max=float(v.max()),median=float(np.median(v)),
            gross_gain_pp=float(np.maximum(v,0).mean()*100) if delta else None,
            gross_loss_pp=float(-np.minimum(v,0).mean()*100) if delta else None,
            harm_gt5pp_parents=int((v<-.05).sum()) if delta else None,
            harm_gt20pp_parents=int((v<-.20).sum()) if delta else None,
            positive_parents=int((v>1e-12).sum()) if delta else None,
            negative_parents=int((v<-1e-12).sum()) if delta else None)
    return dict(parent_sources=len(parents),queries=len(rows),bootstrap_replicates=10000,seed=20261006,
        unit='paired original parent',multiplicity_adjusted=False,historical_cohort=True,metrics=result)


def oracle_evidence(evidence,records,span):
    from scripts.run_decota_corrective_identifiability_v1 import tiou
    result=copy.deepcopy(evidence);quant=[]
    for ev,rec in zip(result['offsets'],records):
        ij=ev['ij'].cpu().numpy();f=np.asarray(rec['frame_ids']);intervals=np.c_[f[ij[0]],f[ij[1]]+1]
        values=tiou(intervals,span);at=int(values.argmax());ev['target']=at
        ev['target_indices']=ij[:,at].tolist();ev['interval']=intervals[at].tolist();quant.append(float(values[at]))
    result['offline_supervised_GT']=True;result['GT_online']=False
    return result,quant


def offline_head(z,span,cfg):
    import torch
    from methods.decota_final_simplified_v1.tensors import ParameterState,detached
    from methods.decota_final_simplified_v1.objectives import prediction,temporal_loss
    from methods.decota_final_simplified_v1.optim import _fit
    from scripts.run_decota_corrective_identifiability_v1 import make_head
    cap=z['capture']
    class Replica(ParameterState):
        def __init__(self):
            self.head=make_head(cap['source_head_state']);self.named=[('head.'+n,p) for n,p in self.head.named_parameters()]
            self.initial=self.state();self.inputs=cap['temporal_inputs'];self.zero=cap['zero']
        def values(self):return {**self.zero,'logits':[self.head(h)[-1] for h in self.inputs]}
    rep=Replica();initial=rep.values()['logits'];error=0.
    for a,b in zip(initial,cap['zero']['logits']):
        err=(a-b).abs();assert torch.all(err<=2e-5+1e-6*b.abs()),float(err.max());error=max(error,float(err.max()))
    native=prediction(initial,cap['zero']['boxes'],cap['records'],cap['frame_ids'])
    assert native['physical_interval']==z['predictions']['Native']['physical_interval']
    target,quant=oracle_evidence(z['fit']['evidence'],cap['records'],span)
    gradients={}
    for label,ev in [('ssl',z['fit']['evidence']),('GT',target)]:
        rep.head.zero_grad(set_to_none=True);temporal_loss(rep.values()['logits'],ev,cfg.margin).backward()
        gradients[label]=torch.cat([p.grad.detach().reshape(-1).double() for _,p in rep.named])
    norms={k:float(v.norm()) for k,v in gradients.items()}
    cosine=float(torch.dot(gradients['ssl'],gradients['GT'])/(norms['ssl']*norms['GT'])) if min(norms.values())>1e-14 else None
    start=time.perf_counter();fit,trace=recorded_fit(rep,cap['records'],cfg,oracle=target)
    elapsed=time.perf_counter()-start;checked=audit(fit,trace,cap['records'],cfg,oracle=True)
    pred=prediction(fit['shrunk']['logits'],cap['zero']['boxes'],cap['records'],cap['frame_ids'])
    assert all(torch.equal(v,rep.state()[k]) for k,v in rep.initial.items())
    # Actual original unrecorded optimizer parity at the first query of each target.
    parity=None
    if z['arrival']==0:
        raw=_fit(rep,rep.zero,lambda v:temporal_loss(v['logits'],target,cfg.margin),lr=cfg.temporal_lr,steps=cfg.temporal_steps,temporal=True,trace=True)
        equal({k:v for k,v in fit.items() if k not in ['evidence','shrunk_state','shrunk','eta','shrunk_parameter_changed']},raw)
        parity=True
    return dict(fit=detached(fit,'cpu'),trajectory=trace,math_audit=checked,prediction=pred,GT_read=True,
        offline_supervised=True,deployment_baseline=False,cost=dict(actual_CPU_head_seconds=elapsed),
        diagnostics=dict(CPU_source_logits_max_error=error,CPU_native_interval_equal=True,source_reset_exact=True,
            original_unrecorded_CPU_fit_bitwise=parity,gradient_cosine=cosine,gradient_norms=norms,
            legal_grid_max_GT_tIoU=quant),GT_span=span)


def source_reset_and_spatial(z,orig,inp,ds):
    import torch
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.run_stvg_opd_p5_budget_precision001 import dispatch_saved
    from scripts.stvg_opd_paper_later_common_v1 import committed
    assert z['capture']['source_model_state_sha256']==inp['source_model_state_sha256']
    first=load(NS/'qualification'/ds/'00000_recorded.pt')['capture']['source_head_state']
    equal(first,z['capture']['source_head_state']);equal(first,z['fit']['initial_state'])
    f=orig['fit'];expected=committed(f['initial'],f['state'],f['config']['writeback'])
    equal(expected,orig['committed']);assert orig['query_reset'] and orig['Adam_reset'] and orig['Native_WHEN_fixed']
    assert sum(v.numel() for v in f['initial'].values())==1792 and torch.count_nonzero(f['initial']['spatial.query_residual'])==0
    if orig['arrival']:
        prev=BASE/'stages'/('P0_'+ds)/'clean/order1/on_policy'/f'{orig["arrival"]-1:05}.pt'
        assert sha(prev)==orig['previous_payload_sha256'];p=load(prev)
        for k,v in f['initial'].items():assert torch.equal(v,torch.zeros_like(v) if k=='spatial.query_residual' else p['committed'][k])
    check=dispatch_saved(f,unpack_expert(inp['expert']),orig,dict(origin_stage='P0_'+ds,reused_complete_identical_stream=True))
    assert check==orig['math_audit'];return check


def score(verified_barrier=None):
    import torch
    from methods.decota_final_simplified_v1.objectives import prediction
    torch.set_num_threads(2);runtime=verify();seal=read(BASE/'P6_PREDICTION_BARRIER.json')
    if verified_barrier is not None:assert verified_barrier==seal
    assert seal['status']=='sealed' and seal['queries']==64 and seal['logical_outputs']==256 and seal['arms']==ARMS and not seal['GT_read']
    assert seal['runtime_sha256']==sha(RUNTIME)
    for ds in ['hc2','vidstg']:
        p=NS/'formal'/ds/'PREDICTION_BARRIER.json';b=read(p)
        assert sha(p)==seal['source_barriers'][ds] and b['status']=='sealed' and b['queries']==32 and b['arms']==ARMS
        assert b['time']<=seal['time'] and not b['GT_read']
    for rc in seal['records']:
        p=BASE/rc['path'];assert rc==read(p.with_suffix('.json')) and sha(p)==rc['sha256'] and p.stat().st_size==rc['bytes']
        assert not rc['GT_read'] and rc['outputs']==4 and rc['time']<=seal['time']
    assert len(seal['records'])==64 and len({(r['dataset'],r['query_ordinal']) for r in seal['records']})==64
    exposure=NS/'GT_EXPOSURE.json'
    if not exposure.exists():write(exposure,dict(status='authorized_postseal_GT',global_deployment_barrier_sha256=sha(BASE/'P6_PREDICTION_BARRIER.json'),
        full_deployment_before_GT=True,logical_outputs=256,time=time.time()))
    # Imports and reads with target GT occur only below the verified global barrier.
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    from vg_tta.tastvg_oracle_event5_v1 import official
    counts=collections.Counter();rows=[];private=[];provenance={};opaque=[];math_records=[]
    start=time.time();allarms=ARMS+['Offline_GT_Head']
    for ds in ['hc2','vidstg']:
        stage=definition(ds);gt,spans,prov=truths(ds,stage);provenance[ds]=prov
        plan=read(ROOT/'artifacts/decota_paper_experiments_v1'/ds/'PLAN.json')
        sourceids={s:i for i,s in enumerate(sorted({r['source'] for r in plan['rows']}))}
        for at,q in enumerate(stage['orders']['order1']):
            p=NS/'formal'/ds/f'{at:05}.pt';rc=read(p.with_suffix('.json'));z=load(p)
            assert z['dataset']==ds and z['query_ordinal']==q and z['arrival']==at and z['config']==config(ds).to_dict() and not z['GT_read']
            original_z,inp,binding=original(ds,q);assert binding==z['original_spatial_fit_binding']
            assert z['frame_ids']==inp['frame_ids'] and z['pixel_sha256']==inp['pixel_sha256'] and z['corruption_spec'] is None
            equal(z['predictions']['Native']['boxes'],inp['native_boxes']);equal(z['predictions']['SpatialOPD'],original_z['predictions']['After'])
            equal(z['original_spatial_before'],original_z['predictions']['Before'])
            assert z['predictions']['Native']['physical_interval']==inp['interval'] and set(z['predictions'])==set(ARMS)
            equal(z['capture']['zero']['boxes'],inp['native_boxes'])
            assert z['capture']['records'][0]['frame_ids']==z['frame_ids'][0::2] and z['capture']['records'][1]['frame_ids']==z['frame_ids'][1::2]
            checked=audit(z['fit'],z['trajectory'],z['capture']['records'],config(ds));assert checked==z['math_audit']
            spatial=source_reset_and_spatial(z,original_z,inp,ds)
            if at<2:
                qp=NS/'qualification'/ds/f'{at:05}_recorded.pt';qual=load(qp)
                for k in ['fit','capture']:equal(z[k],qual[k])
                for a in ARMS[:3]:equal(z['predictions'][a],qual['predictions'][a])
                assert rc['first_formal_qualified_bitwise']['qualified_sha256']==sha(qp);counts['qualified_formal_complete_pairs']+=1
            cap=z['capture'];path_intervals=[]
            for h in z['fit']['path']:
                pr=prediction(h['logits'],cap['zero']['boxes'],cap['records'],cap['frame_ids']);path_intervals.append(pr['physical_interval'])
            offline_path=NS/'offline_GT_head'/ds/f'{at:05}.pt'
            if offline_path.with_suffix('.json').exists():
                orc=read(offline_path.with_suffix('.json'));assert sha(offline_path)==orc['sha256'] and orc['GT_read'];oracle=load(offline_path)
            else:
                oracle=offline_head(z,spans[q],config(ds));save(offline_path,oracle)
                write(offline_path.with_suffix('.json'),dict(status='complete_offline_supervised_CPU_head',dataset=ds,query_ordinal=q,
                    path=str(offline_path.relative_to(BASE)),sha256=sha(offline_path),bytes=offline_path.stat().st_size,GT_read=True,
                    unavailable_at_deployment=True,global_barrier_sha256=sha(BASE/'P6_PREDICTION_BARRIER.json'),time=time.time()))
            assert audit(oracle['fit'],oracle['trajectory'],cap['records'],config(ds),oracle=True)==oracle['math_audit']
            preds={**z['predictions'],'Offline_GT_Head':oracle['prediction']};row=plan['rows'][q]
            r=dict(dataset=ds,source=stage['source'],stage='P6_'+ds,condition='clean',order='order1',arrival=at,query_ordinal=q,
                source_id=sourceids[row['source']],payload_sha256=sha(p),original_spatial_payload_sha256=binding['sha256'],
                offline_head_payload_sha256=sha(offline_path),query_type=row.get('input',{}).get('query_type','unspecified'),
                complete_temporal_cost=z['cost'],original_spatial_cost=original_z['compute'],offline_head_cost=oracle['cost'],
                signal=dict(gradient_cosine=oracle['diagnostics']['gradient_cosine'],legal_grid_max_GT_tIoU=oracle['diagnostics']['legal_grid_max_GT_tIoU']))
            curves={}
            for a,pr in preds.items():
                values=official(pr['boxes'].numpy(),row,gt[q],spans[q],pr['physical_interval'],ds)
                independent,fids,iou=dense_metric(pr['boxes'].numpy(),row,gt[q],spans[q],pr['physical_interval'],ds)
                assert all(abs(values[k]-independent[k])<2e-10 for k in ['v','t','s']);counts['official_dense_scalar_checks']+=3
                for k,v in values.items():r[a+'_'+k]=v
                curves[a]=iou
            for a,b in itertools.combinations(allarms,2):
                for m in ['v','t','s']:r[f'delta_{b}_minus_{a}_{m}']=r[b+'_'+m]-r[a+'_'+m]
            observed=np.isin(fids,[row['frame_ids'][i] for i in original_z['fit']['positions']])
            for label,mask in [('observed',observed),('unobserved',~observed)]:
                r[label+'_spatial_frame_IoU_delta']=float((curves['SpatialOPD']-curves['Native'])[mask].mean()) if mask.any() else None
                counts['observed_unobserved_checks']+=1
            from scripts.run_decota_corrective_identifiability_v1 import tiou
            r['signal'].update(teacher_offset_GT_tIoU=[float(tiou(np.asarray([ev['interval']]),spans[q])[0]) for ev in z['fit']['evidence']['offsets']],
                actual_GPU_SSL_losses=list(z['fit']['losses']),path_tIoU=[float(tiou(np.asarray([iv]),spans[q])[0]) for iv in path_intervals],
                native_tIoU=r['Native_t'],shrunk_temporal_tIoU=r['TemporalHead_t'],oracle_tIoU=r['Offline_GT_Head_t'],
                actual_head_changed=z['fit']['parameter_changed'],shrunk_head_changed=z['fit']['shrunk_parameter_changed'])
            math_records.append(dict(dataset=ds,arrival=at,query_ordinal=q,temporal=checked,spatial=spatial,offline_GT=oracle['math_audit']))
            private.append(dict(dataset=ds,arrival=at,query_ordinal=q,payload_path=str(p.relative_to(BASE)),oracle_path=str(offline_path.relative_to(BASE)),GT_span=spans[q]))
            rows.append(r);counts['formal_queries']+=1;counts['deployment_outputs']+=4;counts['offline_CPU_GT_fits']+=1
            counts['complete_GPU_head_math_dicts']+=1;counts['complete_spatial_math_dicts']+=1;counts['complete_offline_GT_math_dicts']+=1
            counts['head_backward_rounds']+=z['fit']['backwards'];counts['offline_GT_backward_rounds']+=oracle['fit']['backwards']
            counts['spatial_rounds']+=len(original_z['fit']['rounds']);counts['head_path_state_coordinates']+=checked['state_coordinates']
            counts['spatial_state_coordinates']+=1792;counts['offline_GT_path_state_coordinates']+=oracle['math_audit']['state_coordinates']
            opaque.append(dict(path=str(p.relative_to(BASE)),sha256=sha(p),bytes=p.stat().st_size,input_binding=binding))
            status(NS/'POSTSEAL_CPU_STATUS.json',dict(status='running',stage='complete_source_math_dense_offline_head',done=len(rows),expected=64,
                CPU_only=True,GT_after_all_256_deployments_sealed=True,process=os.getpid(),time=time.time()))
            print('P6_CPU_COMPLETE_QUERY',ds,at+1,flush=True)
    assert counts['formal_queries']==64 and counts['deployment_outputs']==256 and counts['qualified_formal_complete_pairs']==4
    fields=[f for f in rows[0] if f.endswith(('_v','_t','_s'))]
    stats={ds:statistic([r for r in rows if r['dataset']==ds],fields) for ds in ['hc2','vidstg']}
    # Independent query-level versus paired-parent recomputation and bootstrap.
    for ds in stats:
        rr=[r for r in rows if r['dataset']==ds];again=statistic(rr,fields);assert again==stats[ds]
        for f in fields:assert abs(stats[ds]['metrics'][f]['parent_macro']-np.mean([r[f] for r in rr]))<2e-14
    for x in opaque:
        p=BASE/x['path'];assert sha(p)==x['sha256'] and p.stat().st_size==x['bytes']
        binding=x['input_binding']
        assert sha(BASE/binding['input_path'])==binding['input_sha256']
        assert sha((BASE/binding['input_path']).with_suffix('.json'))==binding['input_receipt_sha256']
        assert sha(BASE/binding['path'])==binding['sha256'];counts['second_opaque_prediction_input_receipt_SHA_checks']+=4
    write(OUT/'ROWS.json',rows);write(OUT/'STATISTICS.json',stats)
    write(OUT/'ROOT_MATH_STATE_DENSE_READBACK.json',dict(status='pass',scope='all64 original P6 inputs, complete head/spatial/oracle dictionaries/state/dense/paired-parent reads',
        counts=dict(counts),records=math_records,GT_after_global_seal=True,CPU_only=True,new_model_calls=0,time=time.time()))
    write(NS/'PRIVATE_CASE_BINDINGS.json',private)
    outputs={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ROWS.json',OUT/'STATISTICS.json',OUT/'ROOT_MATH_STATE_DENSE_READBACK.json']}
    write(BASE/'P6_CPU_COMPLETION.json',dict(status='pending_actual_root_signal_cases_visual_and_publication',scope='all P6 deployment outputs and separate offline GT-head CPU/root algebra',
        queries=64,parent_sources=64,logical_deployment_outputs=256,offline_GT_head_fits=64,counts=dict(counts),outputs=outputs,
        provenance=provenance,GT_after_global_seal=True,CPU_only=True,new_model_calls=0,elapsed_seconds=time.time()-start,
        global_barrier_sha256=sha(BASE/'P6_PREDICTION_BARRIER.json'),P6_phase_complete=False,paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:score()
    except BaseException:
        status(NS/'POSTSEAL_CPU_FAILURE.json',dict(status='failed',CPU_only=True,process=os.getpid(),traceback=traceback.format_exc(),time=time.time()));raise

"""A bounded CPU-only Old8 quality test; sealed before offline GT scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, collections, functools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_quality_common_v1 import *
import numpy as np
from vg_tta.tastvg_temporal_quality_old8_v1 import activation,feature_edges,decide
from vg_tta.tastvg_temporal_qualification_v1 import critic_scores

BASE_FIELDS=['A_v','quality_v','actual_gain','A_t','quality_t','actual_t_gain','gross_gain','gross_loss']
EXPERT_FIELDS=['native_v','native_t','oracle_v','oracle_t','old_regret','quality_regret','regret_reduction',
    'old_t_regret','quality_t_regret','t_regret_reduction','old_fast_gain','quality_fast_gain',
    'candidate_unique','interval_changed','quality_no_outer_candidates','activation_range']

def prepare():
    import torch
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(PREVIOUS/'FINAL_COMPLETION.json')['status']=='completed_and_verified_publication'
    sys.addaudithook(guard); tick=time.monotonic()
    prior=read(PREVIOUS/'RUNTIME_LOCK.json')
    inputs=prior['inputs']
    for i,(f,h) in enumerate(inputs.items(),1):
        assert sha(ROOT/f)==h,f
        if i%1400==0: print('BIND_IMMUTABLE_INPUTS',i,len(inputs),flush=True)
    cells=read(OLD/'COHORT.json')['cells']
    assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    counts=collections.Counter(c['dataset']+'_'+c['split'] for c in cells)
    write(BASE/'COHORT.json',dict(cells=cells,counts=dict(counts),historical_exposure=True))
    design=dict(version='tastvg_temporal_quality_old8_v1',signal='PE_pooled_query_outer_inner_contrast',
        outer_ratio=.25,precision='numpy_float64',candidate_support='exact_original_Old8',
        baseline='original_max_proposal_confidence_times_interval_IoU',
        video_signal='cosine_of_each_cached_projected_PE_video_column_and_pooled_query_column0',
        integration='fractional_overlap_of_phase0_2Hz_piecewise_constant_bins_last_bin_to_window_end',
        constant_curve_epsilon=1e-12,tie_epsilon=1e-12,ties='native_first_original_candidate_order',
        full_window_no_outer='neutral_zero_contrast_against_same_window_mean_candidate_retained',
        unavailable_or_constant='keep_old_A_selection',
        all_arrivals=1152,expert_arrivals=288,corrupt_expert_arrivals=240,clean_expert_arrivals=48,
        params={ds:plan(ds)['params'] for ds in DATASETS},no_target_training=True,
        new_model_calls=0,new_expert_calls=0,new_views=0,new_backprop=0,
        all_historically_exposed=True,confirmation_not_used_to_choose_formula=True,GT_read=False,
        references=dict(OIC='https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.pdf',
            PE='https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md'))
    write(BASE/'DESIGN.json',design)
    metadata={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'methods/CURRENT_METHOD.json',
        PREVIOUS/'FINAL_COMPLETION.json',PREVIOUS/'RUNTIME_LOCK.json',OLD/'COHORT.json',
        BASE/'COHORT.json',BASE/'DESIGN.json']}
    for ds in DATASETS:
        f=OLD/ds/'PLAN.json';metadata[str(f.relative_to(ROOT))]=sha(f)
        p=plan(ds); assert len(p['rows'])==48
        ss={sp:{c['parent'] for c in cells if c['dataset']==ds and c['split']==sp} for sp in SPLITS}
        assert len(ss['search'])==32 and len(ss['confirm'])==16 and not ss['search'] & ss['confirm']
    code=['protocols/tastvg_temporal_quality_old8_v1.md','docs/tastvg_temporal_quality_old8_v1/EXECUTION.md',
        'vg_tta/tastvg_temporal_quality_old8_v1.py','scripts/tastvg_temporal_quality_common_v1.py',
        'scripts/run_tastvg_temporal_quality_v1.py','scripts/test_tastvg_temporal_quality_v1.py',
        'vg_tta/tastvg_temporal_qualification_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py',
        'scripts/tastvg_oracle_event5_common_v1.py','scripts/tastvg_correction_views_common_v1.py',
        'scripts/decota_matrix_common_v1.py','external/UniversalVTG/universal_vtg_inference.py',
        'external/UniversalVTG/feature_extraction/extract_text_features.py',
        'external/UniversalVTG/feature_extraction/extract_visual_features.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in code},inputs=inputs,
        protected_metadata=metadata,label_hashes_from_predecessor_receipt=prior['label_hashes_from_predecessor_receipt'],
        checkpoint_state_sha256=prior['checkpoint_state_sha256'],GT_read=False,time=time.time()))
    assert not torch.cuda.is_initialized()
    write(BASE/'PREPARATION.json',dict(status='pass',protected_inputs=len(inputs),GT_read=False,
        CUDA_initialized=False,worker_wall_seconds=time.monotonic()-tick,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_unlabelled_scoring',GT_read=False,time=time.time()))
    archive('公式/边界/回退和全部输入已锁，尚无新读出或GT评分')
    print('PREPARED',dict(counts),flush=True)

def generate():
    import torch
    sys.addaudithook(guard); verify();torch.set_num_threads(2);tick=time.monotonic();counts=collections.Counter()
    @functools.lru_cache(maxsize=288)
    def cached_signal(ds,parent,cond,pixel):
        c=dict(dataset=ds,parent=parent,condition=cond,pixel_sha256=pixel)
        e,r=evidence(c);curve,available=activation(e['video_features'].numpy(),e['text_features'].numpy())
        row=plan(ds)['rows'][parent];ids=row['frame_ids'];duration=(ids[-1]-ids[0]+1)/row['input']['fps']
        assert abs(duration-e['duration'])<1e-9 and len(e['picked_observations'])==len(curve)
        edges=feature_edges(len(curve),duration)
        return dict(curve=curve.tolist(),available=available,edges=edges.tolist(),receipt=r,
            proposals=e['proposals'],confidence=e['proposal_confidence'],
            video_shape=list(e['video_features'].shape),text_shape=list(e['text_features'].shape))
    files={};cells=read(BASE/'COHORT.json')['cells']
    for i,c in enumerate(cells,1):
        budget();a=acell(c);x=donor(c);ids=plan(c['dataset'])['rows'][c['parent']]['frame_ids']
        assert torch.equal(x['A']['boxes'],a['slow']['boxes'])
        assert x['persistent_pre_sha']==a['pre_sha'] and x['persistent_post_sha']==a['post_sha']
        assert x['A']['indices']==a['final_indices'] and a['pixel_sha256']==c['pixel_sha256']
        p=dict(cell_key=key(c),dataset=c['dataset'],split=c['split'],source_id=c['parent'],
            condition=c['condition'],order=c['order'],arrival=c['arrival'],expert_scheduled=c['scheduled'],
            persistent_pre_sha=a['pre_sha'],persistent_post_sha=a['post_sha'],A_indices=a['final_indices'],
            native_indices=a['slow']['indices'],old_payload_sha256=sha(oldfile(c)),
            donor_payload_sha256=sha(donorfile(c)),new_expert_calls=0,new_model_calls=0,GT_read=False)
        if c['scheduled']:
            s=cached_signal(c['dataset'],c['parent'],c['condition'],c['pixel_sha256'])
            pool=x['temporal_candidates'];assert pool==a['temporal']['candidates'] and len(pool)==8
            original=critic_scores([q['physical_interval'] for q in pool],s['proposals'],s['confidence']).tolist()
            selected=int(np.argmax(original));assert original==a['temporal']['scores'] and selected==a['temporal']['selected']
            assert pool[selected]['indices']==a['final_indices'] and pool[0]['indices']==a['slow']['indices']
            lo,hi=ids[0],ids[-1]+1
            intervals=[[(q['physical_interval'][0]-lo)/(hi-lo),(q['physical_interval'][1]-lo)/(hi-lo)] for q in pool]
            decision=decide(s['curve'],s['edges'],intervals,selected,s['available'])
            p.update(candidates=pool,intervals_normalized=intervals,old_scores=original,old_selected=selected,
                quality=decision,curve=s['curve'],feature_edges=s['edges'],embedding_available=s['available'],
                expert_receipt=s['receipt'],video_shape=s['video_shape'],text_shape=s['text_shape'],
                quality_indices=pool[decision['selected']]['indices'])
            counts['expert_scored']+=1;counts['original_score_exact_parity']+=1
            counts['fallback_'+decision['fallback_reason']]+=1
            counts['no_outer_candidates']+=sum(q['no_outer_observation'] for q in decision['details'])
        else:p['quality_indices']=a['final_indices'];counts['nonexpert_exact_A']+=1
        f=BASE/c['dataset']/'predictions'/f'{prefix(c)}.json';write(f,p);files[str(f.relative_to(BASE))]=sha(f)
        if i%96==0:
            status(BASE/'STATUS.json',dict(status='unlabelled_scoring_running',done=i,total=1152,GT_read=False,
                worker_pid=os.getpid(),time=time.time()));print('NO_GT_READOUT',i,1152,flush=True)
    assert counts['expert_scored']==counts['original_score_exact_parity']==288 and counts['nonexpert_exact_A']==864
    assert not torch.cuda.is_initialized();verify()
    write(BASE/'GENERATION_RESOURCES.json',dict(status='pass',counts=dict(counts),arrivals=1152,GT_read=False,
        CUDA_initialized=False,new_model_calls=0,new_expert_calls=0,new_backprop=0,
        unique_feature_inputs=cached_signal.cache_info().misses,worker_wall_seconds=time.monotonic()-tick,time=time.time()))
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',arrivals=1152,expert_arrivals=288,
        files=files,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_CPU_GT_score',GT_read=False,time=time.time()))
    archive('1152读出和288同Old8新分数已封存，原critic全部逐值复现；待CPU GT评分')
    print('GLOBAL_SEALED',dict(counts),flush=True)

def summarize(rows):
    out={}
    for group in ['corruption','clean']:
        out[group]={}
        for subset in ['all','expert','nonexpert']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
            z=summary(rr,BASE_FIELDS+(EXPERT_FIELDS if subset=='expert' else []))
            z['counts']=dict(improved=sum(r['actual_gain']>1e-12 for r in rr),
                harmed=sum(r['actual_gain']<-1e-12 for r in rr),unchanged=sum(abs(r['actual_gain'])<=1e-12 for r in rr),
                severe_harm_gt5pp=sum(r['actual_gain']<-.05 for r in rr),
                baseline_good_gt03=sum(r['A_v']>.3 for r in rr),baseline_good_gt05=sum(r['A_v']>.5 for r in rr))
            z['correctness']={str(t):dict(correct_to_wrong=sum(r['A_v']>t and r['quality_v']<=t for r in rr),
                wrong_to_correct=sum(r['A_v']<=t and r['quality_v']>t for r in rr)) for t in [.3,.5]}
            if subset=='expert':
                z['counts'].update(old_rerank_gain_destroyed=sum(r['old_fast_gain']>1e-12 and r['quality_v']<r['A_v']-1e-12 for r in rr),
                    old_positive_rerank=sum(r['old_fast_gain']>1e-12 for r in rr),
                    replacement_v_below_native=sum(r['quality_selected']!=0 and r['quality_v']<r['native_v']-1e-12 for r in rr),
                    replacement_t_below_native=sum(r['quality_selected']!=0 and r['quality_t']<r['native_t']-1e-12 for r in rr),
                    fallback=sum(r['quality_fallback_reason']!='none' for r in rr))
                den=z['metrics']['old_regret']['mean'];num=z['metrics']['regret_reduction']['mean']
                z['regret_recovered_fraction']=num/den if den>1e-12 else None
                valid=[r for r in rr if r['strict_v_pairs']>0]
                z['pairwise_v']=summary(valid,['old_pair_v_accuracy','quality_pair_v_accuracy','pair_v_accuracy_gain'])
                z['pairwise_v']['no_strict_pair_cells']=len(rr)-len(valid)
            out[group][subset]=z
    return out

def pair_accuracy(scores,values):
    good=0.;n=0
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            if abs(values[i]-values[j])<=1e-12:continue
            n+=1;d=(scores[i]-scores[j])*np.sign(values[i]-values[j])
            good+=1. if d>1e-12 else .5 if abs(d)<=1e-12 else 0.
    return good/n if n else 0.,n

def score():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,physical
    torch.set_num_threads(2);verify(labels=True);bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    assert bar['status']=='sealed' and len(bar['files'])==1152
    tick=time.monotonic();checks=collections.Counter();maxerr=0.
    write(BASE/'GT_EXPOSURE.json',dict(scope='offline scoring only after global 1152 readouts sealed',
        global_prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),
        prediction_barrier_time=bar['time'],GT_score_started=time.time(),generation_GT_read=False,
        all_sources_historically_exposed=True))
    for ds in DATASETS:
        p=plan(ds)
        for split in SPLITS:
            labels=read(POOL/ds/f'GT_LABELS_{split}.json');rows=[]
            for c in [c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==split]:
                f=BASE/ds/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                pred=read(f);x=donor(c);row=p['rows'][c['parent']];g=labels[str(c['parent'])]
                truth={int(k):v for k,v in g['truth'].items()};span=g['span'];ids=row['frame_ids']
                dense=DenseTube(x['A']['boxes'],row,truth,span,ds=='hc2')
                intervals=[x['A']['physical_interval'],physical(pred['quality_indices'],ids)]
                values=[dense.score(i) for i in intervals];a,q=values
                r=dict(dataset=ds,split=split,source_id=c['parent'],condition=c['condition'],order=c['order'],
                    arrival=c['arrival'],expert_scheduled=c['scheduled'],A_v=a['v'],quality_v=q['v'],
                    actual_gain=q['v']-a['v'],A_t=a['t'],quality_t=q['t'],actual_t_gain=q['t']-a['t'],
                    gross_gain=max(q['v']-a['v'],0.),gross_loss=max(a['v']-q['v'],0.),
                    A_state_pre_sha256=pred['persistent_pre_sha'],A_state_post_sha256=pred['persistent_post_sha'])
                measured=list(zip(intervals,values))
                if c['scheduled']:
                    pool=pred['candidates'];pv=[dense.score(z['physical_interval']) for z in pool]
                    vs=[z['v'] for z in pv];ts=[z['t'] for z in pv];oi=pred['old_selected'];qi=pred['quality']['selected']
                    assert abs(vs[oi]-a['v'])<1e-12 and abs(vs[qi]-q['v'])<1e-12
                    pa,n=pair_accuracy(pred['old_scores'],vs);qa,_=pair_accuracy(pred['quality']['scores'],vs)
                    r.update(native_v=vs[0],native_t=ts[0],oracle_v=max(vs),oracle_t=max(ts),
                        old_regret=max(vs)-a['v'],quality_regret=max(vs)-q['v'],regret_reduction=q['v']-a['v'],
                        old_t_regret=max(ts)-a['t'],quality_t_regret=max(ts)-q['t'],t_regret_reduction=q['t']-a['t'],
                        old_fast_gain=a['v']-vs[0],quality_fast_gain=q['v']-vs[0],
                        candidate_unique=len({tuple(z['indices']) for z in pool}),interval_changed=int(oi!=qi),
                        old_selected=oi,quality_selected=qi,candidate_v=vs,candidate_t=ts,
                        candidate_indices=[z['indices'] for z in pool],intervals_normalized=pred['intervals_normalized'],
                        old_scores=pred['old_scores'],quality_scores=pred['quality']['scores'],
                        quality_details=pred['quality']['details'],quality_fallback_reason=pred['quality']['fallback_reason'],
                        activation_curve=pred['curve'],feature_edges=pred['feature_edges'],
                        embedding_available=pred['embedding_available'],activation_range=float(np.ptp(pred['curve'])),
                        quality_no_outer_candidates=sum(z['no_outer_observation'] for z in pred['quality']['details']),
                        feature_cache_sha256=pred['expert_receipt']['cache_sha256'],
                        strict_v_pairs=n,old_pair_v_accuracy=pa,quality_pair_v_accuracy=qa,pair_v_accuracy_gain=qa-pa)
                    measured.extend((z['physical_interval'],v) for z,v in zip(pool,pv));checks['expert_cells']+=1
                else:assert r['actual_gain']==0 and r['actual_t_gain']==0 and pred['quality_indices']==pred['A_indices']
                for interval,z in measured:
                    off=official(x['A']['boxes'],row,truth,span,interval,ds)
                    for metric in ['v','t','s']:
                        er=abs(z[metric]-off[metric]);assert er<1e-10,(key(c),metric,er)
                        maxerr=max(maxerr,er);checks['official_scalar_checks']+=1
                rows.append(r);checks['arrivals']+=1
                if checks['arrivals']%192==0:print('SCORE_FIXED_OLD8',checks['arrivals'],1152,flush=True)
            out=PUBLIC/split/ds
            write(out/'ROWS.json',rows);write(out/'SUMMARY.json',summarize(rows))
            write(out/'CASES.json',{m:dict(high=sorted(rows,key=lambda r:-r[m])[:5],low=sorted(rows,key=lambda r:r[m])[:5])
                for m in ['actual_gain','actual_t_gain']})
    assert checks['arrivals']==1152 and checks['expert_cells']==288 and not torch.cuda.is_initialized()
    result=dict(status='pass',checks=dict(checks),max_official_dense_error=maxerr,
        CUDA_initialized=False,worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'SCORE_ROOT_CHECKS.json',result);verify(labels=True)
    status(BASE/'STATUS.json',dict(status='scored_pending_root_audit_report_publication',time=time.time()))
    archive('1152读出/288同池oracle已CPU评分，待独立根审计、报告及GitHub收尾')
    print('SCORED',result,flush=True)

def root():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import official,physical
    tick=time.monotonic();verify(inputs=True,labels=True);bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    exp=read(BASE/'GT_EXPOSURE.json');assert exp['prediction_barrier_time']<exp['GT_score_started']
    checks=collections.Counter();maxerr=0.
    for ds in DATASETS:
        p=plan(ds)
        for split in SPLITS:
            rows=read(PUBLIC/split/ds/'ROWS.json');labels=read(POOL/ds/f'GT_LABELS_{split}.json')
            priorrows=read(ROOT/'results/tastvg_temporal_candidate_coverage/2026-10-03'/split/ds/'ROWS.json')
            priorby={(r['condition'],r['order'],r['arrival']):r for r in priorrows}
            cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==split]
            assert len(rows)==len(cells)
            for r,c in zip(rows,cells):
                f=BASE/ds/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                z=read(f);a=acell(c);x=donor(c);row=p['rows'][c['parent']];ids=row['frame_ids']
                assert torch.equal(a['slow']['boxes'],x['A']['boxes'])
                assert z['persistent_pre_sha']==a['pre_sha']==r['A_state_pre_sha256']
                assert z['persistent_post_sha']==a['post_sha']==r['A_state_post_sha256']
                assert a['final_indices']==z['A_indices'] and r['source_id']==c['parent']
                checks['A_boxes_and_states']+=1
                before=priorby[c['condition'],c['order'],c['arrival']]
                for m in ['A_v','A_t']:assert abs(r[m]-before[m])<1e-12
                checks['prior_A_scalar_parity']+=2
                g=labels[str(c['parent'])];truth={int(k):v for k,v in g['truth'].items()}
                if c['scheduled']:
                    e,receipt=evidence(c);pool=x['temporal_candidates'];assert pool==z['candidates']==a['temporal']['candidates']
                    assert len(pool)==8 and receipt['cache_sha256']==r['feature_cache_sha256']
                    # Independent cosine calculation: explicit sum, no scorer helper.
                    v=e['video_features'].double().numpy();q=e['text_features'][:,0].double().numpy()
                    qnorm=np.sqrt(np.sum(q*q));vnorm=np.sqrt(np.sum(v*v,axis=0));avail=bool(qnorm>0 and np.all(vnorm>0))
                    curve=np.sum(v*q[:,None],axis=0)/(vnorm*qnorm) if avail else np.zeros(v.shape[1])
                    np.testing.assert_allclose(curve,z['curve'],rtol=0,atol=1e-12)
                    scores=critic_scores([t['physical_interval'] for t in pool],e['proposals'],e['proposal_confidence']).tolist()
                    assert scores==z['old_scores']==a['temporal']['scores'] and int(np.argmax(scores))==z['old_selected']
                    assert r['candidate_v']==before['old_candidate_v'] and abs(r['oracle_v']-before['old_oracle_v'])<1e-12
                    checks['old_support_score_and_oracle_parity']+=1;checks['independent_curve_values']+=len(curve)
                    for i,t in enumerate(pool):
                        qoff=official(x['A']['boxes'],row,truth,g['span'],t['physical_interval'],ds)
                        for metric,field in [('v','candidate_v'),('t','candidate_t')]:
                            er=abs(qoff[metric]-r[field][i]);assert er<1e-10;maxerr=max(maxerr,er);checks['official_candidate_scalars']+=1
                else:assert z['quality_indices']==a['final_indices'] and r['actual_gain']==0;checks['nonexpert_identical']+=1
                qoff=official(x['A']['boxes'],row,truth,g['span'],physical(z['quality_indices'],ids),ds)
                for metric,field in [('v','quality_v'),('t','quality_t')]:
                    er=abs(qoff[metric]-r[field]);assert er<1e-10;maxerr=max(maxerr,er);checks['official_actual_scalars']+=1
            assert summarize(rows)==read(PUBLIC/split/ds/'SUMMARY.json');checks['summaries_recomputed']+=1
    assert checks['A_boxes_and_states']==1152 and checks['old_support_score_and_oracle_parity']==288
    from scripts.audit_tastvg_temporal_quality_public_v1 import audit
    anonymous=audit(PUBLIC);assert anonymous['status']=='pass'
    verify(inputs=True,labels=True)
    result=dict(status='pass',checks=dict(checks),max_official_error=maxerr,anonymous_audit=anonymous,
        immutable_inputs=len(read(BASE/'RUNTIME_LOCK.json')['inputs']),global_seal_before_new_GT=True,
        CUDA_initialized=torch.cuda.is_initialized(),worker_wall_seconds=time.monotonic()-tick,time=time.time())
    assert not result['CUDA_initialized'];write(BASE/'FINAL_ROOT_AUDIT.json',result)
    status(BASE/'STATUS.json',dict(status='root_verified_pending_report_publication',time=time.time()))
    print('ROOT_PASS',result,flush=True)

def export():
    verify(inputs=True,labels=True);assert read(BASE/'FINAL_ROOT_AUDIT.json')['status']=='pass'
    runtime=read(BASE/'RUNTIME_LOCK.json');cohort=read(BASE/'COHORT.json')
    write(PUBLIC/'CONFIG.json',dict(version='tastvg_temporal_quality_old8_v1',
        predecessor_commit='cbd7d3b9fabfbb96ffc083f45b7fa5a80768a5ed',design=read(BASE/'DESIGN.json'),
        checkpoint_state_sha256=runtime['checkpoint_state_sha256'],production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        sampling='original_Paper48_observed_grid',confirmation='historical_source_disjoint_within_current_batch',
        source_bootstrap_draws=10000,source_bootstrap_seed=20261003,
        anonymous_source_ordinals={ds:{sp:sorted({c['parent'] for c in cohort['cells'] if c['dataset']==ds and c['split']==sp})
            for sp in SPLITS} for ds in DATASETS},production_unchanged=True))
    write(PUBLIC/'RESOURCES.json',dict(preparation=read(BASE/'PREPARATION.json'),generation=read(BASE/'GENERATION_RESOURCES.json'),
        score=read(BASE/'SCORE_ROOT_CHECKS.json'),root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    # Upstream feature extractors are contract evidence, not vendored code exports.
    pins={f:h for f,h in runtime['pins'].items() if not f.startswith('external/')}
    extra=['scripts/audit_tastvg_temporal_quality_public_v1.py','scripts/report_tastvg_temporal_quality_v1.py']
    for f in extra:pins[f]=sha(ROOT/f)
    write(PUBLIC/'RUNTIME_PROVENANCE.json',dict(pins=pins,
        upstream_contract_pins={f:h for f,h in runtime['pins'].items() if f.startswith('external/')},
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),private_immutable_input_count=len(runtime['inputs']),
        private_assets_not_exported=True,labels_used_for_scoring_only_after_global_prediction_seal=True))
    bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    write(PUBLIC/'BARRIERS.json',dict(global_prediction={k:v for k,v in bar.items() if k!='files'},
        GT_exposure=read(BASE/'GT_EXPOSURE.json'),root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    write(PUBLIC/'ENGINEERING_HISTORY.json',dict(science_changed=False,recoveries=[
        dict(name=f.name,record=read(f/'FAILURE.json'),original_sources_preserved=True) for f in sorted((BASE/'recovery').glob('*'))]))
    files=sorted(set(list(pins)+['docs/TA_TEMPORAL_QUALITY_OLD8_REVIEW.md']+
        [str(f.relative_to(ROOT)) for f in PUBLIC.rglob('*') if f.is_file()]))
    forbidden={'caption','query_text','truth','gt_span','gt_interval','weights','gradients','h','pre_state','post_state'}
    def scan(v):
        if isinstance(v,dict):
            for k,x in v.items():assert k.lower() not in forbidden,k;scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
    for f in PUBLIC.rglob('*.json'):scan(read(f))
    assert all(not any(s in f for s in ['GT_LABELS','.pt','/checkpoints/','external/']) for f in files)
    write(BASE/'PUBLIC_MANIFEST.json',dict(files={f:dict(sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files},
        file_count=len(files),bytes=sum((ROOT/f).stat().st_size for f in files),time=time.time()))
    print('PUBLIC_ALLOWLIST',len(files),flush=True)

if __name__=='__main__':
    command=sys.argv[1];assert command in ['prepare','generate','score','root','export']
    try:globals()[command]()
    except Exception:
        import traceback
        status(BASE/'STATUS.json',dict(status='failed',stage=command,error=traceback.format_exc(),time=time.time()))
        raise

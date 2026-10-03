"""Finite CPU coverage/quality experiment with prediction-before-GT barrier."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_boundary_support_common_v1 import *
import numpy as np
from vg_tta.tastvg_temporal_boundary_support_v1 import expanded,decisions
from vg_tta.tastvg_temporal_qualification_v1 import critic_scores

DIFFS=dict(A_support=('A32','A8'), B_support=('B32','B8'), D_support=('D32','D8'),
    B8_vs_A8=('B8','A8'), D8_vs_A8=('D8','A8'), B32_vs_A32=('B32','A32'),
    D32_vs_A32=('D32','A32'), D32_vs_D8=('D32','D8'))
BASE_FIELDS=[f'{a}_{m}' for a in ARMS for m in ['v','t','gain','t_gain','gross_gain','gross_loss']]
BASE_FIELDS += [f'{name}_{m}' for name in DIFFS for m in ['v','t']]
EXPERT_FIELDS=['O8_v','O32_v','O8_t','O32_t','capacity_gain','t_capacity_gain',
    'old_fast_gain','activation_range','D8_missing_candidates','D32_missing_candidates',
    'D8_selected_both_positive','D32_selected_both_positive']
EXPERT_FIELDS += [f'{a}_{m}' for a in ARMS for m in ['regret','t_regret','changed']]

def prepare():
    import torch
    assert not (BASE/'RUNTIME_LOCK.json').exists()
    assert read(PRIOR/'FINAL_COMPLETION.json')['status']=='completed_and_verified_publication'
    sys.addaudithook(guard);tick=time.monotonic()
    prior=read(PRIOR/'RUNTIME_LOCK.json');bar=read(PRIOR/'GLOBAL_PREDICTION_BARRIER.json')
    inputs=dict(prior['inputs'])
    for f,h in bar['files'].items():inputs[str((PRIOR/f).relative_to(ROOT))]=h
    for i,(f,h) in enumerate(inputs.items(),1):
        assert sha(ROOT/f)==h,f
        if i%1600==0:print('BIND_INPUTS',i,len(inputs),flush=True)
    cells=read(OLD/'COHORT.json')['cells'];assert len(cells)==1152 and sum(c['scheduled'] for c in cells)==288
    write(BASE/'COHORT.json',dict(cells=cells,historical_exposure=True))
    design=dict(version='tastvg_temporal_boundary_support_v1',supports=[8,32],
        append='24_maximin_squared_normalized_physical_endpoints_seeded_by_entire_Old8',
        allocation_ties='lexicographic_start_end_exact',retains_old_A_selection=True,
        scorers=['A_original_UVTG','B_old_inner_outer_alpha025','D_min_start_end_transition'],
        boundary_window_seconds=1.0,inner_clip_to_candidate=True,exterior_clip_to_observed_window=True,
        missing_exterior_transition=0.0,no_positive_gate=True,alpha_B=.25,tie_epsilon=1e-12,
        decision_ties='native_first_retained_candidate_order',
        unavailable_or_constant='same_support_A_selection',spatial_C_removed=True,B_plus_D=False,
        activation='same_PE_pooled_query_cosine_float64_phase0_2Hz_fractional_bins',
        all_arrivals=1152,expert_arrivals=288,corrupt_expert_arrivals=240,clean_expert_arrivals=48,
        params={ds:plan(ds)['params'] for ds in DATASETS},
        checkpoints=prior['checkpoint_state_sha256'],new_model_calls=0,new_expert_calls=0,new_backprop=0,
        new_GPU_jobs=0,confirmation_not_used_to_choose_rule=True,historical_exposure=True,GT_read=False,
        references={'BAM_DETR':'https://arxiv.org/html/2312.00083v2',
            'distinction':'BAM-DETR trains an IoU head; D is an untrained transition proxy, no equivalence'})
    write(BASE/'DESIGN.json',design)
    meta={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'methods/CURRENT_METHOD.json',
        PRIOR/'FINAL_COMPLETION.json',PRIOR/'RUNTIME_LOCK.json',PRIOR/'GLOBAL_PREDICTION_BARRIER.json',
        OLD/'COHORT.json',BASE/'COHORT.json',BASE/'DESIGN.json']}
    for ds in DATASETS:
        f=OLD/ds/'PLAN.json';meta[str(f.relative_to(ROOT))]=sha(f)
        groups={sp:{c['parent'] for c in cells if c['dataset']==ds and c['split']==sp} for sp in SPLITS}
        assert len(groups['search'])==32 and len(groups['confirm'])==16 and not groups['search']&groups['confirm']
    code=['vg_tta/tastvg_temporal_boundary_support_v1.py','scripts/tastvg_boundary_support_common_v1.py',
        'scripts/run_tastvg_boundary_support_v1.py','scripts/test_tastvg_boundary_support_v1.py',
        'scripts/audit_tastvg_boundary_support_public_v1.py','scripts/report_tastvg_boundary_support_v1.py',
        'protocols/tastvg_temporal_boundary_support_v1.md','docs/tastvg_temporal_boundary_support_v1/EXECUTION.md',
        'scripts/tastvg_temporal_quality_common_v1.py','vg_tta/tastvg_temporal_quality_old8_v1.py',
        'vg_tta/tastvg_temporal_qualification_v1.py','vg_tta/tastvg_oracle_event5_v1.py',
        'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py',
        'scripts/tastvg_oracle_event5_common_v1.py','scripts/tastvg_correction_views_common_v1.py',
        'scripts/decota_matrix_common_v1.py']
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in code},inputs=inputs,protected_metadata=meta,
        label_hashes_from_predecessor_receipt=prior['label_hashes_from_predecessor_receipt'],
        checkpoint_state_sha256=prior['checkpoint_state_sha256'],GT_read=False,time=time.time()))
    assert not torch.cuda.is_initialized()
    write(BASE/'PREPARATION.json',dict(status='pass',immutable_inputs=len(inputs),GT_read=False,
        worker_wall_seconds=time.monotonic()-tick,CUDA_initialized=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_pending_unlabelled_generation',GT_read=False,time=time.time()))
    archive('公式/窗口/补池/缺侧规则和输入已锁，尚无新预测评分')
    print('PREPARED',len(inputs),flush=True)

def generate():
    import torch
    torch.set_num_threads(2);sys.addaudithook(guard);verify();tick=time.monotonic();cnt=collections.Counter();files={}
    for at,c in enumerate(read(BASE/'COHORT.json')['cells'],1):
        budget();z=read(prior_prediction(c));a=acell(c);x=donor(c);row=plan(c['dataset'])['rows'][c['parent']]
        ids=row['frame_ids'];assert torch.equal(x['A']['boxes'],a['slow']['boxes'])
        assert z['persistent_pre_sha']==x['persistent_pre_sha']==a['pre_sha']
        assert z['persistent_post_sha']==x['persistent_post_sha']==a['post_sha']
        assert z['A_indices']==a['final_indices'] and a['pixel_sha256']==c['pixel_sha256']
        p={k:z[k] for k in ['cell_key','dataset','split','source_id','condition','order','arrival',
            'expert_scheduled','persistent_pre_sha','persistent_post_sha','A_indices','native_indices']}
        p.update(prior_prediction_sha256=sha(prior_prediction(c)),GT_read=False,new_model_calls=0,new_expert_calls=0)
        if c['scheduled']:
            e,receipt=evidence(c);assert receipt['cache_sha256']==z['expert_receipt']['cache_sha256']
            pool=expanded(z['candidates'],ids);assert pool[:8]==x['temporal_candidates']==a['temporal']['candidates']
            lo,hi=ids[0],ids[-1]+1
            intervals=[[(q['physical_interval'][0]-lo)/(hi-lo),(q['physical_interval'][1]-lo)/(hi-lo)] for q in pool]
            duration=(hi-lo)/row['input']['fps'];assert abs(e['duration']-duration)<1e-9
            p.update(candidates=pool,intervals_normalized=intervals,curve=z['curve'],feature_edges=z['feature_edges'],
                embedding_available=z['embedding_available'],duration_seconds=duration,
                grid_start=[(f-lo)/(hi-lo) for f in ids],grid_end=[(f+1-lo)/(hi-lo) for f in ids],
                feature_cache_sha256=receipt['cache_sha256'],choices={},scores={},details={},fallbacks={})
            for n in [8,32]:
                scores=critic_scores([q['physical_interval'] for q in pool[:n]],e['proposals'],e['proposal_confidence']).tolist()
                ia=int(np.argmax(scores));dd=decisions(p['curve'],p['feature_edges'],intervals[:n],duration,ia,p['embedding_available'])
                p['choices'][f'A{n}']=ia;p['scores'][f'A{n}']=scores
                for name,d in dd.items():
                    arm=f'{name}{n}';p['choices'][arm]=d['selected'];p['scores'][arm]=d['scores']
                    p['details'][arm]=d['details'];p['fallbacks'][arm]=d['fallback_reason']
            assert p['scores']['A8']==z['old_scores'] and p['choices']['A8']==z['old_selected']
            assert p['scores']['B8']==z['quality']['scores'] and p['choices']['B8']==z['quality']['selected']
            assert p['details']['B8']==z['quality']['details']
            assert p['candidates'][p['choices']['A8']]['indices']==z['A_indices']
            cnt['experts']+=1;cnt['A8_B8_exact_parity']+=1;cnt['Old8_and_old_A_retained']+=1
            cnt['D32_missing_side_candidates']+=sum(d['missing_start_context'] or d['missing_end_context'] for d in p['details']['D32'])
        else:cnt['nonexpert_identical_A']+=1
        f=BASE/c['dataset']/'predictions'/f'{prefix(c)}.json';write(f,p);files[str(f.relative_to(BASE))]=sha(f)
        if at%96==0:
            status(BASE/'STATUS.json',dict(status='unlabelled_generation_running',done=at,total=1152,GT_read=False,time=time.time()))
            print('CANDIDATES_AND_SCORES_NO_GT',at,1152,flush=True)
    assert cnt['experts']==cnt['A8_B8_exact_parity']==288 and cnt['nonexpert_identical_A']==864
    assert not torch.cuda.is_initialized();verify()
    write(BASE/'GENERATION_RESOURCES.json',dict(status='pass',counts=dict(cnt),GT_read=False,
        CUDA_initialized=False,new_model_calls=0,new_expert_calls=0,new_backprop=0,
        worker_wall_seconds=time.monotonic()-tick,time=time.time()))
    write(BASE/'GLOBAL_PREDICTION_BARRIER.json',dict(status='sealed',arrivals=1152,expert_arrivals=288,
        files=files,GT_read=False,runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),time=time.time()))
    status(BASE/'STATUS.json',dict(status='sealed_pending_CPU_GT_score',GT_read=False,time=time.time()))
    archive('1152读出/288双候选池A B D全部封存，Old8 A/B逐值复现，待离线GT')

def pair(scores,values):
    wins=[]
    for i in range(len(values)):
        for j in range(i+1,len(values)):
            if abs(values[i]-values[j])<=1e-12:continue
            d=(scores[i]-scores[j])*np.sign(values[i]-values[j])
            wins.append(1. if d>1e-12 else .5 if abs(d)<=1e-12 else 0.)
    return float(np.mean(wins)) if wins else 0.,len(wins)

def summarize(rows):
    out={}
    for group in ['corruption','clean']:
        out[group]={}
        for subset in ['all','expert','nonexpert']:
            rr=[r for r in rows if (r['condition']!='clean')==(group=='corruption') and
                (subset=='all' or r['expert_scheduled']==(subset=='expert'))]
            z=summary(rr,BASE_FIELDS+(EXPERT_FIELDS if subset=='expert' else []));z['arms']={}
            z['comparisons']={name:dict(improved=sum(r[f'{name}_v']>1e-12 for r in rr),
                harmed=sum(r[f'{name}_v']<-1e-12 for r in rr),
                unchanged=sum(abs(r[f'{name}_v'])<=1e-12 for r in rr),
                severe_harm_gt5pp=sum(r[f'{name}_v']<-.05 for r in rr)) for name in DIFFS}
            for arm in ARMS:
                g=f'{arm}_gain'
                az=dict(improved=sum(r[g]>1e-12 for r in rr),harmed=sum(r[g]<-1e-12 for r in rr),
                    unchanged=sum(abs(r[g])<=1e-12 for r in rr),severe_harm_gt5pp=sum(r[g]<-.05 for r in rr),
                    correctness={str(t):dict(correct_to_wrong=sum(r['A8_v']>t and r[f'{arm}_v']<=t for r in rr),
                        wrong_to_correct=sum(r['A8_v']<=t and r[f'{arm}_v']>t for r in rr)) for t in [.3,.5]})
                if subset=='expert':
                    az.update(old_positive_fast=sum(r['old_fast_gain']>1e-12 for r in rr),
                        old_positive_fast_destroyed=sum(r['old_fast_gain']>1e-12 and r[g]<-1e-12 for r in rr))
                z['arms'][arm]=az
            if subset=='expert':
                z['pairwise']={}
                for n in [8,32]:
                    for m in ['v','t']:
                        valid=[r for r in rr if r[f'strict_{n}_{m}_pairs']>0]
                        pp=summary(valid,[f'{a}{n}_pair_{m}' for a in ['A','B','D']])
                        pp['no_strict_pair_cells']=len(rr)-len(valid);z['pairwise'][f'{n}_{m}']=pp
            out[group][subset]=z
    return out

def score():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
    torch.set_num_threads(2);verify(labels=True);bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');tick=time.monotonic()
    assert bar['status']=='sealed' and len(bar['files'])==1152
    write(BASE/'GT_EXPOSURE.json',dict(scope='offline oracle/scoring after every support/score/choice sealed',
        prediction_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),prediction_barrier_time=bar['time'],
        GT_score_started=time.time(),generation_GT_read=False,historical_exposure=True))
    cnt=collections.Counter();err=0.
    for ds in DATASETS:
        pl=plan(ds)
        for sp in SPLITS:
            labels=read(POOL/ds/f'GT_LABELS_{sp}.json');rows=[]
            for c in [c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==sp]:
                f=BASE/ds/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                p=read(f);x=donor(c);row=pl['rows'][c['parent']];g=labels[str(c['parent'])]
                truth={int(k):v for k,v in g['truth'].items()};dense=DenseTube(x['A']['boxes'],row,truth,g['span'],ds=='hc2')
                av=dense.score(x['A']['physical_interval'])
                r={k:p[k] for k in ['dataset','split','source_id','condition','order','arrival','expert_scheduled']}
                r.update(A_state_pre_sha256=p['persistent_pre_sha'],A_state_post_sha256=p['persistent_post_sha'])
                vals=[dense.score(z['physical_interval']) for z in p['candidates']] if c['scheduled'] else []
                for arm in ARMS:
                    v=vals[p['choices'][arm]] if vals else av;gain=v['v']-av['v']
                    r.update({f'{arm}_v':v['v'],f'{arm}_t':v['t'],f'{arm}_gain':gain,
                        f'{arm}_t_gain':v['t']-av['t'],f'{arm}_gross_gain':max(gain,0.),f'{arm}_gross_loss':max(-gain,0.)})
                for name,(a,b) in DIFFS.items():
                    for m in ['v','t']:r[f'{name}_{m}']=r[f'{a}_{m}']-r[f'{b}_{m}']
                if c['scheduled']:
                    vs=[v['v'] for v in vals];ts=[v['t'] for v in vals]
                    assert abs(vs[p['choices']['A8']]-av['v'])<1e-12
                    r.update(candidate_v=vs,candidate_t=ts,candidate_indices=[z['indices'] for z in p['candidates']],
                        choices=p['choices'],scores=p['scores'],details=p['details'],fallbacks=p['fallbacks'],
                        activation_curve=p['curve'],feature_edges=p['feature_edges'],intervals_normalized=p['intervals_normalized'],
                        duration_seconds=p['duration_seconds'],grid_start=p['grid_start'],grid_end=p['grid_end'],
                        embedding_available=p['embedding_available'],feature_cache_sha256=p['feature_cache_sha256'],
                        O8_v=max(vs[:8]),O32_v=max(vs),O8_t=max(ts[:8]),O32_t=max(ts),
                        capacity_gain=max(vs)-max(vs[:8]),t_capacity_gain=max(ts)-max(ts[:8]),
                        old_fast_gain=av['v']-vs[0],activation_range=float(np.ptp(p['curve'])))
                    assert r['capacity_gain']>=0 and r['t_capacity_gain']>=0
                    for n in [8,32]:
                        for m,values in [('v',vs[:n]),('t',ts[:n])]:
                            for a in ['A','B','D']:
                                val,den=pair(p['scores'][f'{a}{n}'],values)
                                r[f'{a}{n}_pair_{m}']=val;r[f'strict_{n}_{m}_pairs']=den
                        dd=p['details'][f'D{n}'];chosen=p['choices'][f'D{n}']
                        r[f'D{n}_missing_candidates']=sum(d['missing_start_context'] or d['missing_end_context'] for d in dd)
                        r[f'D{n}_selected_both_positive']=int(dd[chosen]['both_positive'])
                    for arm in ARMS:
                        n=8 if arm.endswith('8') and arm not in ['A32','B32','D32'] else 32
                        r[f'{arm}_regret']=max(vs[:n])-r[f'{arm}_v']
                        r[f'{arm}_t_regret']=max(ts[:n])-r[f'{arm}_t']
                        r[f'{arm}_changed']=int(p['choices'][arm]!=p['choices']['A8'])
                    measured=list(zip([z['physical_interval'] for z in p['candidates']],vals));cnt['experts']+=1
                else:
                    assert all(r[f'{a}_gain']==r[f'{a}_t_gain']==0 for a in ARMS)
                    measured=[(x['A']['physical_interval'],av)];cnt['nonexperts_identical']+=1
                for interval,v in measured:
                    off=official(x['A']['boxes'],row,truth,g['span'],interval,ds)
                    for m in ['v','t','s']:
                        d=abs(off[m]-v[m]);assert d<1e-10;err=max(err,d);cnt['official_scalars']+=1
                rows.append(r);cnt['arrivals']+=1
                if cnt['arrivals']%96==0:print('CPU_GT_SCORE',cnt['arrivals'],1152,flush=True)
            out=PUBLIC/sp/ds;write(out/'ROWS.json',rows);write(out/'SUMMARY.json',summarize(rows))
            def compact(r):
                ks=['dataset','split','source_id','condition','order','arrival','expert_scheduled']
                return {k:r[k] for k in ks+BASE_FIELDS}
            write(out/'CASES.json',{arm:dict(high=[compact(r) for r in sorted(rows,key=lambda r:-r[f'{arm}_gain'])[:4]],
                low=[compact(r) for r in sorted(rows,key=lambda r:r[f'{arm}_gain'])[:4]]) for arm in ['A32','B32','D8','D32']})
    assert cnt['arrivals']==1152 and cnt['experts']==288 and not torch.cuda.is_initialized()
    verify(labels=True);result=dict(status='pass',checks=dict(cnt),max_official_error=err,
        CUDA_initialized=False,worker_wall_seconds=time.monotonic()-tick,time=time.time())
    write(BASE/'SCORE_CHECKS.json',result);status(BASE/'STATUS.json',dict(status='scored_pending_root_audit',time=time.time()))
    archive('1152到达与双支持oracle已离线计分，待独立根审计/报告/公开')
    print('SCORE_PASS',result,flush=True)

def root():
    import torch
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.audit_tastvg_boundary_support_public_v1 import audit
    torch.set_num_threads(2);tick=time.monotonic();verify(inputs=True,labels=True)
    bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');exp=read(BASE/'GT_EXPOSURE.json')
    assert exp['prediction_barrier_time']<exp['GT_score_started'] and exp['prediction_barrier_sha256']==sha(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    cnt=collections.Counter();err=0.
    for ds in DATASETS:
        pl=plan(ds)
        for sp in SPLITS:
            rows=read(PUBLIC/sp/ds/'ROWS.json');labels=read(POOL/ds/f'GT_LABELS_{sp}.json')
            prev=read(previous.PUBLIC/sp/ds/'ROWS.json');assert len(rows)==len(prev)
            cells=[c for c in read(BASE/'COHORT.json')['cells'] if c['dataset']==ds and c['split']==sp]
            for r,b,c in zip(rows,prev,cells):
                f=BASE/ds/'predictions'/f'{prefix(c)}.json';assert sha(f)==bar['files'][str(f.relative_to(BASE))]
                p=read(f);a=acell(c);x=donor(c);row=pl['rows'][c['parent']];g=labels[str(c['parent'])]
                assert torch.equal(x['A']['boxes'],a['slow']['boxes'])
                assert p['persistent_pre_sha']==a['pre_sha']==r['A_state_pre_sha256']
                assert p['persistent_post_sha']==a['post_sha']==r['A_state_post_sha256']
                assert p['A_indices']==a['final_indices'];cnt['A_states_boxes']+=1
                for field,pf in [('A8_v','A_v'),('A8_t','A_t'),('B8_v','quality_v'),('B8_t','quality_t')]:
                    assert abs(r[field]-b[pf])<1e-12;cnt['predecessor_scalar_parity']+=1
                if not c['scheduled']:
                    assert all(r[f'{arm}_gain']==0 for arm in ARMS);cnt['nonexpert_identical']+=1;continue
                e,receipt=evidence(c);assert receipt['cache_sha256']==p['feature_cache_sha256']
                vv=e['video_features'].double().numpy();q=e['text_features'][:,0].double().numpy()
                qn=np.sqrt((q*q).sum());vn=np.sqrt((vv*vv).sum(0))
                curve=np.sum(vv*q[:,None],0)/(vn*qn) if qn>0 and np.all(vn>0) else np.zeros(vv.shape[1])
                np.testing.assert_allclose(curve,p['curve'],rtol=0,atol=1e-12);cnt['independent_cosine_values']+=len(curve)
                assert p['candidates'][:8]==x['temporal_candidates']==a['temporal']['candidates']
                assert p['scores']['A8']==a['temporal']['scores'] and p['choices']['A8']==a['temporal']['selected']
                assert p['scores']['B8']==b['quality_scores'] and p['choices']['B8']==b['quality_selected']
                truth={int(k):v for k,v in g['truth'].items()}
                for n in [8,32]:
                    scores=critic_scores([z['physical_interval'] for z in p['candidates'][:n]],e['proposals'],e['proposal_confidence']).tolist()
                    assert scores==p['scores'][f'A{n}'] and int(np.argmax(scores))==p['choices'][f'A{n}']
                    cnt['UVTG_score_parity']+=1
                for i,z in enumerate(p['candidates']):
                    off=official(x['A']['boxes'],row,truth,g['span'],z['physical_interval'],ds)
                    for m in ['v','t']:
                        d=abs(off[m]-r[f'candidate_{m}'][i]);assert d<1e-10;err=max(err,d);cnt['official_scalars']+=1
                cnt['support_inclusion_and_prior_parity']+=1
            assert summarize(rows)==read(PUBLIC/sp/ds/'SUMMARY.json');cnt['summary_recomputed']+=1
    public=audit(PUBLIC);assert public['status']=='pass';verify(inputs=True,labels=True)
    result=dict(status='pass',checks=dict(cnt),anonymous_audit=public,max_official_error=err,
        global_seal_before_GT=True,CUDA_initialized=torch.cuda.is_initialized(),
        worker_wall_seconds=time.monotonic()-tick,time=time.time())
    assert cnt['A_states_boxes']==1152 and cnt['support_inclusion_and_prior_parity']==288 and not result['CUDA_initialized']
    write(BASE/'FINAL_ROOT_AUDIT.json',result);status(BASE/'STATUS.json',dict(status='root_verified_pending_report_publication',time=time.time()))
    print('ROOT_PASS',result,flush=True)

def export():
    rt=verify(inputs=True,labels=True);assert read(BASE/'FINAL_ROOT_AUDIT.json')['status']=='pass'
    write(PUBLIC/'CONFIG.json',dict(design=read(BASE/'DESIGN.json'),predecessor_commit='b7b3790ae3697f11c3643fa0260ab4336aba5989',
        checkpoint_state_sha256=rt['checkpoint_state_sha256'],production_method_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'),
        production_unchanged=True,source_bootstrap_draws=10000,source_bootstrap_seed=20261003,
        sampling='original_Paper48_not_Fig1_uniform64',confirmation='historically_exposed_source_disjoint_within_batch'))
    write(PUBLIC/'RESOURCES.json',dict(preparation=read(BASE/'PREPARATION.json'),generation=read(BASE/'GENERATION_RESOURCES.json'),
        score=read(BASE/'SCORE_CHECKS.json'),root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    write(PUBLIC/'RUNTIME_PROVENANCE.json',dict(pins=rt['pins'],runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        private_immutable_input_count=len(rt['inputs']),private_assets_not_exported=True,
        labels_only_after_global_prediction_seal=True,
        engineering_revisions=[read(f) for f in sorted((BASE/'revisions').glob('revision_*.json'))]))
    bar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
    write(PUBLIC/'BARRIERS.json',dict(global_prediction={k:v for k,v in bar.items() if k!='files'},
        GT_exposure=read(BASE/'GT_EXPOSURE.json'),root=read(BASE/'FINAL_ROOT_AUDIT.json')))
    write(PUBLIC/'ENGINEERING_HISTORY.json',dict(recoveries=[dict(name=f.name,record=read(f/'FAILURE.json'))
        for f in sorted((BASE/'recovery').glob('*'))],science_changed=False))
    forbidden={'caption','query_text','truth','gt_span','gt_interval','weights','gradients','h','pre_state','post_state'}
    def scan(v):
        if isinstance(v,dict):
            for k,x in v.items():assert k.lower() not in forbidden,k;scan(x)
        elif isinstance(v,list):
            for x in v:scan(x)
    for f in PUBLIC.rglob('*.json'):scan(read(f))
    files=sorted(set(list(rt['pins'])+['docs/TA_TEMPORAL_BOUNDARY_SUPPORT_REVIEW.md']+
        [str(f.relative_to(ROOT)) for f in PUBLIC.rglob('*') if f.is_file()]))
    assert all(not any(s in f for s in ['GT_LABELS','.pt','/checkpoints/','external/']) for f in files)
    write(BASE/'PUBLIC_MANIFEST.json',dict(files={f:dict(sha256=sha(ROOT/f),bytes=(ROOT/f).stat().st_size) for f in files},
        file_count=len(files),bytes=sum((ROOT/f).stat().st_size for f in files),time=time.time()))
    print('EXPORT_ALLOWLIST',len(files),flush=True)

if __name__=='__main__':
    cmd=sys.argv[1];assert cmd in ['prepare','generate','score','root','export']
    try:globals()[cmd]()
    except Exception:
        import traceback
        status(BASE/'STATUS.json',dict(status='failed',stage=cmd,error=traceback.format_exc(),time=time.time()))
        raise

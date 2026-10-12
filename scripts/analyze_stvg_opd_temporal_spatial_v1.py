"""CPU-only cached factorial assembly/scoring and Q0 mean-readout diagnosis.

No model, optimizer, CUDA or expert calls. Immutable source outputs are bound
before GT exposure; anonymous complete scalar rows are independently reviewed.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections
import gzip
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, load, save

BASE = ROOT/'artifacts/stvg_opd_temporal_spatial_complementarity_v1'
OUT = ROOT/'results/stvg_opd_temporal_spatial_complementarity/2026-10-12'
O = ROOT/'artifacts/stvg_opd_optimization_geometry_v1'
Q = ROOT/'artifacts/stvg_opd_query_observability_q0_v1/Native_support_scope003'
PAPER = ROOT/'artifacts/stvg_opd_paper_hc2_revision_v2'
P6 = PAPER/'P6_existing_temporal_revision001'
SPATIAL = ['Native', 'OPD', 'Direct', 'Capped_Direct']
GROUPS = ['Q_random', 'Q_observable', 'LN_observable']
OWN = ['scripts/analyze_stvg_opd_temporal_spatial_v1.py',
       'scripts/review_stvg_opd_temporal_spatial_v1.py',
       'protocols/stvg_opd_temporal_spatial_complementarity_v1.md']

def array(x):
    return x.detach().cpu().numpy().astype(np.float64) if hasattr(x, 'detach') else np.asarray(x, np.float64)

def rows_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists()
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for r in rows:
                stream.write((json.dumps(r, allow_nan=False, separators=(',', ':'))+'\n').encode())

def rows_read(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]

def verify():
    r = read(BASE/'RUNTIME.json')
    for f, h in r['pins'].items():
        assert sha(ROOT/f) == h, f
    for f, h in r['protected'].items():
        assert sha(ROOT/f) == h, f
    return r

def require_seal():
    b = read(BASE/'GLOBAL_READOUT_BARRIER.json')
    assert b['status'] == 'sealed' and b['cells'] == 192 and b['logical_outputs'] == 1536
    assert not b['GT_read'] and sha(BASE/'RUNTIME.json') == b['runtime_sha256']
    return b

def pair_mean(dp, dm, rp, rm, tau, wp, wm):
    rho = math.tanh((rp-rm)/(2*tau))
    return (wp+wm)*((dp+dm)/2+rho*(dp-dm)/2)

def contracts():
    # Real amplitudes are separately calibrated. Equal-amplitude assumptions
    # would erase the midpoint term and fail this asymmetric fixture.
    dp = np.array([2., 0.]); dm = np.array([-1., 0.]); rp, rm, tau = .3, .2, .25
    v = np.exp(np.array([rp, rm])/tau); v /= v.sum()
    assert np.allclose(pair_mean(dp, dm, rp, rm, tau, *v), v[0]*dp+v[1]*dm, atol=1e-15)
    assert np.linalg.norm(pair_mean(dp, dm, .2, .2, tau, .5, .5)) == .5
    assert safe_ratio(0., 0.) is None and safe_ratio(0., 2.) == 0.
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    row = dict(frame_ids=[2, 4], input=dict(width=10, height=10))
    truth = {i: [0., 0., 10., 10.] for i in range(7)}
    a, ids, iou = dense_metric(np.array([[.5, .5, 1., 1.]]*2), row, truth, [0, 7], [0, 7], 'vidstg')
    assert np.array_equal(iou, [0., 0., 1., 1., 1., 0., 0.]) and a['s'] == 3/7
    assert not (BASE/'GLOBAL_READOUT_BARRIER.json').exists()
    try:
        require_seal()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('preseal GT guard did not reject')
    return dict(status='pass', scope='asymmetric signed-pair algebra/undefined ratios/no-extrapolation/preseal rejection',
                contracts=4, CPU_only=True, GT_read=False, new_model_calls=0)

def prepare():
    import torch
    torch.set_num_threads(2)
    assert not BASE.exists()
    BASE.mkdir(parents=True); OUT.mkdir(parents=True, exist_ok=False)
    assert read(O/'FINAL_COMPLETION.json')['status'] == read(Q/'FINAL_COMPLETION.json')['status'] == 'complete'
    write(BASE/'CPU_CONTRACTS.json', contracts())
    protected = ['methods/CURRENT_METHOD.json', 'scripts/publish_stvg_opd_supplement_v1.py',
        'artifacts/stvg_opd_supplement_v1/actual_root/postseal_continuation006/RUNTIME.json',
        'artifacts/stvg_opd_optimization_geometry_v1/FINAL_COMPLETION.json',
        'artifacts/stvg_opd_query_observability_q0_v1/Native_support_scope003/FINAL_COMPLETION.json',
        'artifacts/stvg_opd_paper_hc2_revision_v2/P6_PREDICTION_BARRIER.json',
        'artifacts/stvg_opd_optimization_geometry_v1/GLOBAL_PREDICTION_BARRIER.json',
        'artifacts/stvg_opd_query_observability_q0_v1/Native_support_scope003/GLOBAL_PREDICTION_BARRIER.json']
    protected = [f for f in protected if (ROOT/f).exists()]
    pins = OWN+['methods/decota_final_simplified_v1/objectives.py', 'methods/decota_final_simplified_v1/config.py',
        'scripts/stvg_opd_legacy_input_bridge003.py', 'vg_tta/tastvg_oracle_event5_v1.py',
        'scripts/review_stvg_opd_revised_p1_v2.py', 'scripts/score_stvg_opd_p1_v1.py']
    write(BASE/'RUNTIME.json', dict(status='locked_cached_CPU_analysis', pins={f: sha(ROOT/f) for f in pins},
        protected={f: sha(ROOT/f) for f in protected}, CPU_only=True, new_GPU_fits=0, new_model_calls=0,
        GT_until_complete_readout_seal=False, historically_exposed=True, bootstrap_replicates=10000,
        seed=20261012, unit='parent', time=time.time()))
    status(BASE/'STATUS.json', dict(status='prepared_inventory_pending', CPU_only=True, GT_read=False, time=time.time()))

def replay_temporal(z):
    from scripts.stvg_opd_p6_existing_common001 import config, check_projection
    from methods.decota_final_simplified_v1.objectives import project, prediction
    c = z['capture']; cfg = config(z['dataset'])
    native = prediction(c['zero']['logits'], c['zero']['boxes'], c['records'], c['frame_ids'])
    assert native['physical_interval'] == z['predictions']['Native']['physical_interval']
    ev = project(c['zero']['logits'], c['zero']['actions'], c['records'], cfg)
    error = check_projection(c['zero']['logits'], c['zero']['actions'], c['records'], ev, cfg)
    intervals = [v['interval'] for v in ev['offsets']]
    interval = [min(v[0] for v in intervals), max(v[1] for v in intervals)]
    assert interval == z['predictions']['ActionnessProjection']['physical_interval']
    assert intervals == z['predictions']['ActionnessProjection']['raw_physical_intervals']
    return dict(T0=native['physical_interval'], T1=interval), error

def inventory():
    verify()
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    p6 = {}; bindings = {}; parents = {}; temporal_checks = 0; maxerr = 0.
    for ds in ['hc2', 'vidstg']:
        for p in sorted((P6/'formal'/ds).glob('*.pt')):
            z = load(p); rc = read(p.with_suffix('.json'))
            assert sha(p) == rc['sha256'] and not z['GT_read'] and not rc['GT_read']
            key = (ds, z['query_ordinal']); assert key not in p6
            p6[key] = z; parents[key] = len(parents)
            intervals, error = replay_temporal(z); maxerr = max(maxerr, error); temporal_checks += 2
            bindings[key] = dict(path=str(p.relative_to(ROOT)), sha256=sha(p), intervals=intervals)
    assert len(p6) == 64
    cohort = read(O/'COMMON_STATE_COHORT.json')['records']; unique = {}; matched = []; missing = []; keys = collections.Counter()
    for r in cohort:
        path = ROOT/r['original_Full']['input_path']
        if str(path) not in unique:
            inp = load(path); n = normalized(inp)
            assert sha(path) == r['original_Full']['input_sha256'] and not inp['GT_read']
            unique[str(path)] = n; keys.update(inp.keys())
        n = unique[str(path)]; key = (r['dataset'], r['query_ordinal']); z = p6.get(key)
        same = z is not None and n['pixel_sha256'] == z['pixel_sha256'] and n['frame_ids'] == z['frame_ids'] \
            and n['source_model_state_sha256'] == z['capture']['source_model_state_sha256'] \
            and n['corruption_spec'] == z['corruption_spec'] and n['interval'] == bindings[key]['intervals']['T0']
        if same:
            assert np.array_equal(array(n['native_boxes']), array(z['predictions']['Native']['boxes']))
            assert not r['development'] and r['condition'] == 'clean' and r['order'] in ['order1','order2']
            matched.append(dict(**r, temporal_binding=bindings[key], parent_id=parents[key]))
        else:
            missing.append(dict(index=r['index'], dataset=r['dataset'], setting=r['setting'], order=r['order'],
                reason='no exact same-input/source temporal capture', input_sha256=r['original_Full']['input_sha256']))
    assert len(cohort) == 1792 and len(unique) == 1536 and len(matched) == 128
    primary = [r for r in matched if r['order'] == 'order1']; assert len(primary) == 64
    design = dict(status='locked_actual_cache_match_before_GT', records=primary+matched,
        primary_records=64, secondary_records=128, complete_O_cells=1792, unique_O_inputs=1536,
        unavailable_O_cells=len(missing), missing=missing, missing_fields=['two-offset frozen temporal logits',
        'two-offset raw actionness logits', 'two-offset physical frame-ID records'],
        spatial_arms=SPATIAL, temporal_arms=['T0','T1'], GT_read=False, new_GPU_fits=0, new_model_calls=0,
        historical_parent_count=64, distinct_extension_parent_count=0, time=time.time())
    write(BASE/'DESIGN_LOCK.json', design)
    write(BASE/'CACHE_SCOPE_READBACK.json', dict(status='pass', scope='all O input inventory and exact P6 source/pixel/IDs/interval/Native-box join',
        cells=1792, unique_inputs=1536, temporal_capture_queries=64, matched_cells=128, primary_cells=64,
        unavailable_cells=1664, actual_temporal_offsets=temporal_checks, maximum_projection_error=maxerr,
        input_schema_field_counts=dict(keys), CPU_only=True, GT_read=False, time=time.time()))

def assemble():
    import torch
    verify(); d = read(BASE/'DESIGN_LOCK.json'); files = {}; source_files = {}; count = 0
    old = read(O/'GLOBAL_PREDICTION_BARRIER.json'); cache = {}
    for index, r in enumerate(d['records']):
        pp = ROOT/r['temporal_binding']['path']
        if str(pp) not in cache: cache[str(pp)] = load(pp)
        p6 = cache[str(pp)]; assert sha(pp) == r['temporal_binding']['sha256']
        inpfile = ROOT/r['original_Full']['input_path']; inp = load(inpfile)
        from scripts.stvg_opd_legacy_input_bridge003 import normalized
        inp = normalized(inp); boxes = dict(Native=inp['native_boxes']); initial = None; fits = {}
        for name, oldarm in [('OPD','Full'),('Direct','Direct'),('Capped_Direct','Direct_capped')]:
            f = O/'common_state'/f"{r['index']:05}"/(oldarm+'.pt'); h = sha(f); rc = read(f.with_suffix('.json'))
            assert h == rc['sha256'] and not rc['GT_read'] and h in old['files'].values()
            z = load(f); fit = z['fit']; assert not z['GT_read'] and z['interval'] == inp['interval']
            if initial is None: initial = fit['initial']
            else: assert all(torch.equal(v,fit['initial'][k]) for k,v in initial.items())
            boxes[name] = fit['final']; fits[name] = dict(path=str(f.relative_to(ROOT)),sha256=h,bytes=f.stat().st_size)
            source_files[str(f.relative_to(ROOT))] = h
        if index < 64:
            assert torch.equal(boxes['OPD'],p6['predictions']['SpatialOPD']['boxes'])
        assert all(v.shape == boxes['Native'].shape == (len(inp['frame_ids']),4) and torch.isfinite(v).all() for v in boxes.values())
        p = BASE/'readouts'/f'{index:05}.pt'
        payload = dict(public_cell=index, panel='P6_primary' if index<64 else 'O_matched_secondary',
            dataset=r['dataset'], query_ordinal=r['query_ordinal'], parent_id=r['parent_id'], order=r['order'],
            frame_ids=inp['frame_ids'], intervals=r['temporal_binding']['intervals'], boxes=boxes,
            temporal_binding=r['temporal_binding'], spatial_bindings=fits, input_binding=r['original_Full'],
            Native_is_frozen_source=True, three_adapted_arms_common_Full_initial=True, GT_read=False)
        save(p,payload); h=sha(p);files[str(p.relative_to(ROOT))]=h
        write(p.with_suffix('.json'),dict(status='accepted_eight_cached_readouts',sha256=h,bytes=p.stat().st_size,
            public_cell=index,outputs=8,CPU_only=True,GT_read=False,time=time.time()))
        source_files[str(pp.relative_to(ROOT))]=sha(pp);source_files[str(inpfile.relative_to(ROOT))]=sha(inpfile)
        count += 8
        status(BASE/'STATUS.json',dict(status='assembling_cached_readouts',done=index+1,total=192,CPU_only=True,GT_read=False,time=time.time()))
    write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='sealed',scope='all available fixed eight-way cached readouts; not unavailable 1664 cells',
        cells=192,logical_outputs=count,distinct_parents=64,primary_cells=64,secondary_cells=128,
        files=files,source_files=source_files,runtime_sha256=sha(BASE/'RUNTIME.json'),
        design_sha256=sha(BASE/'DESIGN_LOCK.json'),CPU_only=True,GT_read=False,new_model_calls=0,new_GPU_fits=0,time=time.time()))

def safe_ratio(a,b):
    return float(a/b) if b>1e-15 else None

def boot(values):
    x=np.asarray(values,float)
    if not len(x):return dict(count=0,mean=None,ci95=None,median=None,min=None,max=None)
    rng=np.random.default_rng(20261012); bs=[]
    for _ in range(100): bs.append(x[rng.integers(len(x),size=(100,len(x)))].mean(1))
    return dict(count=len(x),mean=float(x.mean()),ci95=np.quantile(np.concatenate(bs),[.025,.975]).tolist(),
        median=float(np.median(x)),min=float(x.min()),max=float(x.max()),harm_5pp=int((x<-.05).sum()),
        harm_20pp=int((x<-.2).sum()),positive=int((x>1e-12).sum()),negative=int((x< -1e-12).sum()))

def summarize(rows):
    result={}
    for panel in ['P6_primary','O_matched_secondary']:
        result[panel]={}
        for ds in ['hc2','vidstg']:
            rr=[r for r in rows if r['panel']==panel and r['dataset']==ds]; grouped=collections.defaultdict(list)
            for r in rr:grouped[r['parent_id']].append(r)
            fields=[k for k,v in rr[0].items() if isinstance(v,(int,float)) and k not in ['public_cell','parent_id']]
            result[panel][ds]=dict(cells=len(rr),parents=len(grouped),metrics={f:boot([np.mean([r[f] for r in v]) for v in grouped.values()]) for f in fields})
    return result

def score():
    verify(); b=require_seal()
    from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate
    activate()
    from scripts.score_stvg_opd_p1_v1 import truths
    from scripts.run_decota_paper_main_v1 import read_row
    from vg_tta.tastvg_oracle_event5_v1 import official
    from scripts.review_stvg_opd_revised_p1_v2 import dense_metric
    d=read(BASE/'DESIGN_LOCK.json'); write(BASE/'GT_EXPOSURE.json',dict(status='postseal_offline',
        seal_sha256=sha(BASE/'GLOBAL_READOUT_BARRIER.json'),historical_evaluation_exposure=True,GT_for_branch_selection=False,time=time.time()))
    gt={};spans={}
    for ds in ['hc2','vidstg']:
        gt[ds],spans[ds],_=truths(ds,dict(parents=sorted({r['query_ordinal'] for r in d['records'] if r['dataset']==ds})))
    prior={(r['dataset'],r['query_ordinal']):r for r in read(ROOT/'results/stvg_opd_paper_hc2_revision/2026-10-08/P6/ROWS.json')}
    rows=[];checks=0;controls=0
    for index,r in enumerate(d['records']):
        p=BASE/'readouts'/f'{index:05}.pt';assert sha(p)==b['files'][str(p.relative_to(ROOT))]
        z=load(p);ds=z['dataset'];q=z['query_ordinal'];raw=read_row(ds,q);truth=gt[ds][q];span=spans[ds][q]
        row={k:z[k] for k in ['public_cell','panel','dataset','parent_id','order']}
        for name,boxes in z['boxes'].items():
            spatial=None
            for t,interval in z['intervals'].items():
                values=official(array(boxes),raw,truth,span,interval,ds)
                second,ids,iou=dense_metric(array(boxes),raw,truth,span,interval,ds)
                assert all(abs(values[k]-second[k])<2e-10 for k in values); checks+=3
                if spatial is None:spatial=values['s']
                else:assert spatial==values['s']
                for k,v in values.items():row[f'{name}_{t}_{k}']=v
                for threshold in [.3,.5]:row[f'{name}_{t}_R{int(threshold*100)}']=float(values['v']>threshold)
            for metric in ['v','t','s','R30','R50']:
                row[f'delta_T_{name}_{metric}']=row[f'{name}_T1_{metric}']-row[f'{name}_T0_{metric}']
        for t in ['T0','T1']:
            assert len({row[f'{n}_{t}_t'] for n in SPATIAL})==1
            for comp in ['Direct','Capped_Direct','Native']:
                for metric in ['v','t','s','R30','R50']:
                    row[f'OPD_minus_{comp}_{t}_{metric}']=row[f'OPD_{t}_{metric}']-row[f'{comp}_{t}_{metric}']
        for metric in ['v','t','s','R30','R50']:
            row[f'interaction_OPD_Native_{metric}']=row[f'delta_T_OPD_{metric}']-row[f'delta_T_Native_{metric}']
        ids=np.array(sorted(truth));event=(ids>=span[0])&(ids<span[1]);a,z0=z['intervals']['T0'];c,e=z['intervals']['T1']
        m0=event&(ids>=a)&(ids<z0);m1=event&(ids>=c)&(ids<e);support=(ids>=raw['frame_ids'][0])&(ids<=raw['frame_ids'][-1])
        for k,mask in [('T0_event_frames',m0),('T1_event_frames',m1),('gained_outside_Native_frames',m1&~m0),
                       ('lost_event_frames',m0&~m1),('gained_supported_frames',m1&~m0&support),('gained_unsupported_frames',m1&~m0&~support)]:row[k]=int(mask.sum())
        row['annotated_event_frames']=int(event.sum());row['T0_coverage']=float(m0.sum()/event.sum());row['T1_coverage']=float(m1.sum()/event.sum());row['delta_coverage']=row['T1_coverage']-row['T0_coverage']
        if index<64:
            old=prior[ds,q]
            for name,t,key in [('Native','T0','Native'),('Native','T1','ActionnessProjection'),('OPD','T0','SpatialOPD')]:
                for metric in ['v','t','s']:assert abs(row[f'{name}_{t}_{metric}']-old[f'{key}_{metric}'])<2e-10;controls+=1
        rows.append(row)
        status(BASE/'CPU_STATUS.json',dict(status='actual_offline_factorial_scoring',done=index+1,total=192,CPU_only=True,GT_read=True,time=time.time()))
    rows_write(OUT/'ROWS.jsonl.gz',rows);write(OUT/'SUMMARY.json',summarize(rows))
    write(BASE/'CPU_COMPLETION.json',dict(status='pending_actual_independent_root_PDF_view_publication_archive',
        cells=192,logical_outputs=1536,distinct_parents=64,official_independent_dense_checks=checks,
        original_P6_scalar_controls=controls,files={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'ROWS.jsonl.gz',OUT/'SUMMARY.json']},
        CPU_only=True,GT_read=True,new_model_calls=0,new_GPU_fits=0,time=time.time()))

def diagnose_q0():
    verify(); assert read(Q/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
    qr=read(Q/'DESIGN_LOCK.json')['records'];rows=[];pairs=[];checks=0;maxerr=0.;total=0
    for r in qr:
        p=Q/'formal'/f"{r['index']:05}.pt";rc=read(p.with_suffix('.json'));assert sha(p)==rc['sha256']
        ep=load(p)['episode'];pos=ep['native_positions'];tau=ep['tau'];before=array(ep['before'])
        for name in GROUPS:
            g=ep['groups'][name];cand=g['candidates'];rewards=np.asarray(g['rewards'],float);w=np.asarray(g['weights'],float)
            expected=np.exp((rewards-rewards.max())/tau);expected/=expected.sum();assert np.allclose(w,expected,rtol=0,atol=3e-15)
            deltas=np.stack([array(v['delta']) for v in cand]);weighted=w@deltas
            assert np.array_equal(weighted.astype(np.float32),g['mean_delta'])
            reconstructed=np.zeros(1792);pair_by=collections.defaultdict(dict)
            for c in cand:
                if c['sign']:pair_by[c['direction']][c['sign']]=c
                else:assert np.count_nonzero(c['delta'])==0
            for direction,c in pair_by.items():
                assert set(c)=={-1,1};cp,cm=c[1],c[-1];ip,im=cp['index'],cm['index'];dp,dm=array(cp['delta']),array(cm['delta'])
                v=pair_mean(dp,dm,rewards[ip],rewards[im],tau,w[ip],w[im]);reconstructed+=v
                err=float(np.max(np.abs(v-(w[ip]*dp+w[im]*dm))));assert err<1e-12;maxerr=max(maxerr,err)
                pairs.append(dict(dataset=r['dataset'],parent_id=r['index'],group=name,direction=direction,
                    reward_margin_over_tau=float((rewards[ip]-rewards[im])/tau),absolute_margin_over_tau=float(abs(rewards[ip]-rewards[im])/tau),
                    tanh_retention=float(abs(math.tanh((rewards[ip]-rewards[im])/(2*tau)))),
                    pair_mass=float(w[ip]+w[im]),equal_amplitudes=bool(cp['amplitude']==cm['amplitude']),
                    midpoint_norm=float(np.linalg.norm((dp+dm)/2)),pair_mean_norm=float(np.linalg.norm(v)),
                    amplitude_positive=float(cp['amplitude']),amplitude_negative=float(cm['amplitude'])))
            assert np.allclose(reconstructed,weighted,rtol=0,atol=1e-12); checks+=len(pair_by)*1792+1792;total+=len(cand)
            means=array(g['mean_boxes']);boxmean=np.einsum('i,ijk->jk',w,np.stack([array(c['boxes']) for c in cand]))
            assert np.allclose(boxmean,array(g['box_average_for_Jensen_diagnostic']),rtol=0,atol=3e-15)
            j=means-boxmean; native_j=float(np.sqrt(np.mean(j[pos]**2)));full_j=float(np.sqrt(np.mean(j**2)))
            assert abs(native_j-g['Jensen_RMS'])<3e-9
            mean_rms=float(np.sqrt(np.mean((means-before)[pos]**2)))
            assert abs(mean_rms-g['mean_output_RMS'])<3e-9
            top=g['expert_top1_index'];norms=np.linalg.norm(deltas,axis=1);norm=float(np.linalg.norm(array(g['mean_delta'])))
            row=dict(dataset=r['dataset'],parent_id=r['index'],group=name,candidates=len(cand),pairs=len(pair_by),tau=float(tau),
                mean_norm=norm,weighted_absolute_norm=float(w@norms),cancellation_ratio=safe_ratio(norm,float(w@norms)),
                mean_over_top1_norm=safe_ratio(norm,float(norms[top])),mean_over_max_norm=safe_ratio(norm,float(norms.max())),
                centre_weight=float(sum(w[c['index']] for c in cand if not c['sign'])),
                native_Jensen_RMS=native_j,full_decoder_Jensen_RMS=full_j,
                Jensen_over_budget=safe_ratio(native_j,float(ep['output_RMS_budget'])),
                mean_native_RMS=mean_rms,boxmean_native_RMS=float(np.sqrt(np.mean((boxmean-before)[pos]**2))),
                mean_over_top1_output_RMS=safe_ratio(mean_rms,float(cand[top]['output_RMS'])),
                mean_over_max_output_RMS=safe_ratio(mean_rms,max(float(c['output_RMS']) for c in cand)),
                mean_displacement_float32_rounding_max=float(np.max(np.abs(weighted-array(g['mean_delta'])))),
                attained_signed_candidates=sum(bool(c['budget_attained']) for c in cand if c['sign']),signed_candidates=sum(bool(c['sign']) for c in cand))
            rows.append(row)
    assert len(rows)==192 and total==2808 and len(pairs)==1308
    rows_write(OUT/'Q0_GROUP_ROWS.jsonl.gz',rows);rows_write(OUT/'Q0_PAIR_ROWS.jsonl.gz',pairs)
    summary={}
    for ds in ['hc2','vidstg']:
        summary[ds]={}
        for name in GROUPS:
            rr=[r for r in rows if r['dataset']==ds and r['group']==name];pr=[r for r in pairs if r['dataset']==ds and r['group']==name]
            fields=[k for k,v in rr[0].items() if k not in ['parent_id'] and isinstance(v,(int,float))]
            # None-valued first rows still require inclusion of these quantities.
            fields=sorted(set(fields)|{'cancellation_ratio','mean_over_top1_norm','mean_over_max_norm','mean_over_top1_output_RMS','mean_over_max_output_RMS'})
            summary[ds][name]=dict(groups=len(rr),pairs=len(pr),statistics={f:dict(**boot([r[f] for r in rr if r[f] is not None]),undefined=sum(r[f] is None for r in rr)) for f in fields},
                pair_statistics={f:dict(count=len(pr),mean=float(np.mean([p[f] for p in pr])),median=float(np.median([p[f] for p in pr])),
                    p05_p95=np.quantile([p[f] for p in pr],[.05,.95]).tolist()) for f in ['absolute_margin_over_tau','tanh_retention']},
                unequal_amplitude_pairs=sum(not p['equal_amplitudes'] for p in pr))
    write(OUT/'Q0_DIAGNOSTIC_SUMMARY.json',summary)
    write(BASE/'Q0_CANCELLATION_READBACK.json',dict(status='pass',scope='all corrected64 Q0 signed-pair algebra/softmax/true mean/Jensen; not causal identification',
        queries=64,groups=192,pairs=len(pairs),candidates=total,coordinate_checks=checks,maximum_pair_error=maxerr,
        CPU_only=True,GT_read=False,new_model_calls=0,new_GPU_fits=0,new_parameter_choices=0,time=time.time()))

if __name__ == '__main__':
    import torch
    torch.set_num_threads(2)
    action=sys.argv[1]
    {'prepare':prepare,'inventory':inventory,'assemble':assemble,'score':score,'q0':diagnose_q0}[action]()
    print('CPU_STAGE_COMPLETE',action,flush=True)

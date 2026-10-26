"""Isolate temporal ROI appearance with one fixed spatial geometry; no new GT."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, json, time, hashlib, argparse, csv, shutil, traceback
from pathlib import Path
from functools import lru_cache
R = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(R), str(R/'tools'), str(R/'studies/shared_corrected_roi_v1')]
import shared_common as selected
from shared_common import read, sha, atomic, identity, binding, RUN, PROVIDER
S = Path(__file__).parent
D = R/'outputs/temporal_pooling_isolation_v1'
P = R/'results/temporal_pooling_isolation_v1'
UP = R/'outputs/shared_corrected_roi_sync_v1'
PUB = R/'results/shared_corrected_roi_sync_v1/spatial'
COUNTS = {'hc2': 1902, 'vidstg': 10303}
ALIASES = {'SourceBoxAppearance': 'AuxSourceROI', 'CorrectedBoxAppearance': 'FullMAP'}
fixed = lru_cache(None)(read)

def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()

def array_sha(a):
    import numpy as np
    a = np.asarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def live(w):
    try:
        s = Path(f'/proc/{w["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
        return s[0] != 'Z' and int(s[19]) == w['startticks']
    except FileNotFoundError:
        return False

def other_cpu():
    roots = [(R/'outputs/shared_corrected_roi_CG_v1', ['hc2_CPU_child.json', 'vidstg_CPU_child.json']),
             (R/'outputs/review_gamma_full_v1', ['hc2_CPU_child.json', 'vidstg_CPU_child.json'])]
    return sum(live(read(d/f)) for d, fs in roots for f in fs if (d/f).exists())

def resources(start):
    assert time.perf_counter()-start < 259200
    while other_cpu() >= 2:
        assert time.perf_counter()-start < 259200
        time.sleep(5)
    assert shutil.disk_usage(R).free >= 80*2**30
    assert next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')) >= 8*2**30
    assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file()) < 160*2**30

def guard():
    paths = {(PROVIDER/j['labels']).resolve() for j in read(RUN/'plan.json')['jobs'].values()}
    def no_labels(event, args):
        if event == 'open' and args and isinstance(args[0], (str, bytes, Path)) and Path(os.fsdecode(args[0])).resolve() in paths:
            raise PermissionError('Only previously audited scalar scores; no GT reads here')
    sys.addaudithook(no_labels)

def validate():
    p = read(D/'plan.json')
    assert p['fingerprint'] == digest({k:v for k,v in p.items() if k != 'fingerprint'})
    for field in ['code_sha256', 'input_sha256']:
        for f,h in p[field].items():
            assert sha(R/f) == h, f
    selected.validate()
    return p

def upstream(ds):
    sp = UP/f'spatial_{ds}_sealed.json'
    sc = UP/f'spatial_{ds}_score_receipt.json'
    au = UP/f'spatial_{ds}_reporting_receipt.json'
    se, score, audit = fixed(sp), fixed(sc), fixed(au)
    assert se['queries'] == COUNTS[ds] and not se['GT_read']
    assert score['status'] == 'scored' and score['seal_sha256'] == sha(sp)
    assert score['whole_domain_sealed_before_GT'] and score['queries'] == COUNTS[ds]
    assert audit['status'] == 'verified_whole_domain' and audit['score_sha256'] == sha(sc)
    assert audit['all_arrays_cold_rederived_exact'] and audit['all_means_recomputed']
    assert read(UP/f'spatial_{ds}_domain_exit.json')['code'] == 0
    return se

def compute(ds, row):
    import numpy as np
    from stvg.spatial import SpatialGeometry, evidence_dp
    from stvg.prototype import one_step
    from stvg.current_shared_roi import map_interval
    key = row['id']
    root = selected.PREV/'spatial'
    e = fixed(root/f'{ds}_sealed.json')['files'][key]
    c, proof = selected.checked_payload(root, e)
    assert e['input_binding'] == binding(row, fixed(RUN/'plan.json')['provider_runtime_fingerprint'])
    assert c['parent_capture_sha256'] == fixed(RUN/ds/'sealed.json')['files'][key]['capture_sha256']
    parent = upstream(ds)['files'][key]
    for field,h in [('path','sha256'),('receipt_path','receipt_sha256')]:
        assert sha(UP/parent[field]) == parent[h]
    g = SpatialGeometry.from_cached_geometry(c['geometries']['main'])
    before = digest({k:array_sha(v) for k,v in g.state_dict().items()})
    out = {}
    evidence = {}
    with np.load(UP/parent['path'], allow_pickle=False) as z:
        assert np.array_equal(c['ids'], z['frame_ids']) and np.array_equal(c['pos'], z['positions8'])
        for name,old in ALIASES.items():
            feature = c['features']['AuxSourceROI' if name == 'SourceBoxAppearance' else 'AuxE_ROI']
            q, meta = one_step(feature, c['q0'], kappa=0, beta=1)
            iv = np.asarray(map_interval(c['ids'], q))
            boxes = evidence_dp(g.for_interval(iv.tolist()))['final'].numpy()
            assert np.array_equal(iv, z[old+'_interval']) and np.array_equal(boxes, z[old])
            assert meta['kappa'] == 0 and meta['beta'] == 1 and meta['rounds'] == 1
            out[name] = boxes
            out[name+'_interval'] = iv
            out[name+'_posterior'] = q
            evidence[name] = dict(features_sha256=array_sha(feature), posterior_sha256=array_sha(q),
                                  local_evidence_sha256=array_sha(meta['L']),
                                  event_prototype_sha256=array_sha(meta['prototypes']['event']),
                                  background_prototype_sha256=array_sha(meta['prototypes']['background']),
                                  interval_sha256=array_sha(iv), boxes_sha256=array_sha(boxes))
    assert before == digest({k:array_sha(v) for k,v in g.state_dict().items()}), 'No branch may mutate spatial candidates/weights'
    ce = fixed(selected.D/f'components_{ds}_sealed.json')['files'][key]
    assert sha(selected.D/ce['path']) == ce['sha256'] and sha(selected.D/ce['receipt_path']) == ce['receipt_sha256']
    with np.load(selected.D/ce['path'], allow_pickle=False) as z:
        assert np.array_equal(out['CorrectedBoxAppearance'], z['FullMAP'])
        assert np.array_equal(out['CorrectedBoxAppearance_interval'], z['FullMAP_interval'])
    proof.update(structural_prediction_sha256=parent['sha256'], structural_receipt_sha256=parent['receipt_sha256'],
                 current_selected_prediction_sha256=ce['sha256'], current_selected_receipt_sha256=ce['receipt_sha256'],
                 fixed_main_geometry_sha256=before, original_shared_pool_context_scale=1.5,
                 same_q0_sha256=array_sha(c['q0']), evidence=evidence,
                 shared_spatial_geometry_not_uncorrected=True, final_DP_recomputed_for_each_own_interval=True)
    return out, proof

def prepare():
    assert not (D/'plan.json').exists()
    selected.validate()
    files = [Path(__file__), R/'studies/shared_corrected_roi_v1/shared_common.py',
             R/'studies/shared_corrected_roi_sync_v1/sync_core.py', R/'studies/map_paper_completion_v1/spatial_core.py',
             R/'studies/no_shrink_main_v1/ns_common.py', R/'tools/full_score.py', *sorted((R/'stvg').glob('*.py'))]
    inputs = [R/'CURRENT_METHOD_v2.json', R/'profiles/shared_corrected_roi_v1.json', UP/'plan.json',
              selected.PREV/'raw_reuse_index.json', RUN/'plan.json']
    for ds in COUNTS:
        upstream(ds)
        inputs += [UP/f'spatial_{ds}_{v}.json' for v in ['sealed','score_receipt','reporting_receipt','root_export_receipt']]
        inputs += [UP/'spatial'/ds/'metrics.json', PUB/ds/'report.json', RUN/ds/'sealed.json',
                   selected.PREV/'spatial'/f'{ds}_sealed.json', selected.D/f'components_{ds}_sealed.json']
    p = dict(schema='temporal_pooling_region_isolation_v1', authorization='Latest direct user: 第一优先 only Source-box vs corrected-box appearance; both full target sets.',
             cohorts=COUNTS, arms=ALIASES,
             changed_factor='ONLY temporal appearance ROI: native Source b0 versus shared corrected b_init. Same saved pretext native-grid source features and 1.5 context, feature centering/unit normalization, one all-video prototype pair, kappa0/beta1/one posterior update, identical q0.',
             fixed_spatial='Identical cached MAIN SpatialGeometry for both arms; identical E, expert eight, candidate boxes/availability/appearance+box unary/link weights and final DP parameters. Never NoSharedCorrection/uncorrected geometry. Each arm fits its own prototypes/posterior/legal jointMAP interval, then runs final DP on its own interval.',
             reuse='The exact requested two-arm experiment already exists as AuxSourceROI vs FullMAP in BOTH whole-domain sealed/official-scored/independently all-array-audited selected structural runs. This independent namespace verifies the strict numerical mapping and complete chain per query; does not rename a different control or relaunch Source/teacher/GT scoring.',
             provenance='Original spatial collector used the SAME extract(raw, boxes) for Source/E ROI features on one native grid; all prior SHA/audits preserved. Four actual retained grids re-pool both regions and repeat posterior/own-interval/finalDP exactly; full verification uses their audited sufficient features, not a claim that disposed raw grids are still available.',
             metrics='Primary tIoU; vIoU/R.3/R.5. Query-equal paired differences, parent-video cluster 10000 bootstrap seed20261022, pointwise95CI. Short<=5s/long and all signed/severe>5pp t/v union once/rescues. sIoU/boundaries internal audit, omitted from compact main table.',
             interpretation='Historically exposed full targets, no tuning or equivalence claims. Zero-crossing t/v CI does not support stable temporal localization gain from shared correction. Limit wording to object-region appearance supports temporal correction; no unverified causal benefit of corrected ROI.',
             execution='CPU-only first priority, single consumer CPU14-15/nice19/idleIO/1thread and at most2 jobs including CG/gamma. CG GPU untouched. Corruption paused. No extra model/Source/expert/optimizer/GT calls.',
             code_sha256={str(f.relative_to(R)):sha(f) for f in files}, input_sha256={str(f.relative_to(R)):sha(f) for f in inputs})
    p['fingerprint'] = digest(p)
    atomic(D/'plan.json', p)
    atomic(P/'protocol.json', {k:v for k,v in p.items() if k != 'input_sha256'})
    for f in files:
        q = D/'source_snapshot'/f.relative_to(R)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(f.read_bytes())
    print(p['fingerprint'], flush=True)

def preflight():
    import torch, numpy as np
    from stvg.representation import extract
    from stvg.spatial import SpatialGeometry, evidence_dp
    from stvg.prototype import one_step
    from stvg.current_shared_roi import map_interval
    validate(); guard(); torch.set_num_threads(1)
    cases = []
    reuse = fixed(selected.PREV/'raw_reuse_index.json')
    for ds in COUNTS:
        records = {r['id']:r for r in selected.records(ds)}
        for key,e in list(reuse[ds].items())[:2]:
            row = records[key]
            a,proof = compute(ds,row); b,again = compute(ds,row)
            assert proof == again and all(np.array_equal(a[k],b[k]) for k in a)
            assert sha(e['path']) == e['sha256']
            raw = torch.load(e['path'], map_location='cpu', weights_only=False, mmap=True)
            ce = fixed(selected.PREV/'spatial'/f'{ds}_sealed.json')['files'][key]
            c,_ = selected.checked_payload(selected.PREV/'spatial',ce)
            raw = {**raw, 'frame_ids':c['ids']}
            for name,field,region in [('SourceBoxAppearance','AuxSourceROI','Source'),('CorrectedBoxAppearance','AuxE_ROI','E')]:
                f,_ = extract(raw,c[region]); f2,_ = extract(raw,c[region])
                assert np.array_equal(f,f2) and np.array_equal(f,c['features'][field])
                q,_ = one_step(f,c['q0'],kappa=0,beta=1); iv = map_interval(c['ids'],q)
                g = SpatialGeometry.from_cached_geometry(c['geometries']['main'])
                boxes = evidence_dp(g.for_interval(iv))['final'].numpy()
                assert np.array_equal(q,a[name+'_posterior']) and np.array_equal(boxes,a[name])
            cases.append(dict(dataset=ds,query_sha256=hashlib.sha256(key.encode()).hexdigest(),raw_sha256=e['sha256'],
                              both_actual_ROI_repool_exact=True, own_posterior_interval_finalDP_repeat_exact=True,
                              same_MAIN_spatial_geometry=True, existing_official_predictions_bitwise=True))
    assert not torch.cuda.is_initialized()
    rec = dict(status='passed_four_real_raw_grid_cases',cases=cases,plan_sha256=sha(D/'plan.json'),
               new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0)
    atomic(D/'preflight_receipt.json',rec); atomic(P/'preflight_receipt.json',rec)
    print(rec['status'],flush=True)

def audit(ds):
    import torch, numpy as np
    p = validate(); guard(); torch.set_num_threads(1)
    assert read(D/'preflight_receipt.json')['status'] == 'passed_four_real_raw_grid_cases'
    assert not (D/ds/'worker.json').exists()
    who = identity(); atomic(D/ds/'worker.json',who)
    start = time.perf_counter(); files = {}; changed = dict(features=0,posterior=0,interval=0,boxes=0)
    rows = selected.records(ds)
    try:
        for i,row in enumerate(rows,1):
            resources(start); tick = time.perf_counter()
            a,proof = compute(ds,row)
            for what,field in [('features','features_sha256'),('posterior','posterior_sha256'),('interval','interval_sha256'),('boxes','boxes_sha256')]:
                changed[what] += int(proof['evidence']['SourceBoxAppearance'][field] != proof['evidence']['CorrectedBoxAppearance'][field])
            rp = D/ds/'receipts'/(row['id']+'.json')
            rec = dict(proof=proof,worker=who,GT_read=False,new_Source=0,new_expert=0,
                       actual_cached_chain_rederive_seconds=time.perf_counter()-tick,
                       array_hashes={k:array_sha(v) for k,v in a.items()},fingerprint=p['fingerprint'])
            assert not rp.exists(); atomic(rp,rec)
            files[row['id']] = dict(receipt_path=str(rp.relative_to(D)),receipt_sha256=sha(rp))
            if i%25 == 0 or i == len(rows):
                atomic(D/ds/'progress.json',dict(completed=i,total=len(rows),worker=who,seconds=time.perf_counter()-start,GT_read=False))
                print(ds,i,len(rows),flush=True)
        assert len(files) == COUNTS[ds]
        seal = dict(status='verified_full_two_ROI_chain_exact_to_official_scored_predictions',queries=len(files),files=files,
                    changed_query_counts=changed,worker=who,seconds=time.perf_counter()-start,
                    upstream_seal_sha256=sha(UP/f'spatial_{ds}_sealed.json'),upstream_audit_sha256=sha(UP/f'spatial_{ds}_reporting_receipt.json'),
                    reused_official_score_sha256=sha(UP/f'spatial_{ds}_score_receipt.json'),
                    same_spatial_geometry_and_params=True,both_own_prototype_posterior_interval_finalDP_recomputed=True,
                    all_prediction_receipt_compact_SHA_checked=True,GT_read=False,new_GT_scoring=0,new_model=0,new_Source=0,new_expert=0)
        atomic(D/ds/'sealed.json',seal)
    except BaseException:
        atomic(D/ds/'failure.json',dict(error=traceback.format_exc(),worker=who,preserve_no_retry=True))
        raise

def export(ds):
    import numpy as np
    from full_score import cluster
    p = validate(); guard(); start = time.perf_counter()
    seal = read(D/ds/'sealed.json'); assert seal['status'] == 'verified_full_two_ROI_chain_exact_to_official_scored_predictions'
    assert seal['queries'] == COUNTS[ds] and not live(seal['worker']), 'Root export only after own replay worker exits'
    rr = read(UP/'spatial'/ds/'metrics.json')['private_rows']
    assert len(rr) == COUNTS[ds] and set(seal['files']) == {r['id'] for r in rr}
    original = read(PUB/ds/'report.json')
    rows = []
    for r in rr:
        e = seal['files'][r['id']]; assert sha(D/e['receipt_path']) == e['receipt_sha256']
        rows.append({**r,'values':{a:r['values'][b] for a,b in ALIASES.items()}})
    means = {a:{k:float(np.mean([r['values'][a][k] for r in rows if r['values'][a][k] is not None])) if any(r['values'][a][k] is not None for r in rows) else None for k in rows[0]['values'][a]} for a in ALIASES}
    for a,b in ALIASES.items():
        for k,v in means[a].items():
            old = original['means'][b][k]
            assert (v is None and old is None) or abs(v-old)<1e-12
    fields = [('t',100),('v',100),('vIoU_ge_03',100),('vIoU_ge_05',100),('observed_actual_sIoU',100),('unobserved_actual_sIoU',100),('boundary_MAE_seconds',1)]
    def stats(group):
        out = cluster(group,'CorrectedBoxAppearance','SourceBoxAppearance',fields)
        dt=np.array([100*(r['values']['CorrectedBoxAppearance']['t']-r['values']['SourceBoxAppearance']['t']) for r in group]);dv=np.array([100*(r['values']['CorrectedBoxAppearance']['v']-r['values']['SourceBoxAppearance']['v']) for r in group])
        return dict(queries=len(group),parents=len({r['parent'] for r in group}),paired=out,
                    severe_t=int((dt < -5).sum()),severe_v=int((dv < -5).sum()),severe_union_once=int(((dt < -5)|(dv < -5)).sum()),rescue_union_once=int(((dt > 5)|(dv > 5)).sum()))
    result = stats(rows)
    for k in ['t','v']:
        assert result['paired'][k] == original['paired']['FullMAP-AuxSourceROI'][k]
    result.update(dataset=ds,status='complete_full_audited_ROI_isolation',means=means,
                  timegroups={g:stats([r for r in rows if r['duration_group']==g]) for g in sorted({r['duration_group'] for r in rows})},
                  changed_query_counts=seal['changed_query_counts'],aggregation='query equal; paired parent-video cluster bootstrap 10000 seed20261022; pointwise95CI',
                  inherited_official_checks=read(UP/f'spatial_{ds}_score_receipt.json')['checks'],
                  scientific_scope='Single-factor temporal ROI change; identical spatial candidates/weights, independently updated temporal prototypes/posterior/interval and interval-specific final DP. Already exposed full sets, no tuning or unseen validation.',
                  interpretation='Both tIoU and vIoU paired95CIs cross zero; no evidence for a stable average temporal benefit of shared-corrected versus original Source ROI. No equivalence or no-effect proof.',
                  plan_sha256=sha(D/'plan.json'),new_full_chain_audit_seal_sha256=sha(D/ds/'sealed.json'))
    folder=P/ds; atomic(folder/'report.json',result)
    with (folder/'compact_table.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['Temporal pooling region','tIoU_percent','vIoU_percent','R03_percent','R05_percent'])
        for a,m in means.items():w.writerow([a,*[100*m[k] for k in ['t','v','vIoU_ge_03','vIoU_ge_05']]])
    parents={k:f'P{i:04d}' for i,k in enumerate(sorted({r['parent'] for r in rows}),1)}
    with (folder/'all_signed_queries.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');keys=['t','v','vIoU_ge_03','vIoU_ge_05']
        w.writerow(['query','parent','duration_group',*[a+'_'+k for a in ALIASES for k in keys],'delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','rescue_union_once'])
        for i,r in enumerate(rows,1):
            dt=100*(r['values']['CorrectedBoxAppearance']['t']-r['values']['SourceBoxAppearance']['t']);dv=100*(r['values']['CorrectedBoxAppearance']['v']-r['values']['SourceBoxAppearance']['v'])
            w.writerow([f'{ds[0].upper()}{i:05d}',parents[r['parent']],r['duration_group'],*[r['values'][a][k] for a in ALIASES for k in keys],dt,dv,dt < -5,dv < -5,dt < -5 or dv < -5,dt > 5 or dv > 5])
    cost=dict(new_cached_chain_rederive_elapsed_seconds=seal['seconds'],worker=seal['worker'],sum_query_conditional_seconds=sum(read(D/e['receipt_path'])['actual_cached_chain_rederive_seconds'] for e in seal['files'].values()),
              inherited_structural_replay_cost=read(UP/f'spatial_{ds}_reporting_receipt.json')['worker_CPU_seconds'],
              parent_collector_seal_sha256=sha(selected.PREV/'spatial'/f'{ds}_sealed.json'),
              original_Source_and_expert_cost_nonzero=True,new_model=0,new_Source=0,new_expert=0,new_optimizer=0,new_GT_scoring=0,
              scope='New verification wall is time.perf_counter elapsed, not processCPU/deployment. Original collection and old structural calculation are provenance, not repeated expenditure; all nested costs must not be added. Vid Source producer shared full Tent/SAR.')
    atomic(folder/'cost_segments.json',cost)
    def fmt(m):return f"{m['mean']:+.6f} [{m['CI95'][0]:+.6f}, {m['CI95'][1]:+.6f}]"
    text=f'# Temporal pooling region isolation: {ds}\n\nFull {len(rows)} queries / {result["parents"]} parent videos. This is the exact requested Source-box appearance versus corrected-box appearance comparison, already independently officially scored and all-array audited as AuxSourceROI versus FullMAP in the selected structural study; the new namespace independently verifies the complete two-arm chain for every query. No new inference, experts, optimizer, GT scoring or invented predictions.\n\nOnly the temporal ROI changes. Both use the same original eight observations, pretext Source feature grids, 1.5 context, centering/normalization, kappa0/beta1/one update and main spatial geometry. Each arm reconstructs its own prototypes, posterior and jointMAP interval, and solves final DP in its own interval. This is NOT NoSharedCorrection, which changes spatial geometry too. All-query verification uses exact audited ROI sufficient features; four actual retained grids additionally re-pool both regions. Disposed original raw grids are not claimed recoverable.\n\n|Temporal ROI|tIoU %|vIoU %|R.3 %|R.5 %|\n|---|---:|---:|---:|---:|\n'
    for a,m in means.items():text+=f"|{a}|{100*m['t']:.6f}|{100*m['v']:.6f}|{100*m['vIoU_ge_03']:.6f}|{100*m['vIoU_ge_05']:.6f}|\n"
    text+='\nCorrected minus Source ROI, percentage points, paired parent-video cluster10000bootstrap pointwise95CI:\n\n'
    for k in ['t','v']:text+=f"- {k}: {fmt(result['paired'][k])}.\n"
    text+=f"\nSevere >5pp: t={result['severe_t']}, v={result['severe_v']}, union once={result['severe_union_once']}; rescue union={result['rescue_union_once']}. All positive and negative outcomes and short/long groups are retained. Same candidate construction does not force identical final boxes: different intervals re-solve DP and may change paths. Internal sIoU/boundary checks remain in the report, outside the compact table.\n\n"
    text+='Both tIoU and vIoU confidence intervals cross zero. These results do not support a stable average temporal-localization improvement caused by shared correction of the pooling boxes. They also do not establish equivalence or prove zero effect. The supported wording is limited to object-region appearance supporting temporal correction; this comparison does not justify an additional causal claim for shared-corrected ROI. No manuscript edits or parameter/default changes are made. All full targets were already exposed; this is not unseen confirmation.\n\nCosts and inherited paid acquisition are kept separately in cost_segments.json; cached verification time is not deployment latency.\n'
    (folder/'ANALYSIS.md').write_text(text)
    rec=dict(status='root_complete_full_temporal_ROI_isolation',queries=len(rows),parents=result['parents'],
             code_sha256=sha(Path(__file__)),plan_sha256=sha(D/'plan.json'),audit_seal_sha256=sha(D/ds/'sealed.json'),
             reused_official_score_sha256=sha(UP/f'spatial_{ds}_score_receipt.json'),
             report_sha256=sha(folder/'report.json'),analysis_sha256=sha(folder/'ANALYSIS.md'),
             all_means_recomputed=True,paired_t_v_exact_existing_cluster_CI=True,new_GT_scoring=0,new_model=0,new_Source=0,new_expert=0,
             export_elapsed_seconds=time.perf_counter()-start)
    atomic(D/ds/'root_export_receipt.json',rec);atomic(folder/'root_export_receipt.json',rec)
    print(json.dumps(dict(dataset=ds,queries=len(rows),means={a:{k:100*m[k] for k in ['t','v','vIoU_ge_03','vIoU_ge_05']} for a,m in means.items()},paired={k:result['paired'][k] for k in ['t','v']},severe_union=result['severe_union_once'])),flush=True)

if __name__ == '__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preflight','audit','export']);a.add_argument('--dataset',choices=COUNTS)
    x=a.parse_args()
    if x.command in ['audit','export']:
        assert x.dataset is not None
        globals()[x.command](x.dataset)
    else:
        globals()[x.command]()

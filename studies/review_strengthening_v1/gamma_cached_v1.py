"""Precompute retained raw-grid gamma outputs, never score a partial full set.

This stage covers exactly the existing 192 retained grids per domain. It creates
no GPU owner, model or expert request. Missing raw acquisition and full-domain
official scoring require a separate future frozen execution plan.
"""
from review_common import *
import argparse,gc,shutil

DEST=D/'gamma_retained_v1'
PUBLIC=P/'gamma_retained_v1'


def prepare():
    assert not(DEST/'plan.json').exists()
    reuse=read(selected.PREV/'raw_reuse_index.json')
    files=[Path(__file__),S/'gamma_cached_core_v1.py',S/'review_common.py',
           R/'studies/map_paper_completion_v1/spatial_core.py',R/'stvg/initialization.py',
           R/'stvg/spatial.py',R/'stvg/representation.py',R/'stvg/prototype.py',
           R/'stvg/current_shared_roi.py']
    inputs=[D/'intent.json',selected.PREV/'raw_reuse_index.json',
            D/'gamma_preflight_plan.json',D/'gamma_preflight_receipt.json']
    for ds in COUNTS:inputs += [RUN/ds/'sealed.json',selected.D/f'components_{ds}_sealed.json',selected.D/f'components_{ds}_reporting_receipt.json']
    plan=dict(schema='current_gamma_retained_raw_precompute_v1',queries={ds:len(reuse[ds]) for ds in COUNTS},
              full_domain_queries=COUNTS,values=[0,.25,.5,.75,1],
              scope='Exactly existing retained192rawgrids/domain only, all five full gamma chains. This is partial prediction preparation for eventual full sensitivity, not full-domain metrics or a new sensitivity score. No GT scoring or metric-based choice here.',
              math='Same all-eight shared initialization/calibration; rebuild E, bidirectional geometry, actual native pretext ROI pool, kappa0/beta1 posterior and jointMAP, final DP at each own interval. Default gamma arrays must match current selected full bitwise. Compact retains gamma-specific pooled features/geometry/posterior for independent cold replay; raw grids are still old read-only files.',
              future='Remaining missing raw requires a separate full acquisition/execution freeze and original TA224 Source-or-prefix at most once per query shared across all five arms; zero new expert. Do not substitute pooled vectors for raw grids. Full1902/10303 seal before official labels, allpositive/negative sensitivity/parentCI/rootcurves; no retuning or source-only selection claim.',
              limits=read(D/'intent.json')['limits'],
              code_sha256={str(f.relative_to(R)):sha(f) for f in files},
              input_sha256={str(f.relative_to(R)):sha(f) for f in inputs})
    assert plan['queries']=={'hc2':192,'vidstg':192}
    plan['fingerprint']=digest(plan);atomic(DEST/'plan.json',plan);atomic(PUBLIC/'protocol.json',plan)


def run(ds):
    import torch,numpy as np
    from gamma_cached_core_v1 import build,replay
    gt_guard();torch.set_num_threads(1);p=read(DEST/'plan.json')
    assert p['fingerprint']==digest({k:v for k,v in p.items() if k!='fingerprint'})
    for field in ['code_sha256','input_sha256']:
        for f,h in p[field].items():assert sha(R/f)==h,f
    validate();who=identity();dest=DEST/ds;assert not(dest/'worker.json').exists();atomic(dest/'worker.json',who)
    rows={r['id']:r for r in records(ds)};reuse=read(selected.PREV/'raw_reuse_index.json')[ds]
    files={};started=time.perf_counter();limits=p['limits']
    for i,(key,e) in enumerate(reuse.items(),1):
        assert time.perf_counter()-started<limits['max_domain_seconds']
        assert shutil.disk_usage(R).free>=limits['min_disk_bytes']
        assert next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))>=limits['min_RAM_bytes']
        assert sum(f.stat().st_size for f in DEST.rglob('*') if f.is_file())<=limits['max_private_bytes']
        c,parent=selected.suite().load_parent(ds,rows[key]);assert sha(e['path'])==e['sha256']
        raw=torch.load(e['path'],map_location='cpu',weights_only=False,mmap=True)
        assert raw['parent_capture_sha256']==parent['capture_sha256'] and raw['frame_ids']==c['frame_ids'] and not raw['GT_read']
        pe,ce=paths(ds,key)
        with np.load(selected.D/ce['path']) as z:current={k:z[k].copy() for k in ['FullMAP','FullMAP_interval']}
        tick=time.perf_counter();a,compact,default=build(c,raw,ds,current)
        out=dest/'predictions'/(key+'.npz');cp=dest/'compact'/(key+'.pt');rp=dest/'receipts'/(key+'.json')
        assert not out.exists() and not cp.exists() and not rp.exists()
        cp.parent.mkdir(parents=True,exist_ok=True);tmp=cp.with_suffix('.tmp');torch.save(compact,tmp);os.replace(tmp,cp)
        cold=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);again=replay(cold)
        assert all(np.array_equal(a[k],again[k]) for k in a)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**a)
        os.replace(out.with_suffix('.tmp'),out)
        with np.load(out) as z:assert set(a)==set(z.files) and all(np.array_equal(a[k],z[k]) for k in a)
        rec=dict(path=str(out.relative_to(DEST)),sha256=sha(out),compact_path=str(cp.relative_to(DEST)),compact_sha256=sha(cp),
                 current_prediction_sha256=ce['sha256'],parent_capture_sha256=parent['capture_sha256'],
                 retained_raw_sha256=e['sha256'],worker=who,default_gamma_arm=default,
                 current_default_bitwise=True,actual_five_E_ROI_pools=True,all_cold_arrays_exact=True,
                 query_elapsed_wall_seconds=time.perf_counter()-tick,new_Source=0,new_expert=0,GT_read=False,fingerprint=p['fingerprint'])
        atomic(rp,rec);files[key]=dict(rec,receipt_path=str(rp.relative_to(DEST)),receipt_sha256=sha(rp))
        atomic(dest/'progress.json',dict(completed=i,total=len(reuse),full_domain_total=COUNTS[ds],worker=who,GT_read=False,partial_precompute_only=True))
        del c,raw,compact,cold,again,a;gc.collect()
    atomic(dest/'sealed_retained_subset.json',dict(status='retained_raw_subset_sealed_no_GT_metrics',queries=len(files),full_domain_queries=COUNTS[ds],files=files,worker=who,seconds=time.perf_counter()-started,GT_read=False,new_Source=0,new_expert=0))
    atomic(PUBLIC/f'{ds}_precompute_status.json',dict(status='retained_raw_subset_sealed_no_GT_metrics',queries=len(files),full_domain_queries=COUNTS[ds],full_sensitivity_complete=False,whole_domain_metrics_available=False,new_Source=0,new_expert=0))
    print(json.dumps(dict(dataset=ds,queries=len(files),full_sensitivity_complete=False)))


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','run']);a.add_argument('--dataset',choices=COUNTS)
    args=a.parse_args();globals()[args.command](*([] if args.command=='prepare' else [args.dataset]))

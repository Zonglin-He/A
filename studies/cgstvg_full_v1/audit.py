"""Prelaunch and post-domain audits. No model construction or label access here."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,time,argparse,csv
from pathlib import Path
S=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from run import R,D,P,OLD,FOLLOW,TOTAL,rows,read,sha,atomic,validate,load_legacy,committed,PROFILES,ARMS

def preflight():
    import numpy as np,torch
    from core import feedback,replay
    from cg_calibrate import feedback as prior_feedback
    torch.set_num_threads(1);p=validate();index=read(D/'reuse_index.json');checks={};start=time.perf_counter()
    hc_lookup={x['row']['id']:x for x in read(R/'outputs/paper_hc_refine_v2/index.json')}
    # Exercise each existing acquisition origin in each domain. Every raw cache
    # is checked against its own input/native-grid/evidence key again at use.
    for ds in TOTAL:
        lookup={r['id']:r for r in rows(ds)};checks[ds]=dict(cases=0,profile_arrays_exact=0,cold_sufficient_replay_exact=0)
        chosen=[]
        for origin in ['old_screen','old_confirm','followup_confirm']:
            chosen.extend([k for k,e in index[ds].items()if e['origin']==origin][:2])
        for key in chosen:
            cap,native,e=load_legacy(index[ds][key],lookup[key],ds)
            arrays,compact=feedback(cap,native,ds,PROFILES[ds]);again=replay(compact,PROFILES[ds])
            assert all(np.array_equal(arrays[k],again[k])for k in arrays)
            if ds=='hc2':
                with np.load(R/'outputs/paper_hc_refine_v2/predictions'/(key+'.npz'))as z:
                    assert np.array_equal(arrays['Full'],z['u0.0_l0.0_b0.5'])
                    assert np.array_equal(arrays['Full_interval'],z['u0.0_l0.0_b0.5_interval'])
                    assert np.array_equal(arrays['Source'],z['Source'])
            else:
                reference=prior_feedback(cap,native,ds,PROFILES[ds])
                assert all(np.array_equal(arrays[k],reference[k])for k in reference)
                if index[ds][key]['origin']=='followup_confirm':
                    old=index[ds][key]['entry']
                    with np.load(FOLLOW/old['path'])as z:assert all(np.array_equal(arrays[k],z[k])for k in arrays)
            checks[ds]['cases']+=1;checks[ds]['profile_arrays_exact']+=1;checks[ds]['cold_sufficient_replay_exact']+=1
    result=dict(status='passed',fingerprint=p['fingerprint'],cases=checks,
        full_input_counts=TOTAL,raw_grids_repool_actual=True,no_TA_cache_substitution=True,
        new_model=0,new_Source=0,new_expert=0,GT_read=False,
        prior_native_checkpoint_bridges='4passed HC2+Vid2 in cgstvg_v9 and4passed in followup cg_confirm; referenced, not rerun as CPU tests. First2new acquisitions/domain in this full run will repeat actual model native no-op.',
        prior_bridge_receipt_sha256=sha(OLD/'cgstvg_v9_screen_score_receipt.json'),
        seconds=time.perf_counter()-start)
    atomic(D/'preflight_receipt.json',result);atomic(P/'preflight_receipt.json',result);print(result,flush=True)

def reporting(ds):
    import numpy as np,torch
    from core import replay
    torch.set_num_threads(1);p=validate();sp=D/f'{ds}_sealed.json';seal=read(sp);score=read(D/f'{ds}_score_receipt.json')
    assert score['seal_sha256']==sha(sp)and score['queries']==TOTAL[ds]and score['checks']==TOTAL[ds]*len(ARMS)
    assert score['whole_domain_sealed_before_GT']and not seal['GT_read']
    private=read(D/ds/'metrics.json')['private_rows'];public=read(P/ds/'report.json');assert len(private)==len(seal['files'])==TOTAL[ds]
    table=[];segments={};checks=0;cache_count=0;source_loads={};teacher_loads={};storage=0
    for i,row in enumerate(private,1):
        e=committed(ds,row['id']);assert e==seal['files'][row['id']];checks+=3
        cap=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True)
        with np.load(D/e['prediction_path'])as z:
            again=replay(cap,p['profiles'][ds]);assert all(np.array_equal(again[k],z[k])for k in again)
            assert all(np.isfinite(z[a]).all()for a in ARMS)
        worker=f"{e['worker']['pid']}:{e['worker']['startticks']}"
        seg=segments.setdefault(worker,dict(queries=0,reused_legacy=0,feedback_seconds=0.,pipeline_seconds=0.,new_Source_forwards=0,new_Source_seconds=0.,new_expert=0,expert_original_cost={},teacher_load_seconds=0.))
        seg['queries']+=1;seg['feedback_seconds']+=e['feedback_seconds'];seg['pipeline_seconds']+=e['pipeline_seconds']
        ac=e['acquisition']
        if ac['acquisition_reused']:
            seg['reused_legacy']+=1;cache_count+=1
        else:
            src=ac['Source'];seg['new_Source_forwards']+=src['Source_forwards'];seg['new_Source_seconds']+=src['Source_seconds']
            seg['new_expert']+=ac['expert_new'];seg['teacher_load_seconds']+=ac['expert_load_seconds']
            for name,value in ac['expert_original_cost'].items():seg['expert_original_cost'][name]=seg['expert_original_cost'].get(name,0.)+value
            sw=f"{src['worker']['pid']}:{src['worker']['startticks']}";source_loads.setdefault(sw,src['load']['model_load_seconds'])
        storage+=(D/e['compact_path']).stat().st_size+(D/e['prediction_path']).stat().st_size
        for baseline in ['Source','Space']:
            a=row['values']['Full'];b=row['values'][baseline];dt=100*(a['t']-b['t']);dv=100*(a['v']-b['v'])
            table.append([f'Q{i:05}',row['duration_group'],row['origin'],row['selection_query'],baseline,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5,dt>5 or dv>5])
    for a in ARMS:
        for field,value in public['means'][a].items():
            vals=[r['values'][a][field]for r in private if r['values'][a][field]is not None]
            assert(value is None and not vals)or abs(value-float(np.mean(vals)))<1e-12
    folder=P/ds
    with(folder/'all_signed_severe_timegroups.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','timegroup','previousCGorigin','selection_query','baseline','delta_t_pp','delta_v_pp','severe_t','severe_v','severe_union_once','gain_over5pp_union']);w.writerows(table)
    report=dict(status='verified_whole_domain',queries=TOTAL[ds],SHA_checks=checks,all_compact_array_replays_exact=True,
        all_means_recomputed=True,legacy_reused_queries=cache_count,worker_segments=segments,
        source_model_load_seconds_once_per_actual_worker=source_loads,
        compact_plus_prediction_bytes=storage,source_seal_sha256=sha(sp),score_receipt_sha256=sha(D/f'{ds}_score_receipt.json'),
        severe_vs_Source=public['severe_vs_Source'],timegroups=public['timegroups'],
        cost_scope='Actual worker/PID segments. Historical acquisition receipts are provenance, not new work or zero deployment cost. Source capture/hooks/CPU-transfer and native extras are nested in pipeline; expert subcomponents nested. Loads separate once per actual worker. Cache-prefix CPU time is not an ETA for unseen Source work. Raw grids intentionally discarded only for completed NEW study scratch; sufficient state exactly replays fixed outputs, not arbitrary future repooling.',
        new_model=0,new_Source=0,new_expert=0,new_GT_scoring=0)
    atomic(D/f'{ds}_reporting_receipt.json',report);atomic(folder/'reporting_receipt.json',report);print({'dataset':ds,'status':report['status'],'queries':TOTAL[ds],'SHA_checks':checks})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['preflight','report']);a.add_argument('--dataset',choices=list(TOTAL));z=a.parse_args()
    (preflight if z.command=='preflight'else lambda:reporting(z.dataset))()

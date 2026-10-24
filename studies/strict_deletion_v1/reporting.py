"""Whole-domain postseal strict deletion proof, signed tails and cost segments."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import sys,argparse,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from run import D,P,COUNTS,ARMS,read,sha,atomic,validate,require_gate

def run(ds):
    require_gate();p=validate()
    import torch,numpy as np
    from strict_core import replay
    torch.set_num_threads(1);sp=D/f'{ds}_sealed.json';score=read(D/f'{ds}_score_receipt.json');seal=read(sp)
    assert score['seal_sha256']==sha(sp)and score['queries']==COUNTS[ds]and score['checks']==6*COUNTS[ds]
    rows=read(D/ds/'metrics.json')['private_rows'];report=read(P/ds/'report.json');assert len(rows)==seal['queries']==COUNTS[ds]
    segments={};acquisition_segments={};signed=[];checks=0
    for i,row in enumerate(rows,1):
        e=seal['files'][row['id']]
        assert sha(D/e['path'])==e['sha256']and sha(D/e['compact_path'])==e['compact_sha256']and sha(D/e['receipt_path'])==e['receipt_sha256'];checks+=3
        compact=torch.load(D/e['compact_path'],map_location='cpu',weights_only=False,mmap=True)
        with np.load(D/e['path'])as z:
            again=replay(compact);assert all(np.array_equal(z[k],again[k])for k in again)
            assert np.array_equal(z['NoSpatialTrajectory'],compact['E'])and z['NoTemporalInference_interval'].tolist()==compact['original_interval']
        assert torch.equal(compact['uncorrected_geometry']['E'],torch.as_tensor(compact['Source'],dtype=torch.float64))
        worker=f"{e['worker']['pid']}:{e['worker']['startticks']}";seg=segments.setdefault(worker,dict(queries=0,query_seconds=0.,modes={}))
        seg['queries']+=1;seg['query_seconds']+=e['query_seconds'];seg['modes'][e['mode']]=seg['modes'].get(e['mode'],0)+1
        origin=e['raw_origin']
        if 'worker'in origin:
            aw=f"{origin['worker']['pid']}:{origin['worker']['startticks']}";a=acquisition_segments.setdefault(aw,dict(Source_queries=0,appearance_prefix_queries=0,Source_seconds=0.))
            if origin['mode']=='full_Source':a['Source_queries']+=1
            if origin['mode']=='appearance_prefix':a['appearance_prefix_queries']+=1
            a['Source_seconds']+=origin['Source_seconds']
        for arm in ARMS[1:]:
            dt=100*(row['values']['Full']['t']-row['values'][arm]['t']);dv=100*(row['values']['Full']['v']-row['values'][arm]['v'])
            signed.append([f'Q{i:05}',row['duration_group'],arm,dt,dv,dt< -5,dv< -5,dt< -5 or dv< -5])
    for arm,m in report['means'].items():
        for field,value in m.items():
            vals=[r['values'][arm][field]for r in rows if r['values'][arm][field]is not None]
            assert(value is None and not vals)or abs(value-float(np.mean(vals)))<1e-12
    with(P/ds/'all_signed_deletion_tails.tsv').open('w')as f:
        w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['query','timegroup','deleted_component_arm','Full_minus_arm_t_pp','Full_minus_arm_v_pp','Full_harm_t','Full_harm_v','Full_harm_union_once']);w.writerows(signed)
    receipt=dict(status='complete_verified_strict_deletion_domain',queries=COUNTS[ds],prediction_compact_receipt_SHA_checks=checks,
        every_Full_prior_bitwise=True,all_four_arm_replay_exact=True,NoShared_uses_Source_initialization=True,
        NoSpatial_exact_E_without_teacher_overwrite=True,NoTemporal_nativeMAP_without_MBR=True,
        worker_segments=segments,acquisition_segments=acquisition_segments,new_expert=0,new_GT_scoring=0,new_model=0,
        seal_sha256=sha(sp),score_sha256=sha(D/f'{ds}_score_receipt.json'),
        cost_scope='New Source/prefix work split by actual acquisition PID/start, distinct from CPU-query segments. Original acquisition provenance is not zero deployment cost. Nested times are not additive. Full reuses only exact same trajectory features; deletion branches are repooled. All target histories already exposed; no selection.')
    atomic(D/f'{ds}_reporting_receipt.json',receipt);atomic(P/ds/'reporting_receipt.json',receipt);print(receipt)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=list(COUNTS),required=True);run(p.parse_args().dataset)

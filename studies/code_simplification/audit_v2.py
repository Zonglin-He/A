"""Exact core-preserving refactor: initialization, grid adapter, matching, time."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1] / 'algorithm_simplification')]
import numpy as np
import torch
import run as R
import stvg.evidence as em
import stvg.spatial as sm
from stvg.initialization import initialize as old_initialize
from stvg.predictor import output_writeback
from init_compact import initialize, DirectionalMatches

D=R.ROOT/'outputs/code_simplification_v2'
P=R.ROOT/'results/code_simplification_v2'


def build(features, valid, cap, seeds, boxes, compare=False):
    original=em.affinity
    cache=DirectionalMatches(original)
    def lookup(*args):
        result=cache(*args)
        if compare:assert torch.equal(result,original(*args))
        return result
    with patch.object(em,'affinity',lookup),patch.object(sm,'affinity',lookup):
        evidence=em.SourceEvidence.from_arrays(features,valid,cap['frame_ids'],cap['positions8'],seeds)
        geometry=sm.SpatialGeometry(evidence,cap['E'],boxes)
    return geometry,dict(requests=cache.requests,hits=cache.hits,computed=len(cache.cache),
                         retained_cache_bytes=sum(x.numel()*x.element_size()for x in cache.cache.values()))


def check_initialization(cap, ds, cfg):
    source=np.asarray(cap['Source'],dtype=np.float64)
    expert=np.asarray(cap['geometry']['native_boxes'],dtype=np.float64)
    pos=cap['positions8'];c=cfg[ds];gamma=c['initialization_gamma']
    old,meta=old_initialize(source,pos,expert,c['E_calibration'])
    old=output_writeback(source,old,gamma)
    new,trace=initialize(source,pos,expert,c['E_calibration'],gamma)
    assert np.array_equal(new,old) and np.array_equal(new,np.asarray(cap['E']))
    assert np.array_equal(trace['delta'],meta['delta'])
    assert trace['selected']==tuple(meta['selected_reference_indices'])
    assert trace['objective']==meta['objective']


def main():
    torch.set_num_threads(1)
    parent=R.verify()
    assert not(D/'plan.json').exists()
    assert R.read(R.ROOT/'outputs/code_simplification_v1/state.json')['status']=='failed_exactness_or_engineering'
    code={str(f.relative_to(R.ROOT)):R.sha(f)for f in Path(__file__).parent.glob('*.py')}
    plan=dict(schema='core_preserving_code_simplification_v2',code=code,parent_sha256=R.sha(R.D/'plan.json'),
              prior_failed_plan_sha256=R.sha(R.ROOT/'outputs/code_simplification_v1/plan.json'),
              scope='No model/labels/structural changes. Actual eight-singleton robust estimator retains exact enumeration/objective/ties; identical directed-affinity cache only; original both anchors/cycle/affine/links/DP unchanged. Known native grid shape and temporal equivalent formulas are reused from completed audits.',
              checks='Exact2030Source/E/init delta/admission/objective; exact67all geometry tensors/aux and final boxes/actions/scores; every requested affinity exact; temporal2030 receipt retained, no new GT. Abort on mismatch, no threshold relaxation.',
              timing='Three alternating repeats after warmup on same67; query-local cache memory reported, comparator disabled for timing.',
              budget=dict(initialization_queries=2030,spatial_queries=67,new_model=0,GPU=0,GT=0),production_unchanged=True)
    R.atomic(D/'plan.json',plan);R.atomic(P/'protocol.json',plan)
    for f in code:
        dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((R.ROOT/f).read_bytes())
    R.atomic(D/'worker.json',R.identity());R.atomic(D/'state.json',dict(status='running',**R.identity()))
    cfg=json.loads((R.ROOT/'configs/default.json').read_text())['datasets'];started=time.perf_counter();rows=[]
    try:
        seal=R.read(R.RUN/'hc2/sealed.json');init_count=0
        for e in seal['files'].values():
            cap=torch.load(R.RUN/'hc2'/e['capture_path'],map_location='cpu',weights_only=False)
            check_initialization(cap,'hc2',cfg);init_count+=1
        old_plan=R.read(R.OLD/'plan.json')
        for item in old_plan['items']:
            if item['dataset']=='vidstg':
                cap=torch.load(item['path'],map_location='cpu',weights_only=False)
                check_initialization(cap,'vidstg',cfg);init_count+=1
        assert init_count==2030;print('initialization exact2030',flush=True)
        total_tensors=0
        for n,item in enumerate(parent['items'],1):
            cap=torch.load(item['capture_path'],map_location='cpu',weights_only=False)
            raw=torch.load(item['raw_path'],map_location='cpu',weights_only=False)
            g=sm.SpatialGeometry.__new__(sm.SpatialGeometry);g.__dict__.update(cap['geometry'])
            features,valid=R.known_grid(raw['spatial_inputs'],cap['frame_ids'],g.height,g.width)
            ev=R.C.load_evidence(item,cap)
            new,stats=build(features,valid,cap,ev.seeds,g.native_boxes,compare=True)
            tensor_count=0
            for key,value in g.__dict__.items():
                if torch.is_tensor(value):assert torch.equal(value,getattr(new,key)),key;tensor_count+=1
            q0=R.from_capture(cap);q,_=R.one_step(cap['content_features'],q0)
            iv=R.final_mbr(cap['frame_ids'],q)['interval']
            for span in [[cap['frame_ids'][0],cap['frame_ids'][-1]+1],iv]:
                a=R.evidence_dp(g.for_interval(span));b=R.evidence_dp(new.for_interval(span))
                assert a['score']==b['score']
                for key in ['final','actions']:assert torch.equal(a[key],b[key])
            times={k:[]for k in ['original','directed_cache','old_init','singleton_init']}
            source=np.asarray(cap['Source'],np.float64);expert=np.asarray(g.native_boxes,np.float64);c=cfg['hc2']
            for rep in range(3):
                for kind in (list(times)if rep%2==0 else list(reversed(times))):
                    tick=time.perf_counter()
                    if kind=='original':
                        ee=em.SourceEvidence.from_arrays(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds)
                        sm.SpatialGeometry(ee,cap['E'],g.native_boxes)
                    elif kind=='directed_cache':build(features,valid,cap,ev.seeds,g.native_boxes)
                    elif kind=='old_init':
                        temp,_=old_initialize(source,cap['positions8'],expert,c['E_calibration'])
                        output_writeback(source,temp,c['initialization_gamma'])
                    else:initialize(source,cap['positions8'],expert,c['E_calibration'],c['initialization_gamma'])
                    times[kind].append(time.perf_counter()-tick)
            rows.append(dict(query=f'H{n:03d}',geometry_tensor_fields_exact=tensor_count,**stats,
                             median_seconds={k:float(np.median(v))for k,v in times.items()}));total_tensors+=tensor_count
            if n%16==0:print('spatial exact',n,flush=True)
        R.verify();assert all(R.sha(R.ROOT/f)==h for f,h in code.items())
        assert not torch.cuda.is_initialized()
        result=dict(status='passed',initialization_queries_exact=init_count,spatial_queries_exact=67,
                    all_geometry_tensor_fields_exact=total_tensors,all_requested_affinities_bitwise=True,
                    aux_final_actions_boxes_scores_bitwise=True,core_components_unchanged=True,
                    temporal_exact2030_receipt_sha256=R.sha(R.P/'EXACT_MATH.json'),
                    mean_median_seconds={k:float(np.mean([r['median_seconds'][k]for r in rows]))for k in times},
                    rows=rows,seconds=time.perf_counter()-started,new_models=0,GT_read=False,production_unchanged=True)
        R.atomic(P/'report.json',result);R.atomic(D/'state.json',dict(status='passed',report_sha256=R.sha(P/'report.json')))
        print({k:v for k,v in result.items()if k!='rows'},flush=True)
    except Exception as exc:
        R.atomic(D/'state.json',dict(status='failed_exactness_or_engineering',error=repr(exc),spatial_verified=len(rows),**R.identity()))
        raise


if __name__=='__main__':main()

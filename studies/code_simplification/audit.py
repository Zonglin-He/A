"""Core-preserving code simplification audit; no models/labels/selection."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
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
from matches import PairMatches

D = R.ROOT / 'outputs/code_simplification_v1'
P = R.ROOT / 'results/code_simplification_v1'


def build(features, valid, cap, seeds, boxes, *, compare=False):
    service = PairMatches()
    original = em.affinity
    def lookup(*args):
        result = service(*args)
        if compare:
            reference = original(*args)
            if not torch.equal(result, reference):
                raise AssertionError(dict(message='Transposed/shared affinity is not bitwise exact; stop this optimization',
                                          max_error=float((result-reference).abs().max())))
        return result
    with patch.object(em, 'affinity', lookup), patch.object(sm, 'affinity', lookup):
        evidence = em.SourceEvidence.from_arrays(features, valid, cap['frame_ids'], cap['positions8'], seeds)
        geometry = sm.SpatialGeometry(evidence, cap['E'], boxes)
    return geometry, service.stats()


def main():
    torch.set_num_threads(1)
    parent = R.verify()
    assert not (D / 'plan.json').exists()
    code = {str(f.relative_to(R.ROOT)): R.sha(f) for f in Path(__file__).parent.glob('*.py')}
    plan = dict(schema='core_preserving_code_simplification_v1', authorization='Latest user: verify code simplification, core components unchanged; no article writing.',
                parent_plan_sha256=R.sha(R.D/'plan.json'), code=code, queries=67,
                changes='Reuse unordered frame-pair cosine matrix and previously requested directional affinities within one query; retain exact original top20/softmax, bidirectional fitting, both anchors, cycle/validity/weights, mask transport links, DP and all coefficients. Direct known-shape grid adapter and equivalent temporal formulas already audited separately.',
                required='Every requested affinity bitwise; every geometry tensor bitwise; aux/final paths, scores/actions/boxes bitwise; native posterior <=1e-12 and updated posterior <=1e-11; final interval exact. Abort on any mismatch, no relaxed tolerance or automatic fallback.',
                timing='Original and memoized SourceEvidence+SpatialGeometry, warmup excluded, three alternating repeats per query; no comparator calls during measured timing. Cache is query-local; disclose memory.',
                budget=dict(new_model=0, GPU=0, GT=0, structural_arms=0, queries=67), production_unchanged=True)
    R.atomic(D/'plan.json', plan)
    for f in code:
        dest=D/'source_snapshot'/f;dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes((R.ROOT/f).read_bytes())
    R.atomic(D/'worker.json', R.identity())
    R.atomic(D/'state.json', dict(status='running', **R.identity()))
    R.atomic(P/'protocol.json', plan)
    started=time.perf_counter(); rows=[]; all_tensors=0
    try:
        for n, item in enumerate(parent['items'], 1):
            cap=torch.load(item['capture_path'], map_location='cpu', weights_only=False)
            raw=torch.load(item['raw_path'], map_location='cpu', weights_only=False)
            g=sm.SpatialGeometry.__new__(sm.SpatialGeometry);g.__dict__.update(cap['geometry'])
            features,valid=R.known_grid(raw['spatial_inputs'],cap['frame_ids'],g.height,g.width)
            ev=R.C.load_evidence(item, cap)
            new, stats=build(features,valid,cap,ev.seeds,g.native_boxes,compare=True)
            checked=0
            for key,value in g.__dict__.items():
                if torch.is_tensor(value):
                    assert torch.equal(value,getattr(new,key)),key
                    checked+=1
            old_aux=R.evidence_dp(g.for_interval([cap['frame_ids'][0],cap['frame_ids'][-1]+1]))
            new_aux=R.evidence_dp(new.for_interval([cap['frame_ids'][0],cap['frame_ids'][-1]+1]))
            for key in ['final','actions']:assert torch.equal(old_aux[key],new_aux[key])
            assert old_aux['score']==new_aux['score']
            q0=R.from_capture(cap)
            logs=cap.get('native_logits')
            if logs is None:logs=[x[-1] for x in cap['logits']]
            ids,fast_q0=R.positive_hull(R.probabilities(cap['records'],logs))
            assert np.max(abs(q0-fast_q0))<1e-12
            slow_q,_=R.one_step(cap['content_features'],q0)
            fast_q,_=R.linear_step(cap['content_features'],fast_q0)
            assert np.max(abs(slow_q-fast_q))<1e-11
            interval=R.final_mbr(ids,slow_q)['interval'];assert interval==R.final_mbr(ids,fast_q)['interval']
            old_out=R.evidence_dp(g.for_interval(interval));new_out=R.evidence_dp(new.for_interval(interval))
            for key in ['final','actions']:assert torch.equal(old_out[key],new_out[key])
            assert old_out['score']==new_out['score']
            times={'original':[],'shared_matches':[]}
            for rep in range(3):
                for kind in (['original','shared_matches'] if rep%2==0 else ['shared_matches','original']):
                    tick=time.perf_counter()
                    if kind=='original':
                        ee=em.SourceEvidence.from_arrays(features,valid,cap['frame_ids'],cap['positions8'],ev.seeds)
                        sm.SpatialGeometry(ee,cap['E'],g.native_boxes)
                    else:build(features,valid,cap,ev.seeds,g.native_boxes)
                    times[kind].append(time.perf_counter()-tick)
            rows.append(dict(query=f'H{n:03d}',geometry_tensor_fields_exact=checked,**stats,
                             median_seconds={k:float(np.median(v))for k,v in times.items()}))
            all_tensors+=checked
            if n%16==0:print('verified',n,flush=True)
        R.verify()
        assert all(R.sha(R.ROOT/f)==h for f,h in code.items())
        assert not torch.cuda.is_initialized()
        output=dict(status='passed',queries=67,all_geometry_tensor_fields_exact=all_tensors,
                    all_requested_affinities_bitwise=True,all_aux_final_actions_boxes_scores_exact=True,
                    all_final_intervals_exact=True,posterior_bitwise_claim=False,
                    mean_median_seconds={k:float(np.mean([r['median_seconds'][k]for r in rows]))for k in times},
                    rows=rows,seconds=time.perf_counter()-started,GT_read=False,new_models=0,production_unchanged=True,
                    parent_exact2030_sha256=R.sha(R.P/'EXACT_MATH.json'),plan_sha256=R.sha(D/'plan.json'))
        R.atomic(P/'report.json',output)
        R.atomic(D/'state.json',dict(status='passed',report_sha256=R.sha(P/'report.json')))
        print({k:v for k,v in output.items()if k!='rows'},flush=True)
    except Exception as exc:
        R.atomic(D/'state.json',dict(status='failed_exactness_or_engineering',error=repr(exc),verified_queries=len(rows),**R.identity()))
        raise


if __name__=='__main__':
    main()

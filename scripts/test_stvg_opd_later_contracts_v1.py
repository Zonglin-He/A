"""Synthetic incomplete-stream, paired-parent and figure contracts; no GT/model."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import copy
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.stvg_opd_paper_later_common_v1 import identical_stream, input_equivalent, alias_stage
from scripts.finalize_stvg_opd_later_phase_v1 import independent, matched_contrast
from scripts.score_stvg_opd_p1_v1 import source_stats


def rejects(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError('Malformed stream was accepted')


def run():
    checks = 0
    source = dict(dataset='hc2', source='vidstg', observation_budget=4,
        conditions=['clean'], orders={'order1': [4, 6, 8], 'order2': [8, 4, 6]}, arms=['on_policy'])
    cfg = dict(lr=.03, sigma=.1, tau=.25, steps=20, writeback=1/16, samples=32)
    target = {**source, 'variant_configs': {'on_policy': cfg}, 'reuse_identical_presealed_streams': True}
    identical_stream(source, target, 'on_policy', cfg); checks += 1
    for field, bad in [('dataset', 'vidstg'), ('source', 'hcstvg2'), ('observation_budget', 8),
                       ('conditions', ['frame_drop_5']), ('orders', {'order1': [4, 6]})]:
        rejects(lambda: identical_stream(source, {**target, field: bad}, 'on_policy', cfg)); checks += 1
    rejects(lambda: identical_stream(source, target, 'on_policy', {**cfg, 'writeback': 0})); checks += 1
    assert alias_stage('P3_hc2_cross_clean', target, 'query_only') is None; checks += 1
    assert alias_stage('P5_unified_hc2', {**target, 'reuse_identical_presealed_streams': False}, 'on_policy') is None; checks += 1
    inp = dict(pixel_sha256='synthetic-pixel', source_model_state_sha256='synthetic-source', interval=[2, 9],
        frame_ids=[0, 3, 8], native_boxes=torch.ones(3, 4), expert={'admitted': [.1, .2]})
    input_equivalent(inp, copy.deepcopy(inp)); checks += 1
    for field, bad in [('pixel_sha256', 'different'), ('source_model_state_sha256', 'different'),
                       ('interval', [1, 9]), ('expert', {'admitted': [.3, .2]})]:
        rejects(lambda: input_equivalent(inp, {**inp, field: bad})); checks += 1
    rejects(lambda: input_equivalent(inp, {**inp, 'native_boxes': torch.zeros(3, 4)})); checks += 1
    rows = []
    for source_id, queries in [(0, [(0, .1), (1, .3)]), (1, [(2, .8)])]:
        for q, value in queries:
            for condition in ['clean', 'motion_blur_5']:
                for order in ['order1', 'order2']:
                    rows.append(dict(query_ordinal=q, source_id=source_id, condition=condition, order=order,
                        arrival=q, Frozen_v=.2, Frozen_t=.4, Frozen_s=.5, After_v=value,
                        After_t=.4, After_s=.6, After_R30=float(value>.3), After_R50=float(value>.5),
                        delta_total_v=value-.2))
    fields = ['After_v', 'delta_total_v']
    expected, sources, matrix = independent(rows, fields)
    assert sources == [0, 1] and matrix.shape == (2, 2); checks += 2
    assert expected['cells'] == 12 and expected['queries'] == 3 and expected['sources'] == 2; checks += 3
    assert abs(expected['metrics']['After_v']['mean'] - .5) < 1e-12; checks += 1
    assert abs(expected['metrics']['After_v']['query_macro'] - .4) < 1e-12; checks += 1
    # This suite is source-balanced; P1 handles unequal query counts. A source
    # average of raw arrivals would overweight a multi-query parent here.
    rejects(lambda: independent(rows + [rows[0]], fields)); checks += 1
    rejects(lambda: independent(rows[:-1], fields)); checks += 1
    shuffled = independent(list(reversed(rows)), fields)[0]
    for field in fields:
        assert abs(shuffled['metrics'][field]['mean'] - expected['metrics'][field]['mean']) < 1e-12; checks += 1
        assert np.max(np.abs(np.array(shuffled['metrics'][field]['ci95']) - expected['metrics'][field]['ci95'])) < 1e-12; checks += 1
    control = [{**r, 'After_v': r['After_v']-.07, 'After_t': .4, 'After_s': .58} for r in rows]
    contrast = matched_contrast(rows, control)
    assert abs(contrast['metrics']['Full_minus_v']['mean']-.07) < 1e-12; checks += 1
    assert np.max(np.abs(np.array(contrast['metrics']['Full_minus_v']['ci95'])-.07)) < 1e-12; checks += 1
    rejects(lambda: matched_contrast(rows, [{**control[0], 'Frozen_v': .21}, *control[1:]])); checks += 1
    # Balanced one-query panels must reproduce the scorer's independent RNG chunks.
    balanced = [r for r in rows if r['query_ordinal'] != 1]
    stats = source_stats(balanced, fields); check = independent(balanced, fields)[0]
    for field in fields:
        for key in ['mean', 'query_macro']:
            assert abs(stats['metrics'][field][key]-check['metrics'][field][key]) < 1e-12; checks += 1
        assert np.max(np.abs(np.array(stats['metrics'][field]['ci95'])-check['metrics'][field]['ci95'])) < 1e-12; checks += 1
    return dict(status='pass', synthetic_contracts=checks, real_GPU_qualification=False,
        research_GT_or_predictions_read=False, scope='exact whole-stream reuse and matched independent parent statistics')


if __name__ == '__main__':
    print(json.dumps(run()))

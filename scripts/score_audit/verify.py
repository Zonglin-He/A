"""CPU-only verification of the sealed teacher-reference proposal holdout.

No GT labels, new trajectories, adaptation, models or RGB are read/executed.
After the entire proposal seal exists, this verifier checks all 512 folds and
rebuilds the first three manifest queries' proposals under extreme changes to
their excluded teacher output. This is a downstream non-interference check,
not proof that cached teacher outputs came from independent seven-frame runs.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.score_audit.proposal import build, retained_teacher
from scripts.score_audit.cross_feedback import source_only_affinity, predict_held_support
from scripts.score_audit.run import TR, V0, OLD, validate


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def no_memory(value):
    if isinstance(value, dict):
        assert all('memory' not in str(k).lower() for k in value), 'Proposal contains a memory field'
        for v in value.values(): no_memory(v)
    elif isinstance(value, (tuple, list)):
        for v in value: no_memory(v)


def same(left, right, name):
    if torch.is_tensor(left):
        assert torch.is_tensor(right) and left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right), 'Excluded teacher output changed proposal field: ' + name
    else:
        assert left == right, 'Excluded teacher output changed metadata: ' + name


def extreme_held(raw, held, target_query):
    """Perturb only the withheld row; untouched rows stay bit-identical."""
    logits, boxes, masks = (raw[k].clone() for k in ('pred_logits', 'pred_boxes', 'pred_masks'))
    logits[0, held].fill_(-1e6)
    logits[0, held, target_query].fill_(1e6)
    boxes[0, held, :, :2].fill_(-1e6)
    boxes[0, held, :, 2:].fill_(1e6)
    pattern = torch.arange(masks.shape[-2] * masks.shape[-1]).reshape(masks.shape[-2:]) % 2
    masks[0, held] = torch.where(pattern.bool(), 1e6, -1e6).to(masks.dtype)
    return logits, boxes, masks


def main():
    seal_path = TR / 'proposals/sealed.json'
    if not seal_path.is_file():
        raise SystemExit('Complete 64-query/512-fold proposal seal is required; nothing was run')
    torch.set_num_threads(1)
    begin = time.perf_counter()
    lock = read(TR / 'lock.json')
    seal = validate('proposals', lock)
    assert len(lock['inputs']) == 64
    initial_seal_sha = sha(seal_path)
    initial_lock_sha = sha(TR / 'lock.json')
    fold_checks = 0
    rebuilt = 0
    predicted_checks = 0
    source_checks = 0
    q7_diff_full8 = 0
    full8_selection_changed_by_mutation = 0
    summaries = []
    for ordinal, item in enumerate(lock['inputs']):
        key = item['id']
        entry = seal['files'][key]
        saved = torch.load(TR / 'proposals' / entry['path'], map_location='cpu', weights_only=False)
        assert saved['fingerprint'] == lock['fingerprint'] and saved['GT_read'] is False
        assert len(saved['folds']) == 8
        assert sha(OLD / item['reference_path']) == item['reference_sha256']
        with np.load(OLD / item['reference_path'], allow_pickle=False) as reference:
            assert np.array_equal(saved['source'].numpy(), reference['Source'])
            assert np.array_equal(np.asarray(saved['frame_ids']), reference['frame_ids'])
            assert np.array_equal(np.asarray(saved['positions8']), reference['positions8'])
            assert np.array_equal(np.asarray(saved['interval']), reference['Strong_interval'])
        source_checks += 1
        positions = [int(x) for x in saved['positions8']]
        frames = len(saved['source'])
        assert len(set(positions)) == 8 and frames == len(saved['frame_ids'])
        no_memory(saved)
        for held, fold in enumerate(saved['folds']):
            keep = [j for j in range(8) if j != held]
            assert fold['held_index'] == held and fold['held_node'] == positions[held]
            assert fold['retained_indices'] == keep
            assert fold['retained_positions'] == [positions[j] for j in keep]
            assert int(fold['observed'].sum()) == 7 and not fold['observed'][positions[held]]
            assert not bool((fold['parents'] == positions[held]).any())
            assert torch.equal(fold['boxes'][:, 0], saved['source'])
            assert bool(fold['valid'][:, 0].all()) and not fold['valid'][~fold['active'], 1:].any()
            assert fold['boxes'].shape == (frames, 3, 4)
            assert torch.isfinite(fold['boxes']).all() and (fold['boxes'][..., 2:] > 0).all()
            assert fold['predicted_held'].shape == (frames, 3, *saved['shape'])
            assert not bool(fold['predicted_held'][positions[held]].any()), 'Held node must not score itself'
            fold_checks += 1
        if ordinal >= 3:
            continue
        assert sha(OLD / 'research/native_masks' / item['mask_path']) == item['native_mask_sha256']
        pack = torch.load(OLD / 'research/native_masks' / item['mask_path'], map_location='cpu', weights_only=False)
        receipt = read(V0 / 'stage1/predictions' / (key + '.json'))
        evidence_path = V0 / 'stage1' / receipt['evidence_path']
        assert sha(evidence_path) == receipt['evidence_sha256']
        ev = torch.load(evidence_path, map_location='cpu', weights_only=False)
        assert pack['GT_read'] is ev['GT_read'] is False
        assert pack['source_cache_sha256'] == item['cache_sha256']
        # Only Source-grid feature, validity, geometry and frame-id fields are
        # used. Old all-eight seeds, targets, confidences and banks are not used.
        features, valid = ev['features'].float(), ev['valid'].bool()
        H, W = tuple(saved['shape'])
        assert H == ev['metadata']['grid_height'] and W == ev['metadata']['grid_width']
        assert torch.equal(saved['frame_ids'], ev['frame_ids'])
        raw = pack.get('raw_out', pack.get('out'))
        assert raw['pred_logits'].shape[:3] == raw['pred_boxes'].shape[:3] == raw['pred_masks'].shape[:3] == (1, 8, 15)
        full8_query = int(raw['pred_logits'][0].sigmoid().mean(0).max(-1).values.argmax())
        event = (saved['frame_ids'] >= saved['interval'][0]) & (saved['frame_ids'] < saved['interval'][1])
        affinity_cache = {}
        local_rebuilt = 0
        for held, fold in enumerate(saved['folds']):
            keep = fold['retained_indices']
            pos7 = fold['retained_positions']
            # Mutated excluded rows are removed BEFORE selection/interpolation.
            mutated = extreme_held(raw, held, (full8_query + 1) % 15)
            original7 = [raw[k][0, keep].clone() for k in ('pred_logits', 'pred_boxes', 'pred_masks')]
            mutated7 = [value[0, keep].clone() for value in mutated]
            for a, b in zip(original7, mutated7): assert torch.equal(a, b)
            image_shape = tuple(pack['nested_tensor_shape'][-2:])
            q7, boxes7, seeds7 = retained_teacher(*mutated7, image_shape, (H, W), valid[pos7])
            assert q7 == fold['selected_query']
            q7_diff_full8 += int(q7 != full8_query)
            altered_full8 = int(mutated[0][0].sigmoid().mean(0).max(-1).values.argmax())
            full8_selection_changed_by_mutation += int(altered_full8 != full8_query)
            rebuilt_bank = build(saved['source'], saved['frame_ids'], features, valid, pos7,
                                 seeds7, boxes7, event, (H, W), affinity_cache)
            for name, value in rebuilt_bank.items(): same(value, fold[name], name)
            # Full direct cross-prediction check uses only Source features and
            # already fixed candidate boxes; the held native mask is not read.
            predicted = torch.zeros_like(fold['predicted_held'])
            for node in range(frames):
                if node == positions[held]: continue
                affinity = source_only_affinity(features[node], features[positions[held]],
                    valid[node], valid[positions[held]], source_node=node, held_node=positions[held])
                predicted[node] = predict_held_support(rebuilt_bank['boxes'][node], affinity,
                    source_shape=(H, W), held_shape=(H, W))
            same(predicted, fold['predicted_held'], 'predicted_held')
            predicted_checks += 1
            rebuilt += 1
            local_rebuilt += 1
        summaries.append(dict(query='H' + str(ordinal + 1).zfill(3), held_folds=local_rebuilt,
                              all_proposal_fields_bit_exact=True,
                              all_Source_only_cross_predictions_bit_exact=True))
        del pack, ev, raw
    assert fold_checks == 512 and source_checks == 64 and rebuilt == predicted_checks == 24
    assert validate('proposals', lock) == seal
    assert sha(seal_path) == initial_seal_sha and sha(TR / 'lock.json') == initial_lock_sha
    assert not torch.cuda.is_initialized()
    proof = dict(
        status='passed_downstream_teacher_output_holdout_verification', queries=64, folds=512,
        mutated_queries=3, mutated_folds=24, proposal_fields_bit_exact_folds=rebuilt,
        Source_only_cross_prediction_bit_exact_folds=predicted_checks,
        raw_Source_baseline_bit_exact_queries=source_checks,
        held_excluded_from_observed_and_parent_support_folds=fold_checks,
        proposal_has_memory_fields=False, held_native_target_used_for_reconstruction=False,
        held_mask_feedback_evaluations=0,
        mutation='only held logits/boxes/masks replaced with finite extreme values before keep7 slicing',
        query_selection_checks=dict(retained_query_differs_from_full8=q7_diff_full8,
            full8_query_changes_under_held_mutation=full8_selection_changed_by_mutation,
            denominator=24, diagnostic_only_no_proposal_dependency=True),
        anonymous_checks=summaries, proposal_seal_sha256=initial_seal_sha,
        lock_sha256=initial_lock_sha, fingerprint=lock['fingerprint'],
        verifier_sha256=sha(__file__),
        boundaries=dict(GT_read=False, model_calls=0, expert_calls=0, decoder_calls=0,
                        RGB_reads=0, new_paths=0, adaptation_updates=0, CUDA_initialized=False),
        interpretation='teacher-reference cross-validation/proposal holdout; not an independent seven-frame teacher or GT oracle',
        teacher_context_limit='cached raw outputs retain original eight-frame tracking and temporal-fusion context',
        seconds=time.perf_counter() - begin)
    dest = ROOT / 'results/score_audit/leakage.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    os.replace(temp, dest)
    print(json.dumps({k: proof[k] for k in ('status', 'queries', 'folds', 'mutated_folds', 'seconds')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True,
                        help='Verify existing sealed proposals; performs no model or GT evaluation')
    parser.parse_args()
    main()

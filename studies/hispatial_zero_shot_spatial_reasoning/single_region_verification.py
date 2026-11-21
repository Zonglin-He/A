"""Score individual regions with the frozen HiSpatial decoder on the fixed EmbSpatial panel."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from download import STUDY, read_json, sha256, write_json
from geometry_cohort import stable
from object_region_plan import CONFIG as BANK_CONFIG, DIAG, RUN as BANK, index_rows, read_rgb, render_regions
from oracle_region_binding import RUN as ORACLE, verify_files
from score import jsonl


CONFIG = STUDY / 'SINGLE_REGION_VERIFICATION.json'
RUN = BANK.parent / 'hispatial_single_region_verification_20261121'
BATCH_SIZE = 8
METHODS = ('raw_rank', 'independent', 'joint', 'complement_joint', 'swapped_joint', 'matched_budget_joint')


def concrete(name):
    return name.strip().casefold() != 'unknown'


def panel():
    seal = verify_files(BANK, 'input_seal.json')
    assert seal['protocol'] == sha256(BANK_CONFIG) and not seal['label_access']
    roster = {q: row for q, row in index_rows(BANK / 'roster.jsonl').items() if row['order_control']}
    regions = read_json(BANK / 'regions.json')
    key_set = {item['key'] for item in roster.values()}
    regions = {key: regions[key] for key in key_set}
    assert len(roster) == 96 and sum(concrete(name) for row in roster.values() for name in row['options']) == 379
    assert all(4 <= len(frame['marked']) <= 8 for frame in regions.values())
    return roster, regions


def read_support(key, assets):
    path = BANK / 'views' / f'{key}.support.npy'
    assert sha256(path) == assets[key]['supports']
    packed = np.load(path, allow_pickle=False)
    return np.unpackbits(packed, axis=1, count=448*448).astype(bool).reshape(len(packed), 448, 448)


def physical_groups(masks, limits):
    n = len(masks)
    groups = np.arange(n)
    areas = masks.sum(axis=(1, 2))
    for i in range(n):
        for j in range(i):
            common = np.count_nonzero(masks[i] & masks[j])
            if (common / (areas[i] + areas[j] - common) > limits['marked_max_iou'] or
                    common / min(areas[i], areas[j]) > limits['marked_max_containment']):
                groups[groups == groups[i]] = groups[j]
    return groups.tolist()


def assign(log_odds, groups, mode):
    if mode == 'raw' or mode == 'independent':
        slots = log_odds.argmax(axis=1)
        return [int(j) if mode == 'raw' or log_odds[i, j] > 0 else None for i, j in enumerate(slots)]
    n = log_odds.shape[0]
    members = [np.flatnonzero(np.asarray(groups) == c) for c in sorted(set(groups))]
    representatives = np.column_stack([slots[log_odds[:, slots].argmax(axis=1)] for slots in members])
    best = np.take_along_axis(log_odds, representatives, axis=1)
    scores = np.column_stack([np.where(best > 0, best, -np.inf), np.zeros((n, n))])
    rr, cc = linear_sum_assignment(scores, maximize=True)
    selected = [None] * n
    for i, j in zip(rr, cc, strict=True):
        if j < len(members):
            selected[i] = int(representatives[i, j])
    return selected


def job_layout(roster, regions):
    jobs = []
    for q, row in roster.items():
        for option, name in enumerate(row['options']):
            if concrete(name):
                for slot in range(len(regions[row['key']]['marked'])):
                    jobs.append(dict(key=row['key'], question=q, option=option, slot=slot, name=name,
                                     id=f'{q}/{option}/{slot}'))
    return sorted(jobs, key=lambda r: (r['key'], r['slot'], r['question'], r['option']))


def bindings(roster, regions, groups, responses):
    budget4 = {
        f'{q}/{i}' for q, i in sorted(
            ((q, i) for q, row in roster.items() for i, name in enumerate(row['options']) if concrete(name)),
            key=lambda key: stable(f'verify-budget-v1/{key[0]}/{key[1]}')
        )[:325]
    }
    rows = []
    calls = 0
    baseline_file = DIAG / 'condition_predictions.jsonl'
    assert sha256(baseline_file) == read_json(DIAG / 'predictions_seal.json')['files'][baseline_file.name]
    baseline = index_rows(baseline_file)
    for q, item in roster.items():
        key = item['key']
        candidates = regions[key]['marked']
        n = len(candidates)
        options = [i for i, name in enumerate(item['options']) if concrete(name)]
        positive = np.array([[responses[f'{q}/{i}/{j}']['positive'] for j in range(n)] for i in options])
        negative = np.array([[responses[f'{q}/{i}/{j}']['complement'] for j in range(n)] for i in options])
        budgeted = positive.copy()
        counts = [4 if f'{q}/{i}' in budget4 else 3 for i in options]
        calls += sum(counts)
        for i, k in enumerate(counts):
            budgeted[i, k:] = -np.inf
        sets = {
            'raw_rank': assign(positive, groups[key], 'raw'),
            'independent': assign(positive, groups[key], 'independent'),
            'joint': assign(positive, groups[key], 'joint'),
            'complement_joint': assign((positive-negative)/2, groups[key], 'joint'),
            'swapped_joint': assign(np.roll(positive, -1, axis=0), groups[key], 'joint'),
            'matched_budget_joint': assign(budgeted, groups[key], 'joint'),
        }
        selected = {name: dict(zip(map(str, options), slots, strict=True)) for name, slots in sets.items()}
        original = baseline[q]['aligned']
        answers = {'original_hispatial': original, 'joint_region': original, 'joint_window': original}
        picked = sets['joint']
        depth_status = 'incomplete'
        if all(j is not None for j in picked):
            assert len(set(picked)) == len(options)
            readouts = {}
            for name in ('region', 'window'):
                depth = sorted(((candidates[j][name+'_z'], i) for j, i in zip(picked, options, strict=True)),
                               reverse=item['topic'] == 'far')
                if len(depth) >= 2 and abs(depth[0][0] - depth[1][0]) > 1e-6:
                    readouts['joint_'+name] = 'ABCD'[depth[0][1]]
            if len(readouts) == 2:
                answers.update(readouts)
                depth_status = 'used'
            else:
                depth_status = 'tie'
        rows.append(dict(id=q, key=key, bindings=selected, physical_groups=groups[key],
                         budget_by_option=dict(zip(map(str, options), counts, strict=True)),
                         answers=answers, depth_status=depth_status))
    assert calls == 1462
    return rows


def infer(model_cache):
    import torch
    from predict import feature, input_prompt, release

    if (RUN / 'prediction_seal.json').exists():
        raise FileExistsError(RUN / 'prediction_seal.json')
    RUN.mkdir(parents=True, exist_ok=True, mode=0o700)
    cfg = read_json(CONFIG)
    oracle = read_json(ORACLE / 'summary.json')
    assert oracle['cohorts']['all']['count'] == 1206
    assert oracle['sensitivity']['independent25']['region_correct'] > oracle['cohorts']['all']['correct']['original_hispatial']
    roster, regions = panel()
    assets = read_json(BANK / 'views_sha256.json')
    frames_file = DIAG / 'frames.json'
    assert sha256(frames_file) == read_json(DIAG / 'inputs_seal.json')['files'][frames_file.name]
    frames = read_json(frames_file)
    limits = read_json(BANK_CONFIG)
    jobs = job_layout(roster, regions)
    jsonl(RUN / 'inputs.jsonl', jobs)
    write_json(RUN / 'input_seal.json', dict(panel=96, choices=379, candidate_checks=len(jobs), labels=False,
        original_bank=sha256(BANK / 'input_seal.json'), config=sha256(CONFIG), inputs=sha256(RUN / 'inputs.jsonl'),
        code=sha256(Path(__file__)), oracle_summary=sha256(ORACLE / 'summary.json'),
        oracle_primary_gate=oracle['semantic_verification_gate']))

    cv2.setNumThreads(1)
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    predictor, moge, runtime = release(model_cache)
    del moge
    torch.cuda.empty_cache()
    assert runtime == read_json(DIAG / 'runtime.json')['runtime']
    token_ids = [predictor.processor.tokenizer.encode(v, add_special_tokens=False) for v in ('Yes', 'No')]
    assert token_ids == [[3553], [1294]]
    yes_id, no_id = [v[0] for v in token_ids]
    groups = {}

    @lru_cache(maxsize=6)
    def load(key):
        rgb, size = read_rgb(frames[key])
        row = regions[key]
        assert size == row['size']
        path = DIAG / 'pointmaps' / f'{key}.npy'
        assert sha256(path) == row['xyz_sha256']
        xyz = torch.from_numpy(np.load(path, allow_pickle=False))
        masks = read_support(key, assets)
        assert len(masks) == len(row['marked'])
        groups[key] = physical_groups(masks, limits)
        return rgb, xyz, masks

    @lru_cache(maxsize=12)
    def prepare(key, slot):
        rgb, xyz, masks = load(key)
        overlay = render_regions(rgb, [masks[slot]], [], [])
        return overlay, xyz, feature(predictor, overlay, xyz)

    @torch.inference_mode()
    def logits(block, polarity):
        prepared = [prepare(r['key'], r['slot']) for r in block]
        prompts = [cfg[polarity+'_prompt'].format(name=json.dumps(r['name'], ensure_ascii=False)) for r in block]
        inputs = predictor.processor(text=[input_prompt(p) for p in prompts], images=[x[0] for x in prepared],
                                     padding=True, return_tensors='pt').to(predictor.device)
        inputs['xyz_values'] = torch.stack([p[1] for p in prepared]).to(predictor.device)
        fused = torch.cat([r[2] for r in prepared], dim=0)
        with patch.object(predictor.model, 'get_image_features', return_value=fused):
            output = predictor.model.generate(**inputs, do_sample=False, max_new_tokens=1,
                return_dict_in_generate=True, output_logits=True)
        log_probs = output.logits[0].float().log_softmax(dim=-1)
        return log_probs[:, [yes_id, no_id]].cpu().numpy()

    check = sorted(jobs, key=lambda r: stable('verify-parity-v1/'+r['id']))[:16]
    parity = []
    for polarity in ('positive', 'complement'):
        for i in range(0, len(check), BATCH_SIZE):
            block = check[i:i+BATCH_SIZE]
            single = np.concatenate([logits([r], polarity) for r in block])
            batch = logits(block, polarity)
            x, y = single[:, 0] - single[:, 1], batch[:, 0] - batch[:, 1]
            parity.append(dict(polarity=polarity, n=len(block), max_log_odds_difference=float(np.max(np.abs(x-y))),
                               same_sign=bool(np.all((x>0)==(y>0)))))
    size = BATCH_SIZE if all(p['same_sign'] and p['max_log_odds_difference'] <= 0.002 for p in parity) else 1
    write_json(RUN / 'runtime.json', dict(model=runtime, candidate_batch=size, parity=parity, token_ids=token_ids,
        generation='non-sampled first raw-token logits, both canonical answers', parity_model_calls=64))
    print('parity', parity, 'candidate batch', size, 'candidates', len(jobs), flush=True)

    start = time.monotonic()
    responses = {}
    progress = 128
    partial = RUN / 'scores.jsonl.tmp'
    with partial.open('w') as stream:
        for i in range(0, len(jobs), size):
            block = jobs[i:i+size]
            for r, pos, neg in zip(block, logits(block, 'positive'), logits(block, 'complement'), strict=True):
                record = dict(id=r['id'], positive=float(pos[0]-pos[1]), complement=float(neg[0]-neg[1]),
                              positive_logprobs=pos.tolist(), complement_logprobs=neg.tolist())
                assert np.isfinite([record['positive'], record['complement']]).all()
                responses[r['id']] = record
                stream.write(json.dumps(record, allow_nan=False)+'\n')
            stream.flush()
            if len(responses) >= progress or len(responses) == len(jobs):
                print('candidate pairs', len(responses), '/', len(jobs), round(time.monotonic()-start,1), 's', flush=True)
                progress += 128
    assert len(responses) == len(jobs) and len(groups) == len(regions)
    partial.replace(RUN / 'scores.jsonl')
    resolved = bindings(roster, regions, groups, responses)
    jsonl(RUN / 'predictions.jsonl', resolved)
    assert not any(p.requires_grad for p in predictor.model.parameters())
    write_json(RUN / 'prediction_seal.json', dict(inputs=sha256(RUN / 'input_seal.json'), label_access=False,
        training_steps=0, logical_model_calls=2*len(jobs), panel_rows=len(resolved), candidate_checks=len(jobs),
        gpu_seconds=time.monotonic()-start, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        files={name:sha256(RUN/name) for name in ('scores.jsonl','predictions.jsonl','runtime.json')}))
    print('sealed', len(resolved), len(responses), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path, required=True)
    infer(parser.parse_args().model_cache)

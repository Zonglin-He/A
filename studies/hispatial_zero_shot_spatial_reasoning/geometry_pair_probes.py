"""Expose binary depth comparisons, with a matched four-choice probe and choice-order reversal."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import time

from download import read_json, sha256, write_json
from geometry_cohort import BASELINE, RUN, read_jsonl
from score import jsonl


def make_inputs():
    seal = read_json(RUN / 'inputs_seal.json')
    for name in ('cases.jsonl', 'metadata.jsonl', 'probe_metadata.jsonl', 'frames.json'):
        assert sha256(RUN / name) == seal['files'][name]
    cases = {r['id']: r for r in read_jsonl(RUN / 'cases.jsonl')}
    meta = {r['id']: r for r in read_jsonl(RUN / 'metadata.jsonl')}
    picked = {r['case_id'] for r in read_jsonl(RUN / 'probe_metadata.jsonl') if r['task'] == 'emb_order'}
    assert len(picked) == 96
    rows = []
    for owner in sorted(picked):
        raw = cases[owner]
        row = meta[owner]
        assert row['benchmark'] == 'embspatial' and len(row['options']) == 4
        relation = 'farther from' if row['topic'] == 'far' else 'closer to'
        rows.append(dict(id=owner + '/full', row_id=owner, order=None, key=raw['key'], prompt=raw['prompt']))
        for first, second in combinations(range(4), 2):
            for order in ((first, second), (second, first)):
                names = [row['options'][j] for j in order]
                prompt = (f'Which of these two objects is {relation} the camera by depth? '
                          f'Answer with one letter.\nA. {names[0]}\nB. {names[1]}')
                rows.append(dict(id=f'{owner}/pair/{order[0]}{order[1]}', row_id=owner, order=list(order),
                                 key=raw['key'], prompt=prompt))
    assert len(rows) == 96 * (1 + 6 * 2)
    jsonl(RUN / 'pair_probe_inputs.jsonl', rows)
    write_json(RUN / 'pair_probe_inputs_seal.json', dict(input_sha256=sha256(RUN / 'pair_probe_inputs.jsonl'),
        main_input_seal_sha256=sha256(RUN / 'inputs_seal.json'), official_rows=len(picked),
        probe_rows=len(rows), labels_in_inputs=False))


def predict(model_cache):
    from functools import lru_cache

    import cv2
    import numpy as np
    import torch

    from predict import BATCH_N, IMG_SIZE, batch_query, cached_single, feature, read_image, release

    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    seal = read_json(RUN / 'pair_probe_inputs_seal.json')
    pred_seal = read_json(RUN / 'predictions_seal.json')
    assert seal['main_input_seal_sha256'] == pred_seal['input_seal_sha256'] == sha256(RUN / 'inputs_seal.json')
    assert sha256(RUN / 'pair_probe_inputs.jsonl') == seal['input_sha256']
    main_input = read_json(RUN / 'inputs_seal.json')
    assert sha256(RUN / 'frames.json') == main_input['files']['frames.json']
    assert sha256(RUN / 'pointmap_sha256.json') == pred_seal['files']['pointmap_sha256.json']
    cases = sorted(read_jsonl(RUN / 'pair_probe_inputs.jsonl'), key=lambda r: (r['key'], r['id']))
    keys = {r['key'] for r in cases}
    frames = read_json(RUN / 'frames.json')
    map_hashes = read_json(RUN / 'pointmap_sha256.json')
    assert all(sha256(RUN / 'pointmaps' / f'{key}.npy') == map_hashes[key] for key in keys)
    model, _, runtime = release(model_cache)
    assert runtime == read_json(RUN / 'runtime.json')['runtime']

    @lru_cache(maxsize=8)
    def prepare(key):
        frame = frames[key]
        assert frame['benchmark'] == 'embspatial'
        image = cv2.resize(read_image(BASELINE / 'embspatial' / 'images', frame['image_sha256'], frame['transform']), (IMG_SIZE, IMG_SIZE))
        depth = torch.from_numpy(np.load(RUN / 'pointmaps' / f'{key}.npy', allow_pickle=False))
        return image, depth, feature(model, image, depth)

    sample = cases[::max(1, len(cases)//16)][:16]
    size, equal = BATCH_N, 0
    for index in range(0, len(sample), BATCH_N):
        group = sample[index:index+BATCH_N]
        features = [prepare(r['key']) for r in group]
        shared = batch_query(model, features, group)
        direct = [cached_single(model, f, r['prompt']) for f, r in zip(features, group, strict=True)]
        equal += sum(a == b for a, b in zip(shared, direct, strict=True))
    if equal != len(sample):
        size = 1
    print('auxiliary batch parity', equal, '/', len(sample), 'batch', size, flush=True)
    partial = RUN / 'pair_probe_predictions.tmp'
    start = time.monotonic()
    with partial.open('w') as stream:
        for index in range(0, len(cases), size):
            segment = cases[index:index+size]
            features = [prepare(row['key']) for row in segment]
            if len(segment) > 1:
                output = batch_query(model, features, segment)
            else:
                output = [cached_single(model, features[0], segment[0]['prompt'])]
            for row, response in zip(segment, output, strict=True):
                stream.write(json.dumps(dict(id=row['id'], response=response), ensure_ascii=False) + '\n')
            stream.flush()
            if index % 100 == 0 or index+size >= len(cases):
                print('pair probes', min(index+size, len(cases)), '/', len(cases), round(time.monotonic()-start, 1), 's', flush=True)
    partial.replace(RUN / 'pair_probe_predictions.jsonl')
    write_json(RUN / 'pair_probe_predictions_seal.json', dict(predictions=sha256(RUN / 'pair_probe_predictions.jsonl'),
        inputs=seal['input_sha256'], batch_size=size, batch_anchor_matches=equal, ground_truth_read=False))


def analyze():
    from geometry_audit import select_letter

    seal = read_json(RUN / 'pair_probe_predictions_seal.json')
    assert seal['ground_truth_read'] is False
    assert seal['predictions'] == sha256(RUN / 'pair_probe_predictions.jsonl')
    assert seal['inputs'] == sha256(RUN / 'pair_probe_inputs.jsonl')
    primary = read_json(RUN / 'audit_summary.json')
    assert primary['result_files']['emb_box_depth.jsonl'] == sha256(RUN / 'emb_box_depth.jsonl')
    assert primary['result_files']['paired_rows.jsonl'] == sha256(RUN / 'paired_rows.jsonl')
    metadata = {r['id']: r for r in read_jsonl(RUN / 'emb_box_depth.jsonl')}
    responses = {r['id']: r['response'] for r in read_jsonl(RUN / 'pair_probe_predictions.jsonl')}
    inputs = read_jsonl(RUN / 'pair_probe_inputs.jsonl')
    assert len(responses) == len(inputs) and responses.keys() == {r['id'] for r in inputs}
    items = defaultdict(list)
    for row in inputs:
        items[row['row_id']].append(row)
    summaries, details = defaultdict(Counter), []
    for owner, cases in items.items():
        row = metadata[owner]
        official = row['official_answer']
        full_row = next(r for r in cases if r['order'] is None)
        response = responses[full_row['id']]
        main_prediction = select_letter(response, 'ABCD')
        full_correct = official in response[:2]
        judgments = defaultdict(list)
        for r in cases:
            order = r['order']
            if order is None:
                continue
            choice = select_letter(responses[r['id']], 'AB')
            judgments[tuple(sorted(order))].append(order['AB'.index(choice)] if choice else None)
        stable = {key: p[0] if p[0] is not None and p[0] == p[1] else None for key, p in judgments.items()}
        wins = Counter(v for v in stable.values() if v is not None)
        tournament = next((int(index) for index, n in wins.items() if n == 3), None)
        stable_comp = None
        if main_prediction and main_prediction != official:
            pair = tuple(sorted(('ABCD'.index(main_prediction), 'ABCD'.index(official))))
            winner = stable[pair]
            stable_comp = ('unstable' if all(v is not None for v in judgments[pair]) else 'unparsed') if winner is None else (
                'gold_wins' if 'ABCD'[winner] == official else 'distractor_wins')
        localization = {x['option']: x['localization'] for x in row['localization_probes']}
        both_compatible = all(localization.get(k) in ('contains_box_center', 'overlaps_box') for k in (official, main_prediction))
        both_box_geometry_correct = row['bbox_moge_prediction'] == row['bbox_center_prediction'] == official
        details.append(dict(id=owner, topic=row['topic'], official_answer=official, four_way_response=response,
                            four_way_prediction=main_prediction, four_way_correct=full_correct,
                            box_geometry_correct=both_box_geometry_correct, both_grid_compatible=both_compatible,
                            pair_comparison=stable_comp, binary_winners={''.join(map(str,k)): v for k, v in stable.items()},
                            tournament_prediction='ABCD'[tournament] if tournament is not None else None))
        for scope in (row['topic'], 'both'):
            count = summaries[(scope, 'four_way_correct' if full_correct else 'four_way_wrong')]
            count['questions'] += 1
            count['tournament_valid'] += tournament is not None
            count['tournament_correct'] += tournament is not None and 'ABCD'[tournament] == official
            if stable_comp:
                count['paired_gold_and_distractor_' + stable_comp] += 1
                if both_box_geometry_correct and both_compatible:
                    count['box_and_grid_compatible_' + stable_comp] += 1
    jsonl(RUN / 'pair_probe_comparison_rows.jsonl', details)
    write_json(RUN / 'pair_probe_summary.json', dict(probe_population=96, options_counterbalanced=True,
        predictions_sha256=seal['predictions'], probe_row_scores_sha256=sha256(RUN / 'pair_probe_comparison_rows.jsonl'),
        results={scope: {key: dict(count) for (s, key), count in summaries.items() if s == scope} for scope in ('close', 'far', 'both')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'predict', 'analyze'))
    parser.add_argument('--model-cache', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        make_inputs()
    elif args.action == 'predict':
        if args.model_cache is None:
            parser.error('--model-cache is required for predict')
        predict(args.model_cache)
    else:
        analyze()

"""Verify RGB separation and repeat the frozen score with byte-identical RGB removed."""
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from evaluate_inference import paired, paired_iou, panel, sealed_models
from inference_stats import cluster_weights, intervals
from prepare import RUN, digest, load_blind, publish, read


PAIRWISE = re.compile(r'\b(behind|in front of|next to|between|beside|opposite|adjacent to)\b', re.I)


def anchor_counterfactual(rows):
    a, b = 'prefix_metric', 'prefix_wrong'
    matched = [r for r in rows if a in r['shape'] and b in r['shape']]
    weights = cluster_weights(matched)
    errors = {}
    for key in ('size', 'yaw', 'center'):
        delta = np.array([r['shape'][a][key]-r['shape'][b][key] for r in matched])
        errors[key] = dict(mean_delta=float(delta.mean()), ci=intervals(delta, weights))
    return dict(n=len(matched), errors=errors)


def main():
    _, _, _, plan = sealed_models()
    source, *_ = load_blind()
    test = defaultdict(list)
    for unit in plan['units']:
        path = Path(source[unit['uid']]['frame']['rgb'])
        test[digest(path)].append(unit)
    examples = set()
    recordings = {tuple(x) for x in plan['test_recordings']}
    for group in plan['donors'].values():
        for arm in ('random', 'verified', 'cross'):
            for donor in group[arm]:
                assert (donor['platform'], donor['recording']) not in recordings
                examples.add(digest(Path(donor['rgb'])))
    assert set(test).isdisjoint(examples)

    result = read(RUN/'aggregate.json')
    score_path = RUN/'scores-private.json'
    assert result['score'] == digest(score_path)
    assert result['geometry']['artifact'] == digest(RUN/'geometry-private.json')
    score_rows = read(score_path)
    assert [r['uid'] for r in score_rows] == [u['uid'] for u in plan['units']]
    keep = {same[0]['uid'] for same in test.values()}
    subset = [r for r in score_rows if r['uid'] in keep]
    primary = panel(subset)
    drone = panel([r for r in subset if r['target'] == 'drone'])
    lexical = [r for r in score_rows if PAIRWISE.search(source[r['uid']]['frame']['text'])]
    graph = dict(n=len(lexical),
        versus_som=paired_iou(lexical,'graph','som'), versus_wrong=paired_iou(lexical,'graph','graph_shuffled'),
        identity_versus_som=paired(lexical,'identity','graph','som'),
        identity_versus_wrong=paired(lexical,'identity','graph','graph_shuffled'),
        changed_id_after_shuffle=sum(r['choices']['graph']!=r['choices']['graph_shuffled'] for r in score_rows))
    publish(RUN/'rgb-audit.json', dict(
        test_paths=len(score_rows), unique_test_rgb=len(test), unique_example_rgb=len(examples),
        duplicate_group_sizes=sorted(len(same) for same in test.values() if len(same)>1),
        byte_overlap=0, recording_overlap=0, sensitivity=primary, drone_sensitivity=drone,
        exploratory_graph_pairwise=graph, matched_anchor_counterfactual=anchor_counterfactual(score_rows)))
    print('RGB-content audit', len(score_rows), 'test paths,', len(test),
          'distinct images,', len(examples), 'distinct examples, zero overlaps.')


if __name__ == '__main__':
    main()

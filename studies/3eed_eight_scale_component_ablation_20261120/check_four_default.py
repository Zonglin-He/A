"""Replay the deployed four-scale readout against the sealed no-reference-vote arm."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from common import ROOT, OUT, WORK, catalog, digest, prior, read, row_id, save
from crossplatform.ground_regions import Grounder, regions, selection
from predict_delivery import to_model
from scope183 import ordinary, frame
from score_roles import sensor_frame_box


def compact_seal():
    path = WORK / 'COMPACT_SEAL.json'
    assert digest(path) == read(OUT / 'compression_check.json')['prediction_seal']
    seal = read(path)
    assert not seal['labels_used']
    return seal


def check_frame(task):
    info, items = task
    assert digest(info['lex']) == info['lh'] and digest(info['word']) == info['wh']
    with np.load(info['lex']) as lex, np.load(info['word']) as word:
        part = regions.build(lex['sample'], lex['logits'], lex['xyz'], info['f'])
        assert len(part['cuts']) == 4
        data = regions.assemble(part, word)
    points = regions.Returns(data['xyz'])
    count = 0
    for row, checksum in items:
        assert digest(row['role']) == row['rh'] and digest(row['prior']) == row['ph']
        path = WORK / 'compact' / (row_id(row) + '.npz')
        assert digest(path) == checksum
        with np.load(row['prior']) as base, np.load(row['role']) as roles, np.load(path) as expected:
            backup = to_model(row, base)
            proposals = {k: roles[k] for k in roles.files if k.startswith('t_')}
            nodes = selection.select(data, proposals, row['words'], int(roles['side']),
                                     int(roles['depth']), bool(roles['immediate']), points)
            for node, word, name, old in zip(nodes, row['words'], ('subject', 'reference'),
                                            ('anchored', 'ref_anchored')):
                actual = backup[old][0] if node < 0 else data['completed'][node, word]
                # Entry 1 is the sealed combination, entry 0 includes reference votes.
                np.testing.assert_array_equal(actual, expected[name][1])
            count += 1
    return count


def replay():
    c = catalog()
    files = compact_seal()['files']
    groups = defaultdict(list)
    counts = defaultdict(int)
    for row in c['rows']:
        if row['scope'] in ('ordinary', 'noun'):
            groups[row['frame']].append((row, files[row_id(row)]))
            counts[row['scope']] += 1
    assert set(files) == {row_id(r) for items in groups.values() for r, _ in items}
    with ProcessPoolExecutor(max_workers=6) as pool:
        checks = sum(pool.map(check_frame, ((c['frames'][k], v) for k, v in groups.items()), chunksize=3))
    assert checks == sum(counts.values())
    save(OUT / 'four_default_regions.json', dict(records=counts, frame_source_pairs=len(groups),
         endpoint_boxes_checked=2*checks, equal_coordinate_values_including_nan=True,
         input_proposals='subject only', target_labels=False, FINAL_access=False))


def example_cases():
    records = ordinary()
    c = catalog()
    files = compact_seal()['files']
    by_direction = defaultdict(list)
    for r in c['rows']:
        if r['scope'] == 'ordinary':
            by_direction[r['source'], r['target']].append(r)
    for (src, tgt), rows in sorted(by_direction.items()):
        # Choose a discriminating example by sealed prediction disagreement alone.
        for r in sorted(rows, key=lambda item: item['ordinal']):
            path = WORK / 'compact' / (row_id(r) + '.npz')
            assert digest(path) == files[row_id(r)]
            with np.load(path) as expected:
                if not np.array_equal(expected['subject'][0], expected['subject'][1]):
                    item = records[tgt][r['ordinal']]
                    yield src, 'relation', frame(item), sensor_frame_box(expected['subject'][1], tgt, item['suffix'])
                    break
        else:
            raise ValueError(f'No discriminating relation example for {src} to {tgt}')
        folder = prior.FULL / 'predictions' / (src + '_to_' + tgt)
        assert digest(folder/'all.npz') == read(folder/'SEALED.json')['all']
        with np.load(folder/'all.npz') as old:
            ix = int(np.flatnonzero(~old['registered'])[0])
            expected = old['completed'][ix, prior.old.ARMS.index('shared_cut')]
            yield src, 'fallback', frame(records[tgt][ix]), expected
    examples = {(v['source'],v['uid']):v['frame'] for v in prior.noun_inputs()}
    for src in sorted({r['source'] for r in c['rows'] if r['scope']=='noun'}):
        r = next(r for r in c['rows'] if r['scope']=='noun' and r['source']==src)
        f = examples[src, r['uid']]
        path = WORK / 'compact' / (row_id(r) + '.npz')
        assert digest(path) == files[row_id(r)]
        with np.load(path) as expected:
            from scope183 import VIEWS
            yield src, 'ordered', f, sensor_frame_box(expected['subject'][1], f['target'], VIEWS.index(f['waymo_view']))


def check_entry(package):
    import torch
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    results = []
    engine = None
    current = None
    for src, form, inputs, expected in example_cases():
        if package is not None:
            if any(v['form']==form for v in results):
                continue
            cmd = [sys.executable, '-I', str(package / 'crossplatform/ground_regions.py'),
                   '--asset-root', str(ROOT), '--source', src]
            for k,v in inputs.items():
                cmd.extend(('--'+k.replace('_','-'), str(v)))
            with TemporaryDirectory() as tmp:
                p = subprocess.run(cmd, cwd=tmp, env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'},
                                   capture_output=True, text=True, check=True)
            out = json.loads(p.stdout)
            assert out['frame'] == 'target_lidar'
            actual = np.asarray(out['box'])
            record = {}
        else:
            if current != src:
                engine = None
                engine = Grounder(src)
                current = src
            with patch.object(engine._frontend, 'score_frame', wraps=engine._frontend.score_frame) as front:
                actual = engine.predict(**inputs)
            total = front.call_count
            assert total == (1 if form=='fallback' else 2)
            record = dict(extra_phrase_forwards=total-1)
        error = float(abs(np.asarray(actual)-expected).max())
        assert error < 1e-5, (src,inputs['target'],form,error)
        results.append(dict(source=src,target=inputs['target'],form=form,max_error=error,**record))
    name = 'four_default_public_cli' if package is not None else 'four_default_entry'
    save(OUT / (name + '.json'), dict(cases=results, max_error=max(r['max_error'] for r in results),
         selection_used_target_labels=False, FINAL_access=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('regions','entry','cli'))
    parser.add_argument('--package', type=Path)
    args = parser.parse_args()
    if args.mode == 'regions':
        replay()
    else:
        if args.mode == 'cli' and args.package is None:
            parser.error('cli requires --package')
        check_entry(args.package if args.mode=='cli' else None)

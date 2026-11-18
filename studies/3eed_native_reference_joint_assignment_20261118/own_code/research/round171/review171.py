"""Prepare and seal the new prediction-blind endpoint checks."""
from collections import Counter
import hashlib

import numpy as np
from PIL import Image

from scope171 import HERE, INPUT, R170, SPEC, WORK, atomic_json, lock, read, sha, status
from calibrate168 import image_geometry, render_path
from mechanisms162 import annotation_inventory
from predict171 import FILES
from review170 import STATUSES


REVIEW = WORK / 'blind_review_PRIVATE'
PLATES = REVIEW / 'plates'
KEYS = REVIEW / 'BOX_KEYS_PRIVATE.json'
TRUTH = REVIEW / 'ADJUDICATION_PRIVATE.json'


def verify_predictions():
    pin = lock('prediction', FILES)
    good = read(HERE / 'PREDICTION_SEAL.json')
    assert good['input_seal'] == sha(HERE / 'INPUT_SEAL.json') and good['design'] == pin
    assert good['prior_predictions'] == sha(R170 / 'PREDICTION_SEAL.json')
    assert not good['FINAL_access'] and not good['new_truth_read_by_prediction']
    assert good['source_fingerprints_unchanged'] and good['r170_all_five_endpoints_and_associations_bit_exact']
    assert len(good['files']) == sum(good['rows'].values()) == 168
    assert all(sha(WORK / 'predictions' / k) == v for k, v in good['files'].items())
    return good


def prepare():
    design = lock('visual_review', (*FILES, 'review171.py'))
    pred = verify_predictions()
    assert sha(INPUT) == read(HERE / 'INPUT_SEAL.json')['registry'] and not KEYS.exists()
    PLATES.mkdir(parents=True, mode=0o700, exist_ok=True)
    status('blind_review', new_frames=pred['rows']['new_frames'] // 2, replay_frames=pred['rows']['replay170'] // 2)
    inventory = []
    files = {}
    seed = read(SPEC)['sampling_seed']
    for entry in read(INPUT):
        case = entry['case']
        row = entry['row']
        names, boxes = annotation_inventory(row['metapath'], row['platform'], int(row['suffix'] or 0), row['object_index'])
        token = hashlib.sha256(f'{seed}|blind-key|{case}'.encode()).digest()
        perm = np.random.default_rng(int.from_bytes(token[:8], 'little')).permutation(len(names))
        lo, hi = entry['parsed']['graph']['entities'][entry['role']]['span']
        phrase = row['caption'][lo:hi]
        context = row['caption'][max(0, lo - 11):hi + 46]
        with Image.open(row['imagepath']) as img:
            render_path(case, row, image_geometry(row), [names[i] for i in perm], boxes[perm],
                        [f'Reference mention  {phrase}', f'Context at the selected span  {context}'],
                        img.convert('RGB'), PLATES)
        inventory.append(dict(case=case, display_to_annotation=perm.tolist(), inventory_size=len(names)))
        with Image.open(PLATES / f'{case:02}.jpg') as img:
            img.thumbnail((990, 860))
            img.save(PLATES / f'{case:02}_compact.png')
        for filename in (f'{case:02}.jpg', f'{case:02}_raw.jpg', f'{case:02}_compact.png'):
            files[filename] = sha(PLATES / filename)
    assert len(inventory) * 2 == pred['rows']['new_frames']
    atomic_json(KEYS, inventory)
    atomic_json(HERE / 'REVIEW_PANEL_SEAL.json', dict(n=len(inventory), design=design, keys=sha(KEYS),
                prediction_seal=sha(HERE / 'PREDICTION_SEAL.json'), files=files, displayed_model_outputs=False,
                shuffled_inventory=True, FINAL_access=False))
    print('R171 new prediction-blind sheets', len(inventory), flush=True)


def seal():
    design = lock('visual_review', (*FILES, 'review171.py'))
    verify_predictions()
    panel = read(HERE / 'REVIEW_PANEL_SEAL.json')
    assert panel['design'] == design and panel['keys'] == sha(KEYS)
    assert panel['prediction_seal'] == sha(HERE / 'PREDICTION_SEAL.json') and not panel['displayed_model_outputs']
    assert all(sha(PLATES / k) == v for k, v in panel['files'].items())
    counts = Counter()
    n = 0
    for key, row in zip(read(KEYS), read(TRUTH), strict=True):
        assert key['case'] == row['case']
        for role in ('subject', 'reference'):
            item = row[role]
            kind = item['status']
            assert kind in STATUSES
            assert (0 <= item.get('box', -1) < key['inventory_size']) == (kind == 'verified')
            counts[(role, kind)] += 1
        if row['subject']['status'] == row['reference']['status'] == 'verified':
            assert row['subject']['box'] != row['reference']['box']
            counts[('both', 'verified')] += 1
        n += 1
    assert n == panel['n']
    result = dict(n=n, reader_count=1, prediction_and_official_subject_slot_blind=True, all_frames_retained=True,
                  adjudication=sha(TRUTH), panel_seal=sha(HERE / 'REVIEW_PANEL_SEAL.json'),
                  statuses={f'{r}:{k}': v for (r, k), v in sorted(counts.items())}, FINAL_access=False)
    atomic_json(HERE / 'VISUAL_TRUTH_SEAL.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('prepare', 'seal'))
    {'prepare': prepare, 'seal': seal}[p.parse_args().phase]()

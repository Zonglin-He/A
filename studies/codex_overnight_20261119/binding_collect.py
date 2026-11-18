"""Seal transported role evidence in the source calibration and fixed ordinary panels."""
import argparse
from collections import Counter
import fcntl
import gc
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from edge_predict import HERE, ROOT, R171, OLD_ARMS, INPUT, PRIOR_INPUT, PLATFORMS, SOURCE_SETTINGS, SPLITS, WORK
from edge_predict import atomic_json, read, sha
from binding import factorize, transfer_scores

PRIVATE = HERE / 'private/binding'
SPEC = HERE / 'binding_protocol.json'
ARMS = ('r158_pca','r171_independent','r171_joint','additive_independent','additive_geometric',
        'fixed_exchange_joint','fixed_exchange_mismatched','source_fit_joint','source_fit_mismatched')


def checked_inputs():
    from calibrate168 import CAL, edges_of, parsed
    from panel169 import rewrite
    from retrieval167 import direct_edges

    roster = CAL / 'ROSTER.json'
    certificate = read(ROOT / 'research/round168/SOURCE_INPUT_SEAL.json')
    assert sha(roster) == certificate['source_roster_sha256'] and not certificate['FINAL_access']
    assert certificate['ordinary_frame_content_group_overlap'] == 0
    data = []
    counts = Counter()
    for item in read(roster):
        row = item['row']
        doc = parsed(row)
        edges = direct_edges(edges_of(doc['graph']))
        options = []
        for role, sign in sorted(edges):
            if str(role) in item['truth']:
                views = rewrite(row['caption'], doc, edges, role, sign)
                if views['status'] == 'supported':
                    options.append((role, sign, views))
        if options:
            role, sign, views = options[0]
            data.append(dict(case=item['case'], source=item['source'], row=row, role=role, sign=sign, rewrite=views,
                             recording=sha_recording(row), ref_id=item['truth'][str(role)]))
            counts[item['source']] += 1
    assert len(data) <= 120
    return data, counts


def sha_recording(row):
    physical = Path(row['metapath']).resolve().parent.parent
    return hashlib.sha256(str(physical).encode()).hexdigest()


def pins(phase):
    local = ('binding_protocol.json', 'binding.py', 'binding_collect.py', 'edges.py')
    shared = ('research/round168/SOURCE_INPUT_SEAL.json', 'research/round171/PREDICTION_SEAL.json',
              'research/round169/requery169.py', 'research/round168/calibrate168.py',
              'crossplatform/single_frame_sources.json', 'research/round162/experiment/SOURCE_PARAMETERS.json',
              'crossplatform/ground_single_frame.py')
    paths = [*(HERE / f for f in local), *(ROOT / f for f in shared)]
    return dict(phase=phase, files={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                protected_data_access=False)


def empty(old):
    ns, nr = len(old['subject_boxes']), len(old['D_natural_boxes'])
    base, delta, edge, sc, rc = factorize(np.full(ns+nr, -np.inf), np.full(ns+nr, -np.inf),
                                         old['subject_unary'], old['D_native_unary'], old['native_left'], old['sign'])
    return dict(baseline=base, offset=delta, edge=edge, subject_common=sc, reference_common=rc,
                f=np.full(ns+nr,-np.inf), d=np.full(ns+nr,-np.inf), f_strength=np.zeros(ns+nr),
                d_strength=np.zeros(ns+nr), left=old['native_left'], sign=old['sign'],
                subject_boxes=old['subject_boxes'], reference_boxes=old['D_natural_boxes'], source_time=0.)


def measure(model, entry, setting, old=None):
    from assignment171 import reference_unary
    from common160 import array_sha
    from predict170 import original_evidence
    from predict171 import camera_relation
    from requery169 import native_donors, native_forward
    from replay168 import oriented
    from retrieval167 import support_matrix
    from crossplatform.ground_single_frame import orient_box

    if old is not None and (entry['rewrite']['status'] != 'supported' or not len(old['D_natural_boxes'])):
        return empty(old)
    t = perf_counter()
    row = entry['row']
    if old is not None:
        assert sha(row['imagepath']) == row['image_sha256'] and sha(row['pointpath']) == row['point_sha256']
    ev, original, _, previous = original_evidence(model, row, setting)
    d = native_forward(model, row, ev.points, ev.indices, None, entry['rewrite']['D'])
    n = len(ev.boxes)
    sindex = previous['bank'][0, previous['bank'][0] >= 0]
    whole = np.r_[np.c_[ev.boxes,np.zeros(n)], previous['novel'][np.isfinite(previous['novel']).all(1)]]
    regions = whole[sindex]
    sbox = oriented(row, previous, ev.boxes, sindex, lambda _: ev.points)
    ids, score, _ = native_donors(d)
    raw = d['z']['boxes'][ids]
    rbox = np.stack([orient_box(ev.points, b) for b in raw])
    assert np.array_equal(original['point_indices'], ev.indices)
    assert np.array_equal(d['z']['point_indices'], ev.indices)
    assert np.array_equal(d['z']['xyz'], original['xyz'])
    seeds = np.sort(np.unique(ev.seeds,axis=0,return_index=True)[1])
    su = previous['score'][0, previous['bank'][0] >= 0]
    ru = reference_unary(score)
    if old is None:
        sm = support_matrix(whole, ev.seeds[seeds], n)[sindex]
        rm = support_matrix(np.c_[raw,np.zeros(len(raw))], ev.seeds[seeds], len(raw))
        left, _, _ = camera_relation(sm, rm, ev.uv[seeds], ev.visible[seeds])
        sign = entry['sign']
    else:
        assert old['sample_sha'] == array_sha(ev.indices) and old['point_sha'] == array_sha(ev.points)
        for saved, got in [('subject_unary',su),('D_native_unary',ru),('subject_boxes',sbox),
                           ('D_natural_boxes',rbox),('seeds',ev.seeds[seeds])]:
            np.testing.assert_array_equal(old[saved], got)
        left, sign = old['native_left'], int(old['sign'])
        assert sign == entry['sign']
    all_regions = np.r_[regions,np.c_[raw,np.zeros(len(raw))]]
    points = np.unique(ev.points,axis=0)
    f, fs = transfer_scores(np.c_[ev.boxes,np.zeros(n)], all_regions, points, np.log(ev.probability))
    native_d = d['z']['boxes']
    r, rs = transfer_scores(np.c_[native_d,np.zeros(len(native_d))], all_regions, points, np.log(d['probability']))
    base, delta, edge, sc, rc = factorize(f,r,su,ru,left,sign)
    return dict(baseline=base, offset=delta, edge=edge, subject_common=sc, reference_common=rc,
                f=f, d=r, f_strength=fs, d_strength=rs, left=left, sign=sign,
                subject_boxes=sbox,reference_boxes=rbox,seeds=ev.seeds[seeds], source_time=perf_counter()-t)


def collect(source_phase):
    import torch
    from evidence162 import EvidenceGrounder
    from common160 import npz_atomic
    from model51 import fingerprint
    from predict171 import read_npz
    from review171 import verify_predictions

    dest = PRIVATE / ('source' if source_phase else 'target')
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    seal = dest / 'SEAL.json'
    assert not seal.exists()
    pin = dest / 'LOCK.json'
    expected = pins('source' if source_phase else 'target')
    if pin.exists():
        assert read(pin) == expected
    else:
        atomic_json(pin, expected)
    if source_phase:
        catalog, counts = checked_inputs()
        atomic_json(dest / 'ROSTER.json', catalog)
        batches = {'source':catalog}
        print('Eligible language source rows', counts, flush=True)
    else:
        old = verify_predictions()
        batches = dict(zip(SPLITS,(read(PRIOR_INPUT),read(INPUT)),strict=True))
    records = Counter()
    files = {}
    settings = read(SOURCE_SETTINGS)['sources']
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            initial = fingerprint(model.base), fingerprint(model.matcher)
            for panel, rows in batches.items():
                (dest / panel).mkdir(exist_ok=True)
                selected = [r for r in rows if (r['source'] == source if source_phase else r['target'] != source)]
                for entry in selected:
                    tag = f"{panel}/{entry['case']:02d}_{source}.npz"
                    if source_phase:
                        prior = None
                    else:
                        old_file = WORK / 'predictions' / tag
                        assert sha(old_file) == old['files'][tag]
                        prior = read_npz(old_file)
                    path = dest / tag
                    if path.exists():
                        assert str(read_npz(path)['pin']) == sha(pin)
                    else:
                        with torch.inference_mode():
                            values = measure(model,entry,settings[source],prior)
                        npz_atomic(path, **values, pin=sha(pin))
                    files[tag] = sha(path)
                    records[source] += 1
                    if records[source] % 10 == 0:
                        print(panel, source, records[source], flush=True)
            assert initial == (fingerprint(model.base), fingerprint(model.matcher))
            del model
            gc.collect()
            torch.cuda.empty_cache()
    assert records == counts if source_phase else files.keys() == old['files'].keys()
    atomic_json(seal, dict(files=files, pin=sha(pin), rows=dict(records),
                           roster=sha(dest / 'ROSTER.json') if source_phase else '',
                           frozen=True,target_label_used=False,protected_data_access=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('source','target'))
    collect(parser.parse_args().phase == 'source')

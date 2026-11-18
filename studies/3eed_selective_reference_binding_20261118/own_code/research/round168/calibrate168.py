"""Source-own, recording-disjoint calibration of the reference reject option."""
import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
import fcntl
import gc
import hashlib
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from scope168 import CAL, HERE, ORDER, OUT, PLATFORMS, ROOT, atomic_json, lock, read, roster, sha, status
from retrieval167 import append_sealed_regions, direct_edges, repaired, support_matrix, support_overlap, transfer
from select168 import gate_logits, role_terms
from labels166 import PLURAL, IRREGULAR_PLURALS, mask_owners, physical_labels
from mechanisms162 import annotation_inventory
from score165 import many_iou, write_table
from audit160 import transform_gt
from review167 import render
from scope166 import MAX_HYPOTHESES, MAX_ROLES
from visual162 import projection
from decode160 import WAYMO_SYNONYMS, M3ED_SYNONYMS
from predict166 import camera as archived_camera


@lru_cache(maxsize=1)
def ordinary_content():
    scans, images, frames, recordings = set(), set(), set(), set()
    folder = ROOT / 'crossplatform/round97'
    root = read(folder / 'MATRIX_SEALED.json')
    for target, rows in roster().items():
        source = next(s for s, t in ORDER if t == target)
        tag = f'{source}_to_{target}'
        base = folder / tag
        assert sha(base / 'SEALED.json') == root['seals_sha256'][tag]
        expected = read(base / 'SEALED.json')['files']
        for row in rows:
            name = f"records_PRIVATE/{row['id']}.json"
            assert sha(base / name) == expected[name]
            geo = read(base / name)['geometry']
            scans.add(geo['point_sha256'])
            images.add(geo['image_sha256'])
            frame = Path(row['metapath']).resolve()
            frames.add(frame)
            recordings.add(frame.parent.parent)
    return scans, images, frames, recordings


def sources(source):
    if source == 'waymo':
        return ROOT / 'research/round76/input/SOURCE_TRAIN512_PRIVATE.json'
    return ROOT / 'research/round97/input' / f'SOURCE_{source.upper()}_TRAIN512_PRIVATE.json'


def parsed(row):
    if row['platform'] == 'waymo':
        path = ROOT / 'crossplatform/round89/source_roles640_attempt001'
        info = read(path / 'cases_PRIVATE' / f"{row['id']}.json")
        info['doc'] = read(path / 'dependencies_PRIVATE' / f"{row['id']}.json")
    else:
        info = read(ROOT / 'crossplatform/round97/roles_PRIVATE' / f"{row['id']}.json")
    assert hashlib.sha256(row['caption'].encode()).hexdigest() == info['caption_sha256']
    return info


def edges_of(graph):
    idx = {x['role']: i for i, x in enumerate(graph['entities'])}
    return np.asarray([(idx[e['subject']], idx[e['reference']], 1 if e['predicate'] == 'LEFT' else -1)
                       for e in graph['relations'] if e['known_frame'] and e['predicate'] in ('LEFT', 'RIGHT')],
                     dtype=np.int16).reshape(-1, 3)


def image_geometry(row):
    tag = row['platform']
    view = int(row['suffix'] or 0)
    intrinsic, model_to_camera, distortion, width, height = archived_camera(row['metapath'], row['imagepath'], tag, view)
    return SimpleNamespace(camera_intrinsic=intrinsic, model_to_camera=model_to_camera, distortion=distortion,
                           width=width, height=height)


def source_instances(row, data, refs):
    tag = row['platform']
    names, boxes = annotation_inventory(row['metapath'], tag, int(row['suffix'] or 0), row['object_index'])
    syn = WAYMO_SYNONYMS if tag == 'waymo' else M3ED_SYNONYMS
    attached = []
    for annotation in read(Path(row['metapath']))['ground_info'][row['object_index']].get('others', []):
        box = annotation.get('bbox_3d_other')
        if box is None or len(box) < 7 or min(box[3:6]) <= 0:
            continue
        raw = np.asarray(box[:7] if tag == 'waymo' else (*box[:7], 0., 0.))
        model = transform_gt(raw, tag, int(row['suffix'] or 0))
        owners = many_iou(boxes, model)
        k = int(owners.argmax())
        if k and owners[k] > .96:
            attached.append(k)
    matches = {}
    for r in refs:
        span = data['graph']['entities'][r]['span']
        phrase = row['caption'][span[0]:span[1]].casefold()
        if PLURAL.search(phrase):
            continue
        word = [w for s in data['doc']['sentences'] for w in s['words'] if w['upos'] in ('NOUN', 'PROPN')
                and isinstance(w.get('start_char'), int) and span[0] <= w['start_char'] and w['end_char'] <= span[1]]
        if not word or any(w['text'].casefold() in IRREGULAR_PLURALS or (w['text'].casefold().endswith('s') and
                      w['text'].casefold() != str(w.get('lemma') or w['text']).casefold()) for w in word):
            continue
        compatible = [i for i, n in enumerate(names) if any(re.search(r'\b' + re.escape(v) + r'\b', phrase)
                      for v in syn.get(n, (n,)))]
        specific = sorted(set(compatible) & set(attached))
        chosen = compatible if len(compatible) == 1 else specific
        if len(chosen) == 1 and chosen[0] != 0:
            matches[r] = chosen[0]
    if matches:
        uv = projection(image_geometry(row), boxes[:, :3])
        matches = {r: i for r, i in matches.items() if ((uv[i] >= 0) & (uv[i] <= 1)).all()}
    return matches


def prepare():
    CAL.mkdir(exist_ok=True, parents=True, mode=0o700)
    assert not (CAL / 'ROSTER.json').exists()
    scans, images, frames, recordings = ordinary_content()
    registry = []
    screening = []
    inputs = {}
    for source in PLATFORMS:
        train_path = ROOT / 'crossplatform/round161' / f'{source}_train_PRIVATE.json'
        train = {Path(v['metapath']).resolve() for v in read(train_path)}
        path = sources(source)
        inputs[source] = dict(train=sha(train_path), subset=sha(path))
        seen = set()
        available = defaultdict(list)
        selected = []
        for row in sorted(read(path), key=lambda r: hashlib.sha256(f"R168-source|{r['id']}".encode()).hexdigest()):
            frame = Path(row['metapath']).resolve()
            assert frame in train and row['platform'] == source
            record = frame.parent.parent
            info = parsed(row)
            edges = edges_of(info['graph'])
            direct = direct_edges(edges)
            refs = sorted({r for r, _ in direct})
            flags = dict(source=source, screen=len(screening) + 1, direct_roles=len(refs), known_roles=0,
                         outcome='no_direct_role')
            if refs:
                if frame in frames or record in recordings:
                    flags['outcome'] = 'ordinary_same_recording'
                elif frame in seen:
                    flags['outcome'] = 'repeated_physical_frame'
                else:
                    semantic = source_instances(row, info, refs)
                    flags['known_roles'] = len(semantic)
                    flags['outcome'] = 'unresolved_identity'
                    if semantic:
                        point, image = sha(Path(row['pointpath'])), sha(Path(row['imagepath']))
                        if point in scans or image in images:
                            flags['outcome'] = 'ordinary_content_alias'
                        else:
                            seen.add(frame)
                            available[str(record)].append((row, semantic, len(screening)))
                            flags['outcome'] = 'eligible'
            screening.append(flags)
        queues = [deque(v) for _, v in sorted(available.items())]
        while len(selected) < 72 and any(queues):
            for q in queues:
                if q:
                    row, truth, index = q.popleft()
                    entry = dict(case=len(registry)+1, row=row, source=source, truth=truth)
                    registry.append(entry)
                    selected.append(entry)
                    screening[index]['outcome'] = 'registered'
                    if len(selected) == 72:
                        break
        print('R168 source eligible', source, len(selected), len(available), flush=True)
        assert len(selected) >= 6
        audited = CAL / 'blind_review'
        audited.mkdir(exist_ok=True)
        for x in selected[:4]:
            row = x['row']
            with Image.open(row['imagepath']) as f:
                rgb = f.convert('RGB')
            names, boxes = annotation_inventory(row['metapath'], source, int(row['suffix'] or 0), row['object_index'])
            info = parsed(row)
            clauses = [f"R{r} | {info['graph']['entities'][r]['phrase']}" for r in x['truth']]
            render_path(x['case'], row, image_geometry(row), names, boxes, clauses, rgb, audited)
    atomic_json(CAL / 'ROSTER.json', registry)
    atomic_json(CAL / 'SCREENING.json', screening)
    receipt = dict(rows=dict(Counter(v['source'] for v in registry)), monitored_source_manifests=inputs,
                   source_roster_sha256=sha(CAL / 'ROSTER.json'), screening_sha256=sha(CAL / 'SCREENING.json'),
                   ordinary_frame_content_group_overlap=0, split='source_own_training_partition_readout_calibration_only',
                   directions_used_to_label_reference=False, FINAL_access=False)
    atomic_json(HERE / 'SOURCE_INPUT_SEAL.json', receipt)
    print('R168 source calibration roster', receipt['rows'], flush=True)


def render_path(case, row, cam, names, boxes, clauses, rgb, out):
    # Reuse the existing output-blind renderer without directing it to any target study.
    import review167
    previous = review167.PANEL
    try:
        review167.PANEL = out
        render(dict(case=case, target=row['platform']), row, cam, names, boxes, clauses, rgb)
    finally:
        review167.PANEL = previous


def frozen_pool(ev, settings):
    from support162 import SceneSupport
    from init166 import pca

    scene = SceneSupport(ev)
    starts = np.full((MAX_ROLES, MAX_HYPOTHESES, 7), np.nan)
    budget = np.zeros((MAX_ROLES, 3), dtype=np.int16)
    for r in range(len(ev.role_mask)):
        if not ev.role_mask[r].any():
            continue
        regions = scene.candidates(r, settings['log_dimensions'], 'scene')
        assert len(regions) <= MAX_HYPOTHESES
        for j, item in enumerate(regions):
            starts[r, j] = pca(ev.points, item.box)
        novel = sum(v.query < 0 for v in regions)
        budget[r] = (len(regions), len(regions) - novel, novel)
    return dict(starts=starts, budget=budget, routes=ev.hypotheses)


def algebra_audit(ev, pred, native, new):
    geometry, _ = append_sealed_regions(ev.boxes, dict(starts=pred['novel'][None],
                      budget=np.asarray([[np.isfinite(pred['novel']).all(1).sum(), 0,
                                            np.isfinite(pred['novel']).all(1).sum()]])))
    keep = np.sort(np.unique(ev.seeds, axis=0, return_index=True)[1])
    masks = support_matrix(geometry, ev.seeds[keep], len(ev.boxes))
    count = masks.sum(1)
    hits, overlap = support_overlap(masks)
    allowed = (count >= 2) & (geometry[:, 3:6] > .12).all(1)
    logits = native['query'].astype(float) @ native['tokens'].astype(float).T / .07
    errors = []
    for r, _ in direct_edges(ev.edges):
        positions = pred['bank'][r] >= 0
        if not positions.any():
            continue
        values = logsumexp(logits[:, ev.role_mask[r]], axis=1) - logsumexp(logits, axis=1)
        transferred = transfer(values - values.max(), hits, overlap, len(ev.boxes))
        mass = (masks @ ev.affinity[r, keep]) / np.sqrt(np.maximum(count, 1))
        density = np.log(np.maximum(mass / max(float(mass[allowed].max()), 1e-12), .025))
        x = (transferred + density)[pred['bank'][r, positions]] / 2
        x -= x.max()
        errors.append(float(np.abs(x - new[r, positions]).max()))
    return max(errors, default=0.)


def capture():
    import torch
    from evidence162 import EvidenceGrounder
    from interface66 import capture as project
    from model51 import fingerprint
    from common160 import npz_atomic
    from replay168 import word_masks

    lock()
    proof = read(HERE / 'SOURCE_INPUT_SEAL.json')
    assert sha(CAL / 'ROSTER.json') == proof['source_roster_sha256']
    registry = read(CAL / 'ROSTER.json')
    settings = read(ROOT / 'research/round162/experiment/SOURCE_PARAMETERS.json')['sources']
    save = CAL / 'features'
    save.mkdir(exist_ok=True)
    status('source_calibration', records=len(registry))
    files, numerical = {}, 0.
    with (ROOT / 'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(1)
        for source in PLATFORMS:
            model = EvidenceGrounder(source)
            state = fingerprint(model.base), fingerprint(model.matcher)
            for item in (x for x in registry if x['source'] == source):
                row = item['row']
                name = f"S{item['case']:03d}.npz"
                path = save / name
                if not path.exists():
                    ev = model.evidence_frame(row)
                    original, _ = project(model.base, ev.points, ev.indices, row['caption'])
                    np.testing.assert_array_equal(ev.boxes, original['boxes'])
                    np.testing.assert_array_equal(ev.seeds, original['xyz'][0])
                    source_parse = parsed(row)
                    np.testing.assert_array_equal(ev.edges, edges_of(source_parse['graph']))
                    masks, valid = word_masks(row, source_parse, len(original['tokens']), model.base.tokenizer)
                    np.testing.assert_array_equal(masks, ev.role_mask)
                    previous = frozen_pool(ev, settings[source])
                    pred = repaired(ev, previous, original['query'], original['tokens'])
                    new, _, f = role_terms(ev.boxes, pred['novel'], ev.seeds, original['query'], original['tokens'],
                                           masks, valid, pred['bank'], pred['score'])
                    error = algebra_audit(ev, pred, original, new)
                    assert error < 1.e-9, ('cached native-score replacement differs from independent fusion', error)
                    npz_atomic(path, bank=pred['bank'], score=pred['score'], feature=f, raw=ev.boxes,
                               novel=pred['novel'], seeds=ev.seeds, new=new, verify=error)
                with np.load(path, allow_pickle=False) as saved:
                    numerical = max(numerical, float(saved['verify']))
                files[name] = sha(path)
            assert state == (fingerprint(model.base), fingerprint(model.matcher))
            print('R168 calibrated features', source, len(files), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    atomic_json(HERE / 'SOURCE_FEATURE_SEAL.json', dict(n=len(files), files=files, target_label_read=False,
                roster=proof['source_roster_sha256'], source_weights_unchanged=True,
                algebra_check_max_abs=numerical, FINAL_access=False))


def fit_rule(x, y):
    mean = x.mean(0)
    scale = np.maximum(x.std(0), 1.e-3)
    z = (x - mean) / scale
    v = np.c_[np.ones(len(z)), z]
    ridge = np.asarray([1.0, 1.0, 1.0]) / len(v)

    def objective(w):
        raw = v @ w
        scalar = np.mean(np.logaddexp(0, raw) - y * raw) + .5 * np.dot(ridge, w * w)
        grad = v.T @ (expit(raw) - y) / len(v) + ridge * w
        return scalar, grad

    result = minimize(objective, np.zeros(3), jac=True, method='L-BFGS-B')
    assert result.success and np.isfinite(result.x).all()
    return dict(mean=mean.tolist(), scale=scale.tolist(), coef=result.x.tolist(), lo=z.min(0).tolist(), hi=z.max(0).tolist())


def calibrate():
    lock()
    proof = read(HERE / 'SOURCE_FEATURE_SEAL.json')
    assert sha(CAL / 'ROSTER.json') == proof['roster'] and not proof['FINAL_access'] and not proof['target_label_read']
    registry = read(CAL / 'ROSTER.json')
    table_rows = []
    actual = defaultdict(list)
    for item in registry:
        row, source, case = item['row'], item['source'], item['case']
        name = f'S{case:03d}.npz'
        path = CAL / 'features' / name
        assert sha(path) == proof['files'][name]
        with np.load(path, allow_pickle=False) as a:
            d = dict(a)
        names, boxes, points, ids = physical_labels(row, d['seeds'])
        extra = d['novel'][np.isfinite(d['novel']).all(1)]
        geo = np.concatenate((np.c_[d['raw'], np.zeros(len(d['raw']))], extra))
        owners = mask_owners(support_matrix(geo, points, len(d['raw'])), ids, len(names))
        for r, gold in item['truth'].items():
            r = int(r)
            mask = d['bank'][r] >= 0
            cand = d['bank'][r, mask]
            top = int(owners[cand[int(d['score'][r, mask].argmax())]]) if len(cand) else -2
            new = int(owners[cand[int(d['new'][r, mask].argmax())]]) if len(cand) else -2
            f = d['feature'][r]
            info = dict(case=case, source=source, role=r, physical_available=int(len(cand)>0),
                        reference_covered=int((owners[cand] == gold).any()), old_correct=int(top == gold),
                        new_correct=int(new == gold), old_eval_unknown=int(top == -1),
                        source_mass_enrichment=float(f[0]) if np.isfinite(f[0]) else '',
                        original_margin=float(f[1]) if np.isfinite(f[1]) else '',
                        full_fit_probability='', leave_recording_out_probability='')
            table_rows.append(info)
            if len(cand):
                actual[source].append((recording_key(row), f, int(top == gold), info))
    rules, summaries = {}, []
    for source in PLATFORMS:
        g, x, y, rows = zip(*actual[source], strict=True)
        x, y, g = np.stack(x), np.asarray(y), np.asarray(g)
        assert len(y) >= 6
        rules[source] = fit_rule(x, y)
        probability = expit(gate_logits(x, rules[source]))
        hold = np.full(len(y), np.nan)
        for key in sorted(set(g)):
            test = g == key
            if (~test).any():
                hold[test] = expit(gate_logits(x[test], fit_rule(x[~test], y[~test])))
        for r, p, h in zip(rows, probability, hold, strict=True):
            r.update(full_fit_probability=float(p), leave_recording_out_probability=float(h) if np.isfinite(h) else '')
        valid_loo = np.isfinite(hold).all()
        summaries.append(dict(source=source, observed_roles=len(y), recordings=len(set(g)),
             original_correct=int(y.sum()), replacement_correct=sum(r['new_correct'] for r in rows),
             coverage=int(sum(r['reference_covered'] for r in rows)),
             source_training_accept=int((probability > .5).sum()),
             loo_accept=int((hold > .5).sum()), loo_correct_when_accepted=int(y[hold > .5].sum()),
             loo_brier=float(np.mean((hold - y) ** 2)) if valid_loo else '',
             loo_prior_brier=(float(np.mean([(float(flag) - y[g != rec].mean()) ** 2 for flag, rec in zip(y,g,strict=True)]))
                               if valid_loo else '')))
    OUT.mkdir(exist_ok=True)
    write_table(OUT / 'source_calibration.csv', table_rows)
    write_table(OUT / 'source_screening.csv', read(CAL / 'SCREENING.json'))
    write_table(OUT / 'source_calibration_summary.csv', summaries)
    atomic_json(HERE / 'ABSTENTION_RULE.json', dict(rules=rules, source_feature_sha256=sha(HERE / 'SOURCE_FEATURE_SEAL.json'),
                source_calibration_sha256=sha(OUT / 'source_calibration.csv'),
                source_models_updated=False, target_label_used=False, FINAL_access=False))
    print('R168 source-only reject calibration', summaries, flush=True)


def recording_key(row):
    return str(Path(row['metapath']).resolve().parent.parent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('prepare', 'capture', 'fit'))
    {'prepare': prepare, 'capture': capture, 'fit': calibrate}[parser.parse_args().phase]()

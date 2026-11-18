"""Select physically distinct, registered ordinary frames using language alone."""
from collections import Counter, defaultdict, deque
import hashlib
import re

from scope171 import HERE, INPUT, PLATFORMS, PRIOR_INPUT, R159, REC_INPUT, SPEC, WORK, atomic_json, lock, read, sha, status
from calibrate168 import edges_of, ordinary_content
from panel169 import rewrite
from retrieval167 import direct_edges
from score165 import anonymizer
from select170 import HORIZONTAL, frame, indices, parser, previously_reviewed


OBJECT_NOUN = re.compile(
    r'\b(?:person|people|pedestrians?|man|men|wom[ae]n|child(?:ren)?|cyclists?|bicycles?|'
    r'bikes?|cars?|vehicles?|vans?|suvs?|sedans?|trucks?|pickups?|wagons?|buses?|motorcycles?)\b',
    re.I,
)


def hash_key(*parts):
    return hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()


def pick(groups, n, used, prior_recordings, counts):
    """Round-robin over available recordings, preferring those outside R170."""
    chosen = []
    for _ in range(n):
        while groups:
            recording = min(groups, key=lambda k: (k in prior_recordings, counts[k], groups[k][0]['rank']))
            item = groups[recording].popleft()
            if not groups[recording]:
                del groups[recording]
            row = item['row']
            ids = (frame(row), row['point_sha256'], row['image_sha256'])
            if any(v in seen for v, seen in zip(ids, used, strict=True)):
                continue
            assert sha(row['pointpath']) == ids[1] and sha(row['imagepath']) == ids[2]
            for v, seen in zip(ids, used, strict=True):
                seen.add(v)
            counts[recording] += 1
            chosen.append(item)
            break
    return chosen


def prepare():
    pin = lock('selection', ('select171.py',))
    status('selecting')
    rule = read(SPEC)
    seed = rule['sampling_seed']
    prior = [e['row'] for e in read(PRIOR_INPUT)]
    prior_recordings = {frame(r).parent.parent for r in prior}
    used = ({frame(r) for r in prior}, {r['point_sha256'] for r in prior}, {r['image_sha256'] for r in prior})
    older = ordinary_content()
    reviewed = previously_reviewed()
    report = read(R159 / 'RECORDING_INPUT_SEAL.json')
    assert report['input_sha256'] == sha(REC_INPUT) and not report['target_labels_used_for_selection']
    assert not report['FINAL_access']
    candidates = read(REC_INPUT)
    assert len(candidates) == 1536
    index = indices()
    inspect = parser()
    alias = anonymizer()
    pools = defaultdict(lambda: defaultdict(list))
    filters = defaultdict(Counter)
    for i, raw in enumerate(candidates):
        target = raw['platform']
        row = dict(raw, seed=0, caption=raw['caption'].strip())
        f = frame(row)
        recording = f.parent.parent
        ph, ih = row['point_sha256'], row['image_sha256']
        if recording in older[3] or f in older[2] or ph in older[0] or ih in older[1]:
            filters[target]['r165_overlap'] += 1
            continue
        if any(v in seen for v, seen in zip((f, ph, ih), used, strict=True)):
            filters[target]['r170_same_observation'] += 1
            continue
        if any(v in seen for v, seen in zip((f, ph, ih), reviewed, strict=True)):
            filters[target]['r168_source_review_overlap'] += 1
            continue
        report_key = 'available_recording_previously_in_r170' if recording in prior_recordings else 'available_other_recording'
        filters[target][report_key] += 1
        if not HORIZONTAL.search(row['caption']):
            filters[target]['no_horizontal_word'] += 1
            continue
        parsed = inspect(row['caption'])
        edges = direct_edges(edges_of(parsed['graph']))
        if not edges:
            filters[target]['no_direct_edge'] += 1
            continue
        role, sign = min(edges, key=lambda x: hash_key(seed, 'edge', row['id'], *x))
        lo, hi = parsed['graph']['entities'][role]['span']
        stratum = 'entity_noun' if OBJECT_NOUN.search(row['caption'][lo:hi]) else 'other_span'
        path, object_index = index[row['id']]
        assert path == f
        row['object_index'] = object_index
        pools[target, stratum][recording].append(dict(row=row, target=target, baseline_index=i, parsed=parsed,
                    role=role, sign=sign, rewrite=rewrite(row['caption'], parsed, edges, role, sign), stratum=stratum,
                    rank=hash_key(seed, 'case', row['id'], ph), recording=alias('rec', recording.as_posix()),
                    recording_in_r170=recording in prior_recordings))
        filters[target]['eligible_' + stratum] += 1

    for pool in pools.values():
        for site, items in pool.items():
            pool[site] = deque(sorted(items, key=lambda entry: entry['rank']))
    selected = []
    counts = {}
    n = rule['frames_per_target']
    lexical = rule['lexical_slots_per_target']
    for target in PLATFORMS:
        count = Counter()
        nouns, other = (pools[target, x] for x in ('entity_noun', 'other_span'))
        group = pick(nouns, lexical, used, prior_recordings, count)
        group.extend(pick(other, n - lexical, used, prior_recordings, count))
        group.extend(pick(nouns, n - len(group), used, prior_recordings, count))
        group.extend(pick(other, n - len(group), used, prior_recordings, count))
        assert len(group) == n
        for x in group:
            x.pop('rank')
            x['case'] = len(selected) + 1
            selected.append(x)
        counts[target] = dict(frames=len(group), recordings=len(count),
                              recordings_also_in_r170=sum(k in prior_recordings for k in count),
                              text_strata=dict(Counter(x['stratum'] for x in group)),
                              rewrites=dict(Counter(x['rewrite']['status'] for x in group)),
                              screening=dict(filters[target]))
    assert all(len(x) == len(prior) + len(selected) for x in used)
    WORK.mkdir(parents=True, exist_ok=True, mode=0o700)
    assert not INPUT.exists()
    atomic_json(INPUT, selected)
    evidence = dict(frames=len(selected), direction_records=2 * len(selected), registry=sha(INPUT), design_lock=pin,
                    platforms=counts, identical_observations_shared_with_r170=0, historical_fixed_baseline_exposure=True,
                    no_target_predictions_or_annotations_used_for_selection=True, FINAL_access=False)
    atomic_json(HERE / 'INPUT_SEAL.json', evidence)
    print(evidence, flush=True)


if __name__ == '__main__':
    prepare()

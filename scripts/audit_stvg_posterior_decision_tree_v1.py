"""Independent CPU-only numerical audit of sealed T0/S0 decisions; never GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib, json, time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'artifacts/stvg_posterior_decision_tree_v1'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def direct(ids, q, block=128):
    """Direct physical tIoU for every legal action, independent of four-region sum."""
    ids = np.asarray(ids)
    assert np.issubdtype(ids.dtype, np.integer) and np.all(np.diff(ids) >= 1)
    a, b = np.triu_indices(len(ids), 1)
    start = ids[a].astype(np.float64)
    end = (ids[b] + 1).astype(np.float64)
    probability = q[a, b]
    answer = np.empty(len(a))
    for lo in range(0, len(a), block):
        hi = min(lo + block, len(a))
        overlap = np.maximum(0., np.minimum(end[lo:hi, None], end) -
                             np.maximum(start[lo:hi, None], start))
        union = (end[lo:hi] - start[lo:hi])[:, None] + end - start - overlap
        answer[lo:hi] = np.sum((overlap / union) * probability, axis=1)
    return a, b, answer

def greedy_check(ids, q, trace):
    """Use physical interval membership and explicit capped-coverage differences."""
    ids = np.asarray(ids)
    a, b = np.triu_indices(len(ids), 1)
    weight = q[a, b]
    membership = ((ids[:, None] >= ids[a]) & (ids[:, None] < ids[b] + 1))
    hits = np.zeros(len(a), dtype=int)
    selected = []
    worst = 0.
    for step in trace:
        old = np.sum(weight * np.minimum(hits, 2) / 2.)
        values = np.sum(weight[None] * np.minimum(hits[None] + membership, 2) / 2., axis=1)
        gains = np.sum(membership * (.5 * weight * (hits < 2))[None], axis=1)
        gains[selected] = -np.inf
        allowed = np.isfinite(gains)
        worst = max(worst, float(np.max(np.abs(values[allowed] - old - gains[allowed]))))
        pool = np.flatnonzero(gains == np.max(gains)).tolist()
        if selected:
            chosen = min(pool, key=lambda f: (-min(abs(int(ids[f]) - int(ids[g]))
                                                   for g in selected), int(ids[f])))
        else:
            midpoint = (int(ids[0]) + int(ids[-1]) + 1) / 2
            chosen = min(pool, key=lambda f: (abs(int(ids[f]) - midpoint), int(ids[f])))
        assert chosen == step['position'] and len(pool) == step['tied_candidates']
        assert abs(float(gains[chosen]) - step['marginal']) < 1e-12
        selected.append(chosen)
        hits += membership[chosen]
        assert abs(float(np.sum(weight * np.minimum(hits, 2) / 2.)) - step['model_coverage']) < 1e-12
    assert len(selected) == len(set(selected)) == 4
    assert worst < 1e-12
    return sorted(selected), worst

def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    plan = read(D / 'PLAN_LOCK.json')
    barrier = read(D / 'GLOBAL_READOUT_BARRIER.json')
    assert barrier['status'] == 'complete' and barrier['count'] == 64
    assert barrier['GT_read'] is False
    lock = read(D / 'RUNTIME_LOCK.json')
    for path, sha in {**lock['pins'], **lock['registries']}.items():
        assert digest(ROOT / path) == sha
    assert digest(D / 'PLAN_LOCK.json') == lock['plan_sha256']
    items = []
    checks = 0
    maximum = dict(utility=0., coverage=0., reconstructed_q=0.)
    sources = {row['alias']: row for row in plan['files']}
    for sealed in barrier['files']:
        path = ROOT / sealed['path']
        assert digest(path) == sealed['sha256']
        r = torch.load(path, map_location='cpu', weights_only=False)
        assert r['GT_read'] is False
        ids, q = r['frame_ids'], np.asarray(r['q'])
        assert q.shape == (len(ids), len(ids)) and np.all(np.isfinite(q))
        assert np.min(q) >= 0 and abs(float(q.sum()) - 1) < 1e-12
        assert np.count_nonzero(np.tril(q)) == 0
        source = sources[r['alias']]
        assert digest(ROOT / source['path']) == source['sha256']
        original = torch.load(ROOT / source['path'], map_location='cpu', weights_only=False)
        capture = original['capture']
        distributions = []
        for record, logits in zip(capture['records'], capture['zero']['logits']):
            z = logits.detach().cpu().numpy()[0].astype(np.float64)
            a0, b0 = np.triu_indices(len(z), 1)
            energy = z[a0, 0] + z[b0, 1]
            p = np.exp(energy - energy.max()); p /= p.sum()
            fids = np.asarray(record['frame_ids'])
            distributions.append((fids[a0], fids[b0], p))
        assert len(distributions) == 2
        index = {fid: i for i, fid in enumerate(ids)}
        rebuilt = np.zeros_like(q)
        s1, e1, p1 = distributions[1]
        for s0, e0, p0 in zip(*distributions[0]):
            ss = np.array([index[int(x)] for x in np.minimum(s0, s1)])
            ee = np.array([index[int(x)] for x in np.maximum(e0, e1)])
            np.add.at(rebuilt, (ss, ee), p0 * p1)
        qerr = float(np.max(np.abs(rebuilt - q)))
        assert qerr < 1e-12
        maximum['reconstructed_q'] = max(maximum['reconstructed_q'], qerr)
        a, b, utilities = direct(ids, q)
        error = float(np.max(np.abs(utilities - r['utility_grid'][a, b])))
        assert error < 1e-11
        maximum['utility'] = max(maximum['utility'], error)
        selected = int(np.argmax(utilities))
        assert r['intervals']['final_MBR'] == [ids[a[selected]], ids[b[selected]] + 1]
        imap = int(np.argmax(q))
        si, ei = divmod(imap, len(ids))
        assert r['intervals']['marginal_MAP'] == [ids[si], ids[ei] + 1]
        picked, coverr = greedy_check(ids, q, r['greedy_trace'])
        assert picked == r['layouts']['P']
        maximum['coverage'] = max(maximum['coverage'], coverr)
        for name, layout in r['layouts'].items():
            assert len(layout) == len(set(layout)) == 4
            hits = sum((np.asarray(ids)[a] <= ids[f]) & (ids[f] < np.asarray(ids)[b] + 1)
                       for f in layout)
            value = float(np.sum(q[a, b] * np.minimum(hits, 2) / 2.))
            assert abs(value - r['model_coverage'][name]) < 1e-12
        checks += len(utilities)
        items.append(dict(alias=r['alias'], dataset=r['dataset'], candidate_count=len(a),
                          zero_mass_candidates=int(np.count_nonzero(q[a, b] == 0)),
                          utility_max_error=error, reconstructed_q_max_error=qerr,
                          greedy_max_error=coverr, readout_sha256=sealed['sha256']))
        del original, capture, r
    result = dict(status='pass', count=len(items), GT_read=False, new_models=0, new_DINO=0,
                  source_logits_only=True, direct_all_candidate_checks=checks,
                  integer_physical_grid=True, zero_mass_actions_included=True,
                  exact_greedy_ties_checked=True, maximum_errors=maximum, files=items,
                  barrier_sha256=digest(D / 'GLOBAL_READOUT_BARRIER.json'),
                  plan_sha256=digest(D / 'PLAN_LOCK.json'), script_sha256=digest(__file__),
                  CPU_wall_seconds=time.perf_counter() - started)
    destination = D / 'ROOT_NUMERICAL_AUDIT_NO_GT.json'
    assert not destination.exists()
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))

if __name__ == '__main__':
    main()

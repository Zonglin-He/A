"""Measure paired CPU wall time for all six frozen backend configurations."""
from hashlib import sha256
from time import perf_counter_ns

import cv2
import numpy as np

from replay_minimal import BLIND_ARMS, HERE, RUN, check, digest, publish, read, sealed_inputs, variants, verify_output


def report(rows):
    rng = np.random.default_rng(20261128)
    measured = {arm: np.array([r['ms'][arm] for r in rows]) for arm in BLIND_ARMS}
    full = np.median(measured['Full'], axis=1)
    picks = rng.integers(len(rows), size=(2000, len(rows)))
    out = {}
    for arm, a in measured.items():
        paired = np.median(a, axis=1) - full
        out[arm] = dict(median_ms=float(np.median(a)), p90_ms=float(np.quantile(a, .9)),
                        paired_median_delta_ms=float(np.median(paired)),
                        paired_delta_ci_ms=np.quantile(np.median(paired[picks], axis=1), [.025, .975]).tolist())
    return dict(n=len(rows), replicates=3, metrics=out)


def main():
    cv2.setNumThreads(1)
    check(RUN/'blind.json', 'payload', RUN/'seal.json')
    prior = {r['uid']:r for r in read(RUN/'blind.json')}
    source, _, history, _, full = sealed_inputs()
    methods = variants(full)
    items = []
    for domain in ('waymo', 'drone', 'quad', 'kitti', 'mono'):
        for active in (True, False):
            bank = [u for u,r in source.items() if r['target'] == domain and history[u]['active'] == active]
            bank.sort(key=lambda u: sha256(('minimal-timing-20261128|' + u).encode()).hexdigest())
            assert len(bank) >= 8
            items.extend((domain, active, u) for u in bank[:8])

    rows = []
    for i, (domain, active, uid) in enumerate(items):
        row = source[uid]
        args = row['frame'], row['raw']
        output = prior[uid]['predictions']
        for arm, fn in methods.items():
            verify_output(output[arm], fn(*args))
        ms = {arm: [] for arm in methods}
        for replicate in range(3):
            offset = (i+replicate) % len(methods)
            order = BLIND_ARMS[offset:] + BLIND_ARMS[:offset]
            for arm in order[::(-1 if (i+replicate) % 2 else 1)]:
                started = perf_counter_ns()
                box = methods[arm](*args)
                ms[arm].append((perf_counter_ns() - started) / 1.e6)
                verify_output(output[arm], box)
        rows.append(dict(domain=domain, active=active, ms=ms))
        if len(rows) % 20 == 0:
            print('Timed', len(rows), '/', len(items), flush=True)
    groups = {'all':rows, 'full_C_ray':[r for r in rows if r['active']],
              'full_fallback':[r for r in rows if not r['active']]}
    for p in ('waymo', 'drone', 'quad', 'kitti', 'mono'):
        groups[p] = [r for r in rows if r['domain'] == p]
    timings = {group:report(rows) for group,rows in groups.items()}
    publish(RUN/'timing.json', dict(code=digest(HERE/'time_backend.py'), n=len(items), timing=timings))
    print(timings['all'], flush=True)


if __name__ == '__main__':
    main()

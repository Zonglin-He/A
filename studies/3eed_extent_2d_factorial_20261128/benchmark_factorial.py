"""Time the four public geometry paths on the same 48 sealed queries."""
from functools import partial
from hashlib import sha256
from time import perf_counter_ns

import cv2
import numpy as np

from replay_factorial import ARMS, RUN, check, code_stamp, digest, publish, read, source_inputs


def summaries(timings):
    def describe(values):
        return dict(n=len(values),median_ms=float(np.median(values)),mean_ms=float(np.mean(values)),
                    p90_ms=float(np.quantile(values,.90)))

    out = {}
    for group in ('all','C_ray','fallback','3eed','kitti','mono'):
        rows = [r for r in timings if group=='all' or group==r['group'] or
                group==('C_ray' if r['active'] else 'fallback')]
        values = {arm:np.array([r['ms'][arm] for r in rows]) for arm in ARMS}
        out[group] = {arm:describe(v) for arm,v in values.items()}
        out[group]['axis_2d_minus_height_2d'] = describe(values['axis_2d']-values['height_2d'])
    return out


def main():
    cv2.setNumThreads(1)
    tasks,pins = source_inputs()
    import axiswise_2d as trial

    path = RUN/'predictions-private.json'
    seal = check(path,'payload',RUN/'prediction-seal.json')
    assert seal['code']==code_stamp() and seal['sources']==pins and not seal['labels_read']
    recorded = read(path)
    source = {row[0]:row[1] for row in tasks}
    runner = dict(height_ray=partial(trial.live.predict,height_only=True),
                  axis_ray=partial(trial.live.predict,height_only=False),height_2d=trial.official.predict,axis_2d=trial.predict)
    samples = []
    for group, domains in (('3eed',{'waymo','drone','quad'}),('kitti',{'kitti'}),('mono',{'mono'})):
        for active,count in ((True,12),(False,4)):
            matches = [r for r in recorded if r['target'] in domains and r['active']==active]
            ordered = sorted(matches,key=lambda r:sha256(('extent-timing-20261128|'+r['uid']).encode()).hexdigest())
            assert len(ordered)>=count
            samples.extend((group,r) for r in ordered[:count])
    timing = []
    for i,(group,row) in enumerate(samples):
        frame = source[row['uid']]
        args = frame['frame'],frame['raw']
        for arm,run in runner.items():
            assert run(*args)==row['outputs'][arm]
        for turn in range(3):
            offset = (turn+i)%len(ARMS)
            durations = {}
            for arm in ARMS[offset:]+ARMS[:offset]:
                start = perf_counter_ns()
                result = runner[arm](*args)
                durations[arm] = (perf_counter_ns()-start)/1_000_000.
                assert result==row['outputs'][arm]
            timing.append(dict(group=group,active=row['active'],replicate=turn,ms=durations))
    artifact = RUN/'timing-all-arms.json'
    publish(artifact,timing)
    output = dict(method='single CPU, same 48 hash-selected rows, warm-up plus 3 cyclic-order repetitions',
                  replies_cached=True,model_calls=0,measurements=len(timing),data_sha=digest(artifact),
                  prediction_sha=seal['payload'],breakdown=summaries(timing))
    publish(RUN/'timing-all-arms-summary.json',output)
    print(output['breakdown']['all'],flush=True)


if __name__=='__main__':
    main()

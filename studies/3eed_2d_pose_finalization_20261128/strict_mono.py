"""Seal and score the external recording subset excluding schema examples."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from sys import path

import numpy as np

from prepare_mono import RUN, digest, publish, raw_recordings, read

HERE = Path(__file__).resolve().parent
path.insert(0,str(HERE.parent/'3eed_hypothesis_verification_20261128'))
from inference_stats import cluster_weights, intervals

ARMS = ('native','old','promoted')
SCHEMA_EXAMPLES = (1,2,4)
ROSTER = RUN/'mono-strict-roster-private.json'
SEAL = RUN/'mono-strict-roster-seal.json'


def seal():
    assert not (RUN/'mono-prediction-seal.json').exists() and not (RUN/'mono-scores-private.json').exists()
    original = read(RUN/'mono-input-seal.json')
    assert original['private_index']==digest(RUN/'mono-paired-index-private.json')
    mapping = raw_recordings()
    excluded = {mapping[k][0] for k in SCHEMA_EXAMPLES}
    rows = read(RUN/'mono-paired-index-private.json')
    selected = [r for r in rows if r['recording'] not in excluded]
    assert len(selected)==118 and len({r['recording'] for r in selected})==18
    publish(ROSTER,[r['queries'][0]['uid'] for r in selected])
    publish(SEAL,dict(file=digest(ROSTER), protocol=digest(HERE/'STRICT_TEST_ADDENDUM_ZH.md'),
         source_input_seal=digest(RUN/'mono-input-seal.json'), schema_examples=list(SCHEMA_EXAMPLES),
         input_recording_exclusions=len(excluded),exclusion_units='entire original recording',
         total=original['eligible_frames'], excluded=len(rows)-len(selected),
         n=len(selected), recordings=18, target_scores_unopened=True,
         audited_project_historical_recording_overlap=0))
    print(read(SEAL),flush=True)


def score():
    receipt=read(SEAL)
    assert digest(ROSTER)==receipt['file'] and receipt['source_input_seal']==digest(RUN/'mono-input-seal.json')
    assert receipt['protocol']==digest(HERE/'STRICT_TEST_ADDENDUM_ZH.md')
    whole=read(RUN/'mono-results.json')
    assert whole['only_offline_gt'] and whole['score_payload']==digest(RUN/'mono-scores-private.json')
    expected=set(read(ROSTER))
    xs=[r for r in read(RUN/'mono-scores-private.json') if r['uid'] in expected]
    assert len(xs)==len(expected)==receipt['n'] and len({r['recording'] for r in xs})==receipt['recordings']

    def panel(rs):
        n=len(rs)
        if not n:
            return dict(n=0)
        weights=cluster_weights(rs)
        iou={k:np.array([r['iou'][k] for r in rs]) for k in ARMS}
        hit={k:a>.5 for k,a in iou.items()}
        d={}
        for a,b in (('old','native'),('promoted','old'),('promoted','native')):
            diff=100*(iou[a]-iou[b])
            rate=100*(hit[a].astype(float)-hit[b])
            d[a+'_vs_'+b]=dict(delta_miou=float(diff.mean()),miou_95ci=intervals(diff,weights),
                delta_acc50=float(rate.mean()),acc50_95ci=intervals(rate,weights),
                rescue=int((rate>0).sum()),harm=int((rate<0).sum()))
        return dict(n=n,recordings=len({r['recording'] for r in rs}),switched=sum(r['switched'] for r in rs),
            routes=dict(Counter(r['route'] for r in rs)), valid_2d=sum(r['valid_2d'] for r in rs),
            valid_3d=sum(r['valid_3d'] for r in rs),
            metrics={k:dict(miou=100*float(iou[k].mean()),acc50=100*float(hit[k].mean())) for k in ARMS},
            paired=d)

    result=dict(roster_seal=receipt, all=panel(xs), C_ray=panel([r for r in xs if r['route']=='C_ray']),
                source_scored_payload=whole['score_payload'], only_offline_gt=True)
    publish(RUN/'mono-strict-results.json',result)
    print(result['all'],flush=True)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('stage', choices=('seal','score'))
    {'seal':seal,'score':score}[cli.parse_args().stage]()

"""Reconcile the registered live pair with its sealed earlier 665-query ablation."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '3eed_2d_relocalization_20261128'))
import fresh_relocalize  # Resolve the historical study modules before geometry setup.
from shared import ROOT, canonical, digest, guard_labels, publish, read, setup
from metric_image_common import inputs
from replay import context_pins
from evaluate_structure import stat_panel

EXPERIMENT = ROOT / 'runs/2d-pose-finalization-20261128'
PREVIOUS = ROOT / 'runs/2d-relocalization-20261128'


def run(args):
    from live_2d import paired

    uid, row, target, chosen = args
    old, new = paired(row['frame'],row['raw'],diagnostics=True)
    pool = target['pool']['cray']
    if not pool['boxes']:
        assert old['box'] is None and new['box'] is None
        return uid, False, None, None
    np.testing.assert_allclose(canonical(old['box']), pool['boxes'][0], rtol=0, atol=2e-9)
    candidate = chosen['selected']['cray']['cached']
    np.testing.assert_allclose(canonical(new['box']),pool['boxes'][candidate],rtol=0,atol=2e-9)
    if 'candidate_details' in new:
        info = new['candidate_details']
        if len(pool['boxes']) > 1:
            np.testing.assert_allclose(canonical(info['other_box']),pool['boxes'][1],rtol=0,atol=2e-9)
        return uid, info['switched'], info['extra_geometry_seconds'], [old['heading'], new['heading']]
    return uid, False, None, None


def main():
    HISTORIC = read(PREVIOUS/'score-seal.json')
    assert HISTORIC['scored'] == digest(PREVIOUS/'scored.json')
    for name in ('candidate','choices'):
        assert read(PREVIOUS/f'{name}-seal.json')['file'] == digest(PREVIOUS/f'{name}-blind.json')
    from geometry_predict import guard_study_annotations

    guard_labels()
    guard_study_annotations()
    setup()
    historic_dir=ROOT/'runs/cascaded-oracle-20261128'
    original=read(historic_dir/'o0-seal.json')
    bound=context_pins()
    legacy=HERE.parent/'3eed_qwen_extent_height_only_20261122/LEGACY_REGISTRATION.json'
    bound['registration']=digest(legacy)
    assert original['pins']==bound and not original['target_labels_read']
    source=original['payload']
    assert read(PREVIOUS/'candidate-seal.json')['lock']==source==digest(historic_dir/'o0.json')
    data=inputs()
    prior = {r['uid']:r for r in read(PREVIOUS/'candidate-blind.json') if r['group']=='main'}
    choices = {r['uid']:r for r in read(PREVIOUS/'choices-blind.json') if r['group']=='main'}
    tasks = [(u,r,prior[u],choices[u]) for u,r in data.items()]
    with ProcessPoolExecutor(8) as workers:
        blind = list(workers.map(run,tasks,chunksize=5))
    EXPERIMENT.mkdir(exist_ok=True, parents=True)
    stage = EXPERIMENT/'paired-replay-blind.json'
    publish(stage,blind)
    publish(EXPERIMENT/'paired-replay-seal.json',dict(input=source,file=digest(stage), n=len(blind),
        code=digest(HERE.parent/'3eed_qwen_extent_height_only_20261122/live_2d.py'), target_labels_read=False))
    print('Live matched old and cached-rank predictions',len(blind),flush=True)


def export():
    seal=read(EXPERIMENT/'paired-replay-seal.json')
    assert seal['file']==digest(EXPERIMENT/'paired-replay-blind.json') and seal['n']==702
    assert read(PREVIOUS/'score-seal.json')['scored']==digest(PREVIOUS/'scored.json')
    truth={r['uid']:r for r in read(PREVIOUS/'scored.json') if r['group']=='main'}
    predictions={r[0]:r for r in read(EXPERIMENT/'paired-replay-blind.json')}
    comparisons={}
    for key in ('all','3eed','waymo','drone','quad','kitti'):
        rows=[dict(row,iou={'old':row['pool']['cray']['iou']['formal'],'promoted':row['pool']['cray']['iou']['cached']})
              for row in truth.values() if key=='all' or key==row['target'] or (key=='3eed' and row['target']!='kitti')]
        x=stat_panel(rows, ('old','promoted'), 'old')
        x['changed']=sum(predictions[r['uid']][1] for r in rows)
        dt=[predictions[r['uid']][2] for r in rows if predictions[r['uid']][2] is not None]
        x['added_geometry_seconds_median']=float(np.median(dt))
        comparisons[key]=x
    dest=EXPERIMENT/'development-paired-results.json'
    publish(dest,comparisons)
    print(comparisons['all'],flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='export':
        export()
    else:
        main()

"""Replay the sealed selected carriers with their input box geometry."""

import argparse

import numpy as np

from evaluate import DIRECTIONS, PRIVATE, fingerprint, score_boxes, sensor_truth
from scope import HERE, evaluation_rows, file_sha, protocol, read, save


OUT = HERE / 'geometry_replay_PRIVATE'
ARMS = ('full_pool_unary', 'full_pool_joint', 'source_selected')


def seal():
    incumbent = read(PRIVATE/'SEALED.json')
    assert incumbent['n'] == 3072 and not incumbent['FINAL_access']
    assert read(PRIVATE/'LOCK.json')['implementation'] == fingerprint()
    comparisons = protocol()['comparisons']
    columns = [comparisons.index(name) for name in ARMS]
    OUT.mkdir(mode=0o700)
    files = {}
    for s,t in DIRECTIONS:
        name = f'{s}_to_{t}.npz'
        path = PRIVATE/name
        assert file_sha(path) == incumbent['files'][name]['sha256']
        with np.load(path, allow_pickle=False) as old:
            ids = old['indices']
            pools = old['pool']
            rows = np.arange(len(ids))
            raw = pools[rows[:,None],ids[:,columns]]
            baseline = old['picked'][:,0]
            np.testing.assert_allclose(pools[rows,ids[:,0]],baseline,atol=1e-12,rtol=0)
            raw = np.where((ids[:,columns]==ids[:,:1])[...,None],baseline[:,None],raw)
            f = OUT/name
            np.savez(f, ordinals=old['ordinals'], raw=raw, earlier=old['picked'][:,columns], v1=baseline)
            files[name] = dict(n=len(rows), sha256=file_sha(f))
    save(OUT/'SEALED.json',dict(files=files,arm_order=list(ARMS),n=incumbent['n'],
         original_prediction_seal=file_sha(PRIVATE/'SEALED.json'), identity_unchanged=True,
         new_labels_used_for_replay=False,FINAL_access=False))


def summary(values):
    return dict(Acc25=float((values>.25).mean()*100), Acc50=float((values>.50).mean()*100),
                mIoU=float(values.mean()*100))


def score():
    rows = evaluation_rows()
    pinned = read(OUT/'SEALED.json')
    assert pinned['original_prediction_seal'] == file_sha(PRIVATE/'SEALED.json')
    assert not pinned['new_labels_used_for_replay'] and not pinned['FINAL_access']
    values = {}
    max_error = 0.
    for s,t in DIRECTIONS:
        label = f'{s}_to_{t}'
        path = OUT/f'{label}.npz'
        assert file_sha(path) == pinned['files'][path.name]['sha256']
        with np.load(path, allow_pickle=False) as fields:
            result = []
            for row, ordinal in enumerate(fields['ordinals']):
                key = rows[int(ordinal)]
                target = sensor_truth(t,key['metapath'],key['caption'],key['suffix'])
                predictions = np.concatenate((fields['v1'][row:row+1],
                                              fields['earlier'][row],fields['raw'][row]))
                scored,error = score_boxes(predictions,target)
                max_error = max(max_error,error)
                result.append(scored)
            values[label] = np.stack(result)
    values['six_equal'] = np.concatenate(tuple(values.values()))
    comparisons={}
    for label, results in values.items():
        baseline, old, raw = results[:,0], results[:,1:1+len(ARMS)], results[:,1+len(ARMS):]
        comparisons[label] = {'v1':summary(baseline)}
        for j,name in enumerate(ARMS):
            shift = raw[:,j]-old[:,j]
            comparisons[label][name] = dict(n=len(results),raw=summary(raw[:,j]),earlier=summary(old[:,j]),
                better=int((shift>1e-12).sum()),worse=int((shift<-1e-12).sum()),
                same=int((np.abs(shift)<=1e-12).sum()),
                recovered_Acc25=int(((raw[:,j]>.25)&(old[:,j]<=.25)).sum()),
                lost_Acc25=int(((raw[:,j]<=.25)&(old[:,j]>.25)).sum()),
                recovered_Acc50=int(((raw[:,j]>.50)&(old[:,j]<=.50)).sum()),
                lost_Acc50=int(((raw[:,j]<=.50)&(old[:,j]>.50)).sum()))
    save(HERE/'results/geometry_readout_check.json',dict(
         paired_metrics=comparisons, exact_oriented_iou_max_abs_error=max_error,
         prediction_seal=file_sha(OUT/'SEALED.json'), exploratory_on_previously_scored_panel=True,
         selection_rerun=False, FINAL_access=False))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase',choices=('seal','score'))
    if cli.parse_args().phase=='seal':
        seal()
    else:
        score()

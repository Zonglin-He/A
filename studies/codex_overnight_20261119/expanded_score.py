"""Score an ordinary frame-separated expansion without target-label selection."""
from collections import Counter
from pathlib import Path

import numpy as np

from expanded import ARMS, DATA, HERE, PLATFORMS, PROTOCOL, read, sha, atomic_json
from edge_score import csv_atomic, directions, iou_metrics, macro, METRICS


PROBES = ('r158_pca','r171_native_joint','same_evidence_independent','same_evidence_geometric',
          'role_exchange_mismatched')


def metric_rows(rows):
    stats, comparisons = [],[]
    four = [('all',rows),('relation',[r for r in rows if r['rewrite']]),
            ('cross_supported',[r for r in rows if r['eligible']]),
            ('new_recordings',[r for r in rows if not r['prior_mechanism_recording']]),
            ('new_recording_relations',[r for r in rows if r['rewrite'] and not r['prior_mechanism_recording']])]
    for scope, content in four:
        if not content:
            continue
        groups = {'direction_macro':content}
        groups.update({f"{g[0]['source']}_to_{g[0]['target']}":g for g in directions(content)})
        for label, v in groups.items():
            a = {m:sorted([r for r in v if r['arm']==m],key=lambda r:(r['frame'],r['source'])) for m in ARMS}
            n = len(a[ARMS[0]])
            metadata = dict(scope=scope,route=label,n=n,directions=len(directions(v)),
                            frames=len({r['frame'] for r in v}),recordings=len({r['recording'] for r in v}))
            for name, values in a.items():
                task = macro(values,lambda r:iou_metrics(r['official_iou']))
                stats.append(dict(**metadata,arm=name,**dict(zip(METRICS,task,strict=True)),
                                  reference_proxies=sum(r['reference_proxy_known'] for r in values),
                                  reference_proxy_correct=sum(r['reference_proxy_correct']==1 for r in values),
                                  paired_proxy_correct=sum(r['dual_proxy_correct']==1 for r in values)
                                  if name!=ARMS[0] else '',
                                  reference_emitted=sum(r['reference_emitted'] for r in values)))
            main = a['role_exchange_joint']
            for comparison in PROBES:
                pairs = list(zip(main,a[comparison],strict=True))
                assert all((x['frame'],x['source']) == (y['frame'],y['source']) for x,y in pairs)
                shifted = [dict(x,delta=iou_metrics(x['official_iou'])-iou_metrics(y['official_iou'])) for x,y in pairs]
                z = np.stack([x['delta'] for x in shifted])
                task = macro(shifted,lambda r:r['delta'])
                comparisons.append(dict(**metadata,control=comparison,**dict(zip(METRICS,task,strict=True)),
                                        **{f'{g}_{k}':int((s*z[:,i]>1.e-10).sum())
                                           for i,g in enumerate(METRICS) for k,s in [('gain',1),('loss',-1)]},
                                        dual_gains=sum(x['dual_proxy_correct']==1 and y['dual_proxy_correct']!=1 for x,y in pairs),
                                        dual_losses=sum(x['dual_proxy_correct']!=1 and y['dual_proxy_correct']==1 for x,y in pairs)))
    return stats, comparisons


def scores():
    from calibrate168 import source_instances
    from labels166 import occupied_owners
    from mechanisms162 import annotation_inventory, seed_owners
    from predict171 import read_npz
    from score165 import many_iou, sensor_values

    root = DATA/'predictions'
    receipt = read(root/'SEAL.json')
    assert not receipt['target_label_read'] and not receipt['protected_data_access']
    assert receipt['lock']==sha(root/'LOCK.json') and read(root/'LOCK.json')['design']==sha(PROTOCOL)
    manifest = DATA/'ROSTER.json'
    assert read(DATA/'ROSTER_SEAL.json')['inputs']==sha(manifest)==read(root/'LOCK.json')['roster']
    registry = read(manifest)
    assert len(registry)*2 == receipt['n'] == len(receipt['files'])
    assert all(sha(root/k)==v for k,v in receipt['files'].items())
    out = HERE/'results/expanded'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'SCORE_SEAL.json').exists()
    atomic_json(out/'SCORE_LOCK.json',dict(code=sha(Path(__file__)),inputs=sha(root/'SEAL.json')))
    details, errors = [],[]
    for entry in registry:
        row = entry['row']
        is_query = entry['rewrite']['status']=='supported'
        view = int(row['suffix'] or 0)
        target = entry['target']
        names,gt = annotation_inventory(row['metapath'],target,view,row['object_index'])
        ref = int(source_instances(row,entry['parsed'],[entry['role']]).get(entry['role'],-1)) if is_query else -1
        assert ref!=0 and ref<len(names)
        for s in PLATFORMS:
            if s==target:
                continue
            tag = f"{entry['case']:04d}_{s}.npz"
            pred = read_npz(root/tag)
            assert bool(pred['rewrite'])==is_query and tuple(pred['arms'])==ARMS and pred['lock']==receipt['lock']
            b = pred['boxes']
            y = many_iou(b[:,0],gt[0])
            sensor = sensor_values(b[:,0],gt[0],target,view)
            np.testing.assert_allclose(y,sensor,rtol=0,atol=1.e-7)
            assert np.array_equal(y>.25,sensor>.25) and np.array_equal(y>.5,sensor>.5)
            errors.append(float(np.max(np.abs(y-sensor))))
            if is_query:
                ids = occupied_owners(b,pred['seeds'],seed_owners(pred['seeds'],gt),len(names))
            else:
                ids = np.full((len(ARMS),2),-1)
                np.testing.assert_array_equal(b,np.broadcast_to(b[0],b.shape))
            emitted = np.isfinite(b[:,1]).all(1)
            meta = dict(frame=entry['case'],source=s,target=target,recording=entry['recording'],
                        prior_mechanism_recording=int(entry['mechanism_recording_reuse']),
                        rewrite=int(is_query),eligible=int(pred['eligible']),reference_proxy_known=int(ref>0),
                        active_elapsed_s=float(pred['seconds']))
            for j,name in enumerate(ARMS):
                details.append(dict(**meta,arm=name,official_iou=float(y[j]),subject_annotation_match=int(ids[j,0]==0)
                                    if is_query else '',reference_emitted=int(emitted[j]),
                                    reference_proxy_correct=int(ids[j,1]==ref) if ref>0 and j else '',
                                    dual_proxy_correct=int(ids[j,0]==0 and ids[j,1]==ref) if ref>0 and j else ''))
    assert len(details)==receipt['n']*len(ARMS)
    result, pairs = metric_rows(details)
    for k,v in [('rows',details),('metrics',result),('contrasts',pairs)]:
        csv_atomic(out/(k+'.csv'),v)
    atomic_json(out/'SCORE_SEAL.json',dict(rows=receipt['n'],max_sensor_iou_error=max(errors),physical_frames=len(registry),
        role_reference_is_annotation_proxy_not_independent=True,
        scope=read(DATA/'ROSTER_SEAL.json')['counts'],files={p.name:sha(p) for p in out.glob('*.csv')},
        protected_data_access=False))
    print('Scored all ordinary expansion directions',receipt['n'],flush=True)


if __name__=='__main__':
    scores()

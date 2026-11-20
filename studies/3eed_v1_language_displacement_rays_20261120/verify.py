"""Independent projection, threshold and uncertainty audit for the completed source panel."""
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from capture import ARMS, CFG, HERE, PRIVATE, ROOT, R181, atomic, checked, digest, read, sealed_arrays, source_inputs
from rays import key, selection
from roles import inventory

OUT = HERE / 'results'


def distances(points, origins, vectors, directed=True):
    """Weighted projection formulation with a nonnegative, freely fitted distance."""
    w = np.linalg.norm(vectors,axis=1)**2
    if w.sum() == 0.:
        return np.zeros(len(points))
    vec = points[:,None,:] - origins
    if directed:
        s = np.divide(np.einsum('nki,ki->nk',vec,vectors), w[None,:],
                      out=np.zeros((len(points),len(w))),where=w[None,:]>0)
        vec = vec - np.maximum(0,s)[:,:,None]*vectors[None,:,:]
    return ((vec*vec).sum(2) @ w)/w.sum()


def run():
    dest = PRIVATE / 'source'
    seal = read(dest/'SEALED.json')
    lock = checked(dest/'LOCK.json',seal['lock'])
    assert lock['code']['studies/'+HERE.name+'/rays.py'] == digest(HERE/'rays.py')
    assert seal['n'] == 192 and set(seal['native_state_hashes'])==set(CFG['source_sizes'])
    pre = read(OUT/'source_gate.json')
    assert pre['source_seal']==digest(dest/'SEALED.json')
    metric = list(csv.DictReader((OUT/'source_metrics.csv').open()))
    pairs = list(csv.DictReader((OUT/'source_paired.csv').open()))
    with gzip.open(OUT/'source_all_rows.csv.gz','rt') as f:
        published = list(csv.DictReader(f))
    assert len(published) == 192
    maximum = 0.
    grouped, tails, mixed, certs, examples = defaultdict(list), Counter(), Counter(), Counter(), []
    eligible_differences, all_rows = [], []
    for platform,n in CFG['source_sizes'].items():
        ids = selection('source',platform,n)
        assert ids == lock['roster'][platform]
        truth = sealed_arrays(R181,'SEALED.json','source_scores',platform,['ious'])['ious']
        pool = source_inputs(platform)
        for i in ids:
            rel = f'{platform}/{i:05}.npz'
            assert digest(dest/rel) == seal['files'][rel]
            with np.load(dest/rel,allow_pickle=False) as z:
                slate = z['slate']
                f = z['full'].astype(float)
                a = z['reduced'][-1,:,:3].astype(float)
                v = f[-1,:,:3] - a
                assert len(np.unique(z['ids']))==len(z['ids'])==256 and len(np.unique(slate))==16
                np.testing.assert_array_equal(z['boxes'][:,:3],f[-1,slate,:3])
                perm = z['permutation']
                assert np.array_equal(np.sort(perm),np.arange(256)) and np.all(np.arange(256)!=perm)
                points=z['boxes'][:,:3].astype(float)
                values=np.column_stack((distances(points,a,v,False),distances(points,a,v),
                                        distances(points,a,-v),distances(points,a,v[perm])))
                maximum=max(maximum,float(np.abs(values-z['energy']).max()))
                np.testing.assert_allclose(values,z['energy'],atol=1e-10,rtol=2e-13)
                def pick(cost):
                    return min(range(16),key=lambda k:(cost[k],-z['joint'][k],k))
                choices=z['choices']
                assert pick(-z['seed'][slate])==choices[2]
                verified=np.array([pick(values[:,j]) for j in range(4)])
                np.testing.assert_array_equal(verified,choices[3:])
                ious=truth[i]
                all_iou=ious[choices]
                b,g = all_iou[1],all_iou[4]
                active=bool(v.any())
                grouped['all'].append(all_iou); grouped['active' if active else 'fallback'].append(all_iou)
                grouped[platform].append(all_iou)
                tails['frozen_over50_to_zero'] += bool(b>.50 and g==0)
                tails['frozen_over50_to_atmost25'] += bool(b>.50 and g<=.25)
                tails['frozen_over50_to_25through50'] += bool(b>.50 and .25<g<=.5)
                tails['zero_to_over50'] += bool(b==0 and g>.50)
                tails['pool_has50_but_ray_misses'] += bool(ious.max()>.50 and g<=.50)
                tails['frozen50_and_new50'] += bool(b>.50 and g>.50)
                if abs(g-b)>1e-12:
                    examples.append(dict(platform=platform,frozen_iou=float(b),ray_iou=float(g),
                                         change=float(g-b),mismatch_iou=float(all_iou[6]),
                                         seed_iou=float(all_iou[2]),pool_best_iou=float(ious.max())))
                all_rows.append((key('anonymous','source',platform,i),platform,all_iou))
                if not active:
                    assert not z['valid'] or (f[-1,:,:3]==a).all()
                    assert np.all(choices[3:]==choices[1])
                    continue
                y = inventory(pool[i])[0][0,:3].astype(float)
                er = np.sqrt(distances(y[None],a,v)[0])
                rev = np.sqrt(distances(y[None],a,-v)[0])
                shuffle = np.sqrt(distances(y[None],a,v[perm])[0])
                eligible_differences.append([rev-er,shuffle-er])
                # Sufficient recovery certificate from the weighted line projector.
                w=np.einsum('ki,ki->k',v,v)
                smallest=np.linalg.eigvalsh(np.eye(3)-v.T@v/w.sum())[0]
                ydist=np.linalg.norm(points-y,axis=1)
                for threshold in (.25,.5):
                    hh=ious>threshold
                    if hh.any() and not hh.all():
                        mixed[str(threshold)]+=1
                        certified=np.sqrt(max(0.,smallest))*ydist[~hh].min() > 2*er+ydist[hh].min()
                        certs[str(threshold)]+=bool(certified)
                        assert not certified or g>threshold
    byarm={}
    for group,data in grouped.items():
        z=np.stack(data)
        byarm[group]={}
        for j,arm in enumerate(ARMS):
            ms=np.array([(z[:,j]>.25).mean(),(z[:,j]>.50).mean(),z[:,j].mean()])*100
            byarm[group][arm]=dict(n=len(z),hits25=int((z[:,j]>.25).sum()),hits50=int((z[:,j]>.50).sum()),
                                   mIoU=float(ms[2]))
            if group not in ('active','fallback'):
                groupname='equal' if group=='all' else group
                saved=next(row for row in metric if row['group']==groupname and row['arm']==arm)
                np.testing.assert_allclose(ms,[float(saved[k]) for k in ('Acc25','Acc50','mIoU')],atol=1e-11,rtol=0)
                if arm!='ray':
                    pq=next(row for row in pairs if row['group']==groupname and row['arm']=='ray' and row['reference']==arm)
                    for t in (25,50):
                        b=z[:,j]>t/100
                        h=z[:,4]>t/100
                        assert int((h&~b).sum())==int(pq['rescue'+str(t)]) and int((b&~h).sum())==int(pq['harm'+str(t)])
    for (_,d,expected), p in zip(sorted(all_rows,key=lambda entry:entry[0]),published,strict=True):
        assert p['direction']==d
        np.testing.assert_array_equal(expected,[float(p[a+'_iou']) for a in ARMS])
    diff=np.asarray(eligible_differences)
    np.testing.assert_allclose(np.median(diff,axis=0),[
        pre['paired_true_center_rmse_control_minus_ray_m'][k] for k in ('reverse','mismatch')],atol=1e-10)
    # Integer-threshold checks independent of the score report's gating calculation.
    def hits(a):
        x=byarm['all'][a]
        return x['hits25']+x['hits50']
    necessity=all(hits('ray')>hits(arm) for arm in ('seed','origin','reverse','mismatch'))
    assert pre['checks']['beat_simple_and_wrong_controls']==necessity
    if not pre['expand']:
        assert not (PRIVATE/'ordinary').exists()
    assert checked(dest/'LOCK.json',seal['lock'])['code']['crossplatform/CURRENT_METHOD.json']==digest(ROOT/'crossplatform/CURRENT_METHOD.json')
    audit=dict(verified=192,first32_exact_native_replays=seal['pilot']['rows'],
               max_independent_energy_error_m2=maximum,individual_source_frames_by_scope=byarm,
               threshold_tails=tails,mixed_rows_for_certificate=mixed,sufficient_certificates=certs,
               paired_reverse_minus_ray_rmse_positive=int((diff[:,0]>0).sum()),
               paired_mismatch_minus_ray_rmse_positive=int((diff[:,1]>0).sum()),
               signed_check_n=len(diff),evidence_scope='source own calibration only',
               decision_verified=True,model_registry_sha256=digest(ROOT/'crossplatform/CURRENT_METHOD.json'),
               source_seal_sha256=digest(dest/'SEALED.json'),ordinary_inference_rows=0 if not pre['expand'] else None,
               initial_projection_seconds=seal['pilot']['projected_seconds'],source_capture_seconds=seal['seconds'],
               FINAL_access=False)
    atomic(OUT/'audit.json',audit)
    # Two tails, all changes retained in the full anonymized export.
    extremes = sorted(examples,key=lambda r:r['change'])
    write=[dict(tail='largest_loss',**r) for r in extremes[:3]]+[dict(tail='largest_gain',**r) for r in extremes[-3:][::-1]]
    with (OUT/'source_examples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=write[0]);writer.writeheader();writer.writerows(write)
    print('Verified',len(published),'rows; largest energy discrepancy',maximum,dict(tails))

if __name__=='__main__':
    run()

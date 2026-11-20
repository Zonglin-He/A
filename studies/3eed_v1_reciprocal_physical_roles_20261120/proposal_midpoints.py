"""Anchored proposal fusion and a separate post-seal same-instance geometric existence test."""
import csv
import gzip
from pathlib import Path

import numpy as np

from score_roles import HERE, OUT, PANELS, PRIVATE, anon, csv_file, digest, inventory, many_iou, read, save
from roles_joint import direction_tensor
from shape_path import from_coordinates


def coordinates(boxes):
    return np.column_stack((boxes[:,:3],np.log(boxes[:,5]),direction_tensor(boxes)))


def average_neighbours(anchor, pool):
    selected=pool[many_iou(pool,anchor)>.55]
    assert len(selected)>0
    if len(selected)==1:
        return anchor.copy()
    return from_coordinates(coordinates(selected).mean(0),anchor[6])


def pools(data):
    original=np.unique(data['pool'],axis=0)
    union=np.unique(np.vstack((original,data['other_pools'])),axis=0)
    return original,union


def predict(seal):
    path=PRIVATE/'geometry_averages.npz'
    pin=PRIVATE/'GEOMETRY_AVERAGE_SEAL.json'
    design=digest(HERE/'POSTSEAL.json')
    implementation=digest(Path(__file__))
    if path.exists():
        info=read(pin)
        assert info['predictions']==digest(path) and info['design']==design and info['code']==implementation
        return path
    tags=sorted(seal['files'])
    predicted=[]
    for tag in tags:
        data_path=PRIVATE/tag
        assert digest(data_path)==seal['files'][tag]
        with np.load(data_path,allow_pickle=False) as data:
            anchor=data['raw'][0]
            predicted.append([average_neighbours(anchor,cloud) for cloud in pools(data)])
    np.savez_compressed(path,tags=np.asarray(tags),boxes=np.array(predicted))
    save(pin,dict(n=len(tags),design=design,code=implementation,predictions=digest(path),uses_labels=False))
    return path


def oracle(anchor, pool, labels):
    overlaps=np.stack([many_iou(pool,box) for box in labels])>.25
    eligible=overlaps[0] & (overlaps.sum(0)==1)
    assert eligible.any()
    middle=(coordinates(anchor[None])+coordinates(pool[eligible]))/2
    boxes=np.stack([from_coordinates(row,anchor[6]) for row in middle])
    return float(many_iou(boxes,labels[0]).max()),int(eligible.sum())


def run():
    seal=read(PRIVATE/'SEAL.json')
    outputs=predict(seal)
    with np.load(outputs,allow_pickle=False) as f:
        predictions=dict(zip(f['tags'],f['boxes'],strict=True))
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:
        rows={(r['panel'],r['source'],r['case']):r for r in csv.DictReader(f) if r['arm']=='v1'}
    manifests={panel:read(path) for panel,path in PANELS.items()}
    results=[]
    for tag,predicted in predictions.items():
        panel,idx,source=tag[:-4].rsplit('_',2)
        idx=int(idx)
        case=anon(f'{panel}|{idx}')
        row=manifests[panel][idx]['row']
        labels,_=inventory(row['metapath'],row['platform'],int(row['suffix'] or 0),int(row['object_index']))
        primary=rows[panel,source,case]
        with np.load(PRIVATE/tag,allow_pickle=False) as data:
            anchor=data['raw'][0]
            raw=float(many_iou(anchor[None],labels[0])[0])
            assert abs(raw-float(primary['iou']))<1e-9
            for i,(name,cloud) in enumerate(zip(('A','A_B_D'),pools(data),strict=True)):
                candidate_iou=float(many_iou(predicted[i:i+1],labels[0])[0])
                on_target=primary['baseline_owner']=='T'
                best,n=(None,0) if not on_target else oracle(anchor,cloud,labels)
                results.append(dict(group=primary['group'],source=source,target=row['platform'],case=case,pool=name,
                    raw_iou=raw,direct_iou=candidate_iou,correct_original=int(on_target),
                    best_same_target_midpoint=best,known_same_target_proposals=n,
                    best_all_original=float(primary['best_all_axis_and_pca'])))
    assert len(results)==1002
    with gzip.open(OUT/'proposal_midpoint_rows.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,results[0]);w.writeheader();w.writerows(results)
    comparison=[]
    for group in ('own_proxy','cross_proxy','natural170','natural171'):
        for pool in ('A','A_B_D'):
            rr=[r for r in results if r['group']==group and r['pool']==pool]
            fixed=[r for r in rr if r['correct_original']]
            comparison.append(dict(group=group,pool=pool,n=len(rr),correct_original=len(fixed),
                all_rescue50=sum(r['raw_iou']<=.5<r['direct_iou'] for r in rr),
                all_harm50=sum(r['direct_iou']<=.5<r['raw_iou'] for r in rr),
                fixed_rescue50=sum(r['raw_iou']<=.5<r['direct_iou'] for r in fixed),
                fixed_harm50=sum(r['direct_iou']<=.5<r['raw_iou'] for r in fixed),
                strict_pool_misses=sum(r['best_all_original']<=.5 for r in fixed),
                direct_outside_all=sum(r['best_all_original']<=.5<r['direct_iou'] for r in fixed),
                oracle_midpoint_rescue50=sum(r['raw_iou']<=.5<r['best_same_target_midpoint'] for r in fixed),
                oracle_midpoint_outside_all=sum(r['best_all_original']<=.5<r['best_same_target_midpoint'] for r in fixed)))
    csv_file(OUT/'proposal_midpoints.csv',comparison)
    save(OUT/'proposal_midpoint_audit.json',dict(label_free_predictions=digest(outputs),rows=len(results),
        distinct_sealed_inputs=len(predictions),design=digest(HERE/'POSTSEAL.json'),code=digest(Path(__file__)),
        results_sha256=digest(OUT/'proposal_midpoint_rows.csv.gz'),posthoc=True,FINAL_access=False))


if __name__=='__main__':
    run()

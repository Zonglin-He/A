"""Post-seal oriented IoU with independent sensor geometry and a literal upstream bridge."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
from multiprocessing import get_context
from pathlib import Path
import secrets
import sys

import numpy as np
import torch

from decode165 import ARMS
from scope165 import HERE, ORDER, PAPER, PROTOCOL, ROOT, R96, WORK, atomic_json, canonical, check_lock, ordinary, read, sha
from common160 import npz_atomic
from audit160 import raw_annotations, sensor_frame, transform_gt, published_geometry, oriented_iou, yaw_matrix
from utils.eval_det import iou3d_rotated_vs_aligned

sys.path.insert(0,str(ROOT/'research/round6/source_validation_scoring'))
from batch_yaw_iou import many_iou

METRICS = ('mIoU','Acc25','Acc50')
THRESHOLDS = (.25,.50)
RESULTS = HERE/'results'


def table(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_table(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def official_values(boxes, gt):
    """Compute each distinct endpoint once, with the upstream FP32 corners and clipping."""
    unique, inverse = np.unique(boxes,axis=0,return_inverse=True)
    axis = unique[:,6] == 0
    ious = np.empty(len(unique),dtype=np.float64)
    if axis.any():
        mat,_ = iou3d_rotated_vs_aligned(torch.tensor(gt[None],dtype=torch.float32),
                                       torch.tensor(unique[axis,:6],dtype=torch.float32))
        ious[axis] = mat[0].numpy()
    for i in np.flatnonzero(~axis):
        ious[i] = published_geometry(unique[i],gt)
    return ious[inverse]


def sensor_values(boxes, gt, target, view):
    # Inverse coordinate transforms must retain float64; otherwise assignment into
    # the sealed float32 predictions introduces extra rounding in Waymo and Drone.
    world = np.stack([sensor_frame(b,target,view) for b in np.asarray(boxes,np.float64)])
    truth = sensor_frame(gt,target,view)
    unique,inverse = np.unique(world,axis=0,return_inverse=True)
    truth_rotation=yaw_matrix(truth[6])
    rotations=np.stack([yaw_matrix(b[6]) for b in unique])
    half_extents=(np.abs(rotations) @ unique[:,3:6,None])[:,:,0]/2
    gt_half=np.abs(truth_rotation) @ truth[3:6]/2
    relevant=np.flatnonzero((np.abs(unique[:,:3]-truth[:3])<=half_extents+gt_half+1e-9).all(1))
    measured=np.zeros(len(unique))
    for j in relevant:
        b=unique[j]
        measured[j]=oriented_iou(b[:3],b[3:6],rotations[j],truth[:3],truth[3:6],truth_rotation)
    return measured[inverse]


def fixture():
    gt=np.array([4.5,2.19,3.6,3.1,1.6,1.8,.37])
    shifted=gt.copy(); shifted[0]+=100
    axis=gt.copy(); axis[6]=0
    arms=np.stack([gt,shifted,axis,axis])
    for target,view in (('drone',0),('quad',0),('waymo',2)):
        ref=many_iou(arms,gt)
        replicate=sensor_values(arms,gt,target,view)
        np.testing.assert_allclose(ref,replicate,rtol=0,atol=1e-12)
        assert ref[0]>.99999 and ref[1]==0 and ref[2]==ref[3]
    literal=official_values(arms,gt)
    np.testing.assert_allclose(literal[1:],ref[1:],rtol=0,atol=2e-6)
    return dict(repaired_scoring_claim='Same oriented 3D IoU definition, stable float64 solver and separate literal upstream comparison',
                known_input='Identical rotated boxes in a constructed fixture', robust_IoU=float(ref[0]),
                literal_IoU=float(literal[0]), scoring_policy_changed_before_task_GT=True)


@lru_cache(maxsize=1)
def ordinary_index():
    return ordinary()


@lru_cache(maxsize=3)
def historical_gt(target):
    """Independently captured loader endpoints used only after the prediction seal."""
    name=R96/'experiment'/f'OFFICIAL_{target.upper()}_VAL_ORDER_PRIVATE.json'
    return read(name)


@torch.inference_mode()
def score_direction(job):
    cohort, device, source, target, n, digest = job
    torch.set_num_threads(1)
    name=f'{source}_to_{target}'
    folder=WORK/cohort/name
    out=folder/'IOU_AUDIT_PRIVATE.npz'
    source_file=folder/f'MATCHED_BOXES_{device.upper()}_PRIVATE.npz'
    assert sha(source_file)==digest
    if out.exists():
        with np.load(out,allow_pickle=False) as stored:
            assert str(stored['input_sha256'])==digest and len(stored['values'])==n
        return out
    with np.load(source_file,allow_pickle=False) as p:
        assert tuple(p['arms'])==ARMS
        ordinals, boxes, winners, flags = (p[k] for k in ('ordinals','boxes','winners','flags'))
    assert len(ordinals)==len(boxes)==n
    np.testing.assert_array_equal(winners[:,::2],winners[:,1::2])
    np.testing.assert_array_equal(boxes[:,::2,:6],boxes[:,1::2,:6])
    assert not boxes[:,::2,6].any()
    values, literal = np.empty((n,len(ARMS))),np.empty((n,len(ARMS)))
    rows=ordinary_index()[target]
    reference=historical_gt(target)
    assert len(rows)==len(reference)
    max_error=0.
    for k,ii in enumerate(ordinals):
        row=rows[int(ii)]
        caption, raw=raw_annotations(row['metapath'])[row['object_index']]
        assert canonical(caption)==row['caption']
        view=int(row['suffix'] or 0)
        gt=transform_gt(raw,target,view).astype(np.float64)
        ref=reference[row['anno_index']]
        assert row['object_index']==ref['object_index'] and ref['meta_path']==row['metapath']
        np.testing.assert_array_equal(gt,np.asarray(ref['source_GT7_float32'],np.float32))
        values[k]=many_iou(boxes[k],gt)
        literal[k]=official_values(boxes[k],gt)
        separate=sensor_values(boxes[k],gt,target,view)
        err=float(np.abs(values[k]-separate).max())
        max_error=max(max_error,err)
        if err>1e-7 or any(not np.array_equal(values[k]>t,separate>t) for t in THRESHOLDS):
            raise ValueError(f'Independent sensor geometry mismatch in {name}, ordinal {ii}, max abs {err:.3g}')
        if (k+1)%1024==0:
            print('R165 post-seal paired geometry',name,k+1,'/',n,flush=True)
    npz_atomic(out,ordinals=ordinals,values=values,literal=literal,winners=winners,
               flags=flags,yaw_changed=boxes[:,1::2,6]!=0,input_sha256=digest,sensor_convex_max_error=max_error)
    print('R165 paired scores and sensor audit sealed',name,n,flush=True)
    return out


def measure(values):
    return 100*np.stack((values.mean(0),(values>.25).mean(0),(values>.5).mean(0)),axis=-1)


def summarize(scope, groups, macro, paired):
    results, comparisons = [],[]
    matrices=[measure(a['values']) for a in groups]
    means=np.mean(matrices,axis=0) if macro else measure(np.concatenate([a['values'] for a in groups],axis=0))
    wins=np.concatenate([a['winners'] for a in groups],axis=0)
    values=np.concatenate([a['values'] for a in groups],axis=0)
    nz=np.concatenate([a['yaw'] for a in groups],axis=0)
    for j,arm in enumerate(ARMS):
        results.append(dict(**scope,arm=arm,n=len(values),**dict(zip(METRICS,means[j])),
                            altered_yaw=int(nz[:,j//2].sum()) if j%2 else 0))
    if paired:
        for label,base in read(PROTOCOL)['contrasts']:
            i,j=ARMS.index(label),ARMS.index(base)
            diff=values[:,i]-values[:,j]
            ix=100.*np.array([a['values'][:,i].mean()-a['values'][:,j].mean() for a in groups])
            if macro:
                np.testing.assert_allclose(ix.mean(),means[i,0]-means[j,0],rtol=0,atol=1e-10)
            indicators=[diff]+[(values[:,i]>cut).astype(int)-(values[:,j]>cut).astype(int) for cut in THRESHOLDS]
            row=dict(**scope,after=label,before=base,n=len(values),changed_winner=int((wins[:,i]!=wins[:,j]).sum()))
            for k,(m,d) in enumerate(zip(METRICS,indicators)):
                row.update({m+'_pp':means[i,k]-means[j,k],m+'_up':int((d>1e-10).sum()),m+'_down':int((d< -1e-10).sum())})
            comparisons.append(row)
    return results,comparisons


def anonymizer():
    alias_seed=WORK/'ALIASES_PRIVATE.bin'
    if not alias_seed.exists():
        with alias_seed.open('xb') as stream:
            stream.write(secrets.token_bytes(32))
    key=alias_seed.read_bytes()
    def alias(group,value):
        return group+'_'+hashlib.blake2b(str(value).encode(),digest_size=8,key=key).hexdigest()
    return alias


def record_groups(rows, ordinals):
    locations=[Path(rows[int(j)]['metapath']).resolve() for j in ordinals]
    return np.asarray([p.parent.parent.as_posix() for p in locations]),locations


def paper_bridge(section, matrices):
    names=['_to_'.join(t) for t in ORDER]
    old=read(R96/'frozen_scores_attempt003/ALL_METHOD_ROWS.json')
    p=read(PAPER)
    out={method:{d:[v[0],v[1]] for d,v in zip(p['directions'],p[method])}
         for method in ('BUTD-DETR','EDA','WildRefer','3EED_published')}
    for arm in ('official_bbf_axis','r158_pca'):
        out[arm]={d:measure(matrices[d]['values'])[ARMS.index(arm),1:].tolist() for d in names}
    if section=='ordinary':
        out['R96_official_bbf_native_batch']={f"{r['source']}_to_{r['target']}":[r['Acc25_percent'],r['Acc50_percent']]
            for r in old if r['arm']=='last_official_bbf'}
    results=[]
    for method,mat in out.items():
        assert set(mat)==set(names)
        kind='published_table2' if method in p else 'local_matched'
        if method.startswith('R96'):
            kind='local_historical_batch_sampling'
        for group,sc in list(mat.items())+[('six_equal',np.mean(list(mat.values()),axis=0))]:
            results.append(dict(direction=group,method=method,origin=kind,Acc25=sc[0],Acc50=sc[1]))
    return results


def full_bounds(groups):
    """Bounds induced by the unscored protected frames, without opening them."""
    n_expected=read(PROTOCOL)['official_population']['expected_eligible']
    result=[]
    for method in ('official_bbf_axis','r158_pca','r158_pca_minus_official_bbf_axis'):
        lower,upper=[],[]
        for (source,target),group in zip(ORDER,groups):
            n=n_expected[target]
            k=n-len(group['values'])
            scores=measure(group['values'])*len(group['values'])/n
            if method.endswith('minus_official_bbf_axis'):
                point=scores[ARMS.index('r158_pca')]-scores[ARMS.index('official_bbf_axis')]
                lo=point-100*k/n
            else:
                point=scores[ARMS.index(method)]
                lo=point
            hi=point+100*k/n
            lower.append(lo)
            upper.append(hi)
            result.append(dict(direction=f'{source}_to_{target}',method=method,missing=k,
                               **{f'{m}_{side}':a[j] for j,m in enumerate(METRICS) for side,a in (('lower',lo),('upper',hi))}))
        lo,hi=np.mean(lower,axis=0),np.mean(upper,axis=0)
        result.append(dict(direction='six_equal',method=method,missing=768,
                           **{f'{m}_{side}':a[j] for j,m in enumerate(METRICS) for side,a in (('lower',lo),('upper',hi))}))
    return result


def aggregate(cohort, device, paths, endpoint_seal):
    if cohort=='pilot':
        return
    rows=ordinary_index()
    alias=anonymizer()
    cases, matrix, pairs, recording_tables = [],[],[],[]
    max_error, literal_max_error = 0., 0.
    threshold_mismatch=np.zeros((len(ARMS),2),int)
    literal_invalid=np.zeros(len(ARMS),int)
    matrices, literal_matrices, recording_counts = {}, {}, defaultdict(set)
    for (s,t),path in zip(ORDER,paths):
        d=f'{s}_to_{t}'
        with np.load(path,allow_pickle=False) as f:
            values, literal, wins, yaw=f['values'],f['literal'],f['winners'],f['yaw_changed']
            max_error=max(max_error,float(f['sensor_convex_max_error']))
            error=np.abs(values-literal)
            if np.isfinite(error).any():
                literal_max_error=max(literal_max_error,float(error[np.isfinite(error)].max()))
            literal_invalid+=(~np.isfinite(literal)|(literal<0)|(literal>1.)).sum(0)
            for q,threshold in enumerate(THRESHOLDS):
                threshold_mismatch[:,q]+=((values>threshold)!=(literal>threshold)).sum(0)
            ordinals=f['ordinals']
            flags=f['flags']
        a=dict(values=values,winners=wins,yaw=yaw)
        matrices[d]=a
        literal_matrices[d]=dict(values=literal,winners=wins,yaw=yaw)
        report,pair=summarize(dict(group=d,source=s,target=t,aggregation='direction'),[a],False,True)
        matrix+=report
        pairs+=pair
        recordings, loc=record_groups(rows[t],ordinals)
        recording_counts[t].update(recordings.tolist())
        for recording in np.unique(recordings):
            m=recordings==recording
            entry,_=summarize(dict(group=alias('rec',recording),source=s,target=t,aggregation='recording'),
                              [{k:v[m] for k,v in a.items()}],False,False)
            recording_tables+=entry
        for j,(i,path) in enumerate(zip(ordinals,loc)):
            caption_slot=rows[t][int(i)]['object_index']
            r=dict(source=s,target=t,recording=alias('rec',recordings[j]),
                   frame=alias('frame',path),expression=alias('obs',(str(path),caption_slot)),
                   official_vs_native_switch=int(flags[j,2]))
            r.update(zip(ARMS,values[j].tolist()))
            cases.append(r)
    groups=list(matrices.values())
    for spec,subgroups,is_macro in (
        (dict(group='six_equal',source='all',target='all',aggregation='six_direction_macro'),groups,True),
        (dict(group='all_pooled',source='all',target='all',aggregation='pooled_micro'),groups,False),
        *((dict(group=t,source='both',target=t,aggregation='incoming_macro'),
           [matrices[f'{s}_to_{t}'] for s,x in ORDER if x==t],True) for t in ('waymo','drone','quad')),
    ):
        a,b=summarize(spec,subgroups,is_macro,True)
        matrix+=a
        pairs+=b
    cases.sort(key=lambda r:(r['source'],r['target'],r['expression']))
    recording_tables.sort(key=lambda r:(r['source'],r['target'],r['group'],r['arm']))
    write_table(RESULTS/'all_rows.csv',cases)
    write_table(RESULTS/'metrics.csv',matrix)
    write_table(RESULTS/'paired.csv',pairs)
    write_table(RESULTS/'recordings.csv',recording_tables)
    write_table(RESULTS/'paper_bridge.csv',paper_bridge(cohort,matrices))
    literal_rows=[]
    for d,matrix0 in literal_matrices.items():
        m=measure(matrix0['values'])
        literal_rows.extend(dict(direction=d,arm=a,n=len(matrix0['values']),**dict(zip(METRICS,m[j]))) for j,a in enumerate(ARMS))
    m=np.mean([measure(q['values']) for q in literal_matrices.values()],axis=0)
    literal_rows.extend(dict(direction='six_equal',arm=a,n=len(cases),**dict(zip(METRICS,m[j]))) for j,a in enumerate(ARMS))
    write_table(RESULTS/'literal_upstream_geometry.csv',literal_rows)
    if cohort=='ordinary':
        write_table(RESULTS/'full_cohort_bounds.csv',full_bounds(groups))
    audit=dict(cohort=cohort,official_direction_records=len(cases),arms=len(ARMS),
        primary_scorer='Defined oriented 3D IoU; float64 rectangle intersection and z overlap with independent sensor-frame convex hull',
        independent_gt_transform_checks=len(cases),
        independent_sensor_convex_geometry_endpoint_checks=len(cases)*len(ARMS),
        max_independent_sensor_abs_iou_error=max_error, max_literal_upstream_vs_primary_abs_iou_error=literal_max_error,
        literal_out_of_range_or_nonfinite=dict(zip(ARMS,literal_invalid.tolist())),
        literal_upstream_vs_primary_threshold_disagreements={a:dict(zip(METRICS[1:],v.tolist())) for a,v in zip(ARMS,threshold_mismatch)},
        recording_counts={t:len(x) for t,x in recording_counts.items()},endpoint_seal_sha256=sha(endpoint_seal),
        FINAL_access=False, all_six_directions=True, outputs={p.name:sha(p) for p in RESULTS.glob('*.csv')})
    atomic_json(HERE/'GEOMETRY_AUDIT.json',audit)
    print('Aggregate paired scoring complete',len(cases),max_error,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('cohort',choices=('fixture','pilot','ordinary'))
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    torch.set_num_threads(1)
    evidence=fixture()
    if args.cohort=='fixture':
        print(evidence)
        return
    check_lock()
    scorer_policy=read(HERE/'experiment/SCORER_ADDENDUM.json')
    assert not scorer_policy['box_labels_read_by_r165_before_amendment']
    name=HERE/f'{args.cohort.upper()}_{args.device.upper()}_ENDPOINT_SEAL.json'
    seal=read(name)
    jobs=[(args.cohort,args.device,s,t,item['n'],item['output_sha256'])
          for s,t in ORDER for item in (seal['per_direction'][f'{s}_to_{t}'],)]
    assert sum(j[-2] for j in jobs)==seal['n']
    if args.workers==1:
        paths=list(map(score_direction,jobs))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as pool:
            paths=list(pool.map(score_direction,jobs))
    aggregate(args.cohort,args.device,paths,name)


if __name__=='__main__':
    main()

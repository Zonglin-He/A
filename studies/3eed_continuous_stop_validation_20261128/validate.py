"""One precommitted C_ray-only test of direct dimension priors and image balancing."""
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'3eed_nullspace_solver_20261128'))
from solver import Arm, ITERATIONS, LAMBDA_IMAGE, LAMBDA_PRIOR, axial, image_residual, linearize, physical, ray_depth, scales_for
from run_experiment import ROOT, RUN as PREVIOUS, FrameEvidence, check, digest, load_blind, marks, publish, rays_for, read, rect_for
from shared import canonical

RUN = ROOT/'runs/continuous-stop-20261128'
BASE = Arm('none', image=True)
DIMENSION_COLUMNS = np.eye(7)[3:6]
SQRT_DAMPING = np.sqrt(.02)*np.eye(7)
CONFIG_FILES = (Path(__file__), HERE/'PROTOCOL_ZH.md',
                HERE.parent/'3eed_nullspace_solver_20261128/solver.py')


def singular_values(J):
    return np.linalg.svd(J, compute_uv=False) if J.shape[0] else np.zeros(1)


def rank(J):
    s = singular_values(J)
    return int((s>=max(.025,.08*s[0])).sum())


def evidence_info(J, K, sqrt_weight):
    combined = np.vstack((J,sqrt_weight*K))
    return dict(ray=rank(J), prior_image_weight=rank(np.vstack((J,np.sqrt(LAMBDA_IMAGE)*K))),
                balanced=rank(combined))


def dimensions_loss(box, size_prior, scale, evidence, weight):
    depth, face = ray_depth(box[None,:],evidence.eye,evidence.rays)
    _, measured, _ = physical(depth[0],face[0],evidence.observed,True)
    image, _ = image_residual(box[None,:],evidence)
    projected = weight*np.dot(image[0],image[0])/(2*max(1,image.shape[1]))
    size_delta = (box[3:6]-size_prior)/scale[3:6]
    return (float(measured.mean()) if len(measured) else 0.) + projected + .5*LAMBDA_PRIOR*np.dot(size_delta,size_delta)


def refine(seed, size_prior, evidence):
    """Regularize only the three dimensions, holding the image weight fixed."""
    scale, box = scales_for(seed), seed.copy()
    J, r, K, p, _, _ = linearize(box,scale,evidence,BASE)
    sqrt_weight = max(np.sqrt(LAMBDA_IMAGE),singular_values(J)[0]/max(singular_values(K)[0],.05))
    start = evidence_info(J,K,sqrt_weight)
    weight = sqrt_weight**2
    accepted = 0
    for step_num in range(ITERATIONS):
        if step_num:
            J,r,K,p,_,_=linearize(box,scale,evidence,BASE)
        size_delta = (box[3:6]-size_prior)/scale[3:6]
        jac = np.vstack((J,sqrt_weight*K,np.sqrt(LAMBDA_PRIOR)*DIMENSION_COLUMNS,SQRT_DAMPING))
        residual = np.concatenate((r,sqrt_weight*p,np.sqrt(LAMBDA_PRIOR)*size_delta,np.zeros(7)))
        step = np.linalg.lstsq(jac,-residual,rcond=None)[0]
        span = max(abs(step))
        if span<1e-7:
            break
        step *= min(1.,.4/span)
        before = dimensions_loss(box,size_prior,scale,evidence,weight)
        for fraction in (1.,.50,.25,.125):
            candidate = box+fraction*step*scale
            candidate[6] = axial(candidate[6])
            if min(candidate[3:6])>.08 and dimensions_loss(candidate,size_prior,scale,evidence,weight)<before-1e-10:
                box = candidate
                accepted += 1
                break
        else:
            break
    J,_,K,_,_,_=linearize(box,scale,evidence,BASE)
    return box,dict(start=start,end=evidence_info(J,K,sqrt_weight),steps=accepted,lambda_image=weight)


def one(job):
    uid,row,diag,inv,choice = job
    pool = inv['pool']['cray']['boxes']
    base = None if not pool else pool[choice['selected']['cray']['cached']]
    record = dict(uid=uid,target=row['target'],recording=row['recording'],route=diag['route'],
                  formal=base,refined=base,geometry=None,audit=None)
    if diag['route'] != 'C_ray':
        return record
    ev = FrameEvidence(row['frame'])
    window = rect_for(row,ev)
    exact = rays_for(ev,diag['selection'],window)
    held = rays_for(ev,diag['audit'],window)
    assert base is not None and len(exact.rays)
    model = np.asarray(row['raw']['3d']['value'],dtype=float).copy()
    # Qwen's absolute location is removed before extracting the dimension target.
    model[:3] = 0.
    size_prior = canonical(ev.module.to_sensor(model,ev.mat)[0])[3:6]
    original = canonical(base)
    refined,geometry = refine(original,size_prior,exact)
    record.update(refined=refined.tolist(),geometry=geometry,
                  audit=dict(formal=marks(original,held),refined=marks(refined,held)))
    return record


def predict():
    cv2.setNumThreads(1)
    RUN.mkdir(exist_ok=True,parents=True)
    rows,old,inv,choice,inputs = load_blind()
    prev = check(PREVIOUS/'predictions-private.json','artifact',PREVIOUS/'prediction-seal.json')
    assert prev['planned_input']==inputs
    start=perf_counter()
    jobs=((uid,row,old[uid],inv[uid],choice[uid]) for uid,row in rows.items())
    with ProcessPoolExecutor(8) as workers:
        out=list(workers.map(one,jobs,chunksize=5))
    reference={r['uid']:r['boxes']['formal'] for r in read(PREVIOUS/'predictions-private.json')}
    assert len(out)==len(reference)==702
    assert sum(x['geometry'] is not None for x in out)==320
    assert all(x['formal']==reference[x['uid']] for x in out)
    assert all(x['formal']==x['refined'] for x in out if x['geometry'] is None)
    artifact=RUN/'predictions-private.json'
    publish(artifact,out)
    publish(RUN/'prediction-seal.json',dict(artifact=digest(artifact),n=702,n_cray=320,
            prior_predictions=prev['artifact'],inputs=inputs,
            pins={p.name:digest(p) for p in CONFIG_FILES},target_labels_read=False,
            new_model_calls=0,checkpoint_training=False,cpu_wall_seconds=perf_counter()-start))
    print('Sealed one configuration for',len(out),'blind inputs and 320 refinements.',flush=True)


def quantiles(x):
    v=np.array(list(x))
    return dict(n=len(v), median=float(np.median(v)),p05=float(np.quantile(v,.05)),
                p95=float(np.quantile(v,.95))) if len(v) else dict(n=0)


def pan(rows):
    from evaluate_structure import stat_panel
    from inference_stats import cluster_weights, intervals

    summary=stat_panel(rows,('formal','refined','prior_no_size'), 'formal')
    if not rows:
        return summary
    record_weights=cluster_weights(rows)
    before=np.array([r['iou']['formal'] for r in rows])
    after=np.array([r['iou']['refined'] for r in rows])
    no_size=np.array([r['iou']['prior_no_size'] for r in rows])
    delta=100*(after-no_size)
    acc=100*((after>.5).astype(float)-(no_size>.5))
    summary['versus_prior_no_size']=dict(delta_miou=float(delta.mean()),miou_ci=intervals(delta,record_weights),
          delta_acc50=float(acc.mean()),acc50_ci=intervals(acc,record_weights))
    summary['refinement_n']=sum(r['geometry'] is not None for r in rows)
    summary['changed']=sum(r['changed'] for r in rows)
    summary['rescued_from_045_to_050']=int(((before>=.45)&(before<=.50)&(after>.50)).sum())
    summary['from_045_to_050_total']=int(((before>=.45)&(before<=.50)).sum())
    summary['old_no_size_rescued_from_045_to_050']=int(((before>=.45)&(before<=.50)&(no_size>.50)).sum())
    summary['old_no_size_total_rescue']=int(((before<=.50)&(no_size>.50)).sum())
    summary['old_no_size_total_harm']=int(((before>.50)&(no_size<=.50)).sum())
    summary['delta_center_m']=quantiles(r['delta_center_m'] for r in rows if r['delta_center_m'] is not None)
    eligible=[r for r in rows if r['geometry'] is not None]
    if eligible:
        def counts(stage, key):
            return dict(sorted(Counter(r['geometry'][stage][key] for r in eligible).items()))
        summary['ranks']={stage:{key:counts(stage,key) for key in ('ray','prior_image_weight','balanced')}
                          for stage in ('start','end')}
        summary['actual_rise_n']=sum(x['geometry']['start']['balanced']>x['geometry']['start']['ray'] for x in eligible)
        summary['old_weight_rise_n']=sum(x['geometry']['start']['prior_image_weight']>x['geometry']['start']['ray'] for x in eligible)
        summary['lambda_image']=quantiles(x['geometry']['lambda_image'] for x in eligible)
        summary['steps']=dict(sorted(Counter(x['geometry']['steps'] for x in eligible).items()))
    available=[r for r in rows if r['delta_audit'] is not None]
    if available:
        audit=np.array([r['delta_audit'] for r in available])
        summary['audit']=dict(n=len(audit),mean_m=float(audit.mean()),ci_m=intervals(audit,cluster_weights(available)),
            lower_cost_and_lower_iou=sum(r['delta_audit']<-1e-9 and r['iou']['refined']-r['iou']['formal']<-1e-9 for r in available))
    return summary


def score():
    from metric_image_common import inputs, setup
    from shared import score_boxes
    from truth_score_inputs import truth_data

    artifact=RUN/'predictions-private.json'
    seal=check(artifact,'artifact',RUN/'prediction-seal.json')
    assert seal['pins']=={p.name:digest(p) for p in CONFIG_FILES}
    assert not seal['target_labels_read'] and seal['new_model_calls']==0 and not seal['checkpoint_training']
    archived=check(PREVIOUS/'scores-private.json','scored',PREVIOUS/'aggregate.json')
    assert archived['prediction']['artifact']==seal['prior_predictions']
    previous={r['uid']:r for r in read(PREVIOUS/'scores-private.json')}
    setup()
    truth,_=truth_data(inputs())
    assert len(truth)==665 and truth.keys()==previous.keys()
    rows=[]
    for r in read(artifact):
        uid=r['uid']
        if uid not in truth:
            continue
        score=score_boxes({k:r[k] for k in ('formal','refined')},truth[uid]['box'])
        ious={k:v['iou'] for k,v in score.items()}
        assert abs(ious['formal']-previous[uid]['iou']['formal'])<1e-10
        ious['prior_no_size']=previous[uid]['iou']['joint_no_prior']
        audit=r['audit']
        diff=(audit['refined']['truncated_m']-audit['formal']['truncated_m']
              if audit and all(audit.values()) else None)
        meta={key:r[key] for key in ('uid','target','recording','route','geometry')}
        rows.append(dict(meta,iou=ious,changed=r['refined']!=r['formal'],
                         delta_center_m=score['refined']['center']-score['formal']['center']
                          if 'center' in score['formal'] else None,delta_audit=diff))
    cray=[r for r in rows if r['route']=='C_ray']
    exact=[r for r in rows if r['route']!='C_ray']
    assert len(cray)==315 and len(exact)==350
    assert all(r['geometry'] is None and r['iou']['formal']==r['iou']['refined'] and not r['changed'] for r in exact)
    panels={k:pan(v) for k,v in dict(all=rows,cray=cray,unchanged_fallback=exact).items()}
    for target in ('waymo','drone','quad','kitti'):
        panels[target]=pan([r for r in rows if r['target']==target])
        panels['cray_'+target]=pan([r for r in cray if r['target']==target])
    result=panels['cray']['metrics']['refined']
    passes=all(result[f]>0 for f in ('delta_miou','delta_acc50')) and all(result[f][0]>0 for f in ('miou_ci','acc50_ci'))
    private=RUN/'scores-private.json'
    publish(private,rows)
    publish(RUN/'aggregate.json',dict(panels=panels,passing=passes,primary_panel='cray',prediction=seal,
                score=digest(private),prior_scores=archived['scored'],offline_gt=True))
    print('Stopping rule passed:',passes,flush=True)
    for key in ('cray','all'):
        p=panels[key]
        print(key,p['n'],p['metrics'],p['ranks'],p['lambda_image'],flush=True)


if __name__=='__main__':
    ap=ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('predict','score'))
    {'predict':predict,'score':score}[ap.parse_args().stage]()

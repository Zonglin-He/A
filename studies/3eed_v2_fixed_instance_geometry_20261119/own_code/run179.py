"""Fit on cached source features, seal target choices, then score exact box pairs."""

import argparse
from pathlib import Path
import secrets
import sys
from time import perf_counter

import numpy as np

from quality179 import (BOX_FEATURES, POINT_FEATURES, choose, margins, normalized_ranking,
                       pair_evidence, rankings, supported, threshold_from_source)


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
sys.path.insert(0,str(PRIOR))
from scope import ROOT, PLATFORMS, evaluation_rows, file_sha, read, save
from evaluate import (DIRECTIONS, PRIVATE as EARLIER, fingerprint as earlier_fingerprint,
                      old_boxes, recording_intervals, save_csv, score_boxes, sensor_truth, to_sensor)
from model import NATIVE, VOTES

sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'))
from runner_helpers import sample_points
from crossplatform.round7.camera_projection import WAYMO_ANGLES, z_rotation

CALIBRATION = HERE/'calibration_PRIVATE'
PREDICTIONS = HERE/'predictions_PRIVATE'
METRICS = ('Acc25','Acc50','mIoU')
SELECTORS = ('direct_physical','geometry_only_pair','physical_pair')


def protocol():
    return read(HERE/'PROTOCOL.json')


def fingerprint():
    return {name:file_sha(HERE/name) for name in ('PROTOCOL.json','run179.py','quality179.py')}


def verify_inputs(*, target=False):
    cfg = protocol()
    pins = {
        PRIOR/'features_PRIVATE/SEALED.json': 'source_feature_seal_sha256',
        PRIOR/'inputs_PRIVATE/SEAL.json': 'source_input_seal_sha256',
        PRIOR/'weights_PRIVATE/SEALED.json': 'refinement_weight_seal_sha256',
        ROOT/'crossplatform/CURRENT_METHOD.json': 'v1_registration_sha256',
        ROOT/'crossplatform/ground.py': 'v1_entrypoint_sha256',
        ROOT/'crossplatform/round7/experiment/runner_helpers.py': 'point_sampler_sha256',
    }
    if target:
        pins[EARLIER/'SEALED.json'] = 'old_prediction_seal_sha256'
    for path, key in pins.items():
        if file_sha(path) != cfg[key]:
            raise RuntimeError(f'Frozen input differs from its pin: {key}')


def sensor_points(points, platform, view=None):
    xyz = points.astype(np.float64)
    if platform=='drone':
        xyz[:,2] -= 1.8
    elif platform=='waymo' and view is not None:
        xyz = xyz @ z_rotation(WAYMO_ANGLES[view])
    return xyz


def source_pairs(source, subset, head):
    import torch
    from features import FEATURES
    from train import read_features, take

    data = read_features(FEATURES,source,subset)
    entries = torch.nonzero(data['origin']<2,as_tuple=True)[0]
    feats, sizes, direct, ious = [],[],[],[]
    err = 0.
    for chunk in entries.split(8):
        batch = take(data,chunk)
        out = head.pool(batch,use_votes=True,use_mask=True)
        row = torch.arange(len(chunk),device=chunk.device)
        carrier = batch['v1_index'].long()
        raw = batch['v1_box'].cpu().numpy()
        np.testing.assert_array_equal(out['proposal'][row,carrier,:3].cpu().numpy(),raw[:,:3])
        refined = out['refined'][row,carrier].cpu().numpy()
        points = batch['points'].cpu().numpy()
        truth = batch['gt'][:,0].cpu().numpy()
        views = batch['view_index'].cpu().numpy()
        for p,a,b,gt,view in zip(points,raw,refined,truth,views):
            pair = np.stack((a,b)).astype(np.float64)
            output, error = score_boxes(pair,gt.astype(np.float64))
            both = to_sensor(pair,source,int(view))
            contrast,count,baseline = pair_evidence(sensor_points(p,source,int(view)),*both)
            feats.append(contrast)
            sizes.append(count)
            direct.append(baseline)
            ious.append(output)
            err = max(err,error)
    return dict(x=np.stack(feats), counts=np.stack(sizes), direct=np.asarray(direct),
                ious=np.stack(ious), independent_iou_error=err)


def all_scores(calibrated, x, direct):
    return dict(direct_physical=direct,
                geometry_only_pair=margins(calibrated['geometry_only_pair'],x[:,len(POINT_FEATURES):]),
                physical_pair=margins(calibrated['physical_pair'],x))


def fit():
    import torch
    from train import load_head, prepare_device

    verify_inputs()
    prepare_device()
    cfg = protocol()['source_only']
    start = perf_counter()
    CALIBRATION.mkdir(mode=0o700)
    result = {}
    with torch.inference_mode():
        for source in PLATFORMS:
            refiner,_ = load_head(source)
            fitting, validation = [source_pairs(source,split,refiner) for split in ('fit','calibration')]
            assert len(fitting['x'])==640 and len(validation['x'])==cfg['calibration_counts'][source]
            train_ok, val_ok = supported(fitting['counts']), supported(validation['counts'])
            labels = fitting['ious'][:,1]-fitting['ious'][:,0]
            models = {
                'geometry_only_pair': normalized_ranking(fitting['x'][:,len(POINT_FEATURES):],labels,train_ok,cfg),
                'physical_pair': normalized_ranking(fitting['x'],labels,train_ok,cfg),
            }
            train_scores = all_scores(models,fitting['x'],fitting['direct'])
            heldout = all_scores(models,validation['x'],validation['direct'])
            thresholds, evaluations = {},{}
            for name in SELECTORS:
                tau, audit = threshold_from_source(train_scores[name],train_ok,heldout[name],val_ok,
                                                    validation['ious'],cfg['threshold_quantiles'])
                thresholds[name] = tau
                evaluations[name] = dict(threshold_audit=audit, ranking=rankings(heldout[name],validation['ious'],
                                          val_ok,cfg['tie_epsilon_iou']))
            result[source] = dict(models=models, thresholds=thresholds, validation=evaluations,
                 n_fit=len(train_ok),n_heldout=len(val_ok),heldout_eligible=int(val_ok.sum()),
                 heldout_raw=metrics(validation['ious'][:,0]),heldout_refined=metrics(validation['ious'][:,1]),
                 max_independent_source_iou_error=max(fitting['independent_iou_error'],validation['independent_iou_error']))
            np.savez_compressed(CALIBRATION/f'{source}_pairs.npz',
                **{f'{split}_{key}':value for split,d in [('fit',fitting),('calibration',validation)]
                   for key,value in d.items()})
            print(source,'fit',len(train_ok),'eligible',int(train_ok.sum()),'selected thresholds',thresholds,flush=True)
            del refiner,fitting,validation
            torch.cuda.empty_cache()
    save(CALIBRATION/'SOURCE_LOCK.json',dict(implementation=fingerprint(), sources=result,
         physical_features=list(POINT_FEATURES+BOX_FEATURES), source_only=True,
         target_predictions_used=False, target_labels_used=False, FINAL_access=False,
         elapsed_seconds=perf_counter()-start))


def calibrated():
    data = read(CALIBRATION/'SOURCE_LOCK.json')
    assert data['implementation']==fingerprint()
    assert data['source_only'] and not data['target_labels_used'] and not data['FINAL_access']
    assert set(data['sources'])==set(PLATFORMS)
    return data


def load_past_pair(rows, source, target, previous):
    path = EARLIER/f'{source}_to_{target}.npz'
    assert file_sha(path)==previous['files'][path.name]['sha256']
    ordinal, incumbent = old_boxes(rows,source,target)
    with np.load(path,allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['ordinals'],ordinal)
        scores = archive['picked']
        raw, old_quality = scores[:,0],scores[:,1]
        np.testing.assert_array_equal(raw,incumbent)
        index = archive['indices'][:,0]
        assert np.all((index>=0)&(index<NATIVE))
        row = np.arange(len(raw))
        refined = archive['pool'][row,NATIVE+VOTES+index]
        np.testing.assert_allclose(raw,archive['pool'][row,index],atol=1e-12,rtol=0)
        old_selected = ~np.equal(old_quality,raw).all(axis=1)
        np.testing.assert_allclose(old_quality[old_selected],refined[old_selected],atol=1e-12,rtol=0)
    return dict(ordinals=ordinal,raw=raw,refined=refined,old=old_quality,old_selected=old_selected,carrier=index)


def predict():
    verify_inputs(target=True)
    locked = calibrated()
    previous, lock = read(EARLIER/'SEALED.json'),read(EARLIER/'LOCK.json')
    assert previous['lock_sha256']==file_sha(EARLIER/'LOCK.json') and lock['implementation']==earlier_fingerprint()
    assert not previous['FINAL_access'] and not previous['target_labels_read']
    rows = evaluation_rows()
    cfg = protocol()
    assert len(rows)*2==cfg['evaluation']['direction_records']
    PREDICTIONS.mkdir(mode=0o700)
    files = {}
    checked_points = set()
    start = perf_counter()
    for target in PLATFORMS:
        sources = tuple(s for s in PLATFORMS if s!=target)
        past = {s:load_past_pair(rows,s,target,previous) for s in sources}
        ordinals = past[sources[0]]['ordinals']
        np.testing.assert_array_equal(ordinals,past[sources[1]]['ordinals'])
        assert len(ordinals)==cfg['evaluation']['per_direction']
        evidence = {s:[] for s in sources}
        for j, k in enumerate(ordinals):
            row = rows[int(k)]
            # The frozen paired carriers used seed 0 and this prefix of the v1 sampling.
            point_key = (row['pointpath'],row['point_sha256'])
            if point_key not in checked_points:
                if file_sha(point_key[0])!=point_key[1]:
                    raise RuntimeError('LiDAR bytes differ from the registered pair inputs')
                checked_points.add(point_key)
            xyz,_,_ = sample_points(dict(row,seed=cfg['geometry_pair']['point_seed']))
            points = sensor_points(xyz[:cfg['geometry_pair']['points']],target)
            for source in sources:
                values = past[source]
                evidence[source].append(pair_evidence(points,values['raw'][j],values['refined'][j]))
        for source in sources:
            old = past[source]
            x, counts, direct = (np.stack(v) for v in zip(*evidence[source]))
            eligible = supported(counts)
            cal = locked['sources'][source]
            scores = all_scores(cal['models'],x,direct)
            accepted = [choose(scores[key],eligible,cal['thresholds'][key]) for key in SELECTORS]
            options = np.stack((np.zeros(len(x),bool),np.ones(len(x),bool),old['old_selected'],*accepted),axis=1)
            picked = np.where(options[...,None],old['refined'][:,None],old['raw'][:,None])
            picked[:,2] = old['old']
            assert np.isfinite(picked).all() and (picked[...,3:6]>0.).all()
            name = f'{source}_to_{target}.npz'
            p = PREDICTIONS/name
            np.savez_compressed(p,ordinals=ordinals,carrier=old['carrier'],boxes=picked,
                  switches=options,supported=eligible,margin=np.stack([scores[a] for a in SELECTORS],axis=1))
            files[name] = dict(sha256=file_sha(p),n=len(x))
            print(source,target,'sealed candidates',len(x),'eligible',int(eligible.sum()),flush=True)
    save(PREDICTIONS/'SEALED.json',dict(n=sum(r['n'] for r in files.values()),files=files,
         source_lock_sha256=file_sha(CALIBRATION/'SOURCE_LOCK.json'),
         earlier_sha256=file_sha(EARLIER/'SEALED.json'),implementation=fingerprint(),
         target_labels_used=False,target_updates=0,FINAL_access=False,
         elapsed_seconds=perf_counter()-start))


def metrics(ious):
    return dict(Acc25=float(100*(ious>.25).mean()),Acc50=float(100*(ious>.50).mean()),
                mIoU=float(100*ious.mean()))


def threshold_counts(ious, chosen, eligible, threshold):
    a, b = (ious[:,:2]>threshold).T
    possible = ~a & b
    certainly_harmful = a & ~b
    gained, lost = possible & chosen, certainly_harmful & chosen
    return dict(recoverable=int(possible.sum()), picked_recoverable=int(gained.sum()),
        remaining=int((possible & ~chosen).sum()), support_blocked_recoverable=int((possible & ~eligible).sum()),
        at_risk=int(certainly_harmful.sum()), harmed=int(lost.sum()), net=int(gained.sum()-lost.sum()))


def score():
    verify_inputs(target=True)
    locked = calibrated()
    pin = read(PREDICTIONS/'SEALED.json')
    assert not pin['target_labels_used'] and not pin['FINAL_access']
    assert pin['implementation']==fingerprint() and pin['source_lock_sha256']==file_sha(CALIBRATION/'SOURCE_LOCK.json')
    assert pin['earlier_sha256']==file_sha(EARLIER/'SEALED.json')
    prev = read(EARLIER/'SEALED.json')
    rows = evaluation_rows()
    cfg = protocol()
    arms = tuple(cfg['arms'])
    data, audit, anonymous = {},[],[]
    err, n = 0., 0
    # Verify every complete sealed direction before allowing any target annotation read.
    ready = {}
    for s,t in DIRECTIONS:
        name=f'{s}_to_{t}.npz'
        path=PREDICTIONS/name
        assert file_sha(path)==pin['files'][name]['sha256']
        before=load_past_pair(rows,s,t,prev)
        with np.load(path,allow_pickle=False) as a:
            cols = {key:a[key] for key in a.files}
        np.testing.assert_array_equal(cols['ordinals'],before['ordinals'])
        np.testing.assert_array_equal(cols['carrier'],before['carrier'])
        np.testing.assert_array_equal(cols['boxes'][:,:3],np.stack((before['raw'],before['refined'],before['old']),axis=1))
        assert len(cols['ordinals'])==cfg['evaluation']['per_direction']
        assert not np.any(cols['switches'][:,3:] & ~cols['supported'][:,None])
        picked = np.where(cols['switches'][:,3:,None],before['refined'][:,None],before['raw'][:,None])
        np.testing.assert_array_equal(cols['boxes'][:,3:],picked)
        ready[s,t]=cols
    assert pin['n']==sum(len(x['ordinals']) for x in ready.values())==cfg['evaluation']['direction_records']
    for (s,t), cols in ready.items():
        outcomes, recordings = [],[]
        for j, ordinal in enumerate(cols['ordinals']):
            key = rows[int(ordinal)]
            truth = sensor_truth(t,key['metapath'],key['caption'],key['suffix'])
            ious, delta = score_boxes(cols['boxes'][j],truth)
            err = max(err,delta)
            outcomes.append(ious)
            recordings.append(key['scene_cluster'])
            anonymous.append(dict(source=s,target=t,eligible=int(cols['supported'][j]),
                **{f'switch_{name}':int(cols['switches'][j,k]) for k,name in enumerate(arms)},
                **{name:float(value) for name,value in zip(arms,ious)}))
            n+=1
        data[f'{s}_to_{t}'] = dict(ious=np.stack(outcomes),recordings=np.asarray(recordings),
              eligible=cols['supported'],switches=cols['switches'],margin=cols['margin'])
        print(s,t,'score',len(outcomes),flush=True)
    assert len(anonymous)==n
    table = []
    diag = {}
    for group in (*data,'six_equal'):
        blocks = tuple(data.values()) if group=='six_equal' else (data[group],)
        block = {name:np.concatenate([d[name] for d in blocks],axis=0)
                 for name in ('ious','eligible','switches','margin')}
        ious, eligible = block['ious'],block['eligible']
        diag[group] = dict(n=len(ious),eligible=int(eligible.sum()),
            ranking={name:rankings(block['margin'][:,k],ious,eligible,cfg['source_only']['tie_epsilon_iou'])
                     for k,name in enumerate(SELECTORS)})
        for k,name in enumerate(arms):
            outcome = ious[:,k]
            diff = outcome-ious[:,0]
            selected = block['switches'][:,k]
            d25, d50 = (threshold_counts(ious,selected,eligible,t) for t in (.25,.50))
            table.append(dict(direction=group,arm=name,n=len(ious),switches=int(selected.sum()),
                         **metrics(outcome), win=int((diff>1e-12).sum()),loss=int((diff<-1e-12).sum()),
                         recover25=d25['picked_recoverable'],harm25=d25['harmed'],
                         recover50=d50['picked_recoverable'],harm50=d50['harmed']))
            diag[group][name] = {'25':d25,'50':d50}
        oracle = ious[:,:2].max(axis=1)
        table.append(dict(direction=group,arm='pair_oracle',n=len(oracle),switches=int((ious[:,1]>ious[:,0]).sum()),
                     **metrics(oracle),win=int((oracle-ious[:,0]>1e-12).sum()),loss=0,
                     recover25=int(((oracle>.25)&(ious[:,0]<=.25)).sum()),harm25=0,
                     recover50=int(((oracle>.50)&(ious[:,0]<=.50)).sum()),harm50=0))
    same_cohort={p:len(np.unique(next(d['recordings'] for k,d in data.items() if k.endswith('_to_'+p)))) for p in PLATFORMS}
    intervals = recording_intervals(data,('v1',*arms[1:]))
    intervals['recording_counts']=same_cohort
    intervals['difference_orientation']='method minus v1, percentage points'
    intervals['intervals']={'v1_raw' if method=='v1' else method:[[-high,-low] for low,high in bounds]
                            for method,bounds in intervals['intervals'].items()}
    intervals['limited_recordings_note']='Target Drone and Quad each have two recordings; resampling is descriptive.'
    out = HERE/'results'
    out.mkdir()
    save_csv(out/'direction.csv',table)
    secrets.SystemRandom().shuffle(anonymous)
    save_csv(out/'anonymous_rows.csv',anonymous)
    save(out/'diagnostics.json',diag)
    save(out/'recording_intervals.json',intervals)
    save(out/'source_calibration.json',locked)
    save(out/'audit.json',dict(n=n,exact_base_matches=n,fixed_carrier=n,endpoint_checks=n*len(arms),
         oriented_iou_max_abs_error=err,threshold_disagreements=0,recording_counts=same_cohort,
         target_prediction_seal_sha256=file_sha(PREDICTIONS/'SEALED.json'),
         source_lock_sha256=file_sha(CALIBRATION/'SOURCE_LOCK.json'),label_access_after_seal=True,
         target_updates=0,FINAL_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('fit','predict','score'))
    {'fit':fit,'predict':predict,'score':score}[parser.parse_args().phase]()

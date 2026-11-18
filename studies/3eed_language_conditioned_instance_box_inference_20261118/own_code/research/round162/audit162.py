"""Independently verify task scores in the original sensor frame."""
from collections import defaultdict
import csv
import sys

import numpy as np

from manifest162 import DEV, REC, HERE, ORDER, ROOT, WORK, atomic_json, canonical, prediction_file, read, sha, target_rows
from score162 import FROZEN, HEADLINES, METRICS, table, verified
from run162 import SETTINGS
from audit160 import raw_annotations, sensor_frame, transform_gt, oriented_iou, yaw_matrix
from crossplatform.ground_single_frame import orient_box


def sensor_iou(model_box, gt, platform, view):
    box = sensor_frame(np.asarray(model_box, np.float64),platform,view)
    if np.any(box[3:6] <= 0):
        raise ValueError('Invalid selected dimensions')
    return oriented_iou(box[:3],box[3:6],yaw_matrix(box[6]),gt[:3],gt[3:6],yaw_matrix(gt[6]))


def independent_metrics(values, weights):
    return [100 * sum(w*f(y) for y,w in zip(values,weights)) / sum(weights)
            for f in (lambda x:x,lambda x:x>.25,lambda x:x>.50)]


def main(scope):
    ids = verified(scope)
    dest = HERE / 'results' / scope
    raw = target_rows()
    measurements = table(dest/'rows.csv')
    result_index = {(r['cohort'],int(r['ordinal'])):r for r in measurements}
    assert len(ids) == len(measurements) == len(result_index)
    look = {}
    for file in ('research/round69/input/SCORING_INDEX_PRIVATE.json', 'research/round76/input/SOURCE_TRAIN512_PRIVATE.json'):
        look.update({r['id']: int(r['object_index']) for r in read(ROOT/file)})
    total, oracle_checks, exact_pca, worst, trace_bad = 0, 0, 0, 0., 0
    source_settings = read(SETTINGS)['sources']
    groups = defaultdict(list)
    oracle_subset = defaultdict(int)
    for n,i in enumerate(ids,1):
        entry = raw[i]
        r = entry['r']
        t,s,c,k = (entry[key] for key in ('target','source','cohort','ordinal'))
        row = result_index[c,k]
        assert (s,t)==(row['source'],row['target']) and float(row['weight'])==entry['sampling_weight']
        view = int(r['suffix'] or 0)
        j = look[r['id']] if c==REC else int(r['object_index'])
        expression, gt = raw_annotations(r['metapath'])[j]
        assert canonical(expression) == canonical(r['caption'])
        target = sensor_frame(transform_gt(gt,t,view).astype(float),t,view)
        key = c,s,t
        with np.load(WORK/scope/f'{i:05d}.npz',allow_pickle=False) as pred:
            all_boxes = dict(zip(pred['arms'].tolist(),pred['boxes']))
            all_boxes['r158_full'] = pred['old_pca']
            all_boxes['source_selected_simple'] = all_boxes[source_settings[s]['simple_control']]
            for arm, box in all_boxes.items():
                value = float(sensor_iou(box,target,t,view))
                scored = float(row[arm])
                assert (value>.25,value>.50)==(scored>.25,scored>.50)
                worst = max(worst, abs(scored-value))
                total += 1
            if oracle_subset[key] < 2:
                with np.load(prediction_file(i),allow_pickle=False) as prior:
                    raw6 = prior['raw']
                raw7 = np.c_[raw6,np.zeros(len(raw6))]
                raw7[:,3:6] = np.maximum(raw7[:,3:6],1e-6)
                def vmax(boxes):
                    return max((sensor_iou(box,target,t,view) for box in boxes),default=0.)
                assert abs(vmax(raw7)-float(row['oracle_axis']))<1e-7
                raw7[:,6] = pred['all_pca']
                assert abs(vmax(raw7)-float(row['oracle_pca']))<1e-7
                assert abs(vmax(pred['generated'])-float(row['oracle_lcib']))<1e-7
                oracle_checks += 1
                oracle_subset[key] += 1
            if 'ev_points' in pred and exact_pca < 6*256:
                points, boxes, yaw = pred['ev_points'], pred['ev_boxes'], pred['all_pca']
                for b,angle in zip(boxes,yaw):
                    assert orient_box(points,b)[6] == angle
                    exact_pca += 1
            assert int(pred['sweeps'])==int(pred['serial_sweeps'])
            assert np.isfinite(pred['trace']).all()
            trace_bad += int((np.diff(pred['trace'][:,1])>1e-7).any())
        groups[key].append(row)
        if n%1024==0:
            print('Sensor-frame audit',scope,n,len(ids),flush=True)
    if worst > 1e-7:
        raise ValueError('Sensor-frame IoU discrepancy')
    means = {}
    for r in table(dest/'directions.csv'):
        key=tuple(r[k] for k in ('cohort','source','target'))
        records=groups[key]
        arm = r['arm']
        values = independent_metrics([float(x[arm]) for x in records],[float(x['weight']) for x in records])
        assert len(records)==int(r['n'])
        assert np.allclose(values,[float(r[k]) for k in METRICS],atol=1e-9,rtol=0.)
        means[(*key,arm)]=np.array(values)
    for r in table(dest/'overall.csv'):
        values=np.mean([means[r['cohort'],s,t,r['arm']] for s,t in ORDER],axis=0)
        assert np.allclose(values,[float(r[k]) for k in METRICS],atol=1e-9,rtol=0.)
    for row in table(dest/'paired.csv'):
        c,src,tgt,a,b=(row[k] for k in ('cohort','source','target','before','after'))
        cells=ORDER if src=='six_equal' else [(src,tgt)]
        pairs, up, down=[],np.zeros(3,int),np.zeros(3,int)
        for s,t in cells:
            pairs.append(means[c,s,t,b]-means[c,s,t,a])
            for r in groups[c,s,t]:
                x,y=float(r[a]),float(r[b])
                vals=np.array([y-x,int(y>.25)-int(x>.25),int(y>.50)-int(x>.50)])
                up+=(vals>1e-12)
                down+=(vals< -1e-12)
        for key,d,p,n in zip(METRICS,np.mean(pairs,axis=0),up,down):
            assert abs(d-float(row[key+'_pp']))<1e-9
            assert (int(row[key+'_gains']),int(row[key+'_losses']))==(p,n)
    atomic_json(dest/'AUDIT.json',dict(status='verified',rows=len(ids),endpoint_sensor_frame_checks=total,
                max_abs_IoU_error=worst,independent_oracle_subset_rows=oracle_checks,exact_R155_PCA_checks=exact_pca,
                paired_positive_and_negative_counts_verified=True,serial_support_budget_identical=True,
                joint_objective_nondecreasing_cases=trace_bad,
                score_seal_sha256=sha(dest/'SCORE.json'),result_rows_sha256=sha(dest/'rows.csv'),FINAL_access=False))
    print('Independent audit',scope,len(ids),total,worst,flush=True)


if __name__=='__main__':
    main(sys.argv[1])

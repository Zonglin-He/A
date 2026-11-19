"""Recheck all sealed endpoints and full-model readout choices with a separate scorer."""
from concurrent.futures import ProcessPoolExecutor
import csv
from multiprocessing import get_context

import numpy as np

from study import ARMS, ORDER, PREDICTIONS, PRIVATE, RESULTS, camera_view, canonical, json_atomic, ordinary_rows, read, sha, state
from analyse_factorial import CONTRASTS, METRICS, blocks, metric, merged_boxes, verify_seal
from independent_audit import independent_annotations, sensor_iou, transform_gt
from predict import NAMES as OLD_NAMES


def verify_direction(direction):
    source,target = direction
    name = f'{source}_to_{target}'
    file = PRIVATE/'score'/f'{name}.npz'
    assert sha(file)==read(PRIVATE/'SCORING_SEAL.json')['private_scores'][file.name]
    with np.load(file,allow_pickle=False) as f:
        primary = dict(f)
    values = primary['values']
    targets = ordinary_rows(target)
    old_joint = OLD_NAMES.index('joint')-1
    max_error = 0.
    disagreement = 0
    for x,z in blocks(name):
        ends = merged_boxes(x,z)
        support = z['subject_slots'][:,old_joint]
        for j,idx in enumerate(x['ordinals']):
            row = targets[int(idx)]
            caption,b = independent_annotations(row['metapath'])[row['object_index']]
            assert canonical(caption)==row['caption']
            view = camera_view(row)
            truth = transform_gt(b,target,view)
            delta = sensor_iou(np.concatenate((ends[j],support[j])),truth,target,view)
            s = delta[len(ARMS):]
            u = int(np.argsort(-x['subject_unary'][j],kind='stable')[0])
            pairs = x['joint_score'][j]
            cond = (np.argsort(-pairs,axis=0,kind='stable')[0] if z['is_relation'][j]
                    else np.repeat(u,4))
            obtained = np.r_[delta[:len(ARMS)],s,s[u],s[cond]]
            d = primary['diag'][idx]
            expected = np.r_[values[idx],primary['slots'][idx],d[2],primary['conditional'][idx]]
            max_error = max(max_error,float(np.abs(obtained-expected).max()))
            disagreement += sum(np.count_nonzero((obtained>t)!=(expected>t)) for t in (.25,.50))
            must = primary['gold_columns'][idx]
            if must.any():
                proposals = ((float(pairs[a,b]),a,b) for a in range(4) for b in range(4) if must[b])
                # Stable row-major selection, with the score first.
                _,a,_ = max(proposals,key=lambda c:(c[0],-c[1],-c[2]))
                assert abs(float(s[a])-float(d[8]))<1e-7
            assert abs(float(s.max())-float(d[1]))<1e-7
    assert max_error<1e-7 and disagreement==0
    print('INDEPENDENT_AUDIT',name,len(values),max_error,flush=True)
    return name,dict(n=len(values),max_abs_iou_error=max_error,threshold_disagreements=disagreement)


def read_table(name):
    with (RESULTS/name).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def aggregate():
    rows = read_table('factorial_contrasts.csv')
    metrics = read_table('endpoints.csv')
    dirs = []
    for source,target in ORDER:
        name = f'{source}_to_{target}'
        data = read_table(f'rows_{name}.csv')
        assert len(data)==len(ordinary_rows(target))
        x = np.asarray([[float(r[a]) for a in ARMS] for r in data])
        v = metric(x)
        dirs.append(v.mean(0))
        for i,arm in enumerate(ARMS):
            r = next(r for r in metrics if r['scope']==name and r['cohort']=='all' and r['arm']==arm)
            np.testing.assert_allclose([float(r[m]) for m in METRICS],v[:,i].mean(0)*100,rtol=0.,atol=1e-10)
        for label,coef in CONTRASTS.items():
            r = next(r for r in rows if r['scope']==name and r['cohort']=='all' and r['contrast']==label)
            d = (v*np.asarray(coef)[None,:,None]).sum(axis=1)
            for i,m in enumerate(METRICS):
                assert abs(float(r[m+'_pp'])-100*d[:,i].mean())<1e-10
                assert int(r[m+'_up'])==int((d[:,i]>1e-10).sum())
                assert int(r[m+'_down'])==int((d[:,i]< -1e-10).sum())
    macro = np.asarray(dirs).mean(0)*100.
    for label,coef in CONTRASTS.items():
        r = next(r for r in rows if r['scope']=='six_equal' and r['cohort']=='all' and r['contrast']==label)
        np.testing.assert_allclose(np.asarray(coef)@macro,[float(r[m+'_pp']) for m in METRICS],rtol=0.,atol=1e-10)
    rows_sensitivity = read_table('reference_diagnostics.csv')
    source = [read_table(f'rows_{s}_to_{t}.csv') for s,t in ORDER]
    flat = [r for group in source for r in group]
    for cut in (.25,.50):
        rec = next(r for r in rows_sensitivity if r['cohort']=='all' and float(r['threshold'])==cut)
        miss = [r for r in flat if float(r['joint'])<=cut<float(r['best_subject_slot_iou'])]
        assert len(miss)==int(rec['missed'])
        assert sum(float(r['best_reference_column_iou'])>cut for r in miss)==int(rec['switch_column_rescues'])
        assert sum(float(r['joint'])<=cut<float(r['zero_pair_iou']) for r in flat)==int(rec['without_pair_readout_rescues'])
        assert sum(float(r['zero_pair_iou'])<=cut<float(r['joint']) for r in flat)==int(rec['without_pair_readout_harms'])
    return {'paired_public_outcomes_verified':len(flat),'contrast_counts_reconciled':True,
            'both_reference_cutoff_denominators_reconciled':True}


def main():
    verify_seal()
    scoring = read(PRIVATE/'SCORING_SEAL.json')
    assert scoring['predictions']==sha(PREDICTIONS/'PREDICTION_SEAL.json')
    for f,h in scoring['public_scores'].items():
        assert sha(RESULTS/f)==h
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        directions = dict(pool.map(verify_direction,ORDER))
    result = dict(n=sum(v['n'] for v in directions.values()),directions=directions,results=aggregate(),
                  scoring_sha256=sha(PRIVATE/'SCORING_SEAL.json'),FINAL_access=False)
    assert result['n']==18620
    json_atomic(RESULTS/'independent_audit.json',result)
    state('publication_pending','Completed, sealed and independently audited all ordinary records.')
    print('AUDITED',result['n'],flush=True)


if __name__=='__main__':
    main()

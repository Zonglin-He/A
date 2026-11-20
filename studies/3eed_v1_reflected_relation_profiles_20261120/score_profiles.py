"""Post-seal scoring on the study's original inputs and independently verified natural identities."""
import argparse
import csv
import gzip
from pathlib import Path
import numpy as np

from observe import CONFIG, HERE, PARENT, RESULTS, WORK, digest, read, save
from predict_profiles import DECISIONS, SIGNATURE
from profiles import ARMS
from score_roles import (PANELS, SOURCES, anon, inventory, many_iou, normalize, owners, score_boxes,
                         sensor_frame_box, sensor_truth, state)
from diagnose_roles import COHORTS, historical_endpoints, historical_records


def sealed_inputs():
    signature=read(SIGNATURE)
    trace=read(WORK/'SEALED.json')
    assert signature['observation_seal']==digest(WORK/'SEALED.json')
    assert not signature['labels_used'] and not signature['FINAL_access']
    assert signature['protocol']==digest(CONFIG) and signature['arms']==list(ARMS)
    assert all(digest(HERE/f)==h for f,h in signature['code'].items())
    assert len(signature['files'])==501 and signature['files'].keys()==trace['files'].keys()
    for name,value in signature['files'].items():
        assert digest(DECISIONS/name)==value and digest(WORK/name)==trace['files'][name]
    return signature


def score_entry(panel,case,entry,source,truth,gt_sensor,endpoints,reasons):
    target=entry['row']['platform']
    view=int(entry['row']['suffix'] or 0)
    name=f'{panel}_{case:03d}_{source}.npz'
    with np.load(WORK/name,allow_pickle=False) as trace, np.load(DECISIONS/name,allow_pickle=False) as decision:
        result=decision['boxes']
        raw=decision['raw']
        native=np.vstack((result,raw))
        fitted,e1=score_boxes(native,truth[0])
        transformed=np.stack([sensor_frame_box(b,target,view) for b in native])
        sensor,e2=score_boxes(transformed,gt_sensor)
        tol=float(np.max(abs(fitted-sensor)))
        np.testing.assert_allclose(fitted,sensor,rtol=0.,atol=1.e-8)
        for t in (.25,.50):
            np.testing.assert_array_equal(fitted>t,sensor>t)
        z,initial=np.split(fitted,2)
        raw_owner=owners(raw,truth)
        ref_owner=owners(decision['refs'],truth)
        output_owner=owners(result,truth)
        views=len(trace['subject'])
        anchors=trace['subject'][:2].reshape(-1,7)
        refs=trace['reference'][:2].reshape(-1,7)
        sa=owners(anchors,truth)
        ra=owners(refs,truth) if trace['original_ref_valid'] else []
        native_iou=many_iou(anchors,truth[0])
        best=np.array([[float(many_iou(a,truth[0]).max()) for a in trace[key]] for key in ('pool','axis')])
        original=float(best[:,0].max())
        trial=float(best.max())
        assert views==(4 if trace['active'] else 1)
        meta=dict(panel=panel,group=('own_proxy' if source==target else 'cross_proxy') if panel=='source_proxy' else panel,
            source=source,target=target,case=anon(f'{panel}|{case}'),
            recording=anon(str(Path(entry['row']['metapath']).resolve().parent.parent)),
            active=int(trace['active']),subject_judgment=reasons[0],reference_judgment=reasons[1],
            prior_iou=float(z[0]),prior_subject=state(raw_owner[0],0),prior_human=state(raw_owner[0],endpoints[0]),
            best_original_pca=best[0,0],best_original_union=original,best_all_union=trial,
            best_original_four=float(native_iou[:4].max()),best_coherent=float(native_iou.max()),
            subject_available=('NA' if endpoints[0] is None else int(endpoints[0] in sa)),
            ref_available=('NA' if endpoints[1] is None else int(endpoints[1] in ra)))
        rows=[]
        for idx,arm in enumerate(ARMS):
            o=raw_owner[idx]
            candidate_idx=int(decision['choice'][idx,0])
            assert np.array_equal(raw[idx],anchors[candidate_idx])
            if not trace['active']:
                np.testing.assert_array_equal(result[idx],raw[0])
            rows.append(dict(**meta,arm=arm,iou=float(z[idx]),raw_iou=float(initial[idx]),
                switched=int(not np.array_equal(raw[idx],raw[0])),
                shape_modified=int(not np.allclose(raw[idx],result[idx],rtol=0.,atol=1e-8)),
                raw_subject=state(o,0),subject=state(output_owner[idx],0),
                human_raw=state(o,endpoints[0]),human_subject=state(output_owner[idx],endpoints[0]),
                reference=state(ref_owner[idx],endpoints[1]),source_mirror_candidate=int(candidate_idx>=4),
                new_physical_owner=int(o!=-1 and raw_owner[0]!=-1 and o!=raw_owner[0])))
    return rows,max(e1,e2,tol)


def run(archive):
    signature=sealed_inputs()
    history=historical_records(archive)
    with gzip.open(PARENT/'results/all_rows.csv.gz','rt') as f:
        old={(r['group'],r['source'],r['case']):float(r['iou']) for r in csv.DictReader(f) if r['arm']=='v1'}
    with gzip.open(PARENT/'results/diagnostic_rows.csv.gz','rt') as f:
        ids={(r['group'],r['source'],r['case']):(r['v1_s'],r['v1_r']) for r in csv.DictReader(f)}
    assert len(old)==len(ids)==501
    rows=[]
    worst=0.
    for panel,path in PANELS.items():
        for case,entry in enumerate(read(path)):
            row=entry['row']
            target=row['platform']
            view=int(row['suffix'] or 0)
            truth,caption=inventory(row['metapath'],target,view,int(row['object_index']))
            assert caption==normalize(row['caption'])
            gt_sensor=sensor_truth(target,str(Path(row['metapath']).resolve()),row['caption'],row['suffix'])
            assert abs(many_iou(sensor_frame_box(truth[0],target,view)[None],gt_sensor)[0]-1)<1e-7
            if panel=='source_proxy':
                endpoints,reasons=(0,int(entry['ref_id'])),('official','inventory_proxy')
            else:
                endpoints,reasons=historical_endpoints(COHORTS[panel],entry,truth,history)
            for source in SOURCES:
                if source==target and panel!='source_proxy':
                    continue
                cases,err=score_entry(panel,case,entry,source,truth,gt_sensor,endpoints,reasons)
                baseline=cases[0]
                key=baseline['group'],source,baseline['case']
                assert abs(baseline['iou']-old.pop(key))<1.e-10
                prior_s,prior_r=ids.pop(key)
                assert baseline['human_raw']==prior_s
                # Native no-reference rows have no prediction for that endpoint.
                if baseline['reference']!=prior_r:
                    assert baseline['reference'] in ('U','NA') and baseline['reference_judgment']=='inventory_proxy' and not baseline['active']
                rows.extend(cases)
                worst=max(worst,err)
    assert not old and not ids and len(rows)==501*len(ARMS)
    with gzip.open(RESULTS/'all_rows.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    save(RESULTS/'scoring.json',dict(n=501,arms=list(ARMS),rows=len(rows),active=signature['active'],
          observed_rows_sha256=digest(WORK/'SEALED.json'),decision_sha256=digest(SIGNATURE),
          protocol_sha256=digest(CONFIG),max_abs_independent_iou_disagreement=worst,
          max_inverse_point_error=signature['max_inverse_point_error'],matched_published_v1=501,
          labels_accessed_after_seal=True,updates=0,FINAL_access=False))
    print('Scored',len(rows),'arm rows. Max independent IoU disagreement',worst,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Existing local checkout of the original signed judgments')
    run(parser.parse_args().archive)


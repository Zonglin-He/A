"""Independent scalar recount from the two anonymized source-only case streams."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


OUT=Path(__file__).resolve().parent/'results'


def load(name):
    with gzip.open(OUT/name,'rt') as f:
        return list(csv.DictReader(f))


def run():
    rows=load('source_all_rows.csv.gz')
    boxes=load('identity_conditioned_rows.csv.gz')
    assert len({(r['case'],r['source'],r['arm']) for r in rows})==len(rows)==666
    assert len({(r['case'],r['source']) for r in boxes})==len(boxes)==111
    with (OUT/'task.csv').open() as f:
        for metric in csv.DictReader(f):
            eligible=[r for r in rows if r['arm']==metric['arm'] and metric['source'] in ('macro',r['source'])]
            assert len(eligible)==int(metric['n'])
            for threshold in (25,50):
                groups=defaultdict(list)
                pairs=[]
                for r in eligible:
                    pair=float(r['baseline_iou'])>threshold/100,float(r['iou'])>threshold/100
                    groups[r['source']].append(pair[1])
                    pairs.append(pair)
                assert abs(float(metric['acc'+str(threshold)])-100*mean(mean(v) for v in groups.values()))<1e-9
                assert int(metric['rescue'+str(threshold)])==pairs.count((False,True))
                assert int(metric['harm'+str(threshold)])==pairs.count((True,False))
    with (OUT/'physical.csv').open() as f:
        for metric in csv.DictReader(f):
            part=[r for r in boxes if metric['source'] in ('pooled',r['source'])]
            fixed=[r for r in part if r['eligible']=='1']
            assert len(part)==int(metric['n']) and len(fixed)==int(metric['eligible'])
            assert len({r['recording'] for r in fixed})==int(metric['recordings'])
            if not fixed:
                continue
            for r in fixed:
                h=float(r['ref_displacement_m'])
                a,b,c=[float(r[k]) for k in ('raw_center_m2','shift_center_m2','opposite_center_m2')]
                assert abs(b+c-2*a-2*h*h)<1e-8
                assert abs(a-b-float(r['true_center_gain_m2']))<1e-8
            for name,old in [('mean_raw_center_m2','raw_center_m2'),('mean_shift_center_m2','shift_center_m2'),
                             ('mean_opposite_center_m2','opposite_center_m2'),('mean_raw_iou','raw_iou'),('mean_shift_iou','shift_iou')]:
                assert abs(float(metric[name])-mean(float(r[old]) for r in fixed))<1e-9
            pairs=[(float(r['raw_iou'])>.5,float(r['shift_iou'])>.5) for r in fixed]
            assert int(metric['rescue50'])==pairs.count((False,True)) and int(metric['harm50'])==pairs.count((True,False))
    eligible=[r for r in boxes if r['eligible']=='1']
    recordings={r['recording'] for r in eligible}
    delete_one=[mean(float(r['true_center_gain_m2']) for r in eligible if r['recording']!=key)
                for key in sorted(recordings) if len(recordings)>1]
    projected=None
    if eligible:
        energy=mean(float(r['ref_displacement_m'])**2 for r in eligible)
        inner=.5*(energy+mean(float(r['true_center_gain_m2']) for r in eligible))
        original=mean(float(r['raw_center_m2']) for r in eligible)
        if energy>0 and original>0:
            projected=dict(best_single_coefficient=inner/energy,largest_center_m2_reduction=inner**2/energy,
                           fraction_of_raw_center_m2=inner**2/energy/original,posthoc_only=True)
    result=dict(all_prediction_rows=len(rows),all_source_observations=len(boxes),eligible_identity_tuples=len(eligible),
                eligible_recordings=len(recordings),leave_one_recording_out_mean_center_gain_m2=[min(delete_one),max(delete_one)] if delete_one else None,
                checked_quadratic_sign_identity=True,paired_recount=True,empirical_constant_scale_ceiling=projected,
                hashes={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('source_all_rows.csv.gz','identity_conditioned_rows.csv.gz')})
    (OUT/'recount.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    run()

"""Independently rebuild counts from the anonymized published results using the standard library."""
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean


OUT = Path(__file__).resolve().parent/'results'


def recount_diagnostics(keyed):
    def rows_from(name):
        with gzip.open(OUT/name,'rt') as f:
            return list(csv.DictReader(f))
    diagnostics=rows_from('diagnostic_rows.csv.gz')
    paths=rows_from('shape_path_rows.csv.gz')
    assert len(diagnostics)==len(paths)==501
    arms={key[4] for key in keyed}
    for sample in (diagnostics,paths):
        keys=[(r['group'],r['source'],r['target'],r['case'],'ordered_full') for r in sample]
        assert len(set(keys))==501 and all(k in keyed for k in keys)
    with (OUT/'historical_endpoints.csv').open() as f:
        for metric in csv.DictReader(f):
            group=metric['group']; arm=metric['arm']
            assert arm in arms
            pairs=[(r[arm+'_s'],r[arm+'_r']) for r in diagnostics if r['group']==group]
            assert int(metric['reference_correct'])==sum(b=='T' for a,b in pairs)
            assert int(metric['reference_known'])==sum(b!='NA' for a,b in pairs)
            assert int(metric['pair_known'])==sum('NA' not in pair for pair in pairs)
            assert int(metric['both_correct'])==pairs.count(('T','T'))
    with (OUT/'opportunities.csv').open() as f:
        for metric in csv.DictReader(f):
            threshold=float(metric['threshold'])/100
            group=[r for r in diagnostics if r['group']==metric['group']]
            counts=[0]*4
            for row in group:
                if float(row['v1_iou'])>threshold:
                    counts[0]+=1
                elif float(row['best_slate'])>threshold:
                    counts[1]+=1
                elif float(row['best_pool'])>threshold:
                    counts[2]+=1
                else:
                    counts[3]+=1
            assert counts==[int(metric[k]) for k in ('baseline_hits','good_slot_missed','good_pool_outside_slate','no_good_original_pool')]
    with (OUT/'shape_path.csv').open() as f:
        for metric in csv.DictReader(f):
            fixed=[r for r in paths if r['group']==metric['group'] and r['both_original_subjects_correct']=='1']
            comparison=metric['comparison']
            pairs=[]
            for row in fixed:
                base=float(row['raw_iou'])
                if comparison=='best_from_three':
                    value=max(base,float(row['half_iou']),float(row['end_iou']))
                else:
                    value=float(row[comparison+'_iou'])
                pairs.append((base,value,float(row['best_all'])))
            assert int(metric['rescue50'])==sum(a<=.5<b for a,b,_ in pairs)
            assert int(metric['harm50'])==sum(b<=.5<a for a,b,_ in pairs)
            assert int(metric['outside_all_original'])==sum(c<=.5<b for _,b,c in pairs)
    midpoints=rows_from('proposal_midpoint_rows.csv.gz')
    assert len(midpoints)==1002 and len({(r['group'],r['source'],r['case'],r['pool']) for r in midpoints})==1002
    with (OUT/'proposal_midpoints.csv').open() as f:
        for metric in csv.DictReader(f):
            part=[r for r in midpoints if r['group']==metric['group'] and r['pool']==metric['pool']]
            fixed=[r for r in part if r['correct_original']=='1']
            assert int(metric['n'])==len(part) and int(metric['correct_original'])==len(fixed)
            for rows,prefix in ((part,'all_'),(fixed,'fixed_')):
                steps=[(float(r['raw_iou']),float(r['direct_iou'])) for r in rows]
                assert int(metric[prefix+'rescue50'])==sum(a<=.5<b for a,b in steps)
                assert int(metric[prefix+'harm50'])==sum(b<=.5<a for a,b in steps)
            for method,field in (('direct_outside_all','direct_iou'),('oracle_midpoint_outside_all','best_same_target_midpoint')):
                assert int(metric[method])==sum(float(r['best_all_original'])<=.5<float(r[field]) for r in fixed)


def run():
    with gzip.open(OUT/'all_rows.csv.gz','rt') as f:
        rows=list(csv.DictReader(f))
    keys=[(r['group'],r['source'],r['target'],r['case'],r['arm']) for r in rows]
    assert len(keys)==len(set(keys))==4008
    keyed=dict(zip(keys,rows))
    groups=defaultdict(list)
    for r in rows:
        key=(r['group'],r['arm'])
        groups[(*key,'macro')].append(r)
        groups[(*key,r['source']+'_to_'+r['target'])].append(r)
    with (OUT/'metrics.csv').open() as f:
        for result in csv.DictReader(f):
            chunk=groups[result['group'],result['arm'],result['direction']]
            assert len(chunk)==int(result['n'])
            by_direction=defaultdict(list)
            for r in chunk:
                by_direction[r['source'],r['target']].append(float(r['iou']))
            for limit,k in ((.25,'acc25'),(.5,'acc50'),(None,'miou')):
                calc=100*mean(mean((x>limit if limit else x) for x in values) for values in by_direction.values())
                assert abs(calc-float(result[k]))<1e-9
    with (OUT/'paired.csv').open() as f:
        for result in csv.DictReader(f):
            chunk=groups[result['group'],result['arm'],result['direction']]
            for limit,name in ((.25,'25'),(.5,'50')):
                rescue=harm=0
                for r in chunk:
                    baseline=keyed[r['group'],r['source'],r['target'],r['case'],result['control']]
                    hit=float(r['iou'])>limit
                    old=float(baseline['iou'])>limit
                    rescue+=hit and not old
                    harm+=old and not hit
                assert (rescue,harm)==(int(result['rescue'+name]),int(result['harm'+name]))
    with (OUT/'identity.csv').open() as f:
        for result in csv.DictReader(f):
            chunk=groups[result['group'],result['arm'],result['direction']]
            for a in 'TWU':
                for b in 'TWU':
                    assert int(result[a+'_to_'+b])==sum(r['baseline_owner']==a and r['raw_subject']==b for r in chunk)
            assert int(result['both_T'])==sum(r['raw_subject']==r['reference']=='T' for r in chunk)
    with (OUT/'geometry.csv').open() as f:
        for result in csv.DictReader(f):
            chunk=groups[result['group'],result['arm'],result['direction']]
            on_target=[r for r in chunk if r['baseline_owner']==r['raw_subject']=='T']
            assert len(on_target)==int(result['same_verified_subject'])
            for check,k in [('best_a256','verified_rescue_outside_a256'),('best_all','verified_rescue_outside_all'),
                            ('best_all_axis_and_pca','verified_rescue_outside_all_axis_and_pca')]:
                assert sum(float(r[check])<=.5<float(r['iou']) for r in on_target)==int(result[k])
            assert sum(float(r['raw_iou'])<=.5<float(r['iou']) for r in on_target)==int(result['rescue50'])
            assert sum(float(r['iou'])<=.5<float(r['raw_iou']) for r in on_target)==int(result['harm50'])
    recount_diagnostics(keyed)
    receipt=dict(rows=len(rows),paired_accuracy_recount=True,all_identity_transitions=True,
                 same_subject_geometry=True,controls_and_fallbacks_included=True,
                 historical_endpoints=True,four_slot_opportunities=True,posthoc_shape_paths=True,
                 wider_pool_label_free_and_oracle_recount=True,
                 rows_sha256=hashlib.sha256((OUT/'all_rows.csv.gz').read_bytes()).hexdigest())
    (OUT/'recount.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt)


if __name__=='__main__':
    run()

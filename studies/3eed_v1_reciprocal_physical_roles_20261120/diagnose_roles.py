"""Post-seal decision margins and identification from earlier, independently judged boxes."""
import argparse
from collections import Counter
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen

import numpy as np

from score_roles import ARMS, OUT, PANELS, PRIVATE, ROOT, SOURCES, anon, csv_file, digest, inventory, many_iou, owners, read, save, state

sys.path.insert(0,str(ROOT/'research/round162'))
from geometry162 import in_box


REVISION = 'fa61a76d10e31790e2be031a77126814b8010643'
HISTORY = 'studies/3eed_native_reference_joint_assignment_20261118'
COHORTS = {'natural170': 'replay170', 'natural171': 'new_frames'}


def public_file(repo, name):
    path = f'{HISTORY}/{name}'
    if repo is not None:
        return subprocess.check_output(['git', '-C', str(repo), 'show', f'{REVISION}:{path}'])
    with urlopen(f'https://raw.githubusercontent.com/Zonglin-He/A/{REVISION}/{path}', timeout=45) as res:
        return res.read()


def historical_records(repo):
    pins = json.loads(public_file(repo, 'PREDICTION_SEAL.json'))
    assert not pins['FINAL_access']
    pending = {}
    for name in ('part001.csv', 'part002.csv'):
        for row in csv.DictReader(io.StringIO(public_file(repo, 'results/arms/'+name).decode())):
            key = (row['cohort'], int(row['case']), row['source'], row['arm'])
            assert key not in pending
            pending[key] = row
    cases = {}
    for row in csv.DictReader(io.StringIO(public_file(repo, 'results/cases.csv').decode())):
        key = (row['cohort'], int(row['case']), row['source'])
        assert key not in cases
        cases[key] = row
    ranks = {}
    for row in csv.DictReader(io.StringIO(public_file(repo, 'results/reference_flow.csv').decode())):
        key = (row['cohort'], int(row['case']), row['source'], row['variant'])
        assert key not in ranks
        ranks[key] = row['native_rank']
    assert len(cases)==168 and len(pending)==168*9 and len(ranks)==168*2
    return pins['files'], pending, cases, ranks


def old_seed_owners(proposals, seeds, labels):
    """Reproduce the archived unique physical-point ownership rule used for published ranks."""
    points=np.unique(seeds,axis=0)
    inside=np.stack([in_box(points,box) for box in labels],axis=1)
    point_ids=np.where(inside.sum(1)==1,inside.argmax(1),-1)
    result=[]
    for box in proposals:
        hits=point_ids[in_box(points,box)]
        votes=np.bincount(hits+1,minlength=len(labels)+1)
        winner=1+np.argmax(votes[1:])
        count=votes[winner]
        votes[winner]=0
        reliable=len(hits)>=2 and count>=.55*len(hits) and count-votes.max()>=.15*len(hits)
        result.append(int(winner-1) if reliable else -1)
    return np.array(result)


def historical_endpoints(cohort, entry, labels, history):
    """Use previously measured IoUs to identify an annotated box. Current predictions never enter."""
    pins, records, cases, ranks = history
    case, target = entry['case'], entry['target']
    residuals = [[], []]
    positive = [False, False]
    rank_evidence = []
    declared, agrees = None, None
    for source in SOURCES:
        if source == target:
            continue
        key = cohort, case, source
        meta = cases[key]
        assert meta['target'] == target and meta['recording'] == entry['recording']
        statuses = (meta['subject_label']=='verified', meta['reference_label']=='verified')
        agreement = meta['subject_agrees_with_official']
        if declared is not None:
            assert declared==statuses and agrees==agreement
        declared, agrees = statuses, agreement
        tag = f'{cohort}/{case:02d}_{source}.npz'
        path = ROOT/'crossplatform/round171/predictions'/tag
        assert digest(path)==pins[tag]
        with np.load(path, allow_pickle=False) as data:
            for variant in ('C','D'):
                rank=ranks[(*key,variant)]
                if rank!='':
                    assert statuses[1]
                    candidates=old_seed_owners(data[variant+'_natural_boxes'],data['seeds'],labels)
                    rank_evidence.append((int(rank),candidates))
            for name, endpoints in zip(data['arms'], data['method_boxes'], strict=True):
                row = records[(*key, str(name))]
                assert abs(many_iou(endpoints[:1], labels[0])[0]-float(row['official_iou'])) < 1e-8
                for j, field in enumerate(('verified_subject_iou', 'verified_reference_iou')):
                    if row[field]=='' or not np.isfinite(endpoints[j]).all():
                        continue
                    assert statuses[j]
                    observed = float(row[field])
                    candidates = many_iou(labels, endpoints[j])
                    residuals[j].append(np.abs(candidates-observed))
                    positive[j] |= observed>1e-7
    recovered, reasons = [], []
    for j in range(2):
        if j==0 and declared[0] and agrees=='1':
            if residuals[0]:
                assert np.max(np.array(residuals[0])[:,0])<1e-8
            recovered.append(0)
            reasons.append('historical_explicit_official_agreement')
            continue
        rank_ids = {int(candidates[rank-1]) for rank,candidates in rank_evidence if rank>0} if j==1 else set()
        assert -1 not in rank_ids and len(rank_ids)<=1
        if not declared[j] or (not positive[j] and not rank_ids):
            recovered.append(None)
            reasons.append('no_original_verification' if not declared[j] else 'no_positive_measurement')
            continue
        maximum_error = np.max(residuals[j], axis=0) if residuals[j] else np.zeros(len(labels))
        if rank_ids:
            old_id = next(iter(rank_ids))
            assert maximum_error[old_id]<1e-8
            for rank,candidates in rank_evidence:
                matches=np.flatnonzero(candidates==old_id)
                assert rank==(int(matches[0])+1 if len(matches) else 0)
            recovered.append(old_id)
            reasons.append('historical_positive_rank_and_verified_measurements')
            continue
        indices = np.flatnonzero(maximum_error < 1e-8)
        if len(indices)==1 and (len(maximum_error)==1 or np.partition(maximum_error,1)[1]>1e-6):
            recovered.append(int(indices[0]))
            reasons.append('unique_historical_measurements')
        else:
            recovered.append(None)
            reasons.append('ambiguous_historical_measurements')
    if None not in recovered:
        assert recovered[0]!=recovered[1]
    return recovered, reasons


def gap(logits, qualifying, selected):
    return float(logits[qualifying].max()-logits[selected]) if np.any(qualifying) else None


def summary(cases):
    ranks, independent = [], []
    for group in ('own_proxy','cross_proxy','natural170','natural171'):
        rows = [c for c in cases if c['group']==group]
        for threshold in (25,50):
            t = threshold/100
            available = [r for r in rows if r['v1_iou']<=t<r['best_slate']]
            definite = [r for r in rows if r['wrong_with_correct_slot']]
            field = f'gap{threshold}'
            delta = [r[field+'_full']-r[field+'_v1'] for r in available]
            phys_delta = [r['identity_gap_full']-r['identity_gap_v1'] for r in definite]
            strength = [r['weight'] for r in rows if r['weight']>1e-10]
            ranks.append(dict(group=group,n=len(rows),threshold=threshold,baseline_hits=sum(r['v1_iou']>t for r in rows),
                good_slot_missed=len(available),good_pool_outside_slate=sum(r['best_slate']<=t<r['best_pool'] for r in rows),
                no_good_original_pool=sum(r['best_pool']<=t for r in rows),
                median_good_deficit=float(np.median([-r[field+'_v1'] for r in available])) if available else None,
                good_margin_better=sum(v>1e-8 for v in delta),good_margin_worse=sum(v< -1e-8 for v in delta),
                observed_rescues=sum(r['final_iou']>t for r in available),
                wrong_with_correct_slot=len(definite),identity_margin_better=sum(x>1e-8 for x in phys_delta),
                identity_margin_worse=sum(x< -1e-8 for x in phys_delta),
                median_identity_deficit=float(np.median([-r['identity_gap_v1'] for r in definite])) if definite else None,
                shape_positive=len(strength),positive_weight_median=float(np.median(strength)) if strength else None))
        for arm in ARMS:
            roles=[c['endpoints'][arm] for c in rows]
            counts=Counter(st+rf for st,rf in roles)
            both=sum(st!='NA' and rf!='NA' for st,rf in roles)
            independent.append(dict(group=group,arm=arm,n=len(rows),
               reference_known=sum(rf!='NA' for _,rf in roles),reference_correct=sum(rf=='T' for _,rf in roles),
               pair_known=both,both_correct=counts['TT'],
               reference_label_without_positive=sum(c['ref_reason']=='no_positive_measurement' for c in rows),
               reference_no_original_verification=sum(c['ref_reason']=='no_original_verification' for c in rows)))
    csv_file(OUT/'opportunities.csv',ranks)
    csv_file(OUT/'historical_endpoints.csv',independent)


def run(repo):
    history = historical_records(repo)
    seal = read(PRIVATE/'SEAL.json')
    all_cases=[]
    for panel,path in PANELS.items():
        for case,entry in enumerate(read(path)):
            row=entry['row']
            target=row['platform']
            truth,_=inventory(row['metapath'],target,int(row['suffix'] or 0),int(row['object_index']))
            if panel=='source_proxy':
                endpoints,reasons=(0,int(entry['ref_id'])),('official','inventory_proxy')
            else:
                endpoints,reasons=historical_endpoints(COHORTS[panel],entry,truth,history)
            for source in SOURCES:
                if panel!='source_proxy' and target==source:
                    continue
                tag=f'{panel}_{case:03d}_{source}.npz'
                assert digest(PRIVATE/tag)==seal['files'][tag]
                with np.load(PRIVATE/tag,allow_pickle=False) as z:
                    baseline=int(z['choices'][0])
                    valid = bool(z['active'])
                    logits = np.log(z['scores']) if valid else z['scores']
                    slot_iou = many_iou(z['slate'],truth[0])
                    slot_owners=np.array(owners(z['slate'],truth))
                    ref_owners=owners(z['refs'],truth)
                    r=dict(group=('own_proxy' if source==target else 'cross_proxy') if panel=='source_proxy' else panel,
                        source=source,target=target,case=anon(f'{panel}|{case}'),active=int(valid),
                        best_pool=float(many_iou(z['pool'],truth[0]).max()),best_slate=float(slot_iou.max()),
                        v1_iou=float(slot_iou[baseline]),final_iou=float(many_iou(z['boxes'][-1:],truth[0])[0]),
                        weight=float(z['fraction'][-1]),wrong_with_correct_slot=int(slot_owners[baseline]>0 and np.any(slot_owners==0)),
                        ref_reason=reasons[1],subject_reason=reasons[0])
                    for nm,score in [('v1',logits[0]),('full',logits[-1])]:
                        for thresh in (25,50):
                            r[f'gap{thresh}_{nm}']=gap(score,slot_iou>thresh/100,baseline)
                        r['identity_gap_'+nm]=gap(score,slot_owners==0,baseline)
                    r['endpoints']={arm:(state(slot_owners[z['choices'][i]],endpoints[0]),state(ref_owners[i],endpoints[1]))
                                    for i,arm in enumerate(ARMS)}
                    all_cases.append(r)
    assert len(all_cases)==501
    summary(all_cases)
    with gzip.open(OUT/'diagnostic_rows.csv.gz','wt',newline='') as f:
        writer=None
        for r in all_cases:
            fields={k:v for k,v in r.items() if k!='endpoints'}
            for arm,(s,t) in r['endpoints'].items():
                fields[arm+'_s']=s
                fields[arm+'_r']=t
            if writer is None:
                writer=csv.DictWriter(f,fieldnames=fields)
                writer.writeheader()
            writer.writerow(fields)
    save(OUT/'historical_audit.json',dict(prior_commit=REVISION,old_prediction_hashes_checked=len(history[0]),
        old_arm_rows=len(history[1]),current_prediction_files_verified=len(all_cases),uses_current_predictions_for_label_recovery=False,
        requires_positive_historical_iou_or_rank_for_reference=True,original_predictions_unchanged=True,
        results_sha256=digest(OUT/'diagnostic_rows.csv.gz')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,help='Optional local Git checkout of the published studies')
    run(parser.parse_args().archive)

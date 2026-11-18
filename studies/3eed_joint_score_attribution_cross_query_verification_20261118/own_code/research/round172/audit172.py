"""Check the physical links, exact score attribution and offline oracle independently."""
from itertools import product
from math import fsum

import numpy as np

from scope172 import ARMS, HERE, OUT, PLATFORMS, R171, R171_WORK, SPLITS, WORK, atomic_json, lock, read, sha
from common160 import array_sha
from evidence172 import support, verify_evidence
from labels166 import physical_labels
from predict165 import points_for
from predict171 import read_npz
from score165 import many_iou, sensor_values, table
from score170 import label
from score171 import candidate_owners, select_manifest


STYLES = ('D_only', 'C_D', 'shifted_C_D')
TOL = 1e-10


def count_quality(cmask, dmask):
    """Independent two-sided comparison of unique reciprocal maxima."""
    out = np.zeros(len(dmask), np.int16)
    score = np.zeros(len(dmask))
    if len(cmask) and len(dmask):
        hits = cmask.astype(np.float32) @ dmask.astype(np.float32).T
        union = cmask.sum(1)[:, None] + dmask.sum(1)[None, :] - hits
        value = np.divide(hits, union, out=np.zeros_like(union), where=(union > 0))
        for j in range(len(dmask)):
            col = value[:, j]
            i = int(col.argmax())
            v = float(value[i,j])
            if hits[i,j] >= 3 and v >= .25 and (col == v).sum() == 1 and v == value[i].max() and (value[i]==v).sum()==1:
                out[j] = i+1
                score[j] = v
    return out, score


def check_row(row, truth, d, e, record, identity, points):
    names, gt, seeds, owners = physical_labels(row,d['seeds'])
    s, r = truth
    subject_ids = candidate_owners(d['subject_boxes'], seeds, owners, len(names))
    ref_ids = candidate_owners(d['D_natural_boxes'], seeds, owners, len(names))
    c_ids = candidate_owners(d['C_natural_boxes'], seeds, owners, len(names))
    rank, overlap = count_quality(support(d['C_natural_boxes'],points), support(d['D_natural_boxes'],points))
    np.testing.assert_array_equal(rank,e['c_rank'])
    np.testing.assert_array_equal(overlap,e['overlap'])
    known = [(int(c_ids[rank[j]-1]), int(ref_ids[j])) for j in np.flatnonzero(rank)
             if c_ids[rank[j]-1]>=0 and ref_ids[j]>=0]
    assert len(known)==int(identity['known_links'])
    assert sum(a==b for a,b in known)==int(identity['same_owner_links'])
    assert sum(a!=b for a,b in known)==int(identity['wrong_owner_links'])
    assert int(identity['links']) == int(np.count_nonzero(rank))
    bonus = np.array([1/int(x) if x else 0. for x in rank])
    original = d['D_native_unary']
    np.testing.assert_array_equal(e['C_D'][:-1], original[:-1]+bonus)
    np.testing.assert_array_equal(e['shifted_C_D'][:-1], original[:-1]+np.roll(bonus,int(e['shift'])))
    assert original[-1] == e['C_D'][-1] == e['shifted_C_D'][-1]
    us, ur, delta = list(map(float,d['subject_unary'])), list(map(float,original)), d['native_left'] * int(d['sign'])
    total = lambda pair: us[pair[0]] + (ur[pair[1]] + float(delta[pair]))
    winner = max(product(range(len(us)),range(len(ur))),key=total)
    assert winner == tuple(d['choices'][ARMS.index('D_native_joint')])
    sc, rc = list(np.flatnonzero(subject_ids==s)), list(np.flatnonzero(ref_ids==r))
    if sc and rc:
        good = max(product(sc,rc),key=total)
        i,j = winner
        a,b = good
        diff = (us[i]-us[a], ur[j]-ur[b], float(delta[i,j]-delta[a,b]), total(winner)-total(good))
        for name,value in zip(('subject','D','left','total'),diff,strict=True):
            assert abs(float(record['winner_minus_correct_'+name])-value)<TOL
    else:
        assert record['winner_minus_correct_total']==''
    v = max(product(range(len(us)),rc),key=total) if rc else winner
    box = d['subject_boxes'][v[0]:v[0]+1]
    expected = sensor_values(box,gt[0],row['platform'],int(row['suffix'] or 0))[0]
    independent_subject = many_iou(box,gt[s])[0]
    error = abs(expected-float(record['oracle_iou']))
    assert error<TOL and abs(independent_subject-float(record['oracle_verified_iou']))<TOL
    assert int(v[1]<len(ref_ids) and ref_ids[v[1]]==r and subject_ids[v[0]]==s)==int(record['oracle_dual'])
    for name in STYLES:
        order = sorted(range(len(ref_ids)),key=lambda i: -float(e[name][i]))
        expected_rank = next((k+1 for k,i in enumerate(order) if ref_ids[i]==r),0)
        assert int(identity[name+'_rank'])==expected_rank
        peak = {}
        for i in order:
            if ref_ids[i]>=0:
                peak.setdefault(ref_ids[i],float(e[name][i]))
        others = [val for obj,val in peak.items() if obj!=r]
        if r in peak and others:
            x = [peak[r]-v for v in others]
            acc = fsum((.5 if abs(v)<=TOL else float(v>0.)) for v in x)/len(x)
            assert abs(acc-float(identity[name+'_concordance']))<TOL
        else:
            assert identity[name+'_concordance']==''
    return error


def audit():
    design=lock('audit',('evidence172.py','geometry172.py','analyze172.py','audit172.py'))
    evidence = verify_evidence()
    analysis = read(HERE/'ANALYSIS_SEAL.json')
    assert analysis['evidence_seal']==sha(HERE/'EVIDENCE_SEAL.json') and not analysis['FINAL_access']
    assert analysis['lock']==sha(HERE/'ATTRIBUTION_LOCK.json')
    assert all(sha(OUT/n)==v for n,v in analysis['files'].items())
    assert sha(HERE/'IDENTITY_GATE.json')==analysis['identity_gate']
    keys = lambda r: (r['cohort'],int(r['case']),r['source'])
    q = {keys(r):r for r in table(OUT/'pair_attribution.csv')}
    i = {keys(r):r for r in table(OUT/'identity_rows.csv')}
    assert q.keys()==i.keys() and len(q)==60
    checked = {}
    for cohort, rows in select_manifest().items():
        for entry, blind, lookup in rows:
            truth = label(lookup,blind['subject']),label(lookup,blind['reference'])
            if min(truth)<0:
                continue
            row, case = entry['row'],entry['case']
            xyz, indices = points_for(row)
            values = np.unique(xyz,axis=0)
            for source in PLATFORMS:
                if source==row['platform']:
                    continue
                k = cohort,case,source
                f = f'{cohort}/{case:02d}_{source}.npz'
                d = read_npz(R171_WORK/'predictions'/f)
                e = read_npz(WORK/'evidence_PRIVATE'/f)
                assert str(e['design'])==evidence['design'] and e['sample_sha']==array_sha(indices) and e['point_sha']==array_sha(xyz)
                checked[k] = check_row(row,truth,d,e,q[k],i[k],values)
    assert checked.keys()==q.keys()
    summary = table(OUT/'identity_summary.csv')
    styles = {}
    for row in summary:
        portion = [r for r in i.values() if row['cohort']=='both_panels' or r['cohort']==row['cohort']]
        assert len(portion)==int(row['n'])
        a = row['method']
        assert sum(int(v[a+'_top1']) for v in portion)==int(row['top1'])
        assert abs(fsum(float(v[a+'_rr']) for v in portion)/len(portion)-float(row['mrr']))<TOL
        b = [float(v[a+'_concordance']) for v in portion if v[a+'_concordance']!='']
        assert len(b)==int(row['concordance_n'])
        assert abs(fsum(b)/len(b)-float(row['concordance']))<TOL
        styles[row['cohort'],a] = fsum(b)/len(b),int(row['top1'])
    passed = True
    for co in SPLITS:
        pos = styles[co,'C_D']
        passed &= all(pos[0]>styles[co,h][0]+TOL and pos[1]>=styles[co,h][1] for h in ('D_only','shifted_C_D'))
    assert passed==read(HERE/'IDENTITY_GATE.json')['independent_readout_gate_pass']
    proof = dict(design=design, identity_and_pair_rows_recomputed=len(checked),
                 max_oracle_sensor_iou_error=max(checked.values()), gate_rederived=True, matched_summary_rows=len(summary),
                 source_and_sampling_hash_verified=True, sealed_artifacts=analysis['files'], FINAL_access=False)
    atomic_json(HERE/'SCALAR_AUDIT.json',proof)
    print(proof,flush=True)
    return proof


if __name__=='__main__':
    audit()

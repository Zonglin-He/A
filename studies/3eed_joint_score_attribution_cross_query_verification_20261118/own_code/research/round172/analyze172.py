"""Post-seal attribution, correct-reference oracle and identity discrimination."""
from itertools import chain
from math import fsum
from statistics import median

import numpy as np

from scope172 import ARMS, HERE, OUT, PLATFORMS, R171, R171_WORK, SPLITS, WORK, atomic_json, lock, read, sha, status
from assignment171 import assign
from evidence172 import verify_evidence
from geometry172 import TOL, one_seed, recover
from labels166 import physical_labels, RIGHT
from predict171 import read_npz
from score165 import many_iou, table, write_table
from score170 import label, status_for
from score171 import candidate_owners, select_manifest


D_JOINT = ARMS.index('D_native_joint')
D_LINKED = ARMS.index('D_native_linked_joint')
SCORERS = ('D_only', 'C_D', 'shifted_C_D')


def best_correct_pair(us, ur, rel, subject, reference):
    if not len(subject) or not len(reference):
        return None
    score = us[subject, None] + (ur[None, reference] + rel[np.ix_(subject, reference)])
    a, b = np.unravel_index(score.argmax(), score.shape)
    return int(subject[a]), int(reference[b])


def restrict_reference(us, ur, left, sign, correct):
    u = ur[correct]
    # The labeled restriction excludes every other candidate, including unknown.
    # A dominated placeholder provides the archived assign() interface.
    ext = np.c_[left[:, correct], np.zeros(len(us))]
    i, k = assign(us, np.r_[u, u.min()-2.], ext, sign, 'joint')
    assert k < len(correct)
    j = correct[k]
    enumeration = us[:, None] + (u[None, :] + sign*left[:, correct])
    assert (i, k) == np.unravel_index(enumeration.argmax(), enumeration.shape)
    return i, j


def ordered_rank(unary, correct):
    if not len(correct):
        return 0
    ordering = np.argsort(-unary, kind='stable')
    return int(np.flatnonzero(np.isin(ordering, correct))[0]+1)


def rank_identity(cand, gold, unary):
    """Group physically verified equivalent boxes only for the within-row discrimination statistic."""
    correct = np.flatnonzero(cand == gold)
    r = ordered_rank(unary, correct)
    peak = {k: unary[cand == k].max() for k in np.unique(cand) if k >= 0}
    wrong = [v for k, v in peak.items() if k != gold]
    stat = ''
    if gold in peak and wrong:
        delta = peak[gold] - np.array(wrong)
        stat = float((np.count_nonzero(delta > TOL) + .5*np.count_nonzero(np.abs(delta) <= TOL))/len(wrong))
    return r, int(r == 1), (1/r if r else 0.), stat


def verification(meta, d, evidence, d_ids, c_ids, ref):
    rank, strength = evidence['c_rank'], evidence['overlap']
    assert len(rank) == len(d_ids) and np.array_equal(d['D_native_unary'], evidence['D_only'])
    assert len(c_ids) == len(d['C_natural_boxes'])
    in_both = np.flatnonzero(rank)
    assert len(np.unique(rank[in_both])) == len(in_both) and (strength[in_both] >= .25).all()
    c_link = c_ids[rank[in_both] - 1]
    d_link = d_ids[in_both]
    known = (c_link >= 0) & (d_link >= 0)
    row = dict(**meta, native_c_count=len(c_ids), native_d_count=len(d_ids),
               links=len(in_both), known_links=int(known.sum()), same_owner_links=int(((c_link==d_link)&known).sum()),
               wrong_owner_links=int(((c_link!=d_link)&known).sum()),
               both_c_and_d_find_correct=int((c_ids==ref).any() and (d_ids==ref).any()),
               c_and_d_correct_link=int(((c_link==ref)&(d_link==ref)).any()),
               known_d_instances=len({int(x) for x in d_ids if x >= 0}),
               d_correct_with_c_support=int((d_link == ref).any()),
               effective_shift=int(not np.array_equal(evidence['c_to_d_bonus'], evidence['shifted_bonus'])))
    for method in SCORERS:
        result = rank_identity(d_ids, ref, evidence[method][:-1])
        row.update({method+'_'+k: v for k, v in zip(('rank', 'top1', 'rr', 'concordance'), result, strict=True)})
    return row


def diagnose(meta, d, gt, s_id, r_id, sub, ref, old_arm):
    us, ur, left = d['subject_unary'], d['D_native_unary'], d['native_left']
    sign = int(d['sign'])
    rel = sign * left
    winning = tuple(map(int, d['choices'][D_JOINT]))
    assert winning == assign(us, ur, left, sign, 'joint')
    n = len(r_id)
    sw, rw = winning
    s_owner, r_owner = s_id[sw], (r_id[rw] if rw < n else -1)
    correct_s, correct_r = np.flatnonzero(s_id == sub), np.flatnonzero(r_id == ref)
    both = s_owner == sub and r_owner == ref
    fitted = best_correct_pair(us, ur, rel, correct_s, correct_r)
    win_ref_status = status_for(ref, r_owner, rw < n)
    win_sub_status = status_for(sub, s_owner)
    assert str(win_ref_status) == old_arm['reference_identity']
    assert str(win_sub_status) == old_arm['subject_identity'] and int(both) == int(old_arm['dual_correct'])
    archived_iou = float(old_arm['official_iou'])
    verified_iou = float(old_arm['verified_subject_iou'])
    r158, r158_verified = many_iou(d['method_boxes'][:1, 0], gt[0])[0], many_iou(d['method_boxes'][:1, 0], gt[sub])[0]
    assert abs(many_iou(d['method_boxes'][D_JOINT:D_JOINT+1, 0], gt[0])[0] - archived_iou) < TOL
    assert abs(many_iou(d['method_boxes'][D_JOINT:D_JOINT+1, 0], gt[sub])[0] - verified_iou) < TOL
    recovered = recover(d)
    subject_sets, _ = recovered
    null_margin = float(np.min((ur[:-1][None, :] + rel[:, :-1]).max(1) - ur[-1])) if n else ''
    if n:
        assert rw < n and null_margin >= 1. - TOL
    own_status = ('both_available' if fitted is not None else
                  'missing_both' if not len(correct_s) and not len(correct_r) else
                  'missing_subject' if not len(correct_s) else 'missing_reference')
    subject_rank = ordered_rank(us, correct_s)
    reference_rank = ordered_rank(ur[:-1], correct_r)
    values = dict(**meta, coverage=own_status,
                  subject_candidates=len(s_id), reference_candidates=n,
                  winner_subject_identity=win_sub_status, winner_reference_identity=win_ref_status, winner_dual=int(both),
                  best_correct_subject_rank=subject_rank, best_correct_D_rank=reference_rank,
                  selected_D_rank=rw+1 if rw < n else '', selected_D_unary=float(ur[rw]),
                  selected_subject_unary=float(us[sw]), selected_signed_left=float(rel[winning]),
                  null_margin_min_over_subjects=null_margin,
                  winner_subject_visible=int(d['subject_stats'][sw, 1]),
                  winner_reference_visible=int(d['native_stats'][rw, 1]) if rw < n else '',
                  winner_reference_width=float(d['native_stats'][rw, 2]) if rw < n else '',
                  r171_joint_iou=archived_iou, r171_joint_verified_iou=verified_iou,
                  r158_iou=float(r158), r158_verified_iou=float(r158_verified),
                  d_native_linked_iou=float(many_iou(d['method_boxes'][D_LINKED:D_LINKED+1, 0], gt[0])[0]),
                  ambiguous_mask_variants=sum(len(x)>1 for x in subject_sets))
    if fitted is not None:
        sc, rc = fitted
        deltas = float(us[sw]-us[sc]), float(ur[rw]-ur[rc]), float(rel[winning]-rel[fitted])
        margin = sum(deltas)
        assert margin >= -TOL
        assert abs(margin - ((us[sw]+(ur[rw]+rel[winning]))-(us[sc]+(ur[rc]+rel[fitted])))) < TOL
        assert (abs(margin) <= TOL) if both else True
        source_name = max(zip(('subject', 'D_reference', 'left_right'), deltas), key=lambda k: k[1])[0]
        alternatives = (
            (np.zeros_like(us), ur, left),
            (us, np.r_[np.zeros(n), ur[-1]], left),
            (us, ur, np.zeros_like(left)),
        )
        flips = [assign(*terms, sign, 'joint') for terms in alternatives]
        rescued = [int(a in correct_s and b in correct_r) for a,b in flips]
        extra = dict(best_pair_D_rank=rc+1, best_pair_subject_visible=int(d['subject_stats'][sc,1]),
                     best_pair_reference_visible=int(d['native_stats'][rc,1]),
                     best_pair_reference_width=float(d['native_stats'][rc,2]),
                     best_pair_signed_left=float(rel[fitted]), winner_minus_correct_subject=deltas[0],
                     winner_minus_correct_D=deltas[1], winner_minus_correct_left=deltas[2],
                     winner_minus_correct_total=margin,
                     largest_positive_component=source_name if margin > TOL else '',
                     both_correct_if_drop_subject=rescued[0], both_correct_if_drop_D=rescued[1],
                     both_correct_if_drop_left=rescued[2])
    else:
        deltas = ('','')
        extra = {k: '' for k in ('best_pair_D_rank', 'best_pair_subject_visible',
            'best_pair_reference_visible', 'best_pair_reference_width', 'best_pair_signed_left',
            'winner_minus_correct_subject', 'winner_minus_correct_D', 'winner_minus_correct_left',
            'winner_minus_correct_total', 'largest_positive_component', 'both_correct_if_drop_subject',
            'both_correct_if_drop_D', 'both_correct_if_drop_left')}
    values.update(extra)
    values.update(one_seed(d, recovered, winning, fitted, sum(deltas) if fitted is not None else 0.))
    if len(correct_r):
        ss, _ = restrict_reference(us, ur, left, sign, correct_r)
        box = d['subject_boxes'][ss:ss+1]
        oracle_iou = float(many_iou(box, gt[0])[0])
        oracle_verified_iou = float(many_iou(box, gt[sub])[0])
        oracle_both = int(s_id[ss] == sub)
    else:
        oracle_iou, oracle_verified_iou, oracle_both = archived_iou, verified_iou, int(both)
    return dict(**values, oracle_reference_available=int(bool(len(correct_r))),
                oracle_iou=oracle_iou, oracle_verified_iou=oracle_verified_iou, oracle_dual=oracle_both)


def average(values):
    return fsum(values)/len(values) if values else ''


def summarize(quality, identity):
    attribution, oracle, verification_summary = [], [], []
    for cohort in ('replay170', 'new_frames', 'both_panels'):
        select = lambda rows: [x for x in rows if cohort == 'both_panels' or x['cohort'] == cohort]
        q = select(quality)
        y = select(identity)
        failures = [r for r in q if r['coverage'] == 'both_available' and r['winner_dual'] == 0]
        metrics = dict(cohort=cohort, n=len(q), correct_refs=sum(r['best_correct_D_rank']>0 for r in q),
                       both_available=sum(r['coverage']=='both_available' for r in q), dual_correct=sum(r['winner_dual'] for r in q),
                       feasible_failures=len(failures), null_impossible_nonempty=sum(r['null_margin_min_over_subjects']!='' for r in q),
                       null_margin_min=min(r['null_margin_min_over_subjects'] for r in q if r['null_margin_min_over_subjects']!=''),
                       positive_winning_left_same_endpoint_failure=sum(r['selected_signed_left']>0 and r['winner_dual']==0 for r in q))
        for name in ('subject', 'D', 'left'):
            k = 'winner_minus_correct_'+name
            values = [r[k] for r in failures]
            metrics.update({name+'_favors_winner': sum(v > TOL for v in values),
                            name+'_favors_correct': sum(v < -TOL for v in values),
                            name+'_mean_margin': average(values),
                            name+'_median_margin': median(values) if values else '',
                            name+'_drop_recovers': sum(r['both_correct_if_drop_'+name] for r in failures)})
        for key, name in (('subject','subject'), ('D_reference','D'), ('left_right','left')):
            metrics['largest_'+name] = sum(r['largest_positive_component'] == key for r in failures)
        metrics.update(failed_correct_ranks_median=median([r['best_correct_D_rank'] for r in failures]) if failures else '',
                       failed_winner_visible_median=median([r['winner_reference_visible'] for r in failures]) if failures else '',
                       failed_correct_visible_median=median([r['best_pair_reference_visible'] for r in failures]) if failures else '',
                       failures_tested_by_seed_drop=sum(r['margin_min']!='' for r in failures),
                       failures_margin_reversed_by_seed_drop=sum(r['single_drop_reverses']==1 for r in failures),
                       failures_margin_closed_by_seed_drop=sum(r['single_drop_closes']==1 for r in failures),
                       failures_winner_factor_sign_changes=sum(r['winner_jackknife_sign_changes']>0 for r in failures),
                       failures_winner_factor_opposite_sign=sum(r['winner_jackknife_opposite_sign']>0 for r in failures))
        attribution.append(metrics)
        for scope, rows in (('all_verified', q), ('correct_reference_present', [r for r in q if r['oracle_reference_available']])):
            for method, field, audited, d_field in (('r158_pca','r158_iou','r158_verified_iou',None),
                ('r171_native_joint','r171_joint_iou','r171_joint_verified_iou','winner_dual'),
                ('oracle_correct_reference','oracle_iou','oracle_verified_iou','oracle_dual')):
                values = [r[field] for r in rows]
                count = len(values)
                b = [r['r171_joint_iou'] for r in rows]
                change = [a-other for a, other in zip(values,b,strict=True)]
                oracle.append(dict(cohort=cohort, scope=scope, method=method, n=count,
                    frames=len({(r['cohort'],r['case']) for r in rows}),
                    mIoU=100*average(values) if count else '', Acc25_count=sum(v>.25 for v in values), Acc50_count=sum(v>.50 for v in values),
                    blind_subject_Acc25_count=sum(r[audited]>.25 for r in rows), blind_subject_Acc50_count=sum(r[audited]>.50 for r in rows),
                    dual_correct=sum(r[d_field] for r in rows) if d_field else '',
                    iou_better=sum(v>TOL for v in change), iou_worse=sum(v<-TOL for v in change),
                    acc25_better=sum(a>.25 and z<=.25 for a,z in zip(values,b,strict=True)),
                    acc25_worse=sum(a<=.25 and z>.25 for a,z in zip(values,b,strict=True)),
                    acc50_better=sum(a>.50 and z<=.50 for a,z in zip(values,b,strict=True)),
                    acc50_worse=sum(a<=.50 and z>.50 for a,z in zip(values,b,strict=True))))
        for method in SCORERS:
            hits = lambda x: x[method+'_top1']
            concordance = [x[method+'_concordance'] for x in y if x[method+'_concordance']!='']
            verification_summary.append(dict(cohort=cohort, method=method, n=len(y), top1=sum(hits(x) for x in y),
                mrr=average([x[method+'_rr'] for x in y]), concordance_n=len(concordance), concordance=average(concordance),
                top1_gains=sum(hits(x)>x['D_only_top1'] for x in y), top1_losses=sum(hits(x)<x['D_only_top1'] for x in y),
                links=sum(r['links'] for r in y), link_owner_known=sum(r['known_links'] for r in y),
                links_agree_on_owner=sum(r['same_owner_links'] for r in y), links_disagree_on_owner=sum(r['wrong_owner_links'] for r in y),
                both_find_correct=sum(r['both_c_and_d_find_correct'] for r in y),
                linked_to_correct_in_both=sum(r['c_and_d_correct_link'] for r in y),
                correct_D_with_C_support=sum(r['d_correct_with_c_support'] for r in y),
                rows_permuted_effectively=sum(r['effective_shift'] for r in y)))
    return attribution, oracle, verification_summary


def gate(summary):
    flags = {}
    for scope in SPLITS:
        rows = {x['method']: x for x in summary if x['cohort'] == scope}
        target, other = rows['C_D'], (rows['D_only'], rows['shifted_C_D'])
        assert all(a['concordance_n']==target['concordance_n'] and a['n']==target['n'] for a in other)
        flags[scope] = dict(evaluable=target['concordance_n'],
            improves_on_both_concordance=bool(target['concordance_n'] and all(target['concordance']>a['concordance']+TOL for a in other)),
            no_top1_reduction_vs_either=all(target['top1']>=a['top1'] for a in other))
    return dict(criterion_predeclared=True, independent_readout_gate_pass=all(
        v['improves_on_both_concordance'] and v['no_top1_reduction_vs_either'] for v in flags.values()),
        strata=flags, FINAL_access=False)


def run():
    pin = lock('attribution', ('evidence172.py','geometry172.py','analyze172.py'))
    seal = verify_evidence()
    status('attribution', evaluated_direction_records=60)
    panels = select_manifest()
    evaluation = read(R171 / 'EVALUATION_AUDIT.json')
    assert sha(R171 / 'results/arms.csv') == evaluation['files']['arms.csv']
    inherited = {(r['cohort'],int(r['case']),r['source']):r
                 for r in table(R171 / 'results/arms.csv') if r['arm']=='D_native_joint'}
    quality, identity = [], []
    physical = set()
    for cohort, rows in panels.items():
        for entry, blind, key in rows:
            case, source_row = entry['case'], entry['row']
            assert case == blind['case'] == key['case']
            sub, ref = label(key, blind['subject']), label(key, blind['reference'])
            if min(sub, ref) < 0:
                continue
            assert sub != ref
            physical.add((cohort,case))
            for source in PLATFORMS:
                if source == entry['target']:
                    continue
                path = f'{cohort}/{case:02d}_{source}.npz'
                d = read_npz(R171_WORK / 'predictions' / path)
                e = read_npz(WORK / 'evidence_PRIVATE' / path)
                assert str(e['design']) == seal['design'] and e['point_sha'] == d['point_sha']
                names, gt, seeds, owners = physical_labels(source_row, d['seeds'])
                assert len(gt) == key['inventory_size'] and max(sub,ref)<len(gt)
                s_id = candidate_owners(d['subject_boxes'],seeds,owners,len(names))
                r_id = candidate_owners(d['D_natural_boxes'],seeds,owners,len(names))
                c_id = candidate_owners(d['C_natural_boxes'],seeds,owners,len(names))
                meta = dict(cohort=cohort, case=case, source=source, target=entry['target'], recording=entry['recording'],
                            rewrite=entry['rewrite']['status'])
                quality.append(diagnose(meta,d,gt,s_id,r_id,sub,ref,inherited[cohort,case,source]))
                identity.append(verification(meta,d,e,r_id,c_id,ref))
    assert len(quality) == len(identity) == 60 and len(physical) == 30
    assert [sum(x['cohort']==k for x in quality) for k in SPLITS] == [28,32]
    attribution, oracle, verification_summary = summarize(quality, identity)
    report = gate(verification_summary)
    OUT.mkdir(exist_ok=True)
    files = {}
    tables = (('pair_attribution',quality), ('identity_rows',identity), ('attribution_summary',attribution),
              ('oracle_summary',oracle), ('identity_summary',verification_summary))
    for name, rows in tables:
        dest = OUT / (name+'.csv')
        write_table(dest, rows)
        files[dest.name] = sha(dest)
    atomic_json(HERE / 'IDENTITY_GATE.json', report)
    proof = dict(lock=pin, evidence_seal=sha(HERE / 'EVIDENCE_SEAL.json'), files=files,
                 identity_gate=sha(HERE / 'IDENTITY_GATE.json'), scored_directions=60, physical_frames=30,
                 recovered_R171_geometry_rows=60, target_truth_scope='post_seal_offline_only', FINAL_access=False)
    atomic_json(HERE / 'ANALYSIS_SEAL.json', proof)
    print('R172 scored', proof['scored_directions'], 'identity gate', report, flush=True)


if __name__ == '__main__':
    run()

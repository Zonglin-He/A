"""Tabulate the fixed local text/image review without releasing original data."""
from collections import Counter

from roles163 import STAGES
from scope163 import HERE, WORK, atomic_json, read, sha, target_rows
from score162 import put_table

PATH = ('native_full', 'r88_region', 'r91_init', 'lcib_init', 'lcib_final')
VERIFIED = ('verified_boxes', 'visual_only')


def proxy(assignment):
    known = sum(owner >= 0 for owner in assignment)
    if known == 0:
        return 'neither_assigned'
    if known == 1:
        return 'one_assigned'
    return 'same_box' if assignment[0] == assignment[1] else 'different_boxes'


def compare_visible(assignment, expected):
    if not expected:
        return 'identity_unverified'
    compared = [int(actual == reference) for actual, reference in zip(assignment, expected) if actual >= 0]
    if 0 in compared:
        return 'contradicts_visible_pair'
    if not compared:
        return 'unknown'
    if len(compared) == 1:
        return 'one_endpoint_matches'
    return 'both_endpoints_match'


def direction(entry, expected):
    if not expected or min(expected) < 0:
        return 'unverified'
    a, b = (entry['boxes'][x]['u'] for x in expected)
    if a is None or b is None or not (0 <= a <= 1 and 0 <= b <= 1) or abs(a - b) < .03:
        return 'partial_or_close_projection'
    return 'agrees' if (entry['parsed'] == 'LEFT') == (a < b) else 'opposes'


def main():
    review = WORK/'ROLE_ADJUDICATION_PRIVATE.json'
    captured = WORK/'ROLE_REVIEW_PRIVATE.json'
    scope = read(HERE/'ROLE_SCOPE.json')
    assert sha(captured) == scope['private_review_sha256']
    rows, judgments = read(captured), read(review)
    manifest = target_rows()
    assert [r['case'] for r in rows] == [r['case'] for r in judgments] == list(range(1, 34))
    all_rows, stage_table, groups = [], [], []
    for entry, decision in zip(rows, judgments):
        case_type = decision['type']
        ref = decision['ref']
        expected = []
        assert decision['note'] and case_type in (*VERIFIED, 'unresolved', 'plural', 'image_anchor', 'endpoint_error')
        if case_type in VERIFIED:
            assert sorted(entry['role_indices'])[0] == 0 and entry['role_indices'][0] != entry['role_indices'][1]
            assert ref == -2 if case_type == 'visual_only' else 0 < ref < len(entry['boxes'])
            expected = [0 if role == 0 else ref for role in entry['role_indices']]
        else:
            assert ref == -1
        stats = {stage: proxy(entry['stages'][stage]) for stage in STAGES}
        known = {stage: compare_visible(entry['stages'][stage], expected) for stage in STAGES}
        assert stats['r91_init'] == 'same_box'
        first_wrong = next((stage for stage in PATH if known[stage] == 'contradicts_visible_pair'), 'unverified')
        first_shared = next((stage for stage in PATH if stats[stage] == 'same_box'), 'unassigned')
        all_rows.append(dict(case=entry['case'], source=entry['source'], target=entry['target'],
                             review=case_type, scope=decision['scope'], coreference=decision['coreference'],
                             left_right_projection=direction(entry, expected),
                             first_shared_proxy=first_shared, first_observable_wrong=first_wrong,
                             **{stage+'_assignment': stats[stage] for stage in STAGES},
                             **{stage+'_visible_identity': known[stage] for stage in STAGES}))
        groups.append(manifest[entry['panel_index']]['point_sha'])
    for group in ('all', 'visually_verified', *VERIFIED):
        selected = [r for r in all_rows if group == 'all' or r['review'] == group
                    or (group == 'visually_verified' and r['review'] in VERIFIED)]
        for stage in STAGES:
            assignments = Counter(r[stage+'_assignment'] for r in selected)
            verified = Counter(r[stage+'_visible_identity'] for r in selected)
            stage_table.append(dict(group=group, stage=stage, n=len(selected),
                                    same_box=assignments['same_box'], different_boxes=assignments['different_boxes'],
                                    one_assigned=assignments['one_assigned'], neither_assigned=assignments['neither_assigned'],
                                    contradicts_visible_pair=verified['contradicts_visible_pair'],
                                    both_endpoints_match=verified['both_endpoints_match'],
                                    one_endpoint_matches=verified['one_endpoint_matches'],
                                    unknown=verified['unknown'], identity_unverified=verified['identity_unverified']))
    n = Counter(r['review'] for r in all_rows)
    verified = [r for r in all_rows if r['review'] in VERIFIED]
    result = dict(edges=len(all_rows), counts=dict(n), observed_visually_distinct=len(verified),
                  text_implies_distinct_singletons=n['verified_boxes']+n['visual_only']+n['unresolved'],
                  visually_verified_physical_point_observations=len({groups[r['case']-1] for r in verified}),
                  frame_scope=dict(Counter(r['scope'] for r in all_rows)),
                  verified_left_right_projection=dict(Counter(r['left_right_projection'] for r in verified)),
                  first_wrong=dict(Counter(r['first_observable_wrong'] for r in verified)),
                  first_shared_verified=dict(Counter(r['first_shared_proxy'] for r in verified)),
                  verified_native_top16_wrong=sum(r['native_top16_visible_identity']=='contradicts_visible_pair'
                                                  for r in verified),
                  readers=1, notes='Visual identity checked against the original image and caption. Routing outcome is a box-ownership proxy. Unknown earlier stages do not establish causal origin.',
                  review_sha256=sha(review), FINAL_access=False)
    put_table(HERE/'results/role_edges.csv', all_rows)
    put_table(HERE/'results/role_stage_summary.csv', stage_table)
    atomic_json(HERE/'results/ROLE_REVIEW_SUMMARY.json', result)
    print('Role review complete', result)


if __name__ == '__main__':
    main()

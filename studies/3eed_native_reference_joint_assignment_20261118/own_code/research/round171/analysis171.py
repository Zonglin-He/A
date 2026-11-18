"""Trace the actual use of newly admitted native reference candidates."""
from statistics import median

from scope171 import ARMS, HERE, OUT, SPLITS, WORK, atomic_json, read, sha
from predict171 import read_npz
from score165 import table, write_table


def key(row):
    return row['cohort'], row['case'], row['source']


def summarize():
    audit = read(HERE / 'EVALUATION_AUDIT.json')
    assert not audit['FINAL_access'] and all(sha(OUT / n) == h for n, h in audit['files'].items())
    paths = read(HERE / 'PREDICTION_SEAL.json')['files']
    arms = {(*key(r), r['arm']): r for r in table(OUT / 'arms.csv')}
    selected, summaries = [], []
    for row in table(OUT / 'reference_flow.csv'):
        if row['variant'] != 'D':
            continue
        cohort, case, source = key(row)
        name = f'{cohort}/{int(case):02d}_{source}.npz'
        path = WORK / 'predictions' / name
        assert sha(path) == paths[name]
        d = read_npz(path)
        main = arms[(*key(row), 'D_native_joint')]
        linked = arms[(*key(row), 'D_native_linked_joint')]
        mapped = arms[(*key(row), 'D_mapped_joint')]
        ref = int(d['choices'][ARMS.index('D_native_joint'), 1])
        output = ref < len(d['D_links'])
        assert output == (main['has_reference'] == '1')
        known = row['native_rank'] != ''
        selected.append(dict(cohort=cohort, case=case, source=source, target=row['target'], recording=row['recording'],
            reference_truth_verified=int(known), original_correct_reference_lost_at_mapping=int(row['lost_at_mapping']),
            native_rank=row['native_rank'], mapped_rank=row['mapped_rank'], native_reference_emitted=int(output),
            native_reference_unlinked=int(output and d['D_links'][ref] < 0),
            native_ref_identity=main['reference_identity'], mapped_ref_identity=mapped['reference_identity'],
            native_dual_correct=main['dual_correct'], linked_dual_correct=linked['dual_correct'], mapped_dual_correct=mapped['dual_correct'],
            native_iou=main['official_iou'], linked_iou=linked['official_iou'], mapped_iou=mapped['official_iou'],
            native_ref_visible=model_visible(main), linked_ref_visible=model_visible(linked), mapped_ref_visible=model_visible(mapped),
            signed_geometry=main['signed_relation']))
    assert len(selected) == 168
    for cohort in SPLITS:
        rows = [r for r in selected if r['cohort'] == cohort]
        unlinked = [r for r in rows if r['native_reference_unlinked']]
        lost = [r for r in rows if r['original_correct_reference_lost_at_mapping']]
        worse = [r for r in rows if float(r['native_iou']) < float(r['linked_iou']) - 1e-10]
        better = [r for r in rows if float(r['native_iou']) > float(r['linked_iou']) + 1e-10]
        summaries.append(dict(cohort=cohort, direction_records=len(rows), mapping_loss_cases=len(lost),
            mapping_loss_correct_references_selected=sum(r['native_ref_identity'] == '1' for r in lost),
            mapping_loss_dual_correct=sum(r['native_dual_correct'] == '1' for r in lost),
            selected_unlinked_refs=len(unlinked), selected_unlinked_truth_known=sum(r['reference_truth_verified'] for r in unlinked),
            selected_unlinked_correct=sum(r['native_ref_identity'] == '1' for r in unlinked),
            selected_unlinked_dual_correct=sum(r['native_dual_correct'] == '1' for r in unlinked),
            native_vs_linked_iou_gains=len(better), native_vs_linked_iou_losses=len(worse),
            native_vs_linked_losses_selecting_unlinked=sum(r['native_reference_unlinked'] for r in worse),
            native_visible_ref_median=median(r['native_ref_visible'] for r in rows if r['native_ref_visible'] != ''),
            mapped_visible_ref_median=median(r['mapped_ref_visible'] for r in rows if r['mapped_ref_visible'] != '')))
    dest = {
        'unmapped_reference_cases.csv': selected,
        'unmapped_reference_summary.csv': summaries,
    }
    for name, values in dest.items():
        write_table(OUT / name, values)
    atomic_json(HERE / 'MECHANISM_AUDIT.json', dict(scoring=sha(HERE / 'EVALUATION_AUDIT.json'),
                code=sha(HERE / 'analysis171.py'), files={n: sha(OUT / n) for n in dest}, FINAL_access=False))
    print(summaries, flush=True)


def model_visible(arm):
    return int(float(arm['ref_visible_seeds'])) if arm['ref_visible_seeds'] else ''


if __name__ == '__main__':
    summarize()

"""Grade sealed single-region scores, joint assignments, and their negative controls offline."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

from download import DATA, read_json, sha256, write_json
from geometry_audit import paired_stats
from geometry_cohort import read_jsonl
from object_region_audit import all_distinct_supported, ious, option_box
from object_region_plan import BASELINE, DIAG, RUN as BANK, index_rows
from oracle_region_binding import verify_files
import score
from single_region_verification import CONFIG, METHODS, RUN, concrete, job_layout, panel


READOUTS = ('original_hispatial', 'marked_prototype', 'whole_map_depth', 'joint_window', 'joint_region')
COMPARISONS = (
    ('eight_overlay', 'joint'),
    ('grouped', 'joint'),
    ('grouped', 'matched_budget_joint'),
    ('independent', 'joint'),
    ('swapped_joint', 'joint'),
    ('grouped', 'raw_rank'),
    ('joint', 'complement_joint'),
)


def audit_joint(odds, clusters, slots):
    # Exhaustive alternatives afford a separate objective check at this panel size.
    def valid(seq):
        choice_groups = [clusters[j] for j in seq if j is not None]
        return len(choice_groups) == len(set(choice_groups))

    def utility(seq):
        return sum(odds[i, j] for i, j in enumerate(seq) if j is not None)

    real = [[None, *np.flatnonzero(row>0)] for row in odds]
    assert all(k in real[i] for i, k in enumerate(slots)) and valid(slots)
    attainable = max(utility(seq) for seq in product(*real) if valid(seq))
    assert abs(utility(slots) - attainable) < 1e-8


def old_controls(roster):
    region_summary = read_json(BANK/'summary.json')
    prior_summary = read_json(BANK/'panel_summary.json')
    assert sha256(BANK/'scored_bindings.jsonl') == region_summary['artifacts']['scored_bindings.jsonl']
    assert sha256(BANK/'panel_scored_bindings.jsonl') == prior_summary['scored_bindings']
    assert sha256(BANK/'panel_predictions_seal.json') == prior_summary['prediction_seal']
    prior_seal = verify_files(BANK,'panel_predictions_seal.json')
    assert prior_seal['inference_rows'] == 1462
    old = {(r['id'], r['option']-1):r for r in read_jsonl(BANK/'scored_bindings.jsonl')
           if r['id'] in roster and r['concrete'] and r['annotated']}
    grouped = {(r['id'], r['option']-1):r for r in read_jsonl(BANK/'panel_scored_bindings.jsonl')}
    assert len(old) == len(grouped) == 379 and old.keys() == grouped.keys()
    return old, grouped, prior_summary


def replay(official_source, roster, predictions):
    source_file = DATA / 'embspatial/source.json'
    assert sha256(source_file) == read_json(DIAG/'inputs_seal.json')['baseline']['embspatial']['dataset_source']
    for record in read_json(source_file)['files']:
        assert sha256(source_file.parent/record['file']) == record['sha256']
    official, digest = score.import_official('embspatial', official_source)
    baseline = BASELINE/'embspatial'
    original = read_json(baseline/'predictions_seal.json')
    assert sha256(baseline/'predictions.jsonl') == original['predictions_sha256']
    assert sha256(baseline/'inputs.jsonl') == original['inputs_sha256']
    native = index_rows(baseline/'predictions.jsonl')
    prompts = {k:r['prompt'] for k,r in index_rows(baseline/'inputs.jsonl').items()}
    assert all(prompts[q.removeprefix('embspatial/')] == r['prompt'] for q,r in roster.items())
    verify_files(BANK,'prediction_seal.json')
    older = index_rows(BANK/'predictions.jsonl')
    rows = {}
    for arm in READOUTS:
        condition = 'region_depth' if arm == 'marked_prototype' else arm
        source = older if arm in ('marked_prototype', 'whole_map_depth') else predictions
        updated = {**native,**{q.removeprefix('embspatial/'):dict(response=source[q]['answers'][condition]) for q in roster}}
        where = RUN/'official_replay'/arm
        (where/'official_replay').mkdir(parents=True,exist_ok=True,mode=0o700)
        result,_,metrics = score.score_emb(official, where, updated, prompts)
        write_json(where/'official_metrics.json',metrics)
        rows[arm] = {'embspatial/'+r['id']:r for r in result if 'embspatial/'+r['id'] in roster}
        assert rows[arm].keys() == roster.keys()
    return rows, digest


def analyze(official_source):
    seal = verify_files(RUN, 'prediction_seal.json')
    inputs = read_json(RUN/'input_seal.json')
    assert sha256(RUN/'input_seal.json') == seal['inputs'] and seal['label_access'] is False
    assert sha256(RUN/'inputs.jsonl') == inputs['inputs'] and sha256(CONFIG) == inputs['config']
    assert sha256(CONFIG.with_name('single_region_verification.py')) == inputs['code']
    assert sha256(BANK/'input_seal.json') == inputs['original_bank']
    roster, regions = panel()
    predictions = index_rows(RUN/'predictions.jsonl')
    responses = index_rows(RUN/'scores.jsonl')
    jobs = job_layout(roster, regions)
    assert list(read_jsonl(RUN/'inputs.jsonl')) == jobs
    assert set(responses) == {r['id'] for r in jobs} and set(predictions) == roster.keys()

    geo_path = DIAG/'emb_box_depth.jsonl'
    assert sha256(geo_path) == read_json(DIAG/'audit_summary.json')['result_files'][geo_path.name]
    geo = index_rows(geo_path)
    old, grouped, prev = old_controls(roster)
    comparisons = Counter()
    question_rows, object_rows, auc = [], [], []
    swapped_name_margins, two_object_margins = [], []
    merged_frames = set()
    options = ('eight_overlay','grouped',*METHODS)
    pair_pools = {key:[] for key in ('matching_positive','unrelated_positive','matching_complement','unrelated_complement')}
    for q, item in roster.items():
        prior = geo[q]
        assert prior['options'] == item['options'] and prior['size'] == regions[item['key']]['size']
        candidates = regions[item['key']]['marked']
        boxes = [c['box'] for c in candidates]
        pred = predictions[q]
        clusters = pred['physical_groups']
        opts = [i for i, name in enumerate(item['options']) if concrete(name)]
        targets = [option_box(prior['boxes'][i], prior['size']) for i in opts]
        assert all(x is not None for x in targets)
        overlap = np.array([ious(box, boxes) for box in targets])
        pos = np.array([[responses[f'{q}/{i}/{j}']['positive'] for j in range(len(boxes))] for i in opts])
        neg = np.array([[responses[f'{q}/{i}/{j}']['complement'] for j in range(len(boxes))] for i in opts])
        swap = np.roll(pos, -1, axis=0)
        budget = pos.copy()
        for a, i in enumerate(opts):
            budget[a, pred['budget_by_option'][str(i)]:] = -np.inf

        current = {arm:[pred['bindings'][arm][str(i)] for i in opts] for arm in METHODS}
        for arm, likelihoods in (('joint',pos),('complement_joint',(pos-neg)/2),
                                 ('swapped_joint',swap),('matched_budget_joint',budget)):
            audit_joint(likelihoods, clusters, current[arm])
        question = dict(id=q,scene=prior['scene'],source=prior['source'],topic=item['topic'],conditions={},
                        depth_status=pred['depth_status'])
        scoped = dict(scene=prior['scene'],source=prior['source'],topic=item['topic'])
        maximums = np.column_stack([overlap[:, np.asarray(clusters)==value].max(axis=1)
                                    for value in sorted(set(clusters))])
        for t in (.25, .50):
            comparisons[f'questions_all_candidates_available{int(t*100)}'] += bool(np.all(overlap.max(axis=1)>=t))
            comparisons[f'questions_distinct_candidate_oracle{int(t*100)}'] += all_distinct_supported(overlap,t)
            comparisons[f'questions_distinct_cluster_oracle{int(t*100)}'] += all_distinct_supported(maximums,t)
        if len(clusters) > len(set(clusters)):
            merged_frames.add(item['key'])
        for a, i in enumerate(opts):
            labelled = overlap[a]>=.25
            unrelated = overlap[a]<=.05
            n = len(boxes)
            slots = {arm:current[arm][a] for arm in METHODS}
            previous = old[q,i]['selected_iou']
            group_iou = grouped[q,i]['grouped_selected_iou']
            group_slot = grouped[q,i]['chosen_proposal_index']
            assert abs(group_iou - (overlap[a,group_slot] if group_slot is not None else 0.0)) < 1e-8
            selected_iou = {'eight_overlay':previous,'grouped':group_iou,**{
                arm:float(overlap[a,j]) if j is not None else 0. for arm,j in slots.items()}}
            counts = {arm:dict(hit25=val>=.25,hit50=val>=.50) for arm,val in selected_iou.items()}

            comparisons['all_pairs'] += n
            comparisons['positive_yes'] += int((pos[a]>0).sum())
            comparisons['complement_yes'] += int((neg[a]>0).sum())
            comparisons['complement_inverted_sign_consistent'] += int(((pos[a]>0)==(neg[a]<0)).sum())
            for name, mask in (('matching',labelled),('unrelated',unrelated)):
                pair_pools[name+'_positive'].extend(pos[a,mask].tolist())
                pair_pools[name+'_complement'].extend(neg[a,mask].tolist())
            if labelled.any() and unrelated.any():
                delta = pos[a,labelled][:,None] - pos[a,unrelated][None,:]
                neg_delta = neg[a,unrelated][None,:] - neg[a,labelled][:,None]
                auc.append(dict(positive=float(((delta>0)+.5*(delta==0)).mean()),
                                complement=float(((neg_delta>0)+.5*(neg_delta==0)).mean())))
            next_object = overlap[(a+1) % len(opts)]
            valid_name_pairs = labelled & (next_object <= .05)
            name_difference = pos[a] - swap[a]
            swapped_name_margins.extend(name_difference[valid_name_pairs].tolist())
            swapped_matches = (next_object >= .25) & unrelated
            two_object_margins.extend((name_difference[valid_name_pairs][:,None] -
                                       name_difference[swapped_matches][None,:]).ravel().tolist())
            comparisons['objects_matching_available25'] += bool(labelled.any())
            comparisons['objects_matching_available50'] += bool((overlap[a]>=.50).any())
            comparisons['matched_budget_objects_available25'] += bool((overlap[a,:pred['budget_by_option'][str(i)]]>=.25).any())
            comparisons['matched_budget_objects_available50'] += bool((overlap[a,:pred['budget_by_option'][str(i)]]>=.50).any())
            comparisons['raw_rank_same_after_name_swap'] += int(pos[a].argmax()==swap[a].argmax())
            comparisons['same_joint_binding_after_name_swap'] += int(slots['joint']==slots['swapped_joint'] and slots['joint'] is not None)
            comparisons['both_joint_bound_with_name_swap'] += int(slots['joint'] is not None and slots['swapped_joint'] is not None)
            if not labelled.any():
                failure = 'no_candidate'
            elif counts['joint']['hit25']:
                failure = 'success'
            elif not counts['raw_rank']['hit25']:
                failure = 'bad_raw_rank'
            elif pos[a].max() <= 0:
                failure = 'rejected_correct_raw_rank'
            else:
                failure = 'joint_displaced_correct_raw_rank'
            comparisons['joint_outcome_'+failure] += 1
            object_rows.append(dict(id=q,option=i+1,**scoped,conditions=counts,ious=selected_iou,slots=slots,
                                    previous_grouped_slot=grouped[q,i]['chosen_proposal_index'],
                                    candidate_ious=overlap[a].tolist(),pos=pos[a].tolist(),neg=neg[a].tolist()))
        tail = object_rows[-len(opts):]
        for arm in options:
            question['conditions'][arm] = {m:all(o['conditions'][arm][m] for o in tail) for m in ('hit25','hit50')}
        question_rows.append(question)

    assert comparisons['all_pairs'] == seal['candidate_checks'] and len(object_rows) == 379
    counts = {arm:dict(hit25=sum(r['conditions'][arm]['hit25'] for r in object_rows),
                       hit50=sum(r['conditions'][arm]['hit50'] for r in object_rows),
                       all25=sum(r['conditions'][arm]['hit25'] for r in question_rows),
                       all50=sum(r['conditions'][arm]['hit50'] for r in question_rows))
              for arm in options}
    assert counts['eight_overlay']['hit25'] == prev['counts']['eight_overlay_hit25'] == 65
    assert counts['grouped']['hit25'] == prev['counts']['grouped_final_hit25'] == 81
    assert counts['grouped']['all25'] == prev['counts']['questions_correct_every_option25'] == 0
    for arm in METHODS:
        counts[arm]['selected'] = sum(r['slots'][arm] is not None for r in object_rows)
        counts[arm]['all_selected'] = sum(all(p is not None for p in r['bindings'][arm].values()) for r in predictions.values())
        counts[arm]['all_distinct_clusters'] = sum(
            all(j is not None for j in r['bindings'][arm].values()) and len({r['physical_groups'][j]
                for j in r['bindings'][arm].values()}) == len(r['bindings'][arm]) for r in predictions.values())
    paired = [dict(level=level,**paired_stats(rows, metric, a,b))
              for level,rows in (('object',object_rows),('question',question_rows))
              for metric in ('hit25','hit50') for a,b in COMPARISONS]

    qa = {'gate_passed':counts['joint']['hit25']>counts['grouped']['hit25'] and counts['joint']['all25']>counts['grouped']['all25'],
          'depth_status':dict(Counter(r['depth_status'] for r in predictions.values()))}
    if qa['gate_passed']:
        official, digest = replay(official_source, roster, predictions)
        qa_rows = [dict(id=r['id'],scene=r['scene'],source=r['source'],topic=r['topic'],
                        conditions={arm:{'correct':official[arm][r['id']]['correct']} for arm in READOUTS}) for r in question_rows]
        score.jsonl(RUN/'scored_qa.jsonl',qa_rows)
        qa.update(official_source_sha256=digest, correct={arm:sum(r['conditions'][arm]['correct'] for r in qa_rows) for arm in READOUTS},
                  paired=[paired_stats(qa_rows,'correct','original_hispatial',arm) for arm in READOUTS[1:]]+
                         [paired_stats(qa_rows,'correct','joint_window','joint_region')],
                  scored_qa_sha256=sha256(RUN/'scored_qa.jsonl'))
        assert qa['correct']['original_hispatial']==65 and qa['correct']['marked_prototype']==63

    score.jsonl(RUN/'scored_objects.jsonl',object_rows)
    score.jsonl(RUN/'scored_questions.jsonl',question_rows)
    stats = {k:int(v) for k,v in comparisons.items()}
    stats['frames_with_group_merge'] = len(merged_frames)
    opaque_groups = {(r['source'],r['scene']) for r in question_rows}
    name_advantage = np.asarray(swapped_name_margins)
    swapped_assignment_advantage = np.asarray(two_object_margins)
    yes_no_mass = {}
    for polarity in ('positive','complement'):
        logp = np.array([r[polarity+'_logprobs'] for r in responses.values()])
        yes_no_mass[polarity] = float(np.median(np.exp(logp).sum(axis=1)))
    write_json(RUN/'summary.json', dict(objects=len(object_rows),questions=len(question_rows),scene_clusters=len(opaque_groups),
        physical_frames=len({r['key'] for r in roster.values()}), counts=counts, stats=stats,paired=paired,
        semantic_controls=dict(matched_vs_unrelated_objects=len(auc),
            mean_per_object_rank_auc={k:float(np.mean([x[k] for x in auc])) for k in ('positive','complement')},
            fixed_region_name_control=dict(n=len(name_advantage),median_advantage=float(np.median(name_advantage)),
                                           true_name_higher=int((name_advantage>0).sum())),
            two_object_name_swap=dict(n=len(swapped_assignment_advantage),
                median_advantage=float(np.median(swapped_assignment_advantage)),
                correct_sum_higher=int((swapped_assignment_advantage>0).sum())),
            median_yes_no_probability_mass=yes_no_mass,
            log_odds={k:dict(n=len(v),median=float(np.median(v)),q25=float(np.percentile(v,25)),q75=float(np.percentile(v,75)),
                             positive=sum(x>0 for x in v)) for k,v in pair_pools.items()}),
        qa=qa,runtime=read_json(RUN/'runtime.json'),prediction_seal=seal,
        artifacts={k:sha256(RUN/k) for k in ('input_seal.json','prediction_seal.json','scored_objects.jsonl','scored_questions.jsonl')}))
    print('object and whole question metrics',counts,flush=True)
    print('QA',qa,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-source', type=Path, required=True)
    analyze(parser.parse_args().official_source)

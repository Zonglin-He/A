"""Independent, post-score reference and candidate-loss audit; no new method arm."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import run as R
from reference import hull_posterior, prototype_feedback, correct_once


def main():
    started = time.perf_counter()
    torch.set_num_threads(1)
    plan = R.verify()
    seal = R.read(R.D / 'sealed.json')
    score = R.read(R.D / 'score_receipt.json')
    report = R.read(R.P / 'report.json')
    assert report['status'] == 'complete_scored'
    assert R.sha(R.P / 'report.json') == score['report_sha256']
    files = ['reference.py', 'postseal_audit.py']
    source = {f: R.sha(Path(__file__).parent / f) for f in files}
    binding = dict(parent_plan=R.sha(R.D / 'plan.json'), parent_seal=R.sha(R.D / 'sealed.json'),
                   parent_score=R.sha(R.D / 'score_receipt.json'), code=source,
                   scope='All67 sealed spatial inputs; no new final predictions/GT access. Saved signed results choose cases only after seal and scoring.')
    R.atomic(R.D / 'postseal_audit_plan.json', binding)
    R.atomic(R.D / 'postseal_audit_worker.json', R.identity())
    rows = {r['query']: r for r in report['rows']}
    cases, reference_errors = [], []
    for i in plan['items']:
        key = seal['files'][i['source_key']]['alias']
        cap = torch.load(i['capture_path'], map_location='cpu', weights_only=False)
        raw = torch.load(i['raw_path'], map_location='cpu', weights_only=False)
        current = R.SpatialGeometry.__new__(R.SpatialGeometry)
        current.__dict__.update(cap['geometry'])
        logs = cap.get('native_logits')
        if logs is None:
            logs = [z[-1] for z in cap['logits']]
        grids = R.probabilities(cap['records'], logs)
        ids, q0 = R.mass_table(grids)
        ref_ids, ref_q0 = hull_posterior(grids)
        assert ids == ref_ids
        old_q, old_trace = R.one_step(cap['content_features'], q0)
        feedback = prototype_feedback(cap['content_features'], ref_q0)
        new_q = correct_once(ref_q0, feedback)
        errors = [float(np.max(abs(q0-ref_q0))), float(np.max(abs(old_trace['L']-feedback))), float(np.max(abs(old_q-new_q)))]
        assert max(errors) < 1e-11
        assert R.final_mbr(ids, old_q)['interval'] == R.final_mbr(ids, new_q)['interval']
        reference_errors.append(errors)

        features, valid = R.known_grid(raw['spatial_inputs'], ids, current.height, current.width)
        ev = R.C.load_evidence(i, cap)
        near = R.build(features, valid, ids, cap['positions8'], ev.seeds, cap['E'], current.native_boxes, nearest=True)
        with np.load(R.D / seal['files'][i['source_key']]['path'], allow_pickle=False) as saved:
            interval = saved['Current_interval'].tolist()
            old = R.evidence_dp(current.for_interval(interval))
            new = R.evidence_dp(near.for_interval(interval))
            assert np.array_equal(old['final'].numpy(), saved['Current'])
            assert np.array_equal(new['final'].numpy(), saved['NearestBoxLink'])
        changed = (old['final'] != new['final']).any(-1)
        same_candidate = (near.boxes == old['final'][:, None]).all(-1) & near.valid
        absent = ~same_candidate.any(-1)
        old_slot = old['actions']
        source_j = current.source_anchors[torch.arange(len(ids)), (old_slot-1).clamp_min(0)]
        nearest_j = near.source_anchors[:, 0]
        dropped_other_anchor = changed & (old_slot > 0) & (source_j != nearest_j) & absent
        invalid_nearest = changed & ~near.side_available.any(-1)
        value = rows[key]['values']
        cases.append(dict(query=key, delta_v_pp=100*(value['NearestBoxLink']['v']-value['Current']['v']),
                          delta_unobserved_s_pp=100*(value['NearestBoxLink']['unobserved_actual_sIoU']-value['Current']['unobserved_actual_sIoU']),
                          changed_boxes=int(changed.sum()), old_selected_box_absent_from_new_bank=int((changed & absent).sum()),
                          changed_nodes_losing_other_anchor=int(dropped_other_anchor.sum()),
                          changed_nodes_nearest_fit_unavailable=int(invalid_nearest.sum())))
    rng = np.random.default_rng(20260920)
    for n in (16, 33, 65, 133, 200):
        q = np.triu(rng.random((n, n)), 1); q /= q.sum()
        z = rng.normal(size=(n, 17))
        old, _ = R.one_step(z, q)
        new = correct_once(q, prototype_feedback(z, q))
        assert np.max(abs(old-new)) < 1e-11
        assert np.array_equal(correct_once(q, prototype_feedback(np.ones((n, 17)), q)), q)
    assert not torch.cuda.is_initialized()
    output = dict(status='passed', reference_cases=67, synthetic_frame_counts=[16,33,65,133,200],
                  constant_exact_noop=True, max_hull_feedback_posterior_errors=np.max(reference_errors, axis=0).tolist(),
                  all_reference_intervals_exact=True, new_GT_read=False, new_final_predictions=0, new_models=0,
                  case_scope='Observed associations in post-seal candidate removal, not isolated causal attribution; original native time held fixed.',
                  cases=cases, seconds=time.perf_counter()-started,
                  binding_sha256=hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest())
    R.atomic(R.P / 'REFERENCE_CHECK.json', output)
    R.atomic(R.D / 'postseal_audit_receipt.json', dict(public_sha256=R.sha(R.P/'REFERENCE_CHECK.json'), **R.identity()))
    print(json.dumps({k:v for k,v in output.items() if k != 'cases'}), flush=True)


if __name__ == '__main__':
    main()

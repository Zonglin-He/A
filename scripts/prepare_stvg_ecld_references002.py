"""Keep every dev case; derive matched references solely from frozen R1 replies."""
import copy
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import *
from scripts.stvg_ecld_teacher_protocol_v1 import canonical_sha256
from scripts.prepare_stvg_ecld_teacher_v1 import _minimal_bindings


def main():
    verify_runtime()
    sys.addaudithook(guard)
    barrier = read(BASE / 'TEACHER_GLOBAL_BARRIER.json')
    assert barrier['status'] == 'complete' and barrier['GT_read'] is False
    assert barrier['count'] == 320 and len(barrier['files']) == 320
    assert barrier['job_lock_sha256'] == sha(BASE/'inputs_teacher/TEACHER_JOB_LOCK_V2.json')
    assert barrier['input_barrier_sha256'] == sha(BASE/'inputs_teacher/INPUT_BARRIER_V2.json')
    assert barrier['runtime_sha256'] == sha(BASE/'TEACHER_RUNTIME.json')
    for row in barrier['files']:
        assert sha(ROOT / row['path']) == row['sha256']
    jobs = read(BASE / 'inputs_teacher/TEACHER_JOB_LOCK_V2.json')['records']
    assert {r['path'] for r in barrier['files']} == {
        str((BASE/'teacher'/(j['job_id']+'.json')).relative_to(ROOT)) for j in jobs}
    parent_jobs = {j['job_id']: j for j in read(PARENT/'TEACHER_JOB_LOCK.json')['records']}
    design = read(BASE / 'DESIGN_LOCK.json')
    records = []
    for rec in roster():
        case, _ = load_case(rec)
        cid = rec['probe_alias']
        record = dict(case_id=cid, dataset=rec['dataset'],
                      source_case_sha256=case['source_case_sha256'], GT_read=False)
        for protocol in ['A', 'B']:
            anchors, decisions = [], []
            selected_jobs = [j for j in jobs if j['case_id'] == cid
                             and j['protocol_variant'] == protocol and j['control'] == 'base']
            assert len(selected_jobs) == 4
            for job in sorted(selected_jobs, key=lambda j: j['center_slot']):
                parent = parent_jobs[job['parent_job_id']]
                assert canonical_sha256(parent) == job['parent_job_schema_sha256']
                assert parent['case_id'] == cid and parent['variant'] == 'new_crop'
                assert parent['center_physical_frame_id'] == job['center_physical_frame_id']
                assert _minimal_bindings(parent, 5) == job['candidate_bindings']
                path = BASE / 'teacher' / (job['job_id'] + '.json')
                result = read(path)
                for field in ['job_id', 'case_id', 'dataset', 'center_slot', 'protocol_variant',
                              'control', 'source_case_sha256', 'source_request_sha256',
                              'query_sha256', 'prompt_sha256', 'job_binding_sha256']:
                    assert result[field] == job[field], (cid, field)
                assert result['input_barrier_sha256'] == barrier['input_barrier_sha256']
                assert not result['GT_read']
                if result['runtime_sha256'] != barrier['runtime_sha256']:
                    inherited = read(BASE/'TEACHER_RUNTIME.json')['accepted_prefix'][str(path.relative_to(ROOT))]
                    assert sha(path) == inherited['sha256'] and result['runtime_sha256'] == inherited['runtime_sha256']
                parsed = result['parsed']
                chosen = parsed.get('selected_candidate_id')
                reason = 'invalid_teacher_output'
                anchor = None
                if parsed['status'] == 'valid':
                    decision = parsed['decision']
                    if decision == 'ABSTAIN':
                        reason = 'teacher_abstained'
                    elif decision == 'KEEP':
                        assert chosen == job['source_candidate_id']
                        reason = 'keep_source_no_new_reference'
                    else:
                        assert decision == 'REPLACE' and chosen != job['source_candidate_id']
                        track = next(t for t in parent['candidate_bindings'] if int(t['id']) == chosen)
                        if track['presence'] < design['evidence_gate']['minimum_presence']:
                            reason = 'insufficient_geometric_presence'
                        else:
                            a, b, c, d = track['center_box']
                            assert 0 <= a < c <= 1 and 0 <= b < d <= 1
                            fid = int(job['center_physical_frame_id'])
                            position = list(map(int, case['frame_ids'])).index(fid)
                            grounded = all(parsed.get(k) == v for k, v in
                                           design['evidence_gate']['grounded_fields'].items())
                            anchor = dict(position=position, physical_frame_id=fid,
                                box_cxcywh=[(a+c)/2, (b+d)/2, c-a, d-b],
                                weight=float(track['presence']), candidate_id=chosen,
                                teacher_output_sha256=sha(path), evidence_gate=grounded,
                                evidence_labels={k: parsed[k] for k in
                                                 design['evidence_gate']['grounded_fields']})
                            anchors.append(anchor)
                            reason = 'candidate_reference';
                decisions.append(dict(center_slot=job['center_slot'], selected=chosen,
                                      reason=reason, teacher_output_sha256=sha(path)))
            # The same reference set is passed to original and evidence-aware fitters.
            record[protocol] = dict(anchors=anchors, decisions=decisions)
            record['C' if protocol == 'A' else 'D'] = copy.deepcopy(record[protocol])
        records.append(record)
    write(BASE / 'REFERENCE_LOCK.json', dict(status='locked', records=records,
        teacher_barrier_sha256=sha(BASE / 'TEACHER_GLOBAL_BARRIER.json'),
        job_lock_sha256=sha(BASE / 'inputs_teacher/TEACHER_JOB_LOCK_V2.json'), GT_read=False))
    print('matched reference records', len(records), flush=True)


if __name__ == '__main__':
    main()

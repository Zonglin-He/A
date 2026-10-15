"""Post-inference visual/ID sensitivity diagnostics without GT or Student selection."""
from pathlib import Path
import collections
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, read, write, sha


def main():
    barrier = read(BASE/'TEACHER_GLOBAL_BARRIER.json')
    assert barrier['status'] == 'complete' and barrier['count'] == 320 and not barrier['GT_read']
    assert len(barrier['files']) == 320
    lock = read(BASE/'inputs_teacher/TEACHER_JOB_LOCK_V2.json')
    assert sha(BASE/'inputs_teacher/TEACHER_JOB_LOCK_V2.json') == barrier['job_lock_sha256']
    paths = {r['path']: r for r in barrier['files']}
    assert len(paths) == 320
    rows, lookup = [], {}
    for job in lock['records']:
        path = BASE/'teacher'/(job['job_id']+'.json')
        assert sha(path) == paths[str(path.relative_to(ROOT))]['sha256']
        response = read(path)
        assert response['job_binding_sha256'] == job['job_binding_sha256'] and not response['GT_read']
        parsed = response['parsed']
        r = {k: job[k] for k in ['job_id', 'case_id', 'dataset', 'protocol_variant', 'center_slot', 'control']}
        r.update(parsed=parsed, response_sha256=sha(path),
            input_tokens=response['input_tokens'], visual_tokens=response['visual_tokens'],
            output_tokens=response['output_tokens'], truncated=response['output_truncated_at_cap'],
            model_seconds=response['model_forward_generate_seconds'],
            request_seconds=response['end_to_end_request_seconds'],
            generation_call_count=response['generation_call_count'],
            GPU_peak_bytes=response['GPU_peak_bytes'])
        rows.append(r)
        lookup[job['case_id'], job['center_slot'], job['protocol_variant'], job['control']] = r
    groups, controls = [], []
    for ds in ['hc2', 'vidstg']:
        for protocol in ['A', 'B']:
            rr = [r for r in rows if r['dataset'] == ds and r['protocol_variant'] == protocol and r['control'] == 'base']
            assert len(rr) == 64
            groups.append(dict(dataset=ds, protocol_variant=protocol, total=len(rr),
                valid=sum(r['parsed']['status']=='valid' for r in rr),
                truncated=sum(r['truncated'] for r in rr),
                decisions=dict(collections.Counter(r['parsed']['decision'] if r['parsed']['status']=='valid' else 'invalid' for r in rr))))
    for row in rows:
        if row['control'] == 'base': continue
        baseline = lookup[row['case_id'], row['center_slot'], row['protocol_variant'], 'base']
        a, b = row['parsed'], baseline['parsed']
        valid = a['status']=='valid' and b['status']=='valid'
        controls.append(dict(case_id=row['case_id'], dataset=row['dataset'], protocol_variant=row['protocol_variant'],
            center_slot=row['center_slot'], control=row['control'], both_valid=valid,
            same_stable_candidate=(a['selected_candidate_id']==b['selected_candidate_id']) if valid else None,
            same_decision=(a['decision']==b['decision']) if valid else None,
            nonvisual_grounded_assertion=(a['status']=='valid' and a['actor_role']=='matched'
                and a['action_evidence']=='visible' and a['object_extent']=='complete') if row['control']=='visual_null' else None))
    pairs=[]
    for row in rows:
        if row['control']!='base' or row['protocol_variant']!='A': continue
        b=lookup[row['case_id'],row['center_slot'],'B','base']
        a=row['parsed'];bb=b['parsed']; valid=a['status']=='valid' and bb['status']=='valid'
        pairs.append(dict(case_id=row['case_id'],dataset=row['dataset'],center_slot=row['center_slot'],both_valid=valid,
            same_stable_candidate=(a['selected_candidate_id']==bb['selected_candidate_id']) if valid else None,
            A_decision=a['decision'],B_decision=bb['decision']))
    write(BASE/'TEACHER_DIAGNOSTIC_READBACK.json',dict(status='pass',GT_read=False,CPU_only=True,
        source_barrier_sha256=sha(BASE/'TEACHER_GLOBAL_BARRIER.json'),job_count=320,records=rows,
        main_summary=groups,controls=controls,paired_A_B=pairs,
        total_actual_generation_calls=sum(r['generation_call_count'] for r in rows),
        accepted_prefix_count=barrier['accepted_prefix_count'],
        cost=dict(total_generate_seconds=sum(r['model_seconds'] for r in rows),
            total_request_seconds=sum(r['request_seconds'] for r in rows),
            total_input_tokens=sum(r['input_tokens'] for r in rows),
            total_visual_tokens=sum(r['visual_tokens'] for r in rows),
            total_output_tokens=sum(r['output_tokens'] for r in rows),
            maximum_GPU_peak_bytes=max(r['GPU_peak_bytes'] for r in rows),
            cold_start_count=2,scope='generation/request sums exclude both cold-load intervals and failed attempt overhead; those remain in preserved unit logs'),
        limitation='closed Teacher labels are uncalibrated assertions; null sensitivity and ID invariance are not semantic truth; no GT correctness conclusion',time=time.time()))
    print('Teacher sensitivity readback complete',groups,flush=True)


if __name__ == '__main__':
    main()

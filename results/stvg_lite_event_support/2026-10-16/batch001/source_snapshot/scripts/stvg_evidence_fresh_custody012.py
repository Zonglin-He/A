"""Exact fresh32 source and request custody; no model, annotation or media work."""
from scripts.stvg_evidence_common_v1 import BASE, ROOT, read, sha, load_case

def checked_load_case(record):
    case, path = load_case(record)
    barrier_path = BASE / 'confirmation/SOURCE_CAPTURE_BARRIER.json'
    barrier = read(barrier_path)
    assert barrier['status'] == 'complete' and barrier['count'] == 32 and not barrier['GT_read']
    assert barrier['roster_sha256'] == sha(BASE / 'CONFIRMATION_ROSTER_LOCK.json')
    matches = [x for x in barrier['files'] if ROOT / x['path'] == path]
    assert len(matches) == 1 and matches[0]['sha256'] == sha(path)
    assert matches[0]['bytes'] == path.stat().st_size
    assert len(barrier['files']) == len({x['path'] for x in barrier['files']}) == 32
    return case, path

def check_request(case_id, temporal):
    request_path = BASE / 'confirmation/requests' / (case_id + '.json')
    teacher_path = BASE / 'confirmation/temporal_teacher' / (case_id + '.pt')
    import torch
    teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
    assert teacher['request_sha256'] == sha(request_path) == temporal['request_sha256']
    assert temporal['source_temporal_teacher_sha256'] == sha(teacher_path)
    assert not teacher['GT_read'] and not temporal['GT_read']
    return sha(request_path)

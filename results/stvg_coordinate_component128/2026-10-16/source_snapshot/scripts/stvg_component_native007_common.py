"""Locked Native-clock deletion continuation; imports load no model or labels."""
from pathlib import Path
import copy
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_component_ablation_common_v1 import BASE as COMPONENT, SEARCH, read, write, sha, roster, load_case, verify_final

BASE = COMPONENT / 'native_clock/revision002'
TRIAL = BASE / 'trials/N000'
RUNTIME = BASE / 'EXECUTION_RUNTIME007.json'


def verify():
    runtime = read(RUNTIME)
    assert runtime['status'] == 'locked' and runtime['GT_read'] is False
    for rel, expected in {**runtime['pins'], **runtime['protected']}.items():
        assert sha(ROOT / rel) == expected, rel
    return runtime


def input_readback():
    import numpy as np
    verify_final()
    bar = read(BASE / 'CANDIDATE_INPUT_BARRIER.json')
    lock = read(BASE / 'TEACHER_JOB_LOCK.json')
    assert bar['status'] == 'complete' and bar['count'] == 128 and not bar['GT_read']
    assert lock['status'] == 'locked' and not lock['GT_read'] and lock['case_count'] == 128
    assert lock['count'] == len(lock['records']) == bar['teacher_job_count'] <= 512
    assert bar['bindings']['teacher_job_lock_sha256'] == sha(BASE / 'TEACHER_JOB_LOCK.json')
    assert bar['bindings']['final_parameter_lock_sha256'] == sha(COMPONENT / 'FINAL_PARAMETER_LOCK.json')
    seen = set()
    for x in bar['files']:
        p = ROOT / x['path']
        assert str(p.resolve()).startswith(str(BASE.resolve()) + '/')
        assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256']
        assert x['path'] not in seen
        seen.add(x['path'])
    cases = {r['probe_alias']: r for r in roster()}
    assert len(cases) == len(bar['case_records']) == 128
    case_inputs = {r['case_id']: r for r in bar['case_records']}
    assert set(cases) == set(case_inputs)
    source_hashes = {}
    for cid, rec in cases.items():
        source, p = load_case(rec)
        row = case_inputs[cid]
        assert row['source_case_sha256'] == source['source_case_sha256']
        assert row['native_interval_half_open'] == list(source['Native_interval'])
        assert row['GT_read'] is False and row['temporal_selection_lock_used_for_centers'] is False
        source_hashes[cid] = source['source_case_sha256']
    job_ids = set()
    for job in lock['records']:
        assert job['job_id'] not in job_ids
        job_ids.add(job['job_id'])
        assert job['case_id'] in cases and job['source_case_sha256'] == source_hashes[job['case_id']]
        assert job['clock_source'] == 'source_case.Native_interval' and job['variant'] == 'new_crop'
        assert job['final_parameter_lock_sha256'] == sha(COMPONENT / 'FINAL_PARAMETER_LOCK.json')
        assert hashlib.sha256(job['prompt'].encode()).hexdigest() == job['prompt_sha256']
        assert len(job['videos']) == 2
        for index, v in enumerate(job['videos']):
            p = ROOT / v['path']
            assert v['path'] in seen and sha(p) == v['sha256']
            a = np.load(p, allow_pickle=False)
            assert a.shape == ((4, 3, 448, 448) if index == 0 else (5, 3, 448, 448))
            assert hashlib.sha256(a.tobytes()).hexdigest() == v['tensor_sha256']
    return dict(status='pass', count=128, teacher_job_count=len(job_ids), file_count=len(seen),
                CPU_only=True, GT_read=False, input_barrier_sha256=sha(BASE/'CANDIDATE_INPUT_BARRIER.json'),
                job_lock_sha256=sha(BASE/'TEACHER_JOB_LOCK.json'), scope='all source/native clock, input files and tensors, prompts and job identity; no semantic correctness claim')


def build_reference_locks():
    verify()
    bar = read(BASE / 'TEACHER_GLOBAL_BARRIER.json')
    jobs = read(BASE / 'TEACHER_JOB_LOCK.json')['records']
    assert bar['status'] == 'complete' and bar['count'] == len(jobs) and bar['GT_read'] is False
    files = {x['path']: x for x in bar['files']}
    assert len(files) == len(jobs)
    refs, times = [], []
    for rec in roster():
        cid = rec['probe_alias']
        case, source = load_case(rec)
        anchors = []
        for j in jobs:
            if j['case_id'] != cid:
                continue
            p = BASE / 'teacher' / (j['job_id'] + '.json')
            entry = files[str(p.relative_to(ROOT))]
            assert sha(p) == entry['sha256']
            d = read(p)
            assert d['GT_read'] is False and d['source_case_sha256'] == case['source_case_sha256']
            assert d['job_binding_sha256'] == hashlib.sha256(json.dumps(j, sort_keys=True).encode()).hexdigest()
            chosen = d['parsed']['selected_candidate_id']
            if chosen in (None, 1):
                continue
            tr = next(x for x in j['candidate_bindings'] if int(x['id']) == chosen)
            if tr['presence'] < .6:
                continue
            left, top, right, bottom = tr['center_box']
            assert 0 <= left < right <= 1 and 0 <= top < bottom <= 1
            fid = int(j['center_physical_frame_id'])
            assert case['Native_interval'][0] <= fid < case['Native_interval'][1]
            position = list(map(int, case['frame_ids'])).index(fid)
            anchors.append(dict(position=position, physical_frame_id=fid,
                                box_cxcywh=[(left+right)/2, (top+bottom)/2, right-left, bottom-top],
                                weight=tr['presence'], candidate_id=chosen, teacher_output_sha256=sha(p)))
        assert len({a['position'] for a in anchors}) == len(anchors) <= 4
        refs.append(dict(case_id=cid, dataset=rec['dataset'], source_case_sha256=case['source_case_sha256'], anchors=anchors, GT_read=False))
        times.append(dict(case_id=cid, dataset=rec['dataset'], source_case_sha256=case['source_case_sha256'], selected_interval=list(case['Native_interval']), clock_source='Native_interval', GT_read=False))
    write(BASE/'REFERENCE_LOCK.json', dict(status='locked', count=128, records=refs, GT_read=False))
    write(BASE/'TEMPORAL_LOCK.json', dict(status='locked', count=128, records=times, GT_read=False))
    final = verify_final()
    TRIAL.mkdir(parents=True, exist_ok=False)
    for name in ['REFERENCE_LOCK.json', 'TEMPORAL_LOCK.json']:
        (TRIAL/name).write_bytes((BASE/name).read_bytes())
    design = dict(status='locked', trial_id='N000', variant='native_clock', ablation_flags={},
                  scope='true no-Temporal pipeline: Native clock candidates, same R1 protocol, fixed final M2 Spatial OPD',
                  configs={ds:{'M2':final['configs'][ds]} for ds in ['hc2','vidstg']},
                  response_settings={ds:{'M1':final['response_settings'][ds], 'M2':final['response_settings'][ds]} for ds in ['hc2','vidstg']},
                  family_reuse_from={ds:{'M2':None} for ds in ['hc2','vidstg']}, GT_read=False)
    write(TRIAL/'DESIGN_LOCK.json', design)
    runtime = read(RUNTIME)
    pins = copy.deepcopy(final['pins'])
    pins.update(runtime['pins'])
    for p in [RUNTIME, BASE/'REFERENCE_LOCK.json', BASE/'TEMPORAL_LOCK.json', BASE/'TEACHER_GLOBAL_BARRIER.json', TRIAL/'REFERENCE_LOCK.json', TRIAL/'TEMPORAL_LOCK.json', TRIAL/'DESIGN_LOCK.json']:
        pins[str(p.relative_to(ROOT))] = sha(p)
    write(TRIAL/'RUNTIME.json', dict(status='locked', trial_id='N000', pins=pins, protected=final['protected'], GT_read=False))
    write(BASE/'ROOT_REFERENCE_READBACK007.json', dict(status='pass', count=128,
          teacher_job_count=len(jobs), reference_sources=sum(bool(x['anchors']) for x in refs),
          anchor_count=sum(len(x['anchors']) for x in refs), GT_read=False, CPU_only=True,
          scope='original parsed non-Source choice, presence>=0.6, exact center geometry and Native source binding; not GT reference quality'))

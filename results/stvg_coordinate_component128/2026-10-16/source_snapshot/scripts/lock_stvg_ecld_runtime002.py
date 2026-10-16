"""Pin current local import closure and immutable inputs before bounded inference."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, PARENT, OLD, read, write, sha


def resolve(name):
    path = ROOT.joinpath(*name.split('.'))
    if path.with_suffix('.py').is_file():
        return path.with_suffix('.py')
    if (path/'__init__.py').is_file():
        return path/'__init__.py'


def closure(seeds):
    found, queue = set(), list(seeds)
    while queue:
        path = queue.pop()
        if path in found:
            continue
        found.add(path)
        module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        package = module.rsplit('.', 1)[0] if path.name != '__init__.py' else module[:-9]
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    parts = package.split('.')
                    base = '.'.join(parts[:len(parts)-node.level+1] + ([base] if base else []))
                names = [base] + [base+'.'+x.name for x in node.names if x.name != '*']
            for name in names:
                dep = resolve(name)
                if dep is not None and dep not in found:
                    queue.append(dep)
    return found


def main():
    assert not (BASE/'RUNTIME.json').exists() and not (BASE/'TEACHER_RUNTIME.json').exists()
    seeds = sorted((ROOT/'scripts').glob('*ecld*.py'))
    sources = closure(seeds)
    pins = {str(p.relative_to(ROOT)): sha(p) for p in sorted(sources)}
    for name in ['DESIGN_LOCK.json', 'ROSTER_LOCK.json', 'ROOT_CPU_FIT_CONTRACTS.json',
                 'ROOT_INPUT_PIXEL_REVIEW.json', 'p0/ROOT_SUPPORT_READBACK004.json',
                 'ROOT_TEACHER_INPUT_READBACK.json', 'ROOT_REFERENCE_GEOMETRY_ABI.json',
                 'recovery/cache_payload_size002/CAPTURE.json',
                 'inputs_teacher/TEACHER_JOB_LOCK_V2.json', 'inputs_teacher/INPUT_BARRIER_V2.json',
                 'inputs_teacher/LEGACY_TOP1_REFERENCES_V2.json', 'inputs_teacher/CPU_CONTRACTS_V2.json']:
        p = BASE/name
        pins[str(p.relative_to(ROOT))] = sha(p)
    for p in [PARENT/'RUNTIME006.json', PARENT/'STUDENT_GLOBAL_BARRIER.json',
              PARENT/'TEACHER_GLOBAL_BARRIER.json', PARENT/'CANDIDATE_INPUT_BARRIER.json',
              PARENT/'REFERENCE_LOCK.json', PARENT/'TEACHER_JOB_LOCK.json', OLD/'DEVELOPMENT_ROSTER_LOCK.json',
              ROOT/'artifacts/stvg_weightfree_coordinate_v1/SPATIAL_SELECTION_LOCK.json',
              ROOT/'protocols/stvg_evidence_conditioned_v1.md']:
        pins[str(p.relative_to(ROOT))] = sha(p)
    protected = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT/'methods/CURRENT_METHOD.json', OLD/'FROZEN_METHOD_LOCK.json']}
    from scripts.run_stvg_ecld_teacher002 import EXPECTED_GENERATION
    receipt = ROOT/'artifacts/stvg_target_teacher_distillation_v1/ROOT_OFFICIAL_MODEL_FILE_READBACK.json'
    teacher = dict(schema='stvg-ecld-teacher-runtime-v1', status='locked', GT_read=False,
        design_lock_sha256=sha(BASE/'DESIGN_LOCK.json'),
        development_roster_sha256=sha(OLD/'DEVELOPMENT_ROSTER_LOCK.json'),
        input_barrier_sha256=sha(BASE/'inputs_teacher/INPUT_BARRIER_V2.json'),
        job_lock_sha256=sha(BASE/'inputs_teacher/TEACHER_JOB_LOCK_V2.json'),
        legacy_top1_refs_sha256=sha(BASE/'inputs_teacher/LEGACY_TOP1_REFERENCES_V2.json'),
        model_name='XiaowenZhang/stvg-r1-model-7b', checkpoint_path='checkpoints/STVG-R1-7B',
        model_readback_path=str(receipt.relative_to(ROOT)), model_readback_sha256=sha(receipt),
        model_revision=read(receipt)['revision'], generation=EXPECTED_GENERATION,
        pins=pins, protected=protected, time=time.time())
    teacher['accepted_prefix'] = read(BASE/'recovery/cache_payload_size002/CAPTURE.json')['accepted_prefix']
    write(BASE/'TEACHER_RUNTIME.json', teacher)
    pins[str((BASE/'TEACHER_RUNTIME.json').relative_to(ROOT))] = sha(BASE/'TEACHER_RUNTIME.json')
    versions = {}
    for package in ['torch', 'numpy', 'transformers', 'accelerate', 'safetensors']:
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    write(BASE/'RUNTIME.json', dict(status='locked', GT_read=False, pins=pins, protected=protected,
        local_static_import_closure_count=len(sources), environment_versions=versions,
        limitation='static local Python import closure plus explicit artifacts; external environment versions are recorded, not a full binary supply-chain proof',
        bootstrap_executed=False, full_started=False, time=time.time()))
    print('runtime locked', len(sources), len(pins), flush=True)


if __name__ == '__main__':
    main()

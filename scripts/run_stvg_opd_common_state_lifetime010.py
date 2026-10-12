"""Bounded O capture-memory qualification and exact formal-prefix continuation."""
import collections, gc, os, subprocess, sys, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha, load, save
import scripts.run_stvg_opd_common_state_geometry_lease006 as original
target = original.target
BASE = target.BASE
REC = BASE / 'frozen_attention_lifetime010'
PYTHON = ROOT / '.conda/tubedetr/bin/python'


def setup():
    original.REC = REC
    original.setup()
    import scripts.run_spatial_ssl_gpu_v1 as capture_module
    from vg_tta.stvg_opd_frozen_attention_lifetime010 import frozen_forward
    capture_module.frozen_forward = frozen_forward
    return target.verify()


def qualification():
    setup()
    import torch
    from scripts.stvg_opd_design_ablation_common_v1 import activate_old
    activate_old()
    from scripts.run_decota_paper_main_v1 import model_for
    from methods.decota_final_simplified_v1.tensors import state_hash
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_common_state_geometry_v1 import bare
    design = read(REC / 'DESIGN_LOCK.json')
    cohort = read(BASE / 'COMMON_STATE_COHORT.json')['records']
    lease = target.priority_lease()
    files, count, model, stage = {}, 0, None, None
    try:
        for index in design['qualification_cell_indices']:
            r = cohort[index]
            if stage != r['stage']:
                if model is not None:
                    assert state_hash(model.state_dict()) == model_hash
                    del model
                    gc.collect(); torch.cuda.empty_cache()
                model = model_for(r['source'])
                model_hash = state_hash(model.state_dict())
                stage = r['stage']
            p = ROOT / r['original_Full']['path']
            assert sha(p) == r['original_Full']['sha256']
            old = load(p)
            for repeat in (0, 1):
                torch.cuda.reset_peak_memory_stats()
                base, row, expert = target.capture(model, r, old, collections.OrderedDict())
                for arm in target.ARM:
                    fit, checked = target.fit_arm(base, row, expert, old, arm, True)
                    comparison = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
                    if r['qualify']:
                        q = load(BASE / 'qualification' / f'{index:05}' / (arm + '.pt'))
                        equal(bare(fit), bare(q['fit']), comparison)
                        equal(fit['geometry'], q['fit']['geometry'], comparison)
                    if repeat:
                        first = load(REC / 'qualification' / f'{index:05}' / (arm + '_repeat0.pt'))
                        equal(bare(fit), bare(first['fit']), comparison)
                        equal(fit['geometry'], first['fit']['geometry'], comparison)
                    path = REC / 'qualification' / f'{index:05}' / (arm + f'_repeat{repeat}.pt')
                    z = dict(index=index, dataset=r['dataset'], stage=r['stage'], arm=arm, repeat=repeat,
                             fit=fit, math_audit=checked, original_qualified_cell=r['qualify'],
                             complete_comparison=comparison, GT_read=False,
                             source_state_sha256=model_hash, input_sha256=r['original_Full']['input_sha256'],
                             runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
                             peak_memory_bytes=torch.cuda.max_memory_allocated())
                    save(path, z)
                    write(path.with_suffix('.json'), dict(status='pass', sha256=sha(path),
                          bytes=path.stat().st_size, GT_read=False, time=time.time()))
                    files[str(path.relative_to(ROOT))] = sha(path)
                    count += 1
                    status(REC / 'STATUS.json', dict(status='running_actual_capture_memory_qualification',
                           index=index, dataset=r['dataset'], arm=arm, repeat=repeat,
                           done=count, total=48, pid=os.getpid(), GT_read=False, time=time.time()))
                    print('LIFETIME010_QUALIFICATION', count, 48, index, arm, repeat, flush=True)
                    del fit, checked, z
                del base, row, expert
                gc.collect(); torch.cuda.empty_cache()
            assert state_hash(model.state_dict()) == model_hash
            del old
        assert count == 48
        write(REC / 'QUALIFICATION_BARRIER.json', dict(status='pass', files=files,
              qualified_cells=8, actual_complete_GPU_fits=count, formal_predictions_accepted=0,
              GT_read=False, time=time.time()))
    finally:
        if model is not None:
            del model
        gc.collect(); torch.cuda.empty_cache(); lease.close()


def qualification_root():
    setup()
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from vg_tta.stvg_opd_common_state_geometry_v1 import bare, mathematical_readback
    barrier = read(REC / 'QUALIFICATION_BARRIER.json')
    assert barrier['status'] == 'pass' and barrier['actual_complete_GPU_fits'] == 48
    counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    rounds, checks = 0, 0
    for p, h in barrier['files'].items():
        path = ROOT / p
        assert sha(path) == h
        z = load(path)
        assert mathematical_readback(z['fit']) == z['math_audit']['independent']
        checks += z['math_audit']['independent']['scalar_checks']
        rounds += len(z['fit']['rounds'])
        first = load(path.parent / (z['arm'] + '_repeat0.pt'))
        equal(bare(z['fit']), bare(first['fit']), counts)
        equal(z['fit']['geometry'], first['fit']['geometry'], counts)
        if z['original_qualified_cell']:
            q = load(BASE / 'qualification' / f"{z['index']:05}" / (z['arm'] + '.pt'))
            equal(bare(z['fit']), bare(q['fit']), counts)
            equal(z['fit']['geometry'], q['fit']['geometry'], counts)
    # All accepted original fits are independently rehashed; none are overwritten.
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    for r in capture['accepted_prefix']:
        p = ROOT / r['path']
        assert sha(p) == r['sha256'] and sha(p.with_suffix('.json')) == r['receipt_sha256']
    write(REC / 'ROOT_MEMORY_QUALIFICATION_READBACK.json', dict(status='pass',
          scope='eight actual fixed cells, both sources, three arms, two full repeats; failed cell original Full only',
          actual_complete_GPU_fits=48, complete_rounds=rounds, independent_scalar_checks=checks,
          complete_comparison=counts, accepted_prefix_rehashed=len(capture['accepted_prefix']),
          failed_memory_bitwise_claim=False, formal_predictions_accepted=0, GT_read=False,
          future_OOM_safety_claim=False, time=time.time()))


def formal():
    setup()
    assert read(REC / 'ROOT_MEMORY_QUALIFICATION_READBACK.json')['status'] == 'pass'
    target.gpu_run(False)


def controller():
    for job in ('qualification', 'qualification_root', 'formal'):
        env = {**os.environ, 'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
               'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        if job == 'qualification_root':
            env['CUDA_VISIBLE_DEVICES'] = ''
        else:
            env.pop('CUDA_VISIBLE_DEVICES', None)
        with (REC / (job + '.log')).open('a') as log:
            p = subprocess.Popen([str(PYTHON), '-B', str(Path(__file__)), job], cwd=ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            status(BASE / 'CONTROLLER_STATUS.json', dict(status='running', stage=job,
                   controller_pid=os.getpid(), worker_pid=p.pid, GT_read=False, time=time.time()))
            assert p.wait() == 0, 'Actual lifetime010 failure preserved'
    assert read(BASE / 'GLOBAL_PREDICTION_BARRIER.json')['status'] == 'sealed'
    target.resume_original()
    status(BASE / 'CONTROLLER_STATUS.json', dict(status='deployments_sealed_original_baseline_resumed',
           controller_pid=os.getpid(), GT_read=False, time=time.time()))


if __name__ == '__main__':
    job = sys.argv[1]
    try:
        {'qualification': qualification, 'qualification_root': qualification_root,
         'formal': formal, 'controller': controller}[job]()
    except BaseException:
        p = REC / 'failures' / str(time.time_ns())
        p.mkdir(parents=True, exist_ok=True)
        (p / 'traceback.txt').write_text(traceback.format_exc())
        status(BASE / 'CONTROLLER_STATUS.json', dict(status='failed_preserved_pending_actual_root',
               stage=job, evidence=str(p.relative_to(ROOT)), experiment_complete=False, time=time.time()))
        raise

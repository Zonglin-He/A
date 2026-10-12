"""CPU-only actual first resumed records; no task GT or model execution."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha, load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
from vg_tta.stvg_opd_common_state_geometry_v1 import bare
from vg_tta.stvg_opd_common_state_geometry_arithmetic004 import mathematical_readback
BASE = ROOT / 'artifacts/stvg_opd_optimization_geometry_v1'
REC = BASE / 'frozen_visual_host011'


def run():
    import torch
    assert not torch.cuda.is_initialized()
    runtime = read(REC / 'REVISION_RUNTIME.json')
    for p, h in runtime['pins'].items():
        assert sha(ROOT / p) == h, p
    assert read(REC / 'ROOT_MEMORY_QUALIFICATION_READBACK.json')['status'] == 'pass'
    cohort = read(BASE / 'COMMON_STATE_COHORT.json')['records']
    counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    private, rounds, checks, prediction_bytes, input_bytes = [], 0, 0, 0, 0
    for index in (1001, 1002):
        r = cohort[index]
        p = ROOT / r['original_Full']['path']
        ip = ROOT / r['original_Full']['input_path']
        assert sha(p) == r['original_Full']['sha256']
        assert sha(ip) == r['original_Full']['input_sha256']
        old = load(p)
        input_bytes += ip.stat().st_size
        for arm in ('Full', 'Direct', 'Direct_capped'):
            path = BASE / 'common_state' / f'{index:05}' / (arm + '.pt')
            receipt = read(path.with_suffix('.json'))
            assert sha(path) == receipt['sha256'] and not receipt['GT_read']
            z = load(path)
            assert z['execution_revision_runtime_sha256'] == sha(REC / 'REVISION_RUNTIME.json')
            assert z['input_sha256'] == r['original_Full']['input_sha256']
            assert z['common_initial_original_Full_sha256'] == r['original_Full']['sha256']
            assert z['runtime_sha256'] == sha(BASE / 'GPU_RUNTIME.json') and z['Adam_reset']
            equal(z['fit']['initial'], old['fit']['initial'], counts)
            if arm == 'Full':
                equal(bare(z['fit']), bare(old['fit']), counts)
            if index == 1001:
                q = load(REC / 'qualification' / f'{index:05}' / (arm + '_repeat0.pt'))
                equal(bare(z['fit']), bare(q['fit']), counts)
                equal(z['fit']['geometry'], q['fit']['geometry'], counts)
            math = mathematical_readback(z['fit'])
            assert math == z['math_audit']['independent']
            checks += math['scalar_checks']
            rounds += len(z['fit']['rounds'])
            prediction_bytes += path.stat().st_size
            assert sha(path) == receipt['sha256'] and sha(ip) == r['original_Full']['input_sha256']
            private.append(dict(index=index, arm=arm, prediction_sha256=receipt['sha256'],
                                input_sha256=r['original_Full']['input_sha256']))
    pause = read(BASE / 'BASELINE_PAUSE_RECEIPT.json')
    from scripts.pause_stvg_opd_for_geometry_v1 import process
    for role in ('worker', 'controller'):
        actual = process(pause[role]['pid'])
        assert actual['state'] == 'T' and actual['start_ticks'] == pause[role]['start_ticks']
    for p, h in pause['protected'].items():
        assert sha(ROOT / p) == h
    assert not torch.cuda.is_initialized()
    write(REC / 'ROOT_RESUMPTION_READBACK.json', dict(status='pass',
          scope='first two newly resumed Vid cells, six complete fits; failed-cell qualified/formal complete pairing',
          complete_fits=6, qualified_formal_pairs=3, complete_rounds=rounds,
          state_coordinate_steps=rounds * 1792, independent_scalar_checks=checks,
          original_accepted_prefix=3003, new_prediction_bytes=prediction_bytes, input_bytes=input_bytes,
          complete_comparison=counts, second_prediction_input_SHA_readback=True,
          original_baseline_same_processes_and_protected_source=True, GT_read=False,
          CPU_only=True, new_model_calls=0, O_phase_complete=False, all_tasks_complete=False,
          records=private, time=time.time()))
    print('VISUAL_HOST011_FIRST_FORMAL_ROOT_PASS', 6, rounds, checks, prediction_bytes, flush=True)


if __name__ == '__main__':
    run()

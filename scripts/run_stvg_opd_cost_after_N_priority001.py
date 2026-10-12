"""Original32 audited Full timings after genuine C/N closure; cancelled S stays T."""
import gc
import os
import sys
import time
import weakref
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, sha
from scripts.run_stvg_opd_expanded_after_p8_priority001 import stopped, gpu as priority_gpu
SCIENCE = ROOT/'artifacts/stvg_opd_supplement_v1/cost'
BASE = SCIENCE/'after_N_priority001'
ORIGINAL = ROOT/'scripts/run_stvg_opd_end_to_end_cost_supplement_v1.py'
EXPERTS = []


def completed_parents():
    for parent in ['stvg_opd_claim_strengthening_v1', 'stvg_opd_mechanism_review_v1']:
        z = read(ROOT/'artifacts'/parent/'FINAL_COMPLETION.json')
        assert z['status'] == 'complete'
        assert all(z[k] for k in ['actual_root', 'actual_PDF_pixel_view', 'actual_remote_publication', 'actual_archive'])
    assert read(ROOT/'artifacts/stvg_opd_supplement_v1/baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json')['status'] == 'remaining_formal_cancelled_by_explicit_human_instruction'
    stopped()
    return True


def verify():
    from scripts.run_stvg_opd_end_to_end_cost_supplement_v1 import verify as original_verify
    original_verify()
    completed_parents()
    r = read(BASE/'REVISION_RUNTIME.json')
    for p, h in r['pins'].items():
        assert sha(ROOT/p) == h, p
    assert sha(SCIENCE/'DESIGN_LOCK.json') == r['design_sha256']
    return r


def contracts():
    import torch
    from types import SimpleNamespace
    from scripts.run_stvg_opd_deployment_priority002 import stored_capture
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    checks = coordinates = 0
    for dtype in [torch.float32, torch.float64]:
        module = torch.nn.Linear(3, 2).to(dtype).eval().requires_grad_(False)
        expert = SimpleNamespace(model=torch.nn.Linear(3, 2).to(dtype).eval().requires_grad_(False))
        original = {n: v.clone() for n, v in expert.model.state_dict().items()}
        frames = torch.arange(12, dtype=dtype).reshape(4, 3)
        row = {'ordinal': 0}
        expected = module(frames).clone()
        def forward(m, f, r):
            assert m is module and f is frames and r is row
            assert all(not p.is_cuda for p in expert.model.parameters())
            return m(f)
        actual = stored_capture(forward, module, frames, row, expert)
        assert torch.equal(actual, expected)
        assert all(torch.equal(original[n], v) for n, v in expert.model.state_dict().items())
        checks += 3
        coordinates += actual.numel()+sum(v.numel() for v in original.values())
    assert not torch.cuda.is_initialized()
    write(BASE/'CPU_STORAGE_CONTRACTS.json', dict(status='pass', scope='synthetic exact optional inactive expert storage/forward identity; not actual32 timings or future OOM guarantee',
        contracts=checks, tensor_coordinates=coordinates, CUDA_initialized=False, actual_GPU_fits=0,
        actual_DINO_calls=0, CPU_only=True, GT_read=False, time=time.time()))


def run(ds):
    verify()
    from scripts import run_decota_paper_main_v1 as native
    native.gpu = priority_gpu
    from vg_tta.stvg_opd_frozen_mask_cache012 import frozen_forward as memory_capture
    from scripts import run_spatial_ssl_gpu_v1 as capture_module
    from methods.decota_final_simplified_v1 import observations as expert_module
    from scripts.run_stvg_opd_deployment_priority002 import stored_capture
    original_expert = expert_module.SpatialExpert
    class ResidentExpert(original_expert):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            EXPERTS.append(weakref.ref(self))
    expert_module.SpatialExpert = ResidentExpert
    def capture(model, frames, row):
        stopped()
        alive = [r() for r in EXPERTS if r() is not None]
        assert len(alive) == 1
        return stored_capture(memory_capture, model, frames, row, alive[0])
    capture_module.frozen_forward = capture
    text = ORIGINAL.read_text()
    dependency = "assert read(ROOT/'artifacts/stvg_opd_supplement_v1/baselines/FORMAL_GLOBAL_BARRIER.json')['status']=='sealed'"
    assert text.count(dependency) == 1
    ns = {'__file__': str(ORIGINAL), '__name__': 'cost_after_N_priority001', 'completed_parents': completed_parents}
    exec(compile(text.replace(dependency, 'assert completed_parents()'), str(ORIGINAL), 'exec'), ns)
    ns['execute'](ds)
    verify()
    write(BASE/'worker_exit'/f'{ds}.json', dict(status='pass', scope='original16 full audited timed fits/dataset; exact old scientific fit/fresh detector/input/native/math checks; all storage overhead included in native prefix timer',
        dataset=ds, fits=16, original_source_sha256=sha(ORIGINAL), CPU_only=False, GT_read=False, pid=os.getpid(), time=time.time()))


if __name__ == '__main__':
    if sys.argv[1] == 'cpu_contract':
        contracts()
    else:
        try:
            run(sys.argv[1])
        except BaseException:
            import traceback
            p = BASE/'failures'/str(time.time_ns())
            p.mkdir(parents=True, exist_ok=False)
            (p/'traceback.txt').write_text(traceback.format_exc())
            status(BASE/'STATUS.json', dict(status='failed_preserved_pending_actual_root', evidence=str(p.relative_to(ROOT)), CPU_only=False, GT_read=False, time=time.time()))
            raise

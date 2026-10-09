"""Synthetic CPU precision contracts and rejection controls; no GT or CUDA."""
import copy, json, os, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import load
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha
from vg_tta.decota_spatial_opd_reward_audit_revision005 import reward_check, audit, THRESHOLD
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
REC = BASE/'recovery/P1_reward_precision_005'


def run():
    rng = np.random.default_rng(20261009); cases = 0; rejections = 0
    for scale in (.01, .1, 1., 5., 20.):
        for width in (.0001, .001, .1, .9):
            x = rng.normal(size=(3, 32, 4)).astype(np.float32)*np.float32(scale)
            e = np.array([[.5, .5, width, width], [.95, .05, width, width], [.1, .9, .9, .5]], np.float32)
            saved = aligned_iou(torch.tensor(x).sigmoid(), torch.tensor(e)[:, None, :]).numpy()
            reward_check(x, e, saved); cases += 1
            if rejections < 16:
                wrong = saved.astype(float); wrong[0, 0] += 2*THRESHOLD
                try: reward_check(x, e, wrong)
                except AssertionError: rejections += 1
                else: raise AssertionError('Incorrect reward was accepted')
    failed = load(REC/'FAILED_FIT.pt'); start = time.perf_counter(); checked = audit(failed['fit'], failed['expert'])
    elapsed = time.perf_counter()-start
    assert checked['status'] == 'pass' and not checked['original_absolute_cross_precision_reward_audit_passed']
    # A saved fit with a corrupted reward must fail without changing its weights.
    bad = copy.deepcopy(failed['fit']); bad['rounds'][8]['rollout']['rewards'][0, 0] += 1e-3
    try: audit(bad, failed['expert'])
    except AssertionError: rejections += 1
    else: raise AssertionError('Corrupted saved fit was accepted')
    assert cases == 20 and rejections == 17 and not torch.cuda.is_initialized()
    write(REC/'CPU_CONTRACTS.json', dict(status='pass', scope='synthetic reward arithmetic/rejection and saved complete-fit CPU diagnosis only',
        valid_synthetic_reward_contracts=cases, erroneous_reward_rejections=rejections,
        CUDA_initialized=False, GPU_qualified=False, new_predictions=0, GT_read=False,
        threshold_unchanged=THRESHOLD, failed_fit_sha256=sha(REC/'FAILED_FIT.pt'), time=time.time()))
    write(REC/'CPU_DIAGNOSIS.json', dict(status='pass', scope='no-GT saved full-fit independent reward/softmax/gradient/Adam/chart diagnosis',
        actual_failed_fit_serialized_before_exit=True, failed_fit_sha256=sha(REC/'FAILED_FIT.pt'),
        independent_math=checked, CPU_seconds=elapsed, GT_read=False, time=time.time()))
    print(json.dumps(dict(status='pass', contracts=cases, rejected=rejections, GT_read=False)))


if __name__ == '__main__': run()

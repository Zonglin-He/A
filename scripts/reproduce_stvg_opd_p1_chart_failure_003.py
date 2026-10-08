"""One exact first-missing fit, preserving live state before chart failure.

The prior dead worker did not serialize this fit. This is a deterministic
reproduction, not a comparison with its inaccessible memory. No prediction is
accepted and no chart, optimizer, original code or scientific lock is changed.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, read, write, sha, activate
from scripts.decota_matrix_common_v1 import save
from scripts.run_stvg_opd_p1_softmax_recovery_002 import verify_revision
REC = BASE / 'recovery/P1_box_chart_003'


class CapturedChartFailure(Exception):
    pass


def main():
    verify_revision()
    capture = read(REC / 'CAPTURE_RECEIPT.json')
    assert capture['done'] == 1882 and capture['query_ordinal'] == 10220
    runtime = read(REC / 'REPRODUCTION_RUNTIME.json')
    for f, digest in runtime['pins'].items():
        assert sha(ROOT / f) == digest
    for f, item in capture['original_code'].items():
        assert sha(ROOT / f) == item['sha256']
    assert not (REC / 'REPRODUCED_CHART_FAILURE.pt').exists()
    activate()
    import torch
    import vg_tta.decota_spatial_opd_tunable_v1 as core
    from methods.decota_final_simplified_v1.tensors import detached
    original_values = core.TrickReplay.values
    original_mean = core.mean_coordinates

    def values(replay):
        outputs = []
        hook = replay.base.decoder.decoder.bbox_embed.register_forward_hook(
            lambda module, args, output: outputs.append(output))
        try:
            value = original_values(replay)
        finally:
            hook.remove()
        assert len(outputs) == 12
        raw = torch.stack([outputs[5 + (i % 2) * 6][i // 2, 0]
                           for i in range(len(replay.ids))]).float()
        assert torch.equal(raw.sigmoid(), value['boxes'])
        replay._diagnostic_raw_logits = raw
        return value

    def mean(boxes):
        try:
            return original_mean(boxes)
        except AssertionError:
            fit_frame = sys._getframe(1)
            assert fit_frame.f_code.co_filename.endswith('decota_spatial_opd_tunable_v1.py')
            fit = fit_frame.f_locals
            runner = fit_frame.f_back
            while runner is not None and not runner.f_code.co_filename.endswith('run_stvg_opd_paper_v1.py'):
                runner = runner.f_back
            assert runner is not None
            loc = runner.f_locals
            assert loc['stage_name'] == 'P1_vidstg' and loc['order'] == 'order1'
            assert loc['at'] == 1882 and loc['q'] == 10220
            replay = fit['rp']
            record = dict(stage=loc['stage_name'], order=loc['order'], arrival=loc['at'],
                query_ordinal=loc['q'], input=loc['inputrc'],
                previous_payload_sha256=loc['prevhash'][loc['arm']],
                config=fit['cfg'], failed_round=fit['k'], positions=fit['positions'],
                chart_boxes=detached(boxes, 'cpu'),
                native_pre_sigmoid_logits=detached(replay._diagnostic_raw_logits, 'cpu'),
                failed_state=detached(replay.state(), 'cpu'),
                initial=detached(fit['origin'], 'cpu'),
                path=detached(fit['path'], 'cpu'), rounds=detached(fit['audits'], 'cpu'),
                optimizer_state=detached(fit['opt'].state_dict(), 'cpu'),
                expert=detached(loc['ex'], 'cpu'), frame_ids=loc['row']['frame_ids'],
                row_key=loc['row']['key'], interval=loc['native']['physical_interval'],
                original_failed_fit_not_serialized=True, GT_read=False,
                new_prediction_saved=False, capture_is_reproduction=True)
            if 'mu' in fit:
                record['last_mean_before'] = detached(fit['mu'], 'cpu')
            save(REC / 'REPRODUCED_CHART_FAILURE.pt', record)
            (REC / 'ORIGINAL_CHART_REPRODUCTION_TRACEBACK.txt').write_text(traceback.format_exc())
            bad = (~torch.isfinite(boxes)) | (boxes <= 0) | (boxes >= 1)
            write(REC / 'REPRODUCTION_RECEIPT.json', dict(
                status='original_chart_failure_reproduced', stage=record['stage'],
                order=record['order'], arrival=record['arrival'], query_ordinal=record['query_ordinal'],
                failed_round=record['failed_round'], bad_coordinates=bad.nonzero().cpu().tolist(),
                chart_boxes_all_finite=bool(torch.isfinite(boxes).all()),
                raw_logits_all_finite=bool(torch.isfinite(replay._diagnostic_raw_logits).all()),
                failed_state_sha256=sha(REC / 'REPRODUCED_CHART_FAILURE.pt'),
                exact_saved_prefix_previous_sha256=record['previous_payload_sha256'],
                input=record['input'], original_failed_fit_not_serialized=True,
                equality_with_dead_memory_not_claimed=True, prediction_saved=False,
                GT_read=False, runtime_sha256=sha(REC / 'REPRODUCTION_RUNTIME.json'), time=time.time()))
            raise CapturedChartFailure()

    core.TrickReplay.values = values
    core.mean_coordinates = mean
    from scripts.run_stvg_opd_paper_hc2_revision_v2 import run
    try:
        run('GPU', 'P1_vidstg')
    except CapturedChartFailure:
        print('ACTUAL_CHART_FAILURE_REPRODUCED_FROM_FULL_PREFIX_NO_PREDICTION_ACCEPTED', flush=True)
    finally:
        core.TrickReplay.values = original_values
        core.mean_coordinates = original_mean


if __name__ == '__main__':
    main()

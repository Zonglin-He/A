"""Synthetic integration/skip contracts, not a real-data TTA experiment."""
from pathlib import Path
import hashlib, json
import numpy as np
import torch
from check_regression_adapter_cpu import Synthetic, snapshot, assert_state
from regression_adapter import StepConfig
from camera_projection import CameraSpec
from v1_pipeline import run_episode


class Tokens:
    def batch_encode_plus(self, text, **kwargs):
        assert text == ['the white car']
        return {'offset_mapping': torch.tensor([[[0, 0], [0, 3], [4, 9], [10, 13]]])}


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(2)
    torch.manual_seed(24)
    model = Synthetic()
    model.tokenizer = Tokens()
    points = torch.randn(1, 7, 3) * .1
    points[:, :, 2] += 10
    rgb = np.full((200, 200, 3), 255, np.uint8)
    k = np.array([[100., 0, 100.], [0, 100., 100.], [0, 0, 1.]])
    camera = CameraSpec('quad', k, np.eye(4), np.zeros(5))
    # This numeric configuration belongs to this synthetic fixture only.
    config = StepConfig(steps=3, lr=.01, beta_metres=1.)
    original = snapshot(model)
    result = run_episode(model=model, points_model=points, rgb=rgb,
                         actual_model_text='the white car', camera=camera, step_config=config)
    assert_state(model, original)
    for mode in ('rgb_text', 'pc_only', 'shuffled_color'):
        out = result['modes'][mode]
        assert out['pseudo']['query_index'] == result['query_index']
        assert not out['pseudo']['reliability6'].any(), 'Seven points cannot meet twelve-point minimum'
        assert torch.equal(out['adaptation']['source'], out['adaptation']['parameter_SGD'])
    missing = CameraSpec('quad', k, np.eye(4))
    bad_camera = run_episode(model=model, points_model=points, rgb=rgb,
                             actual_model_text='the white car', camera=missing, step_config=config)
    assert all('camera_contract_rejected' in out['adaptation']['skip_reason'] for out in bad_camera['modes'].values())
    with torch.no_grad():
        model.prediction_heads[5].size_pred_head.net[8].weight.zero_()
        model.prediction_heads[5].size_pred_head.net[8].bias.fill_(-1.)
    bad_size = run_episode(model=model, points_model=points, rgb=rgb,
                           actual_model_text='the white car', camera=camera, step_config=config)
    assert all(out['adaptation']['skip_reason'] == 'nonpositive_source_size' for out in bad_size['modes'].values())
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parent
    files = ['v1_pipeline.py', 'camera_projection.py', 'multimodal_pseudo.py', 'regression_adapter.py', Path(__file__).name]
    receipt = {'state': 'synthetic_end_to_end_contracts_passed_no_real_model_or_task_score',
               'all3_G_modes_wired_to_same_source_and_adapter': True,
               'insufficient_evidence_returns_source_without_update': True,
               'missing_distortion_without_explicit_assumption_fails_closed': True,
               'nonpositive_source_size_never_silently_repaired': True,
               'model_restored_after_all_control_arms': True,
               'synthetic_step_config_not_an_experiment_configuration': True,
               'cuda_initialized': False,
               'source_sha256': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}}
    (root/'CPU_PIPELINE_FIXTURE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

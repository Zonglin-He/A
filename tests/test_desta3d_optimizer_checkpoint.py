from __future__ import annotations

import copy
import json
import torch
import pytest

from vg_tta.optimizer_checkpoint import (
    cpu_clone, restore_optimizer, validate_serialized_optimizer, assert_nested_equal,
)
from scripts import desta3d_v2_aux_backflow_recovery_v2 as recovery


def _step(model, optimizer, turn):
    for index, parameter in enumerate(model.parameters()):
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, ((index % 7) - 2.7) * .003 + turn * .0004)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_integer_state_keys_and_independent_cpu_storage():
    parameter = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([parameter])
    parameter.grad = torch.full_like(parameter, .3)
    opt.step()
    stored = cpu_clone(opt.state_dict())
    validate_serialized_optimizer(stored)
    assert list(stored['state']) == [0]
    opt.state[parameter]['exp_avg'].add_(1)
    assert not torch.equal(stored['state'][0]['exp_avg'], opt.state[parameter]['exp_avg'])


def test_legacy_string_id_is_rejected_and_reproduces_old_reset():
    parameter = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.AdamW([parameter], lr=.01)
    for turn in range(3):
        parameter.grad = torch.full_like(parameter, .1 + turn)
        opt.step(); opt.zero_grad(set_to_none=True)
    saved = cpu_clone(opt.state_dict())
    bad = copy.deepcopy(saved)
    bad['state'] = {str(k): v for k, v in bad['state'].items()}
    replacement = torch.nn.Parameter(parameter.detach().clone())
    broken = torch.optim.AdamW([replacement], lr=.01)
    with pytest.raises(ValueError, match='integer parameter IDs'):
        restore_optimizer(broken, bad)
    broken.load_state_dict(bad)
    assert replacement not in broken.state
    replacement.grad = torch.full_like(replacement, -.3)
    broken.step()
    assert int(broken.state[replacement]['step']) == 1


def test_actual_adapter_three_arm_resume_next_two_updates_exact(tmp_path, monkeypatch):
    """Use real 128-channel adapter/AdamW; only device/RNG APIs are CPU mocks.

    Unlike the superseded round-trip test, compare actual parameter-bound
    moments and TWO subsequent updates to an uninterrupted control.
    """
    worker = recovery.worker
    torch.set_num_threads(2)
    out = tmp_path/'fit'
    out.mkdir(); (out/'LOCK.json').write_text('{"synthetic":true}\n')
    monkeypatch.setattr(recovery, 'OUT', out)
    monkeypatch.setattr(worker, 'OUT', out)
    monkeypatch.setattr(worker.Desta3DAdapterV2, 'cuda', lambda self: self)
    cuda_rng = [torch.tensor([11, 19], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: cpu_clone(cuda_rng))
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda state: cuda_rng.__setitem__(slice(None), cpu_clone(state)))
    adapters, optimizers = {}, {}
    torch.manual_seed(99)
    for arm in worker.MODES:
        model = worker.Desta3DAdapterV2(hidden_dim=128, architecture='dual3d', p1_enabled=False)
        opt = worker.make_source_optimizer(model, 'repaired', 'B')
        for turn in range(3):
            _step(model, opt, turn)
        adapters[arm], optimizers[arm] = model, opt
    state = {'stage': 'B', 'cursor': 12, 'steps': dict.fromkeys(worker.MODES, 3),
             'last_window': {'stage': 'B', 'cursor': 12, 'synthetic': True}}
    recovery.save_state(state, adapters, optimizers)
    wanted_rng = torch.get_rng_state().clone()
    torch.manual_seed(777)
    restored_state, restored_adapters, restored_optimizers = recovery.restore()
    assert restored_state == state and torch.equal(torch.get_rng_state(), wanted_rng)
    assert json.loads((out/'history/B_C0012.json').read_text()) == state['last_window']
    for arm in worker.MODES:
        assert_nested_equal(cpu_clone(optimizers[arm].state_dict()),
                            cpu_clone(restored_optimizers[arm].state_dict()))
        assert all(isinstance(k, torch.nn.Parameter) for k in restored_optimizers[arm].state)
    for turn in (3, 4):
        for arm in worker.MODES:
            _step(adapters[arm], optimizers[arm], turn)
            _step(restored_adapters[arm], restored_optimizers[arm], turn)
            assert_nested_equal(cpu_clone(adapters[arm].state_dict()), cpu_clone(restored_adapters[arm].state_dict()))
            assert_nested_equal(cpu_clone(optimizers[arm].state_dict()), cpu_clone(restored_optimizers[arm].state_dict()))
            assert all(int(s['step']) == turn + 1 for s in restored_optimizers[arm].state.values())


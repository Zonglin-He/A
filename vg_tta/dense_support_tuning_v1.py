"""F40 portable exact temporal-head replay. No labels or metric-based episode choices.

Only the actual two-layer TA-STVG temp_embed is trainable. Frozen inputs are
captured at its true FP32 input; source weights and full-model outputs are checked
by the driver. F39 loss, optimizer, actual proposal backtracking and MAP stay intact.
"""
import hashlib
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def move(x, device):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(device)
    if isinstance(x, list):
        return [move(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move(v, device) for v in x)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    return x


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class HeadReplay:
    def __init__(self, cache, device='cuda'):
        # Use the actual official MLP implementation, including eval-mode dropout.
        spec = importlib.util.spec_from_file_location(
            'f40_tastvg_net_utils', ROOT/'external/TA-STVG/models/net_utils.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.head = module.MLP(256, 256, 2, 2, dropout=.3).to(device).float().eval()
        state = cache['head_state']
        self.head.load_state_dict({n.removeprefix('head.'): v for n, v in state.items()})
        self.named = [('head.'+n, p) for n, p in self.head.named_parameters()]
        self.initial = self.state()
        self.inputs = move(cache['inputs'], device)
        self.zero = move(cache['zero'], device)
        self.teacher = move(cache['teacher'], device)
        self.records, self.ids = cache['records'], cache['frame_ids']
        assert sum(p.numel() for _, p in self.named) == 66306
        assert all(not h.requires_grad for h in self.inputs)
        with torch.no_grad():
            values = self.values()
        assert all(torch.equal(z, q) for z, q in zip(values['logits'], self.zero['logits']))

    def state(self):
        return {n: p.detach().clone() for n, p in self.named}

    def restore(self, state):
        with torch.no_grad():
            for n, p in self.named:
                p.copy_(state[n].to(p))
                p.grad = None

    def values(self):
        return {**self.zero, 'logits': [self.head(h)[-1] for h in self.inputs]}


def compact_fit(result):
    """Keep actual final and initial weights, all logits/losses and step state hashes.

    This only compresses saved artifacts after optimization. It never changes
    optimization, acceptance or selection. Full intermediate states are reproducible.
    """
    result['initial_state'] = result['path'][0]['state']
    for step in result['path']:
        step['state_sha256'] = state_hash(step['state'])
        del step['state']
    result['final'] = result['path'][result['best_step']]
    result['state_sha256'] = state_hash(result['state'])
    assert result['state_sha256'] == result['final']['state_sha256']
    return result

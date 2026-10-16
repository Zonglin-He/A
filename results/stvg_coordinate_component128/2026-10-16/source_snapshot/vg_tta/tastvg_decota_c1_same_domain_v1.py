"""Existing C1 replay at the already-normalized, Frozen encoder boundary."""
import copy
import torch
from methods.decota_final_simplified_v1.replay import SpatialReplay
from methods.decota_final_simplified_v1.tensors import detached, floating32


class NormalizedSpatialReplay(SpatialReplay):
    def __init__(self, model, data):
        self.model, self.n, self.cached = model, len(data['frame_ids']), True
        # Identity is exact: cached H is the output of the original frozen LN.
        self.views = floating32([{**v, 'prefix': v['H']} for v in data['views']])
        self.norm = torch.nn.Identity()
        self.decoder = copy.deepcopy(model.ground_decoder).eval().requires_grad_(False)
        self.head = copy.deepcopy(model.temp_embed).float().eval().requires_grad_(False)
        self.delta = torch.zeros(256, device=next(model.parameters()).device, requires_grad=True)
        self.named = [('spatial.query_residual', self.delta)]
        for ln in ('norm1', 'norm3', 'norm4'):
            for name, param in getattr(self.decoder.decoder.layers[5], ln).named_parameters():
                param.requires_grad_(True)
                self.named.append((f'spatial.layers.5.{ln}.{name}', param))
        assert sum(p.numel() for _, p in self.named) == 1792
        self.initial = self.state()
        self.spatial_inputs, self.temporal_inputs = [], []
        with torch.no_grad():
            self.zero = detached(self.reference_values(capture=True))
            assert torch.equal(self.values()['boxes'], self.zero['boxes'])
        assert len(self.spatial_inputs) == len(self.temporal_inputs) == 2
        expected = data['prediction']
        assert torch.equal(self.zero['boxes'].cpu(), expected['boxes'].cpu()), 'C1 native box parity'
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(self.zero['logits'], expected['logits'])), 'C1 native time parity'


def configuration(dataset):
    from dataclasses import replace
    from methods.decota_final_simplified_v1.config import MethodConfig
    vid = MethodConfig.for_direction('vid_to_hc1')
    hc = MethodConfig.for_direction('hc2_to_vid')
    target, source = (hc, vid) if dataset == 'vidstg' else (vid, hc)
    return replace(target, direction=f'{dataset}_within_domain_C1_Scale06',
        spatial_lr=.03, source_dataset=source.source_dataset,
        checkpoint=source.checkpoint, checkpoint_sha256=source.checkpoint_sha256)


"""Exact frozen-prefix factorization for the fixed, private update interface.

The reference path recomputes both native decoder passes. The cached path only
recomputes the second PosDecoder, INCLUDING all six spatial blocks and every
iterative detached reference. It never freezes a query-dependent spatial prefix.
"""

import copy
import torch

from .tensors import ParameterState, detached, floating32


class SpatialReplay(ParameterState):
    def __init__(self, model, views, n, *, cached=True):
        if len(views) != 2:
            raise ValueError("Expected the two original offsets")
        self.model, self.n, self.cached = model, n, cached
        self.views = floating32(views)
        self.norm = copy.deepcopy(model.ground_encoder.encoder.norm).eval().requires_grad_(False)
        self.decoder = copy.deepcopy(model.ground_decoder).eval().requires_grad_(False)
        self.head = copy.deepcopy(model.temp_embed).float().eval().requires_grad_(False)
        self.delta = torch.zeros(256, device=next(self.norm.parameters()).device, requires_grad=True)
        self.named = [("spatial.query_residual", self.delta)]
        last = len(self.decoder.decoder.layers) - 1
        for layer_norm in ("norm1", "norm3", "norm4"):
            for name, param in getattr(self.decoder.decoder.layers[last], layer_norm).named_parameters():
                param.requires_grad_(True)
                self.named.append((f"spatial.layers.{last}.{layer_norm}.{name}", param))
        if sum(p.numel() for _, p in self.named) != 1792 or last != 5:
            raise RuntimeError("The locked spatial interface is query + three last-block LNs")
        self.initial = self.state()
        self.spatial_inputs, self.temporal_inputs = [], []
        with torch.no_grad():
            self.zero = detached(self.reference_values(capture=True))
        if len(self.spatial_inputs) != 2 or len(self.temporal_inputs) != 2:
            raise RuntimeError("Failed to capture the actual final native interfaces")

    @staticmethod
    def _indices(mask):
        value = torch.nonzero(mask.squeeze()).squeeze().tolist()
        return [value] if isinstance(value, int) else value

    def _queries(self, H, fm, fa, ft, chosen, count):
        with torch.no_grad():
            _, am = self.model.t_spatial_clas(fm[chosen], ft[:, :1])
            _, aa = self.model.s_spatial_clas(fa[chosen], ft[:, :1])
        return ((H[-count:].permute(1, 0, 2)[chosen] * am.unsqueeze(2)).mean((0, 1)),
                (H[:count].permute(1, 0, 2)[chosen] * aa.unsqueeze(2)).mean((0, 1)))

    def reference_values(self, capture=False):
        boxes, logits, actions, gates = [], [], [], []
        with torch.autocast("cuda", enabled=False):
            for view in self.views:
                H = self.norm(view["prefix"])
                info = dict(view["info"])
                info.update(encoded_feature=H, frames_cls=H.mean(0), videos_cls=H.mean(0).mean(0))
                h, w = info["fea_map_size"]
                count, nf = h * w, H.shape[1]
                fm = H[-count:].permute(1, 2, 0).reshape(nf, 256, h, w).detach()
                fa = H[:count].permute(1, 2, 0).reshape(nf, 256, h, w).detach()
                ft = H[count:-count].mean(1).unsqueeze(0).detach()
                with torch.no_grad():
                    lm = self.model.t_temporal_clas(fm, ft)
                    la = self.model.s_temporal_clas(fa, ft)
                    prob = (lm.sigmoid() + la.sigmoid()) / 2
                    fallback = self._indices(prob > 0)
                    first = self._indices(prob > self.model.theta) or fallback
                    itq, isq = self._queries(H, fm, fa, ft, first, count)
                    _, hidden = self.decoder(encoded_info=info, vis_pos=view["vis_pos"], itq=itq, isq=isq)
                    ap = self.model.action_embed(hidden)[-1].squeeze().sigmoid()
                    second = self._indices(ap > .5) or fallback
                itq, isq = self._queries(H, fm, fa, ft, second, count)

                def add_query(module, args, kwargs):
                    if capture:
                        if args:
                            raise RuntimeError("Unexpected positional PosDecoder arguments")
                        self.spatial_inputs.append(detached(kwargs))
                    kwargs = dict(kwargs)
                    q = kwargs["query_tgt"]
                    kwargs["query_tgt"] = q + self.delta.to(q.dtype)[None, None, :]
                    return args, kwargs

                hook = self.decoder.decoder.register_forward_pre_hook(add_query, with_kwargs=True)
                try:
                    pos, hidden = self.decoder(encoded_info=info, vis_pos=view["vis_pos"], itq=itq, isq=isq)
                finally:
                    hook.remove()
                if capture:
                    self.temporal_inputs.append(detached(hidden))
                boxes.append(pos.flatten(1, 2)[-1])
                logits.append(self.head(hidden)[-1])
                actions.append(self.model.action_embed(hidden)[-1])
                gates.append(dict(first=first, second=second))
        return dict(boxes=torch.stack([boxes[i % 2][i // 2] for i in range(self.n)]).float(),
                    logits=logits, actions=actions, gates=gates)

    def values(self):
        if not self.cached:
            return self.reference_values()
        boxes = []
        with torch.autocast("cuda", enabled=False):
            for inputs in self.spatial_inputs:
                kwargs = dict(inputs)
                q = kwargs["query_tgt"]
                kwargs["query_tgt"] = q + self.delta.to(q.dtype)[None, None, :]
                pos = self.decoder.decoder(**kwargs)
                boxes.append(pos.flatten(1, 2)[-1])
        return {**self.zero,
                "boxes": torch.stack([boxes[i % 2][i // 2] for i in range(self.n)]).float()}


class TemporalReplay(ParameterState):
    def __init__(self, source_head, inputs, zero):
        self.head = copy.deepcopy(source_head).float().eval().requires_grad_(True)
        self.named = [("head." + n, p) for n, p in self.head.named_parameters()]
        self.initial = self.state()
        self.inputs, self.zero = detached(inputs), detached(zero)
        if sum(p.numel() for _, p in self.named) != 66306:
            raise RuntimeError("Expected the existing 66306-parameter temp_embed")
        if any(x.requires_grad for x in self.inputs):
            raise RuntimeError("Temporal prefix must be frozen")
        with torch.no_grad():
            values = self.values()
        if any(not torch.equal(a, b) for a, b in zip(values["logits"], zero["logits"])):
            raise RuntimeError("Temporal cached input does not reproduce the native head")

    def values(self):
        with torch.autocast("cuda", enabled=False):
            return {**self.zero, "logits": [self.head(h)[-1] for h in self.inputs]}

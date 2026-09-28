"""ViTTA-episodic (STVG port), isolated from the deployed DeCoTA method.

Contracts follow wlin-at/ViTTA commit c8e01fa63f8a821a2ebdf1f1272872a867b78cdb:
channel-mean L1 mean/variance alignment; undetached view-mean probability L1;
the official ``tta_standard`` branch resets per sample and sets momentum_mvg=1.
The paper/default protocol is continual; this explicitly named port is episodic.

STVG adaptations: decoder parameters only, final-pass last-two-block LayerNorm
features, native interleaved temporal views, physical-time probability alignment,
and eval mode. No source/target localization labels, expert or DeCoTA targets.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from vg_tta.native_probability_interface_v1 import NativeOutput

OFFICIAL_COMMIT = "c8e01fa63f8a821a2ebdf1f1272872a867b78cdb"
DEFAULTS = dict(lr=1e-5, steps=1, momentum=.9, weight_decay=5e-4,
                consistency_weight=.1, momentum_mvg=1.)
FEATURE_LAYERS = tuple(
    f"ground_decoder.{branch}.layers.{layer}.norm{norm}"
    for branch in ("decoder", "time_decoder") for layer in (4, 5)
    for norm in (1, 3, 4)
)
REDUCTION = "equal_video_mean_of_joint_two_view_channel_mean_and_population_variance"


@dataclass(frozen=True)
class View:
    native: NativeOutput
    features: dict[str, Tensor]  # valid positions x channels; final refinement only


def feature_statistics(views):
    """Joint two-view statistics; a population variance, not an unbiased estimate."""
    if len(views) != 2 or not views[0].features:
        raise ValueError("Exactly two nonempty temporal feature views are required")
    keys = set(views[0].features)
    if any(set(v.features) != keys for v in views):
        raise ValueError("Feature layer sets differ between views")
    result = {}
    for name in sorted(keys):
        feats = [v.features[name] for v in views]
        if any(x.ndim != 2 or not x.numel() or not bool(torch.isfinite(x).all()) for x in feats):
            raise ValueError("Invalid/nonfinite feature matrix: " + name)
        if len({x.shape[1] for x in feats}) != 1:
            raise ValueError("Feature channel counts differ: " + name)
        # Native decoder has a single valid frame query at each position.
        if any(x.shape[0] != int(v.native.valid.sum()) for x, v in zip(feats, views)):
            raise ValueError("Feature positions must equal valid temporal positions")
        values = torch.cat(feats, dim=0).float()
        result[name] = dict(mean=values.mean(0), var=values.var(0, unbiased=False))
    return result


class SourceStatistics:
    """Streaming official-code estimator: average within-video moments equally.

    It deliberately does not pool raw positions across videos or add between-video
    variance: corpus/basics.py compute_statistics averages each video's variance.
    """
    def __init__(self):
        self.count = 0
        self.sums = {}

    def add(self, stats):
        if self.count and set(stats) != set(self.sums):
            raise ValueError("Source statistic layers changed")
        for name, values in stats.items():
            mean, var = values["mean"].detach().double().cpu(), values["var"].detach().double().cpu()
            if mean.ndim != 1 or mean.shape != var.shape or not bool(torch.isfinite(mean).all() and torch.isfinite(var).all()) or bool((var < 0).any()):
                raise ValueError("Invalid source moments")
            if name not in self.sums:
                self.sums[name] = dict(mean=torch.zeros_like(mean), var=torch.zeros_like(var))
            if self.sums[name]["mean"].shape != mean.shape:
                raise ValueError("Source channel count changed")
            self.sums[name]["mean"] += mean
            self.sums[name]["var"] += var
        self.count += 1

    def finish(self):
        if not self.count or not self.sums:
            raise ValueError("Cannot finish empty source statistics")
        return {n: {k: (v / self.count).float() for k, v in z.items()} for n, z in self.sums.items()}


def validate_source(source, *, checkpoint_sha256=None, source_dataset=None, layer_names=None):
    required = {"source_dataset", "checkpoint_sha256", "source_manifest_sha256", "official_split",
                "source_count", "query_count", "target_inputs_used", "localization_labels_used", "reduction"}
    p = source.get("provenance", {})
    if required - set(p):
        raise ValueError("Missing source-statistics provenance")
    if p["target_inputs_used"] is not False or p["localization_labels_used"] is not False:
        raise ValueError("ViTTA source statistics cannot use target inputs/localization labels")
    if p["official_split"] != "train" or p["source_count"] <= 0 or p["query_count"] < p["source_count"]:
        raise ValueError("Expected explicitly identified source-training inputs")
    if p["reduction"] != REDUCTION:
        raise ValueError("Source-statistic reduction mismatch")
    for key in ("checkpoint_sha256", "source_manifest_sha256"):
        if len(p[key]) != 64 or any(c not in "0123456789abcdef" for c in p[key]):
            raise ValueError("Invalid source provenance digest: " + key)
    if checkpoint_sha256 is not None and p["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Source statistics belong to a different checkpoint")
    if source_dataset is not None and p["source_dataset"] != source_dataset:
        raise ValueError("Source statistics belong to a different source dataset")
    stats = source.get("statistics", {})
    if not stats or (layer_names is not None and set(stats) != set(layer_names)):
        raise ValueError("Source statistics feature-layer mismatch")
    for name, z in stats.items():
        m, v = z["mean"], z["var"]
        if m.ndim != 1 or m.shape != v.shape or not bool(torch.isfinite(m).all() and torch.isfinite(v).all()) or bool((v < 0).any()):
            raise ValueError("Invalid source moments: " + name)
    return stats


def alignment_loss(target, source):
    if set(target) != set(source):
        raise ValueError("Source/target feature layers differ")
    losses = []
    for name in sorted(target):
        for key in ("mean", "var"):
            t, s = target[name][key], source[name][key]
            if t.shape != s.shape:
                raise ValueError("Source/target channel counts differ")
            losses.append((t - s.to(t)).abs().mean())
    return torch.stack(losses).sum()


def _edges(ids, low, high, reference):
    x = torch.as_tensor(ids, device=reference.device, dtype=reference.dtype)
    if x.numel() < 2 or not bool((x[1:] > x[:-1]).all()):
        raise ValueError("Each temporal view needs at least two increasing valid positions")
    return torch.cat([x.new_tensor([low]), (x[1:] + x[:-1]) / 2, x.new_tensor([high])])


def physical_probabilities(output, full_ids):
    """Map temporal predictions by physical frame IDs, never offset tensor index.

    Endpoint masses are re-binned conservatively from native Voronoi time cells
    into the full-grid cells. Action probabilities are interpolated at physical
    frame centers (nearest-value extension only at the two external boundaries).
    """
    endpoints, action = output.probabilities()
    ids = tuple(i for i, valid in zip(output.frame_ids, output.valid.tolist()) if valid)
    full_ids = tuple(full_ids)
    if len(full_ids) < 4 or full_ids != tuple(sorted(set(full_ids))) or any(i not in full_ids for i in ids):
        raise ValueError("Invalid shared physical grid")
    low = full_ids[0] - (full_ids[1] - full_ids[0]) / 2
    high = full_ids[-1] + (full_ids[-1] - full_ids[-2]) / 2
    src = _edges(ids, low, high, endpoints)
    dst = _edges(full_ids, low, high, endpoints)
    overlap = (torch.minimum(dst[1:, None], src[None, 1:]) -
               torch.maximum(dst[:-1, None], src[None, :-1])).clamp_min(0)
    transfer = overlap / (src[1:] - src[:-1])[None, :]
    rebinned = transfer @ endpoints
    x = endpoints.new_tensor(ids)
    q = endpoints.new_tensor(full_ids).clamp(x[0], x[-1])
    right = torch.searchsorted(x, q).clamp(1, len(x) - 1)
    left = right - 1
    ratio = (q - x[left]) / (x[right] - x[left])
    interpolated = action[left] * (1 - ratio) + action[right] * ratio
    return rebinned, interpolated


def temporal_consistency(views, full_ids):
    if len(views) != 2:
        raise ValueError("Exactly two temporal views required")
    ids = [set(i for i, ok in zip(v.native.frame_ids, v.native.valid.tolist()) if ok) for v in views]
    if ids[0] & ids[1] or ids[0] | ids[1] != set(full_ids):
        raise ValueError("Expected disjoint native offsets covering the full grid")
    mapped = [physical_probabilities(v.native, full_ids) for v in views]
    endpoints = torch.stack([z[0] for z in mapped])
    action = torch.stack([z[1] for z in mapped])
    # Official consistency differentiates through the view mean. Each Bernoulli
    # has two classes; average its class L1 over physical positions.
    return ((endpoints - endpoints.mean(0)).abs().sum((1, 2)).mean()
            + 2 * (action - action.mean(0)).abs().mean())


def _norm(values):
    items = [x.detach().float().square().sum() for x in values if x is not None]
    return torch.stack(items).sum().sqrt() if items else torch.tensor(0.)


@torch.enable_grad()
def adapt(params, closure, final_inference, source, full_ids, config=None):
    """One independent query; closure returns two fresh gradient-connected views."""
    cfg = {**DEFAULTS, **(config or {})}
    if set(cfg) != set(DEFAULTS) or not params or len({id(p) for p in params}) != len(params):
        raise ValueError("Invalid ViTTA configuration/parameter scope")
    if cfg["momentum_mvg"] != 1. or cfg["steps"] < 0 or int(cfg["steps"]) != cfg["steps"]:
        raise ValueError("Episodic ViTTA requires momentum_mvg=1 and integer steps")
    if any(not math.isfinite(float(v)) or v < 0 for v in cfg.values()) or cfg["momentum"] >= 1:
        raise ValueError("Invalid ViTTA optimizer/consistency configuration")
    stats = validate_source(source)
    values = [p.detach().clone() for p in params]
    flags = [p.requires_grad for p in params]
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in params]
    audit = dict(method="ViTTA-episodic (STVG port)", config=cfg, backwards=0,
                 optimizer_steps=0, forward_calls=0, trace=[], gradient_norms=[],
                 unused_parameter_indices=[], GT_online=False, expert_calls=0,
                 target_statistics_reset=True, optimizer_reset=True,
                 official_commit=OFFICIAL_COMMIT, source_provenance=source["provenance"])
    try:
        for p in params:
            p.requires_grad_(True)
        optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
        for step in range(cfg["steps"]):
            optimizer.zero_grad(set_to_none=True)
            views = closure()
            audit["forward_calls"] += 2
            align = alignment_loss(feature_statistics(views), stats)
            consistency = temporal_consistency(views, full_ids)
            loss = align + cfg["consistency_weight"] * consistency
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite/disconnected ViTTA loss")
            loss.backward()
            audit["backwards"] += 1
            norm = _norm([p.grad for p in params])
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("Nonfinite ViTTA gradient")
            audit["gradient_norms"].append(float(norm))
            audit["unused_parameter_indices"].append([i for i, p in enumerate(params) if p.grad is None])
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise RuntimeError("Nonfinite ViTTA parameters")
            audit["optimizer_steps"] += 1
            audit["trace"].append(dict(step=step, alignment=float(align.detach()),
                                       consistency=float(consistency.detach()), loss=float(loss.detach())))
        audit["parameter_delta_l2"] = float(_norm([p - v for p, v in zip(params, values)]))
        audit["parameter_changed"] = any(not torch.equal(p, v) for p, v in zip(params, values))
        # TA-STVG's complete temporal decoder/head selects a numerically
        # different inference path when parameters still require gradients,
        # even inside no_grad (measured no-op drift up to 2.15e-6). Restore the
        # source inference flags, but retain adapted values for this forward.
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)
        audit["final_inference_source_flags"] = True
        with torch.no_grad():
            final = final_inference()
        audit["final_inference_calls"] = 2
        return final, audit
    finally:
        with torch.no_grad():
            for p, value, flag, grad in zip(params, values, flags, gradients):
                p.copy_(value)
                p.requires_grad_(flag)
                p.grad = grad
        audit["source_restored"] = all(torch.equal(p, v) for p, v in zip(params, values))


@contextmanager
def capture_layer_outputs(model, names=FEATURE_LAYERS):
    modules = dict(model.named_modules())
    if set(names) - set(modules):
        raise ValueError("TA-STVG feature hook layers are unavailable")
    captured = {name: [] for name in names}
    hooks = []
    try:
        for name in names:
            if not isinstance(modules[name], torch.nn.LayerNorm):
                raise ValueError("Expected decoder LayerNorm: " + name)
            hooks.append(modules[name].register_forward_hook(
                lambda module, args, output, key=name: captured[key].append(output)))
        yield captured
    finally:
        for hook in hooks:
            hook.remove()


def live_views(model, batch, ids, subject):
    from vg_tta.native_baselines_paper_v1 import live_output
    with capture_layer_outputs(model) as captured:
        native, prediction = live_output(model, batch, ids, subject)
    if any(len(values) != 4 for values in captured.values()):
        raise RuntimeError("Expected two refinement passes per original offset")
    views = []
    for offset in (0, 1):
        features = {}
        for name, values in captured.items():
            value = values[2 * offset + 1]
            if value.ndim != 3 or value.shape[:2] != (len(native[offset].frame_ids), 1):
                raise RuntimeError("Unexpected decoder feature shape: " + name)
            features[name] = value[:, 0, :]
        views.append(View(native[offset], features))
    return views, prediction


def decoder_scope(model):
    # The published method updates its entire action network. This port updates
    # the entire existing query decoder + output heads, retaining frozen prefix
    # encoders and TA-STVG's hard-coded no-grad/detach contracts unchanged.
    selected = [(n, p) for n, p in model.named_parameters()
                if n.startswith(("ground_decoder.", "temp_embed.", "action_embed.", "bbox_embed."))]
    if not selected:
        raise ValueError("Empty TA-STVG decoder scope")
    return [n for n, _ in selected], [p for _, p in selected]


def episode(model, parser, frames, ids, metadata, source, checkpoint_sha256, source_dataset, config=None):
    from methods.decota_final_simplified_v1.backbone import make_batch, validate_input
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.native_baselines_paper_v1 import live_output
    validate_input(frames, ids, metadata)
    validate_source(source, checkpoint_sha256=checkpoint_sha256, source_dataset=source_dataset,
                    layer_names=FEATURE_LAYERS)
    if any(m.training for m in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected fully frozen eval source model")
    before = state_hash(model.state_dict())
    batch = make_batch(frames, ids, metadata, model)
    subject = parser(metadata["caption"])["subject"]
    names, params = decoder_scope(model)
    try:
        with torch.no_grad():
            _, frozen = live_output(model, batch, ids, subject)
        final, audit = adapt(params, lambda: live_views(model, batch, ids, subject)[0],
                             lambda: live_output(model, batch, ids, subject)[1], source, ids, config)
    finally:
        if before != state_hash(model.state_dict()):
            raise RuntimeError("ViTTA failed full source-state restoration")
    audit.update(parameter_names=names, parameter_count=sum(p.numel() for p in params),
                 feature_layers=list(FEATURE_LAYERS), source_state_hash=before,
                 source_state_exact=True, scope="full_query_decoder_and_output_heads",
                 compatibility=["episodic_official_tta_standard_momentum1", "decoder_only",
                                "eval_mode", "native_two_offset_temporal_views", "physical_probability_rebin",
                                "last_two_decoder_blocks_final_refinement", "no_spatial_crop",
                                "native_endpoint_and_actionness_consistency", "full_post_update_forward"])
    return dict(Frozen=frozen, prediction=final, audit=audit)

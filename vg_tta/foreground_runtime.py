"""Versioned, label-free TTS probing. Never modifies the released TA sources."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch


class StopAfterTTS(Exception):
    """Intentional early exit once both relevance heads have executed."""


def capture_tts(model, batch, *, full=False):
    """Cache post-query-conditioning inputs and native logits of both heads.

    Probes stop before ASA; original inference uses the complete native -1
    refinement path. Only explicitly allowlisted non-label metadata is passed.
    """
    from scripts import run_tastvg_span_tta as ta

    device = next(model.parameters()).device
    target = batch["targets"][0]
    allowed = ("item_id", "vid", "frame_ids", "img_size", "ori_size")
    safe_target = {k: copy.deepcopy(target[k]) for k in allowed}
    safe_target["actioness"] = torch.zeros(len(target["frame_ids"]), dtype=torch.int64)
    safe = {"videos": batch["videos"], "texts": list(batch["texts"]),
            "targets": [safe_target], "durations": list(batch["durations"])}
    safe = ta._batch_to_device(safe, device)
    captured, handles = {}, []

    def hook(name):
        def capture(_module, args, output):
            captured[name] = {"features": args[0].detach().float().cpu().reshape(-1, args[0].shape[-1]),
                              "native_logits": output.detach().float().cpu().reshape(-1)}
            if len(captured) == 2 and not full:
                raise StopAfterTTS()
        return capture

    for name, module in (("appearance", model.s_temporal_clas.head),
                         ("motion", model.t_temporal_clas.head)):
        handles.append(module.register_forward_hook(hook(name)))
    outputs = None
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            try:
                outputs = model(safe["videos"], safe["texts"], safe["targets"], iteration_rate=-1)
            except StopAfterTTS:
                if full:
                    raise
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != {"appearance", "motion"}:
        raise RuntimeError("Both pretrained TTS heads must be captured")
    for name, module in (("appearance", model.s_temporal_clas.head),
                         ("motion", model.t_temporal_clas.head)):
        features = captured[name]["features"].to(device)
        # Teacher and adapted heads use FP32. The native boxes and endpoint
        # baseline retain official FP16 inference. Expose numerical difference.
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            captured[name]["logits"] = module(features).reshape(-1).float().cpu()
        captured[name]["fp32_native_max_abs"] = float(
            (captured[name]["logits"] - captured[name]["native_logits"]).abs().max())
    return outputs, captured


def merge_tts(records, total_frames):
    merged = {}
    for branch in ("appearance", "motion"):
        width = records[0][branch]["features"].shape[-1]
        merged[branch] = {"features": torch.empty(total_frames, width),
                          "logits": torch.empty(total_frames),
                          "native_logits": torch.empty(total_frames)}
        for offset, record in enumerate(records):
            for field in ("features", "logits", "native_logits"):
                merged[branch][field][offset::2] = record[branch][field]
    return merged


def erased_batch(batch, boxes_abs, *, random_seed=None):
    """Replace each B0 region by its per-channel mean (constant local fill).

    Random control reflects box positions (not video pixels) about a fixed
    image axis. Rectangular shape and area are exact; center motion magnitude
    is preserved, without wraparound or clipping. Choose randomly among axes
    with mean mask IoU <= .5, or the least-overlapping axis if none qualify.
    Large/central boxes may remain overlapping and are explicitly reported.
    """
    videos = batch["videos"]
    x = videos.tensors.detach().clone()
    n, _, height, width = x.shape
    oh, ow = map(int, batch["targets"][0]["ori_size"])
    boxes = boxes_abs.detach().cpu().numpy().astype(np.float64)
    rects = []
    for box in boxes:
        x0 = int(np.floor(np.clip(box[0] / ow, 0, 1) * width))
        y0 = int(np.floor(np.clip(box[1] / oh, 0, 1) * height))
        x1 = int(np.ceil(np.clip(box[2] / ow, 0, 1) * width))
        y1 = int(np.ceil(np.clip(box[3] / oh, 0, 1) * height))
        x0, y0 = min(x0, width - 1), min(y0, height - 1)
        rects.append([x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)])
    if len(rects) != n:
        raise ValueError("Box count and frame count differ")
    shift = (0, 0)
    reflection = "none"
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        candidates = {}
        for axis in ("x", "y", "xy"):
            overlap = []
            for x0, y0, x1, y1 in rects:
                sx0, sx1 = (width-x1, width-x0) if "x" in axis else (x0,x1)
                sy0, sy1 = (height-y1, height-y0) if "y" in axis else (y0,y1)
                area = (x1-x0)*(y1-y0)
                inter = max(0,min(x1,sx1)-max(x0,sx0))*max(0,min(y1,sy1)-max(y0,sy0))
                overlap.append(inter / max(2*area-inter,1))
            candidates[axis] = float(np.mean(overlap))
        eligible = [axis for axis, value in candidates.items() if value <= .5]
        reflection = eligible[int(rng.integers(len(eligible)))] if eligible else min(candidates,key=candidates.get)
    masks = torch.zeros(n, height, width, dtype=torch.bool)
    intersections, areas = [], []
    for i, (x0, y0, x1, y1) in enumerate(rects):
        sx0, sx1 = (width-x1, width-x0) if "x" in reflection else (x0,x1)
        sy0, sy1 = (height-y1, height-y0) if "y" in reflection else (y0,y1)
        region = x[i, :, sy0:sy1, sx0:sx1]
        x[i, :, sy0:sy1, sx0:sx1] = region.mean(dim=(-1, -2), keepdim=True)
        masks[i, sy0:sy1, sx0:sx1] = True
        area = (x1 - x0) * (y1 - y0)
        inter = max(0, min(x1, sx1) - max(x0, sx0)) * max(0, min(y1, sy1) - max(y0, sy0))
        areas.append(area)
        intersections.append(inter / max(2 * area - inter, 1))
    out = dict(batch)
    out["videos"] = type(videos)(x, videos.mask.clone(), list(videos.durations))
    return out, {"fill": "per-frame per-channel local mean", "reflection_axis": reflection,
                 "mask_areas": areas, "mask_area_fraction_mean": float(np.mean(areas) / (height * width)),
                 "target_control_mask_iou_mean": float(np.mean(intersections)),
                 "random_control_high_overlap": random_seed is not None and float(np.mean(intersections)) > .5,
                 "mask_sha256": hashlib.sha256(masks.numpy().tobytes()).hexdigest(),
                 "pixel_change_mean_abs": float((x - videos.tensors).abs().mean())}


class QuerySubjectParser:
    """Query-only Stanza dependency parser, explicitly not official CoreNLP."""
    def __init__(self, model_dir):
        import stanza
        self.version = stanza.__version__
        self.nlp = stanza.Pipeline("en", dir=str(model_dir), package=None,
            processors={"tokenize": "ewt", "mwt": "ewt", "pos": "ewt_nocharlm",
                        "lemma": "ewt_nocharlm", "depparse": "ewt_nocharlm"},
            use_gpu=False, download_method=None, verbose=False)

    def __call__(self, query):
        words = [w for s in self.nlp(query).sentences for w in s.words]
        chosen, rule = None, None
        # A wh-determiner identifies the queried noun even when it is an object.
        for w in words:
            if w.lemma.lower() in {"which", "what"} and w.deprel == "det":
                chosen = next((v for v in words if v.id == w.head), None)
                if chosen:
                    rule = "wh_determiner_head"
                    break
        if chosen is None:
            for w in words:
                if w.lemma.lower() in {"who", "whom"}:
                    return {"subject": "person", "rule": "explicit_who", "parser_version": self.version}
            if any(w.lemma.lower() == "what" for w in words):
                return {"subject": "", "rule": "unknown_what_no_category_in_query", "parser_version": self.version}
            chosen = next((w for w in words if w.deprel.startswith("nsubj") and w.upos in {"NOUN", "PROPN"}), None)
            rule = "dependency_nominal_subject"
        if chosen is None:
            chosen = next((w for w in words if w.upos in {"NOUN", "PROPN"}), None)
            rule = "first_noun_fallback"
        return {"subject": chosen.lemma.lower() if chosen else "", "rule": rule if chosen else "empty_prefix_fallback",
                "parser_version": self.version}


def set_query_subject(model, target, subject):
    key = str(target["item_id"]) if model.cfg.DATASET.NAME == "VidSTG" else target["vid"]
    model.verb_label2 = {key: {"sub": subject, "verb_index_list": [], "adj_index_list": []}}


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

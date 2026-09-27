"""TA-STVG input/capture/reinsertion. No datasets or evaluation labels are read.

Final inference is STILL a full original-model forward. Source parameters,
temporary hooks, autocast contexts and the query-only lookup are exception-safe.
"""

from contextlib import contextmanager
import numpy as np
import torch

from .objectives import prediction
from .tensors import detached, floating32


def validate_input(frames, ids, metadata):
    allowed = {"caption", "index", "width", "height", "fps", "source", "original_video_id",
               "video_path", "video_sha256", "frame_ids", "frame_count", "duration",
               "start_frame", "end_frame", "kind"}
    if set(metadata) - allowed:
        raise ValueError(f"Unsupported input fields: {sorted(set(metadata) - allowed)}")
    if not {"caption", "index", "width", "height"} <= set(metadata):
        raise ValueError("caption/index/width/height are required")
    if not isinstance(metadata["caption"], str) or not metadata["caption"].strip():
        raise ValueError("Expected a nonempty query")
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 [T,H,W,3]")
    if frames.shape != (len(ids), metadata["height"], metadata["width"], 3):
        raise ValueError("Frame geometry differs from metadata")
    if len(ids) < 4 or any(isinstance(i, (bool, np.bool_)) or int(i) != i for i in ids):
        raise ValueError("At least four integral frame IDs are required")
    if ids[0] < 0 or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("Frame IDs must be nonnegative and strictly increasing")
    if "frame_ids" in metadata and list(metadata["frame_ids"]) != list(ids):
        raise ValueError("Input frame grids differ")


def make_batch(frames, ids, metadata, model):
    from datasets.collate_batch import collate_fn
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.
    height = 224
    width = min(int(height * metadata["width"] / metadata["height"]), int(height * 1.4))
    x = torch.nn.functional.interpolate(x, size=(height, width), mode="bilinear",
                                        align_corners=False, antialias=True)
    x = ((x - torch.tensor([.485, .456, .406])[None, :, None, None])
         / torch.tensor([.229, .224, .225])[None, :, None, None])
    target = dict(item_id=metadata["index"],
                  vid=metadata.get("original_video_id", metadata.get("source", metadata["index"])),
                  frame_ids=list(ids), actioness=torch.zeros(len(ids), dtype=torch.int64),
                  img_size=tuple(x.shape[-2:]), ori_size=(metadata["height"], metadata["width"]))
    batch = collate_fn([(x, metadata["caption"], target)])
    device = next(model.parameters()).device
    batch["videos"] = batch["videos"].to(device)
    batch["targets"][0]["actioness"] = batch["targets"][0]["actioness"].to(device)
    return batch


@contextmanager
def query_subject(model, batch, subject):
    previous = model.verb_label2
    target = batch["targets"][0]
    key = str(target["item_id"]) if model.cfg.DATASET.NAME == "VidSTG" else target["vid"]
    model.verb_label2 = {key: dict(sub=subject, verb_index_list=[], adj_index_list=[])}
    try:
        yield
    finally:
        model.verb_label2 = previous


def offset_batch(batch, offset):
    videos = batch["videos"].subsample(2, start_idx=offset)
    target = dict(batch["targets"][0])
    target["frame_ids"] = target["frame_ids"][offset::2]
    target["actioness"] = target["actioness"][offset::2]
    if len(target["frame_ids"]) != videos.durations[0]:
        raise RuntimeError("Native offset/grid mismatch")
    return dict(videos=videos, durations=list(videos.durations),
                texts=list(batch["texts"]), targets=[target])


@torch.no_grad()
def capture(model, batch):
    views, records = [], []
    for offset in (0, 1):
        prefixes, grounds = [], []
        hooks = [model.ground_encoder.encoder.norm.register_forward_pre_hook(
            lambda m, a: prefixes.append(detached(a[0]))),
            model.ground_decoder.register_forward_pre_hook(
                lambda m, a, kw: grounds.append(detached(kw)) if not grounds else None,
                with_kwargs=True)]
        view = offset_batch(batch, offset)
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                model(view["videos"], view["texts"], view["targets"], iteration_rate=-1)
        finally:
            for hook in hooks:
                hook.remove()
        if len(prefixes) != 1 or len(grounds) != 1:
            raise RuntimeError("Unexpected native encoder/decoder routing")
        views.append(dict(prefix=prefixes[0], info=grounds[0]["encoded_info"],
                          vis_pos=grounds[0]["vis_pos"]))
        records.append(dict(offset=offset, flip=False, frame_ids=view["targets"][0]["frame_ids"]))
    return views, records


@contextmanager
def inserted_state(model, state):
    """Copy only mutable parameters, not the entire frozen spatial decoder."""
    spatial = model.ground_decoder.decoder
    params = dict(model.temp_embed.named_parameters())
    spatial_params = dict(spatial.named_parameters())
    mapped, saved, hooks, contexts, calls = {}, {}, [], [], [0]
    delta = torch.zeros(256, device=next(model.parameters()).device)
    for name, value in state.items():
        if name == "spatial.query_residual":
            delta = value.to(delta)
        elif name.startswith("head."):
            mapped[name] = params[name.removeprefix("head.")]
        elif name.startswith("spatial.layers.5.") and name.split(".")[3] in {"norm1", "norm3", "norm4"}:
            mapped[name] = spatial_params[name.removeprefix("spatial.")]
        else:
            raise ValueError(f"Parameter outside the fixed private interfaces: {name}")
    for name, param in mapped.items():
        if param.shape != state[name].shape:
            raise ValueError(f"Parameter shape differs: {name}")
        saved[name] = param.detach().clone()

    def norm_before(module, args):
        context = torch.autocast("cuda", enabled=False)
        context.__enter__()
        contexts.append(context)
        return floating32(args)

    def query_before(module, args, kwargs):
        calls[0] += 1
        if calls[0] % 2 == 0:
            kwargs = dict(kwargs)
            kwargs["query_tgt"] = kwargs["query_tgt"] + delta[None, None, :]
        return args, kwargs

    def model_after(module, args, output):
        if contexts:
            contexts.pop().__exit__(None, None, None)

    try:
        with torch.no_grad():
            for name, param in mapped.items():
                param.copy_(state[name].to(param))
        hooks.append(model.ground_encoder.encoder.norm.register_forward_pre_hook(norm_before))
        hooks.append(model.ground_decoder.register_forward_pre_hook(
            lambda m, a, kw: (floating32(a), floating32(kw)), with_kwargs=True))
        hooks.append(spatial.register_forward_pre_hook(query_before, with_kwargs=True))
        hooks.append(model.register_forward_hook(model_after, always_call=True))
        yield
    finally:
        for hook in hooks:
            hook.remove()
        while contexts:
            contexts.pop().__exit__(None, None, None)
        with torch.no_grad():
            for name, param in mapped.items():
                param.copy_(saved[name])
                param.grad = None


@torch.no_grad()
def full_prediction(model, batch, ids, records, state, expected):
    boxes, logits = [], []
    with inserted_state(model, state):
        for offset in (0, 1):
            view = offset_batch(batch, offset)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(view["videos"], view["texts"], view["targets"], iteration_rate=-1)
            boxes.append(out["pred_boxes"])
            logits.append(out["pred_sted"])
    merged = torch.stack([boxes[i % 2][i // 2] for i in range(len(ids))]).float()
    result = prediction(logits, merged, records, ids)
    if not torch.equal(result["boxes"], expected["boxes"].cpu()):
        raise RuntimeError("Final full-model boxes differ from saved spatial state")
    if any(not torch.equal(a, b.cpu()) for a, b in zip(result["logits"], expected["logits"])):
        raise RuntimeError("Final full-model temporal logits differ from saved temporal state")
    return result

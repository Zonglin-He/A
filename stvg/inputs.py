from contextlib import contextmanager

import numpy as np

import torch

from stvg.readout import prediction

from stvg.tensors import detached, floating32

def validate_metadata(metadata):
    """Reject label-bearing or unknown metadata before any video decoding."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    allowed = {"caption", "index", "width", "height", "fps", "source", "original_video_id",
               "video_path", "video_sha256", "frame_ids", "frame_count", "duration",
               "start_frame", "end_frame", "kind"}
    if set(metadata) - allowed:
        raise ValueError(f"Unsupported input fields: {sorted(set(metadata) - allowed)}")
    if not {"caption", "index", "width", "height"} <= set(metadata):
        raise ValueError("caption/index/width/height are required")
    if not isinstance(metadata["caption"], str) or not metadata["caption"].strip():
        raise ValueError("Expected a nonempty query")

def validate_input(frames, ids, metadata):
    validate_metadata(metadata)
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

import inspect,textwrap
@contextmanager
def resolution(height):
 from stvg import inputs as backbone
 original=backbone.make_batch;source=textwrap.dedent(inspect.getsource(original));assert source.count('height = 224')==1
 changed=source.replace('height = 224',f'height = {int(height)}');assert changed.replace(f'height = {int(height)}','height = 224')==source
 ns=dict(vars(backbone));exec(compile(changed,__file__,'exec'),ns);backbone.make_batch=ns['make_batch']
 try:yield
 finally:backbone.make_batch=original

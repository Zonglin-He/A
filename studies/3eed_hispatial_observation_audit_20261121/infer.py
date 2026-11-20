"""Run the released frozen HiSpatial decoder on sealed image/point-map conditions."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch

from common import ARMS, ASSETS, OUT, PLATFORMS, PRIVATE, PROTOCOL, SIDE, digest, read, require_repo, require_weights, write
from maps import map_arm


def load_released(sources, weights_root):
    from accelerate import init_empty_weights
    from transformers import PaliGemmaConfig, PaliGemmaProcessor

    source = sources / "HiSpatial"
    require_repo(source, "hispatial")
    sys.path.insert(0, str(source))
    from hispatial.model import HiSpatialVLM
    from hispatial.inference import HiSpatialPredictor

    path = require_weights(weights_root, "hispatial")
    config = PaliGemmaConfig.from_pretrained(path, local_files_only=True)
    with init_empty_weights():
        model = HiSpatialVLM(config)
    checkpoint = torch.load(path / "weights.pt", map_location="cpu", weights_only=True, mmap=True)
    assert isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values())
    load = model.load_state_dict(checkpoint, strict=True, assign=True)
    assert not load.missing_keys and not load.unexpected_keys
    assert all(not p.is_meta for p in model.parameters())
    original_dtypes = sorted({str(v.dtype) for v in checkpoint.values()})
    model = model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
    key_count = len(checkpoint)
    del checkpoint
    processor = PaliGemmaProcessor.from_pretrained(path, local_files_only=True)
    # This object uses the released single-query path with the same FP32
    # tensors. CPU-mapped weights avoid holding two 14 GB copies in RAM.
    official = HiSpatialPredictor.__new__(HiSpatialPredictor)
    official.model = model
    official.processor = processor
    official.img_size = SIDE
    official.device = torch.device("cuda")
    return official, dict(checkpoint_tensor_count=key_count, checkpoint_dtypes=original_dtypes,
        parameters=sum(p.numel() for p in model.parameters()), strict_safe_tensor_load=True)


def read_rgb(frame, mode):
    path = frame["rgb"] if mode == "language" else frame["oracle_rgb"]
    expected = frame["rgb_sha"] if mode == "language" else frame["oracle_sha"]
    assert digest(path) == expected
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def query_inputs(official, rgb, prompts):
    resized = cv2.resize(rgb,(SIDE,SIDE))
    return official.processor(text=["<image>"+p for p in prompts], images=[resized]*len(prompts),
                              padding=True, return_tensors="pt").to("cuda")


@torch.inference_mode()
def feature(official,rgb,xyz):
    x = query_inputs(official,rgb,[" "])
    return official.model.get_image_features(x["pixel_values"],xyz[None].to("cuda"))


@torch.inference_mode()
def query_batch(official, rgb, xyz, prompts, fused):
    inputs = query_inputs(official,rgb,prompts)
    inputs["xyz_values"] = xyz[None].to("cuda").expand(len(prompts),-1,-1,-1)
    # Reuse only the fused RGB/XYZ features. All text, positional state,
    # attention and language model computation is recomputed per prompt.
    def reuse(pixel_values, xyz_values):
        assert pixel_values.shape[0] == xyz_values.shape[0] == len(prompts)
        return fused.expand(len(prompts),-1,-1)
    with patch.object(official.model,"get_image_features",side_effect=reuse):
        generation = official.model.generate(**inputs,max_new_tokens=100,do_sample=False)
    suffix = generation[:,inputs["input_ids"].shape[-1]:]
    return official.processor.batch_decode(suffix,skip_special_tokens=False)


def tokens_clean(text):
    # Batch decoding includes padding after the end token in shorter answers.
    return text.replace("<pad>","").strip()


@torch.inference_mode()
def match_official(official, frames, questions, maps):
    records=[]
    for target in PLATFORMS:
        levels = ("L0","L1","L2","L3")
        frame = next(f for f in frames if f["target"] == target and
                     {q["level"] for q in questions[f["uid"]]["language"]} >= set(levels))
        items = [next(q for q in questions[frame["uid"]]["language"] if q["level"]==level) for level in levels]
        rgb=read_rgb(frame,"language")
        for arm in ("moge","prior_depth"):
            pinned = PRIVATE/"maps"/(frame["uid"]+".npz")
            assert digest(pinned)==maps[frame["uid"]]
            with np.load(pinned,allow_pickle=False) as f:
                xyz=torch.from_numpy(map_arm(f,arm))
            baseline=[official.query(rgb,q["prompt"],xyz_values=xyz) for q in items]
            fused=feature(official,rgb,xyz)
            singles=[query_batch(official,rgb,xyz,[q["prompt"]],fused)[0] for q in items]
            batch=query_batch(official,rgb,xyz,[q["prompt"] for q in items[:3]],fused)
            batch.append(query_batch(official,rgb,xyz,[items[0]["prompt"],items[3]["prompt"]],fused)[1])
            for i,q in enumerate(items):
                records.append(dict(qid=q["qid"],target=target,arm=arm,
                    cached_exact=tokens_clean(baseline[i])==tokens_clean(singles[i]),
                    batch_exact=tokens_clean(baseline[i])==tokens_clean(batch[i]),
                    released=baseline[i]))
    return records


def run(sources,weights_root,batch_size):
    started=time.perf_counter()
    panel=read(OUT/"panel_seal.json")
    assert panel["questions"]==digest(PRIVATE/"questions.json")
    assert panel["frames"]==digest(PRIVATE/"frames.json") and panel["protocol"]==digest(PROTOCOL)
    assert read(OUT/"maps_seal.json")["hashes"]==digest(PRIVATE/"map_hashes.json")
    maps=read(PRIVATE/"map_hashes.json")
    frames=read(PRIVATE/"frames.json")
    questions=defaultdict(lambda:defaultdict(list))
    for q in read(PRIVATE/"questions.json"):
        questions[q["uid"]][q["mode"]].append(q)

    official, meta = load_released(sources, weights_root)
    official.processor.tokenizer.padding_side="left"
    torch.manual_seed(20261121)
    check=match_official(official,frames,questions,maps)
    cache_ok=all(r["cached_exact"] for r in check)
    batch_ok=cache_ok and all(r["batch_exact"] for r in check)
    used_batch=min(3,batch_size) if batch_ok else 1
    write(PRIVATE/"official_smoke.json",check)
    write(OUT/"official_smoke.json",dict(queries=len(check),native_moge_queries=sum(x["arm"]=="moge" for x in check),
        exact_cached=sum(x["cached_exact"] for x in check), exact_in_batch=sum(x["batch_exact"] for x in check),
        frozen_params=True, effective_batch_size=used_batch,
        single_query_released_fallback=not cache_ok, load_and_smoke_seconds=time.perf_counter()-started,
        **meta))
    stamp=dict(panel=digest(OUT/"panel_seal.json"),maps=digest(OUT/"maps_seal.json"),assets=digest(ASSETS),
               code=digest(__file__), cache=cache_ok, batch=used_batch)
    folder=PRIVATE/"answers_frozen"
    folder.mkdir(exist_ok=True,mode=0o700)
    start=time.perf_counter()
    outputs, freshly = {}, 0
    for i,frame in enumerate(frames,1):
        uid=frame["uid"]
        file=folder/(uid+".json")
        if file.exists():
            assert read(file)["stamp"]==stamp
        else:
            path=PRIVATE/"maps"/(uid+".npz")
            assert digest(path)==maps[uid]
            result=[]
            with np.load(path,allow_pickle=False) as maps_file:
                for mode, qs in questions[uid].items():
                    rgb=read_rgb(frame,mode)
                    for arm in ARMS:
                        xyz=torch.from_numpy(map_arm(maps_file,arm))
                        fused=feature(official,rgb,xyz) if cache_ok else None
                        for j in range(0,len(qs),used_batch):
                            batch=qs[j:j+used_batch]
                            prompts=[q["prompt"] for q in batch]
                            if cache_ok:
                                text=query_batch(official,rgb,xyz,prompts,fused)
                            else:
                                text=[official.query(rgb,prompt,xyz_values=xyz) for prompt in prompts]
                            assert len(batch)==len(text)
                            result.extend(dict(qid=q["qid"],arm=arm,answer=tokens_clean(raw))
                                          for q,raw in zip(batch,text))
            write(file,dict(stamp=stamp,answers=result))
            freshly+=len(result)
        outputs[uid]=digest(file)
        if i % 3 == 0:
            print(f"HiSpatial {i}/{len(frames)} frames, {freshly} fresh responses",flush=True)
    write(PRIVATE/"answers_frozen_hashes.json",outputs)
    count=len({q["qid"] for group in questions.values() for qq in group.values() for q in qq})*len(ARMS)
    audit=dict(stamp=stamp, count=count, fresh=freshly, frames=len(outputs), model_device="cuda",
       answers_hashes=digest(PRIVATE/"answers_frozen_hashes.json"),profile_seconds=time.perf_counter()-start,
       peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
       gpu=torch.cuda.get_device_name(),pytorch=torch.__version__,transformers=version("transformers"), target_updates=0,
       official_smoke=digest(OUT/"official_smoke.json"), scoring_key_accessed=False)
    write(OUT/"prediction_seal.json",audit)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources",type=Path)
    parser.add_argument("weights",type=Path)
    parser.add_argument("--batch-size",type=int,default=3)
    args=parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    run(args.sources,args.weights,args.batch_size)

"""Resume the two original frozen Qwen reads on the complete ordinary roster."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from roster import HERE, OLD, OUT, PRIVATE, TASKS, panel, read, sha, write

sys.path.insert(0, str(OLD))
from infer import PROMPTS, parse

PHYSICAL_BATCH = 32


def load_model(model_dir):
    cfg = read(OLD/"PROTOCOL.json")
    assets = read(OLD/"MODEL.json")
    assert (assets["model"],assets["revision"]) == (cfg["model"],cfg["revision"])
    for name, expected in assets["files"].items():
        path = model_dir/name
        assert path.stat().st_size == expected["size"]
        if "sha256" in expected:
            assert sha(path) == expected["sha256"]
        else:
            b = path.read_bytes()
            assert hashlib.sha1(f"blob {len(b)}".encode()+b"\x00"+b).hexdigest() == expected["git_blob_sha1"]
    torch.manual_seed(cfg["inference"]["seed"])
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.image_processor.size = {"shortest_edge":65536, "longest_edge":cfg["inference"]["max_pixels"]}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, device_map={"":"cuda:0"}, attn_implementation="sdpa",
        local_files_only=True, use_safetensors=True
    ).eval().requires_grad_(False)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    return processor, model, cfg["inference"]["max_new_tokens"]


def greedy(processor, model, limit, frames, task):
    images, prompts = [], []
    for f in frames:
        with Image.open(f["rgb"]) as im:
            images.append(im.convert("RGB"))
        msg = [{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPTS[task]+f["text"]}]}]
        prompts.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
    x = processor(text=prompts, images=images, padding=True, return_tensors="pt").to("cuda")
    y = model.generate(**x, do_sample=False, max_new_tokens=limit)
    follow = y[:,x.input_ids.shape[1]:]
    decoded = processor.batch_decode(follow, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    grid = x.image_grid_thw.tolist()
    tokens = (follow!=processor.tokenizer.pad_token_id).sum(1).tolist()
    records = []
    for response, shape, length in zip(decoded,grid,tokens,strict=True):
        value, status = parse(response,task)
        records.append(dict(raw=response,value=value,status=status,visual_grid=shape,generated_tokens=length))
    return records


def batch_read(processor, model, limit, frames, task, batches):
    try:
        results = greedy(processor, model, limit, frames, task)
    except torch.cuda.OutOfMemoryError:
        if len(frames)<=1:
            raise
        batches["memory_split"] += 1
    else:
        batches[str(len(frames))] += 1
        return results
    torch.cuda.empty_cache()
    m = len(frames)//2
    return (batch_read(processor,model,limit,frames[:m],task,batches)
            + batch_read(processor,model,limit,frames[m:],task,batches))


def computation_replay(processor, model, limit, frames, batches, pins):
    """Compare the larger batches against matched, unchanged historical greedy outputs."""
    audit_path = OUT/"batch_replay.json"
    if audit_path.is_file():
        existing = read(audit_path)
        assert existing["pins"] == pins
        return existing
    metrics = []
    start = perf_counter()
    for target in ("waymo","drone","quad"):
        candidates = [f for f in frames if f["target"]==target and "archived" in f]
        assert len(candidates) >= PHYSICAL_BATCH
        # Uniform spread across the archived cases in the original target order.
        chosen = [candidates[i*len(candidates)//PHYSICAL_BATCH] for i in range(PHYSICAL_BATCH)]
        for task in TASKS:
            tick = perf_counter()
            out = batch_read(processor,model,limit,chosen,task,batches)
            saved = [r["archived"][task] for r in chosen]
            metrics.append(dict(platform=target, task=task, count=len(out),
                exact_reply=sum(a["raw"]==b["raw"] for a,b in zip(out,saved,strict=True)),
                exact_decoded=sum((a["value"],a["status"])==(b["value"],b["status"]) for a,b in zip(out,saved,strict=True)),
                equal_status=sum(a["status"]==b["status"] for a,b in zip(out,saved,strict=True)),
                equal_grid=sum(a["visual_grid"]==b["visual_grid"] for a,b in zip(out,saved,strict=True)),
                seconds=perf_counter()-tick))
    assert all(r["count"]==r["equal_grid"] for r in metrics)
    record = dict(pins=pins, rows=metrics, seconds=perf_counter()-start, historical_predictions_replaced=0,
                  description="Scheduling replay, compares labels-free greedy output against exact input matches from archived batch-4 replies.",
                  new_GT_read=False, FINAL_access=False)
    write(audit_path,record)
    print("Larger-batch replay",[(z["platform"],z["task"],z["exact_decoded"],z["count"],round(z["seconds"],2)) for z in metrics],flush=True)
    return record


@torch.inference_mode()
def run(model_dir):
    start = perf_counter()
    frames, seal = panel()
    pins = dict(revision=read(OLD/"MODEL.json")["revision"], inputs=sha(OUT/"input_seal.json"),
                code=sha(__file__), batch=PHYSICAL_BATCH, assets=sha(OLD/"MODEL.json"))
    processor, model, limit = load_model(model_dir)
    count_batches = Counter()
    batch_audit = computation_replay(processor,model,limit,frames,count_batches,pins)
    folder = PRIVATE/"replies"
    folder.mkdir(exist_ok=True, mode=0o700)
    new = initial = 0
    need = [r for r in frames if "archived" not in r]
    queries = len(need)*len(TASKS)
    for target in ("waymo","drone","quad"):
        choices = [r for r in need if r["target"]==target]
        for start_at in range(0,len(choices),PHYSICAL_BATCH):
            block=choices[start_at:start_at+PHYSICAL_BATCH]
            for task in TASKS:
                pending=[]
                for frame in block:
                    path=folder/(frame["uid"]+"_"+task+".json")
                    if path.is_file():
                        saved=read(path)
                        assert saved["pins"]==pins and (saved["uid"],saved["task"])==(frame["uid"],task)
                        initial+=1
                    else:
                        pending.append(frame)
                if not pending:
                    continue
                records=batch_read(processor,model,limit,pending,task,count_batches)
                for frame, record in zip(pending,records,strict=True):
                    uid=frame["uid"]
                    write(folder/(uid+"_"+task+".json"),dict(uid=uid,task=task,pins=pins,**record))
                new+=len(pending)
            if (start_at//PHYSICAL_BATCH)%4==0 or start_at+len(block)==len(choices):
                print("Fresh model reads",new+initial,"/",queries,
                      "latest",target,start_at+len(block),"/",len(choices),"elapsed",round(perf_counter()-start),flush=True)
    keys = {r["uid"]+"_"+t:sha(folder/(r["uid"]+"_"+t+".json")) for r in need for t in TASKS}
    assert len(keys)==queries==new+initial
    write(PRIVATE/"fresh_reply_hashes.json",keys)
    write(OUT/"qwen_seal.json",dict(pins=pins, queries=2*len(frames), fresh_outputs=queries, produced_this_attempt=new,
          cached_original=seal["reuse_queries"], responses=sha(PRIVATE/"fresh_reply_hashes.json"), batch_replay=sha(OUT/"batch_replay.json"),
          actual_batches=dict(count_batches), seconds=perf_counter()-start, torch=torch.__version__, transformers=transformers.__version__,
          gpu=torch.cuda.get_device_name(), memory_peak_bytes=torch.cuda.max_memory_allocated(),
          target_labels_read=False, FINAL_access=False))
    print("Qwen seal complete",new+initial,"/",queries,flush=True)


if __name__=="__main__":
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("model_dir",type=Path)
    run(cmd.parse_args().model_dir)

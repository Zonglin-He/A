"""Greedy matched image reads; select only among box coordinates sealed in the parent."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from endpoints import (ENCODINGS, ETA, HERE, MODES, OUT, PRIVATE, SPEC, SWAP, ancestry,
                       canvas, digest, message, parse, read, write)
from landmarks import check_assets


def read_frame(model, processor, row, obs, saved):
    assert digest(row["rgb"])==row["rgb_sha"]
    with Image.open(row["rgb"]) as im:
        photo=im.convert("RGB")
    modes=MODES if obs["status"]=="eligible" else ("span",)
    asks=[]
    views=[]
    keys=[]
    text_hash={}
    image_hash={}
    for enc in ENCODINGS:
        reverse=enc=="reverse"
        text=message(row,saved,reverse)
        text_hash[enc]=hashlib.sha256(text.encode()).hexdigest()
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}]
        prompt=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True)
        for mode in modes:
            pair=None if mode=="span" else obs["correct" if mode=="endpoint" else "wrong"]
            display=canvas(photo,obs["views"],pair,reverse)
            key=mode+"_"+enc
            image_hash[key]=hashlib.sha256(display.tobytes()).hexdigest()
            keys.append(key); views.append(display); asks.append(prompt)
    assert image_hash["span_forward"]==image_hash["span_reverse"]
    batch=processor(text=asks,images=views,padding=True,return_tensors="pt").to("cuda")
    generated=model.generate(**batch,do_sample=False,max_new_tokens=32)
    answers=processor.batch_decode(generated[:,batch.input_ids.size(1):],skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False)
    return dict(answers=dict(zip(keys,answers,strict=True)),prompts=text_hash,canvases=image_hash)


def decide(result, mode):
    a,b=(parse(result["answers"][mode+"_"+e]) for e in ENCODINGS)
    if a not in ETA or b not in ETA:
        return "H","invalid" if "invalid" in (a,b) else "unknown",a,b
    if a!=SWAP[b]:
        return "H","conflict",a,b
    return a,"agree",a,b


@torch.inference_mode()
def run(model_dir):
    start=perf_counter()
    panel, _, prev, options=ancestry()
    seal=read(OUT/"input_seal.json")
    assert digest(PRIVATE/"observations.json")==seal["observations"] and digest(SPEC)==seal["protocol"]
    assert all(digest(HERE/k)==v for k,v in seal["code"].items()) and not seal["target_labels_read"]
    observed=read(PRIVATE/"observations.json")
    pins=dict(inputs=digest(OUT/"input_seal.json"),code=digest(__file__),revision=check_assets(model_dir))
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True)
    processor.image_processor.size={"shortest_edge":65536,"longest_edge":524288}
    processor.tokenizer.padding_side="left"
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,
        dtype=torch.bfloat16,device_map={"":"cuda:0"},attn_implementation="sdpa").eval().requires_grad_(False)
    replies=PRIVATE/"replies"
    replies.mkdir(mode=0o700,exist_ok=True)
    outputs={}; manifest={}; calls=fresh=0
    for row in panel:
        uid=row["uid"]; obs=observed[uid]; choices=options[uid]
        usable=obs["status"]=="eligible"
        r={}; raw={}
        if obs["movable"]:
            path=replies/(uid+".json")
            if not path.exists():
                raw=read_frame(model,processor,row,obs,prev[uid])
                write(path,dict(**raw,uid=uid,pins=pins))
                fresh+=len(raw["answers"])
            raw=read(path)
            assert raw["uid"]==uid and raw["pins"]==pins
            manifest[uid]=digest(path)
            calls+=len(raw["answers"])
        for arm in (*MODES,"span_all"):
            mode="span" if arm=="span_all" else arm
            if obs["movable"] and (arm=="span_all" or usable):
                label,status,a,b=decide(raw,mode)
            else:
                label,status,a,b="H",obs["status"],"unavailable","unavailable"
            chosen=choices["H"] if label=="H" else choices["discrete_boxes"][label]
            r[arm]=dict(label=label,status=status,forward=a,reverse=b,box=chosen)
        outputs[uid]=r
        if len(outputs)%64==0:
            print("Frames",len(outputs),"/",len(panel),"model calls",calls,flush=True)
    assert len(manifest)==seal["movable"] and calls==2*seal["movable"]+4*seal["statuses"]["eligible"]
    write(PRIVATE/"predictions.json",outputs)
    write(PRIVATE/"reply_hashes.json",manifest)
    write(OUT/"prediction_seal.json",dict(pins=pins,frames=len(outputs),calls=calls,fresh_calls=fresh,
          predictions=digest(PRIVATE/"predictions.json"),reply_hashes=digest(PRIVATE/"reply_hashes.json"),
          seconds=perf_counter()-start,torch=torch.__version__,transformers=transformers.__version__,
          gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          target_labels_read=False,FINAL_access=False))
    print("Sealed endpoint predictions",len(outputs),calls,flush=True)


if __name__=="__main__":
    cmd=argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("model_dir",type=Path)
    run(cmd.parse_args().model_dir)

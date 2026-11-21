"""Restore and stream the original interleaved four-query inference."""
import argparse
from collections import Counter
import hashlib
import os
from pathlib import Path
from time import perf_counter

import torch
import transformers

from roster import CONFIRM, HERE, OLD, OUT, PRIVATE, TASKS, file_sha, panel, read, sha, write
from run_full_qwen import PROMPTS, Image, load_model, parse

CONFIG=HERE/"SCHEDULE.json"
BATCH=4


def predict(processor,model,max_tokens,batch):
    images,prompts=[],[]
    for frame,task in batch:
        assert file_sha(frame["rgb"])==frame["rgb_sha"]
        with Image.open(frame["rgb"]) as image:
            images.append(image.convert("RGB"))
        msg=[{"role":"user","content":[{"type":"image"},{"type":"text","text":PROMPTS[task]+frame["text"]}]}]
        prompts.append(processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True))
    request=processor(text=prompts,images=images,padding=True,return_tensors="pt").to("cuda")
    out=model.generate(**request,do_sample=False,max_new_tokens=max_tokens)
    suffix=out[:,request.input_ids.shape[1]:]
    replies=processor.batch_decode(suffix,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    lengths=(suffix!=processor.tokenizer.pad_token_id).sum(1).tolist()
    grids=request.image_grid_thw.tolist()
    payload=[]
    for (_,task), raw, length, shape in zip(batch,replies,lengths,grids,strict=True):
        value,status=parse(raw,task)
        payload.append(dict(raw=raw,value=value,status=status,visual_grid=shape,generated_tokens=length))
    return payload


def verify_original(processor,model,limit,pins):
    path=OUT/"batch_replay.json"
    if path.is_file():
        a=read(path)
        assert a["pins"]==pins
        return a
    receipts=[]
    for source in (OLD,CONFIRM):
        original=read(source/"private/panel.json")
        manifest=read(source/"private/reply_hashes.json")
        old=read(source/"results/model_seal.json")
        assert hashlib.sha256(__import__("json").dumps(manifest,sort_keys=True).encode()).hexdigest()==old["ordered_reply_hash"]
        # Test whole original batch blocks including frame/task partner and order.
        for first in (0,2,34,66,82):
            items=[(r,t) for r in original[first:first+2] for t in TASKS]
            tick=perf_counter()
            real=predict(processor,model,limit,items)
            expected=[]
            for r,t in items:
                name=r["uid"]+"_"+t
                f=source/"private/replies"/(name+".json")
                assert sha(f)==manifest[name]
                expected.append(read(f))
            for (_,task),x,y in zip(items,real,expected,strict=True):
                receipts.append(dict(
                    source=source.name,task=task,exact_raw=x["raw"]==y["raw"],exact_value=x["value"]==y["value"],
                    exact_status=x["status"]==y["status"],exact_grid=x["visual_grid"]==y["visual_grid"],
                    batch_seconds=perf_counter()-tick))
    assert all(v["exact_grid"] for v in receipts)
    reply=dict(pins=pins,rows=receipts,compared=len(receipts),exact_replies=sum(r["exact_raw"] for r in receipts),
               exact_decoded=sum(r["exact_value"] and r["exact_status"] for r in receipts),
               historical_predictions_replaced=0,new_GT_read=False,FINAL_access=False)
    write(path,reply)
    print("Restored batch-4 replay",reply["exact_decoded"],"/",len(receipts),"decoded",flush=True)
    return reply


@torch.inference_mode()
def run(model_dir):
    start=perf_counter()
    frames,inputs=panel()
    assert read(OLD/"PROTOCOL.json")["inference"]["batch_size"]==BATCH
    pins=dict(revision=read(OLD/"MODEL.json")["revision"], inputs=sha(OUT/"input_seal.json"), code=sha(__file__),
              schedule=sha(CONFIG), helpers=sha(HERE/"run_full_qwen.py"),batch=BATCH,assets=sha(OLD/"MODEL.json"))
    processor,model,limit=load_model(model_dir)
    audit=verify_original(processor,model,limit,pins)
    folder=PRIVATE/"replies"
    folder.mkdir(exist_ok=True,mode=0o700)
    jobs=[(r,t) for r in frames if "archived" not in r for t in TASKS]
    size=len(jobs)
    made=reused=0
    t0=perf_counter()
    for start_at in range(0,size,BATCH):
        group=jobs[start_at:start_at+BATCH]
        todo=[]
        for r,t in group:
            path=folder/(r["uid"]+"_"+t+".json")
            if path.is_file():
                cached=read(path)
                assert cached["pins"]==pins and (cached["uid"],cached["task"])==(r["uid"],t)
                reused+=1
            else:
                todo.append((r,t))
        if todo:
            values=predict(processor,model,limit,group)
            for (f,t),record in zip(group,values,strict=True):
                if (f,t) in todo:
                    uid=f["uid"]
                    write(folder/(uid+"_"+t+".json"),dict(uid=uid,task=t,pins=pins,**record))
                    made+=1
        if (start_at+len(group))%128==0 or start_at==0 or start_at+len(group)==size:
            print("Fresh reads",start_at+len(group),"/",size,"done; elapsed",round(perf_counter()-t0),
                  "sec; calls this process",made,flush=True)
    checks={r["uid"]+"_"+t:sha(folder/(r["uid"]+"_"+t+".json")) for r,t in jobs}
    assert len(checks)==size==made+reused and size+inputs["reuse_queries"]==18620
    write(PRIVATE/"fresh_reply_hashes.json",checks)
    write(OUT/"qwen_seal.json",dict(pins=pins,queries=18620,fresh_outputs=size,produced_this_attempt=made,
          cached_original=inputs["reuse_queries"], responses=sha(PRIVATE/"fresh_reply_hashes.json"),
          batch_replay=sha(OUT/"batch_replay.json"),seconds=perf_counter()-start,
          torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),
          memory_peak_bytes=torch.cuda.max_memory_allocated(),target_labels_read=False,FINAL_access=False))
    print("Complete Qwen seal",len(checks),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir",type=Path)
    run(cli.parse_args().model_dir)

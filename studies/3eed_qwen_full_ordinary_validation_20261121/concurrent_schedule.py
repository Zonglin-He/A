"""Execute original batch-four groups on independent CUDA streams with shared frozen weights."""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoProcessor

from roster import HERE, OLD, OUT, PRIVATE, TASKS, panel, read, sha, write
from run_full_qwen import load_model
from run_original_schedule import BATCH, CONFIG, predict

STREAMS=3


def worker_inputs(model_dir, base, first_processor):
    workers=[]
    for i in range(STREAMS):
        # Keep per-stream rotary deltas and generation state, and share read-only weights.
        model=copy(base)
        model._modules=base._modules.copy()
        core=copy(base.model)
        core._modules=base.model._modules.copy()
        core.rope_deltas=None
        model._modules["model"]=core
        model.generation_config=deepcopy(base.generation_config)
        processor=first_processor if i==0 else AutoProcessor.from_pretrained(model_dir,local_files_only=True)
        processor.image_processor.size=first_processor.image_processor.size.copy()
        processor.tokenizer.padding_side="left"
        workers.append((processor,model,torch.cuda.Stream()))
    torch.cuda.synchronize()
    return workers


@torch.inference_mode()
def block(worker,limit,group):
    p,m,stream=worker
    with torch.cuda.stream(stream):
        answer=predict(p,m,limit,group)
    return answer


def read_games(pool,workers,limit,groups):
    futures=[pool.submit(block,w,limit,g) for w,g in zip(workers,groups,strict=False)]
    return [f.result() for f in futures]


def run(model_dir):
    started=perf_counter()
    frames,inputs=panel()
    original=read(OUT/"batch_replay.json")
    pins=original["pins"]
    assert pins["code"]==sha(HERE/"run_original_schedule.py") and BATCH==pins["batch"]==4
    assert sha(CONFIG)==pins["schedule"]
    folder=PRIVATE/"replies"
    todo=[(r,t) for r in frames if "archived" not in r for t in TASKS]
    groups=[todo[k:k+BATCH] for k in range(0,len(todo),BATCH)]

    def path(frame,task):
        return folder/(frame["uid"]+"_"+task+".json")

    completed=[g for g in groups if all(path(f,t).is_file() for f,t in g)]
    assert len(completed)>=18
    checks=completed[:18]
    p,model,limit=load_model(model_dir)
    workers=worker_inputs(model_dir,model,p)
    with ThreadPoolExecutor(max_workers=STREAMS) as pool:
        # Compare all text and parsed output values on exactly the same original blocks.
        tick=perf_counter()
        exact=0
        for j in range(0,len(checks),STREAMS):
            chunk=checks[j:j+STREAMS]
            values=read_games(pool,workers,limit,chunk)
            for g,answer in zip(chunk,values,strict=True):
                for (r,t),v in zip(g,answer,strict=True):
                    old=read(path(r,t))
                    assert old["pins"]==pins
                    same=all(v[k]==old[k] for k in ("raw","value","status","visual_grid","generated_tokens"))
                    exact+=int(same)
        parallel_seconds=perf_counter()-tick
        baseline=0.
        # Paired timing on the same first six blocks. Readback is already part of the timing.
        tick=perf_counter()
        for g in checks[:6]:
            values=block(workers[0],limit,g)
            baseline+=len(values)
        serial_seconds=perf_counter()-tick
        gain=(len(checks)*BATCH/parallel_seconds)/(baseline/serial_seconds)
        accepted=exact==len(checks)*BATCH and gain>1.12
        info=dict(code=sha(__file__),accepted=accepted,probed_outputs=len(checks)*BATCH,exact_outputs=exact,
                  max_concurrent_batches=STREAMS,parallel_seconds=parallel_seconds,serial_probe_seconds=serial_seconds,
                  serial_probe_outputs=baseline,throughput_factor=gain,
                  complete_original_blocks=True,no_new_GT=True,FINAL_access=False)
        write(OUT/"concurrency_replay.json",info)
        print(info,flush=True)
        if not accepted:
            return
        further=[g for g in groups if not all(path(f,t).is_file() for f,t in g)]
        done=len(groups)*BATCH-sum(map(len,further))
        actual=0
        prepared=perf_counter()
        for j in range(0,len(further),STREAMS):
            chunk=further[j:j+STREAMS]
            values=read_games(pool,workers,limit,chunk)
            for g,answer in zip(chunk,values,strict=True):
                for (r,t),v in zip(g,answer,strict=True):
                    target=path(r,t)
                    if target.is_file():
                        existing=read(target)
                        assert existing["pins"]==pins
                        continue
                    write(target,dict(uid=r["uid"],task=t,pins=pins,execution="batch4_concurrent",**v))
                    actual+=1
            if j%24==0 or j+len(chunk)==len(further):
                print("Fresh reads",(done+actual),"/",len(todo),
                      "new with CUDA scheduling",actual,"elapsed",round(perf_counter()-prepared),flush=True)

    hashes={f["uid"]+"_"+t:sha(path(f,t)) for f,t in todo}
    assert len(hashes)+inputs["reuse_queries"]==18620
    write(PRIVATE/"fresh_reply_hashes.json",hashes)
    write(OUT/"qwen_seal.json",dict(pins=pins,queries=18620,fresh_outputs=len(todo),produced_this_attempt=actual,
         cached_original=inputs["reuse_queries"],responses=sha(PRIVATE/"fresh_reply_hashes.json"),
         batch_replay=sha(OUT/"batch_replay.json"), concurrency_replay=sha(OUT/"concurrency_replay.json"),
         execution_code=sha(__file__),seconds=perf_counter()-started,torch=torch.__version__,
         transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),memory_peak_bytes=torch.cuda.max_memory_allocated(),
         target_labels_read=False,FINAL_access=False))
    print("Complete Qwen seal",len(hashes),flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir",type=Path)
    run(cli.parse_args().model_dir)

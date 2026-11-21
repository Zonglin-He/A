"""Post-primary coordinate-relabeling check. Uses frozen prompts and no evaluation labels."""
import argparse
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from side import (ARMS, HERE, OUT, PRIVATE, crop_photo, digest, inputs, parse, read, write)
from landmarks import check_assets


def reverse(text, legend):
    assert text.count(legend)==1 and legend.startswith("The positive direction runs ")
    assert legend.endswith("Negative is the opposite.")
    mirror = legend.replace("The positive direction", "The negative direction").replace(
                            "Negative is the opposite.", "Positive is the opposite.")
    return text.replace(legend, mirror)


@torch.inference_mode()
def run(model_dir):
    start = perf_counter()
    plan = HERE/"SIGN_CHECK.json"
    primary = read(OUT/"prediction_seal.json")
    seal = read(OUT/"input_seal.json")
    assert primary["pins"]["input_seal"]==digest(OUT/"input_seal.json")
    assert digest(PRIVATE/"prepared.json")==seal["prepared"]
    prepared = read(PRIVATE/"prepared.json")
    panel, frozen = inputs()
    pins = dict(primary=digest(OUT/"prediction_seal.json"), plan=digest(plan),
                code=digest(__file__), revision=check_assets(model_dir))
    assert pins["revision"]==primary["pins"]["model"]
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.image_processor.size = {"shortest_edge":65536,"longest_edge":524288}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
            use_safetensors=True, dtype=torch.bfloat16, device_map={"":"cuda:0"},
            attn_implementation="sdpa").eval().requires_grad_(False)
    folder = PRIVATE/"reversed_replies"
    folder.mkdir(exist_ok=True,mode=0o700)
    hashes = {}
    new = 0
    for row in panel:
        uid = row["uid"]
        item = prepared[uid]
        if "prompts" not in item:
            continue
        path = folder/(uid+".json")
        if not path.exists():
            photo, rect = crop_photo(row,frozen[uid])
            messages = []
            for arm in ARMS:
                text = reverse(item["prompts"][arm],item["legend"])
                msg = [{"role":"user","content":[{"type":"image"},{"type":"text","text":text}]}]
                messages.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
            batch = processor(text=messages,images=[photo]*4,padding=True,return_tensors="pt").to("cuda")
            tokens = model.generate(**batch,do_sample=False,max_new_tokens=32)
            outputs = processor.batch_decode(tokens[:,batch.input_ids.size(1):],skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
            write(path,dict(pins=pins,uid=uid,crop=rect,answers=dict(zip(ARMS,outputs,strict=True))))
            new += len(ARMS)
        obj = read(path)
        assert obj["pins"]==pins and obj["uid"]==uid
        hashes[uid]=digest(path)
        if len(hashes)%80==0:
            print("Relabeled frames",len(hashes),flush=True)
    assert len(hashes)==primary["prompt_frames"]
    write(PRIVATE/"reversed_hashes.json",hashes)
    write(OUT/"reversal_seal.json",dict(pins=pins, hashes=digest(PRIVATE/"reversed_hashes.json"),
          matched_frames=len(hashes),calls=len(hashes)*4,fresh_calls=new,runtime_seconds=perf_counter()-start,
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),target_labels_read=False,FINAL_access=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir",type=Path)
    run(cli.parse_args().model_dir)

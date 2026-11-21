"""Run matched arm prompts in independent greedy sequences; commit every frame atomically."""
import argparse
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from side import (ARMS, HERE, OUT, PRIVATE, QWEN, SPEC, crop_photo, digest,
                  inputs, parse, proposed_box, read, write)
from landmarks import check_assets


@torch.inference_mode()
def run(model_dir):
    t = perf_counter()
    seal = read(OUT/"input_seal.json")
    assert seal["prepared"] == digest(PRIVATE/"prepared.json") and seal["protocol"] == digest(SPEC)
    assert seal["geometry_code"] == digest(HERE/"side.py") and not seal["target_labels_read"]
    panel, frozen = inputs()
    prepared = read(PRIVATE/"prepared.json")
    revision = check_assets(model_dir)
    pin = dict(model=revision, input_seal=digest(OUT/"input_seal.json"), inference_code=digest(__file__),
               crop_code=digest(HERE.parent/"3eed_visible_amodal_span_20261121"/"common.py"),
               assets=digest(QWEN/"MODEL.json"))
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.image_processor.size = {"shortest_edge":65536, "longest_edge":524288}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_dir, local_files_only=True,
            use_safetensors=True, dtype=torch.bfloat16, device_map={"":"cuda:0"},
            attn_implementation="sdpa").eval().requires_grad_(False)
    folder = PRIVATE/"replies"
    folder.mkdir(exist_ok=True, mode=0o700)
    replies = {}
    boxes = {}
    calls = fresh = 0
    for row in panel:
        uid = row["uid"]
        item = prepared[uid]
        h = item["H"]
        sides, output, deltas = {}, {}, {}
        if "prompts" in item:
            path = folder/(uid+".json")
            if not path.is_file():
                img, rect = crop_photo(row, frozen[uid])
                templates = []
                for arm in ARMS:
                    msg = [{"role":"user","content":[{"type":"image"},{"type":"text","text":item["prompts"][arm]}]}]
                    templates.append(processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
                batch = processor(text=templates, images=[img]*4, padding=True, return_tensors="pt").to("cuda")
                generated = model.generate(**batch, do_sample=False, max_new_tokens=32)
                decoded = processor.batch_decode(generated[:,batch.input_ids.size(1):], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                write(path, dict(uid=uid, pins=pin, crop=rect, answers=dict(zip(ARMS, decoded, strict=True))))
                fresh += 4
            replay = read(path)
            assert replay["pins"] == pin and replay["uid"] == uid
            sides = {a:parse(replay["answers"][a]) for a in ARMS}
            replies[uid] = digest(path)
            calls += 4
            if len(replies) % 40 == 0:
                print("Matched frames", len(replies), "of", seal["counts"]["promptable"], flush=True)
        for arm in ARMS:
            side = sides.get(arm, "unavailable")
            delta = item.get("geometry",{}).get("offsets",{}).get(side, 0.)
            output[arm] = proposed_box(h, item.get("geometry",{}).get("axis"), delta)
            deltas[arm] = delta
        options = {s:proposed_box(h, item["geometry"]["axis"], value)
                   for s,value in item.get("geometry",{}).get("offsets",{}).items()}
        boxes[uid] = dict(H=h, boxes=output, sides=sides, offsets_m=deltas, discrete_boxes=options)
    assert calls == 4 * seal["counts"]["promptable"]
    write(PRIVATE/"predictions.json", boxes)
    write(PRIVATE/"reply_hashes.json", replies)
    write(OUT/"prediction_seal.json", dict(pins=pin, frames=len(boxes), prompt_frames=len(replies),
        calls=calls, fresh_calls=fresh, reply_hashes=digest(PRIVATE/"reply_hashes.json"),
        predictions=digest(PRIVATE/"predictions.json"), model_id="Qwen/Qwen3-VL-4B-Instruct",
        runtime_seconds=perf_counter()-t, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=transformers.__version__,
        target_labels_read=False, FINAL_access=False))
    print("Predictions sealed", len(boxes), calls, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir", type=Path)
    run(cli.parse_args().model_dir)

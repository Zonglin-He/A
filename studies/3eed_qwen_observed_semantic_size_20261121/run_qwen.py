"""Generate both semantic labels and the distinct downstream sizes with frozen Qwen."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import torch
import transformers
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from study import (OUT, PRIVATE, SEMANTIC_ARMS, crop_photo, digest, parse_semantic, read, roster,
                   semantic_prompt, size_requests, text_hash, verified_observations, write)
from landmarks import check_assets

FRAME_BATCH = 4
MAX_NEW_TOKENS = 80


def generate(model, processor, jobs):
    if not jobs:
        return []
    text = [
        processor.apply_chat_template([{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}
        ]}], tokenize=False, add_generation_prompt=True)
        for _, prompt in jobs
    ]
    batch = processor(text=text, images=[image for image, _ in jobs], padding=True, return_tensors="pt").to("cuda")
    tokens = model.generate(**batch, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
    answers = processor.batch_decode(tokens[:, batch.input_ids.shape[1]:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
    return [dict(answer=answer, prompt_sha256=text_hash(prompt))
            for answer, (_, prompt) in zip(answers, jobs, strict=True)]


def read_block(model, processor, frames, baseline, observations, pins, directory):
    inputs = []
    semantic_jobs = []
    for frame in frames:
        uid = frame["uid"]
        image, rect = crop_photo(frame, baseline[uid])
        spans = observations[uid]["spans_lwh"]
        inputs.append((frame, image, rect, spans))
        semantic_jobs.extend([
            (image, semantic_prompt(frame["text"])),
            (image, semantic_prompt(frame["text"], spans))
        ])
    semantic_answers = generate(model, processor, semantic_jobs)
    results = []
    size_jobs = []
    for i, (frame, image, rect, spans) in enumerate(inputs):
        replies = dict(zip(SEMANTIC_ARMS, semantic_answers[2*i:2*i+2], strict=True))
        labels = {arm: parse_semantic(replies[arm]["answer"]) for arm in SEMANTIC_ARMS}
        prompts, size_key = size_requests(frame["text"], spans, labels)
        size_jobs.extend((image, prompt) for prompt in prompts.values())
        results.append((frame["uid"], dict(
            pins=pins, crop=rect, image_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
            semantic=replies, size_key=size_key
        ), list(prompts)))
    size_answers = iter(generate(model, processor, size_jobs))
    for uid, record, keys in results:
        write(directory / (uid + ".json"), dict(
            uid=uid, **record, sizes={key: next(size_answers) for key in keys}
        ))
    assert next(size_answers, None) is None
    return len(semantic_jobs), len(size_jobs)


@torch.inference_mode()
def run(model_dir):
    started = perf_counter()
    panel, baseline = roster()
    observations = verified_observations()
    frames = [f for f in panel if observations[f["uid"]]["selected"]]
    assert len(frames) == 320
    pins = dict(inputs=digest(OUT / "input_seal.json"), inference=digest(__file__),
                revision=check_assets(model_dir), frame_batch=FRAME_BATCH, max_new_tokens=MAX_NEW_TOKENS)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 524288}
    processor.tokenizer.padding_side = "left"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
        device_map={"": "cuda:0"}, attn_implementation="sdpa"
    ).eval().requires_grad_(False)
    directory = PRIVATE / "replies"
    directory.mkdir(mode=0o700, exist_ok=True)

    fresh_semantic = fresh_size = 0
    for start in range(0, len(frames), FRAME_BATCH):
        block = [f for f in frames[start:start+FRAME_BATCH] if not (directory / (f["uid"] + ".json")).is_file()]
        if block:
            semantic_count, size_count = read_block(model, processor, block, baseline, observations, pins, directory)
            fresh_semantic += semantic_count
            fresh_size += size_count
        if (start + FRAME_BATCH) % 32 == 0:
            print("Frames", start + FRAME_BATCH, "/", len(frames),
                  "fresh calls", fresh_semantic + fresh_size, flush=True)

    manifest = {}
    semantic_calls = size_calls = direct_calls = 0
    for frame in frames:
        uid = frame["uid"]
        path = directory / (uid + ".json")
        record = read(path)
        assert record["uid"] == uid and record["pins"] == pins
        assert set(record["semantic"]) == set(SEMANTIC_ARMS) and "GD" in record["sizes"]
        manifest[uid] = digest(path)
        semantic_calls += len(record["semantic"])
        size_calls += len(record["sizes"])
        direct_calls += 1
    assert semantic_calls == 640 and direct_calls == 320 and size_calls <= 960
    write(PRIVATE / "reply_hashes.json", manifest)
    write(OUT / "model_seal.json", dict(
        pins=pins, frames=len(frames), replies=digest(PRIVATE / "reply_hashes.json"),
        calls=semantic_calls + size_calls, semantic_calls=semantic_calls, size_calls=size_calls,
        direct_size_calls=direct_calls, distinct_conditioned_size_calls=size_calls - direct_calls,
        fresh_calls=fresh_semantic + fresh_size, seconds=perf_counter() - started,
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), target_labels_read=False, FINAL_access=False
    ))
    print("Sealed model reads", semantic_calls + size_calls, "on", len(manifest), "frames.", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir", type=Path)
    run(cli.parse_args().model_dir)

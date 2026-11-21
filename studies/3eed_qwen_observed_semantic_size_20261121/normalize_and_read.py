"""Reuse identical size requests and fill only the gap left by taxonomy field repair."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from normalization import resolve
from run_qwen import generate
from study import (HERE, OUT, PRIVATE, crop_photo, digest, read, roster, verified_observations, write)
from landmarks import check_assets


@torch.inference_mode()
def run(model_dir):
    started = perf_counter()
    model_seal = read(OUT / "model_seal.json")
    assert not model_seal["target_labels_read"] and not model_seal["FINAL_access"]
    assert model_seal["pins"]["inputs"] == digest(OUT / "input_seal.json")
    assert model_seal["pins"]["inference"] == digest(HERE / "run_qwen.py")
    assert model_seal["replies"] == digest(PRIVATE / "reply_hashes.json")
    original = read(PRIVATE / "reply_hashes.json")
    observations = verified_observations()
    panel, baseline = roster()
    dirs = PRIVATE / "resolved"
    dirs.mkdir(exist_ok=True, mode=0o700)
    pins = dict(
        model_seal=digest(OUT / "model_seal.json"), amendment=digest(HERE / "AMENDMENT.json"),
        normalization=digest(HERE / "normalization.py"), code=digest(__file__), revision=model_seal["pins"]["revision"]
    )
    frames = [f for f in panel if f["uid"] in original]
    model = processor = None
    additional = 0
    for start in range(0, len(frames), 8):
        pending = []
        jobs = []
        for frame in frames[start:start+8]:
            uid = frame["uid"]
            out = dirs / (uid + ".json")
            if out.is_file():
                assert read(out)["pins"] == pins
                continue
            source = PRIVATE / "replies" / (uid + ".json")
            assert digest(source) == original[uid]
            raw = read(source)
            labels, statuses, routes, same, missing = resolve(frame, observations[uid]["spans_lwh"], raw)
            pending.append((out, dict(uid=uid, pins=pins, source=original[uid], labels=labels, status=statuses,
                                      size_key=routes, sizes=same, generated=list(missing))))
            if missing:
                image, crop = crop_photo(frame, baseline[uid])
                assert crop == raw["crop"] and hashlib.sha256(image.tobytes()).hexdigest() == raw["image_sha256"]
                jobs.extend((image, prompt) for prompt in missing.values())

        if jobs and model is None:
            assert check_assets(model_dir) == pins["revision"]
            processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
            processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 524288}
            processor.tokenizer.padding_side = "left"
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_dir, local_files_only=True, use_safetensors=True, dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa"
            ).eval().requires_grad_(False)
        responses = iter(generate(model, processor, jobs))
        for path, r in pending:
            r["sizes"].update({k: next(responses) for k in r["generated"]})
            write(path, r)
        assert next(responses, None) is None
        additional += len(jobs)

    manifest = {}
    calls = 0
    for uid in original:
        path = dirs / (uid + ".json")
        record = read(path)
        assert record["pins"] == pins and record["source"] == original[uid]
        manifest[uid] = digest(path)
        calls += len(record["generated"])
    write(PRIVATE / "resolved_hashes.json", manifest)
    write(OUT / "normalized_seal.json", dict(
        pins=pins, frames=len(manifest), manifest=digest(PRIVATE / "resolved_hashes.json"), additional_size_calls=calls,
        fresh_calls=additional, seconds=perf_counter() - started, target_labels_read=False, FINAL_access=False
    ))
    print("Sealed normalized labels for", len(manifest), "frames. Extra size reads", calls, flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model_dir", type=Path)
    run(cli.parse_args().model_dir)

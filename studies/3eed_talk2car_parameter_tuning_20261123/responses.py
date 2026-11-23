"""Replay locked Qwen prompts and compare bounded inference settings."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
TRANSFER = HERE.parent / "3eed_height_only_transfer_20261123"
sys.path.insert(0, str(TRANSFER))

import nuscenes_eval as external
from common import prior, read_payload, roster, save_payload


OUT = HERE / "results"
PRIVATE = HERE / "private"
PROTOCOL = HERE / "PROTOCOL.json"
TASKS = ("3d", "2d")


def registered():
    config = roster.read(PROTOCOL)
    assert config["FINAL_access"] is False
    assert roster.sha(HERE.parents[1] / "QWEN_CURRENT_METHOD.json") == config["model"]["method_registry_sha256"]
    assert roster.sha(external.INPUT_SEAL) == config["population"]["input_seal_sha256"]
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    return config


def identity():
    return dict(protocol=roster.sha(PROTOCOL), code=roster.sha(__file__), input=roster.sha(external.INPUT_SEAL),
                archived_qwen=roster.sha(external.REPLY_SEAL))


def load(profile):
    config = registered()
    if config["generation"]["profiles"][profile].get("archived"):
        return list(external.read_replies())
    receipt = roster.read(OUT / f"{profile}_receipt.json")
    payload = PRIVATE / f"{profile}_responses.jsonl.gz"
    assert receipt["pins"] == identity() and receipt["profile"] == config["generation"]["profiles"][profile]
    assert roster.sha(payload) == receipt["payload"] and receipt["target_gt_opened"] is False
    return list(read_payload(payload))


def prompt_commands(prompts, variant, suffix):
    if variant == "original":
        return prompts
    assert variant == "single_target"
    output = {}
    for key, body in prompts.items():
        schema, description = body.rsplit("Description: ", 1)
        assert description == ""
        output[key] = schema + suffix.strip() + " Description: "
    return output


def run(weights):
    from concurrent_schedule import STREAMS, block, read_games, worker_inputs
    import run_original_schedule as original_schedule
    from run_full_qwen import load_model
    import torch

    start = perf_counter()
    cfg = registered()
    frames, manifest = external.panel()
    archived = load("original_144")
    assert len(frames) == len(archived) == cfg["population"]["descriptions"] == manifest["descriptions"]
    assert [v["uid"] for v in frames] == [r["uid"] for r in archived]
    prior_answers = {r["uid"]: r["replies"] for r in archived}
    profiles = cfg["generation"]["profiles"]
    pending = [key for key, v in profiles.items() if not v.get("archived") and not (OUT / f"{key}_receipt.json").exists()]
    for key, v in profiles.items():
        if not v.get("archived") and key not in pending:
            assert len(load(key)) == len(frames)
    if not pending:
        return

    print(f"Loading the locked Qwen checkpoint for {len(pending)} inference profiles", flush=True)
    processor, model, frozen_limit = load_model(weights)
    assert frozen_limit == profiles["original_144"]["max_new_tokens"]
    workers = worker_inputs(weights, model, processor)
    work = [(frame, task) for frame in frames for task in TASKS]
    groups = [work[i:i + original_schedule.BATCH] for i in range(0, len(work), original_schedule.BATCH)]
    probe_groups = groups[:STREAMS]
    base_probe = [block(workers[0], frozen_limit, jobs) for jobs in probe_groups]
    for group, answers in zip(probe_groups, base_probe, strict=True):
        for (frame, task), result in zip(group, answers, strict=True):
            assert result == prior_answers[frame["uid"]][task], "Archived original-144 replay mismatch"
    print(f"Archived baseline reproduced on {sum(map(len, probe_groups))} answers", flush=True)
    default_prompts = original_schedule.PROMPTS.copy()

    with ThreadPoolExecutor(max_workers=STREAMS) as pool:
        for key in pending:
            profile_start = perf_counter()
            profile = profiles[key]
            limit = profile["max_new_tokens"]
            prompts = prompt_commands(default_prompts, profile["prompt"], cfg["generation"]["single_target_suffix"])
            torch.cuda.reset_peak_memory_stats()
            with patch.object(original_schedule, "PROMPTS", prompts):
                normal = [block(workers[0], limit, jobs) for jobs in probe_groups]
                simultaneous = read_games(pool, workers, limit, probe_groups)
                streams = STREAMS if simultaneous == normal else 1
                outcomes = {frame["uid"]: {} for frame in frames}
                for index in range(0, len(groups), streams):
                    batch = groups[index:index + streams]
                    answers = read_games(pool, workers, limit, batch)
                    for jobs, completed in zip(batch, answers, strict=True):
                        for (frame, task), answer in zip(jobs, completed, strict=True):
                            outcomes[frame["uid"]][task] = answer
                    if index == 0 or index % 24 == 0:
                        done = sum(len(x) for x in groups[:index + streams])
                        print(f"{key} {done}/{len(work)}", flush=True)
            assert all(set(x) == set(TASKS) for x in outcomes.values())
            replay_below_budget = None
            if key == "original_256":
                below_budget = [(frame["uid"], task) for frame, task in work
                                if prior_answers[frame["uid"]][task]["generated_tokens"] < frozen_limit]
                matches = sum(outcomes[uid][task] == prior_answers[uid][task] for uid, task in below_budget)
                replay_below_budget = dict(total=len(below_budget), exact=matches)
                assert matches == len(below_budget), "Length-only comparison changed an answer that had already stopped"
            payload = save_payload(PRIVATE / f"{key}_responses.jsonl.gz",
                (dict(uid=uid, replies=replies) for uid, replies in outcomes.items()))
            status = {task: dict(Counter(row[task]["status"] for row in outcomes.values())) for task in TASKS}
            roster.write(OUT / f"{key}_receipt.json", dict(status=status, pins=identity(), profile=profile,
                payload=payload, descriptions=len(frames), queries=len(work), streams=streams, batch=original_schedule.BATCH,
                archives_probe=sum(map(len, probe_groups)), replay_below_original_cap=replay_below_budget,
                elapsed_seconds=perf_counter() - profile_start, seen_wall_seconds=perf_counter() - start,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), gpu=torch.cuda.get_device_name(),
                target_gt_opened=False, FINAL_access=False))
            print(f"{key} sealed {len(work)} answers with status counts {status}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    run(parser.parse_args().model)

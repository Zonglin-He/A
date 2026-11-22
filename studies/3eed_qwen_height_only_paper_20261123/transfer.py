"""Register and read frozen Qwen on the fixed recording-disjoint Ordinary TRAIN panel."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from time import perf_counter
from functools import lru_cache


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STUDIES = HERE.parent
FULL = STUDIES / "3eed_qwen_full_ordinary_validation_20261121"
TRAIN = STUDIES / "3eed_v1_supervised_instance_grounding_20261119"
OUT = HERE / "results"
PRIVATE = HERE / "private"
PANEL = PRIVATE / "extension_panel.json"
INPUT_SEAL = OUT / "extension_input_seal.json"
QWEN_SEAL = OUT / "extension_qwen_seal.json"
REPLY_INDEX = PRIVATE / "extension_reply_hashes.json"
PROTOCOL = HERE / "PROTOCOL.json"

sys.path.insert(0, str(FULL))
import roster


def recording(frame):
    return frame["target"], Path(frame["calibration"]).parent.parent.name


def register():
    sys.path.insert(0, str(TRAIN))
    from scope import DATA, canonical, evaluation_rows, protocol

    started = perf_counter()
    manifest = roster.read(ROOT / "QWEN_CURRENT_METHOD.json")
    cfg = roster.read(PROTOCOL)
    assert manifest["method"] == "ray_equivalent_height_only"
    assert cfg["main_lock_sha256"] == roster.sha(ROOT / manifest["method_lock"]["file"])
    assert not cfg["FINAL_access"]
    source_rows = evaluation_rows()
    validation, val_seal = roster.panel()
    val_records = set(map(recording, validation))
    val_images = {r["rgb_sha"] for r in validation}
    val_points = {r["point_sha"] for r in validation}
    platforms = ("waymo", "drone", "quad")
    train_records = {p: set((DATA / "splits" / f"{p}_train.txt").read_text().splitlines()) for p in platforms}
    old, sources = roster.old_replies()
    file_sha = lru_cache(maxsize=None)(roster.sha)
    views = ("F", "FL", "FR", "SL", "SR")
    frames = []
    reused = Counter()
    for row in source_rows:
        target = row["platform"]
        meta = Path(row["metapath"])
        rec = meta.parent.parent.name
        assert meta.name == "meta_info.json" and meta.parents[2] == DATA / target
        assert rec in train_records[target] and (target, rec) not in val_records
        image_sha = file_sha(Path(row["imagepath"]))
        point_sha = file_sha(Path(row["pointpath"]))
        assert image_sha == row["image_sha256"] and image_sha not in val_images
        assert point_sha == row["point_sha256"] and point_sha not in val_points
        original = roster.captions(row["metapath"])
        hits = [i for i, text in enumerate(original) if canonical(text) == canonical(row["caption"])]
        assert len(hits) == 1, ("ordinary train caption slot is ambiguous", target)
        slot, = hits
        suffix = row["suffix"] if target == "waymo" else 0
        frame = dict(
            uid=roster.key("paper-extension-description", target, rec, meta.parent.name, slot),
            target=target, recording=roster.key("paper-extension-recording", target, rec),
            panel="recording_extension", slot=slot, suffix=suffix, waymo_view=views[suffix],
            text=original[slot], rgb=row["imagepath"], points=row["pointpath"], calibration=row["metapath"],
            rgb_sha=image_sha, point_sha=point_sha,
            seed=int(roster.key("R97point", f"{target}|{rec}/{meta.parent.name}/meta_info.json|{slot}")[:16], 16),
        )
        cached = old.get((image_sha, frame["text"]))
        if cached is not None:
            directory, uid, hashes, pins = cached
            for task in roster.TASKS:
                key = f"{uid}_{task}"
                path = directory / "private" / "replies" / (key + ".json")
                assert roster.sha(path) == hashes[key]
                reply = roster.read(path)
                assert (reply["uid"], reply["task"], reply["pins"]) == (uid, task, pins)
                frame.setdefault("archived", {})[task] = {
                    **{k: reply[k] for k in ("raw", "value", "status", "visual_grid", "generated_tokens")},
                    "source_sha256": hashes[key],
                }
            reused[target] += 1
        frames.append(frame)
    n = Counter(f["target"] for f in frames)
    assert n == {p: 512 for p in platforms} and len({f["uid"] for f in frames}) == len(frames) == 1536
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    roster.write(PANEL, frames)
    roster.write(INPUT_SEAL, dict(
        panel=roster.sha(PANEL), protocol=roster.sha(PROTOCOL), code=roster.sha(__file__),
        approved_roster=protocol()["evaluation"]["sha256"], validation_panel=val_seal["panel"],
        descriptions=dict(n), recordings={p: len({f["recording"] for f in frames if f["target"] == p}) for p in platforms},
        reused_descriptions=dict(reused), cached_reply_seals=sources, disjoint_recordings_rgb_lidar=True,
        target_gt_opened=False, FINAL_access=False, seconds=perf_counter()-started,
    ))
    print("Sealed recording-disjoint panel", dict(n), "cached descriptions", dict(reused), flush=True)


def read_panel():
    seal = roster.read(INPUT_SEAL)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["protocol"] == roster.sha(PROTOCOL) and seal["panel"] == roster.sha(PANEL)
    assert seal["code"] == roster.sha(__file__)
    rows = roster.read(PANEL)
    assert len(rows) == 1536 and all(f["panel"] == "recording_extension" for f in rows)
    return rows


def infer(weights):
    import torch
    import transformers
    from concurrent_schedule import STREAMS, block, read_games, worker_inputs
    from run_full_qwen import load_model

    started = perf_counter()
    raw = read_panel()
    folder = PRIVATE / "extension_replies"
    folder.mkdir(exist_ok=True, mode=0o700)
    cfg = roster.read(PROTOCOL)
    assert cfg["revision"] == roster.read(roster.OLD / "MODEL.json")["revision"]
    pins = dict(
        inputs=roster.sha(INPUT_SEAL), protocol=roster.sha(PROTOCOL), code=roster.sha(__file__),
        assets=roster.sha(roster.OLD / "MODEL.json"), revision=cfg["revision"],
        helper=roster.sha(FULL / "run_original_schedule.py"),
        streams=roster.sha(FULL / "concurrent_schedule.py"), batch=4,
    )
    groups, probes = [], []
    for target in ("waymo", "drone", "quad"):
        jobs = [(f, task) for f in raw if f["target"] == target and "archived" not in f for task in roster.TASKS]
        blocks = [jobs[j:j+4] for j in range(0, len(jobs), 4)]
        groups.extend(blocks)
        probes.extend(blocks[:2])

    def path(f, task):
        return folder / f"{f['uid']}_{task}.json"

    def existing(f, task):
        path_ = path(f, task)
        if not path_.is_file():
            return False
        reply = roster.read(path_)
        assert (reply["uid"], reply["task"], reply["pins"]) == (f["uid"], task, pins)
        return True

    pending = [g for g in groups if not all(existing(f, task) for f, task in g)]
    cache = 3072 - sum(map(len, groups))
    made = 0
    with torch.inference_mode():
        print("Loading verified weights. New descriptions", sum(map(len, pending)) // 2, flush=True)
        processor, model, limit = load_model(weights)
        workers = worker_inputs(weights, model, processor)
        with ThreadPoolExecutor(max_workers=STREAMS) as pool:
            exact = 0
            for j in range(0, len(probes), STREAMS):
                bunch = probes[j:j+STREAMS]
                serial = [block(workers[0], limit, g) for g in bunch]
                parallel = read_games(pool, workers, limit, bunch)
                exact += sum(x == y for a, b in zip(serial, parallel, strict=True) for x, y in zip(a, b, strict=True))
            probe_size = sum(map(len, probes))
            concurrency = STREAMS if exact == probe_size else 1
            print("Replay matched", exact, "/", probe_size, "outputs using", concurrency, "streams", flush=True)
            groups_done = len(groups) - len(pending)
            for j in range(0, len(pending), concurrency):
                bunch = pending[j:j+concurrency]
                values = read_games(pool, workers, limit, bunch) if concurrency > 1 else [block(workers[0], limit, bunch[0])]
                for group, replies in zip(bunch, values, strict=True):
                    for (f, task), reply in zip(group, replies, strict=True):
                        if not existing(f, task):
                            roster.write(path(f, task), dict(uid=f["uid"], task=task, pins=pins, **reply))
                            made += 1
                    groups_done += 1
                    if groups_done % 25 == 0 or groups_done == len(groups):
                        print("Completed groups", groups_done, "/", len(groups), "new replies", made,
                              "elapsed seconds", round(perf_counter() - started, 1), flush=True)
    hashes = {f"{f['uid']}_{task}": roster.sha(path(f, task)) for g in groups for f, task in g}
    assert len(hashes) + cache == 3072
    roster.write(REPLY_INDEX, hashes)
    roster.write(QWEN_SEAL, dict(
        pins=pins, input=roster.sha(INPUT_SEAL), code=roster.sha(__file__), replies=roster.sha(REPLY_INDEX),
        queries=3072, cached_prior_replies=cache, new_outputs=len(hashes), produced_this_process=made,
        concurrent_matches=exact, cross_schedule_checked=probe_size, workers=concurrency,
        torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
        memory_peak_bytes=torch.cuda.max_memory_allocated(), seconds=perf_counter()-started,
        target_gt_opened=False, FINAL_access=False,
    ))
    print("All 3,072 extension replies sealed", flush=True)


def replies(frame, seal, hashes):
    if "archived" in frame:
        return frame["archived"]
    result = {}
    for task in roster.TASKS:
        key = f"{frame['uid']}_{task}"
        path = PRIVATE / "extension_replies" / (key + ".json")
        assert roster.sha(path) == hashes[key]
        reply = roster.read(path)
        assert (reply["uid"], reply["task"], reply["pins"]) == (frame["uid"], task, seal["pins"])
        result[task] = reply
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "infer"))
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.phase == "register":
        register()
    else:
        if args.model is None:
            parser.error("infer requires --model")
        infer(args.model)

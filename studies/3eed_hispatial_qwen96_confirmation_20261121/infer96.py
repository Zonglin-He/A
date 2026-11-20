"""Generate paired answers using the released frozen HiSpatial model."""
import argparse
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
import torch

from shared import ARMS, ASSETS, BASE, HERE, OUT, PLATFORMS, PRIVATE, PROTOCOL, digest, read, write
from geometry import make_xyz
from infer import feature, load_released, query_batch, read_rgb, tokens_clean
from maps import map_arm


def load_xyz(file, arm, uid, oracle_hashes):
    if arm != "oracle_depth":
        return torch.from_numpy(map_arm(file, arm))
    path = PRIVATE / "oracle_depth" / (uid + ".npz")
    assert digest(path) == oracle_hashes[uid]
    with np.load(path, allow_pickle=False) as oracle:
        return torch.from_numpy(make_xyz(oracle["depth"], file["mask"], file["rays"]))


def smoke(official, frames, questions, map_hashes, oracle_hashes, batch_size):
    checks = []
    for target in PLATFORMS:
        frame = next(f for f in frames if f["target"] == target and
                     {q["level"] for q in questions[f["uid"]]} == {"L0", "L1", "L2", "L3"})
        uid = frame["uid"]
        items = questions[uid]
        sample = [next(q for q in items if q["level"] == level) for level in ("L0", "L1", "L2", "L3")]
        candidates = sample + [q for q in items if q not in sample]
        sample = candidates[:max(4, batch_size)]
        assert len(sample) >= batch_size
        prompts = [q["prompt"] for q in sample]
        rgb = read_rgb(frame, "language")
        path = PRIVATE / "maps" / (uid + ".npz")
        assert digest(path) == map_hashes[uid]
        with np.load(path, allow_pickle=False) as file:
            for arm in ("moge", "oracle_depth"):
                xyz = load_xyz(file, arm, uid, oracle_hashes)
                released = [tokens_clean(official.query(rgb, prompt, xyz_values=xyz)) for prompt in prompts]
                fused = feature(official, rgb, xyz)
                singles = [tokens_clean(query_batch(official, rgb, xyz, [prompt], fused)[0]) for prompt in prompts]
                batches = []
                for start in range(0, len(prompts), batch_size):
                    batches.extend(tokens_clean(a) for a in query_batch(official, rgb, xyz, prompts[start:start+batch_size], fused))
                for q, a, b, c in zip(sample, released, singles, batches):
                    checks.append(dict(qid=q["qid"], target=target, level=q["level"], arm=arm,
                                       cached_exact=(a == b), batch_exact=(a == c)))
    return checks


def run(sources, weights, batch_size):
    plan = HERE / "ANALYSIS_PLAN.json"
    panel = read(OUT / "panel_seal.json")
    for field, name in (("questions", "questions"), ("frames", "frames"), ("coordinate_prompts", "coordinates")):
        assert panel[field] == digest(PRIVATE / (name + ".json"))
    assert panel["protocol"] == digest(PROTOCOL)
    for seal, hashes in (("maps_seal", "map_hashes"), ("oracle_seal", "oracle_hashes")):
        assert read(OUT / (seal + ".json"))["hashes"] == digest(PRIVATE / (hashes + ".json"))
    map_hashes = read(PRIVATE / "map_hashes.json")
    oracle_hashes = read(PRIVATE / "oracle_hashes.json")
    frames = read(PRIVATE / "frames.json")
    questions = defaultdict(list)
    for q in read(PRIVATE / "questions.json"):
        questions[q["uid"]].append(q)
    coord = {q["qid"]: q["prompt"] for q in read(PRIVATE / "coordinates.json")}
    assert set(coord) == {q["qid"] for group in questions.values() for q in group if q["level"] == "L3"}
    analysis_code = {name: digest(HERE / name) for name in ("score96.py", "diagnose96.py", "recount96.py")}
    analysis_seal = dict(protocol=digest(PROTOCOL), plan=digest(plan), files=analysis_code)
    if (OUT / "analysis_seal.json").exists():
        assert read(OUT / "analysis_seal.json") == analysis_seal
    else:
        write(OUT / "analysis_seal.json", analysis_seal)
    started = time.perf_counter()
    official, load = load_released(sources, weights)
    official.processor.tokenizer.padding_side = "left"
    torch.manual_seed(20261121)
    checks = smoke(official, frames, questions, map_hashes, oracle_hashes, batch_size)
    cache_ok = all(c["cached_exact"] for c in checks)
    batch_ok = cache_ok and all(c["batch_exact"] for c in checks)
    used_batch = batch_size if batch_ok else 1
    proof = dict(queries=len(checks), exact_cached=sum(c["cached_exact"] for c in checks),
                 exact_batch=sum(c["batch_exact"] for c in checks), effective_batch_size=used_batch,
                 checkpoint=load, frozen=True, checks=checks, model_and_smoke_seconds=time.perf_counter()-started)
    write(OUT / "official_smoke.json", proof)
    print(f"Released interface matches cached {proof['exact_cached']}/{len(checks)}, "
          f"batch {proof['exact_batch']}/{len(checks)}. Running batch {used_batch}.", flush=True)
    stamp = dict(panel=digest(OUT / "panel_seal.json"), maps=digest(OUT / "maps_seal.json"),
                 oracle=digest(OUT / "oracle_seal.json"), assets=digest(ASSETS), code=digest(__file__),
                 inherited_infer=digest(BASE / "infer.py"), cache=cache_ok, batch=used_batch,
                 analysis=digest(OUT / "analysis_seal.json"))
    folder = PRIVATE / "answers_frozen"
    folder.mkdir(mode=0o700, exist_ok=True)
    hashes, fresh, total = {}, 0, 0
    began = time.perf_counter()
    for frame in frames:
        uid = frame["uid"]
        path = folder / (uid + ".json")
        items = questions[uid]
        expected = len(items)*len(ARMS) + sum(q["qid"] in coord for q in items)
        if path.exists():
            previous = read(path)
            assert previous["stamp"] == stamp and len(previous["answers"]) == expected
        else:
            rgb = read_rgb(frame, "language")
            map_path = PRIVATE / "maps" / (uid + ".npz")
            assert digest(map_path) == map_hashes[uid]
            records = []
            with np.load(map_path, allow_pickle=False) as file:
                for arm in ARMS:
                    xyz = load_xyz(file, arm, uid, oracle_hashes)
                    fused = feature(official, rgb, xyz) if cache_ok else None
                    requests = [(arm, q["qid"], q["prompt"]) for q in items]
                    if arm == "prior_depth":
                        requests.extend(("oracle_centers", q["qid"], coord[q["qid"]]) for q in items if q["qid"] in coord)
                    for start in range(0, len(requests), used_batch):
                        batch = requests[start:start+used_batch]
                        prompts = [row[2] for row in batch]
                        if cache_ok:
                            outputs = query_batch(official, rgb, xyz, prompts, fused)
                        else:
                            outputs = [official.query(rgb, p, xyz_values=xyz) for p in prompts]
                        assert len(batch) == len(outputs)
                        records.extend(dict(arm=out_arm, qid=qid, answer=tokens_clean(output))
                                       for (out_arm, qid, _), output in zip(batch, outputs))
            assert len(records) == expected
            write(path, dict(stamp=stamp, answers=records))
            fresh += expected
        hashes[uid] = digest(path)
        total += expected
        if len(hashes) % 4 == 0:
            print(f"HiSpatial {len(hashes)}/96 physical frames, {total} responses", flush=True)
    assert total == panel["question_count"]*len(ARMS) + panel["coordinate_count"] and len(hashes) == 96
    write(PRIVATE / "answer_hashes.json", hashes)
    write(OUT / "prediction_seal.json", dict(stamp=stamp, main_answers=panel["question_count"]*len(ARMS),
          coordinate_oracle_answers=panel["coordinate_count"], total=total, frames=len(hashes), fresh=fresh,
          answers_hashes=digest(PRIVATE / "answer_hashes.json"), elapsed_seconds=time.perf_counter()-began,
          gpu=torch.cuda.get_device_name(), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(), torch=torch.__version__, transformers=version("transformers"),
          target_updates=0, ordinary_scoring_key_opened=False, FINAL_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--batch-size", type=int, default=6, choices=range(1, 7))
    args = parser.parse_args()
    run(args.sources, args.weights, args.batch_size)

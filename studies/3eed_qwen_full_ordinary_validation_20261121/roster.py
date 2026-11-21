"""Verify and materialize the existing 9,310 ordinary target descriptions."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
ROOT = HERE.parents[1]
OUT, PRIVATE, SPEC = HERE/"results", HERE/"private", HERE/"PROTOCOL.json"
OLD = STUDIES/"3eed_frozen_qwen_metric_grounding_20261120"
CONFIRM = STUDIES/"3eed_qwen_four_scale_recording_confirmation_20261121"
POSE = STUDIES/"3eed_pose_consistent_visible_surface_grounding_20261121"
VAL = STUDIES/"3eed_v2_ordinary_val_subject_ranking_20261119"
LOCK = STUDIES/"3eed_pose_recording_exposure_audit_20261121"/"CANDIDATE_LOCK.json"
ARMS = ("A", "H", "L7", "C_pca", "C_ray")
TASKS = ("3d", "2d")


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+"\n")
    temporary.replace(path)


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def key(*values):
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()[:24]


def assert_geometry_lock():
    lock = read(LOCK)
    for rel, checksum in lock["local_dependency_sha256"].items():
        assert sha(ROOT/rel) == checksum, rel
    for rel, checksum in lock["published_study_sha256"].items():
        assert sha(POSE/rel) == checksum, rel
    return sha(LOCK)


@lru_cache(maxsize=None)
def file_sha(path):
    return sha(path)


@lru_cache(maxsize=None)
def captions(path):
    import ijson

    with open(path, "rb") as stream:
        return list(ijson.items(stream, "ground_info.item.caption"))


def panel():
    lock = read(OUT/"input_seal.json")
    assert sha(SPEC)==lock["protocol"] and sha(PRIVATE/"panel.json")==lock["panel"]
    assert lock["inference_code"] == sha(OLD/"infer.py") and not lock["target_labels_read"]
    assert lock["code"] == sha(__file__) and lock["geometry_lock"] == sha(LOCK)
    return read(PRIVATE/"panel.json"), lock


def old_replies():
    matches = {}
    sources = {}
    for folder in (OLD, CONFIRM):
        seal_path = folder/"results/model_seal.json"
        seal = read(seal_path)
        pins = seal["pins"]
        protocol = read(folder/"PROTOCOL.json")
        compare = read(OLD/"PROTOCOL.json")
        assert protocol["inference"] == compare["inference"]
        assert not seal["target_labels_read"] and pins["model"] == compare["revision"]
        assert pins["assets"] == sha(OLD/"MODEL.json") and pins["code"] == sha(OLD/"infer.py")
        assert pins["protocol"] == sha(folder/"PROTOCOL.json")
        p = folder/"private/panel.json"
        assert sha(p) == pins["panel"]
        rows = read(p)
        replies = read(folder/"private/reply_hashes.json")
        assert hashlib.sha256(json.dumps(replies, sort_keys=True).encode()).hexdigest() == seal["ordered_reply_hash"]
        assert len(replies)==2*len(rows)
        for r in rows:
            identifier = r["rgb_sha"], r["text"]
            assert identifier not in matches
            matches[identifier] = (folder, r["uid"], replies, pins)
        sources[folder.name] = sha(seal_path)
    return matches, sources


def run():
    t0 = perf_counter()
    assert_geometry_lock()
    sys.path.insert(0, str(VAL))
    from scope183 import VIEWS, canonical, check_input, ordinary

    allowed = ordinary()
    previous, sources = old_replies()
    observed = []
    reused = Counter()
    physical = Counter()
    for target, rows in allowed.items():
        for ordinal, r in enumerate(rows):
            check_input(r)
            meta = Path(r["metapath"])
            slot = r["object_index"]
            text = captions(r["metapath"])[slot]
            assert canonical(text) == r["caption"]
            recording = key("ordinary-val-recording", target, r["recording"])
            timestamp = meta.parent.name.rsplit("_", 1)[0] if target=="waymo" else meta.parent.name
            frame = key("ordinary-val-frame", target, r["recording"], timestamp)
            uid = key("ordinary-val-description", target, r["recording"], meta.parent.name, slot)
            physical[frame] += 1
            r_new = dict(uid=uid, target=target, panel="validation", recording=recording, frame=frame,
                         ordinal=ordinal, slot=slot, text=text, rgb=r["imagepath"], points=r["pointpath"],
                         calibration=r["metapath"], suffix=r["suffix"], waymo_view=VIEWS[r["suffix"]],
                         seed=r["seed"], point_sha=file_sha(r["pointpath"]), rgb_sha=file_sha(r["imagepath"]))
            observed.append(r_new)
            match = previous.get((r_new["rgb_sha"], text))
            if match is None:
                continue
            folder, old_uid, checks, pins = match
            for task in TASKS:
                name = old_uid+"_"+task
                path = folder/"private/replies"/(name+".json")
                assert sha(path) == checks[name]
                saved = read(path)
                assert (saved["uid"], saved["task"], saved["pins"]) == (old_uid, task, pins)
                payload = {k:saved[k] for k in ("raw","value","status","visual_grid","generated_tokens")}
                r_new.setdefault("archived", {})[task] = dict(**payload, source=folder.name, source_sha256=checks[name])
            reused[folder.name] += 1
    assert {k:len(v) for k,v in allowed.items()} == read(SPEC)["roster"]["descriptions_by_platform"]
    assert len({r["uid"] for r in observed}) == len(observed) == 9310
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    OUT.mkdir(exist_ok=True)
    write(PRIVATE/"panel.json", observed)
    write(OUT/"input_seal.json", dict(
        protocol=sha(SPEC), panel=sha(PRIVATE/"panel.json"), geometry_lock=sha(LOCK), inference_code=sha(OLD/"infer.py"),
        code=sha(__file__), ordinary_roster_seal=sha(VAL/"ordinary_PRIVATE/SEALED.json"), frames=len(physical),
        descriptions=len(observed), recordings=len({r["recording"] for r in observed}),
        descriptions_by_platform=dict(Counter(r["target"] for r in observed)), reuse_descriptions_by_source=dict(reused),
        reuse_queries=2*sum(reused.values()), old_model_seals=sources,
        target_labels_read=False, FINAL_access=False, seconds=perf_counter()-t0
    ))
    print("Locked",len(observed),"descriptions on",len(physical),"physical timestamps, reused",2*sum(reused.values()),"queries.",flush=True)


if __name__ == "__main__":
    run()

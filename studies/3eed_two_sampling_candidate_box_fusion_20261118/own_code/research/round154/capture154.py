"""Capture fresh source proposals and complete Frozen-A scores on a second A sampling."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "research/round154"
OUTPUT = ROOT / "crossplatform/round154/captures"
REGISTRY = ROOT / "research/round145/INPUT_REGISTRY_PRIVATE.json"
PARENT = ROOT / "crossplatform/round145/method"
REGISTRY_SHA = "6df8cbc22aff3b8c945c6a50c6a23ec5c3b559d46f200618cfb91e83a220ad85"
PARENT_SEAL_SHA = "ac4a9cf70845431f1509c195afb2664a9593c75388fd48b6880483ff62a6ca31"
ROW_FIELDS = ("id", "platform", "seed", "suffix", "caption", "pointpath", "imagepath", "metapath")
BRIDGE_DIRECTIONS = {"waymo": "drone", "drone": "quad", "quad": "waymo"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def second_seed(seed):
    key = f"R154|{seed}".encode()
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def atomic(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("xb") as out:
            writer(out)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path, value):
    atomic(path, lambda out: out.write((json.dumps(value, indent=2, allow_nan=False)+"\n").encode()))


def registry():
    assert digest(REGISTRY) == REGISTRY_SHA
    records = json.loads(REGISTRY.read_text())
    assert len(records) == 1798
    out = []
    for e in records:
        a = e["A_sensor"]
        assert e["cohort"] == "EXT" and not e["protected_alias_fallback"]
        assert a["eligible_for_extra_media"] and not a["protected_image_content_alias"]
        r = {k: e["row"][k] for k in ROW_FIELDS}
        assert r["platform"] == a["platform"] == e["target"] != e["source"]
        assert e["source"] in BRIDGE_DIRECTIONS and e["target"] in BRIDGE_DIRECTIONS
        for k in ("pointpath", "imagepath", "metapath"):
            assert Path(r[k]).samefile(a[k])
        out.append((e["source"], r, a["point_sha256"], a["image_sha256"]))
    assert len({(s, r["platform"], r["id"]) for s, r, _, _ in out}) == len(out)
    return out


def run(check_only):
    sys.path.insert(0, str(ROOT / "research/round97"))
    from core97 import (base97, fingerprint, freeze91, model97, restore, role97,
                        sample97, setsource97, source97, state, tensor_data,
                        tensor_prefix, tensor_route89)
    sys.path.insert(0, str(ROOT / "research/round135"))
    from core135 import live135
    import torch

    rows = registry()
    assert digest(PARENT / "SEALED.json") == PARENT_SEAL_SHA
    pinned_files = json.loads((PARENT / "SEALED.json").read_text())["files"]
    pin = {"capture_sha256": digest(Path(__file__)), "registry_sha256": REGISTRY_SHA,
           "r145_seal_sha256": PARENT_SEAL_SHA}
    bridge_file = WORK / "CAPTURE_BRIDGE_PRIVATE.json"
    seal_file = OUTPUT / "SEALED_PRIVATE.json"
    if check_only:
        assert not bridge_file.exists(), "existing checked bridge"
    else:
        bridge = json.loads(bridge_file.read_text())
        assert bridge["status"] == "PASS" and bridge["pins"] == pin
        assert {x["source"] for x in bridge["checks"]} == set(BRIDGE_DIRECTIONS)
        assert not seal_file.exists(), "capture already sealed"
    checks, files, source_hashes = [], {}, {}
    started = time.monotonic()
    for source, bridge_target in BRIDGE_DIRECTIONS.items():
        selected = [x for x in rows if x[0] == source]
        if check_only:
            selected = [next(x for x in selected if x[1]["platform"] == bridge_target)]
        setsource97(source)
        base = base97(source)
        model, weights = model97(base, source)
        original_state = fingerprint(base)
        prepared_hash = hashlib.sha256(b"".join(n.encode()+v.tobytes() for n,v in weights.items())).hexdigest()
        source_hashes[source] = {"epoch100_checkpoint_sha256": source97(source)["checkpoint_sha256"],
                                 "prepared_state_sha256": prepared_hash}
        if not check_only:
            assert source_hashes[source] == bridge["sources"][source]
        for _, row, point_sha, image_sha in selected:
            direction = source+"_to_"+row["platform"]
            item = row["id"]+".npz"
            output = OUTPUT / direction / item
            s = row["seed"] if check_only else second_seed(row["seed"])
            if not check_only and output.exists():
                with np.load(output, allow_pickle=False) as existing:
                    assert existing["capture_sha256"].item() == pin["capture_sha256"]
                    assert existing["prepared_state_sha256"].item() == prepared_hash
                    assert existing["row_id"].item() == row["id"] and existing["direction"].item() == direction
                    assert int(existing["seed"]) == s and existing["raw_point_sha256"].item() == point_sha
                    assert existing["raw_image_sha256"].item() == image_sha
                    assert existing["new_boxes"].shape == (256,6) and existing["new_p"].shape == (256,)
                    assert np.isfinite(existing["new_boxes"]).all() and np.isfinite(existing["new_p"]).all()
                    assert np.isclose(existing["new_p"].sum(), 1., atol=1e-9, rtol=0)
                files[direction+"/"+item] = digest(output)
                continue
            r = dict(row, seed=s)
            restore(model, weights)
            freeze91(model)
            with torch.no_grad():
                z0, geometry, receipt, proof = sample97(base, r)
                assert proof["camera_fields_only"] and proof["point_sha256"] == point_sha
                assert proof["image_sha256"] == image_sha
                z, d = tensor_prefix(z0), tensor_data(geometry, r)
                rt, _ = role97(base, model, r, z0, geometry, z, d)
                lp, _, _ = live135(model, base, z, d, z["vis0"][0], tensor_route89(rt))
            prob = lp.exp().cpu().numpy()
            boxes = z0["boxes"]
            assert boxes.shape == (256,6) and prob.shape == (256,)
            assert np.isfinite(boxes).all() and np.isfinite(prob).all() and np.all(prob >= 0)
            assert np.isclose(prob.sum(), 1., atol=1e-9, rtol=0)
            ids = z0["point_indices"]
            assert ids.shape == (16384,) and ids.dtype == np.int64
            with np.load(ROOT / "crossplatform/round131/formal" / direction / "prefix_PRIVATE" / item,
                         allow_pickle=False) as original:
                same_ids = np.array_equal(original["point_indices"], ids)
                original_ids_digest = array_digest(original["point_indices"])
            assert same_ids == check_only
            if check_only:
                rel = direction+"/samples_PRIVATE/"+item
                assert pinned_files[rel] == digest(PARENT / rel)
                with np.load(PARENT / rel, allow_pickle=False) as old:
                    p0 = old["scores"][old["arm_names"].tolist().index("Frozen_A")]
                    assert np.array_equal(boxes, old["source_boxes"])
                    err = float(np.abs(p0-prob).max())
                    assert err < 1e-7 and prob.argmax() == p0.argmax(), (source, err)
                checks.append({"source": source, "target": row["platform"], "id_PRIVATE": row["id"],
                               "source_boxes_bit_exact": True, "original_indices_bit_exact": True,
                               "max_probability_error": err, "same_top1": True})
            else:
                fields = dict(new_boxes=boxes, new_p=prob, seed=np.uint64(s),
                              original_seed=np.uint64(row["seed"]), direction=direction, row_id=row["id"],
                              original_indices_sha256=original_ids_digest, new_indices_sha256=array_digest(ids),
                              new_sample_xyz_sha256=receipt["point_sample_sha256"],
                              new_query_anchor_sha256=array_digest(z0["query_anchor_xyz"]),
                              prepared_state_sha256=prepared_hash, capture_sha256=pin["capture_sha256"],
                              raw_point_sha256=point_sha, raw_image_sha256=image_sha)
                atomic(output, lambda out: np.savez_compressed(out, **fields))
                files[direction+"/"+item] = digest(output)
                if len(files) % 32 == 0:
                    print("CAPTURE154", len(files), round(time.monotonic()-started,1), flush=True)
        assert fingerprint(base) == original_state
        final = state(model)
        assert weights.keys() == final.keys() and all(np.array_equal(v,final[k]) for k,v in weights.items())
        del model, base, weights
        torch.cuda.empty_cache()
    if check_only:
        write_json(bridge_file, dict(status="PASS", pins=pin, sources=source_hashes, checks=checks,
                                    task_GT_read=False, FINAL_access=False))
    else:
        assert len(files) == len(rows)
        write_json(seal_file, dict(n=len(files), files=files, pins=pin, sources=source_hashes,
                                  seconds=time.monotonic()-started, original_A_replaced=False,
                                  target_GT_read=False, FINAL_access=False))
    print("CAPTURE154_COMPLETE", "bridge" if check_only else "second_sampling",
          len(checks) if check_only else len(files), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    run(parser.parse_args().check_only)

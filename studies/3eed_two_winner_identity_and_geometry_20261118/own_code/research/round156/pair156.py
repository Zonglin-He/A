"""Seal both original top-one predictions under shared point support, without target labels."""
import json
import os
import sys
from pathlib import Path
from time import monotonic

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research/round155"))
from predict155 import PCA_FILE, REG, R131, R145, R154, array_hash, digest, load_module, load_raw, read, require

CAP = ROOT / "crossplatform/round154/captures"
LAST = ROOT / "crossplatform/round155"
OUT = ROOT / "crossplatform/round156"
DESIGN = HERE / "experiment/DESIGN_LOCK.json"
PINS = {REG: "6df8cbc22aff3b8c945c6a50c6a23ec5c3b559d46f200618cfb91e83a220ad85",
        CAP / "SEALED_PRIVATE.json": "9f9d6ab42bc870f75d7f0a4bc43c121ac232af1b79c81509c43a4117092615b3",
        R154 / "SEALED.json": "4dcb67195aee3af8cab43571cd43a82d38591a3107a4b64ffa24631d6c44c9da",
        LAST / "SEAL.json": "7e9971a7116e2a6d78e79790ceb3a804d7920814f8db75568d02a31a414ebf15",
        PCA_FILE: "c779a23d4949bd7e06a912c8d8144132ca94bc98fd965a45dccb03bd5aebdcd2"}


def shared_points(entry, receipt, transform):
    row, sensor = entry["row"], entry["A_sensor"]
    require(entry["cohort"] == "EXT" and not entry["protected_alias_fallback"] and
            sensor["eligible_for_extra_media"] and not sensor["protected_image_content_alias"] and
            Path(row["pointpath"]).samefile(sensor["pointpath"]) and row["platform"] == entry["target"] == sensor["platform"],
            "unexpected frame")
    raw = load_raw(sensor["pointpath"], sensor["point_sha256"])
    xyz = np.ascontiguousarray(raw[:, :3], dtype=np.float32).copy()
    if entry["target"] == "waymo":
        view = ("F", "FL", "FR", "SL", "SR")[row["suffix"]]
        xyz, _ = transform(xyz, np.zeros((1, 7)), view)
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    elif entry["target"] == "drone":
        xyz[:, 2] += np.float32(1.8)
    else:
        require(entry["target"] == "quad", "unknown coordinates")
    idx = np.random.default_rng(row["seed"]).choice(len(xyz), 16384, replace=len(xyz) < 16384)
    sampled = xyz[idx]
    require(array_hash(idx) == receipt["point_indices_sha256"] and
            array_hash(sampled) == receipt["point_sample_sha256"], "full-cloud-to-sampled geometry bridge failed")
    return sampled, xyz


def fit(points, box, pca):
    # Intersect before deduplication: filtering and row deduplication commute.
    kept = points[np.all(np.abs(points - box[:3]) <= box[3:] / 2 + 1e-7, axis=1)]
    if (box[3:] <= 0).any():
        return np.r_[box, np.float32(0)], len(np.unique(kept,axis=0)), False
    _, oriented, rec = pca(kept, box)
    return oriented, rec["predicted_crop_points"], rec["reason"] == "updated"


def main():
    start = monotonic()
    lock = digest(DESIGN)
    for path, expected in PINS.items():
        require(digest(path) == expected, "upstream seal changed")
    entries = read(REG)
    original = read(LAST / "SEAL.json")
    require(len(entries) == original["n"] == 1798 and
            digest(LAST / "predictions_PRIVATE.npz") == original["sha256"], "R155 seal")
    with np.load(LAST / "predictions_PRIVATE.npz",allow_pickle=False) as previous:
        base = previous["boxes"][:, 0, :2].copy()
        prior_query = previous["query"][:, 0]
        require(list(previous["states"]) == ["joint_zero","R145"], "historic states")
    cap_files, pred_files = read(CAP / "SEALED_PRIVATE.json")["files"], read(R154 / "SEALED.json")["files"]
    record_files = {d.name: read(d / "SEALED.json")["files"] for d in R131.iterdir() if d.is_dir()}
    boxes = np.empty((len(entries),2,3,7),np.float32)
    sym = np.empty((len(entries),7),np.float32)
    query = np.empty((len(entries),2),np.int16)
    counts = np.empty((len(entries),2,2),np.int32)
    allows = np.empty((len(entries),2,2),bool)
    pca = load_module("r156_historical_pca", PCA_FILE).refine
    from utils.transform_waymo import transform_to_front_view
    output = OUT / "PAIR_PRIVATE.npz"
    seal = OUT / "PAIR_SEAL.json"
    require(not output.exists() and not seal.exists(),"output exists")
    for i,e in enumerate(entries):
        direction = e["source"] + "_to_" + e["target"]
        sid = e["row"]["id"]
        key = direction + "/" + sid + ".npz"
        rkey = "records_PRIVATE/" + sid + ".json"
        rpath = R131 / direction / rkey
        require(digest(rpath) == record_files[direction][rkey] and
                digest(CAP/key) == cap_files[key] and digest(R154/key) == pred_files[key], "parent file hash")
        sampled, full = shared_points(e,read(rpath)["source_prefix"],transform_to_front_view)
        with (np.load(CAP/key,allow_pickle=False) as alternate, np.load(R154/key,allow_pickle=False) as last):
            q = int(alternate["new_p"].argmax())
            alt = alternate["new_boxes"][q].astype(np.float32)
            arm_names = last["arm_names"].tolist()
            prior = last["chosen_boxes"]
            require(np.array_equal(prior[arm_names.index("alternate_single")],alt) and
                    np.array_equal(prior[arm_names.index("joint_zero")],base[i,0,:6]), "historic winners")
            frozen = prior[arm_names.index("frozen_A")].astype(np.float32)
        query[i] = [prior_query[i],q]
        for j, box in enumerate((base[i,0,:6],alt)):
            boxes[i,j,0] = np.r_[box,np.float32(0)]
            for k,cloud in enumerate((sampled, full)):
                boxes[i,j,k+1], counts[i,j,k], allows[i,j,k] = fit(cloud,box,pca)
        sym[i] = fit(sampled,frozen,pca)[0]
        require(np.array_equal(boxes[i,0,:2],base[i]), "original PCA baseline failed exact replay")
        if (i+1)%300 == 0:
            print("Sealed input geometry in memory",i+1,flush=True)
    require(np.isfinite(boxes).all() and np.isfinite(sym).all(),"nonfinite endpoint")
    OUT.mkdir(exist_ok=True,parents=True)
    temp = output.with_suffix(".tmp")
    with temp.open("xb") as stream:
        np.savez_compressed(stream,boxes=boxes,query=query,crop_counts=counts,eligible=allows,
                            symmetric_A_pca=sym,ordinal=np.arange(1,len(entries)+1),
                            winners=["A","D"],geometry=["axis","pca_shared_A","pca_full_raw"])
    os.replace(temp,output)
    info = dict(status="PASS",n=len(entries),prediction_sha256=digest(output),design_sha256=lock,
                code_sha256=digest(Path(__file__)),shared_sample_hash_replays=len(entries),
                baseline_PCA_bit_exact=len(entries),second_winner_exact=len(entries),
                protected_frames_opened=0,new_network_forwards=0,source_training=False,target_GT_read=False,
                seconds=monotonic()-start)
    tmp = seal.with_suffix(".tmp"); tmp.write_text(json.dumps(info,indent=2)+"\n");os.replace(tmp,seal)
    print("R156 pair endpoints sealed",len(entries),flush=True)


if __name__ == "__main__":
    main()

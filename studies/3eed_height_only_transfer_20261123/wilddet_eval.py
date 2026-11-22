"""Test the locked geometric rule with a different frozen 3D box provider."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from common import HERE, OUT, PRIVATE, PROTOCOL, group_summary, independent_tally, lock, prior, read_payload, roster, save_payload, scorer
from live import first_depth, geometry, incumbent, predict
from geometry import cook_rotation, extent_corners


ARMS = ("qwen_frozen", "qwen_roof", "wild_native_rgb", "wild_native_rgb_lidar", "wild_frozen", "wild_roof")
PAIRS = (("qwen_roof", "qwen_frozen"), ("wild_native_rgb_lidar", "wild_native_rgb"), ("wild_frozen", "wild_native_rgb"),
         ("wild_roof", "wild_frozen"), ("wild_roof", "wild_native_rgb_lidar"), ("qwen_roof", "wild_native_rgb_lidar"))
RAW_FILE = PRIVATE / "wilddet3d_boxes.jsonl.gz"
RAW_SEAL = OUT / "wilddet3d_box_provider_seal.json"
PRED = PRIVATE / "wilddet3d_readouts.jsonl.gz"
PRED_SEAL = OUT / "wilddet3d_prediction_seal.json"
SPLIT = HERE / "WILDDET_SPLIT_ADDENDUM.json"


def provider_pins():
    return dict(lock(Path(__file__)), validation=roster.sha(roster.OUT / "input_seal.json"),
                model=roster.sha(roster.OUT / "qwen_seal.json"), split=roster.sha(SPLIT))


def bridge(native):
    # Match the published evaluator: stored W, L, H, with OpenCV local X, Y, Z = L, H, W.
    source = np.asarray(native, dtype=float)
    if source.shape != (10,) or not np.isfinite(source).all() or not (source[3:6] > 0).all() or source[2] <= 0:
        return None, 0.
    if np.linalg.norm(source[6:10]) <= 1.e-8:
        return None, 0.
    q = Rotation.from_quat(source[[7, 8, 9, 6]])
    angles = q.as_euler("xyz") / np.pi
    raw = np.r_[source[:3], source[[4, 5, 3]], angles]
    err = evaluator_corner_error(source, raw)
    assert err < 1.e-8
    return raw.tolist(), err


def evaluator_corner_error(native, box):
    import torch
    from vis4d.data.const import AxisMode
    from vis4d.op.box.box3d import boxes3d_to_corners

    official = boxes3d_to_corners(torch.as_tensor(native[None], dtype=torch.float64), AxisMode.OPENCV)[0].numpy()
    found = extent_corners(box[:3], box[3:6], cook_rotation(box))
    return float(np.linalg.norm(found[:, None] - official[None, :], axis=2).min(axis=1).max())


def verify_checkout(repo, cfg):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    assert git("rev-parse", "HEAD") == cfg["code_revision"]
    pins = {}
    for path in ("third_party/lingbot_depth", "third_party/sam3"):
        mode, kind, sha, _ = git("ls-tree", "HEAD", path).split()
        assert (mode, kind) == ("160000", "commit")
        found = subprocess.check_output(["git", "-C", str(repo/path), "rev-parse", "HEAD"], text=True).strip()
        assert found == sha
        pins[path] = sha
    return pins


def provider(repo, weights, cfg):
    import huggingface_hub
    import torch

    submodules = verify_checkout(repo, cfg)
    assert weights.name == cfg["checkpoint"] and weights.stat().st_size == cfg["checkpoint_bytes"]
    assert roster.sha(weights) == cfg["checkpoint_sha256"]
    geometry_config = OUT / "lingbot_config.json"
    layout = roster.read(geometry_config)
    stub = PRIVATE / "lingbot_config_only.pt"
    torch.save({"model_config": layout["model_config"]}, stub)
    sys.path.insert(0, str(repo))
    from wilddet3d import build_model

    official_download = huggingface_hub.hf_hub_download
    loader = torch.load
    load_weights = torch.nn.Module.load_state_dict
    restore = {}

    def config_hook(repo_id, *args, **kwargs):
        if repo_id == "robbyant/lingbot-depth-postrain-dc-vitl14" and kwargs.get("filename") == "model.pt":
            return str(stub)
        return official_download(repo_id, *args, **kwargs)

    def safe_load(name, *args, **kwargs):
        kwargs["weights_only"] = True
        if name == str(weights):
            kwargs["mmap"] = True
        return loader(name, *args, **kwargs)

    def inspect_state(model, weights, *args, **kwargs):
        info = load_weights(model, weights, *args, **kwargs)
        if model.__class__.__name__ == "WildDet3D":
            expected = set(dict(model.named_parameters()))
            assert not set(info.missing_keys) & expected, sorted(set(info.missing_keys) & expected)[:20]
            restore.update(missing_buffers=info.missing_keys, unused_tensors=len(info.unexpected_keys),
                           loaded_parameter_tensors=len(expected))
        return info

    torch.manual_seed(20261123)
    with patch("huggingface_hub.hf_hub_download", config_hook), patch("torch.load", safe_load), \
            patch.object(torch.nn.Module, "load_state_dict", inspect_state):
        model = build_model(checkpoint=str(weights), skip_pretrained=True, use_predicted_intrinsics=False, canonical_rotation=True,
                            use_depth_input_test=False).eval().requires_grad_(False)
    assert restore and not model.training and not any(p.requires_grad for p in model.parameters())
    return model, dict(submodules=submodules, restored=restore, canonical_rotation=True, lingbot_config=roster.sha(geometry_config),
                       geometry_backbone_revision=layout["revision"], torch=torch.__version__)


def rectified(frame, window):
    import cv2

    spec, size, _ = geometry.camera(frame)
    K, _, distortion, _ = spec.validated()
    with Image.open(frame["rgb"]) as image:
        pixels = np.array(image.convert("RGB"), dtype=np.uint8)
    w, h = map(int, size)
    box = np.asarray(window, dtype=float)*np.tile(size, 2)/1000.
    if np.any(distortion):
        # Use the same K for the undistorted provider input and reference rays.
        uv = np.array([[u, v] for u in np.linspace(box[0], box[2], 9) for v in np.linspace(box[1], box[3], 9)])
        undistorted = cv2.undistortPoints(uv[:, None], K, distortion, P=K)[:, 0]
        box = np.r_[undistorted.min(0), undistorted.max(0)]
        pixels = cv2.undistort(pixels, K, distortion, None, K)
    box = np.clip(box, np.zeros(4), np.array([w-1, h-1, w-1, h-1]))
    _, xyz = geometry.raw_returns(frame, spec, size)
    homogeneous = xyz @ K.T
    points = np.floor(homogeneous[:, :2] / homogeneous[:, 2:3]).astype(int)
    good = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    seen = np.full(h*w, np.inf, dtype=np.float32)
    ij = points[good]
    np.minimum.at(seen, ij[:, 1]*w+ij[:, 0], xyz[good, 2].astype(np.float32))
    seen[~np.isfinite(seen)] = 0.
    return pixels, K, box.tolist(), seen.reshape(h, w), bool(np.any(distortion))


def predict_provider(model, data, window, use_depth):
    model.wilddet3d.use_depth_input_test = use_depth
    predictions = model(images=data["images"], intrinsics=data["intrinsics"][None],
        input_hw=[data["input_hw"]], original_hw=[data["original_hw"]], padding=[data["padding"]],
        input_boxes=[window], prompt_text="geometric", depth_gt=data["depth_gt"] if use_depth else None)
    shape = predictions[1][0].shape
    if shape == (0, 10):
        return None, 0.
    assert shape == (1, 10)
    return bridge(predictions[1][0][0].float().detach().cpu().numpy())


def infer(repo, weights):
    import torch

    start = perf_counter()
    cfg = roster.read(PROTOCOL)["wilddet3d"]
    counts_expected = roster.read(SPLIT)["counts"]
    population = sum(counts_expected.values())
    pins = provider_pins()
    pins.update(wilddet_code=cfg["code_revision"], wilddet_weights=cfg["checkpoint_sha256"],
                lingbot=roster.sha(OUT / "lingbot_config.json"))
    cache = PRIVATE / "wilddet_provider_rows_wlh"
    cache.mkdir(exist_ok=True, mode=0o700)
    model, model_info = provider(repo, weights, cfg)
    from wilddet3d import preprocess

    counts, platform_counts = Counter(), Counter()
    rows = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for frame, qwen, *_ in prior.fixed_validation_jobs():
            assert frame["panel"] == "validation"
            platform_counts[frame["target"]] += 1
            uid = frame["uid"]
            path = cache / (uid + ".json")
            if path.exists():
                row = roster.read(path)
                assert row["uid"] == uid and row["pins"] == pins
            else:
                assert roster.sha(frame["rgb"]) == frame["rgb_sha"] and roster.sha(frame["points"]) == frame["point_sha"]
                window = qwen["2d"]["value"]
                row = dict(uid=uid, pins=pins, mono=None, depth=None, outcome="missing_2d", distortion_rectified=False,
                           depth_pixels=0, worst_corner_error_m=0.)
                if window is not None:
                    rgb, K, rect, dep, distorted = rectified(frame, window)
                    row.update(distortion_rectified=distorted, depth_pixels=int((dep > 0).sum()))
                    if rect[2] > rect[0] and rect[3] > rect[1]:
                        data = preprocess(rgb, K, depth=dep)
                        for key in ("images", "intrinsics", "depth_gt"):
                            data[key] = data[key].cuda()
                        a, err_a = predict_provider(model, data, rect, False)
                        b, err_b = predict_provider(model, data, rect, True)
                        row.update(mono=a, depth=b, outcome="inferred", worst_corner_error_m=max(err_a, err_b))
                    else:
                        row["outcome"] = "degenerate_2d"
                roster.write(path, row)
            rows.append(row)
            counts[row["outcome"]] += 1
            if len(rows) % 64 == 0:
                print("WildDet3D", len(rows), "/", population, dict(counts), "elapsed", round(perf_counter()-start), flush=True)
    assert len(rows) == len({r["uid"] for r in rows}) == population and dict(platform_counts) == counts_expected
    content = save_payload(RAW_FILE, rows)
    roster.write(RAW_SEAL, dict(payload=content, pins=pins, model=model_info, cases=len(rows), platforms=dict(platform_counts), statuses=dict(counts),
        filled_rgb=sum(r["mono"] is not None for r in rows), filled_rgb_lidar=sum(r["depth"] is not None for r in rows),
        max_camera_corner_difference_m=max(r["worst_corner_error_m"] for r in rows), precision="autocast bfloat16",
        gpu=torch.cuda.get_device_name(), peak_bytes=torch.cuda.max_memory_allocated(), seconds=perf_counter()-start,
        target_gt_opened=False, FINAL_access=False))


def construct(job):
    frame, qwen, wild, previous = job
    assert frame["uid"] == wild["uid"] == previous["uid"]
    old, main = [previous["boxes"][a] for a in ("baseline", "height_only")]
    reply = dict(qwen, **{"3d": {"value": wild["mono"], "status": wild["outcome"]}})
    base, context = incumbent(frame, reply, collect_audit=True)
    final = predict(frame, reply)
    assert base["applicable"] == final["applicable"]
    stats = dict(active=base["applicable"], changed=int(final["extent_corrected"]), selection=0,
                 audit=0, selection_changed=0, audit_changed=0)
    if stats["changed"]:
        box, _dimensions, _rotation, _fit, selection, audit, transform = context
        other = np.asarray(final["box"], dtype=float)
        sel, err = prior.reference.invariance(first_depth(box, selection, transform), first_depth(other, selection, transform))
        aud, _ = prior.reference.invariance(first_depth(box, audit, transform), first_depth(other, audit, transform))
        assert sel == 0 and err <= 1.e-8
        stats.update(selection=len(selection), audit=len(audit), selection_changed=sel, audit_changed=aud)
    T = geometry.camera(frame)[2]
    boxes = dict(qwen_frozen=old, qwen_roof=main, wild_frozen=base["box"], wild_roof=final["box"])
    for src, dst in (("mono", "wild_native_rgb"), ("depth", "wild_native_rgb_lidar")):
        raw = wild[src]
        boxes[dst] = geometry.to_sensor(np.asarray(raw), T)[0].tolist() if raw is not None else None
    return dict(uid=frame["uid"], boxes=boxes, readout=base["readout"], depth_pixels=wild["depth_pixels"], **stats)


def generate(workers):
    model = roster.read(RAW_SEAL)
    assert model["payload"] == roster.sha(RAW_FILE) and not model["target_gt_opened"]
    pins = provider_pins()
    assert all(model["pins"][k] == v for k, v in pins.items())
    prev_path, prev_lock = prior.files("validation")
    previous = roster.read(prev_lock)
    assert previous["payload"] == roster.sha(prev_path) and previous["files"] == prior.fingerprints()
    assert not previous["target_gt_opened"] and not previous["FINAL_access"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = ((v[0], v[1], a, b) for v, a, b in zip(prior.fixed_validation_jobs(), read_payload(RAW_FILE), read_payload(prev_path), strict=True))
        payload = save_payload(PRED, pool.map(construct, jobs, chunksize=16))
    roster.write(PRED_SEAL, dict(model_source=roster.sha(RAW_SEAL), qwen_baseline=roster.sha(prev_lock),
        pins=pins, payload=payload, descriptions=model["cases"], target_gt_opened=False, FINAL_access=False))


def score(job):
    frame, guess = job
    assert frame["uid"] == guess["uid"] and frame["panel"] == "validation"
    path = Path(frame["calibration"])
    assert path.name == "meta_info.json" and path.parents[2] == prior.reference.DATA / frame["target"]
    annotations = roster.read(path)["ground_info"]
    truth = scorer.canonical_box(scorer.truth_for_slot(frame, annotations))
    values, err = prior.safe_iou([guess["boxes"][a] for a in ARMS], truth)
    row = {k: frame[k] for k in ("uid", "recording", "target")}
    row.update({k: guess[k] for k in ("readout", "depth_pixels", "active", "changed", "selection", "audit", "selection_changed", "audit_changed")})
    row["category"] = annotations[frame["slot"]]["class"].lower().strip()
    row.update({f"iou_{a}": float(x) for a, x in zip(ARMS, values, strict=True)})
    return row, err


def evaluate(workers):
    lock_ = roster.read(PRED_SEAL)
    assert lock_["pins"] == provider_pins() and lock_["payload"] == roster.sha(PRED) and not lock_["target_gt_opened"]
    assert lock_["model_source"] == roster.sha(RAW_SEAL)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        frames, _ = roster.panel()
        found = list(pool.map(score, zip(frames, read_payload(PRED), strict=True), chunksize=32))
    rows, errs = zip(*found, strict=True)
    assert len(rows) == lock_["descriptions"] and max(errs) < 1.e-7
    from audit_paper import ledger
    expected = {r["uid"]: r for r in ledger("validation")}
    for row in rows:
        for a, b in (("qwen_frozen", "baseline"), ("qwen_roof", "height_only")):
            assert abs(row[f"iou_{a}"] - float(expected[row["uid"]][f"iou_{b}"])) < 1.e-10
    files, interval = group_summary("wilddet3d", rows, ARMS, PAIRS)
    audit = independent_tally("wilddet3d", files)
    roster.write(OUT / "wilddet3d_evaluation.json", dict(files=files, prediction=roster.sha(PRED_SEAL), independent=audit,
        max_two_iou_diff=max(errs), counts={k: sum(r[k] for r in rows) for k in ("active", "changed", "selection",
           "audit", "selection_changed", "audit_changed")}, recording_bootstrap_95pp=interval, FINAL_access=False))
    print(len(rows), "WildDet3D records compared and audited", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("infer", "generate", "evaluate"))
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "infer":
        infer(args.repository, args.weights)
    else:
        (generate if args.phase == "generate" else evaluate)(args.workers)

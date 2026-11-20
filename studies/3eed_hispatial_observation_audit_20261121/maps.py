"""Create frozen model inputs with physically disjoint fit and evaluation returns."""
import argparse
from importlib.metadata import version
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch

from common import ASSETS, PRIVATE, OUT, PROTOCOL, SIDE, digest, read, require_repo, require_weights, write
from geometry import calibrated_grid, calibration, make_xyz, nearest_points, project_camera, wrong_locations


def atomic_npz(path, **values):
    temp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temp, **values)
    temp.replace(path)


def load_frames():
    seal = read(OUT / "panel_seal.json")
    assert seal["frames"] == digest(PRIVATE / "frames.json") and seal["protocol"] == digest(PROTOCOL)
    return read(PRIVATE / "frames.json")


def open_rgb(frame):
    assert frame["rgb_sha"] == digest(frame["rgb"])
    return cv2.cvtColor(cv2.imread(frame["rgb"]), cv2.COLOR_BGR2RGB)


def requirements(sources, name, subdir):
    path = sources / subdir
    require_repo(path, name)
    sys.path.insert(0, str(path))


def cuda_summary(start):
    return dict(runtime_seconds=time.perf_counter()-start, gpu=torch.cuda.get_device_name(),
                allocated_bytes=torch.cuda.max_memory_allocated(), reserved_bytes=torch.cuda.max_memory_reserved(),
                torch=torch.__version__)


@torch.inference_mode()
def run_moge(sources, weights_root):
    requirements(sources, "moge", "MoGe")
    requirements(sources, "hispatial", "HiSpatial")
    from hispatial.inference import MoGeProcessor
    from moge.model.v2 import MoGeModel

    assets = require_weights(weights_root, "moge")
    # The released constructor downloads an unpinned snapshot. Bind the same
    # released class to the already verified local checkpoint.
    processor = MoGeProcessor.__new__(MoGeProcessor)
    processor.device = torch.device("cuda")
    processor.img_size = SIDE
    processor.model = MoGeModel.from_pretrained(assets / "model.pt").eval().requires_grad_(False).to("cuda")
    torch.manual_seed(20261121)
    folder = PRIVATE / "moge"
    folder.mkdir(mode=0o700, exist_ok=True)
    manifest = {}
    stamp = dict(protocol=digest(PROTOCOL), frame=digest(PRIVATE / "frames.json"),
                 model=digest(ASSETS), code=digest(__file__))
    started = time.perf_counter()
    for i, frame in enumerate(load_frames(), 1):
        uid = frame["uid"]
        path = folder / (uid + ".npz")
        sidecar = folder / (uid + ".json")
        if sidecar.exists():
            metadata = read(sidecar)
            assert metadata["stamp"] == stamp and digest(path) == metadata["sha"]
        else:
            xyz = processor.apply_transform(open_rgb(frame)).numpy()
            assert xyz.shape == (4, SIDE, SIDE) and np.isfinite(xyz).all()
            assert set(np.unique(xyz[3])) <= {0.,1.}
            atomic_npz(path, xyz=xyz)
            metadata = dict(sha=digest(path), stamp=stamp)
            write(sidecar, metadata)
        manifest[uid] = metadata["sha"]
        if i % 6 == 0:
            print(f"MoGe {i}", flush=True)
    write(PRIVATE / "moge_hashes.json", manifest)
    write(OUT / "moge_seal.json", dict(count=len(manifest), hashes=digest(PRIVATE / "moge_hashes.json"),
        stamp=stamp, **cuda_summary(started)))


def pda_sparse(rgb, pixels, depth, size):
    factor = min(1., 640 / max(size))
    width, height = np.round(size * factor).astype(int)
    image = cv2.resize(rgb, (width,height), interpolation=cv2.INTER_AREA)
    cells, chosen = nearest_points(pixels, depth, width, height, size)
    sparse = np.zeros((height,width), np.float32)
    sparse[cells[:,1],cells[:,0]] = depth[chosen]
    return image, sparse


@torch.inference_mode()
def run_maps(sources, weights_root):
    requirements(sources, "prior_depth", "Prior-Depth-Anything")
    from prior_depth_anything import PriorDepthAnything

    assets = require_weights(weights_root, "prior_depth")
    completion = PriorDepthAnything(device="cuda:0", version="1.1", mde_dir=str(assets),
        ckpt_dir=str(assets), frozen_model_size="vitb", conditioned_model_size="vitb").eval().requires_grad_(False)
    torch.manual_seed(20261121)
    hashes = read(PRIVATE / "moge_hashes.json")
    assert digest(PRIVATE / "moge_hashes.json") == read(OUT / "moge_seal.json")["hashes"]
    stamp = dict(protocol=digest(PROTOCOL), observation=digest(OUT / "panel_seal.json"), code=digest(__file__),
                 geometry=digest(Path(__file__).with_name("geometry.py")), moge=digest(OUT / "moge_seal.json"),
                 assets=digest(ASSETS))
    folder = PRIVATE / "maps"
    folder.mkdir(mode=0o700, exist_ok=True)
    manifest, counts = {}, []
    began = time.perf_counter()
    for i, frame in enumerate(load_frames(), 1):
        uid = frame["uid"]
        path = folder / (uid + ".npz")
        sidecar = folder / (uid + ".json")
        if sidecar.exists():
            meta = read(sidecar)
            assert meta["stamp"] == stamp and digest(path) == meta["sha"]
        else:
            original = PRIVATE / "moge" / (uid + ".npz")
            obs_path = PRIVATE / "observations" / (uid + ".npz")
            assert digest(original) == hashes[uid] and digest(obs_path) == frame["observation_sha"]
            assert digest(frame["calibration"]) == frame["calibration_sha"]
            with np.load(original, allow_pickle=False) as f:
                xyz = f["xyz"]
            with np.load(obs_path, allow_pickle=False) as f:
                obs = {k:f[k] for k in ("cells","uv","camera","hold","projected_points","multi_occupied_cells","conflict_cells")}
            y, x = obs["cells"][:,1], obs["cells"][:,0]
            mask = (xyz[3] == 1) & (xyz[2] >= .25)
            fit = ~obs["hold"] & mask[y,x]
            evaluated = obs["hold"] & mask[y,x]
            assert not (fit & evaluated).any()
            spec, size, _ = calibration(frame)
            rays, reprojection = calibrated_grid(spec,size)
            ratio = 1.
            if int(fit.sum()) >= 5:
                ratio = float(np.exp(np.median(np.log(obs["camera"][fit,2]) - np.log(xyz[2,y[fit],x[fit]]))))
            global_z = np.clip(xyz[2] * ratio, .25, 250.).astype(np.float32)
            wrong_z = wrong_locations(global_z, mask)
            image, sparse = pda_sparse(open_rgb(frame), obs["uv"][fit], obs["camera"][fit,2], size)
            num_pda = int((sparse > 0).sum())
            if num_pda < 5:
                depth_prior = xyz[2].copy()
                invalid_pda = 0
            else:
                depth_prior = completion.infer_one_sample(image=image, prior=sparse, visualize=False).cpu().numpy()
                assert depth_prior.ndim == 2
                depth_prior = cv2.resize(depth_prior,(SIDE,SIDE),interpolation=cv2.INTER_NEAREST)
                valid_pda = np.isfinite(depth_prior) & (depth_prior > 0.)
                invalid_pda = int((~valid_pda & mask).sum())
                depth_prior = np.where(valid_pda, np.clip(depth_prior,.25,250.), xyz[2]).astype(np.float32)
            expected_pixels = rays[y[evaluated], x[evaluated]]
            if len(expected_pixels):
                unsnapped = obs["uv"][evaluated]
                np.testing.assert_allclose(project_camera(obs["camera"][evaluated].astype(float),spec),
                                           unsnapped, atol=.004, rtol=0)
                cell_error = float(np.median(np.linalg.norm(project_camera(expected_pixels.astype(float),spec)-unsnapped,axis=1)))
            else:
                cell_error = None
            depths = dict(camera_rays=xyz[2],global_scale=global_z,prior_depth=depth_prior,wrong_geometry=wrong_z)
            atomic_npz(path,moge=xyz,mask=mask,rays=rays,fit=fit,held=evaluated,**depths)
            meta = dict(stamp=stamp, sha=digest(path), target=frame["target"],uid=uid,
                fit_n=int(fit.sum()), held_n=int(obs["hold"].sum()), eval_n=int(evaluated.sum()), pda_fit_n=num_pda,
                projected_n=int(obs["projected_points"]), occupied_n=len(y), colliding_n=int(obs["multi_occupied_cells"]),
                conflict_n=int(obs["conflict_cells"]), global_ratio=ratio, valid_fraction=float(mask.mean()),
                pda_fallback=bool(num_pda < 5), pda_bad_valid_pixels=invalid_pda, distortion_assumed_zero=spec.validated()[3],
                grid_projection_error_pixels=reprojection, median_retained_ray_snap_px=cell_error)
            write(sidecar,meta)
        counts.append({k:v for k,v in meta.items() if k not in ("stamp","sha")})
        manifest[uid]=meta["sha"]
        if i % 6 == 0:
            print(f"Prepared point maps {i}", flush=True)
    write(PRIVATE / "map_hashes.json",manifest)
    write(OUT / "maps_seal.json",dict(count=len(manifest), hashes=digest(PRIVATE / "map_hashes.json"), stamp=stamp,
        **cuda_summary(began), fit_only_verified=True, answers_opened=False))
    write(OUT / "alignment_frames.json",counts)


def map_arm(f, name):
    return f["moge"] if name == "moge" else make_xyz(f[name],f["mask"],f["rays"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["moge","maps"])
    p.add_argument("sources",type=Path)
    p.add_argument("weights",type=Path)
    args=p.parse_args()
    if args.stage=="moge":
        run_moge(args.sources,args.weights)
    else:
        run_maps(args.sources,args.weights)

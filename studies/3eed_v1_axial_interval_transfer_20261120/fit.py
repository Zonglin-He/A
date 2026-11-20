"""Fit every arm on its own source platform and report natural and displaced source holdout."""
import csv
from time import perf_counter

import numpy as np

from axial import (ACTIONS, CONFIG, COPIES, HERE, PRIVATE, RESULTS, SOURCE, PLATFORMS, atomic,
                   choose, cloud_to_sensor, digest, evidence, features_at, fit_linear, read, register,
                   reversed_profile, sensor_frame_box, standardizer, standardized, translated)
from batch_yaw_iou import many_iou


def csv_rows(path, rows):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def source_split(platform, subset):
    data = SOURCE/"features_PRIVATE"
    splits = read(data/"SEALED.json")["splits"]
    tag = f"{platform}_{subset}"
    folder = data/tag
    assert digest(folder/"SEAL.json") == splits[tag]["seal_sha256"]
    seal = read(folder/"SEAL.json")
    roster_path = SOURCE/"inputs_PRIVATE"/f"{tag}.json"
    assert digest(roster_path) == seal["input_sha256"]
    roster = read(roster_path)
    old = HERE.parent/"3eed_v2_fixed_instance_geometry_20261119/calibration_PRIVATE"/f"{platform}_pairs.npz"
    with np.load(old, allow_pickle=False) as archived:
        original_ious = archived[f"{subset}_ious"][:, 0]
    features, labels = [], []
    for name, receipt in sorted(seal["files"].items()):
        path = folder/name
        assert digest(path) == receipt["sha256"]
        with np.load(path, allow_pickle=False) as block:
            eligible = block["origin"] < 2
            points = block["points"][eligible]
            boxes = block["v1_box"][eligible]
            targets = block["gt"][eligible, 0]
            views = block["view_index"][eligible]
        for cloud, box, gt, view in zip(points, boxes, targets, views):
            assert cloud.shape == (8192, 3)
            sensor = cloud_to_sensor(cloud, platform, view)
            anchor = sensor_frame_box(box, platform, view)
            target = sensor_frame_box(gt, platform, view)
            starts = translated(anchor, ACTIONS[[0, 2, 4]])
            feat = np.stack([features_at(sensor, b) for b in starts])
            ious = np.stack([many_iou(translated(b, ACTIONS), target) for b in starts])
            assert abs(ious[1, 2] - original_ious[len(labels)]) < 1e-7
            # Positions shared by adjacent starts are identical physical boxes.
            np.testing.assert_allclose(ious[0, 2:], ious[1, :3], atol=1e-9, rtol=0)
            np.testing.assert_allclose(ious[1, 2:], ious[2, :3], atol=1e-9, rtol=0)
            features.append(feat)
            labels.append(ious)
    assert len(labels) == len(roster) == seal["counts"]["natural"] == len(original_ious)
    return np.stack(features), np.stack(labels), roster


def training_examples(features, ious, starts, mean, scale):
    x = features[:, starts].reshape(-1, features.shape[-1])
    hit = (ious[:, starts].reshape(-1, 5) > .5).astype(float)
    train_x = np.concatenate((x, reversed_profile(x)), axis=0)
    train_hit = np.concatenate((hit, hit[:, ::-1]), axis=0)
    return standardized(train_x, mean, scale), train_hit


def validate(arm, x, ious, weight, mean, scale):
    rows = []
    kind = arm.split("_")[1]
    for label, selection in (("natural", [1]), ("displaced", [0, 2])):
        profiles = x[:, selection].reshape(-1, x.shape[-1])
        actual = ious[:, selection].reshape(-1, 5)
        score = evidence(profiles, weight, mean, scale, kind)
        choice = choose(score)
        ref, val = actual[:, 2], actual[np.arange(len(actual)), choice]
        rows.append(dict(arm=arm, starts=label, n=len(val), moved=int((choice!=2).sum()),
                         start_acc50=100*np.mean(ref>.5), Acc25=100*np.mean(val>.25),
                         Acc50=100*np.mean(val>.5), mIoU=100*val.mean(),
                         rescue50=int(((ref<=.5)&(val>.5)).sum()), harm50=int(((ref>.5)&(val<=.5)).sum()),
                         available50=int(((ref<=.5)&(actual.max(1)>.5)).sum())))
    return rows


def run():
    lock = register()
    folder = PRIVATE/"heads"
    folder.mkdir(exist_ok=True, mode=0o700)
    if (folder/"SEAL.json").exists():
        raise RuntimeError("The fitted weights have already been sealed")
    results, training, files = [], {}, {}
    for platform in PLATFORMS:
        tick = perf_counter()
        fit_x, fit_iou, fit_rows = source_split(platform, "fit")
        held_x, held_iou, held_rows = source_split(platform, "calibration")
        assert len(fit_rows) == CONFIG["source"]["natural_fit_per_source"]
        assert not ({r["recording"] for r in fit_rows} & {r["recording"] for r in held_rows})
        assert not ({r["point_sha256"] for r in fit_rows} & {r["point_sha256"] for r in held_rows})
        mean, scale = standardizer(fit_x[:, 1])
        arrays = {s:training_examples(fit_x, fit_iou, ix, mean, scale) for s, ix in COPIES.items()}
        smooth = CONFIG["training"]["l2_per_coefficient_including_intercept"] + .5*max(
            np.linalg.eigvalsh(x.T@x/len(x))[-1] for x, _ in arrays.values())
        weights = dict(mean=mean, scale=scale)
        platform_report = {}
        for start, (x, labels) in arrays.items():
            for kind in ("interval", "bce"):
                arm = f"{start}_{kind}"
                w, numeric = fit_linear(x, labels, kind, smooth)
                assert np.isfinite(w).all() and np.isfinite(numeric["objective"])
                weights[arm] = w
                platform_report[arm] = dict(rows_per_step=len(x), parameters=w.size, **numeric)
                for val in validate(arm, held_x, held_iou, w, mean, scale):
                    results.append(dict(source=platform, **val))
        path = folder/f"{platform}.npz"
        np.savez(path, **weights)
        files[path.name] = digest(path)
        natural = fit_iou[:, 1]
        deliberate = fit_iou[:, [0, 2], 2]
        training[platform] = dict(fit=len(fit_rows), holdout=len(held_rows),
            fit_recordings=len({r["recording"] for r in fit_rows}),
            holdout_recordings=len({r["recording"] for r in held_rows}),
            natural_five_position_repair=int(((natural[:,2]<=.5)&(natural.max(1)>.5)).sum()),
            constructed_reversible_misses=int(((natural[:,2,None]>.5)&(deliberate<=.5)).sum()),
            smoothness=float(smooth), elapsed_seconds=perf_counter()-tick, heads=platform_report)
        print(platform, len(fit_rows), len(held_rows), "four heads finished", round(perf_counter()-tick,1), flush=True)
    assert sum(x["natural_five_position_repair"] for x in training.values()) == 11
    assert sum(x["constructed_reversible_misses"] for x in training.values()) == 2473
    csv_rows(RESULTS/"source_holdout.csv", results)
    atomic(RESULTS/"source_fit.json", training)
    atomic(folder/"SEAL.json", dict(lock=lock, files=files, source_labels_only=True,
                                    target_updates=0, used_target_labels=False, FINAL_access=False))


if __name__ == "__main__":
    run()

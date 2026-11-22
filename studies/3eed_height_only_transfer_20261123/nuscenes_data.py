"""Join released language commands to calibrated single-sweep mini observations."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

import numpy as np
from scipy.spatial.transform import Rotation

from common import HERE, OUT, PRIVATE, PROTOCOL, lock, roster
from live import geometry
from crossplatform.round7.camera_projection import model_to_camera_matrix, project_model_points


TALK_REVISION = "3fde52aee3341312196ab9b2f999337eff0693a0"
INPUT_SEAL = OUT / "talk2car_input_seal.json"
PANEL = PRIVATE / "talk2car_panel.json"
MATCH = PRIVATE / "talk2car_annotation_keys.json"
TABLES = ("sample", "sample_data", "scene", "calibrated_sensor", "sensor", "ego_pose")
ADDENDUM = HERE / "EXTERNAL_BRIDGE_ADDENDUM.json"
SENSOR_AXES = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
SAMPLE_PATHS = re.compile(r"(?:samples|sweeps)/(?:CAM_FRONT/[^/]+\.jpg|LIDAR_TOP/[^/]+\.pcd\.bin)")


def verify_archive(archive):
    cfg = roster.read(PROTOCOL)["external_data"]
    receipt = roster.read(archive.with_name(archive.name + ".receipt.json"))
    assert receipt["source"] == cfg["archive"] and receipt["bytes"] == cfg["expected_bytes"]
    assert archive.stat().st_size == receipt["bytes"] and roster.sha(archive) == receipt["sha256"]
    return receipt


def source_tables(archive, folder):
    keep = {f"v1.0-mini/{t}.json" for t in TABLES}
    marker = folder / "observations.json"
    if marker.exists():
        receipt = roster.read(marker)
        assert receipt["archive"] == roster.read(archive.with_name(archive.name + ".receipt.json"))["sha256"]
        assert all(roster.sha(folder / path) == sha for path, sha in receipt["tables"].items())
        return receipt
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    found = Counter()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            raw = path.as_posix().removeprefix("./")
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                continue
            if raw not in keep:
                continue
            install = folder / raw
            install.parent.mkdir(parents=True, exist_ok=True)
            fileobj = tar.extractfile(member)
            assert fileobj is not None
            with fileobj, install.open("wb") as output:
                shutil.copyfileobj(fileobj, output, length=2**20)
            assert install.stat().st_size == member.size
            found[raw.rsplit("/", 1)[0]] += 1
            if found["v1.0-mini"] == len(TABLES):
                break
    assert found["v1.0-mini"] == len(TABLES)
    receipt = dict(archive=roster.read(archive.with_name(archive.name + ".receipt.json"))["sha256"],
                   extracted=dict(found), tables={x: roster.sha(folder / x) for x in sorted(keep)},
                   target_boxes_extracted=False)
    roster.write(marker, receipt)
    return receipt


def extract_sensor_subset(archive, root, expected):
    waiting = {name for name in expected if not (root / name).is_file()}
    assert all(SAMPLE_PATHS.fullmatch(name) and ".." not in PurePosixPath(name).parts for name in expected)
    if waiting:
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                name = member.name.removeprefix("./")
                if name not in waiting or not member.isfile():
                    continue
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                working = dest.with_suffix(dest.suffix + ".tmp")
                with tar.extractfile(member) as fp, working.open("wb") as output:
                    shutil.copyfileobj(fp, output, length=2**20)
                assert working.stat().st_size == member.size
                working.replace(dest)
                waiting.remove(name)
                if not waiting:
                    break
    assert not waiting and all((root / name).stat().st_size > 0 for name in expected)


def matrix(record):
    wxyz = np.asarray(record["rotation"], dtype=float)
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    out[:3, 3] = np.asarray(record["translation"], dtype=float)
    return out


def obs_index(folder):
    data = {}
    for t in TABLES:
        rows = roster.read(folder / f"v1.0-mini/{t}.json")
        data[t] = {row["token"]: row for row in rows}
        assert len(data[t]) == len(rows)
    front, by_scene = {}, defaultdict(list)
    for item in data["sample_data"].values():
        calibration = data["calibrated_sensor"][item["calibrated_sensor_token"]]
        sensor = data["sensor"][calibration["sensor_token"]]["channel"]
        if sensor == "CAM_FRONT":
            front[item["token"]] = item
        elif sensor == "LIDAR_TOP":
            scene = data["sample"][item["sample_token"]]["scene_token"]
            by_scene[scene].append(item)
    assert len(data["sample"]) == 404 and sum(f["is_key_frame"] for f in front.values()) == 404
    pairs = {}
    times = {s: [item["timestamp"] for item in sorted(group, key=lambda r: r["timestamp"])] for s, group in by_scene.items()}
    scans = {s: sorted(group, key=lambda r: r["timestamp"]) for s, group in by_scene.items()}
    for token, camera in front.items():
        scene = data["sample"][camera["sample_token"]]["scene_token"]
        sort_index = bisect_left(times[scene], camera["timestamp"])
        close = scans[scene][max(0, sort_index-1):sort_index+1]
        most = min(close, key=lambda r: (abs(camera["timestamp"] - r["timestamp"]), r["timestamp"]))
        if abs(camera["timestamp"] - most["timestamp"]) <= 60_000:
            pairs[token] = dict(CAM_FRONT=camera, LIDAR_TOP=most)
    return pairs, data


def eligible_commands(folder, pairs, samples):
    from ijson import parse

    sources = roster.read(folder / "sources.json")
    fields = {"scene_token", "sample_token", "box_token", "command", "command_token", "t2c_img"}
    for split in ("train", "val"):
        src = sources[split]
        path = folder / f"{split}_commands.json"
        assert src["filename"] == path.name and TALK_REVISION in src["source"]
        assert path.stat().st_size == src["bytes"] and roster.sha(path) == src["sha256"]
        with path.open("rb") as stream:
            for key, event, value in parse(stream):
                if key == "commands.item" and event == "start_map":
                    allowed = {}
                elif key == "commands.item" and event == "end_map":
                    assert set(allowed) >= fields - {"t2c_img"}
                    if allowed["sample_token"] in pairs and allowed["box_token"]:
                        camera = pairs[allowed["sample_token"]]["CAM_FRONT"]
                        assert samples[camera["sample_token"]]["scene_token"] == allowed["scene_token"]
                        yield split, allowed
                elif key.startswith("commands.item.") and key.rsplit(".", 1)[1] in fields and event == "string":
                    allowed[key.rsplit(".", 1)[1]] = value


def observation(samples, info, root, token):
    cam, lidar = [samples[token][k] for k in ("CAM_FRONT", "LIDAR_TOP")]
    e_cam, e_lidar = [info["calibrated_sensor"][row["calibrated_sensor_token"]] for row in (cam, lidar)]
    g_cam, g_lidar = [info["ego_pose"][row["ego_pose_token"]] for row in (cam, lidar)]
    lens_to_world = matrix(g_cam) @ matrix(e_cam)
    lidar_to_world = matrix(g_lidar) @ matrix(e_lidar)
    transform = np.linalg.inv(lens_to_world) @ lidar_to_world
    K = np.asarray(e_cam["camera_intrinsic"], dtype=float)
    assert K.shape == (3, 3) and 0 < cam["width"] <= 2048 and 0 < cam["height"] <= 2048
    image, scan = [root / row["filename"] for row in (cam, lidar)]
    assert image.is_file() and scan.is_file() and SAMPLE_PATHS.fullmatch(cam["filename"]) and SAMPLE_PATHS.fullmatch(lidar["filename"])
    case = root / "adapter" / roster.key("mini-sample", token)
    case.mkdir(parents=True, exist_ok=True)
    points = np.fromfile(scan, dtype="<f4").reshape(-1, 5)[:, :4]
    assert len(points) > 0 and np.isfinite(points).all()
    point_path, calib = case / "points.npy", case / "camera.json"
    np.save(point_path, points, allow_pickle=False)
    roster.write(calib, dict(image_intrinsic=K.tolist(), image_extrinsic=(np.linalg.inv(transform) @ SENSOR_AXES).tolist(),
                            camera_distortion=[0.] * 5))
    adapter = dict(target="waymo", suffix=0, waymo_view="F", rgb=str(image), points=str(point_path), calibration=str(calib))
    spec, size, got = geometry.camera(adapter)
    assert (size == [cam["width"], cam["height"]]).all()
    np.testing.assert_allclose(got, transform, atol=5.e-12, rtol=0.)
    np.testing.assert_allclose(model_to_camera_matrix(spec), transform, atol=5.e-12, rtol=0.)
    points = points[::23, :3].astype(float)
    world = geometry.rigid_xyz(points, lidar_to_world)
    xyz = geometry.rigid_xyz(world, np.linalg.inv(lens_to_world))
    front = xyz[:, 2] > .1
    xy = xyz[front] @ K.T
    expected = xy[:, :2] / xy[:, 2:3]
    pp = project_model_points(points, spec, size)
    visible = (expected >= 0).all(axis=1) & (expected < size).all(axis=1)
    np.testing.assert_array_equal(pp["in_image"][front], visible)
    assert not pp["in_image"][~front].any()
    np.testing.assert_allclose(pp["uv"][front][visible], expected[visible], atol=1.e-7, rtol=0.)
    positive = int(pp["in_image"].sum())
    assert positive >= 10
    adapter.update(rgb_sha=roster.sha(image), point_sha=roster.sha(point_path), camera_sha=roster.sha(calib))
    return adapter, np.linalg.inv(lidar_to_world).tolist(), np.linalg.inv(transform).tolist(), cam["timestamp"] - lidar["timestamp"], positive


def register(archive, command_folder):
    pins = lock(Path(__file__))
    asset = verify_archive(archive)
    location = archive.parent / "mini_observations"
    sensor_seal = source_tables(archive, location)
    pairs, data = obs_index(location)
    lines = list(eligible_commands(command_folder, pairs, data["sample"]))
    expected = {sensor["filename"] for _, cmd in lines for sensor in pairs[cmd["sample_token"]].values()}
    assert lines and expected
    extract_sensor_subset(archive, location, expected)
    frames, lookup, reusable, counts = [], {}, {}, Counter()
    time_deltas = []
    for split, cmd in lines:
        sample = cmd["sample_token"]
        uid = roster.key("talk2car-command", cmd["command_token"])
        scene = roster.key("talk2car-scene", cmd["scene_token"])
        if sample not in reusable:
            reusable[sample] = observation(pairs, data, location, sample)
        obs, world_to_lidar, camera_to_lidar, dt, visible = reusable[sample]
        camera = pairs[sample]["CAM_FRONT"]
        frames.append(dict(**obs, uid=uid, recording=scene, panel="talk2car_mini", text=cmd["command"], split=split,
                           camera_keyframe=camera["is_key_frame"],
                           seed=int(roster.key("talk2car-fixed-sampler", sample)[:16], 16)))
        lookup[uid] = dict(sample=camera["sample_token"], box=cmd["box_token"], command=cmd["command_token"],
                           split=split, world_to_lidar=world_to_lidar, camera_to_lidar=camera_to_lidar)
        time_deltas.append(abs(dt)/1000)
        counts[split] += 1
    assert len(frames) == len(lookup) and frames, "No annotation and sensor intersection in mini"
    roster.write(PANEL, frames)
    roster.write(MATCH, lookup)
    roster.write(INPUT_SEAL, dict(pins=pins, code=roster.sha(__file__), panel=roster.sha(PANEL),
        annotation_keys=roster.sha(MATCH), addendum=roster.sha(ADDENDUM), asset=asset, language=roster.read(command_folder / "sources.json"),
        sensor_extraction=sensor_seal, descriptions=len(frames), physical_camera_records=len(reusable), by_annotation_split=dict(counts),
        camera_keyframe_descriptions=sum(r["camera_keyframe"] for r in frames), sensor_files=len(expected),
        scenes=len({x["recording"] for x in frames}), calibrated_input_checks=len(reusable),
        max_camera_lidar_time_offset_ms=max(time_deltas), median_camera_lidar_offset_ms=float(np.median(time_deltas)),
        target_gt_opened=False, FINAL_access=False))
    print("External intersection sealed", dict(counts), "scenes", len({x["recording"] for x in frames}), flush=True)


def panel():
    seal = roster.read(INPUT_SEAL)
    assert seal["code"] == roster.sha(__file__) and seal["pins"] == lock(Path(__file__))
    assert seal["addendum"] == roster.sha(ADDENDUM)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"] and roster.sha(PANEL) == seal["panel"]
    rows = roster.read(PANEL)
    assert len(rows) == seal["descriptions"] and len({r["uid"] for r in rows}) == len(rows)
    return rows, seal


def annotations(archive):
    verify_archive(archive)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name.removeprefix("./") == "v1.0-mini/sample_annotation.json" and member.isfile():
                with tar.extractfile(member) as source:
                    rows = json.load(source)
                return {row["token"]: row for row in rows}
    raise ValueError("Official sample annotations missing from mini")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("commands", type=Path)
    args = parser.parse_args()
    register(args.archive, args.commands)

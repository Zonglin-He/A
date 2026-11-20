"""An axial translation policy with one source-fitted linear readout."""

from hashlib import sha256
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logsumexp, softmax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R183 = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119"
SOURCE = HERE.parent / "3eed_v1_supervised_instance_grounding_20261119"
PAL = HERE.parent / "3eed_k16_utility_loss_20261120"
PRIVATE = HERE / "run_PRIVATE"
RESULTS = HERE / "results"
sys.path[:0] = [str(R183), str(ROOT), str(ROOT/"research/round6/source_validation_scoring"),
                str(ROOT/"crossplatform/round7"), str(ROOT/"crossplatform/round7/experiment"),
                str(ROOT/"crossplatform/round3/p0/upstream")]
from scope183 import DIRECTIONS, PLATFORMS, VIEWS, atomic, digest, ordinary, read
from runner_helpers import sample_points
from camera_projection import WAYMO_ANGLES, z_rotation
from crossplatform.candidate_frontend import sensor_frame_box

CONFIG = read(HERE/"PROTOCOL.json")
ACTIONS = np.array(CONFIG["geometry"]["normalized_actions"])
ARMS = tuple(CONFIG["arms"])
COPIES = {"repeat": [1, 1, 1], "shift": [0, 1, 2]}
INTERVALS = np.zeros((16, 5), dtype=np.float64)
row = 1
for left in range(5):
    for right in range(left, 5):
        INTERVALS[row, left:right+1] = 1
        row += 1
BITS = 2**np.arange(5)
INTERVAL_ID = np.full(32, -1)
INTERVAL_ID[(INTERVALS@BITS).astype(int)] = np.arange(16)


def register():
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    RESULTS.mkdir(exist_ok=True)
    named = {"source_manifest": (SOURCE/"inputs_PRIVATE/SEAL.json", CONFIG["source"]["source_manifest_seal"]),
             "source_features": (SOURCE/"features_PRIVATE/SEALED.json", CONFIG["source"]["source_feature_seal"]),
             "ordinary_roster": (R183/"ordinary_PRIVATE/SEALED.json", CONFIG["target"]["roster_seal"]),
             "v1_predictions": (R183/"predictions_PRIVATE/SEALED.json", CONFIG["target"]["sealed_baseline_boxes"])}
    for p, expected in named.values():
        assert digest(p) == expected
    current = read(ROOT/"crossplatform/CURRENT_METHOD.json")
    assert current["paper_main_method"]["status"] == "locked"
    assert current["rounds"] == 2 and current["reference_slots"] == 4
    for f, pin in current["paper_main_method"]["local_core_sha256"].items():
        assert digest(ROOT/f) == pin
    code = ["axial.py", "fit.py", "predict.py"]
    dependencies = [ROOT/"crossplatform/round7/experiment/runner_helpers.py",
                    ROOT/"crossplatform/round3/p0/upstream/utils/transform_waymo.py",
                    ROOT/"crossplatform/candidate_frontend.py", ROOT/"crossplatform/round7/camera_projection.py",
                    ROOT/"research/round6/source_validation_scoring/batch_yaw_iou.py", R183/"scope183.py"]
    receipt = dict(protocol=digest(HERE/"PROTOCOL.json"),
                   code={f:digest(HERE/f) for f in code},
                   dependencies={str(f.relative_to(ROOT)):digest(f) for f in dependencies},
                   inputs={k:v[1] for k,v in named.items()}, target_labels_accessed=False, FINAL_access=False)
    path = PRIVATE/"LOCK.json"
    if path.exists():
        assert read(path) == receipt
    else:
        atomic(path, receipt)
    return digest(path)


def cloud_to_sensor(cloud, platform, view):
    points = np.asarray(cloud, dtype=np.float64).copy()
    if platform == "waymo":
        return points @ z_rotation(WAYMO_ANGLES[view])
    if platform == "drone":
        points[:, 2] -= 1.8
    return points


def sensor_sample(row):
    from utils.transform_waymo import transform_to_front_view
    cloud, _, _ = sample_points(row)
    cloud = cloud[:8192]
    target, view = row["platform"], row["suffix"]
    if target == "waymo":
        cloud, _ = transform_to_front_view(cloud, np.empty((0, 7)), VIEWS[view])
        cloud = cloud.astype(np.float32)
    return cloud_to_sensor(cloud, target, view)


def frame_geometry(box):
    length, width = box[3:5]
    angle = box[6]
    if width > length:
        length, width = width, length
        angle += np.pi/2
    axis = np.array([np.cos(angle), np.sin(angle)])
    if axis[0] < -1e-12 or (abs(axis[0]) <= 1e-12 and axis[1] < 0):
        axis = -axis
    return axis, length, width


def translated(box, offsets):
    axis, length, _ = frame_geometry(box)
    boxes = np.repeat(box[None], len(offsets), axis=0)
    boxes[:, :2] += np.asarray(offsets)[:, None] * length * axis
    return boxes


def features_at(points, box):
    axis, length, width = frame_geometry(box)
    delta = points - box[:3]
    along = delta[:, :2] @ axis
    across = delta[:, 1]*axis[0] - delta[:, 0]*axis[1]
    keep = (np.abs(along) <= 5*length/6) & (np.abs(across) <= width/2) & (np.abs(delta[:, 2]) <= box[5]/2)
    _, unique = np.unique(points[keep], axis=0, return_index=True)
    bins = np.floor(along[keep][unique]*6/length + 5).astype(int).clip(0, 9)
    counts = np.bincount(bins, minlength=10)
    return np.r_[counts/max(1, len(unique)), np.log1p(len(unique)), width/length, box[5]/length]


def reversed_profile(x):
    y = x.copy()
    y[..., :10] = x[..., 9::-1]
    return y


def standardizer(x):
    combined = np.concatenate((x, reversed_profile(x)), axis=0)
    mean, scale = combined.mean(0), combined.std(0)
    scale[scale <= 1e-12] = 1.
    return mean, scale


def standardized(x, mean, scale):
    return np.column_stack(((x - mean)/scale, np.ones(len(x))))


def losses_and_gradient(x, hit, weight, kind, penalty):
    scores = x @ weight
    if kind == "interval":
        idx = INTERVAL_ID[hit.astype(int)@BITS]
        assert (idx >= 0).all()
        loss = np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(x)), idx])
        residual = softmax(scores, axis=1)
        residual[np.arange(len(x)), idx] -= 1.
    else:
        loss = np.sum(np.logaddexp(0., scores) - hit*scores) / len(x)
        residual = expit(scores) - hit
    gradient = x.T @ residual / len(x) + penalty*weight
    return float(loss + .5*penalty*np.sum(weight**2)), gradient


def fit_linear(x, y, kind, smooth):
    penalty = CONFIG["training"]["l2_per_coefficient_including_intercept"]
    steps = CONFIG["training"]["steps"]
    beta = (np.sqrt(smooth) - np.sqrt(penalty))/(np.sqrt(smooth) + np.sqrt(penalty))
    weight = np.zeros((x.shape[1], 16 if kind == "interval" else 5))
    ahead = weight.copy()
    for _ in range(steps):
        _, gradient = losses_and_gradient(x, y, ahead, kind, penalty)
        new = ahead-gradient/smooth
        ahead = new + beta*(new-weight)
        weight = new
    loss, gradient = losses_and_gradient(x, y, weight, kind, penalty)
    return weight, dict(steps=steps, objective=loss, stationarity_l2=float(np.linalg.norm(gradient)))


def evidence(x, head, mean, scale, kind):
    def ordinary_score(feat):
        z = standardized(feat, mean, scale) @ head
        return softmax(z, axis=1) @ INTERVALS if kind == "interval" else expit(z)
    return .5*(ordinary_score(x) + ordinary_score(reversed_profile(x))[:, ::-1])


def choose(score):
    priority = np.array([2, 3, 1, 4, 0])
    tied = score[:, priority] >= score.max(1, keepdims=True) - 1e-12
    return priority[np.argmax(tied, axis=1)]


def mirror_mask(target, n):
    order = sorted(range(n), key=lambda i: sha256(f"axial-mirror-v1|{target}|{i}".encode()).digest())
    result = np.zeros(n, dtype=bool)
    result[order[:n//2]] = True
    return result


def incumbents(key, parent):
    direction = parent["directions"][key]
    values = []
    for name, receipt in sorted(direction["files"].items()):
        path = R183/"predictions_PRIVATE"/key/name
        assert digest(path) == receipt["sha256"]
        with np.load(path, allow_pickle=False) as block:
            indices = block["ordinals"]
            assert receipt["first"] == sum(len(b) for b in values)
            np.testing.assert_array_equal(indices, np.arange(receipt["first"], receipt["last"]))
            values.append(block["boxes"][np.arange(len(indices)), block["choices"][:, 0]])
    result = np.concatenate(values)
    assert len(result) == direction["n"] and np.isfinite(result).all() and (result[:, 3:6] > 0).all()
    return result

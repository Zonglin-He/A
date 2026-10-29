"""Second CPU implementation for sealed 3EED pilot measurements.

Reads target metadata only after verifying every prediction and plan seal.
Publishes aggregate/per-pseudonym scores, never target coordinates or captions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from obb_iou import oriented_iou


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--schema", choices=['official7', 'historical_invalid9'], required=True,
                    help='The prior full9D interpretation is invalid; official loader uses first7 columns.')
    args = ap.parse_args()
    seal = json.loads((args.run / "sealed.json").read_text())
    assert sha(args.run / "plan.json") == seal["plan_sha256"]
    for name, expected in seal["files"].items():
        assert sha(args.run / name) == expected, name
    first = json.loads((args.run / ('scored_native.json' if args.schema == 'official7' else 'scored.json')).read_text())
    manifest = {x["id"]: x for x in json.loads(args.manifest.read_text())}
    rows, errors = [], []
    identity = np.eye(3)
    for row in first["rows"]:
        sid, arm = row["id"], row["arm"]
        sample = manifest[sid]
        metadata = json.loads(Path(sample["meta_path"]).read_text())
        gt = np.array(metadata["ground_info"][sample["obj_index"]]["bbox_3d"], float)
        if args.schema == 'official7':
            gt = gt[:7]
        yaw, pitch, roll = list(gt[6:]) + [0.] * (9 - len(gt))
        rot = (Rotation.from_euler('z', yaw).as_matrix() @
               Rotation.from_euler('y', pitch).as_matrix() @
               Rotation.from_euler('x', roll).as_matrix())
        yawrot = Rotation.from_euler('z', yaw).as_matrix()
        if sample["platform"] == "drone":
            gt[2] += 1.8
        if sample["platform"] == "waymo":
            index = int(Path(sample["lidar_path"]).parent.name.split('_')[-1])
            angle = [0., -45., 45., -90., 90.][index]
            q = Rotation.from_euler('z', angle, degrees=True).as_matrix()
            gt[:3] = q @ gt[:3]
            rot, yawrot = q @ rot, q @ yawrot
        with np.load(args.run / f"{sid}_{arm}.npz", allow_pickle=False) as pred:
            box = pred['boxes'][row['top_query']]
            measured = oriented_iou(gt[:3], gt[3:6], rot, box[:3], box[3:6], identity)
            yaw_only = oriented_iou(gt[:3], gt[3:6], yawrot, box[:3], box[3:6], identity)
            delta = (max(abs(measured - row['iou_full']), abs(yaw_only - row['iou_yaw_only']))
                     if args.schema == 'historical_invalid9' else abs(yaw_only - row['iou']))
            errors.append(delta)
            # A diagnostic ceiling only; GT never chooses a deployed prediction.
            proposal_ceiling = max(oriented_iou(gt[:3], gt[3:6], rot,
                                               b[:3], b[3:6], identity)
                                   for b in pred['boxes'])
        rows.append({"id": sid, "platform": sample['platform'], "arm": arm,
                     "readout": row['readout'], "iou_full": measured,
                     "iou_yaw_only": yaw_only, "independent_error": delta,
                     "all_proposals_oracle_iou": proposal_ceiling})
    max_error = max(errors)
    if max_error > 1e-6:
        raise RuntimeError(f"Independent IoU mismatch {max_error}")
    result = {"prediction_seal_verified": True,
              "scoring_agrees": True, "maximum_absolute_error": max_error,
              "schema": args.schema,
              "measurement_valid": args.schema == 'official7',
              "rows": rows,
              "scope": "Fixed development pilot. Oracle proposal IoU is diagnostic, never a TTA readout."}
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()

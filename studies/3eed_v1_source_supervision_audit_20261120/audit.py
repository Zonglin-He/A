"""Reconstruct source training supervision and check a predicate reversal without fitting."""
import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "PROTOCOL.json").read_text())
PARENT = "studies/3eed_recurrent_role_support_decoder_20261119"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def historical_preparation(repo, workspace, destination):
    for name in ("common.py", "prepare.py"):
        obj = f"{CONFIG['historical_commit']}:{PARENT}/{name}"
        data = subprocess.check_output(["git", "-C", str(repo), "show", obj])
        (destination / name).write_bytes(data)
    sys.path[:0] = [str(destination), str(workspace / "crossplatform/round7"),
                    str(workspace / "research/round5/audit"), str(workspace / "tools")]
    return importlib.import_module("prepare")


def natural_roster(source, workspace, prep):
    registered = json.loads((workspace / "research/round168/SOURCE_INPUT_SEAL.json").read_text())
    roster_file = workspace / "crossplatform/round161" / f"{source}_train_PRIVATE.json"
    assert digest(roster_file) == registered["monitored_source_manifests"][source]["train"]
    data = (workspace / "crossplatform/dataset_full/3eed_dataset").resolve()
    split_file = data / "splits" / f"{source}_train.txt"
    split = set(split_file.read_text().splitlines())
    raw_rows = json.loads(roster_file.read_text())
    rows = []
    other_fields = Counter()
    for raw in raw_rows:
        if not raw["positive"]:
            continue
        path = Path(raw["metapath"]).resolve()
        platform, recording, _, filename = path.relative_to(data).parts
        assert platform == raw["platform"] == source and recording in split and filename == "meta_info.json"
        caption = prep.canonical(raw["caption"])
        entries = [v for v in prep.meta(str(path))["ground_info"] if prep.canonical(v["caption"]) == caption]
        assert len(entries) == 1
        original = entries[0]
        other_fields.update(tuple(sorted(v)) for v in original.get("others", []))
        subject = prep.gt(original["bbox_3d"], raw)
        assert subject is not None
        row = {key: raw[key] for key in ("platform", "metapath", "pointpath", "imagepath", "suffix")}
        seed = int(prep.stable("source-point-sample", prep.frame(raw), raw["suffix"])[:8], 16)
        rows.append(dict(row, caption=caption, seed=seed, known_reference=False,
                         boxes=[subject.tolist(), [0.] * 7]))
    return rows, {"source_manifest_sha256": digest(roster_file), "official_train_sha256": digest(split_file),
                  "natural_rows": len(rows), "other_entry_schemas": {",".join(k): n for k, n in other_fields.items()}}


def choose_side(positions, low, high, comparable, anchor, sign, margin):
    distance = sign * (positions[comparable] - positions[anchor])
    candidates = comparable[distance > 0]
    if len(candidates) == 0:
        return None, "no_candidate"
    order = np.argsort(distance[distance > 0], kind="stable")
    nearest = candidates[order[0]]
    separation = low[nearest] - high[anchor] if sign > 0 else low[anchor] - high[nearest]
    if separation <= margin:
        return None, "overlapping_intervals"
    if len(order) > 1 and abs(positions[candidates[order[1]]] - positions[nearest]) <= margin:
        return None, "nearest_ambiguous"
    return int(nearest), "unique"


def enumerate_solutions(positions, low, high, comparable, anchor, sign, margin):
    """Check every object against the full defining inequalities without sorting."""
    signed = {i: sign * (positions[i] - positions[anchor]) for i in comparable}
    result = []
    for i, distance in signed.items():
        if distance <= 0:
            continue
        separated = low[i] > high[anchor] + margin if sign > 0 else low[anchor] > high[i] + margin
        distinct = all(j == i or d <= 0 or d - distance > margin for j, d in signed.items())
        if separated and distinct:
            result.append(i)
    return result


def reversal(row, prep):
    meta = prep.meta(row["metapath"])
    objects = prep.inventory(row, meta["ground_info"])
    names = np.asarray([word for word, _ in objects])
    boxes = np.stack([box for _, box in objects])
    indices = []
    for box in row["boxes"]:
        matches = np.flatnonzero(np.all(boxes == box, axis=1))
        assert len(matches) == 1
        indices.append(int(matches[0]))
    subject, anchor = indices
    lo, hi, visible, position = prep.projected_intervals(objects, row, meta)
    assert anchor != subject and visible[anchor] and visible[subject]
    comparable = np.flatnonzero(visible & (names == names[subject]) & (np.arange(len(objects)) != anchor))
    assert len(comparable) >= 2
    sign = row["side"]
    factual, how = choose_side(position, lo, hi, comparable, anchor, sign, prep.XLIMIT)
    assert how == "unique" and factual == subject
    alternative, status = choose_side(position, lo, hi, comparable, anchor, -sign, prep.XLIMIT)
    assert enumerate_solutions(position, lo, hi, comparable, anchor, sign, prep.XLIMIT) == [subject]
    assert enumerate_solutions(position, lo, hi, comparable, anchor, -sign, prep.XLIMIT) == (
        [] if alternative is None else [alternative])
    assert alternative not in (subject, anchor)
    assert -sign * (position[subject] - position[anchor]) < 0
    return status


def check_cache(source, split, rows, cache, reworded):
    labels, flags = [], []
    paths = sorted((cache / source).glob(f"{split}_*.npz"))
    assert paths
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            labels.append(stored["gt"])
            known = stored["known_ref"].astype(bool)
            flags.append(known)
        wording_file = reworded / source / path.name
        rewritten = set()
        if wording_file.is_file():
            with np.load(wording_file, allow_pickle=False) as feature:
                fields = {tuple(k.split("/")) for k in feature.files}
                assert not ({"gt", "known_ref", "quality", "boxes"} & {key for _, key in fields})
                rewritten = {int(offset) for offset, _ in fields}
        assert rewritten == set(np.flatnonzero(known))
    np.testing.assert_array_equal(np.concatenate(labels), np.asarray([r["boxes"] for r in rows], np.float32))
    np.testing.assert_array_equal(np.concatenate(flags), [r["known_reference"] for r in rows])


def run(repo, workspace, cache, reworded):
    output = {"protocol_sha256": digest(HERE / "PROTOCOL.json"),
              "source_audit": {}, "weights_sha256": {}, "rows": [],
              "target_examples_opened": 0, "parameters_trained": 0, "raw_annotations_written": 0}
    current = json.loads((workspace / "crossplatform/CURRENT_METHOD.json").read_text())
    with TemporaryDirectory() as temp:
        prep = historical_preparation(repo, workspace, Path(temp))
        for source in CONFIG["platforms"]:
            asset = current["decoders"][source]
            actual = digest(workspace / "crossplatform" / asset["path"])
            assert actual == asset["sha256"]
            output["weights_sha256"][source] = actual
            rows, scope = natural_roster(source, workspace, prep)
            output["source_audit"][source] = scope
            splits = prep.splits(rows, source)
            assert not ({prep.recording(r) for r in splits["train"]} &
                        {prep.recording(r) for r in splits["monitor"]})
            for split, natural in splits.items():
                derived, _ = prep.make_relations(natural, prep.SYNTHETIC_CAP[split])
                check_cache(source, split, natural + derived, cache, reworded)
                statuses = Counter(reversal(row, prep) for row in derived)
                assert sum(statuses.values()) == len(derived)
                output["rows"].append(dict(source=source, split=split, natural=len(natural), derived=len(derived),
                    original_subject_reproduced=len(derived), original_valid_after_reversal=0,
                    **{k: statuses[k] for k in ("unique", "no_candidate", "overlapping_intervals", "nearest_ambiguous")}))
    target = HERE / "RESULTS.json"
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if target.exists() and target.read_text() != text:
        raise RuntimeError("Audit differs from the published receipt")
    if not target.exists():
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(target)
    for row in output["rows"]:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-repo", required=True, type=Path)
    parser.add_argument("--workspace", type=Path, default=HERE.parents[1])
    parser.add_argument("--historical-source-cache", required=True, type=Path)
    parser.add_argument("--reworded-feature-cache", required=True, type=Path)
    args = parser.parse_args()
    run(args.archive_repo, args.workspace.resolve(), args.historical_source_cache, args.reworded_feature_cache)

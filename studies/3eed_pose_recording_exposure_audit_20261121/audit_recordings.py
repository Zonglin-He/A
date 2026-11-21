"""Check ordinary recording exposure using sealed manifests and ZIP member names."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
STUDIES = ROOT / "studies"
CROSS = ROOT / "crossplatform"
SOURCE = STUDIES / "3eed_v1_supervised_instance_grounding_20261119"
ORDINARY_VAL = STUDIES / "3eed_v2_ordinary_val_subject_ranking_20261119"
QWEN_FIRST = STUDIES / "3eed_frozen_qwen_metric_grounding_20261120"
QWEN_POSE_PANEL = STUDIES / "3eed_qwen_four_scale_recording_confirmation_20261121"
HF_API = "https://huggingface.co/api/datasets/RRRong/3EED/tree/main?expand=true"
PLATFORMS = ("waymo", "drone", "quad")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_qwen_panel(study, checksum):
    path = study / "private/panel.json"
    assert digest(path) == checksum
    return read(path)


def recording_names(rows, platform, *, qwen=False):
    identity = "target" if qwen else "platform"
    path = "calibration" if qwen else "metapath"
    return {Path(r[path]).parent.parent.name for r in rows if r[identity] == platform}


def release_and_recordings():
    receipt_path = CROSS / "full_dataset_verified.json"
    receipt = read(receipt_path)
    archive = Path(receipt["path"])
    assert receipt["official_match"] and receipt["all_members_CRC_testzip_passed"]
    assert archive.stat().st_size == receipt["bytes"]
    with urlopen(HF_API, timeout=15) as stream:
        assets = json.load(stream)
    release = next(f for f in assets if f["path"] == "3eed_dataset.zip")
    assert release["size"] == receipt["bytes"] and release["lfs"]["oid"] == receipt["sha256"]
    recording_ids = defaultdict(set)
    with ZipFile(archive) as file:
        assert len(file.infolist()) == receipt["archive_members"]
        for entry in file.namelist():
            parts = entry.split("/")
            if len(parts) >= 4 and parts[0] == "3eed_dataset" and parts[1] in PLATFORMS:
                recording_ids[parts[1]].add(parts[2])
    return recording_ids, {
        "official_asset": "https://huggingface.co/datasets/RRRong/3EED/blob/main/3eed_dataset.zip",
        "official_asset_lfs_sha256": release["lfs"]["oid"],
        "bytes": release["size"],
        "local_crc_receipt_sha256": digest(receipt_path),
        "zip_central_directory_members": receipt["archive_members"]
    }


def run():
    sys.path[:0] = [str(SOURCE), str(ORDINARY_VAL)]
    from scope import DATA, evaluation_rows, source_rows
    from scope183 import ordinary

    recording_ids, release = release_and_recordings()
    validation = ordinary()
    train_evaluation = evaluation_rows()
    train_receipt_path = ROOT / "research/round159/RECORDING_PREDICTION_SEAL.json"
    train_receipt = read(train_receipt_path)
    assert train_receipt["total"] == 2 * len(train_evaluation) == 3072 and not train_receipt["FINAL_access"]
    val_receipt_path = ORDINARY_VAL / "results/audit.json"
    val_receipt = read(val_receipt_path)
    assert val_receipt["n"] == 2 * sum(map(len, validation.values())) == 18620 and not val_receipt["FINAL_access"]

    first_seal_path = QWEN_FIRST / "results/panel_seal.json"
    pose_seal_path = QWEN_POSE_PANEL / "results/prediction_seal.json"
    first_seal = read(first_seal_path)
    pose_seal = read(pose_seal_path)
    first_panel = checked_qwen_panel(QWEN_FIRST, first_seal["panel"])
    pose_panel = checked_qwen_panel(QWEN_POSE_PANEL, pose_seal["panel"])
    assert len(first_panel) == 96 and len(pose_panel) == 416 and not pose_seal["FINAL_access"]

    source_seal_path = SOURCE / "inputs_PRIVATE/SEAL.json"
    source_seal = read(source_seal_path)
    evidence_paths = [train_receipt_path, val_receipt_path, first_seal_path, pose_seal_path, source_seal_path]
    rows = []
    for platform in PLATFORMS:
        prior_eval = recording_names(train_evaluation, platform)
        fit_cal = {}
        for role in ("fit", "calibration"):
            manifest = source_rows(platform, role)
            feature_path = SOURCE / "features_PRIVATE" / f"{platform}_{role}" / "SEAL.json"
            feature_seal = read(feature_path)
            assert feature_seal["input_sha256"] == source_seal["files"][f"{platform}_{role}.json"]
            # The source extractor also appends synthetic expressions to the natural cohort.
            assert feature_seal["counts"]["natural"] == len(manifest) and not feature_seal["FINAL_access"]
            evidence_paths.append(feature_path)
            fit_cal[role] = recording_names(manifest, platform)
        seen_validation = recording_names(validation[platform], platform)
        qwen = recording_names(first_panel + pose_panel, platform, qwen=True)
        previous = prior_eval | seen_validation | fit_cal["fit"] | fit_cal["calibration"] | qwen
        ordinary_sets = []
        for split in ("train", "val"):
            path = DATA / "splits" / f"{platform}_{split}.txt"
            names = path.read_text().split()
            assert len(names) == len(set(names)) and all(Path(n).name == n for n in names)
            listed = set(names)
            on_disk = {n for n in listed if (DATA / platform / n).is_dir()}
            from_release = listed & recording_ids[platform]
            assert on_disk == from_release
            ordinary_sets.append(listed)
            rows.append(dict(
                platform=platform, split=split, split_names=len(listed), zip_present=len(from_release),
                on_disk=len(on_disk), absent_from_official_zip=len(listed-from_release),
                prior_full_validation=len(on_disk & seen_validation), prior_train_eval=len(on_disk & prior_eval),
                prior_source_fit=len(on_disk & fit_cal["fit"]), prior_source_calibration=len(on_disk & fit_cal["calibration"]),
                prior_qwen=len(on_disk & qwen), exposed_union=len(on_disk & previous),
                train_eval_then_new_source_fit=len(on_disk & (fit_cal["fit"]-prior_eval)),
                train_eval_then_fit_then_new_source_calibration=len(on_disk & (fit_cal["calibration"]-prior_eval-fit_cal["fit"])),
                proven_unexposed_ordinary=len(on_disk-previous), split_sha256=digest(path)
            ))
        train, val = ordinary_sets
        assert train.isdisjoint(val)
        assert recording_ids[platform] <= train | val

    result = OUT / "ordinary_recordings.csv"
    with result.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {
        "official_split_entries": sum(r["split_names"] for r in rows),
        "official_zip_recordings": sum(r["zip_present"] for r in rows),
        "excluded_by_recorded_development": sum(r["exposed_union"] for r in rows),
        "absent_from_official_zip": sum(r["absent_from_official_zip"] for r in rows),
        "proven_unexposed_ordinary": sum(r["proven_unexposed_ordinary"] for r in rows)
    }
    receipt = dict(
        release=release, counts=counts, table_sha256=digest(result),
        verified_manifests={p.relative_to(ROOT).as_posix(): digest(p) for p in evidence_paths},
        criterion="The raw recording identity must be absent from the union of prior project Ordinary evaluation, "
                  "source fitting, source calibration, and Qwen development. No new frame can make an exposed recording independent.",
        archive_scan="ZIP member names only", new_model_predictions=0, new_target_labels_read=False,
        FINAL_access=False
    )
    (OUT / "exposure_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(counts, flush=True)


if __name__ == "__main__":
    run()

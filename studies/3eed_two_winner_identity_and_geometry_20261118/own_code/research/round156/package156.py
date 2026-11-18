"""Build the anonymous R156 public release; choose downstream files from the locked decision."""
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "research/round154"))
from package154 import DIRECTIONS, LIMIT, confidential_literals, dump, scan_release, sha, transport_csv

STUDY = "studies/3eed_two_winner_identity_and_geometry_20261118"
RESULTS = HERE / "results"
LOCAL = ROOT / "crossplatform/round156"
META = "ordinal source target has_B sampling_weight scene_cluster frame_cluster point_content_cluster".split()
ENDPOINTS = "frozen_A R145_pca A_axis A_pca A_full D_axis D_pca D_full A_single_pca".split()
ORACLES = "oracle_R145 oracle_mixed oracle_shared oracle_full oracle_single_frame oracle_axis".split()
AGGS = "n mIoU Acc25 Acc50".split()
IDENTITY = ("case ordinal source target positive_draws negative_draws A_pca D_pca D_minus_A_pp "
            "D_gain_macro_contribution_pp reader1_category reader2_category consensus_category "
            "pair_consensus A_consensus D_consensus").split()
PAIR_SCHEMAS = {
    "pair_rows.csv": META + ENDPOINTS + ORACLES + ["macro_weight", "D_win_gain_pp"],
    "pair_direction.csv": "source target subset aggregation arm n weight_sum mIoU Acc25 Acc50".split(),
    "pair_scene.csv": "source target scene_cluster aggregation arm".split() + AGGS,
    "pair_subset.csv": "aggregation group subset arm".split() + AGGS,
    "pair_contrasts.csv": ("subset aggregation source target arm control n gain_pp better worse unchanged "
                           "rescues25 harms25 rescues50 harms50").split(),
    "pair_overlap_summary.csv": "predicted_pair_3D_IoU n D_better D_worse tie attainable_pp damage_pp".split(),
    "identity_rows.csv": IDENTITY,
}
SYMBOLIC = {"source","target","subset","aggregation","arm","control","group","scope","case","predicted_pair_3D_IoU",
            "reader1_category","reader2_category","consensus_category","pair_consensus","A_consensus","D_consensus",
            "weighting"}


def read(path):
    return json.loads(path.read_bytes())


def read_safe_table(path, columns):
    with path.open(encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != set(columns) or len(columns) != len(reader.fieldnames or []):
            raise ValueError("Unexpected schema for " + path.name)
        rows = list(reader)
    assert rows and all(None not in r and None not in r.values() for r in rows)
    for row in rows:
        for field,value in row.items():
            if field in SYMBOLIC:
                assert re.fullmatch(r"[A-Za-z0-9_]+", value), (path.name, field)
            elif field == "has_B":
                assert value in ("True", "False")
            else:
                assert math.isfinite(float(value)), (path.name, field)
    return rows


def copy_file(path, dest):
    assert path.is_file() and not path.is_symlink()
    dest.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path, dest)


def add_table(build, splits, parent, name, columns, prefix="results"):
    rows = read_safe_table(parent / name, columns)
    if name in ("pair_rows.csv", "all_rows.csv", "center_rows.csv"):
        assert [int(r["ordinal"]) for r in rows] == list(range(1, 1799))
        assert {(r["source"],r["target"]) for r in rows} == set(DIRECTIONS)
    splits.append(transport_csv(build, prefix + "/" + name, rows, columns))
    return rows


def main():
    branch = read(HERE / "DECISION.json")["branch"]
    assert branch == "none" and read(RESULTS / "IDENTITY_SUMMARY.json")["prelocked_branch"] == "UNRESOLVED"
    dest = HERE / "publication/public_export_attempt001"
    if dest.exists():
        raise FileExistsError(dest)
    pred = read(LOCAL / "PAIR_SEAL.json")
    audit = read(RESULTS / "PAIR_SCORE_AUDIT.json")
    assert pred["status"] == audit["status"] == "PASS" and pred["n"] == audit["n"] == 1798
    assert not pred["target_GT_read"] and not pred["source_training"]
    assert pred["protected_frames_opened"] == audit["protected_frames_opened"] == 0
    assert pred["prediction_sha256"] == audit["prediction_sha256"] == sha(LOCAL / "PAIR_PRIVATE.npz")
    assert pred["design_sha256"] == audit["design_sha256"] == sha(HERE / "experiment/DESIGN_LOCK.json")
    assert pred["code_sha256"] == sha(HERE / "pair156.py") and audit["score_code_sha256"] == sha(HERE / "score156.py")
    pred_keys = ("status n prediction_sha256 design_sha256 code_sha256 shared_sample_hash_replays "
                 "baseline_PCA_bit_exact second_winner_exact protected_frames_opened new_network_forwards "
                 "source_training target_GT_read seconds").split()
    dest.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=".build_", dir=dest.parent))
    owns = ["pair156.py", "score156.py", "contributions156.py", "overlap156.py", "visual156.py",
            "identity156.py", "geometric_attribution156.py", "package156.py"]
    splits = []
    try:
        dump(build / "PREDICTION_AUDIT.json", {k:pred[k] for k in pred_keys})
        copy_file(HERE / "experiment/DESIGN_LOCK.json", build / "config/DESIGN_LOCK.json")
        copied = {}
        for name, cols in PAIR_SCHEMAS.items():
            copied[name] = add_table(build, splits, RESULTS, name, cols)
        ids = copied["identity_rows.csv"]
        assert len({r["case"] for r in ids}) == len(ids) and len({r["ordinal"] for r in ids}) == len(ids)
        assert sum(int(r["positive_draws"]) for r in ids)==36 and sum(int(r["negative_draws"]) for r in ids)==6
        for r in ids:
            assert re.fullmatch(r"V[0-9]{2}",r["case"])
            pair = copied["pair_rows.csv"][int(r["ordinal"])-1]
            assert (r["source"],r["target"]) == (pair["source"],pair["target"])
            assert all(abs(float(r[k])-float(pair[k]))<1e-10 for k in ("A_pca","D_pca"))
        for name in ("pair_summary.json","PAIR_SCORE_AUDIT.json","attainable_contributions.json","IDENTITY_SUMMARY.json","geometric_attribution.json"):
            copy_file(RESULTS/name, build / "results" / name)
        for name in owns:
            copy_file(HERE/name, build/"own_code/research/round156"/name)
        # The interpretive account is the report. Only the categorical decision enters the release.
        dump(build/"DECISION.json", {"branch":branch})
        copy_file(HERE/"REPORT_ZH.md", build/"REPORT_ZH.md")
        (build/"REPORT.md").write_text("# R156 two-sampling winners under common yaw refinement\n\n[完整报告与结果](REPORT_ZH.md)\n")
        dump(build/"TRANSPORT_SPLITS.json", splits)
        literals = confidential_literals()
        dump(build/"PUBLIC_EXPORT_AUDIT.json",dict(status="PASS", source_files=len(owns),
             maximum_bytes_exclusive=LIMIT, private_literals_checked=len(literals), branch=branch,
             all_ordinary_records=1798, all_visual_review_rows=len(ids),
             photos_captions_raw_boxes_source_ids_and_private_reader_notes_exported=False))
        entries = [dict(path=p.relative_to(build).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
                   for p in sorted(build.rglob("*")) if p.is_file()]
        dump(build/"MANIFEST.json", {"prefix":STUDY,"files":entries})
        scan_release(build,literals)
        os.rename(build,dest)
    except BaseException:
        shutil.rmtree(build, ignore_errors=True)
        raise
    print("R156_PUBLIC_EXPORT_PASS",len(entries)+1,sha(dest/"MANIFEST.json"),flush=True)


if __name__ == "__main__":
    main()

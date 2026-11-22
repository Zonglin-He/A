"""Recount paired results from the scalar rows without loading labels or geometry."""

from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
sys.path.insert(0, str(HERE.parent / "3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import EXPECTED, check_comparison, check_metrics, read, sha


def table(name):
    opener = gzip.open if name.endswith(".gz") else open
    with opener(OUT / name, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def run():
    seal_path = OUT / "prediction_seal.json"
    seal = read(seal_path)
    evaluation = read(OUT / "evaluation_receipt.json")
    assert evaluation["prediction_seal"] == sha(seal_path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"] and not evaluation["FINAL_access"]
    assert all(sha(HERE / name) == digest for name, digest in seal["files"].items())
    assert all(sha(OUT / name) == digest for name, digest in evaluation["files"].items())
    rows = table("rows.csv.gz")
    assert Counter(r["target"] for r in rows) == EXPECTED
    assert len({r["uid"] for r in rows}) == len(rows) == 9310
    assert sum(r["active"] == "True" for r in rows) == seal["counts"]["active"]
    assert sum(r["gt_scored"] == "True" for r in rows) == evaluation["ordinary_GT_descriptions"]
    assert max(float(r["full_error_m"]) for r in rows) == seal["all_axis_max_error_m"] <= 1e-8
    groups = defaultdict(list)
    for r in rows:
        for key in (("macro", "all"), ("pooled", "all"), ("platform", r["target"]),
                    ("category", r["category"]), ("recording", r["target"] + "/" + r["recording"]),
                    ("case", r["case"])):
            groups[key].append(r)
        for arm in ("ray_equivalent", "height_only"):
            assert int(r[f"{arm}_selection_changed"]) == 0
            assert 0 <= float(r[f"iou_{arm}"]) <= 1 + 1e-8
            if r[f"changed_{arm}"] == "False":
                assert r[f"iou_{arm}"] == r["iou_baseline"]
            else:
                assert r["active"] == r["gt_scored"] == "True"
        if r["changed_height_only"] == "True":
            assert float(r["height_only_trim_height_m"]) > 0.

    for name, check in (("metrics.csv", check_metrics), ("paired.csv", check_comparison)):
        for r in table(name):
            check(groups[r["group"], r["value"]], dict(r, platform=r["group"]))
    for r in table("ray_checks.csv"):
        group = groups[r["group"], r["value"]]
        arm = r["arm"]
        changed = [x for x in group if x[f"changed_{arm}"] == "True"]
        assert len(changed) == int(r["changed"])
        for key in ("selection_rays", "audit_rays", "protected_points"):
            assert sum(int(x[key]) for x in changed) == int(r[key])
        assert sum(int(x[f"{arm}_audit_changed"]) for x in changed) == int(r["audit_changed"])
        assert int(r["selection_changed"]) == 0
    result = dict(
        descriptions=len(rows), groups_recounted=len(groups), counts=seal["counts"],
        all_axis_max_error_m=seal["all_axis_max_error_m"], selection_entries_preserved=True,
        valid=True, FINAL_access=False,
    )
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result, flush=True)


if __name__ == "__main__":
    run()

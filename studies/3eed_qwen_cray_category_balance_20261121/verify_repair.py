"""Independently verify the released metrics from anonymous scalar outcomes."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
EXPECTED = {"waymo": 3782, "drone": 2792, "quad": 2736}
sys.path.insert(0, str(HERE.parent / "3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def run():
    receipt = json.loads((OUT/"evaluation_receipt.json").read_bytes())
    assert receipt["code"] == hashlib.sha256((HERE/"score_repair.py").read_bytes()).hexdigest()
    assert receipt["seal"] == hashlib.sha256((OUT/"prediction_seal.json").read_bytes()).hexdigest()
    for name, checksum in receipt["files"].items():
        assert hashlib.sha256((OUT/name).read_bytes()).hexdigest() == checksum
    rows = []
    for platform, count in EXPECTED.items():
        with gzip.open(OUT/f"rows_{platform}.csv.gz", "rt", newline="") as f:
            block = list(csv.DictReader(f))
        assert len(block) == count and all(r["target"] == platform for r in block)
        rows += block
    assert len(rows) == len({r["uid"] for r in rows}) == 9310
    assert len({r["recording"] for r in rows}) == 98
    assert len({r["frame"] for r in rows}) == 7221
    groups = defaultdict(list)
    activations, controls = Counter(), Counter()
    for r in rows:
        t, c, recording = r["target"], r["category"], r["recording"]
        group = f"{t}/{recording}"
        use = r["use_ray"] == "True"
        shuffled = r["shuffled_use_ray"] == "True"
        assert use == all(r[k] == "True" for k in ("ray_eligible", "label_2d_vehicle", "label_3d_vehicle"))
        assert not shuffled or r["ray_eligible"] == "True"
        assert r["iou_C_ray_applicable"] == r["iou_C_ray" if use else "iou_H"]
        assert r["iou_C_ray_shuffled"] == r["iou_C_ray" if shuffled else "iou_H"]
        activations[group] += use
        controls[group] += shuffled
        for key in (("pooled", "all"), ("macro", "all"), ("platform", t), ("category", c),
                    ("platform_category", t + "/" + c), ("recording", group)):
            groups[key].append(r)
        if r["full"] == r["strict"] == "True":
            groups["matched_support_category", c].append(r)
    assert activations == controls
    checked = {}
    for name, validate in (("metrics.csv", check_metrics), ("paired.csv", check_comparison)):
        with (OUT/name).open(newline="") as f:
            data = list(csv.DictReader(f))
        for report in data:
            key = report["group"], report["value"]
            validate(groups[key], dict(report, platform=report["group"]))
        checked[name] = len(data)
    with (OUT/"coverage.csv").open(newline="") as f:
        for report in csv.DictReader(f):
            group = groups[report["group"], report["value"]]
            assert int(report["n"]) == len(group)
            for name in report.keys() - {"group", "value", "n"}:
                assert int(report[name]) == sum(r[name] == "True" for r in group)
    result = dict(descriptions=len(rows), recordings=len(activations), files_verified=checked,
                  count_matched_per_recording=True, all_negative_rows_retained=True, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(result, flush=True)


if __name__ == "__main__":
    run()

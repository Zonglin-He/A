"""Recompute metrics and paired counts from the published anonymous rows."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE/"results"
sys.path.insert(0, str(HERE.parent/"3eed_qwen_full_ordinary_validation_20261121"))
from verify_full import check_comparison, check_metrics


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def run():
    receipt = json.loads((OUT/"evaluation_receipt.json").read_bytes())
    seal = json.loads((OUT/"prediction_seal.json").read_bytes())
    assert receipt["code"] == sha(HERE/"evaluate.py") == seal["evaluation_code"]
    assert receipt["seal"] == sha(OUT/"prediction_seal.json")
    assert seal["code"] == sha(HERE/"cray_simple.py") and seal["protocol"] == sha(HERE/"PROTOCOL.json")
    assert not receipt["FINAL_access"] and not seal["FINAL_access"]
    assert not seal["target_labels_read"] and seal["new_Qwen_queries"] == 0
    assert seal["max_coordinate_error_vs_2d_only_historical_box"] <= 5.e-10
    for name,checksum in receipt["files"].items():
        assert sha(OUT/name) == checksum
    rows = []
    for t,n in (("waymo",3782),("drone",2792),("quad",2736)):
        with gzip.open(OUT/f"rows_{t}.csv.gz", "rt", newline="") as f:
            block = list(csv.DictReader(f))
        assert len(block) == n and all(r["target"] == t for r in block)
        rows.extend(block)
    assert len(rows) == len({r["uid"] for r in rows}) == 9310
    assert len({r["frame"] for r in rows}) == 7221
    assert len({r["recording"] for r in rows}) == receipt["recordings"] == 98

    groups, routes = defaultdict(list), Counter()
    additional = 0
    for r in rows:
        t,c,recording = r["target"], r["category"], r["recording"]
        readouts = [r[a+"_readout"] for a in ("simple4", "simple2")]
        for name,mode in zip(("simple4","simple2"), readouts, strict=True):
            assert (int(r[name+"_selection_rays"]) >= 5) == (mode == "C_ray")
            assert (r[name+"_heading"] in ("pca","qwen")) == (mode == "C_ray")
            routes[name,mode] += 1
        if r["vehicle_2d"] == "False":
            assert readouts[0] == readouts[1] and readouts[0] != "C_ray"
            assert r["iou_simple4"] == r["iou_simple2"]
        if r["original_uses_ray"] == "True":
            assert readouts[0] == "C_ray"
        else:
            additional += readouts[0] == "C_ray"
        for key in (("pooled","all"),("macro","all"),("category",c),("platform",t),
                    ("platform_category",t+"/"+c),("recording",t+"/"+recording)):
            groups[key].append(r)
    assert additional == seal["newly_eligible_2d_only"]
    recounts = {}
    for name,check in (("metrics.csv",check_metrics),("paired.csv",check_comparison)):
        with (OUT/name).open(newline="") as f:
            table = list(csv.DictReader(f))
        assert len(table) == 3*len(groups)
        for report in table:
            group = report["group"]
            check(groups[group,report["value"]], dict(report,platform=group))
        recounts[name] = len(table)
    with (OUT/"coverage.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            block = groups[row["group"],row["value"]]
            counts = Counter(r[row["arm"]+"_readout"] for r in block)
            assert int(row["n"]) == len(block)
            for mode in ("C_ray","H","A","missing_3d"):
                assert int(row[mode]) == counts[mode]
    verified = dict(descriptions=len(rows), recordings=98, verified_tables=recounts,
                    all_public_file_hashes_verified=True,
                    route_counts={f"{arm}/{mode}":n for (arm,mode),n in sorted(routes.items())},
                    all_negative_rows_retained=True, FINAL_access=False)
    (OUT/"verification.json").write_text(json.dumps(verified,indent=2)+"\n")
    print(verified, flush=True)


if __name__ == "__main__":
    run()

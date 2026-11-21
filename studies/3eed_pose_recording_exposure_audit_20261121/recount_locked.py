"""Recount the locked three-arm historical comparison without opening labels."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

from audit_recordings import HERE, OUT, STUDIES, digest, read

POSE = STUDIES / "3eed_pose_consistent_visible_surface_grounding_20261121"
SEMANTIC = STUDIES / "3eed_qwen_observed_semantic_size_20261121"
ARMS = {"H": "P", "fixed_pca": "C_pca", "pose_ray": "C_ray"}


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def historical_summary(rows):
    subsets = {
        "all_exposed_416": rows,
        "validation_exposed_288": [r for r in rows if r["panel"] == "validation"],
        "separated_from_initial_96_exposed_224": [r for r in rows if r["separation"] == "new_recording"]
    }
    for scope, data in subsets.items():
        by_platform = defaultdict(list)
        for r in data:
            by_platform[r["target"]].append(r)
        for target in (*sorted(by_platform), "macro"):
            strata = list(by_platform.values()) if target == "macro" else [by_platform[target]]
            group = [r for g in strata for r in g]
            summary = dict(scope=scope, platform=target, frames=len(group),
                           recordings=len({(r["target"],r["recording"]) for r in group}))
            for display, arm in ARMS.items():
                values = [[float(r[arm]) for r in g] for g in strata]
                summary[f"{display}_acc25"] = 100*mean(mean(x>.25 for x in v) for v in values)
                summary[f"{display}_acc50"] = 100*mean(mean(x>.50 for x in v) for v in values)
                summary[f"{display}_miou"] = 100*mean(mean(v) for v in values)
            for reference in ("H", "fixed_pca"):
                pairs = [(float(r[ARMS["pose_ray"]]), float(r[ARMS[reference]])) for r in group]
                for threshold, label in ((.25,25),(.50,50)):
                    summary[f"ray_vs_{reference}_rescue{label}"] = sum(a>threshold>=b for a,b in pairs)
                    summary[f"ray_vs_{reference}_harm{label}"] = sum(b>threshold>=a for a,b in pairs)
            yield summary


def run():
    lock = read(HERE / "CANDIDATE_LOCK.json")
    assert all(digest(POSE / file) == checksum for file,checksum in lock["published_study_sha256"].items())
    old = json.loads(subprocess.check_output([sys.executable, str(POSE/"recount.py")], text=True))
    assert old["frames"] == 416
    seal = read(POSE/"results/prediction_seal.json")
    assert seal["isolation"]["retained_withheld_raw"] == 0 and not seal["FINAL_access"]
    data = table(POSE/"results/rows.csv")
    result = list(historical_summary(data))
    assert len(result) == 12
    path = OUT / "historical_three_arms.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)

    semantic_path = SEMANTIC/"results/rows.csv"
    assert digest(semantic_path) == read(SEMANTIC/"results/evaluation_receipt.json")["files"]["rows"]
    semantic = table(semantic_path)
    admissible = [r for r in semantic if r["semantic_format_S"] == r["semantic_format_GS"] == "valid"]
    changed = [r for r in semantic if r["completed_dimensions_changed_GS_S"] == "True"]
    prior_stats = dict(frames=len(semantic), both_raw_schemas_valid=len(admissible),
        same_when_both_raw_schemas_valid=sum(r["semantic_status"]=="same" for r in admissible),
        completed_boxes_changed=len(changed),
        completed_boxes_changed_in_validation=sum(r["panel"]=="validation" for r in changed),
        S_GS_same_Acc50_success_rows=all((float(r["iou_S"])>.5)==(float(r["iou_GS"])>.5) for r in semantic))
    audit = dict(
        rechecked_old_pose_metrics=old, historical_table_sha256=digest(path), semantic_closure_check=prior_stats,
        candidate_lock_sha256=digest(HERE/"CANDIDATE_LOCK.json"),
        historical_scoring_receipt_sha256=digest(POSE/"results/evaluation_receipt.json"),
        no_new_predictions=True, no_new_target_labels_read=True, FINAL_access=False
    )
    (OUT / "recount_receipt.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(audit, flush=True)


if __name__ == "__main__":
    run()

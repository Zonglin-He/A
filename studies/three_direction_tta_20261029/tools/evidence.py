"""Small, model-independent provenance and paired-result checks.

This tool never runs a model and cannot turn a smoke test into task evidence.
Score input is JSONL: sample_id, cluster_id, arm, score (one declared metric).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seal(output, paths):
    output = Path(output).resolve()
    files = [Path(p).resolve(strict=True) for p in paths]
    if not files or len(set(files)) != len(files):
        raise ValueError("An explicit, nonempty, duplicate-free file list is required")
    if output in files or any(not p.is_file() for p in files):
        raise ValueError("Only regular source/prediction files may be sealed")
    manifest = {
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [{"path": str(p), "bytes": p.stat().st_size,
                   "sha256": digest(p)} for p in sorted(files)],
    }
    # Refuse overwrite, including an older protocol or prediction seal.
    with output.open("x") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return {"manifest": str(output), "sha256": digest(output),
            "file_count": len(files)}


def verify(manifest):
    data = json.loads(Path(manifest).read_text())
    failures = []
    for item in data["files"]:
        p = Path(item["path"])
        if not p.is_file():
            failures.append({"path": str(p), "reason": "missing"})
        elif p.stat().st_size != item["bytes"] or digest(p) != item["sha256"]:
            failures.append({"path": str(p), "reason": "content_changed"})
    return {"valid": not failures, "file_count": len(data["files"]),
            "failures": failures}


def paired(path, source, candidate, higher_is_better, repeats=2000, seed=41):
    arms = {source: {}, candidate: {}}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row["arm"] not in arms:
            continue
        key = row["sample_id"]
        if key in arms[row["arm"]]:
            raise ValueError(f"Duplicate sample/arm: {key}, {row['arm']}")
        if not math.isfinite(row["score"]):
            raise ValueError(f"Nonfinite score for {key}")
        if not row.get("cluster_id"):
            raise ValueError("Explicit independent parent/cluster IDs are required")
        arms[row["arm"]][key] = row
    if not arms[source] or arms[source].keys() != arms[candidate].keys():
        raise ValueError("Arms must have the same nonempty sample set; no silent dropping")
    groups, deltas = {}, []
    sign = 1 if higher_is_better else -1
    for key, src in sorted(arms[source].items()):
        dst = arms[candidate][key]
        if src["cluster_id"] != dst["cluster_id"]:
            raise ValueError("Cluster identity changed across arms")
        delta = sign * (dst["score"] - src["score"])
        groups.setdefault(src["cluster_id"], []).append(delta)
        deltas.append({"sample_id": key, "benefit": delta})
    clusters = list(groups)
    rng = random.Random(seed)
    boot = []
    for _ in range(repeats):
        values = [v for k in rng.choices(clusters, k=len(clusters)) for v in groups[k]]
        boot.append(sum(values) / len(values))
    boot.sort()
    return {
        "source": source, "candidate": candidate, "n_samples": len(deltas),
        "n_independent_clusters": len(clusters), "higher_is_better": higher_is_better,
        "source_mean": sum(x["score"] for x in arms[source].values()) / len(deltas),
        "candidate_mean": sum(x["score"] for x in arms[candidate].values()) / len(deltas),
        "mean_benefit": sum(x["benefit"] for x in deltas) / len(deltas),
        "improved": sum(x["benefit"] > 0 for x in deltas),
        "harmed": sum(x["benefit"] < 0 for x in deltas),
        "unchanged": sum(x["benefit"] == 0 for x in deltas),
        "worst_cases": sorted(deltas, key=lambda x: x["benefit"])[:10],
        "bootstrap_percentile_95": [boot[int(.025 * (repeats - 1))],
                                    boot[int(.975 * (repeats - 1))]],
        "bootstrap_seed": seed, "bootstrap_repeats": repeats,
        "interpretation": "Development evidence only. Percentile intervals may be unstable with few independent clusters. Positive benefit means improvement in the declared metric.",
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seal")
    s.add_argument("output")
    s.add_argument("files", nargs="+")
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    p = sub.add_parser("paired")
    p.add_argument("scores")
    p.add_argument("--source", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--direction", choices=["higher", "lower"], required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(args.output, args.files)
    elif args.command == "verify":
        result = verify(args.manifest)
    else:
        result = paired(args.scores, args.source, args.candidate, args.direction == "higher")
    print(json.dumps(result, indent=2))
    if args.command == "verify" and not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

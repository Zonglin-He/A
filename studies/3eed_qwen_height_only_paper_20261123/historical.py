"""Collect previously published 3EED references on matching ordinary counts."""

from collections import defaultdict
import argparse
import csv
import gzip
from hashlib import sha256
from pathlib import Path
import subprocess

from paper_eval import OUT, PLATFORMS, roster, scorer


COMMIT = "0c68f2c94c0d00cf91305f16df4e0884c97300e0"
PUBLISHED = "studies/3eed_official_cross_platform_validation_ablation_20261118/results/metrics.csv"
DIGEST = "14e52420a616028816220141f99ca6911d95930d0c0fc09f6ae367cb9ca66105"
DIRECT = OUT.parents[1] / "3eed_four_scale_direct_subject_20261120" / "results"
COUNTS = dict(waymo=3782, drone=2792, quad=2736)
ARMS = ("official_bbf_axis", "native_text_axis", "official_bbf_pca", "native_text_pca")


def run(repo):
    official = subprocess.check_output(["git", "-C", str(repo), "show", f"{COMMIT}:{PUBLISHED}"])
    assert sha256(official).hexdigest() == DIGEST
    old = list(csv.DictReader(official.decode().splitlines()))
    rows = []
    for record in old:
        arm = record["arm"]
        if arm not in ARMS or record["aggregation"] != "direction":
            continue
        source, target = record["source"], record["target"]
        assert source != target and int(record["n"]) == COUNTS[target]
        rows.append(dict(arm=arm, source=source, target=target, n=COUNTS[target],
            **{name: float(record[k]) for name, k in zip(scorer.STATS, ("Acc25", "Acc50", "mIoU"), strict=True)}))

    simple = DIRECT / "ordinary_rows.csv.gz"
    verified = roster.read(DIRECT / "verification.json")
    assert roster.sha(simple) == verified["ordinary_rows_sha256"] and not verified["FINAL_access"]
    by_direction = defaultdict(list)
    with gzip.open(simple, "rt") as stream:
        for row in csv.DictReader(stream):
            by_direction[row["source"], row["target"]].append(float(row["separate_direct"]))
    for (source, target), scores in sorted(by_direction.items()):
        n = len(scores)
        assert source != target and n == COUNTS[target]
        rows.append(dict(arm="four_scale_v3", source=source, target=target, n=n,
                         acc25=100 * sum(q > .25 for q in scores)/n,
                         acc50=100 * sum(q > .50 for q in scores)/n, miou=100 * sum(scores)/n))

    means = []
    for arm in (*ARMS, "four_scale_v3"):
        group = [r for r in rows if r["arm"] == arm]
        assert len(group) == 6 and len({(r["source"], r["target"]) for r in group}) == 6
        for target in (*PLATFORMS, "macro"):
            sub = group if target == "macro" else [r for r in group if r["target"] == target]
            n = sum(r["n"] for r in sub)
            means.append(dict(arm=arm, source="two_cross_platform_models", target=target, n=n,
                              **{k: sum(r[k] for r in sub)/len(sub) for k in scorer.STATS}))
        if arm in ARMS:
            expected, = [r for r in old if r["arm"] == arm and r["aggregation"] == "six_direction_macro"]
            for key, source in zip(scorer.STATS, ("Acc25", "Acc50", "mIoU"), strict=True):
                assert abs(means[-1][key] - float(expected[source])) < 1.e-10

    scorer.OUT = OUT
    key = scorer.write_rows("historical_3eed.csv", [*rows, *means])
    roster.write(OUT / "historical_3eed.json", dict(
        published_commit=COMMIT, published_file=PUBLISHED, sha256=DIGEST,
        four_scale_rows_sha256=roster.sha(simple), numeric_table_sha256=key,
        ordinary_target_description_counts=COUNTS, historical_direction_records=18620,
        status="Archived source-trained systems, matched ordinary population counts and oriented IoU conventions",
        paired_with_qwen_per_row=False, FINAL_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True, help="Git repository containing the pinned commit")
    run(parser.parse_args().history)

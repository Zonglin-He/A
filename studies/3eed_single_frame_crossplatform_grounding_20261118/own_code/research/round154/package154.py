"""Release allowlisted first-party R154 code and independently verified anonymous scalars."""
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/round154"
RESULTS = HERE / "results"
REGISTRY = ROOT / "research/round131/INPUT_REGISTRY_PRIVATE.json"
PARENT = ROOT / "research/round145"
STUDY = "studies/3eed_two_sampling_candidate_box_fusion_20261118"
LIMIT = 220_000
BLOCK = 208_000
ARMS = ("frozen_A", "joint_zero", "direct_R145", "R145", "wrong_R145", "alternate_single",
        "geometry_zero", "geometry_TTA", "anchored_score_TTA", "union_representative_TTA",
        "fusion_zero", "fusion_TTA", "shuffled_fusion_TTA")
POOLS = ("original_A", "alternate", "geometry", "fusion_zero", "fusion_TTA")
BASES = dict(zip(ARMS[:5], ("Frozen_A", "norm_joint_zero", "norm_direct", "norm_TTA", "norm_wrong_TTA")))
PLATFORMS = ("waymo", "quad", "drone")
DIRECTIONS = [(a, b) for a in PLATFORMS for b in PLATFORMS if a != b]
META = ("ordinal", "source", "target", "has_B", "sampling_weight", "scene_cluster", "frame_cluster", "point_content_cluster")
METRICS = ("mIoU", "Acc25", "Acc50")
CONTRASTS = [("fusion_TTA", a) for a in ARMS if a != "fusion_TTA"] + [
    ("geometry_TTA", "R145"), ("geometry_zero", "joint_zero"), ("geometry_TTA", "geometry_zero"),
    ("fusion_zero", "joint_zero"), ("anchored_score_TTA", "R145"), ("alternate_single", "frozen_A")]
FIELDS = {
    "direction": ("source", "target", "subset", "aggregation", "arm"),
    "scene": ("source", "target", "scene_cluster", "aggregation", "arm"),
    "summary": ("aggregation", "group", "subset", "arm"),
    "subset": ("aggregation", "group", "subset", "arm"),
    "paired": ("source", "target", "subset", "aggregation", "arm", "control")}
PAIR_VALUES = ("n", "gain_pp", "better", "worse", "unchanged", "rescues25", "harms25", "rescues50", "harms50")


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_bytes())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path, columns):
    if path.suffix == ".json":
        records = read(path)
        require(isinstance(records, list) and all(tuple(r) == columns for r in records), path.name + " schema")
    else:
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            require(tuple(reader.fieldnames or ()) == columns, path.name + " schema")
            records = list(reader)
    require(records and all(None not in r and None not in r.values() for r in records), path.name + " incomplete")
    return records


def whole_cohort(task, oracle):
    entries = read(REGISTRY)
    parent = {(r["source"], r["target"], r["id_PRIVATE"]): r for r in read(PARENT / "ALL_ROWS_PRIVATE.json")}
    require(len(entries) == len(parent) == len(task) == len(oracle) == 1798, "cohort size")
    clusters = defaultdict(dict)
    unobserved = 0
    maximum = 0.
    for index, (entry, obs, pool) in enumerate(zip(entries, task, oracle), start=1):
        src, tgt = entry["source"], entry["target"]
        row = entry["row"]
        ref = parent[src, tgt, row["id"]]
        require(entry["cohort"] == "EXT" and not entry["protected_alias_fallback"], "ordinary-only registry")
        require(all(obs[k] == pool[k] for k in META), "task / oracle metadata differ")
        require(int(obs["ordinal"]) == index and (obs["source"], obs["target"]) == (src, tgt), "cohort replay order")
        require(obs["has_B"] == str(ref["has_B"]) and
                float(obs["sampling_weight"]) == ref["sampling_weight"] == entry["sampling_weight"] and
                math.isfinite(ref["sampling_weight"]) and ref["sampling_weight"] > 0, "availability or weights")
        unobserved += int(not ref["has_B"])
        for key, private_key in (("scene_cluster", row["scene"]), ("frame_cluster", entry["frame_PRIVATE"]),
                                 ("point_content_cluster", entry["content_group_PRIVATE"])):
            aliases = clusters[key]
            alias = aliases.setdefault((tgt, private_key), len(aliases) + 1)
            require(str(alias) == obs[key], "inconsistent anonymous cluster")
        for name, old in BASES.items():
            maximum = max(maximum, abs(float(obs[name]) - ref["IoUs"][old]))
        for name in ARMS:
            require(0 <= float(obs[name]) <= 1, "invalid endpoint IoU")
        for name in POOLS:
            require(0 <= float(pool[name]) <= 1, "invalid diagnostic oracle IoU")
    require(unobserved == 1034 and maximum <= 1e-7, "historical endpoints or B subset")
    return dict(original_arms_checked=len(BASES), original_scalar_values=len(BASES) * len(task),
                maximum_parent_IoU_error=maximum, has_B=len(task) - unobserved, no_B=unobserved,
                anonymous_clusters={k: len(v) for k, v in clusters.items()})


def summary_metrics(values, weights):
    scale = 100. / math.fsum(weights)
    return (len(values), *(sum(weights[:, None] * v) * scale for v in (values, values > .25, values > .50)))


def expected_tables(rows, names, include_pairs):
    matrix = np.array([[float(r[k]) for k in names] for r in rows])
    weights = np.array([float(r["sampling_weight"]) for r in rows])
    has_b = np.array([r["has_B"] == "True" for r in rows])
    scenes = np.array([int(r["scene_cluster"]) for r in rows])
    directions = {d: np.array([(r["source"], r["target"]) == d for r in rows]) for d in DIRECTIONS}
    output = {k: {} for k in FIELDS}
    for subset, select in (("all", np.ones(len(rows), dtype=bool)), ("has_B", has_b), ("no_B", ~has_b)):
        for agg, w in (("raw", np.ones(len(rows))), ("design_weighted", weights)):
            macro = defaultdict(list)
            # Each direction receives equal weight in the six-way combined result.
            balanced = np.zeros(len(rows))
            for (src, tgt), mask in directions.items():
                chosen = mask & select
                require(np.any(chosen), "missing direction for " + subset)
                sums = math.fsum(w[chosen])
                balanced[chosen] = w[chosen] / sums
                count, *stats = summary_metrics(matrix[chosen], w[chosen])
                for i, arm in enumerate(names):
                    metrics = [a[i] for a in stats]
                    output["direction"][src, tgt, subset, agg, arm] = (count, sums, *metrics)
                    for group in (tgt, "six_equal"):
                        macro[group, arm].append((count, *metrics))
                if subset == "all":
                    for scene in np.unique(scenes[chosen]):
                        scene_mask = chosen & (scenes == scene)
                        count_scene, *sm = summary_metrics(matrix[scene_mask], w[scene_mask])
                        for i, arm in enumerate(names):
                            output["scene"][src, tgt, str(scene), agg, arm] = (count_scene, *(a[i] for a in sm))
            for (group, arm), values in macro.items():
                n, *stats = zip(*values)
                key = "summary" if subset == "all" else "subset"
                output[key][agg, group, subset, arm] = (sum(n), *(math.fsum(v) / len(v) for v in stats))
            if not include_pairs:
                continue
            for (src, tgt), mask in list(directions.items()) + [(("all", "all"), np.ones(len(rows), dtype=bool))]:
                chosen = select & mask
                values = matrix[chosen]
                ww = balanced[chosen]
                for arm, control in CONTRASTS:
                    new, old = values[:, names.index(arm)], values[:, names.index(control)]
                    delta = new - old
                    better = np.count_nonzero(delta > 1e-12)
                    worse = np.count_nonzero(delta < -1e-12)
                    crosses = [np.count_nonzero(test) for t in (.25, .5) for test in
                               ((new > t) & (old <= t), (new <= t) & (old > t))]
                    output["paired"][src, tgt, subset, agg, arm, control] = (
                        len(new), 100. * math.fsum(delta * ww) / math.fsum(ww),
                        better, worse, len(new) - better - worse, *crosses)
    return output


def validate_table(path, kind, expected):
    dimensions = FIELDS[kind]
    numeric = (("n", "weight_sum") + METRICS if kind == "direction" else
               PAIR_VALUES if kind == "paired" else ("n",) + METRICS)
    records = load_rows(path, dimensions + numeric)
    seen = set()
    maximum = 0.
    for record in records:
        key = tuple(str(record[n]) for n in dimensions)
        require(key in expected and key not in seen, path.name + " duplicate or unregistered group")
        seen.add(key)
        for name, value in zip(numeric, expected[key]):
            observed = float(record[name])
            require(math.isfinite(observed) and abs(observed - value) < 1e-8, path.name + " mismatch in " + name)
            maximum = max(maximum, abs(observed - value))
    require(seen == set(expected), path.name + " missing rows")
    return records, maximum


def transport_csv(directory, logical, rows, columns):
    """Split between records and verify that joining the UTF-8 CSV parts recovers every field in order."""
    def csv_line(cells):
        buffer = io.StringIO(newline="")
        csv.writer(buffer, lineterminator="\n").writerow(cells)
        return buffer.getvalue().encode("utf-8")
    header = csv_line(columns)
    chunks, chunk = [], bytearray(header)
    for row in rows:
        line = csv_line([row[c] for c in columns])
        require(len(line) + len(header) <= BLOCK, "single oversized CSV row")
        if len(chunk) + len(line) > BLOCK:
            chunks.append(chunk)
            chunk = bytearray(header)
        chunk.extend(line)
    chunks.append(chunk)
    start = 0
    parts = []
    for index, payload in enumerate(chunks, start=1):
        path = (directory / logical if len(chunks) == 1 else
                (directory / logical).with_suffix("") / ("part%03d.csv" % index))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        with path.open(encoding="utf-8", newline="") as f:
            restored = list(csv.DictReader(f))
        for offset, record in enumerate(restored):
            require(all(record[c] == str(rows[start + offset][c]) for c in columns), "split round-trip failed")
        start += len(restored)
        parts.append(dict(path=path.relative_to(directory).as_posix(), rows=len(restored), bytes=len(payload)))
    require(start == len(rows), "split lost rows")
    return dict(logical_path=logical, rows=len(rows), parts=parts)


def public_capture_audit():
    bridge = read(HERE / "CAPTURE_BRIDGE_PRIVATE.json")
    captures = read(ROOT / "crossplatform/round154/captures/SEALED_PRIVATE.json")
    require(bridge["status"] == "PASS" and not bridge["task_GT_read"] and not bridge["FINAL_access"], "bridge")
    rows = []
    for check in bridge["checks"]:
        clean = {k: check[k] for k in ("source", "target", "source_boxes_bit_exact", "original_indices_bit_exact",
                                       "max_probability_error", "same_top1")}
        require((clean["source"], clean["target"]) in DIRECTIONS and clean["source_boxes_bit_exact"]
                and clean["original_indices_bit_exact"] and clean["same_top1"], "bridge comparison")
        rows.append(clean)
    require(sorted(r["source"] for r in rows) == sorted(PLATFORMS), "incomplete bridge")
    require(captures["n"] == len(captures["files"]) == 1798 and not captures["target_GT_read"] and
            not captures["FINAL_access"], "capture scope")
    return dict(bridge_rows=rows, complete_captures=captures["n"], measured_capture_seconds=captures["seconds"],
                target_GT_used_in_prediction=False, final_access=False)


def confidential_literals():
    # All private aliases, timestamps, filenames and hashes remain outside the public schema.
    tokens = set()
    keys = ("id", "caption", "scene", "scene_PRIVATE", "frame_PRIVATE", "content_group_PRIVATE",
            "imagepath", "metapath", "pointpath", "image_sha256", "point_sha256", "pair_PRIVATE")
    for entry in read(PARENT / "INPUT_REGISTRY_PRIVATE.json"):
        records = [entry, entry["row"], entry["A_sensor"]]
        if entry.get("has_B"):
            pair = entry["pair_PRIVATE"]
            records.extend([pair, pair["A"], pair["B"]])
        for ob in records:
            for key in keys:
                v = ob.get(key)
                if isinstance(v, str) and len(v) >= 8:
                    tokens.add(v.encode("utf-8"))
            for key in ("pointpath", "metapath", "imagepath"):
                if ob.get(key):
                    frame_folder = Path(ob[key]).parent.name
                    if len(frame_folder) >= 12:
                        tokens.add(frame_folder.encode("utf-8"))
    return tokens


def scan_release(folder, literals):
    credential = re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})")
    for p in sorted(folder.rglob("*")):
        if p.is_dir():
            continue
        require(p.is_file() and not p.is_symlink() and p.suffix in (".csv", ".py", ".md", ".json"), "file type")
        data = p.read_bytes()
        data.decode("utf-8", errors="strict")
        require(0 < len(data) < LIMIT, "transport cap " + p.name)
        require(not credential.search(data) and not any(t in data for t in literals)
                and os.fsencode(str(Path.home()) + "/") not in data
                and os.fsencode(str(ROOT) + "/") not in data, "private material in " + p.name)
        # Generic references in first-party scripts (e.g. private registry field names) are part of the code.
        require(not re.search(rb"(?<![A-Za-z0-9])/(?:home|mnt|media)/[^\s/]+/", data),
                "absolute machine path in " + p.name)


def artifacts():
    design_file = HERE / "experiment/DESIGN_LOCK.json"
    design = read(design_file)
    require(tuple(design["arm_names"]) == ARMS and tuple(design["pool_names"]) == POOLS and
            not design["FINAL_access"] and not design["new_target_fitting"], "protocol")
    code = read(HERE / "experiment/CODE_LOCK.json")["files"]
    code["audit154.py"] = read(HERE / "experiment/AUDITOR_LOCK.json")["audit154.py"]
    require(all(sha(HERE / name) == value for name, value in code.items()), "locked code changed")
    pred = read(HERE / "PREDICTION_AUDIT.json")
    score = read(RESULTS / "AUDIT.json")
    require(score["status"] == pred["status"] == "PASS" and score["n"] == pred["n"] == 1798 and
            set(pred) == {"status", "n", "directions", "matched_pairs", "arm_names", "pool_names",
                          "max_chosen_box_error", "max_pool_box_error", "seal_sha256", "alternate_seal_sha256",
                          "parent_seal_sha256", "capture_forward_independently_reexecuted", "GT_read",
                          "FINAL_access", "utc"} and
            set(score) == {"status", "n", "seal_sha256", "design_lock_sha256", "score_code_sha256",
                           "all_predictions_verified_before_GT", "ordinary_unique_expressions",
                           "independent_general_OBB_checks", "max_selected_IoU_error",
                           "max_original_pool_IoU_error", "max_parent_endpoint_IoU_error", "strict_thresholds",
                           "cluster_counts", "protected_frames_opened"}, "prediction/scoring audits")
    require(pred["GT_read"] is False and pred["FINAL_access"] is False and score["protected_frames_opened"] == 0
            and tuple(pred["arm_names"]) == ARMS and tuple(pred["pool_names"]) == POOLS and
            score["all_predictions_verified_before_GT"] and score["ordinary_unique_expressions"] == 899 and
            score["strict_thresholds"] == [.25, .5] and score["seal_sha256"] == pred["seal_sha256"] ==
            sha(ROOT / "crossplatform/round154/predictions/SEALED.json") and
            score["design_lock_sha256"] == sha(design_file) and score["score_code_sha256"] == sha(HERE / "score154.py"),
            "audit chain")
    task = load_rows(RESULTS / "task_all_rows.csv", META + ARMS)
    oracle = load_rows(RESULTS / "oracle_only_all_rows.csv", META + POOLS)
    checks = whole_cohort(task, oracle)
    require(checks["anonymous_clusters"] == score["cluster_counts"], "cluster counts")
    tables, summaries = {}, {}
    max_error = 0.
    for family, arms, rows in (("task", ARMS, task), ("oracle_only", POOLS, oracle)):
        expected = expected_tables(rows, arms, include_pairs=(family == "task"))
        tables[family + "_all_rows.csv"] = rows
        for kind in ("direction", "scene", "subset", "summary", "paired"):
            if kind == "paired" and family != "task":
                continue
            filename = ("paired.csv" if kind == "paired" else
                        family + "_" + kind + (".json" if kind == "summary" else ".csv"))
            records, discrepancy = validate_table(RESULTS / filename, kind, expected[kind])
            max_error = max(max_error, discrepancy)
            (summaries if kind == "summary" else tables)[filename] = records
    counts = {k: 0 for k in ("better", "worse")}
    changes = []
    for r in task:
        for arm, ctrl in CONTRASTS:
            now, old = float(r[arm]), float(r[ctrl])
            diff = now - old
            if abs(diff) <= 1e-12:
                continue
            counts["better" if diff > 0 else "worse"] += 1
            changes.append({**{k: r[k] for k in META}, "arm": arm, "control": ctrl,
                            "IoU": now, "control_IoU": old, "delta_IoU": diff})
    paired = [r for r in tables["paired.csv"] if r["source"] == "all" and r["subset"] == "all"
              and r["aggregation"] == "raw"]
    require(all(sum(int(r[k]) for r in paired) == v for k, v in counts.items()), "changed-case completeness")
    tables["all_changed_pairs.csv"] = changes
    checks.update(status="PASS", task_rows=len(task), total_arm_IoUs=len(task) * len(ARMS),
                  oracle_rows=len(oracle), total_pool_IoUs=len(oracle) * len(POOLS), input_is_postseal_scalars=True,
                  max_independent_metric_or_contrast_discrepancy=max_error, all_ordinary_rows_retained=True,
                  improved_record_contrasts=counts["better"], worsened_record_contrasts=counts["worse"])
    return tables, summaries, checks, public_capture_audit(), pred, score


REBUILD = """# Reproduction and table units

The research code under `own_code/research/round154` is the executed first-party implementation. The WBF-inspired rule uses two samples, greedy one-to-one AABB matching and a support-count-sensitive grounding readout. All cases and both raw and design-weighted summaries are included; the six-direction summary gives equal weight to directions. Row IoUs range from 0 to 1. Summary mIoU and accuracies use percent. Paired gains use percentage points. The three thresholds used in comparisons are strictly positive change above 1e-12, IoU > 0.25 and IoU > 0.50. Oracle-pool metrics use ground truth after prediction and measure proposal coverage only.

The public tables are sufficient to reproduce arm and pool aggregates and every paired task contrast. Each part is an independent UTF-8 CSV with a header. Rejoin parts in manifest order without repeating headers. `all_changed_pairs.csv` lists each directional record whose IoU changed for each reported contrast. The same record can occur in several contrasts. The ordinal and three integer cluster columns are release-local aliases. No scene name or recording ID is required for the summaries.

Replaying the actual model requires private paired sensors and calibration, the three licensed source checkpoints, source-prepared matching states, R131/R145 ordinary registries, licensed prefix and inference caches, and compatible earlier first-party modules and the original 3EED runtime. These files are outside this release. Under that environment, prepare the original project layout, then run the experiment guard around capture154.py --check-only, capture154.py --run, predict154.py, audit154.py, score154.py, and describe154.py, in that order. The capture bridge compares the original-seed scores and box/point identity before the second sampling starts. Scoring verifies the prediction seal first. package154.py verifies the released scalars against the retained original and writes the publication bundle. No new target parameter fit was performed. The TTA arm suffix identifies inherited R145 adapted scores. The original design and prior-to-scoring cluster analysis are in config/. The expanded proposal sets, target labels, raw coordinates, weights, features and private source paths are not part of the public release.
"""


def main():
    tables, summaries, checks, bridge, pred, score = artifacts()
    transfer = read(HERE / "TRANSFER_SUMMARY.json")
    transfer_keys = {"paired_mIoU_pp", "scope", "eligible_B", "n", "mean_pairs", "mean_new_singletons_in_final256",
                     "winner_type_count", "winner_type_contribution_vs_R145_pp"}
    require(set(transfer) == transfer_keys and transfer["n"] == 1798 and transfer["eligible_B"] == checks["has_B"] and
            sum(transfer["winner_type_count"].values()) == 1798, "transfer summary structure")
    controls = {"frozen_A", "R145", "fusion_zero", "shuffled_fusion_TTA"}
    require(set(transfer["paired_mIoU_pp"]) == controls, "transfer contrasts")
    for r in tables["paired.csv"]:
        if (r["arm"] == "fusion_TTA" and r["source"] == "all" and r["aggregation"] == "design_weighted" and
                r["subset"] == "all" and r["control"] in controls):
            require(abs(float(r["gain_pp"]) - transfer["paired_mIoU_pp"][r["control"]]["point"]) < 1e-8,
                    "transfer-estimate mismatch")
    publication = HERE / "publication"
    publication.mkdir(exist_ok=True)
    destination = publication / "public_export_attempt001"
    require(not destination.exists(), "export exists")
    report = (HERE / "REPORT_ZH.md").read_bytes()
    require(bool(report.strip()), "missing report")
    output = Path(tempfile.mkdtemp(prefix=".building_", dir=publication))
    splits = []
    for filename, records in tables.items():
        columns = tuple(records[0]) if records else META + ("arm", "control", "IoU", "control_IoU", "delta_IoU")
        splits.append(transport_csv(output, "results/" + filename, records, columns))
    for name, summary in summaries.items():
        dump(output / "results" / name, summary)
    for name in ("DESIGN_LOCK.json", "CODE_LOCK.json", "AUDITOR_LOCK.json", "CLUSTER_ANALYSIS.json", "POOL_DECOMPOSITION.json"):
        (output / "config").mkdir(exist_ok=True)
        (output / "config" / name).write_bytes((HERE / "experiment" / name).read_bytes())
    own = [HERE / n for n in ("capture154.py", "predict154.py", "audit154.py", "score154.py",
                               "describe154.py", "package154.py")] + [ROOT / "tools/experiment_guard_round154.py"]
    for p in own:
        target = output / "own_code" / p.relative_to(ROOT)
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(p.read_bytes())
    (output / "REPORT_ZH.md").write_bytes(report)
    (output / "REPORT.md").write_text("# R154 candidate-box fusion\n\n[完整实验报告与结果](REPORT_ZH.md)\n",
                                      encoding="utf-8")
    (output / "REPRODUCTION.md").write_text(REBUILD, encoding="utf-8")
    for name, summary in (("SOURCE_BRIDGE_AND_CAPTURE", bridge), ("PREDICTION_AUDIT", pred),
                          ("SCORE_AUDIT", score), ("SCALAR_AUDIT", checks), ("TRANSFER_SUMMARY", transfer)):
        dump(output / (name + ".json"), summary)
    dump(output / "TRANSPORT_SPLITS.json", {"tables": splits, "record_order_retained": True,
                                            "maximum_file_bytes_exclusive": LIMIT})
    literals = confidential_literals()
    privacy = dict(status="PASS", private_literals_crosschecked=len(literals), allowed_types=["csv", "json", "md", "py"],
                   maximum_file_bytes_exclusive=LIMIT, raw_media_boxes_ids_features_weights_secrets_exported=False,
                   allowlisted_first_party_sources=len(own), destination_prefix=STUDY)
    dump(output / "PUBLIC_EXPORT_AUDIT.json", privacy)
    files = [dict(path=p.relative_to(output).as_posix(), bytes=p.stat().st_size, sha256=sha(p))
             for p in sorted(output.rglob("*")) if p.is_file()]
    dump(output / "MANIFEST.json", dict(prefix=STUDY, files=files))
    scan_release(output, literals)
    os.rename(output, destination)
    dump(HERE / "PUBLIC_EXPORT_AUDIT.json", privacy | {"manifest_sha256": sha(destination / "MANIFEST.json"),
                                                      "manifest_content_files": len(files)})
    print("PASS_R154_PUBLIC_EXPORT", len(files), sha(destination / "MANIFEST.json"), flush=True)


if __name__ == "__main__":
    main()

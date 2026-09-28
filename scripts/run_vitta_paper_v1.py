"""Finite ViTTA-episodic STVG source statistics and evaluation; no automatic queue.

prepare is CPU-only. source/run require an exclusive GPU lease. Each invocation
is bounded and preserves partial receipts. Scoring is a separate offline action.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, status, save, load, sha
from vg_tta.vitta_paper_v1 import DEFAULTS, FEATURE_LAYERS, OFFICIAL_COMMIT, REDUCTION

OUT = ROOT / "artifacts/decota_paper_execution_20260917/vitta"
PARENT = ROOT / "artifacts/decota_final_simplification_v1"
MEDIA_LOCK = ROOT / "artifacts/candidate_utility_p2_v1/lock.json"
CODE = ["vg_tta/vitta_paper_v1.py", "scripts/run_vitta_paper_v1.py", "tests/test_vitta_paper_v1.py",
        "vg_tta/native_baselines_paper_v1.py", "vg_tta/native_probability_interface_v1.py",
        "vg_tta/exact_frame_decode_audit_v2.py", "vg_tta/unanchored_dense_shift_data_v1.py",
        "scripts/evaluate_fullspan_scale_shift_corruptions_v1.py"]


def prepare(direction):
    """Lock available official train media independently of any target scores."""
    from methods.decota_final_simplified_v1.config import MethodConfig
    from methods.decota_final_simplified_v1.release import verify_release
    from vg_tta.unanchored_dense_shift_data_v1 import sampled_ids
    verify_release()
    dest = OUT / direction
    if (dest / "LOCK.json").exists():
        return verify(direction)
    cfg = MethodConfig.for_direction(direction)
    if sha(ROOT / cfg.checkpoint) != cfg.checkpoint_sha256:
        raise RuntimeError("Checkpoint differs from the declared source identity")
    parent = read(PARENT / "LOCK.json")
    old = read(MEDIA_LOCK)
    group = "vid_to_hc" if direction == "vid_to_hc1" else "hc_to_vid"
    cohort = "hcstvg1_test" if direction == "vid_to_hc1" else "vidstg_test"
    ann = ROOT / "data/attribute_tta_official_release" / ("vidstg" if cfg.source_dataset == "vidstg" else "hc-stvg2") / "annos/train.json"
    annotations = read(ann)
    if cfg.source_dataset == "vidstg":
        membership = {(str(z["vid"]), z["sentence"]["description"].lower()): k for k, z in annotations.items()}
    else:
        membership = {(k, z["English"].lower()): k for k, z in annotations.items()}
    pool = old["groups"][group]["queries"]
    # The entire existing pool is an independently prepared source-training
    # roster. A fresh stable hash fixes 64 sources; historical ranker outputs,
    # labels, source holdout scores and target scores are never consumed here.
    selected = sorted(pool, key=lambda r: hashlib.sha256(("vitta-source-v1:" + r["input"]["source"]).encode()).hexdigest())[:64]
    if len(selected) != 64 or len({r["input"]["source"] for r in selected}) != 64:
        raise RuntimeError("Require 64 existing distinct official source-training videos")
    targets = {r["input"]["source"] for r in parent["full_rows"][cohort]}
    records = []
    for ordinal, row in enumerate(selected):
        q = {k: v for k, v in row["input"].items() if k != "video_id"}
        identity = str(q["source"]) if cfg.source_dataset == "vidstg" else Path(q["video_path"]).name
        key = (identity, q["caption"].lower())
        # The old ranker has an internal source train/validation split. Both
        # roles are official source TRAIN; prove that with annotation membership.
        if key not in membership or q["source"] in targets:
            raise RuntimeError("Source membership/query/target isolation mismatch")
        if sha(q["video_path"]) != q["video_sha256"]:
            raise RuntimeError("Existing source media hash changed")
        if q["frame_ids"] != sampled_ids(q, q["kind"]):
            raise RuntimeError("Existing source metadata-only sampling changed")
        records.append(dict(ordinal=ordinal, key=f"{cfg.source_dataset}:train:{membership[key]}",
                            official_annotation_key=membership[key],
                            historical_source_ranker_role=row["split"], input=q))
    del annotations, membership
    source_manifest = dict(source_dataset=cfg.source_dataset, official_split="train", source_count=64,
                           query_count=64, records=records, annotation=str(ann), annotation_sha256=sha(ann),
                           source_media_parent=str(MEDIA_LOCK), source_media_parent_sha256=sha(MEDIA_LOCK),
                           available_pool_sources=len(pool), selection="sha256(vitta-source-v1:source) first64",
                           target_source_overlap=0, target_inputs_used=False, localization_labels_used=False,
                           source_queries_and_clip_metadata_used=True, raw_pixels_redecoded_at_collection=True,
                           historical_use="Source media previously used by an unrelated source-supervised ranker; no ranker outputs, source scores, or labels reused.",
                           sampling="existing metadata-only 5fps cap200 grid, two interleaved original offsets")
    write(dest / "SOURCE_MANIFEST.json", source_manifest)
    dev = [r for r in parent["rows"][cohort] if r["f44_role"] == "development"][:8]
    if len(dev) != 8 or len({r["input"]["source"] for r in dev}) != 8:
        raise RuntimeError("Expected eight locked development sources")
    pins = {str(p): sha(p) for p in [PARENT / "LOCK.json", PARENT / "FINAL_CONFIG.json", ann, MEDIA_LOCK,
                                      dest / "SOURCE_MANIFEST.json", ROOT / cfg.checkpoint,
                                      ROOT / "methods/CURRENT_METHOD.json", ROOT / "methods/CURRENT_WORKING_METHOD.json"]}
    write(dest / "LOCK.json", dict(direction=direction, cohort=cohort, source_dataset=cfg.source_dataset,
          checkpoint=cfg.checkpoint, checkpoint_sha256=cfg.checkpoint_sha256, config=DEFAULTS,
          feature_layers=list(FEATURE_LAYERS), statistic_reduction=REDUCTION, source_count=64,
          source_manifest=str(dest / "SOURCE_MANIFEST.json"), source_manifest_sha256=sha(dest / "SOURCE_MANIFEST.json"),
          official_commit=OFFICIAL_COMMIT, dev=dev, full=parent["full_rows"][cohort],
          code_pins={str(ROOT / n): sha(ROOT / n) for n in CODE}, pins=pins,
          labels=parent["labels"], labels_sha256=parent["labels_sha256"],
          created=time.time(), GT_online=False, target_selection=False, recurring=False,
          parameter_scope="full_query_decoder_and_output_heads", published_original="continual_full_network",
          episodic_reference="official tta_standard branch; momentum_mvg=1; reset weights and optimizer",
          full_requires_dev_barrier=True, finite_invocation_hours=48))
    print("PREPARED", direction, "source64; dev8; full", len(parent["full_rows"][cohort]), flush=True)
    return verify(direction)


def verify(direction):
    from methods.decota_final_simplified_v1.release import verify_release
    verify_release()
    lock = read(OUT / direction / "LOCK.json")
    for path, digest in {**lock["pins"], **lock["code_pins"]}.items():
        if sha(path) != digest:
            raise RuntimeError("Pinned ViTTA input/code changed: " + path)
    if sha(ROOT / lock["checkpoint"]) != lock["checkpoint_sha256"]:
        raise RuntimeError("Pinned weights differ from source-statistic provenance")
    return lock


def model_load(lock):
    import numpy as np
    import torch
    from methods.decota_final_simplified_v1._tastvg_load import load_model_on_device
    from methods.decota_final_simplified_v1.observations import QuerySubjectParser
    torch.set_num_threads(4)
    torch.manual_seed(20260917)
    np.random.seed(20260917)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    runtime = ROOT / "artifacts/tastvg_runtime" / ("vidstg" if lock["source_dataset"] == "vidstg" else "hc-stvg2")
    model, _, _ = load_model_on_device(lock["source_dataset"], runtime,
                                     checkpoint=ROOT / lock["checkpoint"], device="cuda",
                                     source_dataset=lock["source_dataset"])
    return model.eval().requires_grad_(False), QuerySubjectParser(ROOT / ".cache/stanza")


def lease():
    from scripts.run_final_simplification_v1 import lease as common_lease
    guard = common_lease()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    if active:
        guard.close()
        raise RuntimeError("GPU is not exclusive: " + active)
    return guard


def get_receipt(path, lock_hash):
    receipt = path.with_suffix(".json")
    if receipt.exists():
        result = read(receipt)
        if result["lock_sha256"] != lock_hash or sha(path) != result["sha256"]:
            raise RuntimeError("ViTTA receipt/content mismatch")
        return result
    if path.exists():
        raise RuntimeError("Unreceipted output preserved: " + str(path))
    return None


def receipt_save(path, value, lock_hash, key):
    save(path, value)
    receipt = dict(path=str(path), key=key, sha256=sha(path), lock_sha256=lock_hash, completed=time.time())
    write(path.with_suffix(".json"), receipt)
    return receipt


def collect_source(direction, limit=0):
    import torch
    from methods.decota_final_simplified_v1.backbone import make_batch, validate_input
    from methods.decota_final_simplified_v1.tensors import state_hash
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.vitta_paper_v1 import SourceStatistics, feature_statistics, live_views, validate_source
    lock = verify(direction)
    dest = OUT / direction
    manifest = read(lock["source_manifest"])
    lock_hash = sha(dest / "LOCK.json")
    if (dest / "source/COMPLETION.json").exists():
        barrier = read(dest / "source/COMPLETION.json")
        if barrier["lock_sha256"] != lock_hash or sha(dest / "source/statistics.pt") != barrier["statistics_sha256"]:
            raise RuntimeError("Source completion hash mismatch")
        return
    with lease():
        model, parser = model_load(lock)
        before = state_hash(model.state_dict())
        acc, receipts, done = SourceStatistics(), [], 0
        started = time.time()
        for row in manifest["records"]:
            if time.time() - started > lock["finite_invocation_hours"] * 3600:
                raise TimeoutError("Finite source-statistics budget")
            path = dest / "source/episodes" / f'{row["ordinal"]:04d}.pt'
            receipt = get_receipt(path, lock_hash)
            if receipt:
                item = load(path)
            else:
                q = row["input"]
                frames, ids = decode(q)
                validate_input(frames, ids, q)
                batch = make_batch(frames, ids, q, model)
                subject = parser(q["caption"])["subject"]
                tick = time.perf_counter()
                with torch.no_grad():
                    views, _ = live_views(model, batch, ids, subject)
                    stats = {n: {k: v.cpu() for k, v in z.items()} for n, z in feature_statistics(views).items()}
                torch.cuda.synchronize()
                item = dict(key=row["key"], source=q["source"], input=q, statistics=stats,
                            forward_seconds=time.perf_counter() - tick, source_state_hash=before,
                            localization_labels_used=False, target_inputs_used=False, lock_sha256=lock_hash)
                receipt = receipt_save(path, item, lock_hash, row["key"])
                done += 1
                del views, frames, batch
                gc.collect()
                torch.cuda.empty_cache()
            acc.add(item["statistics"])
            receipts.append(receipt)
            status(dest / "source/STATUS.json", dict(status="collecting", done=len(receipts), total=64, updated=time.time()))
            print("SOURCE", direction, len(receipts), "/64", row["key"], flush=True)
            if limit and done >= limit:
                break
        if before != state_hash(model.state_dict()):
            raise RuntimeError("Source-stat collection modified the source model")
        if len(receipts) == 64:
            source = dict(statistics=acc.finish(), provenance=dict(
                source_dataset=lock["source_dataset"], checkpoint_sha256=lock["checkpoint_sha256"],
                source_state_hash=before, source_manifest_sha256=lock["source_manifest_sha256"],
                official_split="train", source_count=64, query_count=64, target_inputs_used=False,
                localization_labels_used=False, reduction=REDUCTION))
            validate_source(source, checkpoint_sha256=lock["checkpoint_sha256"], layer_names=FEATURE_LAYERS)
            save(dest / "source/statistics.pt", source)
            write(dest / "source/COMPLETION.json", dict(status="complete", receipts=receipts,
                  queries=64, sources=64, lock_sha256=lock_hash, source_state_exact=True,
                  statistics_sha256=sha(dest / "source/statistics.pt"), seconds=time.time() - started,
                  target_inference_completed=False))


def run(direction, phase, limit=0):
    import torch
    from vg_tta.exact_frame_decode_audit_v2 import decode
    from vg_tta.vitta_paper_v1 import episode
    from scripts.run_native_baselines_paper_v1 import parity
    lock = verify(direction)
    dest = OUT / direction
    lock_hash = sha(dest / "LOCK.json")
    source_done = read(dest / "source/COMPLETION.json")
    if source_done["lock_sha256"] != lock_hash or sha(dest / "source/statistics.pt") != source_done["statistics_sha256"]:
        raise RuntimeError("Source statistic identity mismatch")
    source = load(dest / "source/statistics.pt")
    if phase == "full":
        barrier = read(dest / "dev/BARRIER.json")
        if not barrier["functional_validation_passed"] or barrier["lock_sha256"] != lock_hash:
            raise RuntimeError("Full evaluation needs a valid development barrier")
    receipts, done, started = [], 0, time.time()
    with lease():
        model, parser = model_load(lock)
        for at, row in enumerate(lock[phase]):
            if time.time() - started > lock["finite_invocation_hours"] * 3600:
                raise TimeoutError("Finite ViTTA invocation budget")
            path = dest / phase / f'{row["ordinal"]:06d}.pt'
            receipt = get_receipt(path, lock_hash)
            if receipt:
                receipts.append(receipt)
                continue
            q = row["input"]
            x = dict(key=row["key"], source=q["source"], input=q, lock_sha256=lock_hash, GT_online=False)
            if row.get("input_unavailable"):
                x.update(status="known_input_unavailable", reason=row["input_unavailable"])
            else:
                parent_path = PARENT / "full" / lock["cohort"] / path.name
                parent_receipt = read(parent_path.with_suffix(".json"))
                if sha(parent_path) != parent_receipt["sha256"]:
                    raise RuntimeError("Parent prediction hash mismatch")
                parent = load(parent_path)
                if parent["input"] != q:
                    raise RuntimeError("Parent input differs")
                frames, ids = decode(q)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                result = episode(model, parser, frames, ids, q, source, lock["checkpoint_sha256"],
                                 lock["source_dataset"], lock["config"])
                torch.cuda.synchronize()
                result.update(seconds_with_frozen_parity=time.perf_counter() - tick,
                              peak_memory_allocated=torch.cuda.max_memory_allocated(),
                              peak_memory_reserved=torch.cuda.max_memory_reserved())
                parity(result["Frozen"], parent["predictions"]["Frozen"])
                if not torch.isfinite(result["prediction"]["boxes"]).all() or not result["audit"]["source_restored"]:
                    raise RuntimeError("Nonfinite ViTTA output or source restoration failure")
                if result["audit"]["backwards"] != lock["config"]["steps"] or max(result["audit"]["gradient_norms"]) <= 0:
                    raise RuntimeError("ViTTA gradient/update validation failed")
                controls = {}
                if phase == "dev" and at == 0:
                    for name, override in [("steps0", dict(steps=0)), ("lr0", dict(lr=0.))]:
                        ctrl = episode(model, parser, frames, ids, q, source, lock["checkpoint_sha256"],
                                       lock["source_dataset"], {**lock["config"], **override})
                        parity(ctrl["prediction"], result["Frozen"])
                        if ctrl["audit"]["parameter_changed"]:
                            raise RuntimeError("No-op control changed source parameters")
                        controls[name] = ctrl["audit"]
                x.update(status="available", result=result, controls=controls, parent_receipt=parent_receipt)
                del frames, parent
            receipts.append(receipt_save(path, x, lock_hash, row["key"]))
            done += 1
            status(dest / phase / "STATUS.json", dict(status="running", done=len(receipts), total=len(lock[phase]), updated=time.time()))
            print(phase, direction, len(receipts), "/", len(lock[phase]), row["key"], x["status"], flush=True)
            del x
            gc.collect()
            torch.cuda.empty_cache()
            if limit and done >= limit:
                break
    if len(receipts) == len(lock[phase]) and not (dest / phase / "BARRIER.json").exists():
        write(dest / phase / "BARRIER.json", dict(complete=True, receipts=receipts, queries=len(receipts),
              functional_validation_passed=True, selection_by_GT=False, lock_sha256=lock_hash))


def score(direction, phase):
    from scripts.analyze_spatial10_components_v1 import checked_score
    from scripts.score_decota_final_freeze_v1 import aggregate
    lock = verify(direction)
    dest = OUT / direction
    barrier = read(dest / phase / "BARRIER.json")
    lock_hash = sha(dest / "LOCK.json")
    if barrier["lock_sha256"] != lock_hash or sha(lock["labels"]) != lock["labels_sha256"]:
        raise RuntimeError("Offline score manifest mismatch")
    labels, rows, missing, audits = read(lock["labels"]), [], [], []
    for receipt in barrier["receipts"]:
        if get_receipt(Path(receipt["path"]), lock_hash) != receipt:
            raise RuntimeError("Prediction receipt mismatch")
        x = load(receipt["path"])
        if x["status"] != "available":
            missing.append(dict(key=x["key"], reason=x["reason"]))
            continue
        if sha(x["parent_receipt"]["path"]) != x["parent_receipt"]["sha256"]:
            raise RuntimeError("Parent comparison prediction changed before scoring")
        parent = load(x["parent_receipt"]["path"])
        predictions = dict(Frozen=x["result"]["Frozen"], ViTTA_episodic=x["result"]["prediction"],
                           DeCoTA=parent["predictions"]["Full_DeCoTA"])
        values = {name: checked_score(p["boxes"], labels[x["key"]], x["input"]["frame_ids"], p["indices"])[0]
                  for name, p in predictions.items()}
        rows.append(dict(key=x["key"], source=x["source"], arms=values))
        audits.append(dict(key=x["key"], audit=x["result"]["audit"], controls=x["controls"]))
    sources = [r["source"] for r in rows]
    metrics = ["vIoU_corrected", "sIoU", "tIoU"]
    names = ["Frozen", "ViTTA_episodic", "DeCoTA"]
    arms = {a: {m: aggregate([r["arms"][a][m] for r in rows], sources) for m in metrics} for a in names}
    paired = {a + " - Frozen": {m: aggregate([r["arms"][a][m] - r["arms"]["Frozen"][m] for r in rows], sources)
                                    for m in metrics} for a in names[1:]}
    analysis = dest / phase / "analysis"
    write(analysis / "ALL_QUERY_RESULTS.json", dict(rows=rows, missing=missing))
    write(analysis / "ALL_SOURCE_RESULTS.json", dict(queries=len(rows), sources=len(set(sources)), arms=arms, paired=paired))
    write(analysis / "AUDIT.json", dict(rows=audits, missing=missing, GT_online=False, historical_target_exposure=True,
                                        original_continual_reproduced=False, source_statistics_queries=64))
    write(analysis / "COMPLETION.json", dict(status="scored_complete", phase=phase, direction=direction,
          files={str(f): sha(f) for f in analysis.iterdir() if f.is_file()}))
    print("SCORED", direction, phase, len(rows), "queries", len(set(sources)), "sources", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "verify", "source", "run", "score"])
    parser.add_argument("--direction", required=True, choices=["vid_to_hc1", "hc2_to_vid"])
    parser.add_argument("--phase", choices=["dev", "full"], default="dev")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.action == "prepare":
        prepare(args.direction)
    elif args.action == "verify":
        verify(args.direction)
        print("VERIFIED", args.direction)
    elif args.action == "source":
        collect_source(args.direction, args.limit)
    elif args.action == "run":
        run(args.direction, args.phase, args.limit)
    else:
        score(args.direction, args.phase)

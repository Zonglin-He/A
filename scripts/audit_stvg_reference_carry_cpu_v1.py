"""Independent CPU readback for saved LN-carry streams; never runs inference."""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import stvg_reference_recovery_common_v1 as common
from scripts import run_stvg_reference_carry_v1 as worker
from scripts.run_stvg_table1_student004 import stable_fit_key
from scripts.stvg_reference_recovery_fit_v1 import locked_config
from methods.decota_final_simplified_v1.tensors import state_hash


def _load(path: Path) -> dict[str, Any]:
    value = common.read(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.relative_to(ROOT)}")
    return value


def _same_state(left: Mapping[str, Any], right: Mapping[str, Any], torch) -> bool:
    if set(left) != set(right):
        return False
    for name in left:
        a, b = left[name], right[name]
        if (not torch.is_tensor(a) or not torch.is_tensor(b) or a.dtype != b.dtype
                or tuple(a.shape) != tuple(b.shape)
                or not torch.equal(a.detach().cpu(), b.detach().cpu())):
            return False
    return True


def _assert_state(state: Mapping[str, Any], torch, label: str) -> tuple[str, ...]:
    if not isinstance(state, Mapping):
        raise ValueError(f"{label} is not a state mapping")
    query = "spatial.query_residual"
    if query not in state or len(state) != 7:
        raise ValueError(f"{label} must contain one Query tensor and six LN tensors")
    ln = [name for name in state if name != query]
    if len(ln) != 6 or sum(state[name].numel() for name in ln) != 1536:
        raise ValueError(f"{label} LN coordinates are not the locked six-by-256 schema")
    for name, value in state.items():
        if not torch.is_tensor(value) or tuple(value.shape) != (256,) or not torch.isfinite(value).all():
            raise ValueError(f"{label} has a nonfinite or unexpected tensor at {name}")
    return tuple(ln)


def _fresh_adam_source_check() -> tuple[str, str]:
    path = ROOT / "scripts/stvg_reference_recovery_fit_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next((node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name == "fit_reference_recovery"), None)
    if function is None:
        raise ValueError("frozen fitter no longer defines fit_reference_recovery")
    calls = []
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "Adam"
                    and isinstance(func.value, ast.Attribute) and func.value.attr == "optim"
                    and isinstance(func.value.value, ast.Name) and func.value.value.id == "torch"):
                calls.append(node)
    if len(calls) != 1:
        raise ValueError("fit_reference_recovery must construct exactly one fresh torch.optim.Adam")
    return str(path.relative_to(ROOT)), common.sha(path)


def _one_stream(dataset: str, order: str, *, require_complete: bool, torch, np,
                roster_index, reference_index, parent_files, orders, runtime_sha,
                fitter_source_sha: str, max_items: int | None = None) -> dict[str, Any]:
    key = (dataset, order)
    if key not in orders:
        raise ValueError(f"missing locked order {dataset}/{order}")
    full_ordered_ids = orders[key]
    ordered_ids = full_ordered_ids
    if max_items is not None:
        if isinstance(max_items, bool) or not 1 <= int(max_items) <= 16:
            raise ValueError("max_items must be an integer from 1 through 16")
        if require_complete and int(max_items) != 16:
            raise ValueError("complete stream audit cannot truncate the ordered stream")
        ordered_ids = ordered_ids[:int(max_items)]
    stream_dir = worker.BASE / "student" / dataset / order
    barrier_path = stream_dir / "STREAM_BARRIER.json"
    present = []
    gap_seen = False
    for cid in ordered_ids:
        path, receipt = worker._stream_paths(dataset, order, cid)
        has_any = path.exists() or receipt.exists()
        if gap_seen and has_any:
            raise ValueError(f"non-prefix stream artifact exists after a missing item: {cid}")
        if not has_any:
            gap_seen = True
            continue
        if not path.is_file() or not receipt.is_file():
            raise ValueError(f"partial saved item; refusing to treat as a sealed prefix: {cid}")
        present.append((cid, path, receipt))
    if not present:
        raise ValueError(f"no saved items found in {dataset}/{order}")
    if require_complete and len(present) != 16:
        raise ValueError(f"{dataset}/{order} is not complete (found {len(present)}/16)")
    if barrier_path.exists():
        barrier = _load(barrier_path)
        if (barrier.get("schema") != "stvg-reference-carry-stream-barrier-v1"
                or barrier.get("status") != "complete" or barrier.get("GT_read") is not False
                or barrier.get("dataset") != dataset or barrier.get("order") != order
                or barrier.get("case_ids") != full_ordered_ids or int(barrier.get("count", -1)) != 16):
            raise ValueError(f"invalid complete stream barrier: {dataset}/{order}")
        if barrier.get("runtime_sha256") != runtime_sha:
            raise ValueError("stream barrier runtime hash changed")
        barrier_files = barrier.get("files", [])
        if len(barrier_files) != 16:
            raise ValueError("complete stream barrier does not list 16 files")
        for cid, item in zip(full_ordered_ids, barrier_files):
            artifact, _receipt = worker._stream_paths(dataset, order, cid)
            if (item.get("path") != str(artifact.relative_to(ROOT)) or not artifact.is_file()
                    or common.sha(artifact) != item.get("sha256")
                    or artifact.stat().st_size != int(item.get("bytes", -1))):
                raise ValueError(f"complete stream barrier artifact mismatch: {cid}")
    if max_items is not None and len(present) != int(max_items):
        raise ValueError(f"bounded prefix requires {int(max_items)} leading items, found {len(present)}")
    elif require_complete and not barrier_path.exists():
        raise ValueError(f"complete stream barrier is missing: {dataset}/{order}")

    previous_sha = None
    previous_state = None
    rows = []
    ln_names_ref = None
    alpha_expected = float(locked_config(dataset, "current5")["writeback"])
    config_expected = locked_config(dataset, "current5")
    reference_lock_sha = common.sha(worker.REFERENCE_PATH)
    for position, (cid, path, receipt_path) in enumerate(present):
        rec = roster_index[cid]
        ref = reference_index[cid]
        receipt = _load(receipt_path)
        if (receipt.get("schema") != "stvg-reference-carry-case-receipt-v1"
                or receipt.get("status") != "sealed" or receipt.get("GT_read") is not False
                or receipt.get("case_id") != cid or receipt.get("dataset") != dataset
                or receipt.get("order") != order or receipt.get("reference_lock_sha256") != reference_lock_sha
                or receipt.get("runtime_sha256") != runtime_sha
                or receipt.get("previous_result_sha256") != previous_sha):
            raise ValueError(f"receipt identity/chain mismatch for {cid}")
        artifact_sha = common.sha(path)
        artifact_bytes = path.stat().st_size
        if receipt.get("sha256") != artifact_sha or int(receipt.get("bytes", -1)) != artifact_bytes:
            raise ValueError(f"saved artifact differs from receipt for {cid}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload.get("schema") != "stvg-reference-carry-case-v1"
                or payload.get("GT_read") is not False or payload.get("case_id") != cid
                or payload.get("dataset") != dataset or payload.get("order") != order
                or int(payload.get("stream_position", -1)) != position
                or payload.get("source_case_sha256") != ref.get("source_case_sha256")
                or payload.get("reference_lock_sha256") != reference_lock_sha
                or payload.get("runtime_sha256") != runtime_sha
                or payload.get("previous_result_sha256") != previous_sha):
            raise ValueError(f"payload identity/chain mismatch for {cid}")
        if payload.get("reference_anchors") != ref.get("anchors"):
            raise ValueError(f"saved anchors differ from the locked anchors for {cid}")
        if set(payload.get("arms", {})) != set(worker.ARMS):
            raise ValueError(f"saved arm set mismatch for {cid}")

        source_case, _ = common.load_case(dict(rec, source_case_sha256=ref["source_case_sha256"]))
        if (source_case.get("source_case_sha256") != ref.get("source_case_sha256")
                or source_case.get("GT_read") is not False
                or source_case.get("source_model_state_sha256") != payload.get("source_model_state_sha256")
                or source_case.get("pixel_sha256") != payload.get("pixel_sha256")
                or list(map(int, source_case.get("frame_ids", []))) != list(map(int, payload.get("frame_ids", [])))):
            raise ValueError(f"source case identity differs from saved output for {cid}")

        transition = payload.get("state_transition", {})
        source_state = transition.get("source_state_before_stream", {})
        light_source = source_case.get("source_light_initial_state", {})
        incoming = transition.get("incoming_state_before_fit", {})
        fitted = transition.get("fit_final_state_before_writeback", {})
        next_state = transition.get("next_stream_state", {})
        fit = payload.get("fit", {})
        fit_initial = fit.get("initial_state", {})
        ln_names = _assert_state(source_state, torch, f"{cid} source state")
        if ln_names_ref is None:
            ln_names_ref = ln_names
        elif set(ln_names) != set(ln_names_ref):
            raise ValueError("LayerNorm tensor names changed across the stream")
        _assert_state(light_source, torch, f"{cid} source_light_initial_state")
        _assert_state(incoming, torch, f"{cid} incoming state")
        _assert_state(fitted, torch, f"{cid} fitted state")
        _assert_state(next_state, torch, f"{cid} next state")
        if not _same_state(source_state, light_source, torch):
            raise ValueError(f"saved Source state differs from source_light_initial_state for {cid}")
        if not _same_state(incoming, fit_initial, torch) or not _same_state(incoming, payload.get("initial_state", {}), torch):
            raise ValueError(f"fit/payload arrival state mismatch for {cid}")
        if not _same_state(fitted, fit.get("fit_final_state", {}), torch) or not _same_state(fitted, fit.get("final_state", {}), torch):
            raise ValueError(f"fit final state differs from transition for {cid}")
        if not _same_state(next_state, payload.get("next_committed", {}), torch):
            raise ValueError(f"transition next state differs from committed payload for {cid}")
        if position == 0:
            expected_ln = {name: source_state[name] for name in ln_names}
            if (payload.get("committed_before") is not None
                    or payload.get("previous_stream_item") is not None):
                raise ValueError("first stream item must not claim a prior committed state")
        else:
            expected_ln = {name: previous_state[name] for name in ln_names}
            if not _same_state(previous_state, payload.get("committed_before", {}), torch):
                raise ValueError(f"incoming committed state does not match previous query for {cid}")
            previous_row = rows[-1]
            if payload.get("previous_stream_item") != {
                "case_id": previous_row["case_id"], "result_sha256": previous_sha,
            }:
                raise ValueError(f"previous stream item binding mismatch for {cid}")
        query = "spatial.query_residual"
        if any(not torch.equal(incoming[name], expected_ln[name]) for name in ln_names):
            raise ValueError(f"incoming LN values are not the Source/previous committed state for {cid}")
        if torch.count_nonzero(incoming[query]).item() != 0:
            raise ValueError(f"Query residual is nonzero at arrival for {cid}")
        if torch.count_nonzero(next_state[query]).item() != 0:
            raise ValueError(f"Query residual was carried after {cid}")
        if (fit.get("optimizer_state_reset_each_query") is not True
                or fit.get("query_residual_reset_at_arrival") is not True
                or fit.get("seed_key") != stable_fit_key(dataset, rec["query_key_sha256"])
                or fit.get("config") != config_expected
                or fit.get("validation_enabled") is not True
                or fit.get("GT_read") is not False):
            raise ValueError(f"fit reset/seed/config contract mismatch for {cid}")
        if not bool(transition.get("carry_ln")) or float(transition.get("writeback_alpha", -1)) != alpha_expected:
            raise ValueError(f"LN carry mode/writeback alpha mismatch for {cid}")
        if transition.get("actual_query_writeback_is_zero") is not True:
            raise ValueError(f"transition does not attest zero Query writeback for {cid}")
        for name in ln_names:
            expected_next = incoming[name] + (fitted[name] - incoming[name]) * alpha_expected
            if not torch.equal(next_state[name], expected_next):
                raise ValueError(f"LN alpha equation mismatch at {cid}:{name}")
        if fit.get("optimizer_steps", -1) < 0 or int(fit.get("optimizer_steps", -1)) > int(fit.get("declared_steps", 0)):
            raise ValueError(f"invalid optimizer step count for {cid}")

        parent_path = parent_files[cid]["path"]
        parent = torch.load(parent_path, map_location="cpu", weights_only=False)
        if parent.get("GT_read") is not False or parent.get("source_case_sha256") != ref.get("source_case_sha256"):
            raise ValueError(f"parent prediction source/GT binding mismatch for {cid}")
        parent_fit = parent.get("fits", {}).get("new_crop", {})
        if parent_fit:
            if (fit.get("seed_key") != parent_fit.get("seed_key")
                    or fit.get("config") != parent_fit.get("config")):
                raise ValueError(f"E3 fit seed/config differ from parent episodic new_crop fit for {cid}")
        elif ref.get("anchors"):
            raise ValueError(f"parent lacks new_crop fit despite nonempty anchors for {cid}")
        parent_arms = parent["arms"]
        for e3_name, parent_name in (("Episodic_Native", "S_new_crop"), ("Episodic_T", "TS_new_crop")):
            e3_arm = payload["arms"][e3_name]
            parent_arm = parent_arms[parent_name]
            if (not np.array_equal(np.asarray(e3_arm["boxes"]), np.asarray(parent_arm["boxes"]))
                    or list(e3_arm["interval"]) != list(parent_arm["interval"])):
                raise ValueError(f"E3 {e3_name} is not exact parent {parent_name} reuse for {cid}")
        if not parent_fit:
            for e3_name, baseline_name in (("Episodic_Native", "F"), ("Episodic_T", "T")):
                if (not np.array_equal(np.asarray(payload["arms"][e3_name]["boxes"]),
                                       np.asarray(parent_arms[baseline_name]["boxes"]))
                        or list(payload["arms"][e3_name]["interval"]) != list(parent_arms[baseline_name]["interval"])):
                    raise ValueError(f"parent empty-reference arm is not exact no-update reuse for {cid}")
        if position == 0:
            for after_name, parent_name in (("After_Native", "S_new_crop"), ("After_T", "TS_new_crop")):
                after = payload["arms"][after_name]
                parent_arm = parent_arms[parent_name]
                if (not np.array_equal(np.asarray(after["boxes"]), np.asarray(parent_arm["boxes"]))
                        or list(after["interval"]) != list(parent_arm["interval"])):
                    raise ValueError(f"first-source {after_name} is not exact episodic {parent_name} output")
        if (payload.get("source_restored") is not True
                or payload.get("source_cold_readback", {}).get("status") != "pass"
                or payload.get("source_cold_readback", {}).get("Source_restored_after_Before_and_After") is not True
                or payload.get("actual_full_readout_calls") != 2
                or payload.get("readouts", {}).get("full_model_redecode") is not True):
            raise ValueError(f"full-model readout/source-restore receipt mismatch for {cid}")
        if int(receipt.get("optimizer_steps", -1)) != int(fit["optimizer_steps"]):
            raise ValueError(f"receipt optimizer count differs from payload for {cid}")
        if receipt.get("next_state_sha256") != state_hash(next_state):
            raise ValueError(f"receipt next-state hash mismatch for {cid}")

        rows.append({
            "case_id": cid,
            "position": position,
            "artifact_sha256": artifact_sha,
            "source_case_sha256": ref["source_case_sha256"],
            "seed_matches_locked_query": fit.get("seed_key") == stable_fit_key(dataset, rec["query_key_sha256"]),
            "parent_fit_seed_match": True if parent_fit else None,
            "parent_new_crop_fit_present": bool(parent_fit),
            "state_tensors_checked": len(source_state),
            "ln_equations_checked": len(ln_names),
            "optimizer_steps": int(fit["optimizer_steps"]),
            "first_source_after_exact_episodic": position == 0,
            "previous_result_sha256": previous_sha,
        })
        previous_state = next_state
        previous_sha = artifact_sha

    checks = {
        "source_state_matches_source_light_initial_state": len(rows),
        "incoming_state_chain_matches": len(rows),
        "all_seven_tensor_roles_checked": len(rows),
        "ln_alpha_equations_checked": sum(r["ln_equations_checked"] for r in rows),
        "query_reset_and_zero_writeback_checked": len(rows),
        "fresh_adam_and_locked_seed_checked": len(rows),
        "parent_fit_seed_match": sum(r["parent_fit_seed_match"] is True for r in rows),
        "parent_fit_seed_not_applicable": sum(r["parent_fit_seed_match"] is None for r in rows),
        "parent_episodic_arms_exactly_reused": len(rows),
        "full_model_source_restoration_checked": len(rows),
        "first_source_after_equals_episodic": sum(bool(r["first_source_after_exact_episodic"]) for r in rows),
    }
    return {
        "dataset": dataset,
        "order": order,
        "status": "pass",
        "count": len(rows),
        "complete": len(rows) == 16 and barrier_path.is_file(),
        "ordered_case_ids": [r["case_id"] for r in rows],
        "checks": checks,
        "rows": rows,
    }


def run(scope: str, dataset: str | None, order: str | None,
        limit: int | None = None) -> dict[str, Any]:
    import numpy as np
    import torch

    runtime = _load(worker.RUNTIME_PATH)
    if runtime.get("status") != "locked" or runtime.get("GT_read") is not False:
        raise ValueError("E3 runtime must be locked and explicitly GT-free")
    runtime_sha = common.sha(worker.RUNTIME_PATH)
    if runtime.get("pins", {}).get("scripts/run_stvg_reference_carry_v1.py") != common.sha(
            ROOT / "scripts/run_stvg_reference_carry_v1.py"):
        raise ValueError("E3 runtime does not pin the exact carry worker being audited")
    rows = common.roster()
    roster_index = {worker._cid(row): row for row in rows}
    if len(roster_index) != 32:
        raise ValueError("E3 roster must contain exactly 32 cases")
    ref_doc = _load(worker.REFERENCE_PATH)
    if ref_doc.get("status") != "locked" or ref_doc.get("GT_read") is not False:
        raise ValueError("E3 reference lock must be locked and GT-free")
    reference_index = {str(r["case_id"]): r for r in ref_doc.get("records", [])}
    if len(reference_index) != 32 or set(reference_index) != set(roster_index):
        raise ValueError("E3 reference lock membership differs from roster")
    orders_doc = _load(worker.ORDERS_PATH)
    orders = {(r["dataset"], r["order"]): list(r["case_ids"]) for r in orders_doc.get("records", [])}
    if len(orders) != 6:
        raise ValueError("ORDERS lock must bind exactly six streams")
    parent_files = worker._parent_files()
    if scope == "prefix":
        if dataset not in worker.DOMAINS or order not in worker.ORDER_NAMES:
            raise ValueError("prefix audit requires one locked dataset and order")
        if limit is None or isinstance(limit, bool) or not 1 <= int(limit) <= 16:
            raise ValueError("prefix audit requires --max-items from 1 through 16")
        streams = [(dataset, order)]
        require_complete = False
    else:
        if dataset is not None or order is not None or limit is not None:
            raise ValueError("all-stream audit does not accept dataset/order/limit filters")
        streams = [(d, o) for d in worker.DOMAINS for o in worker.ORDER_NAMES]
        require_complete = True
    fitter_source, fitter_source_sha = _fresh_adam_source_check()
    results = [_one_stream(d, o, require_complete=require_complete, torch=torch, np=np,
                           roster_index=roster_index, reference_index=reference_index,
                           parent_files=parent_files, orders=orders, runtime_sha=runtime_sha,
                           fitter_source_sha=fitter_source_sha,
                           max_items=limit if scope == "prefix" else None)
               for d, o in streams]
    total = sum(result["count"] for result in results)
    expected_total = int(limit) if scope == "prefix" else 96
    if total != expected_total:
        requirement = "all-stream audit requires exactly 96 saved outputs" if scope == "all" else "bounded prefix must contain exactly the requested item count"
        raise ValueError(f"{requirement}, got {total}")
    return {
        "schema": "stvg-reference-carry-cpu-readback-v1",
        "status": "pass",
        "scope": scope,
        "count": total,
        "stream_count": len(results),
        "streams": results,
        "bindings": {
            "runtime_sha256": runtime_sha,
            "roster_lock_sha256": common.sha(worker.ROSTER_PATH),
            "reference_lock_sha256": common.sha(worker.REFERENCE_PATH),
            "orders_sha256": common.sha(worker.ORDERS_PATH),
            "parent_prediction_barrier_sha256": common.sha(worker.PARENT / "GLOBAL_PREDICTION_BARRIER.json"),
            "fitter_source_path": fitter_source,
            "fitter_source_sha256": fitter_source_sha,
            "audit_source_sha256": common.sha(Path(__file__)),
        },
        "audit_constraints": {"GT_read": False, "model_loaded": False,
                              "GPU_used": False, "media_decoded": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("prefix", "all"), default="prefix")
    parser.add_argument("--dataset", choices=worker.DOMAINS)
    parser.add_argument("--order", choices=worker.ORDER_NAMES)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    if args.scope == "prefix" and (not args.dataset or not args.order or args.max_items is None):
        parser.error("--scope prefix requires --dataset, --order, and --max-items")
    result = run(args.scope, args.dataset, args.order, args.max_items)
    name = "ROOT_PREFIX_CPU_READBACK002.json" if args.scope == "prefix" else "ROOT_STREAM_CPU_READBACK.json"
    path = worker.BASE / name
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable audit receipt: {path}")
    common.write(path, result)
    print(json.dumps({"status": result["status"], "scope": result["scope"],
                      "count": result["count"], "stream_count": result["stream_count"],
                      "receipt": str(path.relative_to(ROOT)), "GT_read": False,
                      "model_loaded": False, "GPU_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()

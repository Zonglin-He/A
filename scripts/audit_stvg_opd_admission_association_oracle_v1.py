"""Additional postseal scalar Oracle/evidence audit; original science is untouched.

Importing this file performs no I/O. run() is the explicit root-invoked entry:
both complete prediction barriers and the original scientific root pass must
exist before it calls the label loader. Geometry/argmax/flags use scalar Python
formulas rather than the experiment's NumPy corners/overlap implementation.
No fit, decoder, GPU, teacher, or optimizer is run. Oracle logical provenance
stays GT-assisted when its canonical payload was fitted without GT.
"""
import copy
import math

FORMAL_ARMS = ("H_Top1", "H_Path", "R_Top1", "R_Path", "R_Add", "R_Shuffle")
EPS = 1e-12


def _list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value.tolist() if hasattr(value, "tolist") else list(value)


def _xyxy(box, width, height, hc):
    cx, cy, bw, bh = map(float, _list(box))
    assert all(math.isfinite(v) for v in (cx, cy, bw, bh)) and bw > 0 and bh > 0
    coordinates = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height]
    return [max(0., v) for v in coordinates] if hc else coordinates


def _iou(first, second):
    a, b = list(map(float, _list(first))), list(map(float, _list(second)))
    assert len(a) == len(b) == 4 and all(math.isfinite(v) for v in a + b)
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    denominator = area_a + area_b - intersection
    return intersection / denominator if denominator > 0 else 0.


def _first_max(values):
    assert values and all(math.isfinite(float(v)) for v in values)
    winner = 0
    for i in range(1, len(values)):
        if values[i] > values[winner]:
            winner = i
    return winner


def _assert_tree(actual, expected):
    """Same existing root numeric tolerance, exact discrete fields and shape."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(actual) == set(expected)
        return sum(_assert_tree(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        return sum(_assert_tree(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, float):
        assert math.isfinite(float(actual)) and math.isfinite(expected)
        assert abs(float(actual) - expected) < 2e-10 * (1 + abs(expected)), (actual, expected)
    else:
        assert actual == expected, (actual, expected)
    return 1


def _origin(row, pools, frame_ids):
    pool = pools[int(row["position"])]
    candidate = int(row["candidate_index"])
    assert 0 <= candidate < len(pool["boxes"]) <= 3
    assert int(row["frame_id"]) == int(pool["frame_id"]) == int(frame_ids[row["position"]])
    assert row["candidate_id"] == pool["candidate_ids"][candidate]
    assert list(map(float, row["box"])) == list(map(float, pool["boxes"][candidate]))
    assert float(row["score"]) == float(pool["scores"][candidate])
    assert row["old_admitted"] == pool["accepted"] and row["context_active"] == pool["context_active"]
    return pool, candidate


def _expected_oracle(selection, frame_ids, truth, span, width, height, dataset):
    pools = {int(p["position"]): p for p in selection["pools"]}
    assert len(pools) == len(selection["pools"])
    original = selection["arms"]["R_Top1"]
    assert original["GT_assisted"] is False
    out = copy.deepcopy(original)
    out["GT_assisted"] = True
    provenance, known = [], 0
    for row in out["rows"]:
        pool, candidate = _origin(row, pools, frame_ids)
        assert candidate == _first_max(pool["scores"])
        fid = int(row["frame_id"])
        use = span[0] <= fid < span[1] and fid in truth
        if use:
            quality = [_iou(_xyxy(box, width, height, dataset == "hc2"), truth[fid]) for box in pool["boxes"]]
            candidate = _first_max(quality)
            row.update(candidate_index=candidate, candidate_id=pool["candidate_ids"][candidate],
                       box=copy.deepcopy(pool["boxes"][candidate]), score=pool["scores"][candidate])
            known += 1
        provenance.append(dict(position=row["position"], GT_selection_used=bool(use),
                               retained_R_Top1_without_label=not use,
                               outside_event=not span[0] <= fid < span[1]))
    return out, provenance, known, len(out["rows"]) - known


def _evidence(record, fixed, selection, frame_ids, truth, span, width, height, arm):
    pools = {int(p["position"]): p for p in selection["pools"]}
    output = []
    for row in fixed["rows"]:
        pool, candidate = _origin(row, pools, frame_ids)
        top1 = _first_max(pool["scores"])
        fid = int(row["frame_id"])
        outside = not span[0] <= fid < span[1]
        known = not outside and fid in truth
        if known:
            quality = [_iou(_xyxy(box, width, height, record["dataset"] == "hc2"), truth[fid]) for box in pool["boxes"]]
            selected_quality, top_quality = quality[candidate], quality[top1]
        else:
            selected_quality = top_quality = None
        descending = sorted(map(float, pool["scores"]), reverse=True)
        output.append(dict(parent_id=record["index"], dataset=record["dataset"], arm=arm,
            GT_assisted=arm == "Oracle", sample_ordinal=row["position"], candidate_index=candidate,
            Top1_index=top1, candidate_count=len(pool["boxes"]), score=row["score"],
            original_margin=descending[0] - descending[1] if len(descending) > 1 else None,
            old_admitted=row["old_admitted"], restored_margin=not row["old_admitted"], reason=pool["reason"],
            context_active=row["context_active"], changed_Top1=candidate != top1, GT_known=bool(known),
            outside_event=bool(outside), missing_annotation=not outside and fid not in truth,
            expert_GT_IoU=selected_quality, Top1_GT_IoU=top_quality,
            quality_delta_vs_Top1=selected_quality - top_quality if known else None,
            correct_Top1_replaced_worse=bool(known and candidate != top1 and top_quality >= .5 and selected_quality < top_quality - EPS),
            wrong_Top1_replaced_better=bool(known and candidate != top1 and top_quality < .5 and selected_quality > top_quality + EPS),
            correct_Top1_destroyed_at_05=bool(known and top_quality >= .5 and selected_quality < .5),
            wrong_Top1_corrected_at_05=bool(known and top_quality < .5 and selected_quality >= .5)))
    return output


def run():
    """Explicit post-root invocation only; writes one additional private receipt."""
    import time
    import torch
    from scripts.stvg_opd_admission_association_common_v1 import BASE, ROOT, ARMS, read, sha, load, write, resolve
    from scripts.score_stvg_opd_admission_association_v1 import require_seal, load_targets, rows_read, OUT
    assert not torch.cuda.is_initialized()
    root_receipt = read(BASE / "ROOT_READBACK.json")
    assert root_receipt["status"] == "pass"
    assert ARMS == list(FORMAL_ARMS)
    # These two calls rehash the sealed logical receipts and canonical predictions.
    formal, oracle = require_seal("formal"), require_seal("oracle")
    assert root_receipt["time"] >= oracle["time"]
    root_sha = sha(BASE / "ROOT_READBACK.json")
    formal_sha = sha(BASE / "GLOBAL_PREDICTION_BARRIER.json")
    oracle_sha = sha(BASE / "ORACLE_PREDICTION_BARRIER.json")
    start = time.time()
    records = read(BASE / "DESIGN_LOCK.json")["records"]
    assert len(records) == 64 and len({r["index"] for r in records}) == 64
    declaration = read(BASE / "ORACLE_DESIGN.json")
    declaration_sha = sha(BASE / "ORACLE_DESIGN.json")
    assert declaration["GT_assisted"] is True and formal["time"] < declaration["time"] <= oracle["time"]
    assert declaration["formal_barrier_sha256"] == formal_sha
    designs = {d["index"]: d for d in declaration["records"]}
    assert len(designs) == 64 and set(designs) == {r["index"] for r in records}
    # The label loader is deliberately below both barriers and the original root pass.
    targets = load_targets(records)
    from scripts.run_decota_paper_main_v1 import read_row
    anonymous = rows_read(OUT / "EVIDENCE_ROWS.jsonl.gz")
    evidence_sha = sha(OUT / "EVIDENCE_ROWS.jsonl.gz")
    saved = {(e["parent_id"], e["arm"], e["sample_ordinal"]): e for e in anonymous}
    assert len(saved) == len(anonymous)
    checks = expected_rows = known_frames = retained_frames = canonical_free = canonical_privileged = 0
    seen = set()
    all_bindings = {}
    for record in records:
        dataset, query = record["dataset"], record["query_ordinal"]
        truths, spans, bindings = targets[dataset]
        for rel, digest in bindings.items(): assert sha(ROOT / rel) == digest
        all_bindings.update(bindings)
        truth = {int(fid): box for fid, box in truths[query].items()}
        span = spans[query]
        raw = read_row(dataset, query)
        width, height = raw["input"]["width"], raw["input"]["height"]
        selection = read(ROOT / record["selection_path"])
        assert sha(ROOT / record["selection_path"]) == record["selection_sha256"]
        expected, provenance, known, retained = _expected_oracle(selection, raw["frame_ids"], truth, span, width, height, dataset)
        design = designs[record["index"]]
        assert design["dataset"] == dataset and design["original_selection_sha256"] == record["selection_sha256"]
        checks += _assert_tree(design["GT_provenance"], bindings)
        checks += _assert_tree(design["selection"], expected)
        checks += _assert_tree(design["GT_selection_provenance"], provenance)
        assert design["known_selection_frames"] == known and design["retained_unscorable_frames"] == retained
        known_frames += known; retained_frames += retained
        for arm in FORMAL_ARMS + ("Oracle",):
            value = resolve(record["index"], arm, "oracle" if arm == "Oracle" else "formal")
            fixed = value["selection"]
            selected = expected if arm == "Oracle" else selection["arms"][arm]
            checks += _assert_tree(fixed, selected)
            assert value["logical_GT_assisted"] == (arm == "Oracle")
            assert value["index"] == record["index"] and value["dataset"] == dataset
            assert value["frame_ids"] == raw["frame_ids"] and value["intervals"] == record["intervals"]
            path = ROOT / value["canonical_payload_path"]
            assert sha(path) == value["canonical_payload_sha256"]
            original = load(path)
            canonical = bool(original["fit"]["GT_used"])
            assert value["canonical_GT_used"] == value["fit"]["GT_used"] == canonical
            assert original["fixed_evidence"]["GT_assisted"] == canonical
            assert original["GT_read"] == canonical
            checks += _assert_tree(fixed["rows"], original["fixed_evidence"]["rows"])
            if arm != "Oracle": assert not canonical
            else:
                canonical_privileged += int(canonical)
                canonical_free += int(not canonical)
            evidence = _evidence(record, fixed, selection, raw["frame_ids"], truth, span, width, height, arm)
            for e in evidence:
                identity = (e["parent_id"], e["arm"], e["sample_ordinal"])
                assert identity in saved and identity not in seen
                seen.add(identity); checks += _assert_tree(saved[identity], e); expected_rows += 1
    assert seen == set(saved) and expected_rows == len(anonymous)
    checks += _assert_tree(declaration["GT_provenance"], all_bindings)
    checks += _assert_tree(root_receipt["GT_provenance"], all_bindings)
    require_seal("formal"); require_seal("oracle")
    assert not torch.cuda.is_initialized() and read(BASE / "ROOT_READBACK.json")["status"] == "pass"
    assert sha(BASE / "ROOT_READBACK.json") == root_sha
    assert sha(BASE / "GLOBAL_PREDICTION_BARRIER.json") == formal_sha
    assert sha(BASE / "ORACLE_PREDICTION_BARRIER.json") == oracle_sha
    assert sha(BASE / "ORACLE_DESIGN.json") == declaration_sha
    assert sha(OUT / "EVIDENCE_ROWS.jsonl.gz") == evidence_sha
    result = dict(status="pass", parents=64, formal_logical_arms=6, Oracle_logical_fits=64,
        independent_scalar_checks=checks, all_anonymous_evidence_rows=expected_rows,
        Oracle_known_selection_frames=known_frames, Oracle_retained_unscorable_frames=retained_frames,
        Oracle_logical_references_to_GT_free_canonical=canonical_free,
        Oracle_logical_references_to_GT_assisted_canonical=canonical_privileged,
        scope="Independent scalar xyxy/IoU/first-argmax of existing R candidates, unknown R-Top1 fallback, all seven-arm anonymous evidence quality/flags and canonical-vs-logical GT provenance; no decoder/Jacobian replay.",
        original_science_root_sha256=root_sha,
        formal_barrier_sha256=formal_sha,
        Oracle_barrier_sha256=oracle_sha,
        Oracle_design_sha256=declaration_sha,
        public_evidence_rows_sha256=evidence_sha,
        auditor_sha256=sha(__file__), CPU_only=True, GT_read=True, GT_only_after_both_prediction_seals=True,
        original_science_files_or_predictions_modified=False, model_calls=0, independent_full_decoder_Jacobian=False,
        time=time.time(), CPU_seconds=time.time() - start)
    write(BASE / "ROOT_ORACLE_SELECTION_READBACK.json", result)
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))

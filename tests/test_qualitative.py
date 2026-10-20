"""Synthetic CPU-only selector/seal tests; no RGB, labels, scoring or model use."""
import hashlib
import json
from pathlib import Path

import pytest

from tools import qualitative as q
from tools.run_formal import canonical, digest


def row(key, parent, dv, dt, ds):
    return dict(id=key, parent=parent, dataset="hc2",
                values=dict(Source=dict(v=.3, t=.4, fixed_actual_sIoU=.4),
                            Full=dict(v=.3+dv, t=.4+dt, fixed_actual_sIoU=.4+ds)))


def test_predeclared_signed_medians_parent_exclusion_and_missing_categories():
    rows = [row("s1", "S1", .01, -.01, .01), row("s2", "S2", .02, 0, .02),
            row("s3", "S3", .03, -.01, .03),
            row("t1", "T1", .01, .01, -.01), row("t2", "S2", .02, .02, 0),
            row("t3", "T3", .03, .03, -.01),
            row("j1", "J1", .01, .01, .01), row("j2", "J2", .02, .02, .02),
            row("j3", "J3", .03, .03, .03),
            row("f1", "F1", -.03, -.01, -.01), row("f2", "F2", -.02, .01, -.02),
            row("f3", "F3", -.01, -.01, .01)]
    selected = q.select_examples(rows)
    assert selected["spatial"]["id"] == "s2"
    # The middle temporal example has the already selected S2 parent; remove
    # it first, then take the predeclared upper median of the remaining two.
    assert selected["temporal"]["id"] == "t3"
    assert selected["joint"]["id"] == "j2"
    assert selected["failure"]["id"] == "f2"
    assert len({v["parent"] for v in selected.values()}) == 4
    assert selected["spatial"]["query"] == "Query000002"
    assert selected["failure"]["delta_v"] < 0
    empty = q.select_examples([rows[6]])
    assert empty["joint"]["status"] == "selected"
    assert all(empty[k]["status"] == "missing" for k in ["spatial", "temporal", "failure"])
    assert q.six_actual_positions([0, 2, 7, 9, 11, 14, 18, 25]) == [0, 1, 3, 4, 6, 7]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def synthetic_job(root):
    folder = root / "outputs/formal"
    output = folder / "tastvg/hc2"
    manifest = root / "data/evaluation/formal/hc2.jsonl"
    cohort = [dict(id="q1", dataset="hc2", key="q1", metadata={"source": "toy"}),
              dict(id="q2", dataset="hc2", key="q2", metadata={"source": "toy"})]
    manifest.parent.mkdir(parents=True)
    manifest.write_text("".join(json.dumps(r)+"\n" for r in cohort))
    source = root / "stvg/synthetic.py"
    source.parent.mkdir(parents=True)
    source.write_text("# synthetic code fingerprint, no inference\n")
    proof = dict(code={"stvg/synthetic.py": digest(source)})
    fingerprint = hashlib.sha256(canonical(proof)).hexdigest()
    mh = digest(manifest)
    resume = hashlib.sha256(canonical(dict(manifest_sha256=mh, runtime_fingerprint=fingerprint,
                                          backbone="tastvg"))).hexdigest()
    run = dict(manifest_sha256=mh, runtime_fingerprint=fingerprint, backbone="tastvg",
               queries=2, labels_read=False, resume_hash=resume)
    save(output / "run.json", run)
    save(output / "runtime_fingerprint.json", proof)
    files = {}
    for record in cohort:
        prediction = output / "predictions" / (record["id"]+".npz")
        prediction.parent.mkdir(parents=True, exist_ok=True)
        prediction.write_bytes(b"synthetic immutable prediction; not an actual array")
        files[record["id"]] = dict(path=str(prediction.relative_to(output)), sha256=digest(prediction))
        save(prediction.with_suffix(".json"), dict(resume_hash=resume, labels_read=False,
             request_sha256=hashlib.sha256(canonical(record)).hexdigest(), prediction_sha256=digest(prediction)))
    save(output / "sealed.json", dict(run, files=files, status="sealed_before_offline_scoring"))
    # Toy metric numbers only; no ground-truth annotation file is created.
    rows = [row("q1", "same_parent", .02, -.01, .02), row("q2", "same_parent", -.02, -.01, -.01)]
    save(output / "metrics.json", dict(status="scored_sealed_predictions", rows=rows,
         provenance=dict(manifest_sha256=mh), datasets=dict(hc2=dict(queries=2))))
    job = dict(id="tastvg-hc2", backbone="tastvg", dataset="hc2", queries=2,
               manifest=str(manifest.relative_to(root)), manifest_sha256=mh,
               labels="data/never_open_labels.jsonl", output=str(output.relative_to(root)))
    plan = dict(schema=1, jobs=[job])
    save(folder / "plan.json", plan)
    state = dict(plan_sha256=hashlib.sha256(canonical(plan)).hexdigest(),
                 jobs={job["id"]: dict(status="scored", metrics_sha256=digest(output/"metrics.json"),
                                      seal_sha256=digest(output/"sealed.json"))})
    save(folder / "state.json", state)
    return output, folder, state


def test_full_cohort_verification_blocks_before_any_label_or_rgb_access(tmp_path, monkeypatch):
    output, folder, state = synthetic_job(tmp_path)
    verified = q.verified_job(tmp_path, "tastvg", "hc2")
    assert len(verified["rows"]) == 2
    assert not verified["labels"].exists()  # The verifier never needed labels.
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append("forbidden")
        raise AssertionError("Labels/RGB must never be reached by a failed seal")

    monkeypatch.setattr(q, "load_selected_truth", forbidden)
    monkeypatch.setattr(q, "render_example", forbidden)
    # q2 will not be selected because it repeats q1's parent. Its bytes must
    # nevertheless be verified before the first selected example can read GT.
    selected = q.select_examples(verified["rows"])
    assert selected["failure"]["status"] == "missing"
    (output / "predictions/q2.npz").write_bytes(b"tampered unselected prediction")
    with pytest.raises(ValueError, match="Sealed prediction changed"):
        q.generate(tmp_path, "tastvg", "hc2")
    assert calls == []
    assert not (tmp_path / "outputs/formal/qualitative").exists()
    state["jobs"]["tastvg-hc2"]["status"] = "running"
    save(folder / "state.json", state)
    with pytest.raises(ValueError, match="to be scored"):
        q.generate(tmp_path, "tastvg", "hc2")
    assert calls == []

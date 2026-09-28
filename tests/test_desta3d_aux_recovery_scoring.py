"""Synthetic controls only; never load source/target labels or predictions."""
import copy
import json

import pytest
import torch

from scripts import score_desta3d_v2_aux_recovery as s


def make_seal(tmp_path):
    pins = {}
    for arm in s.ARMS:
        p = tmp_path / "predictions" / arm / "000.pt"
        p.parent.mkdir(parents=True)
        p.write_bytes(b"synthetic")
        pins[str(p.resolve())] = s.scalar.sha256_file(p)
    seal = {"pins": pins, "arms": list(s.ARMS), "queries_per_arm": 1, "parents": 1,
            "new_predictions": 4, "source_validation_GT_read": False, "target_GT_read": False}
    path = tmp_path / "ALL_PREDICTIONS_SEAL.json"
    path.write_text(json.dumps(seal))
    (tmp_path / "PREDICTIONS_COMPLETE.json").write_text(json.dumps({"seal_sha": s.scalar.sha256_file(path)}))
    return seal


def test_no_complete_marker_cannot_open_labels(tmp_path, monkeypatch):
    def never(*args, **kwargs):
        raise AssertionError("payload reader called before completion barrier")
    monkeypatch.setattr(s.torch, "load", never)
    with pytest.raises(FileNotFoundError):
        s.verify_before_labels(tmp_path)


def test_tampered_file_and_wrong_roster_rejected(tmp_path):
    make_seal(tmp_path)
    assert s.sealed_files(tmp_path, 1, 1)["new_predictions"] == 4
    with pytest.raises(RuntimeError, match="roster"):
        s.sealed_files(tmp_path, 198, 31)
    (tmp_path / "predictions/B0/000.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA"):
        s.sealed_files(tmp_path, 1, 1)


def example():
    row = {"key": "synthetic", "source": "parent", "input": {"frame_ids": [10, 20], "video_sha256": "c" * 64}}
    pred = {"key": "synthetic", "source": "parent", "frame_ids": [10, 20], "video_sha256": "c" * 64,
            "adapter_sha": "a" * 64, "GT_read": False, "preprocess": {"pixel_sha": "b" * 64},
            "event_logits": torch.zeros(1, 2), "positions": [0, 1], "boxes_cxcywh": torch.tensor([[.5, .5, 1, 1]] * 2),
            "geometry_valid": torch.ones(2, dtype=torch.bool), "interval": [0, 1], "format_ok": True}
    label = {"frame_ids": [10, 20], "box_valid": [True, True], "event_active": [True, True],
             "boxes_xyxy": [[0, 0, 1, 1]] * 2, "event_interval": {"begin_fid": 10, "end_fid": 21}}
    return row, pred, label


def test_identity_and_frozen_pixels_before_scoring():
    row, pred, _ = example()
    assert s.check_prediction(pred, row, "a" * 64) == pred["preprocess"]
    bad = copy.deepcopy(pred); bad["frame_ids"] = [10, 19]
    with pytest.raises(RuntimeError, match="frame IDs"):
        s.check_prediction(bad, row, "a" * 64)
    cache = {"key": row["key"], "source": row["source"], "frame_ids": [10, 20], "condition": "clean",
             "GT_read": False, "preprocess": {"pixel_sha": "d" * 64}}
    with pytest.raises(RuntimeError, match="physical pixels"):
        s.frozen_prediction(cache, row, pred["preprocess"])


def test_physical_time_missing_support_and_format_denominators():
    _, pred, label = example()
    score = s.scalar.score_tube_independently(pred, label)
    assert [score[k] for k in s.METRICS] == [1, 1, 1]
    pred["interval"] = [0, 0]
    score = s.scalar.score_tube_independently(pred, label)
    assert score["vIoU"] == .5 and score["sIoU"] == 1 and score["tIoU"] == pytest.approx(1 / 11)
    pred["format_ok"] = False
    score = s.scalar.score_tube_independently(pred, label)
    assert [score[k] for k in s.METRICS] == [0, 0, 0] and score["spatial_support_frames"] == 2
    pred["format_ok"] = True; label["box_valid"] = [False, False]
    assert s.scalar.score_tube_independently(pred, label)["vIoU"] == 0


def test_auc_ties_single_class_and_unequal_query_parent_weight():
    assert s.scalar.binary_auc([0, 1], [0, 0]) == .5
    assert s.scalar.binary_auc([1, 1], [.1, .9]) is None
    rows = [{"source": "a", "metrics": dict.fromkeys(s.METRICS, 1)} for _ in range(3)]
    rows += [{"source": "b", "metrics": dict.fromkeys(s.METRICS, 0)}]
    parents = s.scalar.summarize_parents(rows)
    assert sum(x["vIoU"] for x in parents.values()) / 2 == .5
    zeros = {p: dict.fromkeys(s.METRICS, 0) for p in parents}
    assert s.scalar.paired_parent_bootstrap(parents, zeros, "vIoU")["mean_delta_pp"] == 50


def test_event_detection_empty_positive_denominator():
    row, pred, label = example()
    label["event_active"] = [False, False]
    out = s.event_summary([row], {row["key"]: pred}, {row["key"]: label}, [{"metrics": {"tIoU": 0}}])
    assert out["activation_at_probability_0_5"]["parent_macro_detection"] is None
    assert out["denominators"]["source_AUROC_undefined"] == 1
    json.dumps(out, allow_nan=False)


def test_scalar_geometry_existing_positive_controls():
    s.scalar.synthetic_self_check()

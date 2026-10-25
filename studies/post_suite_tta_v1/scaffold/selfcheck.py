"""Synthetic contract/diagnostic checks, not model or dataset experiments."""
from contracts import QuerySpec, TemporalPrediction, frame_interval_seconds
from diagnostic_math import compare_logits, legal_joint_margin


def main():
    q = QuerySpec("synthetic", "synthetic-parent", "hc2", 2.0, (0.0, 0.5, 1.5))
    TemporalPrediction("synthetic", 0.5, 2.0, "fixture", "STVG_task_adapter", "fixed_source").validate(q)
    assert frame_interval_seconds(1, 3, (0.0, 0.5, 1.5, 2.0)) == (0.5, 2.0)
    try:
        TemporalPrediction("wrong-id", 0.5, 2.0, "fixture", "STVG_task_adapter").validate(q)
    except ValueError:
        pass
    else:
        raise AssertionError("Query mismatch was accepted")
    shift = compare_logits((0, 2, 1), (3, 5, 4))
    assert shift["raw_max_abs_delta"] == 3 and shift["centered_max_abs_delta"] == 0
    assert shift["probability_max_abs_delta"] == 0 and not shift["argmax_changed"]
    changed = compare_logits((0, 2, 1), (0, 2.1, 1))
    assert changed["probability_max_abs_delta"] > 0 and not changed["argmax_changed"]
    margin = legal_joint_margin((0, 5, 0), (0, 0, 5))
    assert margin["best_indices"] == [1, 2] and margin["margin"] == 5
    print("Synthetic contracts/diagnostic math passed; zero model, dataset, GT, optimizer or GPU actions")


if __name__ == "__main__":
    main()

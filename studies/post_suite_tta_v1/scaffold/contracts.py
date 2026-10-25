"""Data-only contracts. No framework, video, model, annotation or GPU imports."""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class QuerySpec:
    query_id: str
    parent_id: str
    dataset: str
    duration_seconds: float
    sampled_seconds: tuple[float, ...]

    def validate(self):
        if not self.query_id or not self.parent_id or self.dataset not in {"hc2", "vidstg"}:
            raise ValueError("Missing identity or unsupported dataset")
        if not isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("Duration must be positive and finite")
        if not self.sampled_seconds:
            raise ValueError("Explicit sampled timestamps are required")
        if any(not isfinite(t) or t < 0 or t > self.duration_seconds for t in self.sampled_seconds):
            raise ValueError("Sample outside physical timeline")
        if any(a > b for a, b in zip(self.sampled_seconds, self.sampled_seconds[1:])):
            raise ValueError("Timeline must preserve chronological order")


@dataclass(frozen=True)
class TemporalPrediction:
    query_id: str
    start_seconds: float
    end_seconds: float
    method: str
    implementation_scope: str
    spatial_provider: str | None = None

    def validate(self, query: QuerySpec):
        query.validate()
        if self.query_id != query.query_id:
            raise ValueError("Query identity mismatch")
        if not all(isfinite(x) for x in (self.start_seconds, self.end_seconds)):
            raise ValueError("Nonfinite interval")
        if not 0 <= self.start_seconds < self.end_seconds <= query.duration_seconds:
            raise ValueError("Invalid physical interval; no silent clamp or fallback")
        if self.implementation_scope not in {"original_full_method", "STVG_task_adapter", "paper_reimplementation"}:
            raise ValueError("Declare original method versus adaptation/reimplementation")


def frame_interval_seconds(start, end_exclusive, frame_edges_seconds):
    """Explicit half-open frame edges; never assume an upstream end convention."""
    if not isinstance(start, int) or not isinstance(end_exclusive, int):
        raise ValueError("Frame indices must be integers")
    if not 0 <= start < end_exclusive < len(frame_edges_seconds):
        raise ValueError("Frame interval outside declared edge grid")
    if any(not isfinite(x) for x in frame_edges_seconds):
        raise ValueError("Nonfinite frame edge")
    if any(a >= b for a, b in zip(frame_edges_seconds, frame_edges_seconds[1:])):
        raise ValueError("Frame edges must be strictly increasing")
    return frame_edges_seconds[start], frame_edges_seconds[end_exclusive]

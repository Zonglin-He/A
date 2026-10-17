"""The single final-method entrypoint: one student, one frozen spatial expert.

Default output contains only this method, not eight research control arms.
audit=True records full optimizer states and hashes; it never changes selection.
"""

import copy
import threading
import time
import torch

from .backbone import capture, full_prediction, make_batch, query_subject, validate_input
from .config import MethodConfig
from .objectives import prediction
from .observations import observations, parse_context, visual_query
from .optim import fit_spatial, fit_temporal
from .replay import SpatialReplay, TemporalReplay
from .tensors import detached, state_hash


class _EpisodeParser:
    """Memoize exact text only, for this episode. Never lowercase a cache key."""
    def __init__(self, parser):
        self.parser = copy.copy(parser)
        self.cache = {}
        original = parser.nlp

        def nlp(text):
            if text not in self.cache:
                self.cache[text] = original(text)
            return self.cache[text]

        self.nlp = nlp
        self.parser.nlp = nlp

    def __call__(self, text):
        return self.parser(text)


class DeCoTAPredictor:
    def __init__(self, model, expert, parser, direction, *, audit=False, cached_replay=True):
        for name, item in (("student", model), ("expert", expert.model)):
            if item.training or any(p.requires_grad for p in item.parameters()):
                raise ValueError(f"{name} must be eval() and fully frozen")
        self.model, self.expert, self.parser = model, expert, parser
        self.config = MethodConfig.for_direction(direction)
        self.audit, self.cached_replay = audit, cached_replay
        self._lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, direction, **options):
        from .loader import load
        from .release import verify_release
        verify_release()
        config = MethodConfig.for_direction(direction)
        return cls(*load(config), direction, **options)

    def predict(self, frames, frame_ids, metadata):
        validate_input(frames, frame_ids, metadata)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent episodes on one predictor are not supported")
        try:
            return self._predict(frames, list(map(int, frame_ids)), dict(metadata))
        finally:
            self._lock.release()

    def _predict(self, frames, ids, metadata):
        started = time.perf_counter()
        if self.audit:
            before = (state_hash(self.model.state_dict()), state_hash(self.expert.model.state_dict()))
        parser = _EpisodeParser(self.parser)
        parses = dict(subject=parser(metadata["caption"])["subject"],
                      context=parse_context(parser, metadata["caption"]),
                      old=visual_query(parser, metadata["caption"]))
        batch = make_batch(frames, ids, metadata, self.model)
        with query_subject(self.model, batch, parses["subject"]):
            views, records = capture(self.model, batch)
            spatial = SpatialReplay(self.model, views, len(ids), cached=self.cached_replay)
            zero = spatial.zero
            native = prediction(zero["logits"], zero["boxes"], records, ids)
            expert = observations(self.expert, parses, frames, ids, native["indices"], audit=self.audit)
            anchors = expert["anchors"]["single4"]
            space = fit_spatial(spatial, zero, anchors, self.config, trace=self.audit)
            head_zero = {**zero, "boxes": space["final"]["boxes"]}
            temporal = TemporalReplay(spatial.head, spatial.temporal_inputs, head_zero)
            time_result = fit_temporal(temporal, records, self.config, trace=self.audit)
            combined = {**space["state"], **time_result["shrunk_state"]}
            final = full_prediction(self.model, batch, ids, records, combined, time_result["shrunk"])
        if self.audit:
            after = (state_hash(self.model.state_dict()), state_hash(self.expert.model.state_dict()))
            if before != after:
                raise RuntimeError("Student or expert source state was not restored")
        result = dict(method="decota_final_simplified_v1", config=self.config.to_dict(),
                      **final, frame_ids=ids, I_seed=native["indices"], I_out=final["indices"],
                      spatial_state=detached(space["state"], "cpu"),
                      temporal_state=detached(time_result["shrunk_state"], "cpu"),
                      spatial_parameter_changed=space["parameter_changed"],
                      temporal_parameter_changed=time_result["shrunk_parameter_changed"],
                      interval_changed=final["indices"] != native["indices"],
                      spatial_selected_step=space["selected_step"],
                      temporal_selected_step=time_result["selected_step"],
                      spatial_losses=space["losses"], temporal_losses=time_result["losses"],
                      failure=dict(spatial=space["failure"], temporal=time_result["failure"]),
                      actual_observation_positions=expert["actual_observation_positions"],
                      anchors=anchors, parses=parses, records=records,
                      teacher=detached(time_result["evidence"], "cpu"),
                      seconds=time.perf_counter() - started, source_restored=True,
                      expert_frozen=True, GT_online=False, student_output_only=True,
                      interpolation_output=False, full_model_forward=True)
        if self.audit:
            result["audit"] = dict(spatial=detached(space, "cpu"),
                                   temporal=detached(time_result, "cpu"), expert=expert,
                                   native=native, source_hashes=before)
        return result

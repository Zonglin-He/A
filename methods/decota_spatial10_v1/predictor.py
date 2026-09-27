"""Thin, label-free packaging of the executed F34 S_PRIVATE10 arm.

No temporal adaptation, warped-video inference, teacher-box insertion, parameter
search, outcome-based fallback or experimental queue is invoked by this API.
The observation, loss, optimizer and replay implementations are inherited, not
approximated. See WORKING_METHOD.json for their versioned dependencies.
"""

import copy
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
METHOD = Path(__file__).resolve().parent
DIRECTIONS = ("vid_to_hc1", "hc2_to_vid")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_config(direction):
    if direction not in DIRECTIONS:
        raise ValueError(f"Expected one of {DIRECTIONS}, got {direction!r}")
    return json.loads((METHOD / "configs.json").read_text())[direction]


def verify_dependencies():
    manifest = json.loads((METHOD / "WORKING_METHOD.json").read_text())
    for name, expected in manifest["inherited_code_pins"].items():
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f"F34 dependency changed: {name}")
    if sha256(METHOD / "configs.json") != manifest["configs_sha256"]:
        raise RuntimeError("Spatial10 locked configuration changed")
    return manifest


def validate_input(frames, ids, metadata):
    """Accept only query and geometry metadata; labels never reach a model path."""
    allowed = {"caption", "index", "width", "height", "fps", "source",
               "original_video_id", "video_path", "video_sha256", "frame_ids",
               "frame_count", "duration", "start_frame", "end_frame", "kind"}
    if set(metadata) - allowed:
        raise ValueError(f"Unsupported input fields: {sorted(set(metadata) - allowed)}")
    if not {"caption", "index", "width", "height"} <= set(metadata):
        raise ValueError("caption/index/width/height are required")
    if not isinstance(metadata["caption"], str) or not metadata["caption"].strip():
        raise ValueError("Expected a nonempty query")
    if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 [T,H,W,3] frames")
    if frames.shape != (len(ids), metadata["height"], metadata["width"], 3):
        raise ValueError("Frame geometry and input metadata differ")
    # Both native temporal offsets need at least two positions for start < end.
    if len(ids) < 4 or any(isinstance(i, (bool, np.bool_)) or int(i) != i for i in ids):
        raise ValueError("Expected at least four integral physical frame IDs")
    if ids[0] < 0 or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("Frame IDs must be nonnegative and strictly increasing")
    if "frame_ids" in metadata and list(metadata["frame_ids"]) != list(ids):
        raise ValueError("Metadata frame grid differs from supplied frames")


class Spatial10Predictor:
    """One episode at a time; the supplied source model and expert stay frozen.

    Use from_pretrained() for the pinned local checkpoints. predict() always
    computes its own automatic parse and fresh expert observations. Its return
    value includes the full actual optimization trace and observation receipts.
    """

    def __init__(self, model, expert, parser, direction):
        verify_dependencies()
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("The source student must be eval() and fully frozen")
        if expert.model.training or any(p.requires_grad for p in expert.model.parameters()):
            raise ValueError("The expert must be eval() and fully frozen")
        self.model, self.expert, self.parser = model, expert, parser
        self.direction = direction
        self._config = load_config(direction)
        self._episode_lock = threading.Lock()

    @classmethod
    def from_pretrained(cls, direction):
        from scripts.run_decota_refine_v1 import configure, student
        from vg_tta.foreground_runtime import QuerySubjectParser
        from vg_tta.tg_spatial_tta_v1 import SpatialExpert

        verify_dependencies()
        config = load_config(direction)
        for spec in (config["checkpoint"], config["expert_weights"]):
            if sha256(spec["path"]) != spec["sha256"]:
                raise RuntimeError(f"Model weights changed: {spec['path']}")
        configure()
        model = student({"ta_checkpoints": {config["source_dataset"]: config["checkpoint"]}},
                        "tastvg", config["loader_group"])
        expert = SpatialExpert(config["expert_snapshot"])
        parser = QuerySubjectParser(ROOT / ".cache/stanza")
        return cls(model, expert, parser, direction)

    @property
    def config(self):
        return copy.deepcopy(self._config)

    def predict(self, frames, ids, metadata):
        validate_input(frames, ids, metadata)
        # The full-model reinsertion helper uses temporary hooks. Concurrent
        # episodes on this predictor would violate the per-sample state contract.
        if not self._episode_lock.acquire(blocking=False):
            raise RuntimeError("Concurrent episodes on one student are not supported")
        try:
            return self._predict(frames, list(map(int, ids)), dict(metadata))
        finally:
            self._episode_lock.release()

    def _predict(self, frames, ids, metadata):
        from scripts.audit_parametric_reinsertion_v1 import full_prediction
        from scripts.run_parametric_observation_v1 import expert_observations
        from vg_tta.decota_spatial_extension_v1 import parse_context
        from vg_tta.decota_tastvg_episode_v1 import fitted_merge, make_batch
        from vg_tta.foreground_runtime import state_digest
        from vg_tta.parametric_observation_v1 import ObservationReplay, fit_student
        from vg_tta.shared_state_v1 import capture_shared
        from vg_tta.tg_spatial_tta_v1 import visual_query

        source_before, expert_before = state_digest(self.model), state_digest(self.expert.model)
        parses = dict(subject=self.parser(metadata["caption"])["subject"],
                      context=parse_context(self.parser, metadata["caption"]),
                      old=visual_query(self.parser, metadata["caption"]))
        batch = make_batch(frames, ids, metadata, parses["subject"], self.model)
        base16, _, records, _, _, views = capture_shared(self.model, batch)
        it = ObservationReplay(self.model, views, len(ids), "spatial")
        with torch.no_grad():
            zero = it.values()
        native = list(fitted_merge(zero["logits"], records, ids))
        native_boxes = zero["boxes"].cpu()
        # Keep the F34 FP32 native seed. Do not use the FP16 endpoint envelope.
        observations = expert_observations(self.expert, {"parses": parses}, frames,
                                           ids, native, include_controls=False)
        anchors = copy.deepcopy(observations["anchors"]["weak"])
        c = self._config
        expected_names = c["trainable_parameters"]
        if {n: p.numel() for n, p in it.named} != expected_names:
            raise RuntimeError("Actual trainable spatial interface differs from F34")
        if set(it.groups.values()) != {"spatial"}:
            raise RuntimeError("Unexpected temporal/shared optimization group")
        result = fit_student(it, anchors, None, records, ids, kind="spatial",
                             steps=c["steps"], lrs={"spatial": c["lr"]},
                             planned=c["planned_anchor_denominator"], kappa=c["kappa"],
                             lambda_s=c["lambda_s"], gamma=c["gamma"], old_mean=False)
        if result["anchors"] != observations["anchors"]["weak"]:
            raise RuntimeError("Teacher targets changed during the episode")
        if any(not torch.equal(a.cpu(), b) for a, b in
               zip(zero["logits"], result["final"]["logits"])):
            raise RuntimeError("Spatial-only adaptation changed temporal logits")
        # This is a real full-video student forward at the saved best state.
        # expected is only an equality assertion, never the returned prediction.
        live = full_prediction(self.model, frames, ids, metadata, parses["subject"],
                               result["state"], result["final"])
        if live["indices"] != native:
            raise RuntimeError("Native temporal output changed")
        if state_digest(self.model) != source_before or state_digest(self.expert.model) != expert_before:
            raise RuntimeError("A source model was not restored after adaptation")
        return dict(method="decota_spatial10_v1", direction=self.direction,
                    boxes=live["boxes"], indices=native, frame_ids=ids,
                    I_seed=list(native), I_out=list(native),
                    temporal_module="native_fp32_no_update", temporal_parameter_updates=False,
                    parameter_updates=result["state_delta"] > 0, adaptation=result,
                    expert=observations, parses=parses, native_boxes=native_boxes,
                    native_logits=[z.cpu() for z in zero["logits"]],
                    native_fp16_indices=list(base16["predicted_indices"]),
                    full_forward_audit=live["audit"], source_restored=True,
                    expert_frozen=True, GT_online=False, student_output_only=True,
                    coverage_coefficient=0, interpolation_output=False)

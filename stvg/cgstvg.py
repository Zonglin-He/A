"""CG-STVG transfer adapter preparation; independent of the TA native capture.

The official checkpoints are hash-bound to their source domains. Import the
official top-level ``models`` package only in a fresh worker process. This file
does not replace the registered TA predictor or introduce another policy.
Real-checkpoint GPU zero-state/capture equivalence is a required release gate.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, ModuleType
import argparse
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external/CGSTVG"
REVISION = "f062d84b72aee8be4425a4a5ab20aa5b077ca142"
HUB_REVISION = "9b0d7b9680c23ae14071ff6ca89f63f6a0990b2d"
CHECKPOINTS = {
    "hc2": ("hcstvg2.pth", "020d77b1013e33d9ae58f01c5b60e77ba6eb3d05e04077a94fa8e4247d635a88"),
    "vidstg": ("vidstg.pth", "cbf3ee4036dab88de3314dc3a520ab0eb2b4d292a2796d12d0f3493c39eed9ef"),
}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 2**20), b""):
            h.update(b)
    return h.hexdigest()


def source_for(target):
    if target not in CHECKPOINTS:
        raise ValueError("Expected target hc2 or vidstg")
    return "vidstg" if target == "hc2" else "hc2"


def checkpoint_for(target):
    source = source_for(target)
    filename, expected = CHECKPOINTS[source]
    path = ROOT / "checkpoints/cgstvg" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required cross-domain CG checkpoint missing: {path}")
    if file_hash(path) != expected:
        raise ValueError(f"CG checkpoint differs from official source-domain hash: {filename}")
    return path, source


def label_free_forward(original):
    """Remove precisely the upstream dead GT read, not its learned inference.

    The caller guards eval mode and iteration_rate < 0. The original training
    function remains intact and is restored after the scoped call.
    """
    source = textwrap.dedent(inspect.getsource(original))
    dead = "        target_boxes, gt_bbox_slice = self.gt_info(targets)\n"
    replacement = "        target_boxes, gt_bbox_slice = None, None  # inference-only dead GT read removed\n"
    if source.count(dead) != 1 or source.count("if iteration_rate >= 0:") != 1:
        raise RuntimeError("Unrecognized official CG PosDecoder; refusing a loose patch")
    modified = source.replace(dead, replacement)
    assert modified.replace(replacement, dead) == source
    namespace = dict(original.__globals__)
    exec(compile(modified, str(VENDOR / "models/grounding_model/query_decoder.py"), "exec"), namespace)
    return namespace[original.__name__], {
        "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "inference_sha256": hashlib.sha256(modified.encode()).hexdigest(),
        "only_dead_gt_read_removed": True,
    }


@contextmanager
def inference_only(decoder):
    """Instance-scoped guard; targets can be an object forbidding every read."""
    original = type(decoder).forward
    if "forward" in decoder.__dict__:
        raise RuntimeError("CG decoder already has an instance forward override")
    patched, audit = label_free_forward(original)
    signature = inspect.signature(original)

    def checked(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        rate = bound.arguments.get("iteration_rate")
        if self.training or rate is None or rate >= 0:
            raise RuntimeError("The no-GT wrapper is restricted to eval iteration_rate < 0")
        return patched(self, *args, **kwargs)

    decoder.forward = MethodType(checked, decoder)
    try:
        yield audit
    finally:
        del decoder.forward
        assert type(decoder).forward is original


def host_roi_function(original):
    """Batch only integer ROI-coordinate synchronization; arithmetic unchanged."""
    source = textwrap.dedent(inspect.getsource(original))
    # The frozen upstream line carries a trailing explanatory comment.
    coordinate_line = next((line for line in source.splitlines(keepends=True)
                            if line.startswith("    bboxs = torch.stack(") and ").long()" in line), None)
    unpack = "        x1, y1, x2, y2 = bboxs[i]\n"
    insertion = "    _roi_coordinates = bboxs.detach().cpu().tolist()\n"
    if coordinate_line is None or source.count(unpack) != 1 or source.count("bboxs = torch.stack(") != 1:
        raise RuntimeError("Unrecognized CG ROI function; refusing a loose synchronization patch")
    changed = source.replace(coordinate_line, coordinate_line + insertion).replace(
        unpack, "        x1, y1, x2, y2 = _roi_coordinates[i]\n")
    recovered = changed.replace(insertion, "").replace(
        "        x1, y1, x2, y2 = _roi_coordinates[i]\n", unpack)
    assert recovered == source
    namespace = dict(original.__globals__)
    exec(compile(changed, __file__, "exec"), namespace)
    return namespace[original.__name__], dict(
        applied=True, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        synchronized_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        coordinate_sync="one detached integer CPU list after round/ceil/long",
        ROI_clone_slice_mean_unchanged=True)


@contextmanager
def roi_coordinates_on_host(decoder):
    """Scope the real official method; synthetic replacement functions stay as-is."""
    bound = decoder.generate_roi_feature
    original = getattr(bound, "__func__", None)
    native = VENDOR / "models/grounding_model/query_decoder.py"
    if original is None or Path(original.__code__.co_filename).resolve() != native.resolve():
        yield dict(applied=False, reason="not_the_official_CG_ROI_function")
        return
    if "generate_roi_feature" in decoder.__dict__:
        raise RuntimeError("Official CG ROI function already has an instance override")
    patched, audit = host_roi_function(original)
    decoder.generate_roi_feature = MethodType(patched, decoder)
    try:
        yield audit
    finally:
        del decoder.generate_roi_feature
        assert decoder.generate_roi_feature.__func__ is original


def select_boxes(output):
    """CG returns a three-element list, unlike the TA tensor return."""
    if not isinstance(output, (tuple, list)) or len(output) != 3:
        raise ValueError("Expected CG [layer anchors, confidences, fake anchors]")
    anchors = output[0]
    if anchors.ndim != 4 or anchors.shape[1] != 1 or anchors.shape[-1] != 4:
        raise ValueError("Expected CG anchors [layers,1,T,4]")
    return anchors.flatten(1, 2)[-1]


def strong_distribution(records, native_layers, hires_layers):
    """Use this CG model's own six layers and its own high-resolution output."""
    from stvg.temporal import probabilities, mass_table, final_mbr
    if len(native_layers) != 2 or len(hires_layers) != 2:
        raise ValueError("Expected the two physical offsets")
    if any(x.ndim != 4 or x.shape[0] != 6 or x.shape[1] != 1 or x.shape[-1] != 2
           for x in [*native_layers, *hires_layers]):
        raise ValueError("Expected six CG temporal layers [6,1,T,2]")
    base = [mass_table(probabilities(records, [x[k] for x in native_layers])) for k in range(6)]
    hu, hq = mass_table(probabilities(records, [x[-1] for x in hires_layers]))
    if any(u != hu for u, _ in base):
        raise ValueError("CG native/high-resolution physical supports differ")
    q = .5 * hq + .5 * np.mean([x[1] for x in base], axis=0)
    return dict(frame_ids=hu, posterior=q, interval=final_mbr(hu, q)["interval"])


def frozen_prefix_forward(original):
    """CG-native operations with TA's precision boundary and host staging.

    The image/motion/multimodal prefix executes under the caller's fp16
    autocast. The actual CG decoder and heads execute fp32. Visual arithmetic
    remains the original CG code; only its tensor residency is staged around
    the memory-heavy Video-Swin call.
    """
    from stvg.capture import transfer
    from stvg.tensors import floating32
    import gc
    source = textwrap.dedent(inspect.getsource(original))
    motion = "    vid_features = self.vid(videos.tensors, len(videos.tensors))\n"
    decoding = "    outputs_pos, outputs_time = self.ground_decoder(encoded_info=encoded_info, vis_pos=vis_pos_embed, targets=targets)\n"
    if source.count(motion) != 1 or source.count(decoding) != 1:
        raise RuntimeError("Unrecognized CG pipeline; refusing memory/precision adaptation")
    staged = (
        "    assert not self.training and not _cg_torch.is_grad_enabled()\n"
        "    _visual_device = vis_features.device\n"
        "    vis_outputs = vis_outputs.to('cpu')\n"
        "    vis_pos_embed = _cg_transfer(vis_pos_embed, 'cpu')\n"
        "    del vis_features, vis_mask\n"
        "    _cg_gc.collect()\n"
        "    _cg_torch.cuda.empty_cache()\n"
        "    try:\n"
        "        vid_features = self.vid(videos.tensors, len(videos.tensors))\n"
        "    finally:\n"
        "        vis_outputs = vis_outputs.to(_visual_device)\n"
        "        vis_features, vis_mask, vis_durations = vis_outputs.decompose()\n"
        "        vis_pos_embed = _cg_transfer(vis_pos_embed, _visual_device)\n")
    staged_source = source.replace(motion, staged)
    assert staged_source.replace(staged, motion) == source
    before, after = staged_source.split(decoding)
    tail = decoding + after
    boundary = (
        "    with _cg_torch.autocast('cuda', enabled=False):\n"
        "        encoded_info = _cg_floating32(encoded_info)\n"
        "        vis_pos_embed = _cg_floating32(vis_pos_embed)\n")
    modified = before + boundary + textwrap.indent(tail, "    ")
    assert modified == before + boundary + textwrap.indent(tail, "    ")
    namespace = dict(original.__globals__, _cg_torch=torch, _cg_transfer=transfer,
                     _cg_floating32=floating32, _cg_gc=gc)
    exec(compile(modified, __file__, "exec"), namespace)
    return namespace[original.__name__], dict(
        native_pipeline_sha256=hashlib.sha256(source.encode()).hexdigest(),
        capture_pipeline_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        frozen_prefix="float16_autocast", decoder_and_heads="float32",
        visual_tensor_host_staging=True)


@contextmanager
def frozen_capture(model):
    """Apply storage-only Swin wrappers plus the explicit precision boundary."""
    from stvg.source import transient_native_masks
    from stvg._attention import frozen_attention_storage
    from stvg._host import inactive_modules_on_host
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("CG prefix must be frozen in eval mode")
    if "forward" in model.__dict__:
        raise RuntimeError("CG model already has an instance forward override")
    forward, audit = frozen_prefix_forward(type(model).forward)
    model.forward = MethodType(forward, model)
    try:
        with transient_native_masks(model), frozen_attention_storage(model), inactive_modules_on_host(model):
            yield audit
    finally:
        del model.forward


@torch.no_grad()
def capture_offsets(model, frames, ids, metadata, height=224):
    """Capture the real CG decoder interfaces, without TA query reconstruction.

    The frozen prefix uses fp16 autocast, followed by the real CG fp32 decoder
    and heads. Compatibility must be checked on each loaded model before any
    formal evaluation. The unused RGB/target argument references
    are stripped from the saved replay kwargs; their only GT read is removed
    by the narrowly scoped inference wrapper.
    """
    from stvg import inputs
    from stvg.capture import transfer
    from stvg.tensors import detached
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("CG capture requires a frozen eval Source")
    with inputs.resolution(height):
        batch = inputs.make_batch(frames, ids, metadata, model)
    device = next(model.parameters()).device
    batch["videos"] = batch["videos"].to("cpu")
    records, spatial, layers, outputs = [], [], [], []
    for offset in (0, 1):
        view = inputs.offset_batch(batch, offset)
        view["videos"] = view["videos"].to(device)
        captured, hidden = [], []

        def before(_module, args, kwargs):
            if args or kwargs.get("iteration_rate") != -1:
                raise RuntimeError("Unexpected CG decoder routing")
            # generate_roi_feature uses only height and width, never [2].
            kw = dict(kwargs)
            kw["targets"] = None
            kw["feature_size"] = (*kw["feature_size"][:2], None)
            captured.append(detached(kw, "cpu"))

        hooks = [model.ground_decoder.decoder.register_forward_pre_hook(before, with_kwargs=True),
                 model.ground_decoder.time_decoder.register_forward_hook(
                     lambda _m, _a, out: hidden.append(detached(out, "cpu")))]
        try:
            with frozen_capture(model), inference_only(model.ground_decoder.decoder), roi_coordinates_on_host(model.ground_decoder.decoder):
                with torch.autocast(device.type, dtype=torch.float16):
                    out = model(view["videos"], view["texts"], view["targets"], iteration_rate=-1)
        finally:
            for hook in hooks:
                hook.remove()
        if len(captured) != 1 or len(hidden) != 1:
            raise RuntimeError("Unexpected CG spatial/time decoder call count")
        layer_logits = model.temp_embed(hidden[0].to(device)).detach().cpu()
        if not torch.equal(layer_logits[-1], out["pred_sted"].detach().cpu()):
            raise RuntimeError("CG temporal capture does not reproduce native output")
        records.append(dict(offset=offset, flip=False, frame_ids=view["targets"][0]["frame_ids"]))
        spatial.extend(captured)
        layers.append(layer_logits)
        outputs.append(out["pred_boxes"].detach().cpu())
        del view, out
    source = torch.stack([outputs[p % 2][p // 2] for p in range(len(ids))]).float()
    return dict(Source=source, records=records, spatial_inputs=spatial, temporal_layers=layers)


def official_imports():
    """Fail rather than mix TA/CG packages named models/config/utils/datasets."""
    for name in ("models", "config", "utils", "datasets"):
        module = sys.modules.get(name)
        if module is not None:
            f = getattr(module, "__file__", None)
            if not f or not Path(f).resolve().is_relative_to(VENDOR.resolve()):
                raise RuntimeError(f"{name} already imported outside CG; use a fresh CG worker")
    sys.path.insert(0, str(VENDOR))
    # The frozen RoBERTa path does not use eagerly imported torchtext/LSTM.
    if "torchtext" not in sys.modules:
        shim = ModuleType("torchtext")
        shim.__path__ = []
        sys.modules["torchtext"] = shim
    from config import cfg
    from models import build_model
    return cfg, build_model


def load_source(target, device="cpu"):
    """Strict official EMA load; no target-shaped head or training execution."""
    path, source = checkpoint_for(target)
    cfg, build = official_imports()
    cfg = cfg.clone()
    cfg.merge_from_file(str(VENDOR / "experiments" / ("vidstg.yaml" if source == "vidstg" else "hcstvg2.yaml")))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.USE_LSTM = False
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(ROOT / "checkpoints/torch")
    runtime = ROOT / "runtime/cgstvg"
    if not (runtime / "model_zoo/roberta-base").is_dir():
        raise FileNotFoundError("Prepare CG model_zoo links with tools/setup_cgstvg.py first")
    previous = Path.cwd()
    original_empty = torch.cuda.empty_cache
    try:
        os.chdir(runtime)
        torch.cuda.empty_cache = lambda: None
        model, criterion, weights = build(cfg)
        del criterion, weights
    finally:
        os.chdir(previous)
        torch.cuda.empty_cache = original_empty
    raw = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    if "model_ema" not in raw:
        raise ValueError("Official CG checkpoint lacks model_ema")
    state = raw["model_ema"]
    expected = model.state_dict()
    extras = set(state) - set(expected)
    if extras - {"text_encoder.body.embeddings.position_ids"}:
        raise ValueError(f"Unexpected CG checkpoint keys: {sorted(extras)}")
    if set(expected) - set(state):
        raise ValueError("CG checkpoint misses learned/model state; refusing partial load")
    if any(state[k].shape != v.shape for k, v in expected.items()):
        raise ValueError("CG checkpoint shape mismatch; refusing target-shaped reconstruction")
    model.load_state_dict({k: state[k] for k in expected}, strict=True)
    audit = dict(source_dataset=source, target_dataset=target,
                 checkpoint_sha256=CHECKPOINTS[source][1], model_key="model_ema",
                 state_keys=len(state), ignored_deterministic_extras=sorted(extras),
                 all_model_weights_loaded=True, pretrained_weights_frozen=True)
    del raw, state
    return model.to(device).eval().requires_grad_(False), audit


class SpatialReplay:
    """Same 8x256 temporary state and existing policy, CG-native decoder replay."""
    def __init__(self, model, inputs, source, E, positions, check=lambda: None, device="cuda"):
        from stvg.capture import transfer
        from stvg.tensors import floating32
        self.decoder = copy.deepcopy(model.ground_decoder.decoder).to(device).eval().requires_grad_(False)
        self.inputs = floating32(transfer(inputs, device))
        self.source = torch.as_tensor(source, device=device, dtype=torch.float32)
        self.E = torch.as_tensor(E, device=device, dtype=torch.float64)
        self.positions = list(positions)
        if len(self.positions) != 8 or len(set(self.positions)) != 8 or len(self.inputs) != 2:
            raise ValueError("Expected eight distinct physical positions and two CG offsets")
        self.state = torch.zeros(8, 256, device=device, requires_grad=True)
        self.calls = self.backwards = 0
        self.seconds = 0.
        self.check = check
        with torch.no_grad():
            zero = self.values()
        if not torch.equal(zero["native"], self.source) or not torch.equal(zero["boxes"], self.E):
            raise RuntimeError("CG zero-state replay differs; do not begin adaptation")

    def values(self):
        import time
        from stvg.state import inject
        from stvg.geometry import corrected
        self.check()
        start = time.perf_counter()
        raw = []
        with inference_only(self.decoder), roi_coordinates_on_host(self.decoder), torch.autocast(self.state.device.type, enabled=False):
            for offset, original in enumerate(self.inputs):
                kw = dict(original)
                kw["query_tgt"] = inject(kw["query_tgt"], self.state, self.positions, offset)
                if kw.get("iteration_rate") != -1:
                    raise ValueError("CG cached replay is not the label-free inference path")
                raw.append(select_boxes(self.decoder(**kw)))
            native = torch.stack([raw[p % 2][p // 2] for p in range(len(self.source))]).float()
            boxes = corrected(native, self.source, self.E)
        if self.state.is_cuda:
            torch.cuda.synchronize()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return dict(native=native, boxes=boxes)


class Predictor:
    """CG-bound form of the frozen algorithm; requires real GPU regression.

    Numeric dataset configuration is copied, not searched or overwritten. The
    source path alone is rebound to the official CG checkpoint. Construct and
    use this object in its isolated worker; it is not the default TA predictor.
    """
    def __init__(self, dataset, check=lambda: None):
        config = json.loads((ROOT / "configs/legacy_opd.json").read_text())["datasets"][dataset]
        self.config = copy.deepcopy(config)
        self.config["backbone"] = "CG-STVG"
        self.config["source_checkpoint"] = "checkpoints/cgstvg/" + CHECKPOINTS[source_for(dataset)][0]
        self.config["temporal"] = "Strong CG-STVG only, original MBR"
        self.config["E_calibration_provenance"] = (
            "Unchanged TA-STVG source-domain calibration constants transferred to CG-STVG; "
            "not recalibrated on CG source training data; E_calibration.model_hash identifies TA-STVG")
        self.dataset = dataset
        self.source = None
        self.check = check

    @torch.no_grad()
    def capture(self, frames, frame_ids, metadata):
        import gc
        import time
        from stvg.temporal import probabilities, mass_table, final_mbr
        from stvg.readout import prediction
        from stvg.positions import observation_positions
        from stvg.initialization import initialize
        from stvg.predictor import output_writeback
        from stvg.teacher_bridge import refer_native
        self.check()
        begin = time.perf_counter()
        tick = begin
        if self.source is None:
            self.source, audit = load_source(self.dataset, "cpu")
            self.load_audit = audit
        load_seconds = time.perf_counter() - tick
        model = self.source.cuda()
        tick = time.perf_counter()
        native = capture_offsets(model, frames, frame_ids, metadata, height=224)
        torch.cuda.synchronize()
        native_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        high = capture_offsets(model, frames, frame_ids, metadata, height=448)
        torch.cuda.synchronize()
        high_seconds = time.perf_counter() - tick
        records = native["records"]
        logits = [x[-1] for x in native["temporal_layers"]]
        source = native["Source"]
        p = prediction(logits, source, records, list(frame_ids))
        positions = observation_positions(list(frame_ids), p["indices"])
        U, q = mass_table(probabilities(records, logits))
        strong = strong_distribution(records, native["temporal_layers"], high["temporal_layers"])
        if U != list(frame_ids) or high["records"] != records:
            raise RuntimeError("CG physical frame supports changed with resolution")
        del high
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        teacher = refer_native(frames[positions], metadata["caption"])
        tick = time.perf_counter()
        raw, initialization = initialize(source.double().numpy(), positions,
                                         teacher["boxes"].numpy(), self.config["E_calibration"])
        gamma = self.config["initialization_gamma"]
        E = output_writeback(source.double().numpy(), raw, gamma)
        initialization = dict(initialization, initialization_gamma=gamma,
                              unscaled_delta=initialization["delta"],
                              delta=(gamma * np.asarray(initialization["delta"])).tolist())
        cost = dict(Source_load_seconds=load_seconds, Source224_seconds=native_seconds,
                    Source448_seconds=high_seconds, ERef_seconds=time.perf_counter() - tick,
                    ReferDINO=teacher["cost"], ReferDINO_load_seconds=teacher["model_load_seconds"],
                    ReferDINO_process_seconds=teacher["process_wall_seconds"],
                    ReferDINO_query_forwards=1, ReferDINO_frames=8,
                    independent_GroundingDINO_forwards=0, expert_instances=1,
                    peak_GPU_bytes=max(torch.cuda.max_memory_allocated(), teacher["peak_GPU_bytes"]),
                    capture_wall_seconds=time.perf_counter() - begin)
        return dict(Source=source, E=E, frame_ids=U, planned_positions=positions,
                    spatial_inputs=native["spatial_inputs"], teacher_native=teacher,
                    initialization=initialization,
                    intervals=dict(Native=final_mbr(U, q)["interval"], Strong=strong["interval"]),
                    posterior=dict(Native=q, Strong=strong["posterior"]), cost=cost,
                    GT_read=False, load_audit=self.load_audit)

    def adapt(self, cache, key):
        import gc
        from stvg.policy import fit
        from stvg.initialization import evidence_hash
        from stvg.predictor import output_writeback
        if self.source is None:
            self.source, self.load_audit = load_source(self.dataset, "cpu")
        replay = SpatialReplay(self.source, cache["spatial_inputs"], cache["Source"], cache["E"],
                               cache["planned_positions"], self.check)
        try:
            out = fit(replay, cache["teacher_native"]["boxes"], key, "Live", 41, self.config)
            if evidence_hash(out["target"].numpy()) != cache["initialization"]["evidence_sha256"]:
                raise RuntimeError("CG initialization and OPD did not reuse the same expert evidence")
            raw = out["final"].numpy()
            final = output_writeback(cache["E"], raw, self.config["output_alpha"])
            return dict(fit=out, Source=np.asarray(cache["Source"]), E_Ref=cache["E"],
                        Live_raw=raw, Live=final, initialization=cache["initialization"],
                        intervals=cache["intervals"], posterior=cache["posterior"],
                        output_alpha=self.config["output_alpha"],
                        cost=dict(actual_updates=out["actual_updates"], fit_seconds=out["seconds"],
                                  decoder_calls=replay.calls), GT_read=False)
        finally:
            del replay
            gc.collect()
            torch.cuda.empty_cache()

    def predict(self, frames, frame_ids, metadata, key):
        cache = self.capture(frames, frame_ids, metadata)
        out = self.adapt(cache, key)
        return dict(out, boxes=out["Live"], interval=cache["intervals"]["Strong"],
                    capture_cost=cache["cost"])


def run_worker(target, output):
    """CPU strict-load audit in a fresh interpreter, never in the TA process."""
    command = [sys.executable, "-B", "-m", "stvg.cgstvg", "--target", target, "--load-audit", str(output)]
    return subprocess.run(command, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description="Isolated CG-STVG strict load or frozen prediction")
    p.add_argument("--target", choices=list(CHECKPOINTS), required=True)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--load-audit")
    mode.add_argument("--request", help="Label-free existing video/text request")
    p.add_argument("--output", help="Prediction .npz; required with --request")
    args = p.parse_args()
    if args.request:
        import time
        from stvg.runtime import setup, lease
        from stvg.decode import decode
        from stvg.inputs import validate_input
        if not args.output:
            p.error("--request requires --output")
        request = json.loads(Path(args.request).read_text())
        if set(request) - {"dataset", "key", "metadata"} or request["dataset"] != args.target:
            raise ValueError("CG target/request fields differ")
        setup()
        lock = lease()
        begin = time.perf_counter()
        try:
            frames, ids = decode(request["metadata"], args.target)
            validate_input(frames, ids, request["metadata"])
            predictor = Predictor(args.target)
            out = predictor.predict(frames, ids, request["metadata"], request.get("key", "single-query"))
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.savez(destination, Source=out["Source"], E_Ref=out["E_Ref"],
                     Live_raw=out["Live_raw"], Live=out["Live"], frame_ids=np.asarray(ids),
                     positions8=np.asarray(out["fit"]["positions"]),
                     Native_interval=out["intervals"]["Native"], Strong_interval=out["intervals"]["Strong"])
            audit = dict(status="CG_prediction_completed", dataset=args.target, key=request.get("key", "single-query"),
                         configuration=predictor.config, load_audit=predictor.load_audit,
                         prediction_sha256=file_hash(destination), input_sha256=file_hash(args.request),
                         expert_count=1, labels_read=False, zero_state_GPU_equivalence_verified=True,
                         total_seconds=time.perf_counter() - begin, capture=out["capture_cost"], adapt=out["cost"])
            destination.with_suffix(".json").write_text(json.dumps(audit, indent=2) + "\n")
        finally:
            lock.close()
        return
    model, audit = load_source(args.target, device="cpu")
    audit.update(status="CPU_strict_checkpoint_loaded", GPU_forward_executed=False,
                 zero_state_GPU_equivalence_verified=False, formal_evaluation_completed=False)
    Path(args.load_audit).parent.mkdir(parents=True, exist_ok=True)
    Path(args.load_audit).write_text(json.dumps(audit, indent=2) + "\n")
    del model


if __name__ == "__main__":
    main()

"""TubeDETR interface for the frozen single-ReferDINO OPD method.

The upstream model is unchanged. We capture its actual decoder arguments and
inject eight content vectors at the matching physical-frame positions. The
encoder and the network parameters never receive gradients.
"""
from contextlib import contextmanager
import copy
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.special import logsumexp

from .runtime import ROOT, sha
from .tensors import detached
from .geometry import corrected
from .initialization import initialize, evidence_hash
from .positions import observation_positions
from .policy import fit
from .temporal import final_mbr


def official_imports():
    root = ROOT / 'external/TubeDETR'
    if 'models' in sys.modules:
        if not Path(sys.modules['models'].__file__).resolve().is_relative_to(root):
            raise RuntimeError('Use a separate process for each backbone/teacher import namespace')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module('models')


@contextmanager
def checkpoint_only_construction(roberta_directory):
    """Avoid unrelated initialization downloads; strict restore follows construction."""
    import torchvision.models
    from transformers import RobertaConfig, RobertaModel
    original_resnet = torchvision.models.resnet101
    original_roberta = RobertaModel.from_pretrained

    def resnet(*args, **kwargs):
        kwargs.pop('pretrained', None)
        kwargs['weights'] = None
        return original_resnet(*args, **kwargs)

    def roberta(*args, **kwargs):
        config = RobertaConfig.from_pretrained(str(roberta_directory), local_files_only=True)
        config._attn_implementation = 'eager'
        return RobertaModel(config)

    torchvision.models.resnet101 = resnet
    RobertaModel.from_pretrained = roberta
    try:
        yield
    finally:
        torchvision.models.resnet101 = original_resnet
        RobertaModel.from_pretrained = original_roberta


def load_source(config, device='cuda'):
    upstream = official_imports()
    checkpoint_path = ROOT / config['source_checkpoint']
    checkpoint_sha256 = sha(checkpoint_path)
    if checkpoint_sha256 != config['source_checkpoint_sha256']:
        raise ValueError('TubeDETR source checkpoint differs from its recorded official fingerprint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = checkpoint.get('args')
    if saved is None:
        raise ValueError('Official checkpoint must contain its architecture arguments')
    args = SimpleNamespace(**copy.deepcopy(saved if isinstance(saved, dict) else vars(saved)))
    # Released checkpoints predate the upstream argument rename. Their
    # pool_size=2 is precisely the current stride=2, not a new hyperparameter.
    legacy_pool_size = getattr(args, 'pool_size', None)
    if not hasattr(args, 'stride'):
        if legacy_pool_size is None:
            raise ValueError('Checkpoint has neither stride nor legacy pool_size')
        args.stride = legacy_pool_size
    if args.num_queries != 1 or args.hidden_dim != 256 or args.dec_layers != 6:
        raise ValueError('Unsupported TubeDETR decoder architecture')
    if args.stride != config['stride'] or args.resolution != config['native_resolution']:
        raise ValueError('Checkpoint/configuration stride or resolution mismatch')
    args.device = device
    args.aux_loss = True  # read all existing heads; no new learned parameters
    args.text_encoder_type = str(ROOT / 'runtime/tubedetr/roberta-base')
    with checkpoint_only_construction(args.text_encoder_type):
        model, criterion, _ = upstream.build_model(args)
    del criterion
    state_key = 'model_ema' if checkpoint.get('model_ema') is not None else 'model'
    state = dict(checkpoint[state_key])
    # The official evaluation also regenerates this deterministic time table.
    state.pop('transformer.time_embed.te', None)
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {'transformer.time_embed.te'}
    allowed_unexpected = {'transformer.text_encoder.embeddings.position_ids',
                          'transformer.text_encoder.embeddings.token_type_ids'}
    learned = set(dict(model.named_parameters()))
    if (set(incompatible.missing_keys) - allowed_missing
            or set(incompatible.unexpected_keys) - allowed_unexpected
            or set(incompatible.missing_keys) & learned):
        raise RuntimeError(f'Incomplete checkpoint restore: {incompatible}')
    model.eval().requires_grad_(False).to(device)
    model.load_audit = dict(checkpoint_sha256=checkpoint_sha256, state_key=state_key,
                            missing=incompatible.missing_keys,
                            unexpected=incompatible.unexpected_keys,
                            all_learned_parameters_covered=True,
                            legacy_pool_size=legacy_pool_size,
                            stride=args.stride, resolution=args.resolution,
                            upstream_revision=config['upstream_revision'])
    return model


def endpoint_posterior(logits):
    """Single TubeDETR full-grid endpoint posterior; no artificial offset views."""
    z = torch.as_tensor(logits).detach().cpu().numpy()
    if z.ndim != 3 or z.shape[0] != 1 or z.shape[-1] != 2 or z.dtype != np.float32:
        raise ValueError('Expected native FP32 [1,T,2] logits')
    if not np.isfinite(z).all():
        raise ValueError('Nonfinite temporal output')
    v = z[0].astype(np.float64)
    i, j = np.triu_indices(len(v), 1)
    scores = v[i, 0] + v[j, 1]
    q = np.zeros((len(v), len(v)), dtype=np.float64)
    q[i, j] = np.exp(scores - logsumexp(scores))
    return q


@torch.no_grad()
def native_forward(model, frames, frame_ids, caption, resolution=224):
    official_imports()
    from datasets.video_transforms import make_video_transforms
    from util.misc import NestedTensor
    from models.postprocessors import PostProcessSTVG
    if len(frames) != len(frame_ids) or len(frames) < 8:
        raise ValueError('Eight or more real frames are required')
    # Exact upstream spatial preprocessing, with empty targets and no annotation loader.
    transform = make_video_transforms('test', cautious=True, resolution=resolution)
    x, _ = transform(frames, None)
    device = next(model.parameters()).device
    fast = NestedTensor.from_tensor_list([x], False).to(device)
    slow = NestedTensor.from_tensor_list([x[:, ::model.stride]], False).to(device) if model.stride else fast
    fast_arg = fast if model.fast else None
    captured = []

    def capture(module, args, kwargs):
        captured.append((detached(args), detached(kwargs)))

    memory = model(slow, [len(frames)], [caption], encode_and_save=True, samples_fast=fast_arg)
    hook = model.transformer.decoder.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        output = model(slow, [len(frames)], [caption], encode_and_save=False, memory_cache=memory)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError('Expected one native full-grid decoder call')
    # Source uses the upstream MAP postprocessor, including its tie ordering.
    interval = PostProcessSTVG()(output, [list(frame_ids)], [0])[0]
    interval = list(map(int, interval))
    extent = [list(frame_ids).index(interval[0]), list(frame_ids).index(interval[1] - 1)]
    layers = [x['pred_sted'].float().cpu() for x in output['aux_outputs']] + [output['pred_sted'].float().cpu()]
    return dict(source=output['pred_boxes'].float().cpu(), logits=output['pred_sted'].float().cpu(),
                layers=layers, decoder_inputs=captured[0], frame_ids=list(frame_ids),
                interval=interval, extent=extent)


def inject_content(tgt, state, positions):
    if tgt.ndim != 3 or tgt.shape[1:] != (1, 256):
        raise ValueError('Expected TubeDETR full-grid [T,1,256] content')
    if state.shape != (8, 256) or len(set(positions)) != 8:
        raise ValueError('Expected eight independent content states')
    index = torch.as_tensor(positions, device=tgt.device)
    return tgt + torch.zeros_like(tgt).index_add(0, index, state[:, None])


class TubeReplay:
    def __init__(self, model, cache, E, positions, check=lambda: None):
        # Reuse immutable modules instead of copying a decoder every query.
        self.decoder = model.transformer.decoder
        self.head = model.bbox_embed
        self.args, self.kwargs = cache['decoder_inputs']
        self.source = cache['source'].to(next(self.decoder.parameters()).device)
        self.E = torch.as_tensor(E, dtype=torch.float64, device=self.source.device)
        self.positions = list(positions)
        self.state = torch.zeros(8, 256, device=self.source.device, requires_grad=True)
        self.check = check
        self.calls = self.backwards = 0
        self.seconds = 0.
        with torch.no_grad():
            zero = self.values()
        if not torch.equal(zero['native'], self.source) or not torch.equal(zero['boxes'], self.E):
            raise RuntimeError('Zero content replay does not exactly restore native output/initialization')

    def values(self):
        self.check()
        start = time.perf_counter()
        args = list(self.args)
        args[0] = inject_content(args[0], self.state, self.positions)
        hidden = self.decoder(*args, **self.kwargs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        native = self.head(hidden.transpose(1, 2).flatten(1, 2)).sigmoid()[-1].float()
        boxes = corrected(native, self.source, self.E)
        if boxes.device.type == 'cuda':
            torch.cuda.synchronize()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        if not torch.isfinite(boxes).all() or not (boxes[:, 2:] > 0).all():
            raise ValueError('Invalid decoder output')
        return dict(native=native, boxes=boxes)


class TubePredictor:
    def __init__(self, dataset, check=lambda: None, teacher=None, model=None):
        base = json.loads((ROOT / 'configs/default.json').read_text())
        adapter = json.loads((ROOT / 'configs/tubedetr.json').read_text())
        self.config = {**base['datasets'][dataset], **adapter['common'], **adapter['datasets'][dataset]}
        self.model = model
        self.teacher = teacher
        self.check = check

    def predict(self, frames, frame_ids, metadata, key):
        from .predictor import output_writeback
        from .teacher_bridge import refer_native
        start = time.perf_counter()
        load_tick = time.perf_counter()
        if self.model is None:
            self.model = load_source(self.config)
        source_load_seconds = time.perf_counter()-load_tick
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        native = native_forward(self.model, frames, frame_ids, metadata['caption'], self.config['native_resolution'])
        torch.cuda.synchronize()
        source_seconds = time.perf_counter() - tick
        native_peak_bytes = torch.cuda.max_memory_allocated()
        positions = observation_positions(frame_ids, native['extent'])
        tick = time.perf_counter()
        hr = native_forward(self.model, frames, frame_ids, metadata['caption'], self.config['strong_resolution'])
        strong = .5 * endpoint_posterior(hr['logits']) + .5 * np.mean([endpoint_posterior(z) for z in native['layers']], axis=0)
        interval = final_mbr(frame_ids, strong)['interval']
        del hr
        torch.cuda.synchronize()
        strong_seconds = time.perf_counter() - tick
        # The teacher owns a separate CUDA allocator. Release the completed
        # high-resolution forward's unused cache before handing it the GPU.
        torch.cuda.empty_cache()
        source_idle_bytes = torch.cuda.memory_allocated()
        teacher = (self.teacher or refer_native)(frames[positions], metadata['caption'])
        raw, init = initialize(native['source'].double().numpy(), positions, np.asarray(teacher['boxes']), self.config['E_calibration'])
        E = output_writeback(native['source'].numpy(), raw, self.config['initialization_gamma'])
        replay = TubeReplay(self.model, native, E, positions, self.check)
        result = fit(replay, teacher['boxes'], key, 'Live', 41, self.config)
        if evidence_hash(result['target'].numpy()) != init['evidence_sha256']:
            raise RuntimeError('Initialization and OPD evidence differ')
        # fit resets in finally; check both content reset and native restoration.
        with torch.no_grad():
            reset = replay.values()
        if replay.state.count_nonzero() or not torch.equal(reset['native'], native['source'].to(reset['native'])):
            raise RuntimeError('Per-query reset failed')
        live = output_writeback(E, result['final'].numpy(), self.config['output_alpha'])
        cost = dict(source_seconds=source_seconds, source_load_seconds=source_load_seconds, strong_seconds=strong_seconds,
                    adaptation_seconds=result['seconds'], teacher=teacher['cost'],
                    teacher_load_seconds=teacher['model_load_seconds'],
                    total_seconds=time.perf_counter()-start,
                    source_peak_bytes=native_peak_bytes,
                    teacher_peak_bytes=teacher['peak_GPU_bytes'],
                    peak_combined_allocated_bytes=max(torch.cuda.max_memory_allocated(), source_idle_bytes+teacher['peak_GPU_bytes']),
                    actual_updates=result['actual_updates'], decoder_calls=replay.calls)
        return dict(Source=native['source'].numpy(), E_Ref=E, Live_raw=result['final'].numpy(), Live=live,
                    frame_ids=frame_ids, positions8=positions, Native_interval=native['interval'],
                    Strong_interval=interval, cost=cost, configuration=self.config,
                    checkpoint=self.model.load_audit, initialization=init, GT_read=False)

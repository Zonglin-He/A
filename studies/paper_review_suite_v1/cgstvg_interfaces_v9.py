"""CG-native thin interfaces; no model is constructed by importing this module.

CG has one native frame grid. These adapters never request TA offsets, hidden
layers, word classifiers or higher-resolution passes. Runtime execution is a
separate last-stage gate after all TA studies have sealed and scored.
"""
from contextlib import contextmanager
import hashlib
import inspect
import textwrap
from types import MethodType

import numpy as np
from scipy.special import logsumexp


def frame_grid(metadata, dataset):
    """Official TEST sampling from input metadata, without dataset/label loaders."""
    if dataset == 'hc2':
        count = int(metadata['frame_count'])
        frames = list(range(count - 1))  # official preprocess excludes count-1
        rate = 6.4 / (count / 20.)      # TEST doubles official SAMPLE_FPS=3.2
        selected = [0]
        for i, frame in enumerate(frames):
            if int(frames[selected[-1]] * rate) < int(frame * rate):
                selected.append(i)
        if selected[-1] != len(frames)-1:
            selected.append(len(frames)-1)
        ids = [frames[i] for i in selected]
    elif dataset == 'vidstg':
        count = int(metadata['frame_count'])
        begin = int(metadata['start_frame'])
        end = min(int(metadata['end_frame']), count-1)
        assert 0 <= begin <= end < count
        # Frozen input manifest: source segment end may be last-valid or one-past.
        # Convert to the official inclusive segment BEFORE native sampling.
        frames = list(range(begin, end+1))
        # Official TEST doubles TRAIN_SAMPLE_NUM=64; no temporal GT cropping.
        ids = frames if len(frames) <= 128 else [frames[int(i)] for i in np.linspace(0, len(frames)-1, 128)]
    else:
        raise ValueError(dataset)
    assert len(ids) >= 8 and ids == sorted(set(ids))
    return ids


def last_layer_posterior(logits):
    """Single-grid legal endpoint posterior from the native final FP32 head."""
    if hasattr(logits, 'detach'):
        logits = logits.detach().cpu().numpy()
    z = np.asarray(logits)
    assert z.dtype == np.float32 and z.ndim == 3 and z.shape[0] == 1 and z.shape[2] == 2
    assert np.isfinite(z).all()
    n = z.shape[1]
    ij = np.triu_indices(n, 1)
    score = z[0, ij[0], 0].astype(np.float64) + z[0, ij[1], 1].astype(np.float64)
    q = np.zeros((n, n), np.float64)
    q[ij] = np.exp(score - logsumexp(score))
    assert abs(q.sum()-1) < 1e-12
    return q


def native_map_interval(logits, ids):
    """Exact official FP32 log-softmax/add/mask/flatten argmax operation order."""
    import torch
    import torch.nn.functional as F
    z = torch.as_tensor(logits)
    assert z.dtype == torch.float32 and z.shape == (1, len(ids), 2)
    n = len(ids)
    scores = (torch.ones(n, n, device=z.device) * -1e32).tril(0)
    scores += F.log_softmax(z[:, :, 0], dim=1)[0, :, None] + F.log_softmax(z[:, :, 1], dim=1)[0, None, :]
    k = int(scores.flatten().max(dim=0)[1])
    a, b = divmod(k, n)
    assert a < b
    return [int(ids[a]), int(ids[b])+1]


def pool_single_grid(projected, valid, boxes, context=1.5):
    """Same overlap-weighted ROI math as production, single CG time axis."""
    from stvg.representation import weights, pool
    assert context == 1.5, 'Frozen main context; no CG target parameter selection'
    if hasattr(projected, 'detach'):
        projected = projected.detach().cpu().numpy()
    if hasattr(valid, 'detach'):
        valid = valid.detach().cpu().numpy()
    x = np.asarray(projected)
    v = np.asarray(valid, bool)
    boxes = np.asarray(boxes, np.float64)
    assert x.ndim == 4 and v.shape == (x.shape[0], x.shape[2], x.shape[3])
    assert boxes.shape == (len(x), 4) and np.isfinite(x).all()
    assert (boxes[:, 2:] > 0).all()
    _, _, h, w = x.shape
    output = []
    for a, mask, box in zip(x, v, boxes):
        support, _ = weights(box, h, w, mask)
        output.append(pool(a.transpose(1, 2, 0), support))
    return np.asarray(output, np.float64)


def label_free_forward(original):
    """Bypass the official eval-dead gt_info call, leaving learned ops intact."""
    source = textwrap.dedent(inspect.getsource(original))
    dead = '        target_boxes, gt_bbox_slice = self.gt_info(targets)\n'
    replacement = '        target_boxes, gt_bbox_slice = None, None\n'
    assert source.count(dead) == 1 and source.count('if iteration_rate >= 0:') == 1
    modified = source.replace(dead, replacement)
    assert modified.replace(replacement, dead) == source
    namespace = dict(original.__globals__)
    exec(compile(modified, original.__code__.co_filename, 'exec'), namespace)
    return namespace[original.__name__], dict(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        eval_sha256=hashlib.sha256(modified.encode()).hexdigest(),
        change='only unused gt_info read; no GT input; eval iteration_rate<0 only')


@contextmanager
def inference_only(decoder):
    original = type(decoder).forward
    assert 'forward' not in decoder.__dict__
    patched, audit = label_free_forward(original)
    signature = inspect.signature(original)

    def checked(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        rate = bound.arguments.get('iteration_rate')
        assert not self.training and rate is not None and rate < 0
        return patched(self, *args, **kwargs)

    decoder.forward = MethodType(checked, decoder)
    try:
        yield audit
    finally:
        del decoder.forward
        assert type(decoder).forward is original


class ForbidLabels:
    def __getitem__(self, key):
        raise AssertionError('Inference attempted to read targets')

    def __iter__(self):
        raise AssertionError('Inference attempted to iterate targets')

    def __len__(self):
        raise AssertionError('Inference attempted to measure targets')

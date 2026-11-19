"""Fit two matched subject heads on ordinary source recordings only."""

import argparse
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ranker182 import HERE, WIDTH, K, SubjectRanker, choose, protocol
from capture182 import FIT_FEATURES, archive_block, capture_fingerprint, source_expected
from scope import PLATFORMS, file_sha, read, save
from train import metrics, prepare_device

TRAINED = HERE / 'rankers_PRIVATE'


def multi_positive(scores, positive):
    eligible = positive.any(-1)
    if not eligible.any():
        return scores.new_zeros(())
    s, good = scores[eligible], positive[eligible]
    return (s.logsumexp(-1) - s.masked_fill(~good, -torch.inf).logsumexp(-1)).mean()


def objective(scores, ious, hard, upweight):
    positive = ious > .25
    pairs = positive[:, :, None] & ~positive[:, None, :]
    weight = pairs.float() * (1 + (3 if upweight else 0) * hard[:, None, :].float())
    cost = F.softplus(scores[:, None, :] - scores[:, :, None])
    pair_loss = (cost * weight).sum((1, 2)) / weight.sum((1, 2)).clamp_min(1)
    return multi_positive(scores, positive) + .5 * multi_positive(scores, ious > .50) + .5 * pair_loss.mean()


def evidence_scores(model, block):
    return model(torch.from_numpy(block['features'])).cpu().numpy()


def hard_metrics(scores, ious, hard):
    good = ious > .25
    pair = good[:, :, None] & hard[:, None, :]
    eligible = pair.any(axis=(1, 2))
    count = int(pair.sum())
    if not count:
        return dict(rows=0, pairs=0, ordered_pairs=0, correct_top_competition=0)
    ranks = scores[:, :, None] - scores[:, None, :]
    best_good = np.where(good, scores, -np.inf).max(-1)
    best_hard = np.where(hard, scores, -np.inf).max(-1)
    return dict(rows=int(eligible.sum()), pairs=count, ordered_pairs=int(((ranks > 0) & pair).sum()),
                correct_top_competition=int((eligible & (best_good > best_hard)).sum()))


def score_heldout(block, output):
    idx = choose(output, block['incumbent'])
    val = block['ious'][np.arange(len(idx)), idx]
    return dict(**metrics(val), switches=int((idx != block['incumbent']).sum()),
                competing=hard_metrics(output, block['ious'], block['hard']))


def fit_one(seed, initial, heldout, use_hard):
    cfg = protocol()['ranker']
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = SubjectRanker()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    x = torch.from_numpy(initial['features'])
    ious = torch.from_numpy(initial['ious'])
    hard = torch.from_numpy(initial['hard'])
    independent = torch.from_numpy(initial['scores'][:, :, 1])
    eligible = torch.where((ious > .25).any(-1))[0]
    assert len(eligible) > 0 and x.shape[1:] == (K, WIDTH)
    assert not (hard & (ious > .25)).any()
    initial_loss = float(objective(independent[eligible], ious[eligible], hard[eligible], use_hard))
    best, checkpoints, history = None, [], []
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        total = 0.
        order = eligible[torch.randperm(len(eligible), generator=rng)]
        for idx in order.split(cfg['batch_size']):
            values = independent[idx] + model(x[idx])
            loss = objective(values, ious[idx], hard[idx], use_hard)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite source ranking loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        history.append(total / len(eligible))
        if epoch not in cfg['checkpoint_epochs']:
            continue
        model.eval()
        with torch.inference_mode():
            residual = evidence_scores(model, heldout)
        score = heldout['scores'][:, :, 0] + residual
        stat = score_heldout(heldout, score)
        stat['epoch'] = epoch
        checkpoints.append(stat)
        key = (stat['Acc25'], stat['Acc50'], stat['mIoU'], -epoch)
        if best is None or key > best[0]:
            best = key, {name: value.detach().clone() for name, value in model.state_dict().items()}, stat
    return best[1], dict(selected=best[2], checkpoints=checkpoints, epoch_losses=history,
                         initial_loss=initial_loss, fit_eligible=int(len(eligible)),
                         trained_parameters=sum(p.numel() for p in model.parameters()))


def fit(smoke=False):
    directory = HERE / 'source_smoke_PRIVATE' if smoke else FIT_FEATURES
    seal = read(directory / 'SEALED.json')
    assert seal['n'] == (24 if smoke else source_expected()) and seal['source_only']
    assert seal['implementation'] == capture_fingerprint()
    assert not seal['target_labels_read'] and not seal['FINAL_access']
    target = HERE / 'rank_smoke_PRIVATE' if smoke else TRAINED
    target.mkdir(mode=0o700)
    reports, start = {}, perf_counter()
    cfg = protocol()['ranker']
    for i, source in enumerate(PLATFORMS):
        fit_data = archive_block(directory, f'{source}_fit.npz', seal)
        heldout = archive_block(directory, f'{source}_calibration.npz', seal)
        reports[source] = dict(
            samples=seal['sources'][f'{source}_fit.npz'], heldout=seal['sources'][f'{source}_calibration.npz'],
            v1=metrics(heldout['ious'][np.arange(len(heldout['ious'])), heldout['incumbent']]),
            frozen=score_heldout(heldout, heldout['scores'][:, :, 0]), models={})
        for name in ('hard', 'generic'):
            params, score = fit_one(cfg['seed'] + i, fit_data, heldout, name == 'hard')
            path = target / f'{source}_{name}.pth'
            torch.save(dict(state=params, source=source, name=name), path)
            score['weights_sha256'] = file_sha(path)
            reports[source]['models'][name] = score
            print(source, name, 'epoch', score['selected']['epoch'], 'source Acc25/50',
                  round(score['selected']['Acc25'], 4), round(score['selected']['Acc50'], 4),
                  'seconds', round(perf_counter() - start, 1), flush=True)
    save(target / 'FITTED.json', dict(sources=reports, source_only=True,
         source_feature_seal=file_sha(directory / 'SEALED.json'), protocol_sha256=file_sha(HERE / 'PROTOCOL.json'),
         code_sha256=file_sha(HERE / 'fit182.py'), core_sha256=file_sha(HERE / 'ranker182.py'),
         FINAL_access=False, target_labels_read=False, elapsed_seconds=perf_counter() - start))


def fitted():
    value = read(TRAINED / 'FITTED.json')
    assert value['source_only'] and not value['FINAL_access'] and not value['target_labels_read']
    assert value['protocol_sha256'] == file_sha(HERE / 'PROTOCOL.json')
    assert value['code_sha256'] == file_sha(HERE / 'fit182.py')
    assert value['core_sha256'] == file_sha(HERE / 'ranker182.py')
    assert value['source_feature_seal'] == file_sha(FIT_FEATURES / 'SEALED.json')
    return value


def load_heads(source):
    registered = fitted()['sources'][source]['models']
    models = {}
    for name in ('hard', 'generic'):
        path = TRAINED / f'{source}_{name}.pth'
        assert file_sha(path) == registered[name]['weights_sha256']
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value['source'] == source and value['name'] == name
        model = SubjectRanker()
        model.load_state_dict(value['state'], strict=True)
        models[name] = model.eval().requires_grad_(False)
    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    prepare_device()
    fit(parser.parse_args().smoke)

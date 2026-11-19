"""Training on the sealed per-platform source cache, once per predefined arm."""
from collections import defaultdict
from pathlib import Path
import gc
import os
import time

import numpy as np
import torch

from common import ARMS, CACHE, PRIVATE, SPEC, PLATFORMS, json_atomic, read, sha, verify_inputs
from decoder import Decoder, in_boxes, objective
from evidence import FIELDS, source_features

TRAIN_FIELDS = (*FIELDS, 'in_image', 'gt', 'known_ref', 'side')


def load_split(source, split):
    proof = read(PRIVATE/source/'FEATURE_SEAL.json')
    n = proof['counts'][split]
    paths = sorted(p for p in proof['files'] if p.startswith(split+'_'))
    data, offset = {}, 0
    for name in paths:
        path = CACHE/source/name
        assert sha(path) == proof['files'][name]
        with np.load(path, allow_pickle=False) as part:
            count = len(part['xyz'])
            for k in TRAIN_FIELDS:
                arr = torch.from_numpy(part[k])
                if k not in data:
                    data[k] = torch.empty((n, *arr.shape[1:]), dtype=arr.dtype, device='cuda')
                data[k][offset:offset+count].copy_(arr)
            offset += count
    assert offset == n
    return data


def batch(data, ids):
    example = {k: v[ids] for k, v in data.items()}
    length = max(8, int(example['valid'].shape[1]))
    # Source batches keep a fixed token width, avoiding data-dependent shapes on the GPU.
    assert example['text'].shape[1] == length
    return example


def filtered_data(data, keep):
    return {k: v[keep] for k, v in data.items()}


@torch.inference_mode()
def monitor(model, data, arm, batch_size):
    from batch_yaw_iou import many_iou

    totals = defaultdict(float)
    model.eval()
    for j in range(0, len(data['xyz']), batch_size):
        part = batch(data, slice(j, j+batch_size))
        found = model(part)
        boxes = found['boxes'].cpu().numpy()
        if arm == 'fixed_proposals':
            best = found['proposal_logits'].argmax(-1)
            # Source-monitor diagnostics use axis original seeds. The actual target
            # arm applies the historic R155 PCA on the matched 16,384 input points.
            prior = torch.cat((part['prior'], torch.zeros_like(part['prior'][..., :1])), -1)
            boxes[:, 0] = prior[torch.arange(len(best), device=best.device), best].cpu().numpy()
        labels = part['gt'].cpu().numpy()
        known = part['known_ref'].cpu().numpy()
        subj = np.asarray([many_iou(b[None, 0], t[0])[0] for b, t in zip(boxes, labels, strict=True)])
        ref = np.zeros(len(known))
        ref[known] = [many_iou(b[None, 1], t[1])[0]
                      for b, t in zip(boxes[known], labels[known], strict=True)]
        support = (found['support'] * in_boxes(part['xyz'], part['gt'])).sum(-1).cpu().numpy()
        for k, val in (
            ('n', len(known)), ('relation_n', known.sum()), ('subject_iou', subj.sum()),
            ('subject25', (subj>.25).sum()), ('subject50', (subj>.50).sum()),
            ('subject_support', support[:, 0].sum()), ('reference_support', support[known, 1].sum()),
            ('reference25', (ref>.25).sum()), ('reference50', (ref>.50).sum()),
            ('relationship_subject25', (known & (subj>.25)).sum()),
            ('pair25', ((subj>.25) & (ref>.25)).sum()),
            ('pair50', ((subj>.50) & (ref>.50)).sum())):
            totals[k] += float(val)
    return dict(totals)


def fit_arm(source, arm, train, diagnostics, options):
    folder = PRIVATE/source
    ckpt = folder/f'{arm}.pth'
    receipt = folder/f'{arm}_TRAINING.json'
    hashes = dict(protocol=sha(SPEC), features=sha(folder/'FEATURE_SEAL.json'),
                  decoder=sha(Path(__file__).with_name('decoder.py')), trainer=sha(Path(__file__)))
    if receipt.exists():
        value = read(receipt)
        assert value['inputs'] == hashes and sha(ckpt) == value['checkpoint']
        return value
    results = []
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    rng = torch.Generator(device='cuda').manual_seed(options['seed']+1)
    engine = Decoder(arm).cuda()
    initial = [p.detach().clone() for p in engine.parameters()]
    opt = torch.optim.AdamW(engine.parameters(), lr=options['learning_rate'],
                            weight_decay=options['weight_decay'], fused=True)
    epochs, batch_size = options['epochs'], options['batch_size']
    n = len(train['xyz'])
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=options['learning_rate']*.1)
    origin = time.perf_counter()
    for epoch in range(epochs):
        engine.train()
        loss_sum = torch.zeros((), device='cuda')
        permutation = torch.randperm(n, generator=rng, device='cuda')
        for j in range(0, n, batch_size):
            part = batch(train, permutation[j:j+batch_size])
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = engine(part)
                loss = objective(part, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(engine.parameters(), options['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            loss_sum += loss.detach()*len(part['xyz'])
        lr = schedule.get_last_lr()[0]
        schedule.step()
        value = (loss_sum/n).item()
        results.append(dict(epoch=epoch+1, train_loss=value, lr=lr))
        print('SOURCE_TRAIN', source, arm, epoch+1, epochs, 'loss', round(value, 5), 'seconds',
              round(time.perf_counter()-origin, 1), flush=True)
    state = {k: v.detach().cpu() for k, v in engine.state_dict().items()}
    tmp = ckpt.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, ckpt)
    delta = torch.stack([(p-q).norm().square() for p, q in zip(engine.parameters(), initial, strict=True)]).sum().sqrt().item()
    measures = monitor(engine, diagnostics, arm, batch_size)
    res = dict(source=source, arm=arm, inputs=hashes, checkpoint=sha(ckpt), train=results,
               n_train=n, budget_updates=epochs*((n+batch_size-1)//batch_size), weight_delta_l2=delta,
               parameters=sum(p.numel() for p in engine.parameters()), diagnostic=measures,
               wall_seconds=time.perf_counter()-origin, source_monitor_used_for_selection=False)
    json_atomic(receipt, res)
    return res


def train_source(source):
    verify_inputs()
    source_features(source)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    options = read(SPEC)['source_protocol']['training']
    train, diagnostics = (load_split(source, part) for part in ('train', 'monitor'))
    for arm in ARMS:
        fit_arm(source, arm, train, diagnostics, options)
    # Fixed output ablation shares the identical training objective and random stream.
    # Verify this implementation property before using it as an output-only ablation.
    folder = PRIVATE/source
    a = torch.load(folder/'joint.pth', weights_only=True)
    b = torch.load(folder/'fixed_proposals.pth', weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a), 'Unintended training difference in output-only arm'
    del a, b, train, diagnostics
    gc.collect()
    torch.cuda.empty_cache()
    seal = dict(source=source, inputs=sha(PRIVATE/'INPUT_LOCK.json'), shared_labels_and_budget=True,
                fixed_and_joint_weights_bit_equal=True,
                arms={arm: sha(folder/f'{arm}_TRAINING.json') for arm in ARMS})
    json_atomic(folder/'MODEL_SEAL.json', seal)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', choices=PLATFORMS, nargs='+')
    options = parser.parse_args()
    for s in options.sources:
        train_source(s)

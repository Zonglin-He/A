"""Train only the two missing cells using the original deterministic budget."""
import gc
import os
import time

import torch

from study import HERE, NEW, OLD_CELL, OLD_PRIVATE, PLATFORMS, PRIVATE, json_atomic, pin_prior, read, sha, state
from decoder import objective
from factor_decoder import FactorDecoder
from train import batch, load_split, monitor


def fit(source, cell, train, diagnostics, options):
    folder = PRIVATE/source
    folder.mkdir(exist_ok=True, mode=0o700)
    ckpt, receipt = folder/f'{cell}.pth', folder/f'{cell}_TRAINING.json'
    contract = dict(input_sha256=sha(PRIVATE/'INPUT_LOCK.json'),
                    feature_sha256=sha(OLD_PRIVATE/source/'FEATURE_SEAL.json'),
                    decoder_sha256=sha(HERE/'factor_decoder.py'))
    if receipt.exists():
        r = read(receipt)
        assert r['contract'] == contract and r['checkpoint'] == sha(ckpt)
        return r
    torch.manual_seed(options['seed'])
    torch.cuda.manual_seed_all(options['seed'])
    rng = torch.Generator(device='cuda').manual_seed(options['seed']+1)
    model = FactorDecoder(cell).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=options['learning_rate'],
                            weight_decay=options['weight_decay'], fused=True)
    epochs, batch_size = options['epochs'], options['batch_size']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=options['learning_rate']*.1)
    n = len(train['xyz'])
    curve = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device='cuda')
        permutation = torch.randperm(n, generator=rng, device='cuda')
        for j in range(0, n, batch_size):
            part = batch(train, permutation[j:j+batch_size])
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(part)
                loss = objective(part, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'], error_if_nonfinite=True)
            opt.step()
            loss_sum += loss.detach()*len(part['xyz'])
        value, lr = (loss_sum/n).item(), scheduler.get_last_lr()[0]
        scheduler.step()
        curve.append(dict(epoch=epoch+1, train_loss=value, lr=lr))
        print('TRAIN', source, cell, epoch+1, epochs, round(value, 5),
              round(time.perf_counter()-start, 1), flush=True)
    tmp = ckpt.with_suffix('.tmp')
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, tmp)
    os.replace(tmp, ckpt)
    diag = monitor(model, diagnostics, cell, batch_size)
    record = dict(source=source, cell=cell, checkpoint=sha(ckpt), contract=contract, curve=curve, diagnostic=diag,
                  n_train=n, updates=epochs*((n+batch_size-1)//batch_size),
                  parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-start)
    for original in OLD_CELL.values():
        r = read(OLD_PRIVATE/source/f'{original}_TRAINING.json')
        assert (record['n_train'], record['updates'], record['parameters']) == (r['n_train'], r['budget_updates'], r['parameters'])
    json_atomic(receipt, record)
    return record


def main():
    options = pin_prior()['recipe']
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state('source_training', 'Fitting the two missing cells for each source with the sealed original features.')
    seals = {}
    for source in PLATFORMS:
        train, monitor_data = (load_split(source, part) for part in ('train', 'monitor'))
        for cell in NEW:
            fit(source, cell, train, monitor_data, options)
            file = PRIVATE/source/f'{cell}_TRAINING.json'
            seals[f'{source}/{cell}'] = sha(file)
        del train, monitor_data
        gc.collect()
        torch.cuda.empty_cache()
    json_atomic(PRIVATE/'TRAINING_SEAL.json', dict(inputs=sha(PRIVATE/'INPUT_LOCK.json'), receipts=seals,
                                                n_new_runs=6, no_target_training=True))
    print('SEALED_TRAINING', len(seals), flush=True)


if __name__ == '__main__':
    main()

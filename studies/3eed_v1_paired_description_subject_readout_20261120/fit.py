"""Train matched readouts on fit pairs. The two training runs never see held-out labels."""
import csv
from time import perf_counter
from paired import (ARMS, HERE, WORK, PUBLIC, F, anon, archive_npz, atomic, digest,
                    initialize, logit, np, pair_margins, read, torch)


def load(split, seal):
    path = WORK/f'{split}.npz'
    assert digest(path) == seal['splits'][split]['sha256']
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def eligible(data):
    return (data['slots'] >= 0).all(1)


def objective(m, arm):
    if arm == 'mean':
        return F.softplus(-m).mean()
    if arm == 'worst':
        return F.softplus(-m.amin(-1)).mean()
    raise ValueError(arm)


def fit(arm, data, initial, cfg):
    use = eligible(data)
    x, base, which = [torch.from_numpy(data[k][use]) for k in ('h', 'j', 'slots')]
    head = initial.clone().requires_grad_()
    optim = torch.optim.Adam([head], lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    trace = []
    for step in range(cfg['steps'] + 1):
        m = pair_margins(logit(x, initial, base, head), which)
        loss = objective(m, arm)
        trace.append(dict(step=step, loss=loss.item(), both=int((m > 0).all(-1).sum()),
                          l2=(head-initial).norm().item()))
        if step < cfg['steps']:
            optim.zero_grad()
            loss.backward()
            optim.step()
    assert torch.isfinite(head).all() and trace[-1]['l2'] > 0
    return head.detach(), trace


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def review(split, data, heads, initial):
    ok = eligible(data)
    has = (data['slots'] >= 0).sum(1)
    assert np.all(data['slots'][ok,0] != data['slots'][ok,1])
    source = []
    aggregate = {}
    for arm, head in zip(ARMS, heads):
        sc = logit(torch.from_numpy(data['h']), initial, torch.from_numpy(data['j']), head)
        if arm == 'v1':
            np.testing.assert_array_equal(sc.numpy(), data['j'])
        margin = pair_margins(sc[ok], torch.from_numpy(data['slots'][ok])).numpy()
        choices = sc.numpy().argmax(-1)
        co = data['hits'][np.arange(len(ok))[:,None], np.arange(2)[None], choices]
        idx = 0
        for i in range(len(ok)):
            m = margin[idx] if ok[i] else None
            idx += int(ok[i])
            source.append(dict(split=split, sort=data['key'][i], arm=arm,
                available_endpoints=int(has[i]), shared_slot_possible=bool(ok[i]),
                ambiguous_endpoint_count=int(data['ambiguous'][i].sum()),
                source_primary_in_pair=int(data['primary'][i]),
                positive_margins='' if m is None else int((m>0).sum()),
                m_a='' if m is None else float(m[0]), m_b='' if m is None else float(m[1]),
                both_pair_correct='' if m is None else int((m>0).all()),
                both_four_way_correct=int(co[i].all()), matched_four_way_ends=int(co[i].sum()),
                both_have_50=int((data['ious'][i].max(-1) > .50).all())))
        valid_both = (margin > 0).all(-1)
        ascending = np.min(margin, axis=-1)
        aggregate[arm] = dict(pairs=len(ok), both_available=int(ok.sum()),
                one_available=int((has==1).sum()), neither_available=int((has==0).sum()),
                valid_frames=len(set(data['frame_key'][ok])), valid_recordings=len(set(data['rec_key'][ok])),
               both_pair_correct=int(valid_both.sum()), only_one_correct=int(((margin>0).sum(1)==1).sum()),
                both_four_way_correct=int(co.all(1).sum()),
                positive_directions=int((margin>0).sum()),
                mean_worse_margin=float(ascending.mean()) if len(margin) else None,
                geometric_both_50=int((data['ious'].max(-1)>.50).all(-1).sum()))
    return source, aggregate


def main():
    cfg = initialize()
    seal = read(WORK/'SOURCE_SEAL.json')
    assert seal['protocol'] == digest(HERE/'PROTOCOL.json')
    assert all(digest(HERE/k)==v for k,v in seal['capture_code'].items())
    assert digest(WORK/'initial.npz') == seal['initial']
    initial = torch.from_numpy(np.load(WORK/'initial.npz')['head'])
    assert initial.shape == (97,)
    data = load('fit', seal)
    if not eligible(data).any():
        raise ValueError('No same-slate supervised fit pairs. No trained model can be reported.')
    start = perf_counter()
    heads = [initial]
    traces = {}
    for arm in ARMS[1:]:
        h, trace = fit(arm, data, initial, cfg['optim'])
        heads.append(h)
        traces[arm] = trace
    archive_npz(WORK/'heads.npz', heads=torch.stack(heads).numpy())
    atomic(WORK/'HEADS_LOCK.json', dict(heads=digest(WORK/'heads.npz'), source=digest(WORK/'SOURCE_SEAL.json'),
           protocol=digest(HERE/'PROTOCOL.json'), training_code=digest(HERE/'fit.py'), arms=list(ARMS),
           eligible_training_pairs=int(eligible(data).sum()), optimizer_steps=cfg['optim']['steps'],
           parameters=initial.numel(), target_labels_read=False, FINAL_access=False,
           elapsed_training_seconds=perf_counter()-start))
    # Open the held-out features and labels only after the head lock has been written.
    summaries, all_rows = {}, []
    for split in ('fit', 'calibration'):
        rows, summaries[split] = review(split, load(split, seal), heads, initial)
        identifiers = sorted({r['sort'] for r in rows})
        index = {v:f'{split[0].upper()}{i:03d}' for i,v in enumerate(identifiers,1)}
        for row in sorted(rows,key=lambda r:(r['sort'], ARMS.index(r['arm']))):
            row['pair_id'] = index[row.pop('sort')]
            all_rows.append(row)
    write_csv(PUBLIC/'source_pairs.csv', all_rows)
    atomic(PUBLIC/'source_summary.json', summaries)
    write_csv(PUBLIC/'training.csv', [dict(arm=arm, **step) for arm, rows in traces.items() for step in rows])
    print(summaries, flush=True)


if __name__ == '__main__':
    main()

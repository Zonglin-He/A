"""Cross-check a compact source-trained identity read on held-out recordings."""
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from appearance_collect import DATA, HERE, MIN_SEEDS, PLATFORMS, PROTOCOL, atomic_json, order, read, sha
from edge_score import csv_atomic


ARMS = ('rgb_inside', 'rgb_exterior_contrast', 'lidar_exterior_contrast')
EPOCHS = 16


@dataclass
class Example:
    frame: int
    group: str
    source: str
    role: str
    inside: np.ndarray
    exterior: np.ndarray
    text: np.ndarray
    base: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    same_class: np.ndarray
    visible: np.ndarray


class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Linear(288,16,bias=False)
        self.local = nn.Linear(352,16,bias=False)

    def forward(self, text, local):
        left = F.normalize(self.text(text),dim=-1)
        right = F.normalize(self.local(local),dim=-1)
        return 4. * (left[:,None,:]*right).sum(dim=-1).tanh()


def read_examples():
    from predict171 import read_npz

    root = DATA/'source'
    receipt = read(root/'CAPTURE_SEAL.json')
    assert receipt['lock']==sha(root/'CAPTURE_LOCK.json')
    assert not receipt['target_labels_read'] and not receipt['protected_data_access']
    assert sha(root/'ROSTER.json')==read(root/'ROSTER_SEAL.json')['roster']
    roster = read(root/'ROSTER.json')
    assert len(roster)==len(receipt['files'])
    examples = []
    for entry in roster:
        tag = f"{entry['case']:03d}_{entry['source']}.npz"
        assert sha(root/tag)==receipt['files'][tag]
        packet = read_npz(root/tag)
        for role in ('s','r'):
            if role+'_txt' not in packet:
                continue
            def val(field):
                return packet[role+'_'+field]
            inner = val('inner_count')
            outer = val('outer_count')
            good = val('owners')==val('truth')
            same = val('class')
            visible = (inner>=MIN_SEEDS) & (outer>=MIN_SEEDS) & (np.linalg.norm(val('txt'))>0)
            chosen = np.flatnonzero(visible&same)
            w = np.zeros(len(good))
            for index in np.unique(val('owners')[chosen]):
                ids = chosen[val('owners')[chosen]==index]
                w[ids] = 1. / len(ids)
            field = Example(entry['case'],entry['group'],entry['source'],role,val('inside'),val('exterior'),
                            val('txt'),val('base'),good,w,same,visible)
            examples.append(field)
    return examples,receipt


def informative(r):
    valid = r.weight>0
    return (valid&r.target).any() and (valid&~r.target).any()


def pack(examples, arm):
    n = len(examples)
    k = max(len(x.target) for x in examples)
    feat = np.zeros((n,k,352),np.float32)
    base = np.zeros((n,k),np.float32)
    truth = np.zeros((n,k),bool)
    weights = np.zeros((n,k),np.float32)
    phrase = np.stack([r.text for r in examples])
    for i,r in enumerate(examples):
        m = len(r.target)
        feat[i,:m] = r.inside if arm=='rgb_inside' else r.inside-r.exterior
        base[i,:m] = r.base
        truth[i,:m] = r.target
        weights[i,:m] = r.weight
    if arm=='lidar_exterior_contrast':
        feat[:,:,288:] = 0.
    return tuple(torch.from_numpy(n) for n in (feat,base,truth,weights,phrase))


def train(examples,arm,seed):
    torch.manual_seed(seed)
    net = Readout()
    net.train()
    opt = torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01)
    if examples:
        feat,base,truth,weight,phrase = pack(examples,arm)
        rng = torch.Generator().manual_seed(seed+1)
        for epoch in range(EPOCHS):
            for ix in torch.randperm(len(phrase),generator=rng).split(24):
                z = base[ix] + .5*net(phrase[ix],feat[ix])
                viable = weight[ix]>0
                z = z + weight[ix].clamp_min(1.e-20).log()
                z = z.masked_fill(~viable,-1.e4)
                positive = (truth[ix] & viable)
                logloss = torch.logsumexp(z,dim=1)-torch.logsumexp(z.masked_fill(~positive,-1.e4),dim=1)
                opt.zero_grad(set_to_none=True)
                loss = logloss.mean()
                assert torch.isfinite(loss)
                loss.backward()
                opt.step()
    net.eval()
    return net


def infer(net,r,arm):
    with torch.inference_mode():
        local = r.inside if arm=='rgb_inside' else r.inside-r.exterior
        if arm=='lidar_exterior_contrast':
            local = local.copy()
            local[:,288:] = 0.
        extra = .5*net(torch.from_numpy(r.text[None]),torch.from_numpy(local[None])).numpy()[0]
    return r.base+extra*r.visible


def measure(r,score):
    eligible = informative(r)
    both = r.weight>0
    local_idx = int(np.argmax(np.where(both,score,-np.inf)))
    known = r.target.any()
    if eligible:
        evidence = float(np.max(score[both&r.target])-np.max(score[both&~r.target]))
    else:
        evidence = ''
    return dict(eligible=int(eligible),eligible_right=int(r.target[local_idx]) if eligible else '',
                available=int(known),has_visible_positive=int((r.target&r.visible).any()),
                winner_right=int(r.target[np.argmax(score)]),same_class_gap=evidence)


def run():
    from common160 import npz_atomic
    from collections import Counter

    torch.set_num_threads(1)
    start = time.perf_counter()
    rows, receipt = read_examples()
    out = HERE/'results/appearance_source'
    path = DATA/'source'
    fit = path/'FIT_SEAL.json'
    assert not fit.exists()
    atomic_json(path/'FIT_LOCK.json',dict(code=sha(Path(__file__)),design=sha(PROTOCOL),
                capture=sha(path/'CAPTURE_SEAL.json'),exp_epochs=EPOCHS))
    all_metrics, summaries = [],[]
    final, fit_files = {},{}
    for source_idx, source in enumerate(PLATFORMS):
        cases = [r for r in rows if r.source==source]
        if not cases:
            final[source] = False
            print(source, 'no eligible source roster', flush=True)
            continue
        groups = sorted({r.group for r in cases},key=order)
        f = min(5,len(groups))
        k = {g:i%f for i,g in enumerate(groups)}
        train_sizes = []
        metrics = []
        for fold in range(f):
            train_cases = [r for r in cases if k[r.group]!=fold and informative(r)]
            eval_cases = [r for r in cases if k[r.group]==fold]
            train_sizes.append(len(train_cases))
            nets = {a:train(train_cases,a,251119+100*source_idx+fold) for a in ARMS}
            for item in eval_cases:
                sources = [('frozen',item.base)] + [(a,infer(m,item,a)) for a,m in nets.items()]
                for a,z in sources:
                    metrics.append(dict(source=source,anonymous_frame=item.frame,fold=fold,role=item.role,arm=a,
                                        **measure(item,z)))
        all_metrics.extend(metrics)
        status = {}
        for role in ('s','r'):
            for arm in ('frozen',*ARMS):
                relevant = [r for r in metrics if r['arm']==arm and r['role']==role]
                active = [r for r in relevant if r['eligible']]
                numbers = dict(source=source,role=role,arm=arm,roles=len(relevant),recordings=len(groups),
                               visible_retrievable=sum(r['has_visible_positive'] for r in relevant),
                               retrieved_annotated=sum(r['available'] for r in relevant),
                               hard_cases=len(active),hard_correct=sum(r['eligible_right'] for r in active),
                               correct_all=sum(r['winner_right'] for r in relevant),
                               hard_groups=len({x.group for x in cases if x.role==role and informative(x)}),
                               hard_folds=len({r['fold'] for r in active}),fit_min=min(train_sizes) if train_sizes else 0)
                summaries.append(numbers)
                status[role,arm] = numbers
        tested = status['r','rgb_exterior_contrast']
        enough = tested['hard_cases']>=8 and tested['hard_groups']>=2 and min(train_sizes,default=0)>0
        adopted = enough and all(tested['hard_correct']>status['r',a]['hard_correct'] for a in ('frozen','rgb_inside'))
        final[source] = bool(adopted)
        if adopted:
            for variant in ARMS:
                head = train([r for r in cases if informative(r)],variant,251119+100*source_idx)
                h = path/f'{source}_{variant}.npz'
                npz_atomic(h,**{k:v.detach().numpy() for k,v in head.state_dict().items()})
                fit_files[h.name] = sha(h)
        print(source,'source cases',Counter(r.role for r in cases),'source gate',adopted,flush=True)
    csv_atomic(out/'fold_rows.csv',all_metrics)
    csv_atomic(out/'metrics.csv',summaries)
    atomic_json(fit,dict(source=sha(path/'CAPTURE_SEAL.json'),fit_lock=sha(path/'FIT_LOCK.json'),
                         metrics=sha(out/'metrics.csv'),source_use_gate=final,private_heads=fit_files,
                         elapsed_s=time.perf_counter()-start,protected_data_access=False,target_label_used=False))


if __name__=='__main__':
    run()

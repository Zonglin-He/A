"""Train a one-sentence readout with a physical two-object/two-sentence assignment."""
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F
from common import ARMS, HERE, PREV, RESULTS, WORK, archive_npz, atomic, digest, np, read, setup, torch
from paired import logit, pair_margins
from fit import write_csv


def loaded(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}


def broken_links(category, scene, margin):
    # Minimize change in the baseline difficulty of the opposite sentence. Every
    # original second-direction comparison is used exactly once, with its true label.
    ids = np.arange(len(category))
    perm = ids.copy()
    for cls in np.unique(category):
        ix = ids[category == cls]
        distance = np.abs(margin[ix,None] - margin[None,ix])
        same = scene[ix,None] == scene[None,ix]
        penalty = (distance.max()+1)*(len(ix)+1)
        cost = distance + penalty*same
        a,b = linear_sum_assignment(cost)
        perm[ix[a]] = ix[b]
    assert np.array_equal(np.sort(perm),ids) and (category == category[perm]).all()
    return perm


def loss(margin,arm,permutation):
    if arm == 'independent':
        return F.softplus(-margin).mean()
    if arm == 'worst':
        return F.softplus(-margin.amin(1)).mean()
    if arm == 'assignment':
        return F.softplus(-margin.sum(1)).mean()
    if arm == 'broken_pair':
        return F.softplus(-margin[:,0]-margin[permutation,1]).mean()
    raise ValueError(arm)


def fit_heads(data,original):
    use = data['available'].all(1)
    h,j,slots = [torch.from_numpy(data[k][use]) for k in ('h','j','slots')]
    valid = slots.numpy()
    assert (valid>=0).all() and (valid[:,0] != valid[:,1]).all()
    m = pair_margins(j,slots)
    scene = data['frame_key'][use]
    perm = broken_links(data['category'][use],scene,m[:,1].numpy())
    traces,heads = [],[original]
    for arm in ARMS[1:5]:
        weight = original.clone().requires_grad_()
        optimizer = torch.optim.Adam([weight],lr=.001)
        for step in range(65):
            margin = pair_margins(logit(h,original,j,weight),slots)
            objective = loss(margin,arm,perm)
            traces.append(dict(arm=arm,step=step,loss=objective.item(),
                correct_both=int((margin>0).all(1).sum()),weight_distance=(weight-original).norm().item()))
            if step < 64:
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
        assert torch.isfinite(weight).all()
        heads.append(weight.detach())
    return heads,perm,traces


def change_count(base,other,scale):
    return int(((base + scale*(other-base)).argmax(-1) != base.argmax(-1)).sum())


def match_scale(base,other,count):
    if change_count(base,other,1.) <= count:
        return 1.
    # The original winner's upper-envelope membership can only be lost once
    # along a score segment, hence the number of switches is monotone.
    lo,hi = 0.,1.
    for _ in range(56):
        mid = (lo+hi)/2
        if change_count(base,other,mid) <= count:
            lo = mid
        else:
            hi = mid
    return lo


def all_scores(h,j,w0,heads):
    x = torch.as_tensor(h)
    baseline = torch.as_tensor(j)
    return np.stack([logit(x,w0,baseline,w).double().numpy() for w in heads])


def versions(scores,scales):
    versions = dict(zip(ARMS,scores))
    for idx,arm in enumerate(ARMS):
        if arm != 'v1':
            versions[arm+'_matched'] = scores[0]+scales[arm]*(scores[idx]-scores[0])
    return versions


def source_review(data,split,heads,scales):
    lines = []
    summaries = []
    keys = data['key']
    for probe, pre in [('pretruncation',''),('two_nominations','free_')]:
        h,j,slots,hits = [data[pre+k] for k in ('h','j','slots','hits')]
        scores = versions(all_scores(h,j,heads[0],heads),scales)
        eligible = (slots>=0).all(1)
        both0 = None
        for arm, logits in scores.items():
            margins = np.full((len(j),2),np.nan)
            rows = np.flatnonzero(eligible)
            marg = pair_margins(torch.from_numpy(logits[rows]),torch.from_numpy(slots[rows])).numpy()
            margins[rows] = marg
            both = (margins>0).all(1)
            interaction = np.sum(margins,axis=1)
            choices = logits.argmax(-1)
            four_both = hits[np.arange(len(j))[:,None],np.arange(2)[None,:],choices].all(1)
            if arm == 'v1':
                both0 = both.copy()
            summaries.append(dict(split=split,probe=probe,arm=arm,potential=len(j),
                available=int(eligible.sum()),both=int(both.sum()), assignment_correct=int((interaction>0).sum()),
                both_four_way=int(four_both.sum()),recovered=int((both & ~both0).sum()),harmed=int((~both & both0).sum()),
                mean_min_margin=float(marg.min(1).mean()),mean_contrast=float(marg.sum(1).mean())))
            for i in range(len(keys)):
                lines.append(dict(split=split,probe=probe,arm=arm,pair=str(keys[i])[:16],
                    recording=str(data['rec_key'][i])[:12],frame=str(data['frame_key'][i])[:12],
                    available=bool(eligible[i]),both=bool(both[i]),both_four_way=bool(four_both[i]),
                    m_a=float(margins[i,0]) if eligible[i] else '',m_b=float(margins[i,1]) if eligible[i] else ''))
    return summaries,lines


def main():
    setup()
    seal = read(WORK/'SOURCE_SEAL.json')
    assert digest(WORK/'fit.npz') == seal['splits']['fit']['hash']
    previous = read(PREV/'run_PRIVATE/HEADS_LOCK.json')
    assert digest(PREV/'run_PRIVATE/heads.npz') == previous['heads']
    old = torch.from_numpy(loaded(PREV/'run_PRIVATE/heads.npz')['heads'])
    data = loaded(WORK/'fit.npz')
    heads,perm,trace = fit_heads(data,old[0])
    heads.extend([old[1],old[2]])
    archive_npz(WORK/'heads.npz', heads=torch.stack(heads).numpy(),permutation=perm)
    active = data['available'].all(1)
    frame = data['frame_key'][active]
    atomic(WORK/'HEADS_LOCK.json',dict(heads=digest(WORK/'heads.npz'),input=digest(WORK/'SOURCE_SEAL.json'),
        protocol=digest(HERE/'PROTOCOL.json'),code=digest(HERE/'fit_duels.py'),arms=list(ARMS),
        n_fit=int(active.sum()),pair_broken_frames=int((frame != frame[perm]).sum()),
        target_labels=False,source_holdout_labels_used_for_fit=False))
    assert digest(WORK/'native_cal.npz') == seal['native_cal']
    native = loaded(WORK/'native_cal.npz')
    scores = all_scores(native['h'],native['j'],old[0],heads)
    counts = {a:change_count(scores[0],scores[i],1.) for i,a in enumerate(ARMS)}
    scales = {'v1':1.}
    for comp in (ARMS[1:5], ARMS[5:]):
        count = min(counts[a] for a in comp)
        for arm in comp:
            scales[arm] = match_scale(scores[0],scores[ARMS.index(arm)],count)
    actual = {a:change_count(scores[0],scores[i],scales[a]) for i,a in enumerate(ARMS)}
    atomic(WORK/'DECODER_LOCK.json',dict(head_lock=digest(WORK/'HEADS_LOCK.json'),scales=scales,
           unscaled=counts,matched=actual,calibration_rows=len(native['j']),target_labels=False))
    write_csv(RESULTS/'fit.csv',trace)
    assert digest(WORK/'calibration.npz') == seal['splits']['calibration']['hash']
    summaries,lines = [],[]
    for split in ('fit','calibration'):
        s,l = source_review(loaded(WORK/(split+'.npz')),split,heads,scales)
        summaries.extend(s)
        lines.extend(l)
    write_csv(RESULTS/'source_summary.csv',summaries)
    from score import compressed_csv
    compressed_csv(RESULTS/'source_pairs.csv.gz',lines)
    atomic(RESULTS/'calibration.json',dict(native=counts,matched=actual,scales=scales,
              waymo_source_rows=160,broken_pairs_outside_frame=int((frame!=frame[perm]).sum()),total_pairs=len(perm)))
    print(read(RESULTS/'calibration.json'),flush=True)


if __name__=='__main__':
    main()

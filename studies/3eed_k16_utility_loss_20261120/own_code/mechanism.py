"""Check the 3D construction and differentiate through the exact frozen-source scores."""
import numpy as np
import torch
from scipy.optimize import minimize

from common import ARMS, HERE, OUT, PLATFORMS, atomic, digest, register, source_data, write_csv
from losses import loss, pair_only
from train import many_iou
from ranker182 import protocol


def counterexample():
    def box(x):
        return [x, .5, .5, 3., 1., 1., 0.]
    b = np.array([box(x) for x in (0.,-1.5,1.5)])
    gt = np.array([box(x) for x in (-.75,.75,-2.25,2.25)])
    overlaps = np.stack([many_iou(b, target) for target in gt])
    np.testing.assert_array_equal(overlaps>.25, overlaps>.50)
    multiplicity = [3,3,2,2]
    freq = np.average(overlaps>.25, weights=multiplicity, axis=0)
    np.testing.assert_allclose(freq,[.6,.5,.5])
    result = {}
    for size in (4,16):
        dummy_boxes = np.array([box(30.+4*j) for j in range(size-3)])
        assert not np.any([many_iou(dummy_boxes, target) for target in gt])
        ious = torch.from_numpy(np.pad(np.repeat(overlaps,multiplicity,axis=0), ((0,0),(0,size-3))))
        for tag in [ARMS[0], ARMS[1], ARMS[2], ARMS[3], "pair_only"]:
            def fun(ab):
                weights = torch.tensor(ab,dtype=torch.float64,requires_grad=True)
                vec = torch.cat((weights,weights.new_zeros(1),weights.new_full((size-3,),-1000.)))
                s = vec.expand(len(ious),-1)
                value = pair_only(s,ious) if tag=="pair_only" else loss(tag,s,ious)
                grad, = torch.autograd.grad(value,weights)
                return value.item(), grad.numpy()
            solutions = []
            for initial in ([0.,0.],[1.,-1.],[-1.,1.],[2.,.5],[-.5,-2.]):
                opt = minimize(fun,initial,jac=True,method="BFGS",options={"gtol":1e-11,"maxiter":400})
                assert np.max(np.abs(opt.jac))<1e-7
                logits = torch.tensor([*opt.x,0.],dtype=torch.float64)
                solutions.append(logits.softmax(0).numpy())
            np.testing.assert_allclose(solutions, np.repeat([solutions[0]],len(solutions),axis=0),atol=2e-7,rtol=0)
            result[f"{tag}_K{size}"] = solutions[0].tolist()
    checks = {(ARMS[0],16):[.20921981,.39539010,.39539010],
              (ARMS[1],16):[.2,.4,.4],(ARMS[2],16):[.375,.3125,.3125],
              (ARMS[3],16):[.30,.35,.35]}
    for (a,k), expected in checks.items():
        np.testing.assert_allclose(result[f"{a}_K{k}"],expected,atol=2e-7,rtol=0)
    assert result["pair_only_K4"][0] > result["pair_only_K4"][1]
    assert result["pair_only_K16"][0] < result["pair_only_K16"][1]
    return dict(ious=overlaps.tolist(),success=freq.tolist(),probabilities_on_three_real_candidates=result)


def gradient(arm, original, ious, count):
    base = original.clone().requires_grad_(True)
    scores = torch.cat((base,base.new_full((len(base),count),-1000.)),dim=-1)
    padded = torch.cat((ious,ious.new_zeros((len(base),count))),dim=-1)
    value = pair_only(scores,padded) if arm=="pair_only" else loss(arm,scores,padded)
    derivative, = torch.autograd.grad(value,base)
    return derivative.detach().numpy()*len(base)


def run():
    register()
    torch.set_num_threads(1)
    theory = counterexample()
    rows = []
    total = 0
    for source_index,s in enumerate(PLATFORMS):
        block = source_data(s,"fit")
        scores = torch.from_numpy(block["scores"][:,:,0]).double()
        ious = torch.from_numpy(block["ious"]).double()
        mass = (ious>.25).sum(dim=-1).numpy()
        eligible = torch.where((ious>.25).any(-1))[0]
        gen = torch.Generator().manual_seed(protocol()["ranker"]["seed"]+source_index)
        order=eligible[torch.randperm(len(eligible),generator=gen)]
        total+=len(eligible)
        for idx in order.split(32):
            npos=mass[idx.numpy()]
            for arm in (ARMS[0], ARMS[1], ARMS[2], ARMS[3], "pair_only"):
                before=gradient(arm,scores[idx],ious[idx],0)
                for pad in (0,16,48):
                    after=gradient(arm,scores[idx],ious[idx],pad) if pad else before
                    norms=np.linalg.norm(before,axis=1)
                    norms_after=np.linalg.norm(after,axis=1)
                    for i in range(len(idx)):
                        valid=bool(norms[i]>1e-12 and norms_after[i]>1e-12)
                        ratio=float(norms_after[i]/norms[i]) if valid else None
                        cosine=float(np.clip(after[i]@before[i]/(norms_after[i]*norms[i]),-1.,1.)) if valid else None
                        delta=float(np.max(np.abs(after[i]-before[i])))
                        m=int(npos[i])
                        if arm in (ARMS[1], ARMS[2], ARMS[3]):
                            assert delta<1e-12
                        if arm=="pair_only" and valid:
                            np.testing.assert_allclose(ratio,(16-m)/(16+pad-m),atol=1e-10)
                            assert cosine>1-1e-10
                        rows.append(dict(source=s, row=int(idx[i]), positives=m, arm=arm, dummy=pad,
                                         max_abs_change=delta, norm_ratio=ratio, cosine=cosine))
    groups=[]
    for arm in (ARMS[0],ARMS[1],ARMS[2],ARMS[3],"pair_only"):
        for pad in (0,16,48):
            for m in ["all",*sorted({r["positives"] for r in rows})]:
                selected=[r for r in rows if r["arm"]==arm and r["dummy"]==pad and (m=="all" or r["positives"]==m)]
                if not selected:
                    continue
                valid=[r for r in selected if r["cosine"] is not None]
                groups.append(dict(arm=arm,dummy=pad,positives=m,n=len(selected),nonzero=len(valid),
                              max_abs_change=max(r["max_abs_change"] for r in selected),
                              median_norm_ratio=float(np.median([r["norm_ratio"] for r in valid])) if valid else None,
                              median_cosine=float(np.median([r["cosine"] for r in valid])) if valid else None,
                              direction_changed=sum(r["cosine"]<1-1e-6 for r in valid)))
    write_csv(OUT/"source_gradient_by_count.csv",groups)
    atomic(OUT/"mechanism.json",dict(counterexample=theory,fit_eligible=total,dummy_logit=-1000,
           aggregated=[r for r in groups if r["positives"]=="all"],fundamental_code=digest(HERE/"losses.py"),
           target_labels_read=False,FINAL_access=False))
    print("Mechanism experiment complete on",total,"eligible source rows",flush=True)


if __name__=="__main__":
    run()

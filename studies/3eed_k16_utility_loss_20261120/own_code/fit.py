"""Fit the five matched heads with identical randomization and source-only checkpoint selection."""

from hashlib import sha256
from time import perf_counter

import numpy as np
import torch

from common import ARMS, HERE, OFFSETS, OUT, PLATFORMS, PRIVATE, R182, atomic, digest, read, register, source_data, write_csv
from losses import loss
from fit182 import fit_one as historical_fit
from ranker182 import SubjectRanker, choose, protocol
from scope import source_rows
from train import metrics, prepare_device

WEIGHTS=PRIVATE/"heads"


@torch.inference_mode()
def score(model, data):
    model.eval()
    raw = model(torch.from_numpy(data["features"])).numpy()
    choices = choose(data["scores"][:,:,0]+raw, data["incumbent"])
    iou = data["ious"][np.arange(len(choices)), choices]
    return choices, metrics(iou)


def one(arm,seed,fit,cal,cfg):
    torch.manual_seed(seed)
    generator=torch.Generator().manual_seed(seed)
    head=SubjectRanker()
    initial=sha256(b"".join(t.detach().numpy().tobytes() for t in head.parameters())).hexdigest()
    assert sum(p.numel() for p in head.parameters()) == 14449
    optimizer=torch.optim.AdamW(head.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    features=torch.from_numpy(fit["features"])
    joint=torch.from_numpy(fit["scores"][:,:,0])
    ious=torch.from_numpy(fit["ious"])
    eligible=torch.where((ious>.25).any(-1))[0]
    held, losses = [], []
    best = None
    orders=sha256()
    for epoch in range(1, cfg["epochs"]+1):
        head.train()
        order=eligible[torch.randperm(len(eligible),generator=generator)]
        orders.update(order.numpy().tobytes())
        accumulated=0.
        for indices in order.split(cfg["batch_size"]):
            value=loss(arm, joint[indices]+head(features[indices]),ious[indices])
            if not torch.isfinite(value):
                raise ValueError(f"Invalid loss in {arm}, seed {seed}")
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            accumulated+=float(value.detach())*len(indices)
        losses.append(accumulated/len(order))
        if epoch in cfg["checkpoint_epochs"]:
            choices, result = score(head,cal)
            held.append(dict(epoch=epoch,**result))
            key = (result["Acc25"],result["Acc50"],result["mIoU"],-epoch)
            if best is None or key>best[0]:
                best=(key, {k:v.detach().clone() for k,v in head.state_dict().items()},
                      epoch, choices.copy(), result)
    return best[1], best[3], dict(epoch=best[2],heldout=best[4],epoch_losses=losses,checkpoints=held,
                    initialization=initial,sampling=orders.hexdigest(),eligible=int(len(eligible)))


def train():
    register()
    assert read(OUT/"mechanism.json")["fit_eligible"]>0
    prepare_device()
    WEIGHTS.mkdir(mode=0o700,exist_ok=True)
    cfg=protocol()["ranker"]
    old=read(R182/"rankers_PRIVATE/FITTED.json")
    start=perf_counter()
    results=[]
    verified={}
    files={}
    logs={}
    for platform_index,s in enumerate(PLATFORMS):
        fit=source_data(s,"fit")
        cal=source_data(s,"calibration")
        fit_rows=source_rows(s,"fit")
        cal_rows=source_rows(s,"calibration")
        assert len(fit_rows)==len(fit["features"]) and len(cal_rows)==len(cal["features"])
        assert not {r["recording"] for r in fit_rows} & {r["recording"] for r in cal_rows}
        assert not {r["point_sha256"] for r in fit_rows} & {r["point_sha256"] for r in cal_rows}

        # A single archival replay checks that the historical one-seed readout comparison
        # differs only in the training score path; it is never used to choose a new model.
        original_seed=cfg["seed"]+platform_index
        state, trace=historical_fit(original_seed,fit,cal,False)
        parent=R182/"rankers_PRIVATE"/f"{s}_generic.pth"
        assert digest(parent)==old["sources"][s]["models"]["generic"]["weights_sha256"]
        archived=torch.load(parent,map_location="cpu",weights_only=True)["state"]
        errors={k:float((v-archived[k]).abs().max()) for k,v in state.items()}
        if any(errors.values()):
            raise ValueError(f"Original R182 training is not an exact replay: {s}, {errors}")
        assert trace["selected"]==old["sources"][s]["models"]["generic"]["selected"]
        verified[s]=dict(weights_max_abs_diff=max(errors.values()),selected_epoch=trace["selected"]["epoch"],
                         fit_records=len(fit_rows),eligible=trace["fit_eligible"],heldout_records=len(cal_rows),
                         fit_recordings=len({r["recording"] for r in fit_rows}),
                         heldout_recordings=len({r["recording"] for r in cal_rows}))
        cal_choices=[]
        for k,offset in enumerate(OFFSETS):
            packed=[]
            matches=[]
            for arm in ARMS:
                state, choices, info=one(arm,original_seed+offset,fit,cal,cfg)
                name=f"{s}_{k}_{arm}"
                dest=WEIGHTS/(name+".pt")
                torch.save(dict(source=s,seed=original_seed+offset,arm=arm,state=state),dest)
                files[dest.name]=digest(dest)
                packed.append(choices)
                matches.append((info["initialization"],info["sampling"]))
                logs[name]=info
                results.append(dict(source=s,seed_id=k,seed=original_seed+offset,arm=arm,eligible=info["eligible"],
                                    epoch=info["epoch"],**info["heldout"]))
            assert len(set(matches))==1
            cal_choices.append(np.stack(packed,axis=-1))
            print(s,"seed",k,"5 heads selected on source recordings",round(perf_counter()-start,1),flush=True)
        hold=WEIGHTS/f"{s}_heldout_choices.npz"
        np.savez_compressed(hold,choices=np.stack(cal_choices,axis=1))
        files[hold.name]=digest(hold)
    write_csv(OUT/"source_checkpoints.csv",results)
    atomic(PRIVATE/"training_trace.json",logs)
    atomic(OUT/"source_fit_audit.json",dict(source=verified,matched_fits=len(results),
           historical_exact_replays=3,matched_initialization_and_batch_order=True,extra_parameter_count=0,
           target_labels_read=False, FINAL_access=False))
    atomic(WEIGHTS/"LOCK.json",dict(files=files,protocol=digest(PRIVATE/"LOCK.json"),
           training_implementation=digest(HERE/"fit.py"),loss_implementation=digest(HERE/"losses.py"),
           source_summary=digest(OUT/"source_checkpoints.csv"),source_split_checks=digest(OUT/"source_fit_audit.json"),
           source_trace=digest(PRIVATE/"training_trace.json"),target_labels_read=False,FINAL_access=False,
           elapsed_seconds=perf_counter()-start))


if __name__=="__main__":
    train()

"""Describe the locked heads on source data; never selects configurations."""
import numpy as np
import torch

from common import ARMS, OFFSETS, OUT, PLATFORMS, source_data, write_csv
from predict import head_lock, models
from ranker182 import choose


@torch.inference_mode()
def run():
    torch.set_num_threads(1)
    tables = []
    lock = head_lock()
    for source in PLATFORMS:
        heads = models(source, lock)
        for split in ("fit", "calibration"):
            data = source_data(source, split)
            x = torch.from_numpy(data["features"])
            prior = torch.from_numpy(data["scores"][:, :, 0])
            iou = data["ious"]
            good = iou > .25
            number = good.sum(1)
            supported = number > 0
            multiple = number > 1
            for seed in range(len(OFFSETS)):
                methods = [("frozen_joint", None), *zip(ARMS, heads[seed], strict=True)]
                for arm, head in methods:
                    logits = prior if head is None else prior + head(x)
                    picks = choose(logits.numpy(), data["incumbent"])
                    chosen = iou[np.arange(len(iou)), picks]
                    logp = logits.double().log_softmax(-1).numpy()
                    pal = -(good * logp).sum(1)
                    # Spread of probability among successful boxes; unity is uniform.
                    conditional = logits.double().masked_fill(~torch.from_numpy(good), -torch.inf)
                    p = conditional[multiple].softmax(-1).numpy()
                    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(1)
                    tables.append(dict(source=source, split=split, seed=seed, arm=arm,
                        records=len(iou), eligible=int(supported.sum()), multiple=int(multiple.sum()),
                        Acc25=(chosen > .25).mean()*100, Acc50=(chosen > .50).mean()*100, mIoU=chosen.mean()*100,
                        PAL_per_eligible_row=pal[supported].mean(),
                        probability_on_passing_set=(np.exp(logp[supported]) * good[supported]).sum(1).mean(),
                        normalized_entropy_within_multiple_positives=(entropy / np.log(number[multiple])).mean()))
    write_csv(OUT / "source_profile.csv", tables)


if __name__ == "__main__":
    run()

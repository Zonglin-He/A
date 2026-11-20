"""Restore the original 299 features once for all arms and seeds. No target scoring."""
import argparse
from functools import lru_cache
import os
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from common import HERE, R182, R183, PRIVATE, DIRECTIONS, PLATFORMS, atomic, digest, read, register
from scope183 import ordinary, frame, check_input
from predict183 import LockedEvaluator, ranking_fields
from features import observation
from scope import evaluation_rows
from train import prepare_device
import readout181

CAP = PRIVATE / "features"


def capture(source):
    register()
    CAP.mkdir(mode=0o700, exist_ok=True)
    lock = dict(protocol=digest(PRIVATE/"LOCK.json"), extractor=digest(HERE/"capture.py"),
                parent=digest(R183/"predictions_PRIVATE/SEALED.json"))
    parent = read(R183/"predictions_PRIVATE/SEALED.json")
    dev_seal = read(R182/"target_PRIVATE/SEALED.json")
    assert not parent["target_labels_read"] and not dev_seal["target_labels_read"]
    rows = ordinary()
    prepare_device()
    torch.set_num_threads(2)
    model = LockedEvaluator(source)
    model.grounder._frontend.parser = lru_cache(maxsize=12000)(model.grounder._frontend.parser)
    started = perf_counter()
    with patch.object(readout181, "encode", ranking_fields):
        for s,t in DIRECTIONS:
            if s != source:
                continue
            key = f"{s}_to_{t}"
            folder = CAP/key
            folder.mkdir(mode=0o700, exist_ok=True)
            dev = R182/"target_PRIVATE"/f"{key}.npz"
            assert digest(dev) == dev_seal["files"][dev.name]["sha256"]
            with np.load(dev) as z:
                ordinal, expected = z["ordinals"][0], z["features"][0]
            test = model.evaluate(observation(evaluation_rows()[int(ordinal)]), debug=True)
            np.testing.assert_array_equal(test["features"], expected)
            blocks = {}
            for name, spec in sorted(parent["directions"][key]["files"].items()):
                dest = folder/name
                marker = dest.with_suffix(".json")
                if marker.exists():
                    tag = read(marker)
                    assert tag["inputs"] == lock and tag["boxes"] == spec["sha256"] and digest(dest) == tag["sha256"]
                    blocks[name] = tag["sha256"]
                    continue
                native = R183/"predictions_PRIVATE"/key/name
                assert digest(native) == spec["sha256"]
                with np.load(native) as f:
                    ids, boxes, picks = f["ordinals"], f["boxes"], f["choices"]
                    np.testing.assert_array_equal(ids,np.arange(spec["first"],spec["last"]))
                feat, scores = [], []
                for local, i in enumerate(ids):
                    row = rows[t][int(i)]
                    check_input(row)
                    obs = model.evaluate(frame(row), debug=True)
                    np.testing.assert_array_equal(obs["boxes"],boxes[local])
                    np.testing.assert_array_equal(obs["choices"],picks[local])
                    feat.append(obs["features"])
                    scores.append(obs["scores"][:,0])
                tmp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(tmp,features=feat,joint=scores,inc=picks[:,0],ordinals=ids)
                os.replace(tmp,dest)
                blocks[name] = digest(dest)
                atomic(marker,dict(inputs=lock,boxes=spec["sha256"],sha256=blocks[name]))
                print(key,spec["last"],"/",len(rows[t]),round(perf_counter()-started,1),flush=True)
            assert len(rows[t]) == parent["directions"][key]["n"]
            atomic(folder/"SEAL.json",dict(inputs=lock,n=len(rows[t]),files=blocks,dev_feature_exact=True,
                       all_K16_boxes_and_7_parent_arms_exact=True,target_labels_read=False,FINAL_access=False))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",choices=PLATFORMS)
    capture(p.parse_args().source)

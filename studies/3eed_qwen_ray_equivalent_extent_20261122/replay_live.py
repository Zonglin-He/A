"""Prove normal-frame inference matches the sealed shadow method on the full roster."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from time import perf_counter

import numpy as np

from live import predict
import run_trial as trial


def one(job):
    frame,expected,official,checksum,pins = job
    assert frame["uid"]==expected["uid"]==official["uid"]
    base=official["simple4"]
    archived=trial.origin.prediction(frame,checksum,pins)
    answer=trial.origin.replies(frame,archived,trial.replies_hashes)
    actual=predict(frame,answer)
    assert actual["applicable"]==expected["active"]==base["applicable"]
    expected_correction=expected.get("variants",{}).get("ray_equivalent")
    expected_box=base["box"] if expected_correction is None else expected_correction["box"]
    assert actual["extent_corrected"]==(expected_correction is not None)
    assert actual["heading"]==base["heading"] and actual["selection_rays"]==base["selection_rays"]
    if expected_box is None:
        assert actual["box"] is None
        error=0.
    else:
        error=float(np.max(abs(np.asarray(actual["box"])-np.asarray(expected_box))))
        assert error<5.e-9
    return error,actual["extent_corrected"]


def run(workers):
    started=perf_counter()
    seal_path=trial.OUT/"prediction_seal.json"
    seal=trial.roster.read(seal_path)
    assert not seal["target_gt_opened"] and not seal["FINAL_access"]
    assert seal["payload"]==trial.roster.sha(trial.PAYLOAD)
    frames,pins,checks,index,official=trial.origin.inputs()
    count,changed,error=0,0,0.
    with gzip.open(trial.PAYLOAD,"rt") as stream, \
            ProcessPoolExecutor(max_workers=workers,initializer=trial.initialize,initargs=(index,)) as pool:
        jobs=((frame,json.loads(row),base,checks[frame["uid"]],pins) for frame,row,base in zip(frames,stream,official,strict=True))
        for e,c in pool.map(one,jobs,chunksize=10):
            count+=1
            changed+=c
            error=max(error,e)
    assert count==9310 and changed==5904
    trial.roster.write(trial.OUT/"live_replay.json",dict(
        descriptions=count,changed=changed,max_box_error_m=error,
        code=trial.roster.sha(trial.HERE/"live.py"),driver=trial.roster.sha(__file__),
        prediction_seal=trial.roster.sha(seal_path),target_gt_opened=False,new_Qwen_queries=0,
        production_prediction_changes=0,FINAL_access=False,seconds=perf_counter()-started,
    ))
    print("Live entry audited on",count,"descriptions, maximum box difference",error,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    run(parser.parse_args().workers)

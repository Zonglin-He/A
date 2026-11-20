"""Apply all locked, source-selected heads, then seal every decision before task scoring."""
import argparse
import os
import time

import numpy as np
import torch

from common import ARMS, DIRECTIONS, HERE, OFFSETS, PRIVATE, R182, R183, atomic, digest, read, register
from ranker182 import SubjectRanker, choose

READOUTS=("v1","frozen_joint","R183_history","zero_joint",*ARMS)
CAP=PRIVATE/"features"
PRED=PRIVATE/"predictions"
WEIGHTS=PRIVATE/"heads"


def head_lock():
    obj=read(WEIGHTS/"LOCK.json")
    assert obj["protocol"]==digest(PRIVATE/"LOCK.json") and not obj["target_labels_read"]
    assert obj["training_implementation"]==digest(HERE/"fit.py") and obj["loss_implementation"]==digest(HERE/"losses.py")
    return obj


def models(s,lock):
    result=[]
    for k,offset in enumerate(OFFSETS):
        run=[]
        for arm in ARMS:
            path=WEIGHTS/f"{s}_{k}_{arm}.pt"
            assert digest(path)==lock["files"][path.name]
            data=torch.load(path,weights_only=True,map_location="cpu")
            assert data["source"]==s and data["arm"]==arm
            model=SubjectRanker()
            model.load_state_dict(data["state"],strict=True)
            run.append(model.eval().requires_grad_(False))
        result.append(run)
    return result


@torch.inference_mode()
def run_block(s,t,name,source_heads,observed,parent):
    key=f"{s}_to_{t}"
    path=CAP/key/name
    native=R183/"predictions_PRIVATE"/key/name
    assert digest(path)==observed["sha256"] and digest(native)==parent["sha256"]==observed["boxes"]
    with np.load(path,allow_pickle=False) as f, np.load(native,allow_pickle=False) as old:
        xs=torch.from_numpy(f["features"])
        joint,inc,ids=f["joint"],f["inc"],f["ordinals"]
        held=old["choices"]
        np.testing.assert_array_equal(ids,old["ordinals"])
        np.testing.assert_array_equal(ids,np.arange(parent["first"],parent["last"]))
        np.testing.assert_array_equal(inc,held[:,0])
    zero=choose(joint,inc)
    np.testing.assert_array_equal(joint.argmax(-1),held[:,1])
    assert xs.shape==(len(ids),16,299) and torch.isfinite(xs).all()
    endpoints=np.empty((len(ids),len(OFFSETS),len(READOUTS)),dtype=np.uint8)
    endpoints[:,:,:4]=np.stack((inc,held[:,1],held[:,5],zero),axis=1)[:,None,:]
    for k,heads in enumerate(source_heads):
        for a,model in enumerate(heads):
            # Use the deployed single-example matrix operation for each independent prediction.
            residual=torch.stack([model(x) for x in xs]).numpy()
            assert residual.shape==joint.shape and np.isfinite(residual).all()
            endpoints[:,k,4+a]=choose(joint+residual,inc)
    assert (endpoints<16).all()
    tmp=(PRED/key/name).with_suffix(".tmp.npz")
    np.savez_compressed(tmp,choices=endpoints,ordinals=ids)
    os.replace(tmp,PRED/key/name)
    return digest(PRED/key/name)


def predict(watch):
    register()
    torch.set_num_threads(1)
    lock=head_lock()
    trained={s:models(s,lock) for s in dict(DIRECTIONS)}
    capture_pins=dict(protocol=digest(PRIVATE/"LOCK.json"),extractor=digest(HERE/"capture.py"),
                      parent=digest(R183/"predictions_PRIVATE/SEALED.json"))
    expected=read(R183/"predictions_PRIVATE/SEALED.json")
    pins=dict(source_heads=digest(WEIGHTS/"LOCK.json"),predict=digest(HERE/"predict.py"),capture=capture_pins,
              validation_manifest=digest(R183/"ordinary_PRIVATE/SEALED.json"))
    PRED.mkdir(exist_ok=True,mode=0o700)
    verified={}
    while True:
        count=len(verified)
        completed={}
        for s,t in DIRECTIONS:
            key=f"{s}_to_{t}"
            folder=PRED/key
            folder.mkdir(exist_ok=True,mode=0o700)
            block=expected["directions"][key]
            for name,parent in sorted(block["files"].items()):
                if (key,name) in verified:
                    continue
                observation=(CAP/key/name).with_suffix(".json")
                if not observation.is_file():
                    continue
                proof=read(observation)
                assert proof["inputs"]==capture_pins and proof["boxes"]==parent["sha256"]
                record=dict(inputs=pins,features=proof["sha256"],native=parent["sha256"],
                            first=parent["first"],last=parent["last"])
                receipt=(folder/name).with_suffix(".json")
                if not receipt.exists():
                    file_hash=run_block(s,t,name,trained[s],proof,parent)
                    atomic(receipt,dict(record,sha256=file_hash))
                saved=read(receipt)
                assert saved==dict(record,sha256=digest(folder/name))
                verified[key,name]=saved
            cap=CAP/key/"SEAL.json"
            if cap.exists() and all((key,name) in verified for name in block["files"]):
                stamp=read(cap)
                assert stamp["inputs"]==capture_pins and stamp["n"]==block["n"] and not stamp["target_labels_read"]
                assert stamp["files"]=={n:verified[key,n]["features"] for n in block["files"]}
                completed[key]=dict(n=block["n"],feature_seal=digest(cap),
                                    files={n:verified[key,n] for n in block["files"]})
        n=sum(r["last"]-r["first"] for r in verified.values())
        if count!=len(verified):
            print(n,"/ 18620 fixed records",flush=True)
        if len(completed)==6:
            assert n==18620
            result=dict(n=n,arms=list(READOUTS),seed_offsets=list(OFFSETS),directions=completed,
                        inputs=pins,target_labels_read=False,FINAL_access=False)
            path=PRED/"SEAL.json"
            if path.exists():
                assert result==read(path)
            else:
                atomic(path,result)
            print("sealed",n,"direction records",digest(path),flush=True)
            return
        if not watch:
            return
        time.sleep(5)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    predict(p.parse_args().watch)

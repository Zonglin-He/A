"""Counterfactual check for ordinal questions under a positive global scale."""
from collections import defaultdict

import numpy as np

from common import OUT, PRIVATE, PLATFORMS, digest, read, write
from maps import map_arm
from score import parse
from summarize import read_csv, save_csv


FAMILIES=("pixel_order","object_order","static_right","yaw_90")


def run():
    summaries={}
    scale_info={}
    result=read(OUT/"alignment_frames.json")
    hashes=read(PRIVATE/"map_hashes.json")
    for row in result:
        uid=row["uid"]
        file=PRIVATE/"maps"/(uid+".npz")
        assert digest(file)==hashes[uid]
        with np.load(file,allow_pickle=False) as maps:
            mask=maps["mask"]
            a=map_arm(maps,"camera_rays")[:3,mask].astype(float)
            b=map_arm(maps,"global_scale")[:3,mask].astype(float)
        absolute=np.max(abs(b-row["global_ratio"]*a),axis=0)
        scale_info[uid]=dict(target=row["target"], scale=row["global_ratio"],
            uniform=bool((absolute<=1e-4).all()), non_uniform_pixels=int((absolute>1e-4).sum()),
            max_float_error_m=float(absolute.max()))
    write(OUT/"scale_check_frames.json",scale_info)

    raw={}
    seal=read(OUT/"prediction_seal.json")
    assert seal["answers_hashes"]==digest(PRIVATE/"answers_frozen_hashes.json")
    for uid,hasher in read(PRIVATE/"answers_frozen_hashes.json").items():
        file=PRIVATE/"answers_frozen"/(uid+".json")
        assert digest(file)==hasher
        for row in read(file)["answers"]:
            if row["arm"] in ("camera_rays","global_scale"):
                raw[row["qid"],row["arm"]]=parse(row["answer"],"A")
    by_question=defaultdict(dict)
    for r in read_csv("qa_rows.csv"):
        if r["mode"]=="language" and r["family"] in FAMILIES and r["arm"] in ("camera_rays","global_scale"):
            if scale_info[r["uid"]]["uniform"]:
                by_question[r["qid"]][r["arm"]]=r
    for base,changed in [(v["camera_rays"],v["global_scale"]) for v in by_question.values()]:
        assert base["qid"]==changed["qid"]
        q=base["qid"]
        a,b=raw[q,"camera_rays"],raw[q,"global_scale"]
        both=a is not None and b is not None
        row = dict(valid_pair=int(both),flip=int(both and a!=b),n=1,
                   rescue=int(base["correct"]=="0" and changed["correct"]=="1"),
                   harm=int(base["correct"]=="1" and changed["correct"]=="0"),
                   direct_change=int(base["direct_correct"]!=changed["direct_correct"]))
        for key in [(base["target"],base["family"]),("pooled",base["family"])]:
            sums=summaries.setdefault(key,dict(target=key[0],family=key[1],**dict.fromkeys(row,0)))
            for k in row:
                sums[k]+=row[k]
    data=sorted(summaries.values(),key=lambda r:(r["target"],FAMILIES.index(r["family"])))
    save_csv("scale_invariance.csv",data)


if __name__=="__main__":
    run()

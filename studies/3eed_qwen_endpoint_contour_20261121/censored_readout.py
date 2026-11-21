"""Retain H when an unobserved contour edge leaves multiple center choices possible."""
import numpy as np

from endpoints import HERE, OUT, PRIVATE, ancestry, digest, read, write
from outline_geometry import ARMS, SIDES, readout


def single(v):
    frac=v["gap_fractions"]
    if frac is None:
        return v
    neg,pos=frac
    # Reaching the limit supplies only a lower bound on the actual distance to the edge.
    # The projection of the feasible split onto [0,1] is monotone in pos and -neg.
    measured=(1+pos-neg)/2
    low=0. if neg==1. else measured
    high=1. if pos==1. else measured
    def quant(a):
        d=np.abs(np.array([0,.5,1])-np.clip(a,0,1))
        ks=np.flatnonzero(np.isclose(d,d.min(),atol=1.e-10,rtol=0))
        return SIDES[ks[0]] if len(ks)==1 else "H"
    ends=quant(low),quant(high)
    action=ends[0] if ends[0]==ends[1] else "H"
    return dict(action=action,reason="observed_choice" if action!="H" else "unresolved_edge",
                gap_fractions=frac)


def run():
    parent=read(OUT/"outline_prediction_seal.json")
    assert not parent["target_labels_read"] and parent["predictions"]==digest(PRIVATE/"outline_predictions.json")
    old=read(PRIVATE/"outline_predictions.json")
    _,_,_,options=ancestry()
    preds={}
    for uid,arms in old.items():
        preds[uid]={}
        for a in ARMS:
            entry=arms[a]
            views=[single(v) for v in entry["views"]]
            if views:
                result,status=readout([(v["action"],v["reason"],v["gap_fractions"]) for v in views])
            else:
                result,status="H",entry["status"]
            b=options[uid]
            preds[uid][a+"_censored"]=dict(action=result,status=status,views=views,
                                    box=b["H"] if result=="H" else b["discrete_boxes"][result])
    write(PRIVATE/"censored_predictions.json",preds)
    write(OUT/"censored_prediction_seal.json",dict(parent=digest(OUT/"outline_prediction_seal.json"),code=digest(__file__),
          predictions=digest(PRIVATE/"censored_predictions.json"),frames=len(preds),target_labels_read=False,
          FINAL_access=False,scope="Post-score identifiability audit. A path still inside the outline at the maximum shift only lower-bounds the edge distance. Keep H if any compatible pair of edge distances changes the nearest allowed center in either view. No threshold changes or additional images."))


if __name__=="__main__":
    run()

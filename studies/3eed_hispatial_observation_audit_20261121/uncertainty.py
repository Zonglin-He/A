"""Recording-level paired uncertainty for the main frozen comparisons."""
from collections import defaultdict

import numpy as np

from common import ARMS, OUT, PLATFORMS, write
from summarize import read_csv


def bootstrap(rows, fields, seed=392126):
    elements=defaultdict(lambda:defaultdict(list))
    for r,v in rows:
        elements[r["target"]][r["recording"]].append(v)
    random=np.random.default_rng(seed)
    point=[]
    sampled=[]
    sizes={}
    for platform in PLATFORMS:
        recs=elements[platform]
        if not recs:
            continue
        sums=np.array([np.asarray(r,float).sum(axis=0) for r in recs.values()])
        count=np.array([len(r) for r in recs.values()])
        draw=random.integers(len(recs),size=(3000,len(recs)))
        point.append(sums.sum(0)/count.sum())
        sampled.append(sums[draw].sum(1)/count[draw].sum(1)[:,None])
        sizes[platform]=len(recs)
    if not sampled:
        return None
    model=np.mean(sampled,axis=0)
    q=np.quantile(model,[.025,.975],axis=0)
    return dict(recordings=sizes,n=len(rows),
        effects={f:dict(mean_macro=float(v),low_95=float(l),high_95=float(h)) for f,v,l,h in zip(fields,np.mean(point,axis=0),q[0],q[1])})


def run():
    summary=[]
    qa=read_csv("qa_rows.csv")
    qid=defaultdict(dict)
    for r in qa:
        qid[r["qid"]][r["arm"]]=r
    geo=read_csv("geometry_rows.csv")
    samples=defaultdict(dict)
    for r in geo:
        if r["group"]=="all":
            samples[r["uid"]][r["arm"]]=r
    for arm, ref in [("camera_rays","moge"),("global_scale","camera_rays"),("prior_depth","moge"),
                     ("prior_depth","global_scale"),("wrong_geometry","global_scale")]:
        for mode in ("language","oracle_identity"):
            for task in ("L0","L1","L2","L3","pixel_xyz","distance","height","object_order","separation","static_right","yaw_90"):
                pp=[(q[arm],q[ref]) for q in qid.values() if q[arm]["mode"]==mode and task in (q[arm]["level"],q[arm]["family"])]
                if not pp:
                    continue
                effects=[(q,[100*(int(q["correct"])-int(b["correct"]))]) for q,b in pp]
                row=dict(arm=arm,reference=ref,mode=mode,task=task,**bootstrap(effects,("paired_accuracy_pp",)))
                quantitative=[(q,[float(q["abs_error"])-float(b["abs_error"])]) for q,b in pp if q["abs_error"] and b["abs_error"]]
                if quantitative and task in ("pixel_xyz","distance","height","separation"):
                    row["paired_mae"]=bootstrap(quantitative,("paired_MAE_m",))
                summary.append(row)
        obs=[(p[arm],[float(p[arm]["depth_mae"])-float(p[ref]["depth_mae"]),
                      float(p[arm]["xyz_mae"])-float(p[ref]["xyz_mae"])])
             for p in samples.values() if p[arm]["depth_mae"] and p[ref]["depth_mae"]]
        summary.append(dict(arm=arm,reference=ref,mode="geometry",task="heldout_meters",
                            **bootstrap(obs,("depth_mae","xyz_mae"))))
    write(OUT/"recording_uncertainty.json",summary)


if __name__=="__main__":
    run()

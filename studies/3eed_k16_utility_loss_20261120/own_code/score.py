"""Paired, post-seal scoring on all 18,620 ordinary records, including unrecoverable records."""
import csv
import gzip
import io
import numpy as np

from common import ARMS, DIRECTIONS, HERE, METRICS, OFFSETS, OUT, PRIVATE, R183, atomic, digest, read, write_csv
from predict import PRED, READOUTS

COMPARISONS = list(dict.fromkeys(
    [(a,b) for a in ARMS for b in ("v1","frozen_joint")] +
    [(ARMS[2],b) for b in (ARMS[0],ARMS[1],ARMS[3],ARMS[4],"zero_joint","R183_history")] +
    [(ARMS[0],"R183_history"),("zero_joint","frozen_joint"),("frozen_joint","v1")]))


def measured(iou):
    return np.stack((iou>.25,iou>.50,iou),axis=-1).astype(float)


def ratio(good, available):
    if not np.any(available):
        return np.full(good.shape[1:],np.nan)
    return good[available].mean(axis=0)*100


def average_nonempty(stack):
    a=np.stack(stack)
    den=np.isfinite(a).sum(axis=0)
    return np.divide(np.nansum(a,axis=0),den,out=np.full(a.shape[1:],np.nan),where=den>0)


def recording_interval(blocks):
    # A recording is resampled as a unit, with the same weight for both sources
    # evaluated on that target. The estimand weights all six directions equally.
    rng=np.random.default_rng(4171)
    shifts=np.zeros((2000,len(METRICS)))
    counts={}
    for target in ("waymo","drone","quad"):
        keys=[k for k in DIRECTIONS if k[1]==target]
        recordings=np.unique(np.concatenate([blocks[k][0] for k in keys]))
        n=len(recordings)
        counts[target]=n
        freq=rng.multinomial(n,[1/n]*n,size=len(shifts))
        for key in keys:
            labels,delta=blocks[key]
            observations=np.array([(labels==r).sum() for r in recordings])
            totals=np.stack([delta[labels==r].sum(0) for r in recordings])
            den=freq@observations
            assert (den>0).all()
            shifts+=(freq@totals)/den[:,None]/len(DIRECTIONS)
    bounds=np.percentile(100*shifts,[2.5,97.5],axis=0)
    return dict(recordings=counts,metric_order=list(METRICS),paired_percentage_point_intervals=bounds.tolist())


def score_boxes_once():
    # Import ground truth routines only after checking the whole, immutable four-baseline/five-loss matrix.
    path=PRED/"SEAL.json"
    seal=read(path)
    assert seal["n"]==18620 and tuple(seal["arms"])==READOUTS and tuple(seal["seed_offsets"])==OFFSETS
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    assert seal["inputs"]["source_heads"]==digest(PRIVATE/"heads/LOCK.json")
    assert seal["inputs"]["predict"]==digest(HERE/"predict.py")
    from scope183 import ordinary, check_input
    from score183 import truth
    from evaluate import score_boxes
    rows=ordinary()
    prior=read(R183/"audited_iou_PRIVATE/SEALED.json")
    assert prior["prediction_seal"]==seal["inputs"]["capture"]["parent"]
    result={}
    worst=0.
    old_error=0.
    detail=PRIVATE/"scores"
    detail.mkdir(mode=0o700,exist_ok=True)
    hashes={}
    for s,t in DIRECTIONS:
        key=f"{s}_to_{t}"
        r=rows[t]
        packs=seal["directions"][key]
        count=len(r)
        assert count==packs["n"]
        rec_names={v:f"{t}_{j:03d}" for j,v in enumerate(sorted({row["recording"] for row in r}),1)}
        ious=np.full((count,16),np.nan)
        choices=np.full((count,len(OFFSETS),len(READOUTS)),-1,dtype=np.int16)
        original=np.full(count,np.nan)
        for name,pin in sorted(packs["files"].items()):
            pred=PRED/key/name
            raw=R183/"predictions_PRIVATE"/key/name
            assert digest(pred)==pin["sha256"] and digest(raw)==pin["native"]
            with np.load(pred) as z, np.load(raw) as k16:
                ids=z["ordinals"]
                np.testing.assert_array_equal(ids,np.arange(pin["first"],pin["last"]))
                np.testing.assert_array_equal(ids,k16["ordinals"])
                assert (choices[ids]==-1).all()
                choices[ids]=z["choices"]
                boxes,four=k16["boxes"],k16["four"]
                archived=k16["choices"]
            for j,i in enumerate(ids):
                row=r[int(i)]
                check_input(row)
                target=truth(row["metapath"],row["object_index"],row["caption"],t,row["suffix"])
                measurements,err=score_boxes(np.concatenate((boxes[j],four[j])),target)
                worst=max(worst,err)
                ious[i]=measurements[:16]
                original[i]=measurements[16:].max()
            np.testing.assert_array_equal(choices[ids,0,:3],archived[:,[0,1,5]])
        assert np.isfinite(ious).all() and (choices>=0).all() and (choices<16).all() and np.isfinite(original).all()
        selected=ious[np.arange(count)[:,None,None],choices]
        old=R183/"audited_iou_PRIVATE"/f"{key}.npz"
        assert digest(old)==prior["files"][old.name]["sha256"]
        with np.load(old,allow_pickle=False) as past:
            for now,previous in ((selected[:,0,:3],past["ious"][:,[0,1,5]]),
                                 (ious.max(1),past["k16_best"]),(original,past["four_best"])):
                np.testing.assert_allclose(now,previous,rtol=0,atol=1e-12)
                old_error=max(old_error,float(np.max(np.abs(now-previous))))
                for level in (.25,.50): np.testing.assert_array_equal(now>level,previous>level)
        rec=np.asarray([rec_names[row["recording"]] for row in r])
        dest=detail/f"{key}.npz"
        np.savez_compressed(dest,candidate_ious=ious,choices=choices,four_best=original,recording=rec)
        hashes[dest.name]=digest(dest)
        result[key]=dict(iou=selected,choices=choices,m25=(ious>.25).sum(1),m50=(ious>.50).sum(1),
                         four=original,rec=rec)
        print(key,count,"ordinary boxes scored",flush=True)
    atomic(detail/"SEAL.json",dict(files=hashes,source_prediction_seal=digest(path),records=18620,
          candidate_measurements=18620*16,four_slot_measurements=18620*4,max_dual_IoU_error=worst,
          max_prior_IoU_error=old_error,threshold_disagreements=0,FINAL_access=False))
    return result


def summarize(blocks):
    all_rows={k:np.concatenate([x[k] for x in blocks.values()]) for k in next(iter(blocks.values()))}
    metrics_rows,pair_rows,strata_rows,recording_rows=[],[],[],[]
    main={}
    cond25={}
    cond50={}
    for key,b in blocks.items():
        main[key]=measured(b["iou"]).mean(0)*100
        cond25[key]=ratio(b["iou"]>.25,b["m25"]>0)
        cond50[key]=ratio(b["iou"]>.50,b["m50"]>0)
    for dest in (main,cond25,cond50):
        dest["six_equal"]=average_nonempty(list(dest.values()))
    main["pooled"]=measured(all_rows["iou"]).mean(0)*100
    cond25["pooled"]=ratio(all_rows["iou"]>.25,all_rows["m25"]>0)
    cond50["pooled"]=ratio(all_rows["iou"]>.50,all_rows["m50"]>0)

    for key,b in [*blocks.items(),("six_equal",all_rows),("pooled",all_rows)]:
        v,chosen=b["iou"],b["choices"]
        four_missing=(b["m25"]>0)&(b["four"]<=.25)
        successful=((v>.25)&four_missing[:,None,None]).sum(0)
        for k in (0,1,2,"mean"):
            def take(a):
                return a.mean(0) if k=="mean" else a[k]
            local=take(main[key])
            for j, arm in enumerate(READOUTS):
                metrics_rows.append(dict(direction=key,seed_id=k,arm=arm,n=len(v),
                    **dict(zip(METRICS,local[j],strict=True)),
                    with_Acc25_candidate=int((b["m25"]>0).sum()),with_Acc50_candidate=int((b["m50"]>0).sum()),
                    success_given_covered25=float(take(cond25[key])[j]),success_given_covered50=float(take(cond50[key])[j]),
                    four_misses_reachable=int(four_missing.sum()),selected_when_four_miss=float(take(successful)[j])))
            for after,before in COMPARISONS:
                i,j=READOUTS.index(after),READOUTS.index(before)
                a,z=v[:,:,i],v[:,:,j]
                diff=a-z
                counts=dict(switched=(chosen[:,:,i]!=chosen[:,:,j]).sum(0),
                            iou_improved=(diff>1e-12).sum(0),iou_worsened=(diff< -1e-12).sum(0))
                for n,threshold in ((25,.25),(50,.50)):
                    p,q=a>threshold,z>threshold
                    counts[f"rescue{n}"]=(p&~q).sum(0)
                    counts[f"harm{n}"]=(q&~p).sum(0)
                    assert np.array_equal(counts[f"rescue{n}"]-counts[f"harm{n}"],p.sum(0)-q.sum(0))
                pair_rows.append(dict(direction=key,seed_id=k,method=after,reference=before,n=len(v),
                      **{m+"_pp":local[i,idx]-local[j,idx] for idx,m in enumerate(METRICS)},
                      **{name:float(take(val)) for name,val in counts.items()}))

    bins=((0,0,"0"),(1,1,"1"),(2,3,"2-3"),(4,7,"4-7"),(8,16,"8-16"))
    for lo,hi,label in bins:
        outputs={}
        ns={}
        for key,b in blocks.items():
            mask=(b["m25"]>=lo)&(b["m25"]<=hi)
            ns[key]=int(mask.sum())
            if ns[key]:outputs[key]=measured(b["iou"][mask]).mean((0,1))*100
            else:outputs[key]=np.full((len(READOUTS),3),np.nan)
        nonempty=sum(n>0 for n in ns.values())
        outputs["six_equal"]=average_nonempty(list(outputs.values()))
        ns["six_equal"]=sum(ns.values())
        for key,mat in outputs.items():
            for i,arm in enumerate(READOUTS):
                strata_rows.append(dict(direction=key,positive_count=label,n=ns[key],arm=arm,
                      contributing_directions=nonempty if key=="six_equal" else int(ns[key]>0),
                      **{m:float(mat[i,j]) if np.isfinite(mat[i,j]) else "" for j,m in enumerate(METRICS)}))

    for key,b in blocks.items():
        for rec in np.unique(b["rec"]):
            mask=b["rec"]==rec
            per_seed=measured(b["iou"][mask]).mean(0)*100
            for i,arm in enumerate(READOUTS):
                recording_rows.append(dict(direction=key,recording=rec,arm=arm,n=int(mask.sum()),
                  **dict(zip(METRICS,per_seed.mean(0)[i],strict=True))))
    write_csv(OUT/"target_metrics.csv",metrics_rows)
    write_csv(OUT/"paired.csv",pair_rows)
    write_csv(OUT/"positive_count_strata.csv",strata_rows)
    write_csv(OUT/"recordings.csv",recording_rows)

    # Average the training seeds inside each record before resampling.
    intervals={}
    for a,b in COMPARISONS:
        i,j=READOUTS.index(a),READOUTS.index(b)
        matrix={}
        for (s,t) in DIRECTIONS:
            item=blocks[f"{s}_to_{t}"]
            difference=(measured(item["iou"][:,:,i])-measured(item["iou"][:,:,j])).mean(1)
            matrix[s,t]=(item["rec"],difference)
        intervals[f"{a}_minus_{b}"]=recording_interval(matrix)
    atomic(OUT/"recording_intervals.json",dict(repeats=2000,seed=4171,effects=intervals))
    path=OUT/"all_rows.csv.gz"
    tmp=OUT/"all_rows.tmp.gz"
    with tmp.open("wb") as raw, gzip.GzipFile(filename="",fileobj=raw,mode="wb",mtime=0) as gz, io.TextIOWrapper(gz,newline="") as stream:
        names=["direction","row","recording","positive_count25","positive_count50","four_best",
               *[f"iou_{a}" for a in READOUTS[:4]],*[f"iou_{a}_seed{k}" for a in ARMS for k in range(3)]]
        writer=csv.writer(stream)
        writer.writerow(names)
        for key,b in blocks.items():
            for i,v in enumerate(b["iou"]):
                writer.writerow([key,i,b["rec"][i],b["m25"][i],b["m50"][i],b["four"][i],
                                 *v[0,:4],*v[:,4:].T.ravel()])
    tmp.replace(path)
    products=("target_metrics.csv","paired.csv","positive_count_strata.csv","recordings.csv","recording_intervals.json","all_rows.csv.gz")
    atomic(OUT/"score_receipt.json",dict(files={s:digest(OUT/s) for s in products},evidence=digest(PRIVATE/"scores/SEAL.json"),
          source_prediction_seal=digest(PRED/"SEAL.json"),records=len(all_rows["iou"]),models=45,score_code=digest(HERE/"score.py"),FINAL_access=False))
    print("Scored all paired seeds, readouts and recording intervals",flush=True)


if __name__=="__main__":
    summarize(score_boxes_once())

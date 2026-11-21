"""Read the sealed sign-change replies and reuse the already scored fixed boxes."""
import csv
from side import ARMS, ETA, OUT, PRIVATE, TOL, digest, parse, read, write
from evaluate import COMPARISONS, effect, mean_platform, scopes, table

SWAP = {"positive":"negative","negative":"positive","balanced":"balanced"}


def run():
    seal = read(OUT/"reversal_seal.json")
    primary = read(OUT/"evaluation_receipt.json")
    assert seal["pins"]["primary"] == digest(OUT/"prediction_seal.json")
    assert seal["hashes"] == digest(PRIVATE/"reversed_hashes.json")
    assert primary["files"]["rows"] == digest(OUT/"rows.csv")
    hashes = read(PRIVATE/"reversed_hashes.json")
    prepared = read(PRIVATE/"prepared.json")
    with (OUT/"rows.csv").open() as f:
        rows = list(csv.DictReader(f))
    summary, pairs, public_rows = [],[],[]
    for r in rows:
        uid = r["uid"]
        raw = {}
        if uid in hashes:
            path = PRIVATE/"reversed_replies"/(uid+".json")
            assert digest(path)==hashes[uid]
            response = read(path)
            assert response["uid"]==uid and response["pins"]==seal["pins"]
            raw = {arm:parse(answer) for arm,answer in response["answers"].items()}
        r["C_D_same"] = {"True":True,"False":False,"":None}[r["C_D_same"]]
        r["iou_H"] = float(r["iou_H"])
        r["shift_H"] = 0.
        out = {k:r[k] for k in ("uid","target","recording","panel","separation")}
        for arm in ARMS:
            key = "side_"+arm
            side = r[key]
            other = raw.get(arm, "unavailable")
            explicit = side in ETA and other in ETA
            if explicit:
                status = "agree" if side==SWAP[other] else "conflict"
            else:
                status = "abstain" if raw else "unavailable"
            r["gauge_"+arm] = status
            shifted = status=="agree"
            literal_box = SWAP.get(other,"H")
            out[key] = side
            out["reversed_literal_"+arm] = other
            out["status_"+arm] = status
            out["primary_iou_"+arm] = float(r["iou_"+arm])
            out["relabeled_iou_"+arm] = float(r["iou_"+literal_box])
            r["iou_"+arm] = float(r["iou_"+side]) if shifted else r["iou_H"]
            r["shift_"+arm] = prepared[uid]["geometry"]["offsets"][side] if shifted else 0.
            out["agreement_only_iou_"+arm] = r["iou_"+arm]
        public_rows.append(out)
    for scope,t,group in scopes(rows):
        label = dict(scope=scope,target=t,n=len(group))
        for arm in ARMS:
            statuses = [r["gauge_"+arm] for r in group]
            xs = [r["iou_"+arm] for r in group]
            summary.append(dict(**label,arm=arm,**{k:statuses.count(k) for k in
                           ("agree","conflict","abstain","unavailable")},hits50=sum(x>.5 for x in xs),
                           acc50=100*mean_platform(group,[x>.5 for x in xs]),
                           moved=sum(abs(r["shift_"+arm])>TOL for r in group)))
        for arm,ref in COMPARISONS:
            pairs.append(dict(**label,arm=arm,reference=ref,**effect(group,arm,ref)))
    results = {k:table(k,v) for k,v in (("reversal_rows",public_rows),("reversal_agreement",summary),
                                         ("reversal_paired",pairs))}
    write(OUT/"reversal_receipt.json",dict(seal=digest(OUT/"reversal_seal.json"),
          scored_primary=digest(OUT/"evaluation_receipt.json"), files=results, code=digest(__file__),
          new_GT_reads=0, FINAL_access=False))
    print("Counted",len(public_rows),"paired sign encodings",flush=True)


if __name__=="__main__":
    run()

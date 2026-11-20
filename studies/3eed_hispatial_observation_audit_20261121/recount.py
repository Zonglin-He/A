"""Independently verify released counts, rates and matched identity diagnostics."""
from collections import Counter, defaultdict
import csv
from statistics import mean

from common import ARMS, OUT, PLATFORMS, digest, read, write
from summarize import read_csv


def run():
    rec=read(OUT/"score_receipt.json")
    questions=read_csv("qa_rows.csv")
    geometry=read_csv("geometry_rows.csv")
    consistency=read_csv("consistency_rows.csv")
    assert (len(questions),len(geometry),len(consistency))==(3245,1890,1035)
    assert len({r["qid"] for r in questions})==649
    assert Counter({t:len({r["uid"] for r in questions if r["target"]==t}) for t in PLATFORMS})==dict.fromkeys(PLATFORMS,18)
    byq=defaultdict(dict)
    for r in questions:
        assert r["arm"] not in byq[r["qid"]]
        byq[r["qid"]][r["arm"]]=r
    assert all(set(group)==set(ARMS) for group in byq.values())
    depth_rows={(r["uid"],r["arm"]):r for r in geometry if r["group"]=="all"}
    assert len(depth_rows)==54*len(ARMS)
    assert all(r["n"]==depth_rows[uid,"moge"]["n"] for (uid,arm),r in depth_rows.items())
    audited={"qa_max_pp_error":0.,"geom_max_m_error":0.}
    for r in read_csv("qa_summary.csv"):
        fields=[v for v in questions if v["arm"]==r["arm"] and v["mode"]==r["mode"] and r["task"] in (v["level"],v["family"])]
        counters=[(len(s),sum(int(v["correct"]) for v in s)) for s in
            ([v for v in fields if v["target"]==t] for t in PLATFORMS)]
        if r["target"]=="platform_macro":
            total=sum(n for n,k in counters)
            accuracy=100*mean(k/n for n,k in counters)
        elif r["target"]=="pooled":
            total=sum(n for n,k in counters)
            accuracy=100*sum(k for n,k in counters)/total
        else:
            total,hit=counters[PLATFORMS.index(r["target"])]
            accuracy=100*hit/total
        assert total==int(r["n_questions"])
        audited["qa_max_pp_error"]=max(audited["qa_max_pp_error"],abs(accuracy-float(r["acc_pct"])))
    for r in read_csv("geometry_summary.csv"):
        if r["group"]!="all":
            continue
        available=[v for (u,a),v in depth_rows.items() if a==r["arm"] and
                   (r["target"] not in PLATFORMS or r["target"]==v["target"])]
        estimate=mean(float(v["depth_mae"]) for v in available)
        assert sum(int(v["n"]) for v in available)==int(r["n_holdout"])
        audited["geom_max_m_error"]=max(audited["geom_max_m_error"],abs(estimate-float(r["depth_mae"])))
    assert audited["qa_max_pp_error"] < 1e-10 and audited["geom_max_m_error"] < 1e-10

    improvements={}
    frames={uid for uid,arm in depth_rows}
    for arm in ARMS:
        diff=[float(depth_rows[u,arm]["depth_mae"])-float(depth_rows[u,"moge"]["depth_mae"]) for u in frames]
        improvements[arm]=dict(lower=sum(v<0 for v in diff),higher=sum(v>0 for v in diff),same=sum(v==0 for v in diff))
    write(OUT/"recount.json",dict(**audited,rows=len(questions),exact_question_pairings=len(byq),
        heldout_unique_cells=sum(int(depth_rows[u,"moge"]["n"]) for u in frames),
        geometry_frames_each_arm=len(frames),within_frame_depth_mae_change=improvements,
        sealed_score_receipt=digest(OUT/"score_receipt.json")))

    # Matched language and marked-image questions about exactly the same pair.
    control={(r["uid"],r["task"],r["arm"]):r for r in questions if r["mode"]=="oracle_identity"}
    outcomes=defaultdict(list)
    for r in questions:
        if r["mode"]!="language" or r["task"] not in ("static_right","after_turn_right","after_turn_left"):
            continue
        oracle=control.get((r["uid"],r["task"],r["arm"]))
        if not oracle:
            continue
        assert oracle["static_changes"]==r["static_changes"]
        for target in (r["target"],"pooled"):
            outcomes[target,r["family"],r["arm"],"all"].append((int(r["correct"]),int(oracle["correct"])))
            if r["static_changes"]=="1" and r["static_available"]=="1":
                outcomes[target,r["family"],r["arm"],"requires_turn_change"].append((int(r["correct"]),int(oracle["correct"])))
    output=[]
    for (target,task,arm,which),rows in outcomes.items():
        delta=[b-a for a,b in rows]
        output.append(dict(target=target,task=task,arm=arm,subset=which,n_matched=len(rows),
            language_correct=sum(a for a,b in rows),marked_correct=sum(b for a,b in rows),
            rescue=delta.count(1),harm=delta.count(-1)))
    with (OUT/"matched_identity.csv").open("w",newline="",encoding="utf8") as f:
        out=csv.DictWriter(f,fieldnames=list(output[0]),lineterminator="\n")
        out.writeheader()
        out.writerows(output)
    print(audited, "54 frames,", len(byq), "paired questions")


if __name__=="__main__":
    run()

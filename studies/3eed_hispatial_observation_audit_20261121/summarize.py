"""Recompute platform summaries and paired outcomes from anonymous scalars."""
from collections import defaultdict
import csv
import gzip
from pathlib import Path
from statistics import mean, median

from common import ARMS, OUT, PLATFORMS, digest, write


COMPARISONS=(("camera_rays","moge"),("global_scale","camera_rays"),("global_scale","moge"),
             ("prior_depth","camera_rays"),("prior_depth","moge"),("prior_depth","global_scale"),
             ("wrong_geometry","moge"),("global_scale","wrong_geometry"),("prior_depth","wrong_geometry"))
TARGETS=(*PLATFORMS,"platform_macro","pooled")


def read_csv(name):
    path=OUT/name
    source=path.open("rt",encoding="utf8") if path.exists() else gzip.open(str(path)+".gz","rt",encoding="utf8")
    with source as f:
        return list(csv.DictReader(f))


def save_csv(name,rows):
    path=OUT/name
    with path.open("w",newline="",encoding="utf8") as f:
        fieldnames=list(dict.fromkeys(k for r in rows for k in r))
        out=csv.DictWriter(f,fieldnames=fieldnames,lineterminator="\n")
        out.writeheader()
        out.writerows(rows)


def values(group,key):
    return [float(r[key]) for r in group if r.get(key) not in ("",None)]


def stat(group,key,fn=mean):
    x=values(group,key)
    return fn(x) if x else None


def subsets(rows):
    for target in (*PLATFORMS,"pooled"):
        sub=[r for r in rows if target=="pooled" or r["target"]==target]
        if sub:
            yield target,sub


def add_macros(rows, fields, scores):
    grouped=defaultdict(list)
    for row in rows:
        if row["target"] in PLATFORMS:
            grouped[tuple(row[f] for f in fields)].append(row)
    for group in grouped.values():
        if len(group) != len(PLATFORMS):
            continue
        metadata={k:group[0][k] for k in fields}
        by_count={k:sum(r[k] for r in group) for k in ("n_questions","n_frames","n_valid","n_holdout","n_pairs") if k in group[0]}
        rows.append(dict(target="platform_macro",**metadata,**by_count,**{k:stat(group,k) for k in scores}))
    return rows


def question_summary(rows):
    output=[]
    groups=defaultdict(list)
    for r in rows:
        for task in (r["family"],r["level"]):
            groups[(r["mode"],task,r["arm"])].append(r)
    for (mode,task,arm),block in groups.items():
        for target,r in subsets(block):
            family=task not in ("L0","L1","L2","L3")
            output.append(dict(
                target=target,mode=mode,task=task,arm=arm,n_questions=len(r),
                n_frames=len({v["uid"] for v in r}),n_valid=sum(map(int,[v["valid"] for v in r])),
                acc_pct=100*stat(r,"correct"),valid_pct=100*stat(r,"valid"),
                mae_m=stat(r,"abs_error") if family else None,medae_m=stat(r,"abs_error",median) if family else None,
                xyz_mae_m=stat(r,"point_error") if task=="pixel_xyz" else None,
                raw_changes=sum(int(v["raw_change"]) for v in r),
                answer_changes=sum(int(v["answer_change"]) for v in r),
                direct_acc_pct=100*stat(r,"direct_correct") if r[0]["level"]=="L0" else None,
                direct_valid_pct=100*stat(r,"direct_valid") if r[0]["level"]=="L0" else None,
                direct_mae_m=stat(r,"direct_depth_error") if task=="pixel_xyz" else None,
                direct_xyz_mae_m=stat(r,"direct_point_error") if task=="pixel_xyz" else None))
    return add_macros(output,("mode","task","arm"),
                       ("acc_pct","valid_pct","mae_m","medae_m","xyz_mae_m","direct_acc_pct",
                        "direct_valid_pct","direct_mae_m","direct_xyz_mae_m"))


def paired_summary(rows):
    summary=[]
    questions=defaultdict(dict)
    for r in rows:
        assert r["arm"] not in questions[r["qid"]]
        questions[r["qid"]][r["arm"]]=r
    assert all(set(arms)==set(ARMS) for arms in questions.values())
    for arm,base in COMPARISONS:
        groups=defaultdict(list)
        for answers in questions.values():
            a,b=answers[arm],answers[base]
            for task in (a["family"],a["level"]):
                groups[a["mode"],task].append((a,b))
        for (mode,task),block in groups.items():
            for target in (*PLATFORMS,"pooled"):
                pairs=[(a,b) for a,b in block if target=="pooled" or a["target"]==target]
                if not pairs:
                    continue
                ab=[int(a["correct"])-int(b["correct"]) for a,b in pairs]
                error=[float(a["abs_error"])-float(b["abs_error"]) for a,b in pairs
                       if a["abs_error"] and b["abs_error"]] if task not in ("L0","L1","L2","L3") else []
                summary.append(dict(target=target,mode=mode,task=task,arm=arm,reference=base,
                    n_questions=len(pairs),n_frames=len({a["uid"] for a,b in pairs}),
                    rescue=ab.count(1),harm=ab.count(-1),delta_acc_pp=100*mean(ab),
                    n_valid_both=len(error),delta_mae_m=mean(error) if error else None))
    return add_macros(summary,("mode","task","arm","reference"),("delta_acc_pp","delta_mae_m"))


def geometry_summary(rows):
    summary=[]
    for arm in ARMS:
        groups=defaultdict(list)
        for r in rows:
            if r["arm"]==arm:
                groups[r["group"]].append(r)
        for group,block in groups.items():
            for target,rr in subsets(block):
                summary.append(dict(target=target,group=group,arm=arm,
                    n_frames=sum(int(r["n"])>0 for r in rr),n_holdout=sum(int(r["n"]) for r in rr),
                    **{k:stat(rr,k) for k in ("depth_mae","depth_rmse","depth_absrel","xyz_mae")}))
    return add_macros(summary,("group","arm"),("depth_mae","depth_rmse","depth_absrel","xyz_mae"))


def logic_summary(rows):
    output=[]
    groups=defaultdict(list)
    for r in rows:
        groups[r["mode"],r["task"],r["arm"]].append(r)
    for (mode,task,arm),rr in groups.items():
        for target,r in subsets(rr):
            output.append(dict(target=target,mode=mode,task=task,arm=arm,n_pairs=len(r),
                valid_pct=100*stat(r,"valid_both"),consistency_pct=100*stat(r,"consistent"),
                both_correct_pct=100*stat(r,"both_correct")))
    return add_macros(output,("mode","task","arm"),("valid_pct","consistency_pct","both_correct_pct"))


def perspective(rows):
    lookup={(r["uid"],r["mode"],r["arm"]):r for r in rows if r["task"]=="static_right"}
    data=[]
    for mode in ("language","oracle_identity"):
        for arm in ARMS:
            selected=[r for r in rows if r["level"]=="L3" and r["mode"]==mode and r["arm"]==arm]
            for subset,questions in [("all",selected),("changes_static",[r for r in selected if r["static_available"]=="1" and r["static_changes"]=="1"])]:
                for target,cases in subsets(questions):
                    n=0
                    carried=[]
                    got=[]
                    identical=[]
                    for row in cases:
                        static=lookup.get((row["uid"],mode,arm))
                        if static:
                            n+=1
                            carried.append(int(static["valid"]=="1" and static["correct"]!=row["static_changes"]))
                            got.append(int(row["correct"]))
                            if static["valid"]=="1" and row["valid"]=="1":
                                same_grade = static["correct"]==row["correct"]
                                identical.append(int(same_grade == (row["static_changes"]=="0")))
                    data.append(dict(target=target,mode=mode,subset=subset,arm=arm,
                        n_questions=len(cases),n_static=n,accuracy_pct=100*stat(cases,"correct"),
                        accuracy_static_matched_pct=100*mean(got) if n else None,
                        static_carry_pct=100*mean(carried) if n else None,
                        static_and_turn_valid=len(identical), answer_same_as_static_pct=100*mean(identical) if identical else None))
    return add_macros(data,("mode","arm","subset"),("accuracy_pct","accuracy_static_matched_pct",
                      "static_carry_pct","answer_same_as_static_pct"))


def mechanisms(rows):
    by_q=defaultdict(dict)
    for r in rows:
        if r["mode"]=="language":
            by_q[r["qid"]][r["arm"]]=r
    result=[]
    for arm in ("global_scale","prior_depth","wrong_geometry"):
        comparisons=[(group[arm],group["moge"]) for group in by_q.values()]
        for level in ("L0","L1","L2","L3"):
            q=[(a,b) for a,b in comparisons if a["level"]==level]
            for target in (*PLATFORMS,"pooled"):
                block=[(a,b) for a,b in q if target=="pooled" or a["target"]==target]
                supports=[(a,b) for a,b in block if int(a["min_object_holdout"])>=3]
                tests=[
                    ("global_depth_improved",[(a,b) for a,b in block if a["global_depth_mae"] and b["global_depth_mae"]
                      and float(a["global_depth_mae"])<float(b["global_depth_mae"])-1e-5],len(block)),
                    ("all_named_objects_depth_improved",[(a,b) for a,b in supports if a["all_objects_depth_improved"]=="1"],len(supports))]
                for group, selected, eligible in tests:
                    if not eligible:
                        continue
                    effects=[int(a["correct"])-int(b["correct"]) for a,b in selected]
                    result.append(dict(target=target,arm=arm,level=level,condition=group,
                      n_eligible=eligible,n_questions=len(selected),n_frames=len({a["uid"] for a,b in selected}),
                      rescue=effects.count(1),harm=effects.count(-1),same=effects.count(0)))
    return result


def run():
    q=read_csv("qa_rows.csv")
    g=read_csv("geometry_rows.csv")
    c=read_csv("consistency_rows.csv")
    assert len({(r["qid"],r["arm"]) for r in q})==len(q)
    assert len({r["uid"] for r in q})==54
    assert all(len({r["uid"] for r in q if r["target"]==t})==18 for t in PLATFORMS)
    assert len(q)==len({r["qid"] for r in q})*len(ARMS)
    assert all(len({r["n"] for r in g if (r["uid"],r["group"])==key})==1
               for key in {(r["uid"],r["group"]) for r in g})
    specs={
        "qa_summary.csv":question_summary(q),
        "paired.csv":paired_summary(q),
        "geometry_summary.csv":geometry_summary(g),
        "consistency_summary.csv":logic_summary(c),
        "perspective.csv":perspective(q),
        "mechanisms.csv":mechanisms(q),
    }
    for name,rows in specs.items():
        save_csv(name,rows)
    write(OUT/"summary_receipt.json",dict(questions=len(q)//len(ARMS),answers=len(q),
         geometry_frame_arm_groups=len(g),consistency_pairs=len(c), files={k:digest(OUT/k) for k in specs}))


if __name__=="__main__":
    run()

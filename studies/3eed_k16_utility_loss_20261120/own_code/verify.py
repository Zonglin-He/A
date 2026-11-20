"""Independently reconstruct the score tables, historical controls and a sample of decisions."""
import csv
import gzip
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F

from common import ARMS, DIRECTIONS, HERE, METRICS, OUT, PRIVATE, R183, atomic, digest, read, register, write_csv


def table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def independent_replay(pins):
    """Use functional linear layers, independent argmax and explicit incumbent handling."""
    torch.set_num_threads(1)
    checks = 0
    for (s, t) in DIRECTIONS:
        key = f"{s}_to_{t}"
        files = sorted(pins["directions"][key]["files"])
        for filename in (files[0], files[len(files)//2], files[-1]):
            with np.load(PRIVATE/"features"/key/filename) as f, np.load(PRIVATE/"predictions"/key/filename) as q:
                for row in (0, len(f["ordinals"])//2, len(f["ordinals"])-1):
                    x = torch.from_numpy(f["features"][row])
                    prior = torch.from_numpy(f["joint"][row])
                    incumbent = int(f["inc"][row])
                    for seed in range(3):
                        for col, arm in enumerate(ARMS, start=4):
                            path = PRIVATE/"heads"/f"{s}_{seed}_{arm}.pt"
                            w = torch.load(path, weights_only=True, map_location="cpu")["state"]
                            h = F.gelu(F.linear(x,w["score.0.weight"],w["score.0.bias"]))
                            s_new = prior + F.linear(h,w["score.3.weight"],w["score.3.bias"]).flatten()
                            ix = int(s_new.argmax())
                            if s_new[ix] == s_new[incumbent]:
                                ix = incumbent
                            assert ix == q["choices"][row,seed,col]
                            checks += 1
    return checks


def window(values):
    return np.stack([(values > .25).mean(0), (values > .50).mean(0), values.mean(0)], axis=-1)*100


def near(x,y):
    np.testing.assert_allclose(x, y, rtol=0, atol=2e-10)


def audit():
    register()
    score_receipt = read(OUT/"score_receipt.json")
    for file, hash_value in score_receipt["files"].items():
        assert digest(OUT/file) == hash_value
    seals = read(PRIVATE/"predictions/SEAL.json")
    checks = independent_replay(seals)
    labels = tuple(seals["arms"])
    data = defaultdict(list)
    with gzip.open(OUT/"all_rows.csv.gz","rt",newline="") as stream:
        for item in csv.DictReader(stream):
            data[item["direction"]].append(item)
    assert set(data) == {f"{s}_to_{t}" for s,t in DIRECTIONS}
    rebuilt, measurements, support25, fourouts = {}, {}, {}, {}
    margin_tables = []
    phase = lambda x: (x > .25).astype(int) + (x > .5).astype(int)
    total = 0
    for key, rows in data.items():
        ids = [int(x["row"]) for x in rows]
        assert ids == list(range(len(rows)))
        total += len(rows)
        arr = np.empty((len(rows), 3, len(labels)))
        for j,a in enumerate(labels):
            for k in range(3):
                col = f"iou_{a}" if j < 4 else f"iou_{a}_seed{k}"
                arr[:,k,j] = [float(x[col]) for x in rows]
        assert np.isfinite(arr).all() and ((0<=arr)&(arr<=1)).all()
        with np.load(PRIVATE/"scores"/f"{key}.npz") as target:
            ref = target["candidate_ious"]
            picks = target["choices"]
            np.testing.assert_array_equal(arr,ref[np.arange(len(rows))[:,None,None],picks])
            counts25 = (ref>.25).sum(1)
            counts50 = (ref>.50).sum(1)
            np.testing.assert_array_equal(counts25,[int(r["positive_count25"]) for r in rows])
            np.testing.assert_array_equal(counts50,[int(r["positive_count50"]) for r in rows])
            eligible4 = (counts25>0) & (target["four_best"]<=.25)
            fourouts[key] = (eligible4, (arr[eligible4]>.25).sum(0))
        support25[key] = counts25
        rebuilt[key], measurements[key] = window(arr), arr
        after = phase(arr[:,:,labels.index(ARMS[2])])
        for arm in (ARMS[0],ARMS[1],ARMS[3],ARMS[4],"frozen_joint"):
            before = phase(arr[:,:,labels.index(arm)])
            for seed in range(3):
                flow = np.bincount(3*before[:,seed]+after[:,seed],minlength=9).reshape((3,3))
                assert flow.sum() == len(rows)
                for a in range(3):
                    for b in range(3):
                        margin_tables.append(dict(direction=key, seed=seed, prior=arm,
                            prior_iou_band=("miss","25_only","50")[a], PAL_iou_band=("miss","25_only","50")[b],
                            count=int(flow[a,b])))
    assert total == 18620
    rebuilt["six_equal"] = np.mean(list(rebuilt.values()),axis=0)
    pooled = np.concatenate(list(measurements.values()))
    rebuilt["pooled"] = window(pooled)
    checks_metric = 0
    for r in table(OUT/"target_metrics.csv"):
        key, j = r["direction"], labels.index(r["arm"])
        matrix = rebuilt[key]
        trial = matrix.mean(0)[j] if r["seed_id"]=="mean" else matrix[int(r["seed_id"]),j]
        near(trial,[float(r[m]) for m in METRICS])
        if key in data:
            eligible, passed = fourouts[key]
            near(float(r["four_misses_reachable"]),eligible.sum())
            selected = passed.mean(0)[j] if r["seed_id"]=="mean" else passed[int(r["seed_id"]),j]
            near(float(r["selected_when_four_miss"]),selected)
        checks_metric += 3
    checks_pair = 0
    for r in table(OUT/"paired.csv"):
        key, i,j = r["direction"], labels.index(r["method"]), labels.index(r["reference"])
        v = measurements.get(key,pooled)
        for threshold,column in ((.25,"25"),(.50,"50")):
            a,b = v[:,:,i]>threshold, v[:,:,j]>threshold
            for count,kind in (((a&~b).sum(0),"rescue"),((b&~a).sum(0),"harm")):
                n = count.mean() if r["seed_id"]=="mean" else count[int(r["seed_id"])]
                near(n,float(r[kind+column]))
                checks_pair += 1
    legacy = {"v1":0,"frozen_joint":1,"rank_joint":2}
    checks_old = 0
    for r in table(R183/"results/direction.csv"):
        if r["arm"] in legacy:
            key = "pooled" if r["direction"]=="all_pooled" else r["direction"]
            near(rebuilt[key][0,legacy[r["arm"]]],[float(r[m]) for m in METRICS])
            checks_old += 3
    n4 = sum(int(x[0].sum()) for x in fourouts.values())
    assert n4==4001
    contributions = []
    for arm in ("frozen_joint", ARMS[1], ARMS[3]):
        j, i = labels.index(arm), labels.index(ARMS[2])
        for threshold, metric in ((.25,"Acc25"),(.50,"Acc50")):
            for seed in range(3):
                total_shift = 0.
                for lo, hi, name in ((0,0,"0"), (1,1,"1"), (2,3,"2-3"), (4,7,"4-7"), (8,16,"8-16")):
                    rescues, harms, effect = 0, 0, 0.
                    for key,v in measurements.items():
                        mask=(support25[key]>=lo)&(support25[key]<=hi)
                        before,after=v[mask,seed,j]>threshold,v[mask,seed,i]>threshold
                        r,h=int((after&~before).sum()),int((before&~after).sum())
                        rescues+=r
                        harms+=h
                        effect+=(r-h)*100/(len(v)*len(DIRECTIONS))
                    total_shift+=effect
                    contributions.append(dict(arm=ARMS[2],reference=arm,metric=metric,seed=seed,
                           positives25=name,rescues=rescues,harms=harms,contribution_to_six_equal_pp=effect))
                ix=METRICS.index(metric)
                near(total_shift,rebuilt["six_equal"][seed,i,ix]-rebuilt["six_equal"][seed,j,ix])
    write_csv(OUT/"PAL_threshold_transitions.csv", margin_tables)
    write_csv(OUT/"PAL_positive_count_contributions.csv", contributions)
    atomic(OUT/"audit.json",dict(records=total,directions=len(data),heads=45,per_model_example_replays=checks,
           metric_cells_recomputed=checks_metric,paired_count_cells_recomputed=checks_pair,
           historical_metric_cells_reproduced=checks_old,four_slot_oracle_misses_reachable_in_K16=n4,
           target_positive_rows=int(sum((m>0).sum() for m in support25.values())),
           post_score_hash=digest(OUT/"score_receipt.json"),diagnostic_transitions=digest(OUT/"PAL_threshold_transitions.csv"),
           positive_count_contributions=digest(OUT/"PAL_positive_count_contributions.csv"),
           source_profile=digest(OUT/"source_profile.csv"),production_lock_verified=True,FINAL_access=False))
    print("Independently reproduced the decisions, metrics, paired counts and historical controls.",flush=True)


if __name__=="__main__":
    audit()

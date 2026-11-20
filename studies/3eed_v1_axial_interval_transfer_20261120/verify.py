"""Independent scalar replay and descriptive accounting after prediction sealing."""
import csv
import gzip
from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np

from axial import ARMS, CONFIG, PLATFORMS, PRIVATE, R183, RESULTS, SOURCE, atomic, digest, ordinary, read
from fit import csv_rows


def load_csv(name):
    with (RESULTS/name).open(newline="") as f:
        return list(csv.DictReader(f))


def rotation(view):
    a = np.deg2rad([0, -45, 45, -90, 90][view])
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def read_sample(row, seed):
    path = Path(row["pointpath"])
    full = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.fromfile(path, np.float32).reshape(-1, 4)
    indices = np.random.default_rng(seed).choice(full.shape[0], size=16384, replace=full.shape[0] < 16384)[:8192]
    model = np.array(full[indices, :3], dtype=np.float32)
    p = row["platform"]
    if p == "drone":
        model[:, 2] += np.float32(1.8)
    elif p == "waymo":
        rot = rotation(int(row["suffix"]))
        model = (model @ rot.T).astype(np.float32)
    sensor = model.astype(np.float64)
    if p == "waymo":
        sensor = sensor @ rot
    if p == "drone":
        sensor[:, 2] -= 1.8
    return model, sensor


def center_axis(box):
    theta = box[6] + (np.pi / 2 if box[4] > box[3] else 0)
    e = np.array([np.cos(theta), np.sin(theta)])
    if (e[0] < -1e-12) or (abs(e[0]) <= 1e-12 and e[1] < 0):
        e *= -1
    return e


def histogram(cloud, box):
    long, short = max(box[3:5]), min(box[3:5])
    e = center_axis(box)
    xy = cloud[:, :2] - box[:2]
    u = xy @ e
    v = xy @ np.array([-e[1], e[0]])
    inside = (np.abs(u) <= long*5/6) & (np.abs(v) <= short/2) & (np.abs(cloud[:, 2]-box[2]) <= box[5]/2)
    points = np.unique(cloud[inside], axis=0)
    offsets = (points[:, :2] - box[:2]) @ e
    # Count the nine internal boundaries strictly to the right of a return.
    bins = np.searchsorted(long*np.arange(-4, 5)/6, offsets, side="right")
    counts = np.bincount(bins, minlength=10)
    return np.r_[counts/max(1,len(points)), np.log1p(len(points)), short/long, box[5]/long]


def policy(x, weights, kind):
    def once(a):
        f = np.c_[(a-weights["mean"])/weights["scale"], np.ones(len(a))]
        s = f @ weights[kind]
        if kind.endswith("bce"):
            return np.exp(-np.logaddexp(0., -s))
        p = np.exp(s-s.max(axis=1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        spans = [(lo, hi) for lo in range(5) for hi in range(lo, 5)]
        return np.stack([p[:,[i+1 for i,(lo,hi) in enumerate(spans) if lo<=j<=hi]].sum(1) for j in range(5)], axis=1)
    xflip = np.c_[x[:,:10][:,::-1], x[:,10:]]
    return (once(x) + once(xflip)[:,::-1]) / 2


def recount_transitions(val, base):
    return {f"{name}{cut}":int(mask.sum())
            for cut in (25,50)
            for name,mask in (("rescue",(val>cut/100)&(base<=cut/100)),
                              ("harm",(val<=cut/100)&(base>cut/100)))}


def run():
    from score183 import truth
    from batch_yaw_iou import many_iou

    original = read(R183/"predictions_PRIVATE/SEALED.json")
    predseal = read(PRIVATE/"predictions/SEAL.json")
    goldseal = read(PRIVATE/"scores/SEAL.json")
    assert predseal["target_updates"]==0 and not predseal["target_labels_read"] and not goldseal["FINAL_access"]
    assert goldseal["prediction_seal"] == digest(PRIVATE/"predictions/SEAL.json")
    roster = ordinary()
    directions = load_csv("directions.csv")
    pairs = load_csv("paired.csv")
    summary = read(RESULTS/"source_fit.json")
    nsource = 0
    source_seed = read(SOURCE/"protocol.json")["point_seed"]
    for platform in PLATFORMS:
        for subset in ("fit","calibration"):
            tag = f"{platform}_{subset}"
            manifest = read(SOURCE/"inputs_PRIVATE"/f"{tag}.json")
            folder = SOURCE/"features_PRIVATE"/tag
            offset = 0
            for file in sorted(read(folder/"SEAL.json")["files"]):
                with np.load(folder/file, allow_pickle=False) as block:
                    cached = block["points"][block["origin"]<2]
                for cube, row in zip(cached, manifest[offset:offset+len(cached)]):
                    model, _ = read_sample(row, source_seed)
                    np.testing.assert_array_equal(model, cube)
                offset += len(cached)
            assert offset == len(manifest)
            nsource += offset

    with gzip.open(RESULTS/"all_rows.csv.gz", "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 18620
    breakdown, largest = [], []
    profile_error = prob_error = center_error = 0.
    max_raw_diff = 0.
    checked_rates = {}
    for target, inputs in roster.items():
        sources = [p for p in PLATFORMS if p!=target]
        files = {}
        for src in sources:
            key = f"{src}_to_{target}"
            record = predseal["files"][key]
            filename = PRIVATE/"predictions"/f"{key}.npz"
            assert digest(filename)==record["sha256"]
            with np.load(filename, allow_pickle=False) as saved:
                files[src] = {k:saved[k] for k in ("boxes","choices","profiles","hit_probabilities","mirror")}
            upstream = R183/"predictions_PRIVATE"/key
            parts = []
            for name in sorted(original["directions"][key]["files"]):
                with np.load(upstream/name, allow_pickle=False) as block:
                    parts.append(block["boxes"][np.arange(len(block["choices"])), block["choices"][:,0]])
            np.testing.assert_array_equal(np.vstack(parts), files[src]["boxes"][:,0])
        for i,row in enumerate(inputs):
            _, raw = read_sample(row, row["seed"])
            for src in sources:
                profile = histogram(raw, files[src]["boxes"][i,0])
                profile_error = max(profile_error,float(np.max(np.abs(profile-files[src]["profiles"][i]))))

        for src, pack in files.items():
            key = f"{src}_to_{target}"
            with np.load(PRIVATE/"heads"/f"{src}.npz", allow_pickle=False) as data:
                weights = dict(data)
            with np.load(PRIVATE/"scores"/f"{key}.npz", allow_pickle=False) as gold:
                five, k16 = gold["five"], gold["k16_best"]
                np.testing.assert_array_equal(gold["choices"], pack["choices"])
            assert digest(PRIVATE/"scores"/f"{key}.npz") == goldseal["files"][key]
            n = len(five)
            order = sorted(range(n),key=lambda i:sha256(f"axial-mirror-v1|{target}|{i}".encode()).hexdigest())
            mask = np.zeros(n, dtype=bool)
            mask[order[:n//2]] = True
            np.testing.assert_array_equal(mask, pack["mirror"])
            for j, arm in enumerate(ARMS[1:],1):
                x = pack["profiles"].copy()
                if arm.endswith("_mirror"):
                    x[mask,:10] = x[mask,:10][:,::-1]
                score = policy(x, weights, arm.removesuffix("_mirror"))
                prob_error = max(prob_error,float(np.max(np.abs(score-pack["hit_probabilities"][:,j-1]))))
                pref = sorted(range(5), key=lambda k:(abs(k-2),-k))
                choice = np.array([next(k for k in pref if v[k]>=v.max()-1e-12) for v in score])
                np.testing.assert_array_equal(choice, pack["choices"][:,j])
            anchors = pack["boxes"][:,0]
            e = np.stack([center_axis(b) for b in anchors])
            expected = (pack["choices"].astype(float)-2)/6*anchors[:,3:5].max(1)[:,None]
            delta = pack["boxes"][:,:,:2] - anchors[:,None,:2]
            center_error = max(center_error,float(np.abs(delta-expected[:,:,None]*e[:,None,:]).max()))
            np.testing.assert_array_equal(pack["boxes"][:,:,2:],np.repeat(anchors[:,None,2:],len(ARMS),axis=1))
            selected = np.take_along_axis(five,pack["choices"],axis=1)
            possible = (five[:,2]<=.5) & (five.max(1)>.5)
            outside = possible & (k16<=.5)
            # Rebuild the previous feasibility study's unsigned PCA action set.
            # Canonicalizing the axis may reverse the order, but must preserve
            # each record's eligibility for the original 491-item comparison.
            prior_mask = []
            for box, row, old_best in zip(anchors, inputs, k16):
                theta = box[6] + (np.pi/2 if box[4]>box[3] else 0.)
                direction = np.array([np.cos(theta),np.sin(theta)])*max(box[3:5])
                old_actions = np.tile(box,(5,1))
                old_actions[:,:2] += np.arange(-2,3)[:,None]/6*direction
                label = truth(row["metapath"],row["object_index"],row["caption"],target,row["suffix"])
                old_values = many_iou(old_actions,label)
                prior_mask.append(old_best<=.5 and old_values.max()>.5)
            np.testing.assert_array_equal(outside,prior_mask)
            public = [r for r in rows if r["direction"]==key]
            def public_key(r):
                return (int(r["outside_k16_opportunity50"]),float(r["five_best_iou"]),float(r["k16_best_iou"]),
                        int(r["mirror_applied"]), *(float(r[a+"_iou"]) for a in ARMS), *(int(r[a+"_moved"]) for a in ARMS))
            local = Counter((int(outside[i]), max(five[i]),k16[i],int(mask[i]),*selected[i],*(pack["choices"][i]!=2)) for i in range(n))
            assert local == Counter(public_key(r) for r in public)
            for j,arm in enumerate(ARMS):
                actual = np.array([float(r[arm+"_iou"]) for r in public])
                base = np.array([float(r["v1_iou"]) for r in public])
                rates = np.array([100*np.count_nonzero(actual>.25)/n, 100*np.count_nonzero(actual>.5)/n, 100*np.mean(actual)])
                checked_rates[key,arm] = rates
                report = next(r for r in directions if r["direction"]==key and r["arm"]==arm)
                np.testing.assert_allclose(rates,[float(report[k]) for k in ("Acc25","Acc50","mIoU")],rtol=0,atol=1e-11)
                assert all(int(report[k])==v for k,v in recount_transitions(actual,base).items())
                for label,filt in (("outside_k16",outside),("all_repairable",possible)):
                    has_left = (five[:,:2]>.5).any(1)
                    has_right = (five[:,3:]>.5).any(1)
                    assert ((has_left ^ has_right)[filt]).all()
                    act = pack["choices"][:,j].astype(int)-2
                    good = selected[:,j]>.5
                    counts = dict(rescued=int((filt&good).sum()), held_center=int((filt&(act==0)).sum()),
                          incorrect_side=int((filt & (((act<0)&has_right)|((act>0)&has_left))).sum()),
                          correct_side_missed_magnitude=int((filt&~good&(((act>0)&has_right)|((act<0)&has_left))).sum()))
                    assert sum(counts.values())==filt.sum()
                    breakdown.append(dict(direction=key,arm=arm,group=label,n=int(filt.sum()),**counts))
            for r in pairs:
                if r["direction"]!=key:
                    continue
                a,b = (ARMS.index(r[k]) for k in ("arm","ref"))
                assert all(int(r[k])==v for k,v in recount_transitions(selected[:,a],selected[:,b]).items())
                assert int(r["different_position"])==(pack["choices"][:,a]!=pack["choices"][:,b]).sum()
                d = checked_rates[key,ARMS[a]]-checked_rates[key,ARMS[b]]
                max_raw_diff=max(max_raw_diff,float(np.max(np.abs(d-[float(r[k]) for k in ("Acc25_pp","Acc50_pp","mIoU_pp")]))))
            j=ARMS.index("shift_interval")
            for case, ok in (("rescue",(five[:,2]<=.5)&(selected[:,j]>.5)),("harm",(five[:,2]>.5)&(selected[:,j]<=.5))):
                candidates = np.flatnonzero(ok)
                if len(candidates):
                    i = candidates[np.argmax(np.abs(selected[candidates,j]-five[candidates,2]))]
                    largest.append(dict(direction=key,transition50=case,v1_iou=five[i,2],selected_iou=selected[i,j],
                         center_shift_in_sixths=int(pack["choices"][i,j])-2,k16_best=k16[i],outside_k16_opportunity=int(outside[i])))
    for r in directions+pairs:
        if r["direction"]!="six_equal":
            continue
        paired = "ref" in r
        values = [v-(checked_rates[key,r["ref"]] if paired else 0.) for (key,a),v in checked_rates.items() if a==r["arm"]]
        fields = [m+("_pp" if paired else "") for m in ("Acc25","Acc50","mIoU")]
        np.testing.assert_allclose(np.mean(values,axis=0),[float(r[m]) for m in fields],atol=1e-11,rtol=0)
        comparable = [t for t in (pairs if paired else directions) if t["direction"]!="six_equal" and t["arm"]==r["arm"]
                      and (not paired or t["ref"]==r["ref"])]
        for col in ("n","rescue25","harm25","rescue50","harm50","different_position" if paired else "rescued_no_k16_50"):
            assert int(r[col])==sum(int(t[col]) for t in comparable)
    totals = []
    for arm in ARMS:
        for label in ("outside_k16","all_repairable"):
            selected = [r for r in breakdown if r["arm"]==arm and r["group"]==label]
            totals.append(dict(direction="all_records",arm=arm,group=label,**{k:sum(r[k] for r in selected)
                  for k in ("n","rescued","held_center","incorrect_side","correct_side_missed_magnitude")}))
    assert profile_error<1e-12 and center_error<1e-12 and prob_error<1e-12 and max_raw_diff<1e-11
    csv_rows(RESULTS/"opportunity_breakdown.csv",breakdown+totals)
    csv_rows(RESULTS/"largest_threshold_crossings.csv",largest)
    for data in summary.values():
        for h in data["heads"].values():
            assert h["steps"]==1024 and h["rows_per_step"]==3840 and h["stationarity_l2"]<1e-6
    access = dict(protocol_sha256=digest(PRIVATE.parent/"PROTOCOL.json"),lock_sha256=digest(PRIVATE/"LOCK.json"),
             head_seal=digest(PRIVATE/"heads/SEAL.json"),prediction_seal=digest(PRIVATE/"predictions/SEAL.json"),
             score_seal=digest(PRIVATE/"scores/SEAL.json"))
    atomic(RESULTS/"audit.json",dict(records=len(rows),unique_manifest_expressions=sum(map(len,roster.values())),
           source_raw_samples_reproduced=nsource,complete_raw_target_profiles=len(rows),
           profile_max_difference=profile_error,checked_target_arm_predictions=len(rows)*len(ARMS),
           largest_probability_difference=prob_error,horizontal_translation_max_error=center_error,
           noncenter_coordinate_changes=0,different_winner_vs_registered_v1=0,anonymous_scalar_rows_reconciled=len(rows),
           direction_metric_rows=len(directions),paired_metric_rows=len(pairs),max_direction_pair_difference_pp=max_raw_diff,
           source_budget=dict(heads=12,source_instances_per_head=640,exposures_per_step=3840,steps_per_head=1024),
           oriented_ious_crosschecked=goldseal["boxes_crosschecked"],iou_max_error=goldseal["independent_iou_max_difference"],
           prior_outside_k16_membership_compared=18620,prior_outside_k16_membership_disagreements=0,
           v1_replay_max_iou_error=goldseal["v1_original_iou_max_difference"],target_updates=0,FINAL_access=False,receipts=access))
    print(nsource, "source samples;",len(rows), "target profiles and anonymized rows audited")


if __name__ == "__main__":
    run()

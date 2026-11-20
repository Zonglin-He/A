"""Score the one sealed target run. All thresholds use strict oriented 3D IoU."""
import csv
import gzip
from time import perf_counter

import numpy as np

from axial import (ACTIONS, ARMS, CONFIG, DIRECTIONS, PAL, PLATFORMS, PRIVATE, R183, RESULTS,
                   atomic, digest, ordinary, read, register, translated)
from fit import csv_rows

LEVELS = (.25, .5)
COMPARISONS = [(a, "v1") for a in ARMS[1:]] + [
    ("shift_interval", "repeat_interval"), ("shift_bce", "repeat_bce"),
    ("shift_interval", "shift_bce"), ("repeat_interval", "repeat_bce"),
    ("shift_interval", "shift_interval_mirror"), ("shift_bce", "shift_bce_mirror")]
NAMES = ("Acc25", "Acc50", "mIoU")


def measures(x):
    return np.stack((x > .25, x > .5, x), axis=-1).astype(float)


def score_once():
    prediction = PRIVATE/"predictions"
    seal = read(prediction/"SEAL.json")
    assert seal["lock"] == register() and seal["direction_records"] == 18620
    assert tuple(seal["arms"]) == ARMS and len(seal["files"]) == 6
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    assert seal["heads"] == digest(PRIVATE/"heads/SEAL.json")
    for name, receipt in seal["files"].items():
        assert digest(prediction/f"{name}.npz") == receipt["sha256"]

    # Target task labels and pre-existing scored pools first enter after all arms are sealed.
    from score183 import truth
    from evaluate import score_boxes
    old_pool = PAL/"run_PRIVATE/scores"
    assert digest(old_pool/"SEAL.json") == CONFIG["target"]["postseal_candidate_iou_seal"]
    pool_seal = read(old_pool/"SEAL.json")
    roster = ordinary()
    out = PRIVATE/"scores"
    out.mkdir(exist_ok=True, mode=0o700)
    if (out/"SEAL.json").exists():
        raise RuntimeError("Scores have already been sealed")
    tick = perf_counter()
    files, dual_error, historical_error = {}, 0., 0.
    for s,t in DIRECTIONS:
        key = f"{s}_to_{t}"
        path = prediction/f"{key}.npz"
        with np.load(path, allow_pickle=False) as pack:
            choices, boxes = pack["choices"], pack["boxes"]
            mask = pack["mirror"]
            np.testing.assert_array_equal(pack["ordinals"], np.arange(len(roster[t])))
        prior_file = old_pool/path.name
        assert digest(prior_file) == pool_seal["files"][path.name]
        with np.load(prior_file, allow_pickle=False) as pack:
            old = pack["candidate_ious"]
            winner = pack["choices"][:, 0, 0]
            recordings = pack["recording"]
        actual = np.empty((len(choices), len(ACTIONS)))
        for i, row in enumerate(roster[t]):
            target = truth(row["metapath"], row["object_index"], row["caption"], t, row["suffix"])
            actions = translated(boxes[i,0], ACTIONS)
            np.testing.assert_array_equal(actions[choices[i]], boxes[i])
            actual[i], err = score_boxes(actions, target)
            dual_error = max(dual_error, err)
        prior = old[np.arange(len(old)), winner]
        historical_error = max(historical_error, float(np.abs(actual[:, 2]-prior).max()))
        np.testing.assert_allclose(actual[:, 2], prior, atol=1e-12, rtol=0)
        for tau in LEVELS:
            np.testing.assert_array_equal(actual[:, 2]>tau, prior>tau)
        dest = out/path.name
        np.savez_compressed(dest, five=actual, choices=choices, k16_best=old.max(1),
                            mirror=mask, recording=recordings)
        files[key] = digest(dest)
        print(key, len(actual), "scored", round(perf_counter()-tick,1), flush=True)
    atomic(out/"SEAL.json", dict(files=files, prediction_seal=digest(prediction/"SEAL.json"),
             n=18620, boxes_crosschecked=18620*5, independent_iou_max_difference=float(dual_error),
             v1_original_iou_max_difference=historical_error, threshold_disagreements=0,
             score_seconds=perf_counter()-tick, FINAL_access=False))


def read_blocks():
    folder = PRIVATE/"scores"
    seal = read(folder/"SEAL.json")
    assert seal["prediction_seal"] == digest(PRIVATE/"predictions/SEAL.json")
    blocks = {}
    for key, pin in seal["files"].items():
        file = folder/f"{key}.npz"
        assert digest(file) == pin
        with np.load(file, allow_pickle=False) as p:
            b = {k:p[k] for k in p.files}
        b["picked"] = np.take_along_axis(b["five"], b["choices"], axis=1)
        b["possible"] = (b["five"].max(1)>.5)&(b["five"][:,2]<=.5)
        b["outside"] = b["possible"] & (b["k16_best"]<=.5)
        blocks[key] = b
    return blocks


def paired_counts(x, ref):
    counts = {}
    for cutoff, tag in ((.25, "25"), (.5, "50")):
        counts["rescue"+tag] = int(((x>cutoff)&(ref<=cutoff)).sum())
        counts["harm"+tag] = int(((x<=cutoff)&(ref>cutoff)).sum())
    return counts


def summarize(blocks):
    outcomes, paired, scope, intervals = [], [], [], []
    raw = {k:b["picked"] for k,b in blocks.items()}
    all_b = {k:np.concatenate([b[k] for b in blocks.values()]) for k in
             ("five","choices","k16_best","picked","possible","outside","mirror")}
    assert all_b["possible"].sum() == 606 and all_b["outside"].sum() == 491
    assert ((all_b["five"][:, 2]>.5)&(all_b["five"].min(1)<=.5)).sum() == 1845
    direction_rates = {k:100*measures(v).mean(axis=0) for k,v in raw.items()}
    total_rates = np.mean(list(direction_rates.values()), axis=0)
    for key, b in list(blocks.items()) + [("six_equal", all_b)]:
        rates = total_rates if key == "six_equal" else direction_rates[key]
        val, rest = b["picked"], b["outside"]
        count = len(val)
        for j, arm in enumerate(ARMS):
            counts = paired_counts(val[:,j], val[:,0])
            outcomes.append(dict(direction=key, n=count, arm=arm,
                **dict(zip(NAMES, rates[j])), delta25_pp=rates[j,0]-rates[0,0],
                delta50_pp=rates[j,1]-rates[0,1], **counts,
                moved=int((b["choices"][:,j]!=2).sum()),
                opportunities50=int(b["possible"].sum()), no_k16_opportunities50=int(rest.sum()),
                rescued_no_k16_50=int((rest&(val[:,j]>.5)).sum()),
                rescued_with_k16_50=int((~rest&(val[:,j]>.5)&(val[:,0]<=.5)).sum())))
        for arm, ref in COMPARISONS:
            j, k = ARMS.index(arm), ARMS.index(ref)
            paired.append(dict(direction=key, n=count, arm=arm, ref=ref,
                **{m+"_pp":rates[j,i]-rates[k,i] for i,m in enumerate(NAMES)},
                **paired_counts(val[:,j], val[:,k]),
                different_position=int((b["choices"][:,j]!=b["choices"][:,k]).sum())))
        for j, arm in enumerate(ARMS):
            for label, filt in (("outside_k16_opportunities", rest), ("all_606_opportunities", b["possible"]),
                                ("v1_success_50", val[:,0]>.5), ("masked_half", b["mirror"])):
                if not filt.any():
                    continue
                scope.append(dict(direction=key, arm=arm, stratum=label, n=int(filt.sum()),
                    successes50=int((val[filt,j]>.5).sum()),
                    moved=int((b["choices"][filt,j]!=2).sum()), **paired_counts(val[filt,j],val[filt,0])))
    csv_rows(RESULTS/"directions.csv", outcomes)
    csv_rows(RESULTS/"paired.csv", paired)
    csv_rows(RESULTS/"strata.csv", scope)

    samples = np.zeros((2000,len(ARMS),3))
    rng = np.random.default_rng(4171)
    for t in PLATFORMS:
        grouped = [b for key,b in blocks.items() if key.endswith("_to_"+t)]
        rec, idx = np.unique(grouped[0]["recording"], return_inverse=True)
        occurrences = np.bincount(idx)
        freq = rng.multinomial(len(rec), np.ones(len(rec))/len(rec), size=len(samples))
        den = (freq @ occurrences)[:,None,None]
        for b in grouped:
            np.testing.assert_array_equal(grouped[0]["recording"], b["recording"])
            m = measures(b["picked"])
            sums = np.stack([m[idx==j].sum(0) for j in range(len(rec))])
            samples += 100*np.einsum("br,rak->bak", freq, sums)/den/6
    for arm, ref in COMPARISONS:
        j, k = ARMS.index(arm), ARMS.index(ref)
        bounds = np.percentile(samples[:,j]-samples[:,k], [2.5,97.5], axis=0)
        for i, metric in enumerate(NAMES):
            intervals.append(dict(arm=arm, ref=ref, metric=metric, delta_pp=total_rates[j,i]-total_rates[k,i],
                                  lo95_pp=bounds[0,i], hi95_pp=bounds[1,i]))
    csv_rows(RESULTS/"recording_intervals.csv", intervals)


def anonymous_rows(blocks):
    lines = []
    for key,b in blocks.items():
        for i, row in enumerate(b["picked"]):
            line = dict(direction=key, outside_k16_opportunity50=int(b["outside"][i]),
                        five_best_iou=float(b["five"][i].max()), k16_best_iou=float(b["k16_best"][i]),
                        mirror_applied=int(b["mirror"][i]))
            for j, arm in enumerate(ARMS):
                line[arm+"_iou"] = float(row[j])
                line[arm+"_moved"] = int(b["choices"][i,j]!=2)
            lines.append(line)
    # Strip original order and all physical identifiers from the public evidence.
    import secrets
    secrets.SystemRandom().shuffle(lines)
    with gzip.open(RESULTS/"all_rows.csv.gz", "wt", newline="") as output:
        writer = csv.DictWriter(output, list(lines[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(lines)


if __name__ == "__main__":
    score_once()
    scored = read_blocks()
    summarize(scored)
    anonymous_rows(scored)

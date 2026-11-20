"""Label-free, single-frame, frozen prediction for all ordinary directions."""
from time import perf_counter

import numpy as np

from axial import (ACTIONS, ARMS, CONFIG, PLATFORMS, PRIVATE, R183, atomic, choose, digest,
                   evidence, features_at, incumbents, mirror_mask, ordinary, read, register,
                   reversed_profile, sensor_sample, translated)
from scope183 import check_input


def infer(profiles, heads, mask):
    x_mirror = profiles.copy()
    x_mirror[mask] = reversed_profile(profiles[mask])
    choices = [np.full(len(mask), 2, dtype=np.int8)]
    scores = []
    for arm in ARMS[1:]:
        base = arm.removesuffix("_mirror")
        x = x_mirror if base != arm else profiles
        value = evidence(x, heads[base], heads["mean"], heads["scale"], base.split("_")[1])
        assert np.isfinite(value).all() and (value>=0.).all() and (value<=1.+1e-12).all()
        if base != arm:
            original = scores[ARMS.index(base)-1]
            np.testing.assert_allclose(value[~mask], original[~mask], atol=1e-12, rtol=0)
            np.testing.assert_allclose(value[mask], original[mask, ::-1], atol=1e-12, rtol=0)
        choices.append(choose(value))
        scores.append(value)
    return np.stack(choices, axis=1).astype(np.int8), np.stack(scores, axis=1)


def run():
    lock = register()
    trained = PRIVATE/"heads"
    seal = read(trained/"SEAL.json")
    assert seal["lock"] == lock and not seal["used_target_labels"]
    heads = {}
    for platform in PLATFORMS:
        f = trained/f"{platform}.npz"
        assert digest(f) == seal["files"][f.name]
        with np.load(f, allow_pickle=False) as pack:
            heads[platform] = {k:pack[k] for k in pack.files}
    out = PRIVATE/"predictions"
    out.mkdir(exist_ok=True, mode=0o700)
    if (out/"SEAL.json").exists():
        raise RuntimeError("Target predictions already sealed")
    parent = read(R183/"predictions_PRIVATE/SEALED.json")
    assert not parent["target_labels_read"] and not parent["FINAL_access"]
    roster = ordinary()
    receipts = {}
    tick = perf_counter()
    for target in PLATFORMS:
        sources = [p for p in PLATFORMS if p!=target]
        boxes = {s:incumbents(f"{s}_to_{target}", parent) for s in sources}
        rows = roster[target]
        profiles = {s:np.empty((len(rows), 13)) for s in sources}
        for i, row in enumerate(rows):
            check_input(row)
            points = sensor_sample(row)
            for s in sources:
                profiles[s][i] = features_at(points, boxes[s][i])
        mask = mirror_mask(target, len(rows))
        assert mask.sum()*2 == len(rows)
        for s in sources:
            name = f"{s}_to_{target}"
            actions, scores = infer(profiles[s], heads[s], mask)
            output = np.stack([translated(b, ACTIONS[choice]) for b, choice in zip(boxes[s], actions)])
            np.testing.assert_array_equal(output[:, 0], boxes[s])
            np.testing.assert_array_equal(output[:, :, 2:], np.broadcast_to(boxes[s][:,None,2:], output[:,:,2:].shape))
            path = out/f"{name}.npz"
            np.savez_compressed(path, choices=actions, hit_probabilities=scores, boxes=output,
                                profiles=profiles[s], mirror=mask, ordinals=np.arange(len(rows)))
            receipts[name] = dict(n=len(rows), sha256=digest(path), masked=int(mask.sum()))
        print(target, len(rows)*2, "frozen predictions", round(perf_counter()-tick,1), flush=True)
    assert sum(k["n"] for k in receipts.values()) == 18620
    atomic(out/"SEAL.json", dict(input_roster=CONFIG["target"]["roster_seal"], lock=lock,
          heads=digest(trained/"SEAL.json"), files=receipts, arms=list(ARMS), direction_records=18620,
          target_updates=0, target_labels_read=False, FINAL_access=False, seconds=perf_counter()-tick))


if __name__ == "__main__":
    run()

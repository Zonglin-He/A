"""Prepare the four matched inputs and the possible translations without task labels."""
from collections import Counter

from side import (ARMS, BASE, HERE, OUT, PRIVATE, SPEC, TOL, digest, geometry, inputs,
                  legend, prompts, read, write)


def run():
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True, mode=0o700)
    parent = digest(BASE/"results/prediction_seal.json")
    assert parent == read(SPEC)["parent_prediction_seal_sha256"]
    panel, frozen = inputs()
    prepared = {}
    counts = Counter()
    for row in panel:
        uid = row["uid"]
        entry = frozen[uid]
        result = {"H":entry["boxes"]["H"], "status":entry["status"], "selected":entry["selected"]}
        if entry["selected"]:
            g = geometry(entry)
            frame = legend(row, entry, g)
            result.update(geometry=g, legend=frame)
            if frame is not None:
                result["prompts"] = prompts(row["text"], frame, g)
                counts["promptable"] += 1
                counts["C_D_identical_prompts"] += result["prompts"]["C"] == result["prompts"]["D"]
            counts["selected"] += 1
            counts["positive_slack"] += g["slack_m"] > TOL
        prepared[uid] = result
    write(PRIVATE/"prepared.json", prepared)
    write(OUT/"input_seal.json", dict(parent=parent, protocol=digest(SPEC), frames=len(panel),
          counts=dict(counts), prepared=digest(PRIVATE/"prepared.json"),
          geometry_code=digest(HERE/"side.py"), preparation_code=digest(__file__),
          target_labels_read=False, FINAL_access=False))
    print("Sealed", len(prepared), "frames", dict(counts), flush=True)


if __name__ == "__main__":
    run()

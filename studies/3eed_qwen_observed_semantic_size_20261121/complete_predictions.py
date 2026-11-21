"""Complete frozen observed boxes using the parsed prior and seal every prediction."""
from collections import Counter

import numpy as np

from normalization import resolve
from study import (ARMS, HERE, OUT, PRIVATE, SEMANTIC_ARMS, complete, digest, geometry_interface,
                   parse_semantic, parse_size, read, roster, semantic_prompt, size_requests,
                   text_hash, verified_observations, write)


def checked_replies():
    seal = read(OUT / "model_seal.json")
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    assert seal["pins"]["inputs"] == digest(OUT / "input_seal.json")
    assert seal["pins"]["inference"] == digest(HERE / "run_qwen.py")
    assert seal["replies"] == digest(PRIVATE / "reply_hashes.json")
    return seal, read(PRIVATE / "reply_hashes.json")


def checked_normalization():
    seal = read(OUT / "normalized_seal.json")
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    assert seal["pins"]["model_seal"] == digest(OUT / "model_seal.json")
    for key, file in (("amendment", "AMENDMENT.json"), ("normalization", "normalization.py"),
                      ("code", "normalize_and_read.py")):
        assert seal["pins"][key] == digest(HERE / file)
    manifest = PRIVATE / "resolved_hashes.json"
    assert seal["manifest"] == digest(manifest)
    return seal, read(manifest)


def decode_replies(frame, record, spans, pins):
    assert record["uid"] == frame["uid"] and record["pins"] == pins
    expression = frame["text"]
    labels = {}
    for arm in SEMANTIC_ARMS:
        response = record["semantic"][arm]
        assert response["prompt_sha256"] == text_hash(semantic_prompt(expression, spans if arm == "GS" else None))
        labels[arm] = parse_semantic(response["answer"])
    prompts, route = size_requests(expression, spans, labels)
    assert route == record["size_key"] and set(prompts) == set(record["sizes"])
    for key, prompt in prompts.items():
        assert record["sizes"][key]["prompt_sha256"] == text_hash(prompt)

    dimensions = {key: parse_size(record["sizes"][key]["answer"]) for key in prompts}
    return labels, dimensions, route


def decoded_inputs(frame, spans, initial_seal, initial_hash, normalized_seal, normalized_hash):
    uid = frame["uid"]
    archive = PRIVATE / "replies" / (uid + ".json")
    change = PRIVATE / "resolved" / (uid + ".json")
    assert digest(archive) == initial_hash and digest(change) == normalized_hash
    raw, record = read(archive), read(change)
    decode_replies(frame, raw, spans, initial_seal["pins"])
    assert record["uid"] == uid and record["pins"] == normalized_seal["pins"] and record["source"] == initial_hash
    labels, statuses, route, reused, new = resolve(frame, spans, raw)
    assert record["labels"] == labels and record["status"] == statuses and record["size_key"] == route
    assert set(record["generated"]) == set(new) and set(record["sizes"]) == set(reused) | set(new)
    for key, response in reused.items():
        assert record["sizes"][key] == response
    for key, prompt in new.items():
        assert record["sizes"][key]["prompt_sha256"] == text_hash(prompt)
    dimensions = {key: parse_size(r["answer"]) for key, r in record["sizes"].items()}
    return labels, dimensions, route, statuses


def run():
    observations = verified_observations()
    panel, baseline = roster()
    seal, hashes = checked_replies()
    normalization, normalized_hashes = checked_normalization()
    assert set(hashes) == set(normalized_hashes)
    parent = geometry_interface()
    predictions = {}
    fallbacks = Counter()
    repairs = Counter()
    shared = 0
    for frame in panel:
        uid = frame["uid"]
        old = baseline[uid]
        observation = observations[uid]
        h = old["boxes"]["H"]
        by_arm = {"H": {"box": h, "status": "original", "size_prior": old["boxes"]["A"][3:6] if h is not None else None}}
        if not observation["selected"]:
            assert uid not in hashes
            for arm in ARMS[1:]:
                by_arm[arm] = {"box": h, "status": "no_region", "semantic": None, "semantic_format": "unavailable",
                               "size_prior": None, "request": None}
                fallbacks[(arm, "no_region")] += 1
            predictions[uid] = by_arm
            continue

        labels, sizes, route, statuses = decoded_inputs(
            frame, observation["spans_lwh"], seal, hashes[uid], normalization, normalized_hashes[uid]
        )
        repairs.update(statuses.items())
        reused = bool(route["S"] and route["S"] == route["GS"])
        shared += int(reused)
        for arm in ARMS[1:]:
            key = "GD" if arm == "GD" else route[arm]
            dimensions = sizes.get(key)
            box = h
            if key is None:
                status = "invalid_semantic"
            elif dimensions is None:
                status = "invalid_size"
            else:
                new = complete(parent, frame, observation["observed_box_model"], dimensions)
                assert np.isfinite(new).all() and (new[3:6] > 0.).all() and new[6] == h[6]
                box = new.tolist()
                status = "completed"
            if status != "completed":
                fallbacks[(arm, status)] += 1
            by_arm[arm] = {"box": box, "status": status, "semantic": labels.get(arm),
                           "size_prior": dimensions, "request": key}
        if reused:
            assert by_arm["GS"] == by_arm["S"]
        for arm in SEMANTIC_ARMS:
            by_arm[arm]["semantic_format"] = statuses[arm]
        predictions[uid] = by_arm

    assert len(predictions) == 416 and len(hashes) == 320
    output = PRIVATE / "predictions.json"
    write(output, predictions)
    write(OUT / "prediction_seal.json", dict(
        frames=len(predictions), selected=len(hashes), shared_size_requests=shared,
        semantic_format_by_arm={arm: {status: n for (source, status), n in repairs.items() if source == arm}
                                for arm in SEMANTIC_ARMS},
        fallback_by_arm={arm: {s: fallbacks[(arm, s)] for s in ("no_region", "invalid_semantic", "invalid_size")}
                         for arm in ARMS[1:]},
        model_seal=digest(OUT / "model_seal.json"), normalized_seal=digest(OUT / "normalized_seal.json"),
        input_seal=digest(OUT / "input_seal.json"),
        code=digest(__file__), predictions=digest(output), target_labels_read=False, FINAL_access=False
    ))
    print("Sealed", len(predictions), "predictions and", shared, "reused S/GS size requests.", flush=True)


if __name__ == "__main__":
    run()

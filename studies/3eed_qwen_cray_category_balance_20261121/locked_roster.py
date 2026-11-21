"""Read the previous Ordinary Validation seals and match payloads by description."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL = HERE.parent / "3eed_qwen_full_ordinary_validation_20261121"
OUT = HERE / "results"
WORK = HERE / "private"
sys.path.insert(0, str(FULL))

import roster
import score_full


def locked_inputs():
    frames, manifest, seal, predictions = score_full.verify_seals()
    assert len(frames) == manifest["descriptions"] == seal["descriptions"] == 9310
    assert not seal["target_labels_read"] and not seal["FINAL_access"]
    reply_hashes = roster.PRIVATE / "fresh_reply_hashes.json"
    model_seal = roster.read(roster.OUT / "qwen_seal.json")
    assert roster.sha(reply_hashes) == model_seal["responses"]
    return frames, manifest, seal, predictions, roster.read(reply_hashes)


def prediction(frame, checksum, pins):
    uid = frame["uid"]
    path = roster.PRIVATE / "predictions" / (uid + ".json")
    assert roster.sha(path) == checksum
    data = roster.read(path)
    assert data["uid"] == uid and data["pins"] == pins
    return data


def replies(frame, data, reply_hashes):
    found = {}
    uid = frame["uid"]
    for task in roster.TASKS:
        if "archived" in frame:
            payload = frame["archived"][task]
            checksum = payload["source_sha256"]
        else:
            name = uid + "_" + task
            path = roster.PRIVATE / "replies" / (name + ".json")
            checksum = roster.sha(path)
            assert checksum == reply_hashes[name]
            payload = roster.read(path)
            assert payload["uid"] == uid and payload["task"] == task
            assert payload["pins"] == data["pins"]["model"]
        assert checksum == data["replies"][task]
        found[task] = payload
    return found

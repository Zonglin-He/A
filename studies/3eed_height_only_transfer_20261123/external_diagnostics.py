"""Count frozen reply and geometry outcomes without publishing model text."""

from collections import Counter
import csv
import gzip

from common import OUT, PRIVATE, read_payload, roster


def run():
    seal = OUT / "talk2car_qwen_seal.json"
    source = OUT / "talk2car_evaluation.json"
    replies = PRIVATE / "talk2car_replies.jsonl.gz"
    scores = OUT / "talk2car_rows.csv.gz"
    assert roster.sha(replies) == roster.read(seal)["payload"]
    evaluated = roster.read(source)
    assert roster.sha(scores) == evaluated["output_sha256"][scores.name]
    with gzip.open(scores, "rt") as stream:
        rows = {row["uid"]: row for row in csv.DictReader(stream)}
    parsed = {row["uid"]: row["replies"] for row in read_payload(replies)}
    assert rows.keys() == parsed.keys() and len(rows) == evaluated["independent"]["rows"]
    stats = {name: dict(Counter(r[name]["status"] for r in parsed.values())) for name in ("2d", "3d")}
    routes = dict(Counter(row["route"] for row in rows.values()))
    edited = Counter("better" if float(r["iou_roof"]) > float(r["iou_frozen"]) else
                     "worse" if float(r["iou_roof"]) < float(r["iou_frozen"]) else "same"
                     for r in rows.values() if int(r["changed"]))
    roster.write(OUT / "talk2car_frontend_diagnostics.json", dict(qwen=roster.sha(seal), labels=roster.sha(source),
        descriptions=len(rows), reply_statuses=stats, routes=routes, iou_on_edited=dict(edited), FINAL_access=False))


if __name__ == "__main__":
    run()

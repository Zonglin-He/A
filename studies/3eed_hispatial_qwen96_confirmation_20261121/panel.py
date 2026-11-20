"""Copy the exact 96-frame roster and create local pre-prediction review pages."""
from collections import Counter
from itertools import islice

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from shared import BASE, OUT, PRIVATE, PROTOCOL, QWEN, PLATFORMS, digest, key, read, write
from author_panel import inventory, pairs, ordinary
from scope183 import check_input
from geometry import calibration, observations


COLORS = ("#ff4143", "#48a5ff")


def capture():
    PRIVATE.mkdir(mode=0o700, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    observation_dir = PRIVATE / "observations"
    observation_dir.mkdir(mode=0o700, exist_ok=True)
    qwen_path = QWEN / "private/panel.json"
    qwen_seal = read(QWEN / "results/panel_seal.json")
    assert digest(qwen_path) == qwen_seal["panel"] and not qwen_seal["FINAL_access"]
    rows = read(qwen_path)
    ordinary_rows = ordinary()
    previous = BASE / "private/captured.json"
    assert digest(previous) == read(BASE / "results/capture_seal.json")["capture"]
    old_hashes = {frame["point_sha"] for frame in read(previous)}
    assert len(rows) == len({r["point_sha"] for r in rows}) == 96
    assert Counter(v["target"] for v in rows) == Counter({t: 32 for t in PLATFORMS})
    assert not ({r["point_sha"] for r in rows} & old_hashes)

    frames, labels = [], {}
    counts = {target: Counter() for target in PLATFORMS}
    for rank, existing in enumerate(rows):
        target = existing["target"]
        admitted = ordinary_rows[target][existing["ordinal"]]
        for old, new in (("rgb", "imagepath"), ("points", "pointpath"), ("calibration", "metapath"),
                         ("slot", "object_index"), ("suffix", "suffix")):
            assert existing[old] == admitted[new]
        check_input(admitted)
        frame = {k: existing[k] for k in ("target", "suffix", "points", "rgb", "calibration", "point_sha", "rgb_sha")}
        assert digest(frame["rgb"]) == frame["rgb_sha"] and digest(frame["points"]) == frame["point_sha"]
        uid = key("hispatial96", target, frame["point_sha"])
        frame.update(uid=uid, ordinal=rank, panel_index=rank % 32 + 1,
                     recording=key("hispatial96_recording", existing["recording"]),
                     calibration_sha=digest(frame["calibration"]))
        spec, size, transform = calibration(frame)
        frame["size"] = size.tolist()
        metadata = read(frame["calibration"])
        objects, conflict = inventory(metadata, spec, transform, size)
        noun_counts = Counter(o["noun"] for o in objects)
        selected = None if conflict else pairs(objects, uid)
        if selected is None:
            selected = sorted(objects, key=lambda o: (noun_counts[o["noun"]],
                -np.prod(o["rect"][2:] - o["rect"][:2]), key(uid, o["box"].tolist())))[:1]
        labels[uid] = {"objects": [{**obj, "box": obj["box"].tolist(), "rect": obj["rect"].tolist(),
                                   "noun_unique": noun_counts[obj["noun"]] == 1} for obj in selected]}
        obs = observations(frame, spec, size, transform)
        obs_file = observation_dir / (uid + ".npz")
        np.savez_compressed(obs_file, **obs)
        frame["observation_sha"] = digest(obs_file)
        counts[target].update({"frames": 1, f"candidate_count_{len(selected)}": 1,
                               "holdout": int(obs["hold"].sum()), "fit": int((~obs["hold"]).sum()),
                               "zero_distortion_assumed": int(spec.validated()[3])})
        frames.append(frame)

    write(PRIVATE / "captured.json", frames)
    write(PRIVATE / "candidate_truth.json", labels)
    write(OUT / "capture_seal.json", dict(
        protocol=digest(PROTOCOL), capture=digest(PRIVATE / "captured.json"),
        labels=digest(PRIVATE / "candidate_truth.json"), prior_qwen_panel=digest(qwen_path),
        previous_hispatial_overlap=0, ordinary_physical_frames=len(frames), exact_qwen_frame_order=True,
        FINAL_access=False, counts=counts,
        recordings={t: len({f["recording"] for f in frames if f["target"] == t}) for t in PLATFORMS}))
    print(dict(counts), flush=True)


def review_pages():
    frames = read(PRIVATE / "captured.json")
    labels = read(PRIVATE / "candidate_truth.json")
    folder = PRIVATE / "review"
    folder.mkdir(exist_ok=True, mode=0o700)
    font = ImageFont.truetype("DejaVuSans.ttf", 14)
    width, height = 490, 333
    for platform in PLATFORMS:
        subset = iter(f for f in frames if f["target"] == platform)
        for page in range(4):
            canvas = Image.new("RGB", (width * 2, height * 4), "#161a1e")
            draw = ImageDraw.Draw(canvas)
            for idx, frame in enumerate(islice(subset, 8)):
                x, y = (idx % 2) * width, (idx // 2) * height
                with Image.open(frame["rgb"]) as source:
                    original = source.convert("RGB")
                shown = original.copy()
                shown.thumbnail((width - 8, 234))
                scaled = np.asarray(shown.size) / original.size
                inset = ImageDraw.Draw(shown)
                objects = labels[frame["uid"]]["objects"]
                for j, obj in enumerate(objects):
                    rectangle = (np.asarray(obj["rect"]) * np.tile(scaled, 2)).tolist()
                    inset.rectangle(rectangle, outline=COLORS[j], width=2)
                    inset.text((rectangle[0], max(0, rectangle[1] - 16)), "AB"[j], font=font, fill=COLORS[j],
                               stroke_width=1, stroke_fill="black")
                    crop = original.crop(tuple(map(round, obj["rect"])))
                    crop.thumbnail((150, 55))
                    canvas.paste(crop, (x + 4 + j * 156, y + 259))
                    draw.text((x + 4 + j * 156, y + 315), "AB"[j] + " " + obj["noun"], font=font, fill=COLORS[j])
                canvas.paste(shown, (x + 4, y + 22))
                draw.text((x + 5, y + 4), f"{platform.upper()} {frame['panel_index']:02d}", font=font, fill="white")
            canvas.save(folder / f"{platform}_{page+1}.jpg", quality=91)


if __name__ == "__main__":
    capture()
    review_pages()

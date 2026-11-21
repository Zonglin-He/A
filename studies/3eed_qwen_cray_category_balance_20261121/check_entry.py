"""Exercise the registered inference entry on saved inputs without reading labels."""
from cray import predict
from locked_roster import HERE, OUT, WORK, locked_inputs, prediction, replies, roster


def run():
    frames, _, history, checksums, response_index = locked_inputs()
    locked = roster.read(OUT / "prediction_seal.json")
    path = WORK / "predictions.json"
    assert locked["selector"] == roster.sha(HERE / "cray.py")
    assert locked["payload"] == roster.sha(path)
    chosen = {}
    for frame, saved in zip(frames, roster.read(path), strict=True):
        assert frame["uid"] == saved["uid"]
        box = saved["boxes"]["C_ray_applicable"]
        group = "ray" if saved["use_ray"] else "missing" if box is None else "fallback"
        chosen.setdefault((frame["target"], group), (frame, saved))
    assert len(chosen) == 9
    for (platform, group), (frame, saved) in sorted(chosen.items()):
        old = prediction(frame, checksums[frame["uid"]], history["pins"])
        expected = saved["boxes"]["C_ray_applicable"]
        value = predict(frame, replies(frame, old, response_index))
        assert value["box"] == expected
        assert value["route"] == ("C_ray" if group == "ray" else "H")
        print(platform, group, "exact", flush=True)
    roster.write(OUT/"inference_entry_check.json", dict(
        code=roster.sha(HERE/"cray.py"), matched_outputs=len(chosen), replay="first row per platform and route",
        routes=["ray", "fallback", "missing"], used_target_annotations=False, qwen_inferences=0, FINAL_access=False))


if __name__ == "__main__":
    run()

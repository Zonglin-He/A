"""Record native v1 scores and support moments without the target labels."""
import argparse
from functools import lru_cache
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R183 = HERE.parent / "3eed_v2_ordinary_val_subject_ranking_20261119"
sys.path[:0] = [str(ROOT), str(R183),
                str(HERE.parent / "3eed_v1_retrieval_binding_refinement_20261119")]
from scope183 import PLATFORMS, atomic, check_input, digest, frame, ordinary, read
from mechanisms import CandidateGeometry, distinct_entity_relation
from crossplatform.ground import Grounder

PRIVATE = HERE / "run_PRIVATE"
BLOCK = 128


def inputs():
    cfg = read(HERE / "PROTOCOL.json")
    for path, checksum in {**cfg["authority"], **cfg["decoder_source_sha256"]}.items():
        assert digest(ROOT / path) == checksum
    return dict(protocol=digest(HERE / "PROTOCOL.json"), capture=digest(Path(__file__)),
                prediction=digest(R183 / "predictions_PRIVATE/SEALED.json"))


def sensor_boxes(value, platform, view):
    boxes = np.asarray(value, dtype=np.float64).copy()
    if platform == "waymo":
        from camera_projection import WAYMO_ANGLES, z_rotation
        angle = WAYMO_ANGLES[view]
        boxes[:, :3] = boxes[:, :3] @ z_rotation(angle)
        boxes[:, 6] = (boxes[:, 6] - np.deg2rad(angle) + np.pi / 2) % np.pi - np.pi / 2
    elif platform == "drone":
        boxes[:, 2] -= 1.8
    return boxes


class NativeTrace:
    def __init__(self, decoder):
        self.round_statistics = []
        self.slot_outputs = []
        self.pair_outputs = []
        self.handles = [
            decoder.statistics.register_forward_pre_hook(
                lambda _, args: self.round_statistics.append(args[0].detach())),
            decoder.slot_score.register_forward_hook(
                lambda _, args, out: self.slot_outputs.append(out.detach())),
            decoder.pair.register_forward_hook(
                lambda _, args, out: self.pair_outputs.append(out.detach())),
        ]

    def collect(self, decoder, ev):
        for values in (self.round_statistics, self.slot_outputs, self.pair_outputs):
            values.clear()
        out = decoder(ev)
        assert len(self.round_statistics) == len(self.slot_outputs) == len(self.pair_outputs) == 2
        support = torch.stack([v.reshape(8, 11) for v in self.round_statistics])
        a = ev["base_logit"][0]
        raw_ref = ev["ref_logit"][0]
        ref_prior = .22 * (raw_ref - raw_ref.max()).clamp_min(-15)
        edit = .75 * self.slot_outputs[-1].reshape(8).float().tanh()
        log_pi = (ref_prior + edit[4:]).log_softmax(-1)
        active = bool(ev["has_reference"][0])
        q = .5 * self.pair_outputs[-1].reshape(4, 4).float().tanh() * active
        relation = (q + log_pi[None]).logsumexp(-1) if active else torch.zeros_like(a)
        score = out["subject_logits"][0]
        torch.testing.assert_close(a + edit[:4] + relation, score, atol=2e-6, rtol=0.)
        assert a[0].item() == a.max().item() == 0.
        return dict(prior=a.cpu().numpy(), final=score.cpu().numpy(),
                    unary=edit[:4].cpu().numpy(), rel=relation.cpu().numpy(),
                    support=support.cpu().numpy(), active=active,
                    choices=out["choices"][0].cpu().numpy().astype(np.int8))

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def observation(model, trace, row, expected_four, expected_winner):
    stored = {}
    scene, ev = model._read_observation(frame(row), capture=stored)
    v = trace.collect(model._decoder, ev)
    assert v["choices"][0] == expected_winner
    four = ev["subject_boxes"][0].cpu().numpy()
    refs = ev["reference_boxes"][0].cpu().numpy()
    v["same_selected_box"] = bool(v["active"] and np.array_equal(
        four[v["choices"][0]], refs[v["choices"][1]]))
    v["explicit_lr"] = distinct_entity_relation(stored["graph"], v["active"])
    # Geometry uses the same points and the same PCA function as the original cohort.
    geo = CandidateGeometry(scene.boxes, scene.sampled_points)
    native = np.stack([geo.orient(i) for i in range(len(scene.boxes))])
    matches = np.all(four[:, None] == native[None], axis=-1)
    assert matches.any(-1).all()
    # Ties denote identical seven-dimensional boxes, equivalent for oriented IoU.
    slots = matches.argmax(-1)
    v["pool"] = sensor_boxes(native, scene.target, scene.view_index)
    np.testing.assert_allclose(v["pool"][slots], expected_four, atol=1e-10, rtol=0.)
    v["slots"] = slots.astype(np.uint8)
    return v


def capture(source):
    setup = inputs()
    old = read(R183 / "predictions_PRIVATE/SEALED.json")
    assert old["n"] == 18620 and not old["target_labels_read"] and not old["FINAL_access"]
    roster = ordinary()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    model = Grounder(source)
    model._frontend.parser = lru_cache(maxsize=256)(model._frontend.parser)
    trace = NativeTrace(model._decoder)
    start = perf_counter()
    try:
        for target in PLATFORMS:
            if source == target:
                continue
            key = f"{source}_to_{target}"
            folder = PRIVATE / key
            folder.mkdir(exist_ok=True, mode=0o700)
            receipts = {}
            for name, proof in sorted(old["directions"][key]["files"].items()):
                before = R183 / "predictions_PRIVATE" / key / name
                assert digest(before) == proof["sha256"]
                out = folder / name
                pin = out.with_suffix(".json")
                if pin.exists():
                    record = read(pin)
                    assert record["inputs"] == setup and record["parent"] == proof["sha256"]
                    assert digest(out) == record["sha256"]
                else:
                    with np.load(before, allow_pickle=False) as previous:
                        group = []
                        four, chosen = previous["four"], previous["choices"][:, 0]
                        ordinals = previous["ordinals"]
                        for idx, quartet, winner in zip(ordinals, four, chosen):
                            row = roster[target][int(idx)]
                            check_input(row)
                            group.append(observation(model, trace, row, quartet, winner))
                    arrays = {k: np.stack([x[k] for x in group]) for k in group[0]}
                    tmp = out.with_suffix(".tmp.npz")
                    np.savez_compressed(tmp, ordinals=ordinals, **arrays)
                    os.replace(tmp, out)
                    record = dict(inputs=setup, parent=proof["sha256"], sha256=digest(out),
                                  n=len(group), original_four_and_winner_checked=len(group))
                    atomic(pin, record)
                receipts[name] = record["sha256"]
                print(key, proof["last"], "/", len(roster[target]), round(perf_counter()-start, 1), flush=True)
            atomic(folder / "SEALED.json", dict(inputs=setup, n=len(roster[target]), files=receipts,
                   labels_used=False, target_updates=0, FINAL_access=False))
    finally:
        trace.close()


def seal():
    setup = inputs()
    files = {}
    n = 0
    for source in PLATFORMS:
        for target in PLATFORMS:
            if target == source:
                continue
            name = f"{source}_to_{target}"
            p = PRIVATE / name / "SEALED.json"
            info = read(p)
            assert info["inputs"] == setup and not info["labels_used"] and not info["FINAL_access"]
            for path, signature in info["files"].items():
                assert digest(p.parent / path) == signature
            files[name] = digest(p)
            n += info["n"]
    assert n == 18620
    atomic(PRIVATE / "SEALED.json", dict(inputs=setup, n=n, directions=files,
           original_four_and_winners_checked=n, target_labels_used=False, target_updates=0, FINAL_access=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=[*PLATFORMS, "seal"])
    phase = parser.parse_args().phase
    seal() if phase == "seal" else capture(phase)

"""Replay v1 components on sealed, non-protected inputs. Writes sensor-frame predictions."""
import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import os
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import torch

from crossplatform.candidate_frontend import PLATFORMS, WAYMO_VIEWS, orient_box, sensor_frame_box
from crossplatform.ground import Grounder
from crossplatform.v1_decoder import JointDecoder
from crossplatform.v1_evidence import role_masks, subject_anchors


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    draft = path.with_suffix('.tmp')
    draft.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(draft, path)


def input_rows(protocol):
    path = ROOT / protocol['input_registry']
    sealed = read(ROOT / protocol['input_seal'])
    assert digest(path) == sealed['input_sha256'] == protocol['input_sha256']
    assert sealed['FINAL_access'] is False and sealed['target_labels_used_for_selection'] is False
    rows = read(path)
    assert len(rows) == len({r['id'] for r in rows}) == 3 * protocol['per_target_expressions']
    for target in PLATFORMS:
        assert sum(r['platform'] == target for r in rows) == protocol['per_target_expressions']
    return rows


class AblationDecoder(JointDecoder):
    """One-factor interventions on an exact copy of the frozen decoder."""

    def replay(self, evidence, arm):
        self.arm = arm
        self.read_index = 0
        self.pair_index = 0
        if arm == 'no_reference_interaction':
            evidence = dict(evidence, has_reference=torch.zeros_like(evidence['has_reference']))
        elif arm == 'no_predicate_vector':
            mask = evidence['role_mask'].clone()
            mask[:, 2] = False
            evidence = dict(evidence, role_mask=mask)
        elif arm == 'no_anchor_pca':
            evidence = dict(evidence)
            for key in ('subject_boxes', 'reference_boxes'):
                evidence[key] = evidence[key].clone()
                evidence[key][..., 6] = 0
        unit = {'no_ref_to_subj': self.from_ref, 'no_subj_to_ref': self.from_subject}.get(arm)
        if unit is None:
            return self(evidence)
        hook = unit.register_forward_hook(lambda _module, _args, out: torch.zeros_like(out))
        try:
            return self(evidence)
        finally:
            hook.remove()

    def _read(self, state, language, key, value, spatial, points, uv, visible, anchors, prior):
        self.read_index += 1
        if self.arm == 'no_subject_geometry' and self.read_index == 2:
            spatial = .65 * self._spatial_bias(points, anchors)
        return super()._read(state, language, key, value, spatial, points, uv, visible, anchors, prior)

    def _pair_scores(self, state, boxes, angle, pixels, mass, active):
        self.pair_index += 1
        if self.arm == 'no_final_pair' and self.pair_index == 2:
            return state.new_zeros((len(state), self.slots, self.slots))
        if self.arm == 'no_pair_projection':
            mass = torch.zeros_like(mass)
        return super()._pair_scores(state, boxes, angle, pixels, mass, active)


def tapped_read(model, frame):
    """Use the deployed frontend once, retaining only the tensors needed for matched ablations."""
    import core89

    matcher = model._frontend.matcher
    call = matcher.main
    route = core89.route89
    staged = []
    route_args = []

    def read_match(*args, **kwargs):
        out = call(*args, **kwargs)
        staged.append((args, out[0]))
        return out

    def read_route(*args):
        route_args.append(args)
        return route(*args)

    with patch.object(matcher, 'main', read_match), patch.object(core89, 'route89', read_route):
        observed, evidence = model._read_observation(frame)
    assert len(staged) == 2 and len(route_args) == 1
    return observed, evidence, staged[0], route_args[0]


def lidar_probability(model, initial_call, route_args):
    from core89 import route89, tensor_route89

    matcher = model._frontend.matcher
    base, prefix, pixels, mode = initial_call
    assert mode == 'bidirectional' and matcher.route is None and not matcher.use_interface
    stripped = dict(pixels, image=torch.full_like(pixels['image'], .5))
    for name in ('p2i', 'i2p'):
        stripped[name] = torch.full_like(pixels[name], -1)
        stripped[name + '_weight'] = torch.zeros_like(pixels[name + '_weight'])
    layer = base.cross_encoder.layers[-1].cross_layer.cross_lv
    attention = []
    hook = layer.register_forward_hook(lambda _m, _a, out: attention.append(out[1][0].cpu().numpy()))
    try:
        p0, _, details = matcher.main(base, prefix, stripped, 'independent')
    finally:
        hook.remove()
    assert len(attention) == 1 and details['late_RGB_delta'].count_nonzero().item() == 0
    row, raw, graph, roles, spans, _attention, offsets, doc = route_args
    binding, _ = route89(row, dict(raw, frozen_probability=p0.cpu().numpy()), graph,
                         roles, spans, attention[0], offsets, doc)
    matcher.route = tensor_route89(binding)
    matcher.use_interface = bool(len(matcher.route['token_roles']))
    try:
        score, _, output = matcher.main(base, prefix, stripped, 'independent')
        assert output['late_RGB_delta'].count_nonzero().item() == 0
        return score.cpu().numpy()
    finally:
        matcher.route = None
        matcher.use_interface = False
        matcher.local = {}


def alternative_subjects(frame, evidence, arm, probability=None):
    from grounding158 import select_grounding_box
    from reader158 import overlaps

    if probability is not None:
        obs = replace(frame, probability=probability)
        boxes, priors = subject_anchors(obs.boxes, obs.probability, obs.sampled_points)
    else:
        if arm == 'no_slot_diversity':
            _, scores = select_grounding_box(frame.boxes, frame.probability)
            indices = np.argsort(-scores, kind='stable')[:4]
        else:
            assert arm == 'no_overlap_prior'
            scores = frame.probability
            ordered = np.argsort(-scores, kind='stable')
            overlap = overlaps(frame.boxes)
            keep = [int(ordered[0])]
            for index in ordered[1:]:
                if (overlap[index, keep] <= .35).all():
                    keep.append(int(index))
                    if len(keep) == 4:
                        break
            keep.extend(int(index) for index in ordered if index not in keep)
            indices = keep[:4]
        log = np.log(np.maximum(scores[indices], 1e-80))
        priors = (log - log[0]).clip(-20., 0.).astype(np.float32)
        boxes = np.stack([orient_box(frame.sampled_points, frame.boxes[k]) for k in indices])
    current = evidence['subject_boxes']
    return dict(evidence, subject_boxes=torch.as_tensor(boxes[None], device=current.device),
                base_logit=torch.as_tensor(priors[None], device=current.device))


@torch.inference_mode()
def predict_row(model, head, row, arms, seed):
    source = model._frontend.source
    target = row['platform']
    view = int(row['suffix'] or 0) if target == 'waymo' else 0
    frame = dict(target=target, points=row['pointpath'], rgb=row['imagepath'],
                 calibration=row['metapath'], text=row['caption'], seed=seed, waymo_view=WAYMO_VIEWS[view])
    observed, evidence, (main_args, initial), role_args = tapped_read(model, frame)
    original = model._decoder(evidence)
    control = head.replay(evidence, 'v1')
    for name in ('choices', 'boxes', 'subject_logits'):
        torch.testing.assert_close(control[name], original[name], atol=0, rtol=0)
    rgb_off_probability = lidar_probability(model, main_args, role_args)
    old_box = np.r_[observed.boxes[np.argmax(role_args[1]['scores'])], np.float32(0)]
    old_box[3:6] = np.maximum(old_box[3:6], 1e-6)
    retained = {'frozen_text': old_box, 'v1': original['boxes'][0, 0].cpu().numpy(),
                'direct': evidence['subject_boxes'][0, 0].cpu().numpy()}
    retained['no_output_pca'] = retained['v1'].copy()
    retained['no_output_pca'][6] = 0
    same_box = {name: False for name in retained}
    same_box['v1'] = bool(torch.equal(original['boxes'][0, 0], original['boxes'][0, 1]))
    for arm in arms:
        if arm in retained:
            continue
        if arm in ('no_rgb', 'no_competition', 'no_overlap_prior', 'no_slot_diversity'):
            probability = {'no_rgb': rgb_off_probability, 'no_competition': initial.cpu().numpy()}.get(arm)
            changed = alternative_subjects(observed, evidence, arm, probability)
        else:
            changed = evidence
        result = head.replay(changed, arm)
        assert torch.isfinite(result['subject_logits']).all()
        retained[arm] = result['boxes'][0, 0].cpu().numpy()
        same_box[arm] = bool(torch.equal(result['boxes'][0, 0], result['boxes'][0, 1]))
    parsed_reference = bool(evidence['has_reference'][0])
    # The same parser supplied the deployed role mask. A direct relation additionally
    # requires one edge connecting the target to the single parsed reference.
    graph = role_args[3]
    direct_edges = [e for e in graph['relations'] if 'T' in (e['subject'], e['reference'])]
    direct_left_right = parsed_reference and len(direct_edges) == 1 and direct_edges[0]['predicate'] in ('LEFT', 'RIGHT')
    assert source != target
    boxes = np.stack([sensor_frame_box(retained[name], target, view) for name in arms])
    assert np.isfinite(boxes).all() and (boxes[:, 3:6] > 0).all()
    return boxes, [parsed_reference, direct_left_right], [same_box[name] for name in arms]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--engineering', action='store_true', help='one ordinary record per direction, no labels')
    args = cli.parse_args()
    protocol = read(HERE / 'protocol.json')
    names = tuple(protocol['arms'])
    rows = input_rows(protocol)
    private = HERE / ('engineering_PRIVATE' if args.engineering else 'predictions_PRIVATE')
    private.mkdir(mode=0o700)
    lock = dict(protocol_sha256=digest(HERE / 'protocol.json'), registry_sha256=protocol['input_sha256'],
                method_sha256=digest(ROOT / 'crossplatform/CURRENT_METHOD.json'),
                implementation={str(path.relative_to(ROOT)): digest(path) for path in
                                [Path(__file__), *(ROOT / 'crossplatform' / f for f in
                                  ('ground.py', 'v1_decoder.py', 'candidate_frontend.py', 'v1_evidence.py',
                                   'single_frame_sources.json'))]}, arms=list(names),
                FINAL_access=False, target_labels_read=False, predictions_only=True)
    atomic_json(private / 'LOCK.json', lock)
    files = {}
    run_start = perf_counter()
    for source in PLATFORMS:
        model = Grounder(source)
        head = AblationDecoder().to('cuda').eval().requires_grad_(False)
        head.load_state_dict(model._decoder.state_dict(), strict=True)
        for target in PLATFORMS:
            if source == target:
                continue
            content = [(i, row) for i, row in enumerate(rows) if row['platform'] == target]
            if args.engineering:
                content = content[:1]
            boxes, roles, same = [], [], []
            started = perf_counter()
            for k, (_ordinal, row) in enumerate(content, 1):
                b, rel, duplicate = predict_row(model, head, row, names, protocol['seed'])
                boxes.append(b)
                roles.append(rel)
                same.append(duplicate)
                if k % 64 == 0 or k == len(content):
                    print(source, target, k, '/', len(content), 'seconds', round(perf_counter() - started, 1), flush=True)
            name = source + '_to_' + target + '.npz'
            draft = private / (name + '.tmp')
            with draft.open('wb') as f:
                np.savez_compressed(f, arms=names, ordinals=[i for i, _ in content], boxes=boxes, roles=roles, same=same)
            os.replace(draft, private / name)
            files[name] = dict(n=len(content), sha256=digest(private / name))
        del model, head
        torch.cuda.empty_cache()
    n = sum(record['n'] for record in files.values())
    assert n == (6 if args.engineering else protocol['direction_records'])
    atomic_json(private / 'SEALED.json',
                dict(n=n, files=files, lock_sha256=digest(private / 'LOCK.json'),
                     elapsed_seconds=perf_counter() - run_start, FINAL_access=False, target_labels_read=False))


if __name__ == '__main__':
    main()

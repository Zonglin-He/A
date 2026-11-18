"""Export frozen R76/R91 evidence with the registered single-frame deployment."""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import hashlib
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / 'crossplatform/round7'))
from crossplatform.ground_single_frame import SingleFrameGrounder, WAYMO_VIEWS, orient_box


@dataclass
class Evidence:
    boxes: np.ndarray
    probability: np.ndarray
    utility: np.ndarray
    old_winner: int
    old_pca: np.ndarray
    points: np.ndarray
    indices: np.ndarray
    seeds: np.ndarray
    seed_sample_indices: np.ndarray
    anchors: np.ndarray
    role_mask: np.ndarray
    role_spans: np.ndarray
    edges: np.ndarray
    affinity: np.ndarray
    supports: np.ndarray
    hypotheses: np.ndarray
    route_weights: np.ndarray
    subject_weights: np.ndarray
    p2i: np.ndarray
    p2i_weight: np.ndarray
    p2i_learned: np.ndarray
    i2p: np.ndarray
    i2p_weight: np.ndarray
    i2p_learned: np.ndarray
    ionly: np.ndarray
    visible: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    camera_intrinsic: np.ndarray
    model_to_camera: np.ndarray
    distortion: np.ndarray
    width: int
    height: int
    forward_seconds: float


def attention_weights(module, arguments, result, verify=False):
    """Reconstruct the two local attention reads from their original arguments."""
    import torch
    query, key, index, weight = arguments
    valid = (index >= 0) & (weight > 0)
    selected = index.clamp_min(0)
    logits = (module.q(query)[:, None] * module.k(key)[selected]).sum(-1) / 8
    logits = logits + weight.clamp_min(1e-12).log()
    weights = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
    if verify:
        read = (weights[:, :, None] * module.v(key)[selected]).sum(1)
        torch.testing.assert_close(module.o(read) * valid.any(1)[:, None], result, rtol=0, atol=0)
    return weights


class EvidenceGrounder(SingleFrameGrounder):
    """Use the original capture, role routing and frozen matching computations."""

    def evidence_frame(self, row, *, rgb_mode='correct', other_image=None, verify=False):
        import torch
        from PIL import Image
        from camera_projection import CameraSpec, model_to_camera_matrix, read_camera_allowlist
        from core89 import route89, tensor_route89, token_roles88
        from grounding158 import select_grounding_box
        from interface66 import capture
        from model51 import SENTINEL
        from model76 import tensor_data, tensor_prefix
        from parse97 import dependency_document, parse85, parse_subject, repair_inventory87
        from prepare76 import geometry
        from runner_helpers import sample_points
        from utils.transform_waymo import transform_to_front_view

        if rgb_mode not in {'correct', 'off', 'other_frame'} or (rgb_mode == 'other_frame') != (other_image is not None):
            raise ValueError('Expected one registered RGB intervention')
        started = time.perf_counter()
        row = dict(row)
        caption = row['caption']
        if not caption.strip() or len(self.base.tokenizer(caption + SENTINEL)['input_ids']) > 256:
            raise ValueError('Invalid native expression length')
        row['id'] = hashlib.sha256(caption.encode()).hexdigest()
        view = int(row['suffix'] or 0)
        with self._lock, torch.inference_mode():
            points, indices, _ = sample_points(row)
            if row['platform'] == 'waymo':
                points, _ = transform_to_front_view(points, np.empty((0, 7)), WAYMO_VIEWS[view])
                points = np.ascontiguousarray(points, dtype=np.float32)

            point_order = []
            def exact_seed_order(_module, _args, output):
                first, second = output['sa_ins'][1:3]
                point_order.append(first.long().gather(1, second.long())[0].cpu().numpy())

            hook = self.base.backbone_net.register_forward_hook(exact_seed_order)
            try:
                raw, _ = capture(self.base, points, indices, caption)
            finally:
                hook.remove()
            seeds = raw['xyz'][0]
            seed_index, = point_order
            if not np.array_equal(seeds, points[seed_index]):
                raise ValueError('Composed PointNet++ indices differ from observed seed coordinates')
            spatial, _ = geometry(row, raw)
            prefix = tensor_prefix(raw)
            pixels = tensor_data(spatial, row)
            if rgb_mode == 'off':
                # "independent" skips both early cross-modal residuals. The empty
                # point-to-image graph also forces the late anchor RGB delta to zero.
                pixels['image'] = torch.full_like(pixels['image'], .5)
                for name in ('p2i', 'i2p'):
                    pixels[name] = torch.full_like(pixels[name], -1)
                    pixels[name + '_weight'] = torch.zeros_like(pixels[name + '_weight'])
            elif other_image is not None:
                with Image.open(other_image) as im:
                    altered = np.array(im.convert('RGB').resize((384, 256), Image.Resampling.BILINEAR))
                pixels['image'] = torch.as_tensor(altered, device='cuda').float().permute(2, 0, 1)[None] / 255
            mode = 'independent' if rgb_mode == 'off' else 'bidirectional'
            doc = dependency_document(self.parser(caption))
            subject = parse_subject(caption, doc)
            roles = repair_inventory87(caption, doc, parse85(caption, doc, subject))
            token_roles, offsets = token_roles88(self.base, row, raw, roles, doc)
            captured = {}

            def record_lang(_module, _args, output):
                captured['attention'] = output[1][0].cpu().numpy()

            def record_local(name):
                def record(module, arguments, output):
                    captured[name] = attention_weights(module, arguments, output, verify).cpu().numpy()
                return record

            def record_image(_module, _args, output):
                captured['ionly'] = output.cpu().numpy()

            matcher = self.matcher
            layer = self.base.cross_encoder.layers[-1].cross_layer.cross_lv
            with ExitStack() as hooks:
                hooks.callback(layer.register_forward_hook(record_lang).remove)
                initial, _, _ = matcher.main(self.base, prefix, pixels, mode)
                route_attention = captured['attention']
                route, _ = route89(row, dict(raw, frozen_probability=initial.cpu().numpy()),
                                   spatial, roles, token_roles, route_attention, offsets, doc)
                hooks.callback(matcher.image_to_point.register_forward_hook(record_local('p2i')).remove)
                hooks.callback(matcher.point_to_image.register_forward_hook(record_local('i2p')).remove)
                hooks.callback(matcher.image_adapter.register_forward_hook(record_image).remove)
                matcher.route = tensor_route89(route)
                matcher.use_interface = bool(len(token_roles))
                try:
                    probability, _, detail = matcher.main(self.base, prefix, pixels, mode)
                finally:
                    matcher.route = None
                    matcher.use_interface = False
                    matcher.local = {}
            if verify:
                np.testing.assert_array_equal(route_attention, captured['attention'])
            if rgb_mode == 'off' and (captured.keys() & {'p2i', 'i2p'} or detail['late_RGB_delta'].count_nonzero()):
                raise ValueError('An RGB residual survived the full no-RGB intervention')
            prob = probability.cpu().numpy()
            winner, utility = select_grounding_box(raw['boxes'], prob)
            att = captured['attention']
            affinity = (token_roles @ att) / np.maximum(1, token_roles.sum(1))[:, None]
            names = [e['role'] for e in roles['entities']]
            relations = []
            for edge in roles['relations']:
                if edge['known_frame'] and edge['predicate'] in {'LEFT', 'RIGHT'}:
                    a, b = names.index(edge['subject']), names.index(edge['reference'])
                    relations.append((a, b, 1 if edge['predicate'] == 'LEFT' else -1))
            with Image.open(row['imagepath']) as im:
                width, height = im.size
            cam = CameraSpec.from_camera_fields(row['platform'], read_camera_allowlist(row['metapath']),
                                                waymo_view_suffix=view, missing_distortion_policy='assume_zero_declared')
            intrinsic, _, distortion, _ = cam.validated()
            maps = {n: pixels[n].cpu().numpy() for n in ('p2i', 'p2i_weight', 'i2p', 'i2p_weight')}
            return Evidence(
                boxes=raw['boxes'], probability=prob, utility=utility, old_winner=winner,
                old_pca=orient_box(points, raw['boxes'][winner]), points=points, indices=indices, seeds=seeds,
                seed_sample_indices=seed_index, anchors=raw['query_points_sample_inds'],
                role_mask=token_roles, role_spans=np.array([e['span'] for e in roles['entities']]).reshape(-1, 2),
                edges=np.asarray(relations, dtype=np.int16).reshape(-1, 3),
                affinity=affinity.astype(np.float32), supports=route['supports'],
                hypotheses=route['hypotheses_PRIVATE'], route_weights=route['weights_plain'],
                subject_weights=detail['subject_weights'].cpu().numpy(),
                p2i_learned=captured.get('p2i', np.zeros_like(maps['p2i_weight'])),
                i2p_learned=captured.get('i2p', np.zeros_like(maps['i2p_weight'])),
                ionly=captured['ionly'], visible=spatial['visible'], uv=spatial['uv_grid'] / [48, 32],
                depth=spatial['point_depth'], camera_intrinsic=intrinsic, model_to_camera=model_to_camera_matrix(cam),
                distortion=distortion, width=width, height=height,
                forward_seconds=time.perf_counter() - started, **maps)

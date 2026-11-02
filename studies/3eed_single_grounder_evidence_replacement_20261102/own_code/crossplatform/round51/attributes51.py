"""R51 measured sparse RGB/point attributes, without external vision models.

Color supports characterize only uniquely associated visible point pixels.
They are not learned category, complex appearance or identity predictions.
Text graph fields are retained as inert data, never interpreted as commands.
"""
import copy
import hashlib
import json
import re
import numpy as np


MINIMUM_COLOR_PIXELS = 8
MINIMUM_GEOMETRY_POINTS = 8
COLOR_BETA_PSEUDOCOUNT = .5
COLOR_LOGODDS_CLIP = 4.
COLOR_RGB_PROTOTYPES = {
    'red': (190, 40, 40), 'orange': (225, 120, 35), 'yellow': (225, 210, 40),
    'green': (50, 140, 65), 'cyan': (50, 180, 190), 'blue': (45, 80, 180),
    'purple': (130, 65, 160), 'pink': (225, 145, 175), 'brown': (120, 75, 40),
    'beige': (200, 180, 140), 'white': (235, 235, 235), 'gray': (135, 135, 135),
    'black': (30, 30, 30),
}
COLOR_ALIASES = {'grey': 'gray', 'silver': 'gray', 'gold': 'yellow', 'golden': 'yellow',
                 'violet': 'purple', 'turquoise': 'cyan', 'tan': 'beige',
                 'maroon': 'red', 'navy': 'blue'}


def rgb_to_hsv(rgb):
    x = np.asarray(rgb, np.float64) / 255.
    hi, lo = x.max(-1), x.min(-1)
    d = hi - lo
    hue = np.zeros_like(hi)
    chromatic = d > 0
    winner = x.argmax(-1)
    for k in range(3):
        keep = chromatic & (winner == k)
        a, b = ((1, 2), (2, 0), (0, 1))[k]
        hue[keep] = (x[keep, a] - x[keep, b]) / d[keep] + 2 * k
    hue = (hue / 6) % 1.
    sat = np.divide(d, hi, out=np.zeros_like(d), where=hi > 0)
    return np.stack([hue, sat, hi], -1)


def rgb_to_lab(rgb):
    x = np.asarray(rgb, np.float64) / 255.
    linear = np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750],
                             [.0193339, .1191920, .9503041]]).T
    xyz /= np.array([.95047, 1., 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16,
                     500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def color_support(rgb_pixels):
    """Fixed smooth HSV/Lab descriptors; supports need not sum to one."""
    pixels = np.asarray(rgb_pixels)
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.dtype != np.uint8:
        raise ValueError('Original uint8 Nx3 RGB samples required')
    if not len(pixels):
        return {name: None for name in COLOR_RGB_PROTOTYPES}
    hsv = rgb_to_hsv(pixels)
    lab = rgb_to_lab(pixels)
    h, s, v = hsv.T
    L = lab[:, 0]
    result = {}
    for name, prototype in COLOR_RGB_PROTOTYPES.items():
        ph, ps, pv = rgb_to_hsv(np.array([prototype]))[0]
        hue_distance = np.minimum(np.abs(h - ph), 1 - np.abs(h - ph))
        chromatic = np.exp(-.5 * (hue_distance / .075) ** 2) * np.clip(s / .35, 0, 1)
        if name == 'black':
            hsv_support = np.clip((.35 - v) / .25, 0, 1)
        elif name == 'white':
            hsv_support = np.clip((v - .60) / .30, 0, 1) * np.clip((.30 - s) / .25, 0, 1)
        elif name == 'gray':
            hsv_support = np.clip((.30 - s) / .25, 0, 1) * np.exp(-.5 * ((v - .52) / .25) ** 2)
        elif name == 'brown':
            hsv_support = chromatic * np.exp(-.5 * ((L - 40) / 20) ** 2)
        elif name in ('pink', 'beige'):
            hsv_support = chromatic * np.exp(-.5 * ((s - ps) / .25) ** 2) * np.exp(-.5 * ((L - 75) / 25) ** 2)
        else:
            hsv_support = chromatic * np.clip((v - .08) / .25, 0, 1)
        plab = rgb_to_lab(np.array([prototype]))[0]
        lab_support = np.exp(-.5 * np.sum(((lab - plab) / np.array([30., 30., 30.])) ** 2, axis=1))
        result[name] = float(np.mean(.5 * hsv_support + .5 * lab_support))
    return result


def _entity_requests(graph):
    if not isinstance(graph, dict):
        raise ValueError('Text graph must be a plain JSON object')
    # Round23 preserves Round20 subject fields in base_text. Do not reparse the
    # entire caption and accidentally attach a reference's color to the target.
    base = graph.get('base_text', graph)
    if not isinstance(base, dict):
        raise ValueError('Invalid pure-text base graph')
    subject_text = base.get('own_attributes_literal', graph.get('subject_query', ''))
    entities = [{'entity': 'T', 'role': 'subject', 'literal_attributes': subject_text}]
    refs = graph.get('references', [])
    if not isinstance(refs, list):
        raise ValueError('Text reference list required')
    for ref in refs:
        if not isinstance(ref, dict):
            raise ValueError('Invalid pure-text reference')
        entities.append({'entity': ref.get('entity'), 'role': 'reference',
                         'literal_attributes': ref.get('own_query', ref.get('phrase', ''))})
    for entity in entities:
        text = entity['literal_attributes']
        if not isinstance(text, str):
            raise ValueError('Literal attribute text must be a string')
        words = re.findall(r"[a-z]+", text.casefold())
        entity['recognized_color_words'] = sorted({COLOR_ALIASES.get(w, w) for w in words
                                                  if w in COLOR_RGB_PROTOTYPES or w in COLOR_ALIASES})
        # Remainder is retained literally, not interpreted as a negative label.
        color_tokens = set(COLOR_RGB_PROTOTYPES) | set(COLOR_ALIASES)
        entity['unrecognized_literal_tokens'] = [w for w in words if w not in color_tokens]
        entity['unimplemented_appearance_status'] = 'UNKNOWN_not_inferred_from_rules'
    return entities


def measure_attributes(*, rgb, raw_points, instance_arrays, text_graph=None):
    """Return JSON-safe measured attributes plus unchanged inert text graph.

    No per-candidate preference or penalty is produced. Missing evidence is
    UNKNOWN, not incompatible. Geometry spans are visible surface measurements
    and must never replace the source complete-box extent or object center.
    """
    image = np.asarray(rgb)
    raw = np.asarray(raw_points)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Original RGB uint8 HxWx3 required')
    if raw.ndim != 2 or raw.shape[1] != 3 or not np.isfinite(raw).all():
        raise ValueError('Finite raw Nx3 required')
    a = instance_arrays
    offsets = a['instance_crop_mask_pixel_offsets']
    point_offsets = a['instance_visible_point_offsets']
    if len(offsets) != len(point_offsets):
        raise ValueError('Inconsistent instance bundle')
    scene_xy = a['visible_pixels_xy']
    if len(scene_xy) and ((scene_xy < 0).any() or (scene_xy[:, 0] >= image.shape[1]).any() or (scene_xy[:, 1] >= image.shape[0]).any()):
        raise ValueError('Scene visible point pixels out of bounds')
    scene_color = color_support(image[scene_xy[:, 1], scene_xy[:, 0]])
    nscene = len(scene_xy)
    rows = []
    for i in range(len(offsets) - 1):
        xy = a['instance_crop_mask_pixel_xy'][offsets[i]:offsets[i + 1]]
        ids = a['instance_visible_point_indices'][point_offsets[i]:point_offsets[i + 1]]
        if len(xy) != len(ids) or len(np.unique(xy, axis=0)) != len(xy):
            raise ValueError('Only unique visible point pixels may supply attributes')
        if len(xy) and ((xy < 0).any() or (xy[:, 0] >= image.shape[1]).any() or (xy[:, 1] >= image.shape[0]).any()):
            raise ValueError('Attribute pixel out of original image bounds')
        if len(ids) and ((ids < 0).any() or (ids >= len(raw)).any()):
            raise ValueError('Attribute point out of original XYZ bounds')
        pixels = image[xy[:, 1], xy[:, 0]]
        enough_color = len(xy) >= MINIMUM_COLOR_PIXELS
        enough_geometry = len(ids) >= MINIMUM_GEOMETRY_POINTS
        local_color = color_support(pixels)
        corrected_color = {name: None for name in COLOR_RGB_PROTOTYPES}
        if enough_color and nscene >= MINIMUM_COLOR_PIXELS:
            for name in corrected_color:
                p = (len(xy) * local_color[name] + COLOR_BETA_PSEUDOCOUNT) / (len(xy) + 2 * COLOR_BETA_PSEUDOCOUNT)
                b = (nscene * scene_color[name] + COLOR_BETA_PSEUDOCOUNT) / (nscene + 2 * COLOR_BETA_PSEUDOCOUNT)
                corrected_color[name] = float(np.clip(np.log(p / (1 - p)) - np.log(b / (1 - b)),
                                                      -COLOR_LOGODDS_CLIP, COLOR_LOGODDS_CLIP))
        geometry = {'status': 'UNKNOWN_insufficient_visible_points',
                    'visible_surface_span_xyz_m': None, 'visible_surface_pca_eigenvalues': None,
                    'span_coordinate_frame': 'raw_pointcloud_not_complete_model_box',
                    'visible_depth_camera_q10_q50_q90_m': None,
                    'complete_object_size': None, 'physical_object_center': None}
        if enough_geometry:
            p = raw[ids].astype(np.float64)
            eigenvalues = np.linalg.eigvalsh(np.cov(p.T))[::-1]
            geometry.update(status='MEASURED_visible_surface_only',
                            visible_surface_span_xyz_m=np.ptp(p, axis=0).tolist(),
                            visible_surface_pca_eigenvalues=eigenvalues.tolist(),
                            visible_depth_camera_q10_q50_q90_m=np.quantile(
                                a['visible_depth_camera_m'][a['visible_point_owner_instance_ids'] == i],
                                [.1, .5, .9]).tolist())
        rows.append({'instance_id': i, 'n_unique_visible_rgb_pixels': len(xy),
                     'color_status': 'MEASURED_sparse_visible_support' if enough_color else 'UNKNOWN_insufficient_visible_pixels',
                     'color_soft_support': local_color if enough_color else {k: None for k in COLOR_RGB_PROTOTYPES},
                     'color_logodds_relative_scene_visible': corrected_color,
                     'color_evidence_reliability': float(len(xy) / (len(xy) + MINIMUM_COLOR_PIXELS)) if enough_color else 0.,
                     'rgb_median': np.median(pixels, axis=0).tolist() if len(pixels) else None,
                     'geometry': geometry,
                     'semantic_category': None, 'complex_appearance': None,
                     'absence_of_attribute_evidence_is_not_negative': True})
    graph = copy.deepcopy(text_graph) if text_graph is not None else {}
    # Only JSON data; prohibit Python objects/callables/NaN introduced by caller.
    graph_json = json.dumps(graph, sort_keys=True, allow_nan=False)
    return {'schema': 'R51_sparse_measured_attributes_v1',
            'minimum_color_pixels': MINIMUM_COLOR_PIXELS,
            'minimum_geometry_points': MINIMUM_GEOMETRY_POINTS,
            'color_prototypes_rgb': {k: list(v) for k, v in COLOR_RGB_PROTOTYPES.items()},
            'color_aliases': COLOR_ALIASES.copy(), 'instances': rows,
            'scene_visible_color_soft_support': scene_color,
            'n_scene_visible_pixels': nscene,
            'color_background_definition': 'all_global_zbuffer_visible_point_pixels_including_unassigned_and_ambiguous_not_rect_background',
            'color_logodds_beta_pseudocount': COLOR_BETA_PSEUDOCOUNT,
            'color_logodds_clip': COLOR_LOGODDS_CLIP,
            'rgb_sha256': hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
            'text_graph_unchanged': graph,
            'text_graph_sha256': hashlib.sha256(graph_json.encode()).hexdigest(),
            'entity_attribute_requests': _entity_requests(graph),
            'point_support_pixels_only': True, 'external_vision_model_or_mask_consumed': False,
            'complex_appearance_not_rule_guessed': True}

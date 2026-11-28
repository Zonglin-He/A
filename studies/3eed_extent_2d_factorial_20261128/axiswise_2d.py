"""Independent axis-wise fitting and first-rectangle ranking for the locked C_ray pair."""
from dataclasses import dataclass

import numpy as np

import live_2d as official

geometry = official.geometry
live = official.live
HEADINGS = ('pca', 'qwen')
METHOD = 'ray_equivalent_extent_2d_rank'


@dataclass(frozen=True)
class PosePair:
    spec: object
    image_size: np.ndarray
    transform: np.ndarray
    rectangle: np.ndarray | None
    boxes: tuple
    ray_winner: int
    support: np.ndarray
    selection: np.ndarray
    dimensions: np.ndarray
    rotation: np.ndarray


def prepare(frame, replies):
    """Use the registered vehicle gate, sampling and pre-fit pose selection."""
    raw = replies['3d']['value']
    pixels = replies['2d']['value']
    if raw is None or pixels is None or not live.vehicle_reply(replies['2d']):
        return None
    spec, size, transform = geometry.camera(frame)
    uv, returns = geometry.raw_returns(frame, spec, size)
    window = np.asarray(pixels, dtype=float)*np.tile(size, 2)/1000.
    central, cells = geometry.z_buffer(uv, returns, np.flatnonzero(geometry.inside(uv, window, .5)))
    central = central[geometry.reserve(cells) == 0]
    if len(central) < live.MIN_RETURNS:
        return None
    selected = official.prepare_pair(frame, window, spec, size, uv, returns, transform, raw, central)
    if selected is None:
        return None
    pair, ray_winner, support, selection, dimensions, rotation = selected
    return PosePair(spec, size, transform, official.visible_rectangle(window, size), tuple(pair), ray_winner,
                    support, selection, dimensions, rotation)


def witnesses(ctx, box):
    depth = live.first_depth(box, ctx.selection, ctx.transform)
    return live.protected_points(box, ctx.support, ctx.selection, depth, ctx.transform)


def fit(ctx, box, protected, *, height_only):
    return live.clip_extent(box, ctx.dimensions, ctx.rotation, *protected, height_only=height_only)


def rank(ctx, boxes):
    if ctx.rectangle is None:
        return ctx.ray_winner, (None, None)
    scores = tuple(official.projection_overlap(b, ctx.rectangle, ctx.spec, ctx.transform, ctx.image_size)
                   for b in boxes)
    ray = ctx.ray_winner
    alt = 1-ray
    improves = scores[alt] is not None and scores[alt] > (scores[ray] if scores[ray] is not None else 0.)
    return (alt if improves else ray), scores


def fallback(frame, replies):
    return dict(live.frozen_predict(frame, replies), extent_corrected=False)


def output(ctx, boxes, winner):
    box = boxes[winner]
    return dict(box=box.tolist(), route='C_ray', applicable=True, readout='C_ray',
                selection_rays=len(ctx.selection), heading=HEADINGS[winner],
                extent_corrected=box is not ctx.boxes[winner])


def predict(frame, replies):
    """Project both poses independently, then use the original 2D selection rule."""
    ctx = prepare(frame, replies)
    if ctx is None:
        return fallback(frame, replies)
    boxes = tuple(fit(ctx, box, witnesses(ctx, box), height_only=False) for box in ctx.boxes)
    winner, _ = rank(ctx, boxes)
    return output(ctx, boxes, winner)

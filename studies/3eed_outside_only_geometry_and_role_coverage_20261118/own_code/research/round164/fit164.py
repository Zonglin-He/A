"""Matched fixed-winner LCIB fits with either signed-surface or outside-only loss."""
import gc
import hashlib
import time

import numpy as np
from scipy.optimize import minimize

from scope164 import HERE, PLATFORMS, SETTINGS, W163, WORK, read, sensor_row, sha, source_index, source_scope, write_seal
from cycle163 import first_update, part_state, winner_box, winner_support
from evidence162 import EvidenceGrounder
from geometry162 import axial, axes, effective_count, surface_cost, to_box, to_parameters
from run162 import array_file, pool, submit, sync
from solver162 import Options, Relationships, build_roles, choose
from support162 import SceneSupport

ARMS = ('r158', 'initial', 'surface_s0', 'outside_s0', 'surface_s1', 'outside_s1', 'surface_s1_no_relation')
FITS = ((0, surface_cost, True), (0, None, True), (1, surface_cost, True), (1, None, True), (1, surface_cost, False))


def outside_cost(points, params, scale, derivative=False):
    """Use the original pseudo-Huber on distance outside the box, without inside attraction."""
    rotation = axes(params[6])
    local = (points - params[:3]) @ rotation
    half = np.exp(params[3:6]) / 2
    excess = np.maximum(np.abs(local) - half, 0.)
    distance = np.linalg.norm(excess, axis=1) / scale
    root = np.sqrt(1 + (distance / .2)**2)
    loss = .2**2 * (root - 1)
    if not derivative:
        return loss
    normal = excess / (root[:, None] * scale**2)
    local_grad = normal * np.sign(local)
    center = -local_grad @ rotation.T
    size = -normal * half
    yaw = local_grad[:, 0] * local[:, 1] - local_grad[:, 1] * local[:, 0]
    return loss, np.c_[center, size, yaw]


def fit(fitter, points, weights, relation, cost):
    initial = fitter.initial
    if len(points) < 8 or effective_count(weights) < 6 or weights.sum() < .75:
        return initial.copy(), 0, 0, False
    assert not fitter.fixed_dims
    x0 = to_parameters(initial)
    half = np.r_[.45*np.maximum(initial[3:6], .30), np.full(3, .48), 1.1]
    denominator = weights.sum() + fitter.missing_mass

    def objective(params):
        per_point, jacobian = cost(points, params, fitter.scale, derivative=True)
        penalty, grad = fitter.regularize(params)
        value = fitter.mean_cost(per_point, weights) + penalty
        grad += weights @ jacobian / denominator
        if relation is not None:
            loss, partial = relation(params[:3])
            value += loss
            grad[:3] += partial
        return value, grad

    start = objective(x0)[0]
    result = minimize(objective, x0, method='L-BFGS-B', jac=True, bounds=list(zip(x0-half,x0+half)),
                      options={'maxiter':fitter.fit_iterations, 'maxls':8, 'ftol':1e-10})
    accepted = np.isfinite(result.fun) and result.fun <= start + 1e-10
    return to_box(result.x) if accepted else initial.copy(), int(result.nfev), int(result.status), bool(accepted)


def held_out(points, raw_indices):
    """Split original returns deterministically and never split coincident returns."""
    assert len(raw_indices) == len(np.unique(raw_indices)) == len(points)
    _, inverse = np.unique(points, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    first = np.full(int(inverse.max())+1, np.iinfo(np.int64).max, dtype=np.int64) if len(inverse) else np.empty(0,dtype=np.int64)
    np.minimum.at(first, inverse, raw_indices)
    order = sorted(range(len(first)), key=lambda j:hashlib.sha256(f'R164-fit-split|{first[j]}'.encode()).digest())
    held = np.zeros(len(first), dtype=bool)
    held[order[:len(first)//5]] = True
    return held[inverse]


def empty(initial):
    return [(initial.copy(), 0, 0, False) for _ in FITS]


def replay(path, ev, options, prior, sigma, parent_path):
    scene = SceneSupport(ev)
    roles = build_roles(scene, options, prior, sigma)
    relation = Relationships(ev, roles, options.relation_weight)
    h0 = part_state(roles, choose(scene, roles, relation))
    old = ev.old_pca.astype(float)
    initial = winner_box(roles, h0, old)
    seed0 = winner_support(scene, roles, h0)
    dense0 = roles[0].hypotheses[h0].point_weights(roles[0].state[:, h0]) if h0 >= 0 else np.empty(0)
    first_update(roles)
    seed1 = winner_support(scene, roles, h0)
    off_active = False
    if h0 >= 0:
        role = roles[0]
        region = role.hypotheses[h0]
        assert options.init != 'inside' and np.array_equal(region.box, initial)
        selected = scene.select_near(initial)
        points = region.points
        np.testing.assert_array_equal(scene.points[selected], points)
        raw = ev.indices[scene.unique[selected]]
        weights = np.stack((dense0, region.point_weights(role.state[:, h0])))
        held = held_out(points, raw)
        term = relation.neighbors(0, h0, region.box[:3])
        off_active = term is not None

        def scaled(center):
            val, grad = term(center)
            return val/region.fit_weight, grad/region.fit_weight

        scale = region.fitter.scale

        def arm(is_full, support, cost, use_relation):
            include = np.ones(len(raw),bool) if is_full else ~held
            penalty = scaled if use_relation and off_active else None
            return fit(region.fitter, points[include], weights[support, include], penalty, cost or outside_cost)

        full = [arm(True, *spec) for spec in FITS]
        split = [arm(False, *spec) for spec in FITS]
    else:
        points, raw, weights = np.empty((0,3)), np.empty(0,dtype=int), np.empty((2,0))
        held, scale = np.empty(0,dtype=bool), 1.
        full, split = empty(initial), empty(initial)

    boxes = np.stack((old, initial, *(x[0] for x in full)))
    split_boxes = np.stack((old, initial, *(x[0] for x in split)))
    with np.load(parent_path, allow_pickle=False) as ref:
        names = ref['arms'].tolist()
        np.testing.assert_array_equal(boxes[0], ref['boxes'][names.index('r158')])
        np.testing.assert_array_equal(boxes[1], ref['boxes'][names.index('initial')])
        assert h0 == int(ref['h0'])
        np.testing.assert_array_equal(np.stack((seed0,seed1)), ref['supports'][:2])
        np.testing.assert_array_equal(ev.seeds, ref['seeds'])
        np.testing.assert_allclose(boxes[ARMS.index('surface_s1')], ref['boxes'][names.index('all7')], atol=1e-10,rtol=0.)
    assert (boxes[:,3:6] > 0).all() and np.isfinite(boxes).all()
    assert (split_boxes[:,3:6] > 0).all() and np.isfinite(split_boxes).all()
    array_file(path, arms=np.asarray(ARMS), full=boxes, split=split_boxes, points=points, raw_indices=raw,
               weights=weights, held=held, scale=scale, relation_active=off_active, has_winner=h0>=0,
               full_fits=np.asarray([x[1:] for x in full]), split_fits=np.asarray([x[1:] for x in split]))


def check_cost():
    """Check the requested sphere counterexample, exterior equivalence and all seven derivatives."""
    box = np.array([5.,0.,0.,2.,2.,2.,0.])
    half_gap = np.array([[4.4,.8,0.]])
    val, jac = outside_cost(half_gap, to_parameters(box), 1., True)
    assert (val==0).all() and (jac==0).all() and surface_cost(half_gap,to_parameters(box),1.)[0]>0
    rng = np.random.default_rng(164)
    box = np.array([.6,-.4,.1,1.3,2.,1.8,.3])
    points = rng.normal(0, 1.6, (80,3))
    params = to_parameters(box)
    scale = .72
    v, g = outside_cost(points,params,scale,True)
    vv, gg = surface_cost(points,params,scale,True)
    mask = v>0
    np.testing.assert_allclose(v[mask], vv[mask], atol=1e-14,rtol=0.)
    np.testing.assert_allclose(g[mask], gg[mask], atol=1e-13,rtol=0.)
    np.testing.assert_array_equal(g[~mask],0.)
    for i,e in enumerate(np.eye(7)*1e-6):
        grad = (outside_cost(points,params+e,scale) - outside_cost(points,params-e,scale)) / 2e-6
        np.testing.assert_allclose(g[:,i], grad, atol=1e-8,rtol=0.)


def main():
    import torch

    assert read(HERE/'STATE.json')['state'] == 'running'
    assert not (HERE/'GEOMETRY_SEAL.json').exists()
    check_cost()
    refs = source_index()
    records = source_scope()
    settings = read(SETTINGS)['sources']
    folder = WORK/'geometry'
    folder.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with pool() as executor:
        for source in PLATFORMS:
            spec = settings[source]
            opts = Options(**spec['options'])
            prior, sigma = [np.asarray(spec[k]) for k in ('log_dimensions','log_scale')]
            model = EvidenceGrounder(source)
            pending = []
            for i, row in enumerate(records):
                if row['source'] != source:
                    continue
                name = f'{i:05d}.npz'
                p = W163/'source_predictions'/name
                assert sha(p) == refs[name]
                ev = model.evidence_frame(sensor_row(row))
                submit(executor,pending,replay,folder/name,ev,opts,prior,sigma,p)
            sync(pending)
            print('Verified parent fitter, dense support and new matched arms',source,64,flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    write_seal('GEOMETRY',folder,range(192),parent_seal_sha256=sha(W163/'SOURCE_FILES_PRIVATE.json'),
               sphere_and_gradient_checked=True, original_C_replayed=192, frozen_forward_count=192,
               ground_truth_in_fits=False, elapsed=time.perf_counter()-start)


if __name__=='__main__':
    main()

"""Predict geometry interventions from frozen Qwen replies and unlabeled returns."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '3eed_qwen_four_scale_recording_confirmation_20261121'
OUT = HERE / 'results'
PRIVATE = HERE / 'private'
SPEC = HERE / 'PROTOCOL.json'
sys.path.insert(0, str(BASE))

from cohort import checked_lock, digest, read, read_panel, write
from geometry import WAYMO_ANGLES, camera, raw_returns, rigid_xyz, to_sensor, variants
from predict import aligned_completion, fit_sample, sensor_frame_box, to_long_axis
from region_depth import (held_cells, inside, model_sample, model_to_camera_matrix,
                          partition, project_model_points, regions, transfer, z_buffer)
from score_confirmation import ray_residual


FACTORS = ('P', 'Q_yaw', 'Q_center', 'Q_size', 'Q_joint')
LEVELS = tuple(f'L{l}' for l in regions.FOUR_LEVEL_INDICES)
SCALES = LEVELS + ('Euclidean', 'Window')
FAMILIES = ('F', 'C')
STRICT = tuple(f'{name}_{arm}' for name in FAMILIES for arm in ('pca', 'qwen', 'ray'))
ARMS = ('A',) + FACTORS + SCALES + STRICT + ('P_transfer',)


def reserve(cells):
    bits = (cells[:, 0]*73856093) ^ (cells[:, 1]*19349663)
    return np.where((bits & 1)==0, 0, 1+((bits>>1)&1))


def choose(mapped, window_mask, central_count):
    """Apply the same original window-attribution vote to arbitrary components."""
    if central_count < 5 or not window_mask.any() or (mapped < 0).all():
        return None
    whole, central = mapped[:, :len(window_mask)], mapped[:, len(window_mask):]
    overlap = whole[:, window_mask]
    count = int(mapped.max()) + 1
    exposed = np.bincount(whole[whole>=0], minlength=count)
    intersect = np.bincount(overlap[overlap>=0], minlength=count)
    center = np.bincount(central[central>=0], minlength=count)
    valid = np.flatnonzero((center>=5) & (intersect>0))
    if not len(valid):
        return None
    weight = intersect[valid].astype(float)**2/(exposed[valid]*int(window_mask.sum()))
    return int(valid[int(np.argmax(weight))])


def tree_winner(sample, catalog, returns, window_mask, central_count):
    cuts, assign, births, _ = catalog
    if not len(births):
        return None
    mapped = transfer(sample, returns, cuts, assign, births)
    idx = choose(mapped, window_mask, central_count)
    if idx is None:
        return None
    level = births[idx]
    return dict(node=idx, level=int(level), points=sample[assign[level]==idx],
                central=np.flatnonzero(mapped[level, len(window_mask):]==idx))


def one_level(sample, labels, radius, nearest, distances, window_mask, central_count):
    matched = np.where(distances <= radius, labels[nearest], -1)[None, :]
    idx = choose(matched, window_mask, central_count)
    return None if idx is None else sample[labels==idx]


def euclidean_labels(sample, foreground, radius):
    cloud = sample[foreground]
    contacts = cKDTree(cloud).query_pairs(radius, output_type='ndarray')
    adjacency = coo_matrix((np.ones(len(contacts),bool), (contacts[:,0], contacts[:,1])),
                           shape=(len(cloud),len(cloud)))
    _, label = connected_components(adjacency,directed=False)
    eligible = np.bincount(label)[label]>=8
    labels = np.full(len(sample),-1,int)
    labels[foreground] = np.where(eligible,label,-1)
    return labels


def fixed_size_anchor(points, terrain, yaw, dims, viewpoint):
    """Use the candidate's camera-facing slab and bottom with the same extents."""
    c,s = np.cos(yaw),np.sin(yaw)
    axes = np.array([[c,s],[-s,c]])
    lower,upper = (op(points[:,:2]@axes.T,axis=0) for op in (np.min,np.max))
    eye = viewpoint[:2]@axes.T
    middle = (lower+upper)/2
    center = np.where(eye<lower,lower+dims[:2]/2,np.where(eye>upper,upper-dims[:2]/2,middle))@axes
    low = float(points[:,2].min())
    if np.isfinite(terrain).all():
        low = min(low,float(center@terrain[:2]+terrain[2]))
    return np.r_[center, low+dims[2]/2, dims, yaw]


def disagreement(pair, points, transform, cap):
    """A shared pixel-level loss, with a finite maximum loss for non-hit rays."""
    result = []
    for candidate in pair:
        hit, residual = ray_residual(np.asarray(candidate),points,transform)
        clipped = np.where(hit, np.minimum(residual/cap,1.), 1.)
        result.append((hit,residual,clipped))
    return result


def factorial(box, qpose, qyaw):
    base = to_long_axis(box)
    assert abs(np.sin(qpose[6]-qyaw)) < 1.e-12
    yaw = np.r_[base[:6], qyaw]
    center = np.r_[qpose[:3], base[3:6], qyaw]
    extent = np.r_[base[:3], qpose[3:6], qyaw]
    return dict(P=box.tolist(), Q_yaw=yaw.tolist(), Q_center=center.tolist(),
                Q_size=extent.tolist(), Q_joint=qpose.tolist())


def predict_frame(row, reply, archived):
    target,view = row['target'],row['suffix']
    raw = reply['3d']
    pixel = reply['2d']
    baseline = archived['boxes']['A']
    record = dict(boxes=dict.fromkeys(ARMS,baseline), full=False, strict=False, scales_valid=[],
                  isolation={}, selection_rays=0, audit_rays=0, fit_match=None, chooses={},
                  transfer=False, status='missing_reply')
    boxes = record['boxes']
    if raw is None or pixel is None:
        return record
    raw = np.asarray(raw,float)
    assert pixel==archived['pixel']
    spec,size,transform = camera(row)
    uv, returns = raw_returns(row,spec,size)
    window = np.asarray(pixel)*np.tile(size,2)/1000.
    a, _, _ = variants(raw,pixel,spec,size,uv,returns)
    prior, _ = to_sensor(a['ray_depth'],transform)
    np.testing.assert_array_equal(prior,baseline)
    dims,qyaw = prior[3:6],prior[6]
    qmodel = qyaw + (np.deg2rad(WAYMO_ANGLES[view]) if target=='waymo' else 0.)
    transform_model = model_to_camera_matrix(spec)
    xyz = rigid_xyz(returns,np.linalg.inv(transform_model))
    def to_output(box):
        return sensor_frame_box(box,target,view).tolist()

    indices,cells = z_buffer(uv,returns,np.arange(len(uv)))
    parts = reserve(cells)
    np.testing.assert_array_equal(parts>0,held_cells(cells))
    centers,center_cells = z_buffer(uv,returns,np.flatnonzero(inside(uv,window,.5)))
    center = centers[reserve(center_cells)==0]
    n_center = len(center)
    if n_center < 5:
        record['status']='insufficient_original_center'
        return record
    scene = indices[parts==0]
    in_window = inside(uv[scene],window)
    combined = xyz[np.r_[scene,center]]
    sample = model_sample(row)
    catalog = partition(sample)
    full = tree_winner(sample,catalog,combined,in_window,n_center)
    plane = regions.terrain(sample)
    foreground = sample[:,2]-sample[:,:2]@plane[:2]-plane[2] > .20
    if full is not None:
        p = regions.complete(regions.measured_box(full['points'],plane),dims)
        q = aligned_completion(full['points'],plane,qmodel,dims)
        boxes.update(factorial(np.asarray(to_output(p)),np.asarray(to_output(q)),qyaw))
        boxes['P_transfer'] = boxes['P']
        record.update(full=True, full_node=full['node'], full_level=full['level'],
                      face_center_shift=float(np.linalg.norm(p[:3]-q[:3])))

    cuts,labels = regions.levels(sample,foreground)
    distances,nearest = cKDTree(sample).query(combined)
    def complete_support(name, cloud):
        if cloud is not None:
            boxes[name] = to_output(regions.complete(regions.measured_box(cloud,plane),dims))
            record['scales_valid'].append(name)

    for name,lab,radius in zip(LEVELS,labels,cuts):
        complete_support(name,one_level(sample,lab,radius,nearest,distances,in_window,n_center))
    euclid_radius = cuts[regions.FOUR_LEVEL_INDICES.index(5)]
    if np.isfinite(euclid_radius):
        lab = euclidean_labels(sample,foreground,euclid_radius)
        complete_support('Euclidean',one_level(sample,lab,euclid_radius,nearest,distances,in_window,n_center))
    projected = project_model_points(sample,spec,size)
    window_support = sample[foreground & projected['in_image'] & inside(projected['uv'],window)]
    if len(window_support)>=8:
        complete_support('Window',window_support)

    fit, audit = fit_sample(row,spec,size)
    record['isolation'] = audit
    fit_catalog = partition(fit)
    chosen = tree_winner(fit,fit_catalog,combined,in_window,n_center)
    if chosen is None:
        record['status']='no_fit_node'
        return record
    fit_plane = regions.terrain(fit)
    measured = regions.measured_box(chosen['points'],fit_plane)
    p = regions.complete(measured,dims)
    q = aligned_completion(chosen['points'],fit_plane,qmodel,dims)
    p_model = to_long_axis(p)
    eye = np.linalg.inv(transform_model)[:3,3]
    fixed_p = fixed_size_anchor(chosen['points'],fit_plane,p_model[6],dims,eye)
    fixed_q = fixed_size_anchor(chosen['points'],fit_plane,qmodel,dims,eye)
    pairs = {'F':(p,q),'C':(fixed_p,fixed_q)}
    cuts,assign,births,_ = fit_catalog
    window_indices = indices[inside(uv[indices],window) & (parts>0)]
    matched = transfer(fit,xyz[window_indices],cuts,assign,births)[chosen['level']]==chosen['node']
    selected_ids = window_indices[matched]
    filtered_cells = (uv[selected_ids]//4).astype(np.int64)
    fold = reserve(filtered_cells)
    assert (fold>0).all() and len(np.unique(filtered_cells,axis=0))==len(filtered_cells)
    pick = selected_ids[fold==1]
    report = selected_ids[fold==2]
    assert not np.intersect1d(pick,report).size
    record.update(strict=True, status='usable_pair' if len(pick)>=5 else 'too_few_selection_rays',
                  selection_rays=len(pick), audit_rays=len(report))
    cap = max(.1, .5*min(dims[:2]))
    for name,(p,q) in pairs.items():
        converted = [to_output(b) for b in (p,q)]
        choice = 0
        costs = None
        if len(pick)>=5:
            costs = [float(clipped.mean()) for _,_,clipped in disagreement(converted,returns[pick],transform,cap)]
            choice = int(costs[1]<costs[0]-1.e-12)
        record['chooses'][name] = dict(qwen=bool(choice), selection_loss=costs)
        for arm,b in zip(('pca','qwen','ray'),[*converted,converted[choice]]):
            boxes[f'{name}_{arm}'] = b
    # This payload is consumed only by the separate offline evaluator.
    audit_window = indices[(parts==2) & inside(uv[indices],window)]
    audit_uses_node = np.isin(audit_window,report)
    assert int(audit_uses_node.sum())==len(report) and not np.intersect1d(pick,audit_window).size
    record.update(audit_payload=dict(optical=returns[audit_window].tolist(), fit_node=audit_uses_node.tolist(),
                                     transform=transform.tolist(), cap=cap))
    if full is not None:
        central_match = len(np.intersect1d(full['central'],chosen['central']))/len(np.union1d(full['central'],chosen['central']))
        record['fit_match'] = central_match
        if central_match>=.5 and record['chooses']['F']['qwen']:
            boxes['P_transfer'] = boxes['Q_joint']
            record['transfer'] = True
    return record


def load_prior():
    seal = read(BASE/'results/prediction_seal.json')
    prior = BASE/'private/predictions.json'
    model_file = BASE/'results/model_seal.json'
    hashes = read(BASE/'private/reply_hashes.json')
    model = read(model_file)
    assert digest(model_file)==seal['qwen_seal']
    assert hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()==model['ordered_reply_hash']
    assert digest(prior)==seal['predictions'] and not seal['task_labels_read']
    assert seal['method_lock']==checked_lock() and not seal['FINAL_access']
    predictions = read(prior)
    panel = read_panel()
    assert len(hashes)==2*len(panel)==832 and set(predictions)=={p['uid'] for p in panel}
    return panel,predictions,model,hashes


def run():
    start = perf_counter()
    panel,old,model,hashes = load_prior()
    table, counts, isolation = {},Counter(),Counter()
    for frame in panel:
        uid = frame['uid']
        answers = {}
        for mode in ('2d','3d'):
            key = f'{uid}_{mode}'
            path = BASE/'private/replies'/f'{key}.json'
            assert digest(path)==hashes[key]
            reply = read(path)
            assert reply['pins']==model['pins'] and reply['task']==mode and reply['uid']==uid
            answers[mode] = reply['value']
        assert digest(frame['points'])==frame['point_sha']
        entry = predict_frame(frame,answers,old[uid])
        assert entry['full']==old[uid]['selected']
        if entry['full']:
            assert entry['full_node']==old[uid]['node']
        for a,b in (('A','A'),('P','H'),('Q_yaw','H_yaw_only'),('Q_joint','H_qwen_reanchored')):
            np.testing.assert_array_equal(entry['boxes'][a],old[uid]['boxes'][b])
        counts[entry['status']]+=1
        isolation.update(entry['isolation'])
        table[uid]=entry
    path = PRIVATE/'predictions.json'
    write(path,table)
    assert isolation['retained_withheld_raw']==0
    write(OUT/'prediction_seal.json',dict(protocol=digest(SPEC), predictions=digest(path),
        implementation=digest(__file__), prior_prediction_seal=digest(BASE/'results/prediction_seal.json'),
        panel=digest(BASE/'private/panel.json'), frames=len(table), exact_previous_box_comparisons=4*len(table),
        exact_full_region_replay=sum(p['full'] for p in table.values()), isolation=dict(isolation),
        stages=dict(counts), qwen_calls=0, target_labels_read=False, FINAL_access=False,
        seconds=perf_counter()-start))
    print('Sealed',len(table),'frames',dict(counts),flush=True)


if __name__=='__main__':
    run()

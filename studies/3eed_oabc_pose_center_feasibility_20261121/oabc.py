"""Label-free center search on the previous frozen Qwen observations."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from feasible_geometry import CameraRays, bounds, continuous_fit, inside, line_witnesses, observe_axes, rotation


HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / '3eed_pose_consistent_visible_surface_grounding_20261121'
PRIVATE, OUT, SPEC = HERE/'private', HERE/'results', HERE/'PROTOCOL.json'
sys.path.insert(0,str(PREVIOUS))

from surface_grounding import (BASE, camera, digest, fit_sample, fixed_size_anchor, held_cells,
    inside as pixel_inside, load_prior, model_to_camera_matrix, partition, raw_returns,
    read, reserve, rigid_xyz, sensor_frame_box, to_long_axis, to_sensor, transfer, tree_winner,
    write, z_buffer, regions, WAYMO_ANGLES)
from predict import to_raw_support


POSES = ('pca','qwen')
ARMS = ('A','anchor_pca','anchor_qwen','anchor_pair','L7','plain_pca','plain_qwen','plain_pair',
        'feasible_pca','feasible_qwen','feasible_pair','B','C','D_no_yaw_prior','D','D_L7')


def evenly_spaced(points, cap):
    if len(points)<=cap:
        return points
    return points[((np.arange(cap)+.5)*len(points)/cap).astype(int)]


def center_box(center, theta, dims):
    return np.r_[np.asarray(center)@rotation(theta).T, dims, theta].tolist()


def fit_pose(support, anchor, prior, views, transform, cfg, allow_plain):
    theta = anchor[6]
    dims = np.asarray(anchor[3:6])
    basis = rotation(theta)
    local = support @ basis
    anchor_center, qcenter = anchor[:3]@basis, prior[:3]@basis
    epsilon = cfg['tolerance_m']
    strict = bounds(local,dims)
    low,high = bounds(local,dims,cfg['trim_quantile'],epsilon)
    feasible = (low<=high).all()
    cap = max(.1,.5*min(dims[:2]))
    data = dict(theta=float(theta), strict=[x.tolist() for x in strict], domain=[low.tolist(),high.tolist()],
                feasible=bool(feasible), anchor=anchor.tolist(), solved=False)
    if len(views)<cfg['minimum_rays']:
        return data
    rays = CameraRays(views,transform,theta,dims,cap)
    n_grid,max_eval = cfg['grid_samples_per_axis'],cfg['powell_evaluations_per_start']
    if allow_plain:
        outer_l,outer_h = np.minimum(anchor_center-dims,low), np.maximum(anchor_center+dims,high)
        plain, value, _, _ = continuous_fit(rays,outer_l,outer_h,anchor_center,n_grid,max_eval)
        data.update(plain=center_box(plain,theta,dims), plain_fit_loss=value)
    if not feasible:
        return data
    minimum, energy, candidates, losses = continuous_fit(rays,low,high,anchor_center,n_grid,max_eval)
    edge = line_witnesses(rays,minimum,low,high,energy+epsilon)
    tested = np.unique(np.vstack((candidates,edge)),axis=0)
    losses = rays.losses(tested)
    best = int(np.lexsort((np.linalg.norm(tested-anchor_center,axis=1),np.round(losses,12)))[0])
    minimum,energy = tested[best],float(losses[best])
    center, adopted_loss, counts, local_width, strong = observe_axes(rays,minimum,low,high,qcenter,
        epsilon,cfg['min_active_face_rays'],energy)
    tested = np.vstack((tested,center))
    losses = np.r_[losses,adopted_loss]
    assert inside(tested,low,high).all()
    physical = np.all(np.abs(local-center) <= dims/2.+epsilon,axis=1)
    data.update(solved=True, geometry=center_box(minimum,theta,dims), oabc=center_box(center,theta,dims),
        fit_min=energy, fit_oabc=adopted_loss, ray_caps_m=cap, witness_centers=tested.tolist(), witness_losses=losses.tolist(),
        support_fraction=float(physical.mean()), local_width=local_width.tolist(), face_counts=counts.tolist(),
        prior_move=np.abs(center-minimum).tolist(), strong=strong.tolist())
    return data


def readout(poses, field, select, transform, cfg, fallback, yaw_prior=False):
    valid = [p for p in POSES if field in poses[p]]
    if not valid or len(select)<cfg['minimum_rays']:
        return fallback, None, {}
    if field!='plain':
        best = min(poses[p]['fit_min'] for p in valid)
        valid = [p for p in valid if poses[p]['fit_min'] <= best+cfg['tolerance_m']+1.e-10]
    scores = {}
    for name in valid:
        box = np.asarray(poses[name][field])
        rays = CameraRays(select,transform,box[6],box[3:6],max(.1,.5*min(box[3:5])))
        scores[name] = float(rays.losses(box[:3]@rotation(box[6]))[0])
    chosen = min(valid,key=scores.__getitem__)
    if yaw_prior and 'qwen' in scores and scores['qwen'] <= scores[chosen]+cfg['tolerance_m']:
        chosen='qwen'
    return poses[chosen][field],chosen,scores


def camera_evidence(row, wanted, frozen):
    spec,size,transform = camera(row)
    uv, optical = raw_returns(row,spec,size)
    window = np.asarray(wanted[0])*np.tile(size,2)/1000.
    matrix = model_to_camera_matrix(spec)
    xyz = rigid_xyz(optical,np.linalg.inv(matrix))
    all_ids,cells = z_buffer(uv,optical,np.arange(len(uv)))
    folds = reserve(cells)
    np.testing.assert_array_equal(folds>0,held_cells(cells))
    central,central_cells = z_buffer(uv,optical,np.flatnonzero(pixel_inside(uv,window,.5)))
    central = central[reserve(central_cells)==0]
    assert len(central)>=5
    scene = all_ids[folds==0]
    fit,iso = fit_sample(row,spec,size)
    in_window = pixel_inside(uv[scene],window)
    combined = xyz[np.r_[scene,central]]
    catalog = partition(fit)
    node = tree_winner(fit,catalog,combined,in_window,len(central))
    assert node is not None
    selected = pixel_inside(uv[all_ids],window)
    return dict(row=row, transform=transform, points=xyz[all_ids[selected]], rays=optical[all_ids[selected]],
        fold=folds[selected], fit=fit, catalog=catalog, node=node, isolation=iso, plane=regions.terrain(fit),
        eye=np.linalg.inv(matrix)[:3,3], prior=to_sensor(np.asarray(wanted[1]),transform)[0],
        old_isolation=frozen['isolation'], combined=combined, in_window=in_window, central_count=len(central))


def pose_anchors(context, node):
    frame = context['row']
    target,view = frame['target'],frame['suffix']
    cloud,plane = node['points'],context['plane']
    dims = context['prior'][3:6]
    measured = regions.measured_box(cloud,plane)
    p = to_long_axis(regions.complete(measured,dims))[6]
    q = context['prior'][6] + (np.deg2rad(WAYMO_ANGLES[view]) if target=='waymo' else 0.)
    return {name:sensor_frame_box(fixed_size_anchor(cloud,plane,angle,dims,context['eye']),target,view)
            for name,angle in zip(POSES,(p,q))}


def solve_region(context, node, catalog, anchors, cfg, allow_plain):
    cuts,labels,births,_=catalog
    membership = transfer(context['fit'],context['points'],cuts,labels,births)[node['level']]==node['node']
    fit,selector,audit = [context['rays'][membership & (context['fold']==k)] for k in range(3)]
    views = evenly_spaced(fit,cfg['fit_ray_cap'])
    frame = context['row']
    cloud = to_raw_support(node['points'],frame['target'],frame['suffix'])
    poses = {name:fit_pose(cloud,box,context['prior'],views,context['transform'],cfg,allow_plain)
             for name,box in anchors.items()}
    active = [d for d in poses.values() if d['solved']]
    near_minimum = min((min(d['witness_losses']) for d in active),default=None)
    for p in active:
        allowed = np.asarray(p['witness_losses']) <= near_minimum+cfg['tolerance_m']+1.e-10
        witnesses = np.array(p['witness_centers'])[allowed]
        p.update(near_count=int(allowed.sum()), near_width=np.ptp(witnesses,axis=0).tolist() if len(witnesses) else None)
    return dict(poses=poses, support=cloud.tolist(), fit_raw_count=len(fit), fit_rays=views.tolist(),
                selection=selector,audit=audit.tolist(),near_minimum=near_minimum)


def process(frame, reply, previous, locked, cfg):
    old = previous['boxes']
    assert old['P']==locked['boxes']['H']
    boxes = dict(A=old['P'], anchor_pca=old['C_pca'],anchor_qwen=old['C_qwen'],anchor_pair=old['C_ray'],L7=old['L7'])
    boxes.update(plain_pca=boxes['anchor_pca'],plain_qwen=boxes['anchor_qwen'],plain_pair=boxes['anchor_pair'],
        feasible_pca=boxes['anchor_pca'],feasible_qwen=boxes['anchor_qwen'],feasible_pair=boxes['anchor_pair'],
        B=boxes['anchor_pca'],C=boxes['anchor_qwen'],D_no_yaw_prior=boxes['anchor_pair'], D=boxes['anchor_pair'],D_L7=boxes['anchor_pair'])
    outcome = dict(boxes=boxes, fit_node=False, status=previous['status'])
    if not previous['strict']:
        return outcome
    context = camera_evidence(frame,reply,previous)
    assert context['isolation']==context['old_isolation']
    anchors = pose_anchors(context,context['node'])
    for name,a in anchors.items():
        np.testing.assert_allclose(a,boxes[f'anchor_{name}'],rtol=0,atol=1.e-12)
    np.testing.assert_array_equal(anchors['pca'][3:6],anchors['qwen'][3:6])
    primary = solve_region(context,context['node'],context['catalog'],anchors,cfg,True)
    assert len(primary['selection'])==previous['selection_rays'] and len(primary['audit'])==previous['audit_rays']
    poses,select = primary['poses'],primary.pop('selection')
    assert set(poses)==set(POSES)
    for name,p in poses.items():
        boxes[f'plain_{name}']=p.get('plain',boxes[f'anchor_{name}'])
        boxes[f'feasible_{name}']=p.get('geometry',boxes[f'anchor_{name}'])
        boxes['B' if name=='pca' else 'C']=p.get('oabc',boxes[f'anchor_{name}'])
    choices = {}
    for arm,field,yaw in (('plain_pair','plain',False),('feasible_pair','geometry',False),
            ('D_no_yaw_prior','oabc',False),('D','oabc',True)):
        boxes[arm],chosen,ranks = readout(poses,field,select,context['transform'],cfg,boxes['anchor_pair'],yaw)
        choices[arm]=dict(pose=chosen,selection_losses=ranks)

    level = regions.FOUR_LEVEL_INDICES[-1]
    sample, plane = context['fit'],context['plane']
    foreground = sample[:,2]-sample[:,:2]@plane[:2]-plane[2] > .20
    cut,labels=regions.levels(sample,foreground,indices=(level,))
    single = cut,labels,np.zeros(int(labels.max())+1,dtype=int),()
    node = tree_winner(sample,single,context['combined'],context['in_window'],context['central_count'])
    single_box = None
    if node is not None:
        small = pose_anchors(context,node)
        extra = solve_region(context,node,single,small,cfg,False)
        single_choice = extra['selection']
        default = small['pca'].tolist()
        if len(single_choice)>=cfg['minimum_rays']:
            losses = []
            for a in small.values():
                cam=CameraRays(single_choice,context['transform'],a[6],a[3:6],max(.1,.5*min(a[3:5])))
                losses.append(cam.losses(a[:3]@rotation(a[6]))[0])
            if losses[1] < losses[0]-1.e-12:
                default = small['qwen'].tolist()
        boxes['D_L7'],single_box,_ = readout(extra['poses'],'oabc',single_choice,context['transform'],cfg,default,True)
    assert set(boxes)==set(ARMS)
    outcome.update(primary,fit_node=True,selection_count=len(select), dims=anchors['pca'][3:6].tolist(),
        transform=context['transform'].tolist(), raw_qwen_center=context['prior'][:3].tolist(),
        isolation=context['isolation'], choices=choices, single_node=node is not None, single_choice=single_box)
    return outcome


def verify_sources():
    before = read(PREVIOUS/'results/prediction_seal.json')
    plan = PREVIOUS/'PROTOCOL.json'
    predictions = PREVIOUS/'private/predictions.json'
    assert digest(predictions)==before['predictions'] and digest(plan)==before['protocol']
    assert digest(PREVIOUS/'surface_grounding.py')==before['implementation']
    assert not before['target_labels_read'] and not before['FINAL_access']
    panel,locked,model,hashes = load_prior()
    assert before['panel']==digest(BASE/'private/panel.json')
    previous = read(predictions)
    work = []
    cfg = read(SPEC)
    for frame in panel:
        uid=frame['uid']
        answers=[]
        for name in ('2d','3d'):
            slug=f'{uid}_{name}'
            path=BASE/'private/replies'/f'{slug}.json'
            assert digest(path)==hashes[slug]
            value=read(path)
            assert value['pins']==model['pins'] and value['task']==name and value['uid']==uid
            answers.append(value['value'])
        assert digest(frame['points'])==frame['point_sha']
        work.append((frame,answers,previous[uid],locked[uid],cfg))
    return panel,work


def evaluate_one(args):
    return process(*args)


def run():
    began = perf_counter()
    panel,work=verify_sources()
    data={}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for row, result in zip(panel,pool.map(evaluate_one,work,chunksize=4)):
            data[row['uid']]=result
            if len(data)%64==0:
                print('Predicted',len(data),flush=True)
    fate = Counter(r['status'] for r in data.values())
    iso = Counter()
    for r in data.values():
        iso.update(r.get('isolation',{}))
    path=PRIVATE/'predictions.json'
    write(path,data)
    assert len(data)==416 and iso['retained_withheld_raw']==0
    write(OUT/'prediction_seal.json',dict(protocol=digest(SPEC), predictions=digest(path),
        implementation={name:digest(HERE/name) for name in ('oabc.py','feasible_geometry.py')},
        previous_prediction_seal=digest(PREVIOUS/'results/prediction_seal.json'),panel=digest(BASE/'private/panel.json'),
        frames=len(data),replayed_qwen_replies=832, qwen_calls=0,baseline_exact=len(data),
        fit_node_exact=sum(r['fit_node'] for r in data.values()),ray_isolation=dict(iso),stages=dict(fate),
        elapsed_seconds=perf_counter()-began,target_labels_read=False, FINAL_access=False))
    print('Sealed',len(data),'frames',dict(fate),flush=True)


if __name__=='__main__':
    run()

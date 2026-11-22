"""Fit an upright enclosure independently of the prior, then audit top-only clipping."""
from itertools import product

import numpy as np

from object_region_audit import ious
from object_region_plan import proposal_regions, valid_xyz


STRIDE = 4
EPS = 1e-7
UP = np.array([0., -1., 0.])
COS_UP = np.cos(np.deg2rad(30))
COS_AGREE = np.cos(np.deg2rad(12))


def unit(vector):
    return vector / np.linalg.norm(vector)


def unit_rows(rows):
    norm = np.linalg.norm(rows, axis=-1, keepdims=True)
    return np.divide(rows, norm, out=np.zeros_like(rows), where=norm > 1e-12)


def inside_rect(x, y, bbox):
    x0, y0, x1, y1 = bbox
    return (x+0.5 >= x0) & (x+0.5 < x1) & (y+0.5 >= y0) & (y+0.5 < y1)


def estimate_up(grid, valid, gx, gy, boxes):
    origin = grid[:-1, :-1]
    across = grid[:-1, 1:] - origin
    down = grid[1:, :-1] - origin
    normal = unit_rows(np.cross(across, down))
    normal = np.where((normal @ UP < 0)[..., None], -normal, normal)
    x, y = gx[:-1, :-1], gy[:-1, :-1]
    select = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & (y >= 224)
    for box in boxes:
        select &= ~(inside_rect(x, y, box) | inside_rect(x+STRIDE, y, box) | inside_rect(x, y+STRIDE, box))
    z = origin[...,2]
    select &= (normal @ UP >= COS_UP)
    select &= (np.linalg.norm(across,axis=-1) <= .10*z) & (np.linalg.norm(down,axis=-1) <= .10*z)
    select &= (normal * origin).sum(axis=-1) < 0
    p, n = origin[select], normal[select]
    xx, yy = x[select], y[select]
    details = {'eligible_triangles':len(p)}
    if len(p) < 32:
        return None, details

    candidate = np.linspace(0, len(p)-1, min(len(p),64), dtype=int)
    offsets = (n[candidate] * p[candidate]).sum(axis=1)
    near = np.abs(p @ n[candidate].T - offsets) <= .015*p[:,2,None]
    compatible = (n @ n[candidate].T >= COS_AGREE) & near
    chosen = compatible[:,compatible.sum(axis=0).argmax()]
    if chosen.sum() < max(32, .35*len(p)):
        return None, details
    centroid = p[chosen].mean(axis=0)
    _, spread, axes = np.linalg.svd(p[chosen] - centroid, full_matrices=False)
    if spread[1] < .10*spread[0]:
        return None, details
    up = axes[2] if axes[2] @ UP > 0 else -axes[2]
    chosen = (n @ up >= COS_AGREE) & (np.abs((p-centroid)@up) <= .015*p[:,2])
    if up @ UP < COS_UP or centroid @ up >= 0 or chosen.sum() < max(32,.35*len(p)):
        return None, details
    xx, yy = xx[chosen], yy[chosen]
    tiles = xx//112 + 4*(yy//112)
    if np.ptp(xx) < .20*448 or len(np.unique(tiles)) < 3:
        return None, details
    details.update(consensus_triangles=int(chosen.sum()),
                   camera_up_angle_deg=float(np.degrees(np.arccos(np.clip(up@UP,-1,1)))),
                   plane_below_camera_m=float(-centroid@up), plane_tiles=len(np.unique(tiles)))
    return up, details


def upright_enclosure(points, up):
    center = points.mean(axis=0)
    _, spectrum, pca = np.linalg.svd(points-center, full_matrices=False)
    if spectrum[1] < 1e-7:
        return None
    vertical = np.argmax(np.abs(pca@up))
    horizontal_axes = np.delete(pca,vertical,axis=0)
    projected = horizontal_axes - (horizontal_axes@up)[:,None]*up
    local = (points-center)@pca.T
    lengths = np.ptp(local,axis=0)
    direction = unit(projected[np.argmax(np.delete(lengths,vertical)*np.linalg.norm(projected,axis=1))])
    world_to_box = np.column_stack((direction, np.cross(up,direction), up))
    corners = np.array(list(product(*zip(local.min(axis=0),local.max(axis=0),strict=True))))
    upright_corners = (corners@pca + center) @ world_to_box
    bounds = np.stack((upright_corners.min(axis=0), upright_corners.max(axis=0)))
    if (np.ptp(bounds, axis=0) <= 1e-6).any():
        return None
    return world_to_box, bounds


def cube_entry(directions, bounds):
    """Slab intersections for rays from the camera origin in an orthonormal box frame."""
    lo, hi = bounds
    parallel = np.abs(directions) < 1e-14
    outside = (lo > 0) | (hi < 0)
    a = np.divide(lo, directions, out=np.zeros_like(directions), where=~parallel)
    b = np.divide(hi, directions, out=np.zeros_like(directions), where=~parallel)
    enter = np.where(parallel, -np.inf, np.minimum(a,b)).max(axis=1)
    leave = np.where(parallel, np.inf, np.maximum(a,b)).min(axis=1)
    near = np.maximum(enter,0)
    hit = (leave >= near-EPS) & (leave > 0) & ~np.any(parallel & outside,axis=1)
    return np.where(hit,near,np.nan)


def ray_difference(old, new):
    hit0, hit1 = np.isfinite(old), np.isfinite(new)
    common = hit0 & hit1
    depth = np.abs(old[common]-new[common])
    changed = np.count_nonzero(depth > EPS)
    return dict(total=len(old), incumbent_hits=int(hit0.sum()), hit_changes=int((hit0!=hit1).sum()),
                common_depth_changes=int(changed), max_common_entry_delta_m=float(depth.max(initial=0)))


def split_support(xyz, support, bbox, size, grid, valid):
    xx, yy = np.meshgrid(np.arange(112)*STRIDE,np.arange(112)*STRIDE)
    width,height = size
    used = set()
    splits = []
    for offset in (2,1,3):
        x, y = xx+offset, yy+offset
        raw = (y*height//448)*width + x*width//448
        # Assign every source pixel to its first split, including Fit outside this region.
        unshared = ~np.isin(raw, list(used))
        used.update(raw.ravel().tolist())
        mask = support & inside_rect(x,y,bbox) & valid_xyz(xyz[:,y,x]) & unshared
        observed = xyz[:3,y,x].transpose(1,2,0)
        if offset != 2:
            mask &= valid & (np.linalg.norm(observed-grid,axis=-1) <= .10*np.maximum(observed[...,2],grid[...,2]))
        raw_selected = raw[mask]
        _, first = np.unique(raw_selected, return_index=True)
        unique = observed[mask][first].astype(np.float64)
        splits.append(unique)
    return splits


def geometry(rgb, xyz, bbox, boxes, size, prior, proposal_config):
    from warnings import catch_warnings, filterwarnings

    grid = xyz[:3,2::4,2::4].transpose(1,2,0).astype(np.float64)
    valid = valid_xyz(xyz[:,2::4,2::4])
    gx, gy = np.meshgrid(np.arange(2,448,4),np.arange(2,448,4))
    up, plane = estimate_up(grid,valid,gx,gy,boxes)
    result = dict(status='no_scene_plane',plane=plane)
    if up is None:
        return result
    with catch_warnings():
        filterwarnings('ignore',message='Got image with third dimension of 6',category=RuntimeWarning)
        regions,_ = proposal_regions(rgb,xyz,proposal_config)
    overlap = ious(np.array(bbox),[r['box'] for r in regions])
    result.update(status='no_region',region_iou=float(overlap.max(initial=0)))
    if not len(overlap) or overlap.max() < .25:
        return result
    support = regions[overlap.argmax()]['grid_mask']
    fit, selection, audit = split_support(xyz,support,bbox,size,grid,valid)
    result.update(status='not_enough_fit',split_counts=[len(fit),len(selection),len(audit)])
    if len(fit) < 24:
        return result

    enclosure = upright_enclosure(fit,up)
    if enclosure is None:
        result.update(status='degenerate_fit')
        return result
    axes,bounds = enclosure
    fit, sel = fit@axes, selection@axes
    bottom, top = bounds[:,2]
    h0 = top-bottom
    hvis = np.ptp(fit[:,2])
    fit_required = fit[:,2].max()-bottom
    assert np.all(fit>=bounds[0]-EPS) and np.all(fit<=bounds[1]+EPS)
    result.update(status='missing_height_prior',up=up.tolist(),axes=axes.tolist(),initial_bounds=bounds.tolist(),
                  visible_height_m=float(hvis),initial_height_m=float(h0),height_m=float(h0),no_ray_height_m=float(h0),
                  width_m=float(bounds[1,0]-bounds[0,0]),bottom_m=float(bottom),
                  four_scale_region_cells=int(support.sum()),raw_top_slack_m=float(h0-fit_required))
    if prior is None:
        return result
    if np.all((bounds[0] <= 0) & (bounds[1] >= 0)):
        result.update(status='camera_inside')
        return result
    rays = (selection / selection[:,2,None]) @ axes
    audit_rays = (audit / audit[:,2,None]) @ axes
    entry = cube_entry(rays,bounds)
    audit_entry = cube_entry(audit_rays,bounds)
    hit = np.isfinite(entry)
    if not hit.any():
        result.update(status='no_selection_hit')
        return result
    inside = np.all((sel>=bounds[0]-EPS)&(sel<=bounds[1]+EPS),axis=1)
    measured = np.vstack((fit,sel[inside]))
    entries = rays[hit]*entry[hit,None]
    if np.any(entries < bounds[0]-2*EPS) or np.any(entries > bounds[1]+2*EPS):
        result.update(status='entry_outside')
        return result
    point_floor = float(measured[:,2].max()-bottom+EPS)
    entry_floor = float(entries[:,2].max()-bottom+EPS)
    witness_floor = max(point_floor,entry_floor)
    with_point = min(h0,max(prior,point_floor))
    clipped = min(h0,max(prior,witness_floor))
    new_bounds = bounds.copy()
    new_bounds[1,2] = bottom+clipped
    proof = ray_difference(entry,cube_entry(rays,new_bounds))
    point_bounds = bounds.copy()
    point_bounds[1,2] = bottom+with_point
    result.update(status='invariant_fallback',witness_required_height_m=witness_floor,
                  fit_selection_required_height_m=point_floor,selection_entry_required_height_m=entry_floor,
                  selection_outside_count=int((~inside).sum()),selection=proof,
                  point_only_selection=ray_difference(entry,cube_entry(rays,point_bounds)),
                  selection_entry_on_top=int(np.count_nonzero(entries[:,2]>=top-EPS)))
    if proof['hit_changes'] or proof['common_depth_changes']:
        return result
    result.update(status='clipped' if h0-clipped > 1e-6 else 'unchanged',
                  height_m=float(clipped),no_ray_height_m=float(with_point),final_bounds=new_bounds.tolist(),
                  audit=ray_difference(audit_entry,cube_entry(audit_rays,new_bounds)))
    return result

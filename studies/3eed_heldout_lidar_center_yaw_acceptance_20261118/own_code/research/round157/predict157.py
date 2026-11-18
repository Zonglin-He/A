"""R157 preregistered no-label A/H geometry selection using held-out returns."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "research/round157"
OUT = ROOT / "crossplatform/round157"
UPSTREAM = ROOT / "crossplatform/round156"
DESIGN = HERE / "experiment/DESIGN_LOCK.json"
CODE_LOCK = HERE / "experiment/CODE_LOCK.json"
MODES = ("ray", "surface", "combined")
PARENT_SEAL = "1513f9addb268ebaa505d43337ae8a784905288cb094614a80e45991123795e9"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def basis(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def local(points, box):
    result = np.asarray(points, np.float64) - box[:3]
    result[..., :2] = result[..., :2] @ basis(box[6])
    return result


def pair_intersects(a, b):
    """Strictly positive-volume intersection of two yaw-oriented bounding boxes."""
    if min(a[2]+a[5]/2, b[2]+b[5]/2) <= max(a[2]-a[5]/2, b[2]-b[5]/2):
        return False
    axes_a, axes_b = basis(a[6]), basis(b[6])
    axes = np.column_stack([axes_a, axes_b])
    radius = a[3:5]/2 @ abs(axes_a.T @ axes) + b[3:5]/2 @ abs(axes_b.T @ axes)
    return bool(np.all(abs((a[:2]-b[:2]) @ axes) < radius))


def common_component(points, a, h):
    """Find an unambiguous connected component at the shared footprint.

    Each occupied voxel that contains a return inside both padded footprints votes
    once. The denominator contains all such voxels, across the entire local ROI.
    No winning cluster without strict majority or with too few common voxels.
    """
    voxel = np.clip(.15*min(a[3:6]), .10, .30)
    lo = np.minimum(a[:2] - abs(basis(a[6])) @ (a[3:5]/2+voxel),
                    h[:2] - abs(basis(h[6])) @ (h[3:5]/2+voxel))
    hi = np.maximum(a[:2] + abs(basis(a[6])) @ (a[3:5]/2+voxel),
                    h[:2] + abs(basis(h[6])) @ (h[3:5]/2+voxel))
    mask = ((points[:, 2] >= a[2]-.4*a[5]) & (points[:, 2] <= a[2]+.4*a[5])
            & (points[:, 0] >= lo[0]) & (points[:, 0] <= hi[0])
            & (points[:, 1] >= lo[1]) & (points[:, 1] <= hi[1]))
    ids = np.flatnonzero(mask)
    nearby = points[ids]
    inside_a = (abs(local(nearby,a)[:, :2]) <= a[3:5]/2+voxel).all(1)
    inside_h = (abs(local(nearby,h)[:, :2]) <= h[3:5]/2+voxel).all(1)
    take = inside_a | inside_h
    ids = ids[take]
    if len(ids) < 8:
        return None, "too_few_local_points"
    common = (inside_a & inside_h)[take]
    cells, inverse = np.unique(np.floor(points[ids]/voxel).astype(np.int64),axis=0,return_inverse=True)
    common_cells = np.unique(inverse[common])
    if len(common_cells) < 4:
        return None, "too_few_shared_cells"
    edges = cKDTree(cells).query_pairs(1.,p=np.inf,output_type="ndarray")
    graph = coo_matrix((np.ones(len(edges),np.int8),(edges[:,0],edges[:,1])),shape=(len(cells),len(cells)))
    n_comp, labels = connected_components(graph,directed=False)
    votes = np.bincount(labels[common_cells], minlength=n_comp)
    winner = int(votes.argmax())
    if 2*votes[winner] <= len(common_cells) or votes[winner] < 4:
        return None, "ambiguous_components"
    keep = labels[inverse] == winner
    chosen = ids[keep]
    if len(chosen) < 8:
        return None, "too_few_component_points"
    # Each occupied cell has equal total weight; all original returns are retained.
    _, compact, cell_counts = np.unique(inverse[keep],return_counts=True,return_inverse=True)
    weights = 1. / (cell_counts[compact] * len(cell_counts))
    return (chosen, weights, len(cell_counts), int(votes[winner]), len(common_cells)), "eligible"


def surface_residual(points, box):
    """Unsigned Euclidean distance to a bounded box surface, including interior points."""
    q = abs(local(points, box)) - box[3:6]/2
    return np.linalg.norm(np.maximum(q, 0.),axis=1) - np.minimum(q.max(axis=1),0.)


def ray_first(points, origin, box):
    """Distance along each half-ray from a verified physical sensor to first box hit."""
    axes_origin = local(origin,box)
    to_point = points-origin
    ranges = np.linalg.norm(to_point,axis=1)
    unit = np.divide(to_point,ranges[:,None],out=np.zeros_like(to_point),where=ranges[:,None]>0)
    unit[:, :2] = unit[:, :2] @ basis(box[6])
    parallel = abs(unit) < 1e-12
    h = box[3:6]/2
    lower = np.divide(-h-axes_origin,unit,out=np.zeros_like(unit),where=~parallel)
    upper = np.divide(h-axes_origin,unit,out=np.zeros_like(unit),where=~parallel)
    entry = np.where(parallel,-np.inf,np.minimum(lower,upper)).max(axis=1)
    exit_dist = np.where(parallel,np.inf,np.maximum(lower,upper)).min(axis=1)
    hit = ((~(parallel & (abs(axes_origin)>h)).any(axis=1)) & (entry<=exit_dist) &
           (entry>=0) & (ranges>0))
    return entry,hit,ranges


def evaluate(points, origin, a, d, h):
    rec = dict(eligible_ray=False, eligible_surface=False,eligible_combined=False,
               choose_ray=False,choose_surface=False,choose_combined=False,
               ray_A=np.nan,ray_H=np.nan,miss_A=np.nan,miss_H=np.nan,
               miss_raw_A=np.nan,miss_raw_H=np.nan,
               surface_A=np.nan,surface_H=np.nan,n_cluster_points=0,n_cluster_voxels=0,
               n_winner_shared_voxels=0,n_shared_voxels=0,
               ray_reason="origin_unknown" if origin is None else "locality_unavailable")
    if not (np.all(a[3:6]>0) and np.all(d[3:6]>0)):
        return dict(rec, geometry_reason="invalid_box")
    if not pair_intersects(a,d):
        return dict(rec, geometry_reason="disjoint_A_D")
    if (abs(local(a[:3],h)[:2])>h[3:5]/2).any() or (abs(local(h[:3],a)[:2])>a[3:5]/2).any():
        return dict(rec, geometry_reason="different_local_centers")
    if np.array_equal(a[:2],h[:2]) and abs(np.sin(a[6]-h[6]))<=1e-10:
        return dict(rec, geometry_reason="same_proposal")
    support,reason = common_component(points,a,h)
    if support is None:
        return dict(rec,geometry_reason=reason)
    ids,weight,ncells,winning_votes,votes = support
    selected = np.asarray(points[ids],np.float64)
    cap = max(.2,min(a[3],a[4])/2)
    rec.update(n_cluster_points=len(ids),n_cluster_voxels=ncells,n_winner_shared_voxels=winning_votes,
               n_shared_voxels=votes,geometry_reason="eligible",eligible_surface=True)
    for name,box in (("A",a),("H",h)):
        rec["surface_"+name] = float(weight @ np.minimum(surface_residual(selected,box),cap))
    rec["choose_surface"] = rec["surface_H"] + 1e-12 < rec["surface_A"]
    if origin is None:
        rec.update(eligible_combined=True,choose_combined=rec["choose_surface"])
        return rec
    origin=np.asarray(origin,np.float64)
    if origin.ndim==2:
        require(origin.shape==points.shape,"per-beam origin does not match points")
        origin=origin[ids]
    else:
        require(origin.shape==(3,),"sensor origin shape")
    require(np.isfinite(origin).all(),"nonfinite verified physical origin")
    if any(np.any((abs(local(origin,box)) <= box[3:6]/2).all(axis=-1)) for box in (a,h)):
        return dict(rec,ray_reason="origin_in_box")
    if np.any(np.linalg.norm(selected-origin,axis=1)<=0):
        return dict(rec,ray_reason="zero_range")
    rec.update(eligible_ray=True,eligible_combined=True,ray_reason="eligible")
    for name,box in (("A",a),("H",h)):
        entry,hit,ranges=ray_first(selected,origin,box)
        cost = np.full(len(selected),cap)
        cost[hit]=np.minimum(abs(ranges[hit]-entry[hit]), cap)
        rec["ray_"+name]=float(weight @ cost)
        rec["miss_"+name]=float(np.clip(weight @ (~hit),0.,1.))
        rec["miss_raw_"+name]=float(np.mean(~hit))
    rec["choose_ray"] = rec["ray_H"]+1e-12 < rec["ray_A"]
    rec["choose_combined"] = rec["choose_ray"]
    return rec


def shuffle_switches(flags, eligible, source, target, has_b, origin_ok, seed):
    result=np.zeros(len(flags),bool)
    rng=np.random.default_rng(seed)
    for s,t,b,ok in sorted(set(zip(source,target,has_b,origin_ok))):
        strata=(source==s)&(target==t)&(has_b==b)&(origin_ok==ok)&eligible
        chosen=flags[strata]
        result[strata]=chosen[rng.permutation(len(chosen))]
    require(result.sum()==flags.sum() and not any(result & ~eligible),"shuffle switch budget")
    return result


def synthetic():
    a = np.array([5.,0.,0.,2.,2.,2.,0.])
    orig = np.zeros(3)
    rays = np.array([[4.,0.,0.],[0.,3.,0.],[-4.,0.,0.],[6.,0.,0.],[4.,1.,1.]])
    e,h,r = ray_first(rays,orig,a)
    np.testing.assert_allclose(e[h],np.array([4.,4.,np.sqrt(18.)]))
    np.testing.assert_array_equal(h,[1,0,0,1,1])
    # A back-surface return lies on the box but the first-hit discrepancy is 2 m.
    assert r[3]-e[3] == 2. and surface_residual(rays[3:4],a)[0] == 0.
    rotated = a.copy()
    rotated[4],rotated[6] = 1.,np.pi/2
    assert np.isclose(ray_first(rays[:1],orig,rotated)[0][0],4.5)
    points = np.array([[4.,y,z] for y in np.linspace(-.75,.75,15) for z in [-.28,-.14,0.,.14,.28]])
    wrong = a.copy(); wrong[0] -= .22
    rec = evaluate(points,orig,wrong,a,a)
    assert rec["eligible_ray"] and rec["choose_ray"] and rec["choose_surface"] and rec["choose_combined"]
    assert rec["ray_H"] < 1e-14 and rec["n_cluster_points"] == len(points)
    assert rec["miss_H"] == 0. and rec["n_winner_shared_voxels"] == rec["n_shared_voxels"]
    assert not evaluate(points,orig,a,wrong,wrong)["choose_ray"]
    assert not evaluate(points[:1],orig,a,wrong,wrong)["eligible_surface"]
    unknown=evaluate(points,None,wrong,a,a)
    assert not unknown["choose_ray"] and unknown["choose_combined"] and np.isnan(unknown["miss_H"])
    beam=evaluate(points,np.tile(orig,(len(points),1)),wrong,a,a)
    assert np.isclose(beam["ray_A"],rec["ray_A"]) and beam["choose_ray"]
    far=a.copy(); far[0] += 3.
    assert evaluate(points,orig,a,far,far)["geometry_reason"] == "disjoint_A_D"
    # Shared-cell votes for two well separated, equally sized groups reject both.
    two=np.array([[4.,y,z] for y in [-.8,-.7,.7,.8] for z in [-.39,-.24,0.,.19,.36]])
    assert common_component(two,wrong,a)[1] == "ambiguous_components"
    np.testing.assert_allclose(surface_residual(np.array([[6.3,1.4,0.],[5,0,0.]]),a),[.5,1])
    print("R157_SYNTHETIC_PASS",flush=True)


def run():
    from heldout157 import iter_heldout
    start=monotonic()
    data=UPSTREAM/"PAIR_PRIVATE.npz"
    seal=json.loads((UPSTREAM/"PAIR_SEAL.json").read_bytes())
    require(digest(data)==PARENT_SEAL==seal["prediction_sha256"] and seal["status"]=="PASS","sealed parent")
    code=json.loads(CODE_LOCK.read_bytes())["files"]
    for fname in ("predict157.py","heldout157.py"):
        require(code[fname]==digest(HERE/fname),fname+" differs from lock")
    design_sha=digest(DESIGN)
    with np.load(data,allow_pickle=False) as f:
        a=f["boxes"][:,0,1].copy()
        d=f["boxes"][:,1,1].copy()
        ordinal=f["ordinal"].copy()
    h=a.copy()
    h[:,[0,1,6]]=d[:,[0,1,6]]
    require(a.shape==d.shape==h.shape==(1798,7) and np.array_equal(ordinal,np.arange(1,1799)),"cohort")
    dest=OUT/"PREDICTIONS_PRIVATE.npz"
    receipt=OUT/"PREDICTION_SEAL.json"
    require(not dest.exists() and not receipt.exists(),"refuse to overwrite predictions")
    rows=[]
    for i,(points,origin,meta) in enumerate(iter_heldout()):
        require(meta["ordinal"]==ordinal[i] and len(points)==meta["n_holdout"],"heldout order/count")
        require(points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all(),"heldout points")
        row={k:meta[k] for k in ("ordinal","source","target","strict_e2e","registration_exposure",
             "origin_reason","n_raw","n_unique_sample_indices","n_holdout_indices","n_holdout","alias_exclusions",
             "n_strict_unique","n_origin_unresolved")}
        row["origin_ok"]=origin is not None
        row["has_B"]=row["registration_exposure"]=="accepted_B"
        row.update(evaluate(points,origin,a[i].astype(float),d[i].astype(float),h[i].astype(float)))
        rows.append(row)
        if (i+1)%300==0:
            print("held-out geometry processed",i+1,flush=True)
    require(len(rows)==1798 and sum(r["has_B"] for r in rows)==764,"incomplete or changed cohort")
    result={k:np.array([r[k] for r in rows]) for k in rows[0]}
    # Use one reproducibly seeded permutation per mode. Matching strata retain geometry and observation coverage.
    for k in MODES:
        result["choose_shuffle_"+k]=shuffle_switches(result["choose_"+k],result["eligible_"+k],
            result["source"],result["target"],result["has_B"],result["origin_ok"],157)
        require(not np.any(result["choose_"+k] & ~result["eligible_"+k]),"ineligible H choice")
    result.update(A=a,D=d,H=h)
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(".tmp")
    with tmp.open("xb") as stream:
        np.savez_compressed(stream,**result)
    os.replace(tmp,dest)
    summary=dict(status="PASS",n=len(rows),prediction_sha256=digest(dest),parent_sha256=PARENT_SEAL,
                 GT_read=False,FINAL_access=False,source_training=False,model_forwards=0,
                 new_parameter_updates=0,design_sha256=design_sha,code_lock_sha256=digest(CODE_LOCK),
                 seconds=monotonic()-start,eligible={k:int(result["eligible_"+k].sum()) for k in MODES},
                 switches={k:int(result["choose_"+k].sum()) for k in MODES},
                 geometry_coverage=dict(Counter(str(x) for x in result["geometry_reason"])),
                 origin_coverage=dict(Counter(str(x) for x in result["origin_reason"])),
                 strict_e2e=int(result["strict_e2e"].sum()))
    tmp=receipt.with_suffix(".tmp")
    with tmp.open("x") as f:
        f.write(json.dumps(summary,indent=2)+"\n")
    os.replace(tmp,receipt)
    print("R157_PREDICTIONS_SEALED",json.dumps(summary),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--run",action="store_true")
    args=ap.parse_args()
    synthetic()
    if args.run:
        run()

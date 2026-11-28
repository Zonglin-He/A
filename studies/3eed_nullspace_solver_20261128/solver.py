"""Training-free IRLS using camera-proxy first-hit and image-edge Jacobians."""
from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np


FD = .005
RAY_DELTA = .20
MISS = .75
LAMBDA_PRIOR = .04
LAMBDA_IMAGE = .0625
ITERATIONS = 4
SIGNS = np.array(list(product((-.5,.5),repeat=3)))


@dataclass(frozen=True)
class Inputs:
    eye: np.ndarray
    rays: np.ndarray
    observed: np.ndarray
    matrix: np.ndarray
    K: np.ndarray
    distortion: np.ndarray
    size: np.ndarray
    window: np.ndarray | None


@dataclass(frozen=True)
class Arm:
    prior: str
    image: bool = False
    robust: bool = True
    shape_only: bool = False
    seed_center: bool = False


ARMS = {
    'ray_only': Arm('none'),
    'isotropic': Arm('isotropic'),
    'projected': Arm('hard'),
    'joint': Arm('hard', True),
    'joint_quadratic': Arm('hard', True, False),
    'joint_soft': Arm('soft', True),
    'joint_shape': Arm('hard', image=True, shape_only=True),
    'joint_no_prior': Arm('none', image=True),
    'joint_no_prior_quadratic': Arm('none', image=True, robust=False),
    'joint_seed_center': Arm('hard', image=True, seed_center=True),
}


def axial(angle):
    return (angle + np.pi / 2) % np.pi - np.pi / 2


def basis(boxes):
    cos, sin = np.cos(boxes[:,6]), np.sin(boxes[:,6])
    z, one = np.zeros(len(cos)), np.ones(len(cos))
    return np.stack((np.stack((cos,sin,z),axis=1),
                     np.stack((-sin,cos,z),axis=1), np.stack((z,z,one),axis=1)),axis=1)


def ray_depth(boxes, eye, rays):
    """Intersect rays, returning positive optical depth and the entry face ID."""
    A = basis(boxes)
    if not len(rays):
        shape = (len(boxes),0)
        return np.zeros(shape), np.full(shape,-1)
    start = np.einsum('bij,bj->bi', A, eye-boxes[:,:3])
    direction = np.einsum('bij,nj->bni', A, rays)
    half = boxes[:,None,3:6]/2
    parallel = abs(direction)<1e-10
    lower = np.divide(-half-start[:,None,:], direction, out=np.zeros_like(direction), where=~parallel)
    upper = np.divide(half-start[:,None,:], direction, out=np.zeros_like(direction), where=~parallel)
    near = np.where(parallel,-np.inf,np.minimum(lower,upper))
    far = np.where(parallel,np.inf,np.maximum(lower,upper))
    t = near.max(axis=2)
    axis = near.argmax(axis=2)
    face = 2*axis + (np.take_along_axis(direction,axis[:,:,None],axis=2)[:,:,0] >= 0)
    hit = (t>0) & (far.min(axis=2)>=t) & ~(parallel & (abs(start[:,None,:])>half)).any(axis=2)
    return np.where(hit,t,0.), np.where(hit,face,-1)


def image_residual(boxes, evidence):
    """Signed image-edge distances, with one-sided out-of-frame edges."""
    window = evidence.window
    if window is None:
        return np.zeros((len(boxes),0)), np.empty((len(boxes),0),dtype=int)
    A = basis(boxes)
    local = SIGNS[None,:,:]*boxes[:,None,3:6]
    world = local @ A + boxes[:,None,:3]
    cam = world @ evidence.matrix[:3,:3].T + evidence.matrix[:3,3]
    valid = (cam[:,:,2]>.10).all(axis=1)
    safe = cam.copy()
    safe[~valid] = np.array([0.,0.,1.])
    if evidence.distortion is not None and np.any(evidence.distortion):
        uv = cv2.projectPoints(safe.reshape(-1,3),np.zeros(3),np.zeros(3),
                              evidence.K,evidence.distortion)[0].reshape(len(boxes),8,2)
    else:
        uvz = safe @ evidence.K.T
        uv = uvz[:,:,:2]/uvz[:,:,2:3]
    sides = np.c_[uv.min(axis=1),uv.max(axis=1)]
    which = np.c_[uv.argmin(axis=1),uv.argmax(axis=1)]
    span = np.tile(window[2:]-window[:2],2)
    residual = sides-window
    low = (window[:2]<=0.)[None,:] & (residual[:,:2]<=0.)
    high = (window[2:]>=evidence.size)[None,:] & (residual[:,2:]>=0.)
    residual[np.c_[low,high]] = 0.
    which[np.c_[low,high]] += 8
    residual[~valid] = span
    which[~valid] = -1
    return residual/span, which


def physical(depth, face, observed, robust):
    r = np.where(face>=0,depth-observed,MISS)
    a = np.minimum(abs(r),MISS)
    if robust:
        cost = np.where(a<=RAY_DELTA,a*a/2,RAY_DELTA*(a-RAY_DELTA/2))
        weight = np.minimum(1., RAY_DELTA/np.maximum(abs(r),1e-12))
    else:
        cost, weight = a*a/2,np.ones_like(r)
    return r, cost, np.where((face>=0)&(abs(r)<MISS), weight, 0.)


def scales_for(box):
    return np.r_[np.clip(box[3:6]*.30,.18,1.),box[3:6]*.40,np.deg2rad(24.)]


def weak_prior(J, arm):
    # Full right basis is required even if only one ray is available.
    if not np.any(J):
        values, v = np.zeros(7),np.eye(7)
    else:
        _, values, v = np.linalg.svd(J,full_matrices=len(J)<7)
        values = np.pad(values,(0,7-len(values)))
    tau = max(.025,.08*values[0])
    rank = int((values>=tau).sum())
    if arm.prior == 'hard':
        reg = v[rank:]
    elif arm.prior == 'soft':
        reg = (tau/np.hypot(tau,values))[:,None]*v
    elif arm.prior == 'isotropic':
        reg = np.eye(7)
    else:
        reg = np.empty((0,7))
    if arm.shape_only:
        reg = reg @ np.diag([0.,0.,0.,1.,1.,1.,1.])
    return reg, values, rank


def spectra(J, K, values):
    s = np.linalg.svd(np.vstack((J,np.sqrt(LAMBDA_IMAGE)*K)),compute_uv=False)
    rank_joint = int((s>=max(.025,.08*s[0])).sum()) if len(s) else 0
    m = np.linalg.norm(K,ord=2) if K.size else 0.
    A = J/max(values[0],1e-12)
    B = K/max(m,1e-12)
    sv = np.linalg.svd(np.vstack((A,B)),compute_uv=False)
    modal = int((sv>=.08*sv[0]).sum()) if len(sv) and sv[0]>1e-12 else 0
    return rank_joint, modal


def signed_delta(box, prior, scales):
    offset = box-prior
    offset[6] = axial(offset[6])
    return offset/scales


def linearize(box, scales, evidence, arm):
    e = np.diag(scales*FD)
    batch = np.vstack((box,box+e,box-e))
    depth, face = ray_depth(batch,evidence.eye,evidence.rays)
    r, _, w = physical(depth[0],face[0],evidence.observed,arm.robust)
    stable = (face[0]>=0) & (face==face[0]).all(axis=0)
    weight = np.sqrt(stable*w/max(1,len(evidence.rays)))
    J = ((depth[1:8]-depth[8:])/(2*FD)).T*weight[:,None]

    image, active = image_residual(batch,evidence)
    stable_edges = (active[0]>=0) & (active==active[0]).all(axis=0)
    normalization = np.sqrt(max(1,len(image[0])))
    K = ((image[1:8]-image[8:])/(2*FD)).T*stable_edges[:,None]/normalization
    reg, sv, rank = weak_prior(J,arm)
    joint, modal = spectra(J,K,sv)
    info = dict(rank=rank,joint_rank=joint,modal_rank=modal, stable=int(stable.sum()),
                weighted=int((weight>0).sum()), edges=int(stable_edges.sum()))
    return J,r*weight,K,image[0]/normalization,reg,info


def energy(box, prior, scales, evidence, arm, reg):
    ys = box[None,:]
    d, face = ray_depth(ys,evidence.eye,evidence.rays)
    _, cost, _ = physical(d[0],face[0],evidence.observed,arm.robust)
    total = float(cost.mean()) if len(cost) else 0.
    if arm.image:
        image = image_residual(ys,evidence)[0][0]
        total += LAMBDA_IMAGE*np.dot(image,image)/(2*max(1,len(image)))
    delta = reg @ signed_delta(box, prior, scales)
    return total + .5*LAMBDA_PRIOR*np.dot(delta,delta)


def solve(seed, prior, evidence, arm):
    """Four iterations with relinearization and a frozen spectrum inside each line search."""
    box, scale = seed.copy(), scales_for(seed)
    if arm.seed_center:
        prior = np.r_[seed[:3], prior[3:]]
    origin, final = None, None
    accepted = 0
    for _ in range(ITERATIONS):
        J,r,K,p,reg,final = linearize(box,scale,evidence,arm)
        if origin is None:
            origin = final
        matrices = [J,np.sqrt(LAMBDA_PRIOR)*reg,np.sqrt(.02)*np.eye(7)]
        residuals = [r,np.sqrt(LAMBDA_PRIOR)*(reg @ signed_delta(box,prior,scale)),np.zeros(7)]
        if arm.image:
            matrices.append(np.sqrt(LAMBDA_IMAGE)*K)
            residuals.append(np.sqrt(LAMBDA_IMAGE)*p)
        step = np.linalg.lstsq(np.vstack(matrices),-np.concatenate(residuals),rcond=None)[0]
        size = np.max(abs(step))
        if size<1e-7:
            break
        step *= min(1.,.4/size)
        previous = energy(box,prior,scale,evidence,arm,reg)
        for fraction in (1.,.5,.25,.125):
            proposed = box+fraction*step*scale
            proposed[6] = axial(proposed[6])
            if min(proposed[3:6])>.08 and energy(proposed,prior,scale,evidence,arm,reg)<previous-1e-10:
                box = proposed
                accepted += 1
                break
        else:
            break
    if accepted:
        final = linearize(box,scale,evidence,arm)[-1]
    return box, dict(start=origin,end=final,accepted=accepted)

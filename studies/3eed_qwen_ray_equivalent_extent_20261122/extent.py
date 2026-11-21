"""Shrink the incumbent footprint within its first-intersection equivalence class."""
import numpy as np

from center_sets import axes, local


TOL = 1.e-8


def intrinsic_dimensions(dimensions, rotation):
    vertical = int(np.argmax(abs(rotation[2])))
    return np.r_[np.sort(np.delete(dimensions,vertical))[::-1], dimensions[vertical]], vertical


def wall_motion(box, intrinsic, camera):
    half = box[3:6]/2.
    shrink = np.maximum(box[3:6]-intrinsic,0.)
    eye = local(camera[None],box)[0]
    share = np.where(eye>half,1.,np.where(eye< -half,0.,.5))
    share[2] = 0.  # Retain the existing terrain or visible-support bottom plane.
    return shrink*share, shrink*(1.-share)


def allowed_fraction(half, advance, retreat, points, *, margin=TOL):
    if not len(points):
        return np.ones(3)
    if not (np.abs(points)<=half+TOL).all():
        raise ValueError("A protected point lies outside the incumbent box")
    rooms = (np.maximum(0.,points.min(0)+half-margin),np.maximum(0.,half-points.max(0)-margin))
    limits = (np.divide(room,rate,out=np.ones(3),where=rate>0.) for room,rate in zip(rooms,(advance,retreat),strict=True))
    return np.minimum.reduce((np.ones(3),*limits))


def shrink_box(box, advance, retreat, fraction):
    lo = -box[3:6]/2. + fraction*advance
    hi = box[3:6]/2. - fraction*retreat
    result = box.copy()
    result[:3] += ((lo+hi)/2) @ axes(box[6]).T
    result[3:6] = hi-lo
    return result


def contained(points, box):
    return (np.abs(local(points,box))<=box[3:6]/2+TOL).all(1)

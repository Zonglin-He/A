"""Inference entry from a normal frame and frozen Qwen 2D/3D replies."""
import numpy as np

from run_trial import contract, first_depth, geometry
from cray_simple import (FOUR_SCALES, MIN_RETURNS, choose_pose, choose_region,
                         predict as frozen_predict, vehicle_reply)


def predict(frame, replies):
    """Use the locked grounding path with a ray-equivalent extent correction."""
    raw = replies["3d"]["value"]
    pixel = replies["2d"]["value"]
    if raw is None or pixel is None or not vehicle_reply(replies["2d"]):
        return dict(frozen_predict(frame,replies),extent_corrected=False)
    spec,size,transform = geometry.camera(frame)
    uv,returns = geometry.raw_returns(frame,spec,size)
    window = np.asarray(pixel)*np.tile(size,2)/1000.
    central,cells = geometry.z_buffer(uv,returns,np.flatnonzero(geometry.inside(uv,window,.5)))
    central = central[geometry.reserve(cells)==0]
    if len(central)<MIN_RETURNS:
        return dict(frozen_predict(frame,replies),extent_corrected=False)
    sample,_ = geometry.fit_sample(frame,spec,size)
    if len(sample)<9:
        return dict(frozen_predict(frame,replies),extent_corrected=False)
    ids,cells = geometry.z_buffer(uv,returns,np.arange(len(uv)))
    split = geometry.reserve(cells)
    train = ids[split==0]
    select = ids[(split==1)&geometry.inside(uv[ids],window)]
    model_camera = geometry.model_to_camera_matrix(spec)
    inv = np.linalg.inv(model_camera)
    xyz = geometry.rigid_xyz(returns,inv)
    terrain = geometry.regions.terrain(sample)
    support,holding = choose_region(sample,terrain,FOUR_SCALES,xyz[np.r_[train,central]],
                                    geometry.inside(uv[train],window),xyz[select])
    if support is None or holding.sum()<MIN_RETURNS:
        return dict(frozen_predict(frame,replies),extent_corrected=False)
    raw = np.asarray(raw,float)
    upright,rotation = geometry.to_sensor(raw,transform)
    dims,yaw = upright[3:6],upright[6]
    if frame["target"]=="waymo":
        yaw += np.deg2rad(geometry.WAYMO_ANGLES[frame["suffix"]])
    held = returns[select[holding]]
    old,heading = choose_pose(support,terrain,dims,yaw,inv[:3,3],held,transform,frame)
    fit = geometry.rigid_xyz(support,np.linalg.inv(transform)@model_camera)
    variants,_,_ = contract(old,raw[3:6],rotation,fit,held,np.empty((0,3)),first_depth(old,held,transform),transform)
    corrected = variants.get("ray_equivalent")
    return dict(box=corrected["box"] if corrected is not None else old.tolist(), route="C_ray",
                applicable=True, readout="C_ray", selection_rays=len(held), heading=heading,
                extent_corrected=corrected is not None)

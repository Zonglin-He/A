"""Camera-box conversion and label-free first-return geometry."""
from itertools import product
import sys
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from paths import ROOT

sys.path.insert(0,str(ROOT if (ROOT/'crossplatform').is_dir() else ROOT/'v1'))
from crossplatform.round7.camera_projection import (CameraSpec, read_camera_allowlist,
     model_to_camera_matrix, raw_to_model_points, project_model_points, WAYMO_ANGLES, z_rotation)

SIGNS=np.array(list(product((-.5,.5), repeat=3)))
ARMS=('raw','image_ray','depth_only','ray_depth','mismatched_depth')

def camera(row):
    spec=CameraSpec.from_camera_fields(row['target'],read_camera_allowlist(row['calibration']),
                    waymo_view_suffix=row['suffix'],missing_distortion_policy='assume_zero_declared')
    with Image.open(row['rgb']) as img:size=np.array(img.size)
    # Model coordinate transforms in the 3EED loader cancel out for raw LiDAR.
    raw_to_model=np.eye(4)
    if row['target']=='drone':raw_to_model[2,3]=1.8
    if row['target']=='waymo':raw_to_model[:3,:3]=z_rotation(WAYMO_ANGLES[row['suffix']])
    return spec,size,model_to_camera_matrix(spec) @ raw_to_model

def rigid_xyz(points,mat):
    return np.asarray(points) @ mat[:3,:3].T + mat[:3,3]

def cook_rotation(box):
    return Rotation.from_euler('xyz',np.asarray(box[6:])*np.pi).as_matrix()

def extent_corners(c,dim,rot):
    return (SIGNS*np.asarray(dim)) @ rot.T + c

def axial(theta):
    return (theta+np.pi/2)%np.pi-np.pi/2

def upright(center,dim,rot):
    # The horizontal projection of a cuboid is a zonotope. Each hull edge is
    # parallel to a projected cuboid edge, so these three directions suffice.
    verts=(SIGNS*np.asarray(dim)) @ rot.T
    candidates=[]
    for axis in rot.T:
        u=axis[:2]
        if np.linalg.norm(u)<1e-12:continue
        u=u/np.linalg.norm(u);v=np.array([-u[1],u[0]])
        length=np.ptp(verts[:,:2]@u);width=np.ptp(verts[:,:2]@v)
        theta=np.arctan2(u[1],u[0])
        if width>length:length,width=width,length;theta+=np.pi/2
        candidates.append((length*width,length,width,axial(theta)))
    _,length,width,theta=min(candidates,key=lambda k:k[0])
    return np.r_[center,length,width,np.ptp(verts[:,2]),theta]

def to_sensor(box,mat):
    inv=np.linalg.inv(mat)
    c=rigid_xyz(box[:3],inv)
    rot=inv[:3,:3]@cook_rotation(box)
    return upright(c,box[3:6],rot),rot

def raw_returns(row,spec,size):
    path=row['points']
    arr=np.load(path,allow_pickle=False) if path.endswith('.npy') else np.fromfile(path,'<f4').reshape(-1,4)
    xyz=np.asarray(arr[:,:3],dtype=float)
    xyz=xyz[np.isfinite(xyz).all(1)]
    model=raw_to_model_points(xyz,row['target'],row['suffix'])
    proj=project_model_points(model,spec,size)
    cam=rigid_xyz(model,model_to_camera_matrix(spec))
    valid=proj['in_image']
    return proj['uv'][valid],cam[valid]

def pixel_rays(uv,spec):
    k,_,dist,_=spec.validated()
    uv=np.asarray(uv,float).reshape(-1,2)
    if np.any(dist):
        import cv2
        xy=cv2.undistortPoints(uv[:,None,:],k,dist)[:,0]
        return np.column_stack((xy,np.ones(len(xy))))
    return np.column_stack((uv,np.ones(len(uv))))@np.linalg.inv(k).T

def sensor_pixels(camera_points,spec):
    k,_,dist,_=spec.validated()
    camera_points=np.asarray(camera_points,float)
    if (camera_points[:,2]<=.1).any():return None
    if np.any(dist):
        import cv2
        return cv2.projectPoints(camera_points,np.zeros(3),np.zeros(3),k,dist)[0][:,0]
    p=camera_points@k.T
    return p[:,:2]/p[:,2:3]

def cell_points(uv,cam,rect):
    center=(rect[:2]+rect[2:])/2
    half=(rect[2:]-rect[:2])/4
    ids=np.flatnonzero((abs(uv-center)<=half).all(1))
    bins=(uv[ids]//4).astype(int)
    order=np.lexsort((cam[ids,2],bins[:,1],bins[:,0]))
    bins=bins[order];ids=ids[order]
    if not len(ids):return cam[:0],cam[:0]
    first=np.r_[True,(bins[1:]!=bins[:-1]).any(1)]
    bins=bins[first];ids=ids[first]
    hold=((bins[:,0]*73856093)^(bins[:,1]*19349663))%2==1
    return cam[ids[~hold]],cam[ids[hold]]

def front_offset(box,ray):
    # Central ray (dx/dz,dy/dz,1) through the 3D center. Measured in optical z.
    direction=abs(cook_rotation(box).T@ray)
    return np.min(np.divide(np.asarray(box[3:6])/2,direction,out=np.full(3,np.inf),where=direction>1e-12))

def intersections(box,cam):
    if not len(cam):return np.zeros(0,bool),np.zeros(0)
    rot=cook_rotation(box)
    u=(cam/cam[:,2:3])@rot
    origin=(-np.asarray(box[:3]))@rot
    half=np.asarray(box[3:6])/2
    parallel=abs(u)<=1e-12
    a=np.divide(-half-origin,u,out=np.zeros_like(u),where=~parallel)
    b=np.divide( half-origin,u,out=np.zeros_like(u),where=~parallel)
    near=np.where(parallel,-np.inf,np.minimum(a,b)).max(1)
    far=np.where(parallel,np.inf,np.maximum(a,b)).min(1)
    hit=(~(parallel&(abs(origin)>half)).any(1)) & (far>=near) & (near>0)
    return hit,np.where(hit,abs(cam[:,2]-near),np.nan)

def variants(base,rect,spec,size,uv,cam):
    boxes={k:np.array(base,float) for k in ARMS}
    diag={'fit_bins':0,'heldout_bins':0,'wrong_fit_bins':0}
    if rect is None:return boxes,diag,cam[:0]
    rect=np.asarray(rect,float)*np.tile(size,2)/1000
    center=(rect[:2]+rect[2:])/2
    guide=pixel_rays(center,spec)[0]
    ray=base[:3]/base[2]
    # Displace the actual fit window by half the image width. Clip by field of
    # view naturally; no borrowing returns from another scene or target label.
    dx=size[0]/2 if center[0]<size[0]/2 else -size[0]/2
    wrong=rect+np.array([dx,0,dx,0])
    fit,held=cell_points(uv,cam,rect)
    bad,_=cell_points(uv,cam,wrong)
    boxes['image_ray'][:3]=guide*base[2]
    for name,local,direct in [('depth_only',fit,ray),('ray_depth',fit,guide),('mismatched_depth',bad,guide)]:
        z=base[2] if len(local)<5 else np.median(local[:,2])+front_offset(base,direct)
        boxes[name][:3]=direct*z
    diag.update(fit_bins=len(fit),heldout_bins=len(held),wrong_fit_bins=len(bad))
    return boxes,diag,held

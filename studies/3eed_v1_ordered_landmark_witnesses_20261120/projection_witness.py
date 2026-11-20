"""Calibrated projections in the 3EED model coordinate system."""
import numpy as np
from collect import camera
from crossplatform.round7.camera_projection import project_model_points, model_to_camera_matrix

CORNERS = np.array([(x,y,z) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])

def projection(boxes, target, view, calib, image):
    spec, size = camera(target, int(view or 0), str(calib), str(image))
    boxes = np.asarray(boxes)
    corners = boxes[:,None,3:6] * CORNERS
    x, y = corners[...,0].copy(), corners[...,1].copy()
    co, si = np.cos(boxes[:,6,None]), np.sin(boxes[:,6,None])
    corners[...,0],corners[...,1] = co*x-si*y, si*x+co*y
    center = project_model_points(boxes[:,:3],spec,size)
    verts = project_model_points((corners+boxes[:,None,:3]).reshape(-1,3),spec,size)
    img = verts['uv'][:,0].reshape(-1,8) / size[0]
    depth = verts['depth'].reshape(-1,8)
    radius = np.linalg.norm((np.c_[boxes[:,:3],np.ones(len(boxes))] @ model_to_camera_matrix(spec).T)[:,:3],axis=1)
    return np.column_stack((center['uv'][:,0]/size[0], center['depth'], img.min(1), img.max(1),
                            depth.min(1), depth.max(1), radius,
                            center['in_image'] & np.isfinite(img).all(1) & (depth.min(1)>.1)))

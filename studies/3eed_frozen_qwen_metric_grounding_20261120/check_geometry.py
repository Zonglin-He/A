"""Coordinate, unit, upright-box and held-out-evidence positive controls. No GT."""
import numpy as np
from scipy.spatial import ConvexHull
from geometry import (camera,cook_rotation,extent_corners,to_sensor,upright,rigid_xyz,axial,
    front_offset,intersections,cell_points,sensor_pixels,pixel_rays,SIGNS,raw_to_model_points,
    project_model_points)
from paths import OUT,PRIV,read,write
from math import cos,sin

def check():
    rng=np.random.default_rng(752)
    rotation_error=0.;area_error=0.;projection_error=0.;inverse_error=0.;ray_error=0.
    for _ in range(180):
        angles=rng.uniform(-1,1,3);p,y,r=angles*np.pi
        rx=np.array([[1.,0.,0.],[0.,cos(p),-sin(p)],[0.,sin(p),cos(p)]])
        ry=np.array([[cos(y),0.,sin(y)],[0.,1.,0.],[-sin(y),0.,cos(y)]])
        rz=np.array([[cos(r),-sin(r),0.],[sin(r),cos(r),0.],[0.,0.,1.]])
        dims=rng.uniform(.2,6,3);b=np.r_[rng.uniform(-2,2,2),12.,dims,angles];rot=cook_rotation(b)
        rotation_error=max(rotation_error,abs(rot-rz@ry@rx).max())
        bb=upright(b[:3],dims,rot);v=extent_corners(b[:3],dims,rot);h=v[ConvexHull(v[:,:2]).vertices,:2]
        hm=np.roll(h,-1,axis=0)-h;axes=hm/np.linalg.norm(hm,axis=1)[:,None]
        a=np.ptp(h@axes.T,axis=0)*np.ptp(h@np.column_stack((-axes[:,1],axes[:,0])).T,axis=0)
        area_error=max(area_error,abs(min(a)-bb[3]*bb[4]))
    covered=set()
    for row in read(PRIV/'panel.json'):
        spec,size,m=camera(row)
        xyz=np.array([[0.,0.,10.],[.15,-.3,5.],[3.,-.3,15.],[-1.,.1,8.]])
        raw=rigid_xyz(xyz,np.linalg.inv(m))
        model=raw_to_model_points(raw,row['target'],row['suffix'])
        p=project_model_points(model,spec,size);pix=sensor_pixels(xyz,spec)
        projection_error=max(projection_error,abs(p['uv']-pix).max())
        inverse_error=max(inverse_error,abs(rigid_xyz(raw,m)-xyz).max())
        ray_error=max(ray_error,abs(pixel_rays(pix,spec)*xyz[:,2:]-xyz).max());covered.add(row['target'])
    b=np.array([0.,0.,10.,2.,4.,6.,0.,0.,0.])
    dist=front_offset(b,[0.,0.,1.]);np.testing.assert_allclose(dist,3.)
    hits,errors=intersections(b,np.array([[0,0,7],[0,0,8],[9,0,7]],float))
    assert hits.tolist()==[True,True,False] and np.allclose(errors[:2],[0.,1.])
    pix=np.array([[10.1,10.1],[10.2,10.2],[14.1,10.1],[18.1,10.1]])
    pts=np.column_stack((np.zeros((4,2)),[4,6,7,9]))
    fit,held=cell_points(pix,pts,np.array([0,0,32,24]))
    assert sorted(np.r_[fit[:,2],held[:,2]].tolist())==[4,7,9] and len(fit)*len(held)>0
    assert rotation_error<1e-12 and area_error<1e-10 and projection_error<1e-7 and inverse_error<1e-10 and ray_error<1e-5
    write(OUT/'coordinate_checks.json',dict(random_cuboids=180,calibrations=96,targets=sorted(covered),
        max_camera_euler_error=rotation_error,max_upright_area_vs_hull_error=area_error,
        max_projection_roundtrip_pixels=projection_error,max_rigid_roundtrip_m=inverse_error,max_ray_unprojection_m=ray_error,
        analytic_rayface_and_disjoint_sampling_pass=True))
if __name__=='__main__':check()

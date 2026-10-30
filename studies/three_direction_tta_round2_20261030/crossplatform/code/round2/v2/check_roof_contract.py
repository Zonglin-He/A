"""Analytic geometric conditions only; not dataset scores or held-out evidence."""
from pathlib import Path
import sys,json
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import roof_teacher as roof
def main():
    source=np.array([0.,0,0,4,2,2]);yaw=np.pi/6;x,y=np.meshgrid(np.linspace(-2,2,65),np.linspace(-1,1,35));xy=np.c_[x.ravel(),y.ravel()]@roof.rotation(yaw).T;p=np.c_[xy,np.ones(len(xy))];perm=np.random.default_rng(20261030).permutation(len(p));fit,rf=roof.roof_observation(p[perm[::2]],source);check,rc=roof.roof_observation(p[perm[1::2]],source);assert fit is not None and check is not None,(rf,rc);box,e,d=roof.fit_roof(fit,check,source);assert d['accepted'],d;error=roof.old.anglediff(box[6],yaw);grid_resolution_bound=np.arctan(2*(4/64)/min(source[3:5]));assert error<grid_resolution_bound,(error,grid_resolution_bound)
    rectcrop=np.all(abs(p)<=1.25*source[3:6]/2,axis=1);radius=1.25*.5*np.hypot(source[3],source[4]);cylinder=np.linalg.norm(p[:,:2],axis=1)<=radius;assert not rectcrop.all() and cylinder.all()
    x,y=np.meshgrid(np.linspace(-2,2,65),np.linspace(-.05,.05,15));strip=np.c_[x.ravel(),y.ravel(),np.ones(x.size)];stripobs,stripreason=roof.roof_observation(strip,source);assert stripobs is None,stripreason
    x,z=np.meshgrid(np.linspace(-1,1,35),np.linspace(-1,1,35));front=np.c_[np.ones(x.size)*2,x.ravel(),z.ravel()];_,frontreason=roof.roof_observation(front,source);assert frontreason=='insufficient_measured_z_normal_roof_points'
    tilt=p.copy();a=np.pi/4;rot3=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]);tilt=tilt@rot3.T;_,tiltreason=roof.roof_observation(tilt,source);assert tiltreason=='insufficient_measured_z_normal_roof_points',tiltreason
    x,y=np.meshgrid(np.linspace(-radius,radius,101),np.linspace(-radius,radius,101));xy=np.c_[x.ravel(),y.ravel()];xy=xy[np.linalg.norm(xy,axis=1)<radius];fake=np.c_[xy,np.ones(len(xy))];_,cropreason=roof.roof_observation(fake,source);assert cropreason=='observed_boundary_touches_cylinder_crop',cropreason
    result={'scope':'synthetic analytic engineering only, no dataset prediction/GTscore','complete_30degree_roof_accepted':True,'known_yaw_error_radians':error,'old_rectcrop_would_truncate':True,'circumscribed_cylinder_retains_all_roof':True,'thin_roof_strip_rejected':stripreason,'vertical_frontface_rejected':frontreason,'tilted45degree_plane_unsupported':tiltreason,'crop_created_boundary_rejected':cropreason,'coverage':d};(R/'ROOF_ENGINEERING_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='coverage'},indent=2))
if __name__=='__main__':main()

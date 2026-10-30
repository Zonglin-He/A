"""Deterministic, GT-free geometric orientation baselines on one shared crop.
Box layout: centerXYZ, extentXYZ, yaw; local x-axis is extent[0].
MIT-derived common-implementation objectives are credited in PROVENANCE.md.
No model, metadata, target score, candidate selection or GPU imports.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,importlib.util
import numpy as np
from scipy.optimize import minimize

@dataclass(frozen=True)
class Config:
    min_points:int=8
    crop_tolerance_m:float=1e-7
    relative_rank_tolerance:float=1e-12
    theta_step_deg:float=1.
    closeness_floor_m:float=.01
    variance_ddof:int=0
    optimizer_maxiter:int=20
    optimizer_ftol:float=1e-9
    optimizer_maxls:int=15

DEFAULT=Config()
METHODS=('PCA_standard_axisaware','minimum_area_exact','Lshape_area','Lshape_closeness','Lshape_variance','surface_energy_orientationonly_optimization')

def wrap_pi(theta):return float((float(theta)+np.pi/2)%np.pi-np.pi/2)
def angle_difference(a,b):return abs(wrap_pi(float(a)-float(b)))
def rotation(theta):
    theta=float(theta);c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,-s],[s,c]],dtype=np.float64)

def source7(source):
    x=np.asarray(source)
    if x.shape not in [(6,),(7,)]:raise ValueError('source box must have6 or7 entries')
    dtype=x.dtype if np.issubdtype(x.dtype,np.floating) else np.float64
    y=np.empty(7,dtype=dtype);y[:6]=x[:6];y[6]=x[6] if len(x)==7 else 0
    if not np.isfinite(y).all():raise ValueError('nonfinite source box')
    return y

def unique_xyz(points):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]<3:raise ValueError('expected Nx3+ observed point array')
    if not np.isfinite(p[:,:3]).all():raise ValueError('nonfinite observed point')
    return np.unique(p[:,:3],axis=0)

def crop_source_aabb(points,source,config=DEFAULT):
    """One fixed AABB crop, not recropped after a method changes yaw."""
    raw=np.asarray(points)
    if raw.ndim!=2 or raw.shape[1]<3 or not np.isfinite(raw[:,:3]).all():raise ValueError('invalid observed pointcloud')
    p=np.unique(raw[:,:3],axis=0);s=source7(source)
    if np.any(s[3:6]<=0):return p[:0].astype(np.float64),{'reason':'invalid_nonpositive_source_size','unique_points':len(p),'crop_points':0}
    # Exact legacyPCA membership arithmetic: select in source/cache dtype first.
    # Only after one common selection cast tofloat64 for geometry/rotation.
    mask=np.all(np.abs(p-s[:3])<=s[3:6]/2+config.crop_tolerance_m,axis=1);crop=p[mask].astype(np.float64)
    digest=hashlib.sha256(np.ascontiguousarray(crop,dtype='<f8').tobytes()).hexdigest()
    return crop,{'reason':'source_AABB','unique_points':len(p),'crop_points':len(crop),'crop_sha256_float64_xyz':digest,'source_yaw_ignored_for_AABB':True,'membership_arithmetic':'native point/box dtype, identicallegacyPCA, thenfloat64 geometry'}

def support(crop,config=DEFAULT):
    p=unique_xyz(crop)
    if len(p)<config.min_points:return p,'fewer_than_shared_min_points'
    xy=p[:,:2]-p[:,:2].mean(0);singular=np.linalg.svd(xy,compute_uv=False)
    if len(singular)<2 or singular[0]<=config.relative_rank_tolerance:return p,'coincident_XY'
    if singular[1]<=singular[0]*config.relative_rank_tolerance:return p,'collinear_XY'
    return p,None

def convex_hull_xy(xy):
    """Andrew monotone chain, exact all-hull-edge candidates (float64)."""
    points=np.unique(np.asarray(xy,dtype=np.float64),axis=0)
    if len(points)<=1:return points
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    low=[];high=[]
    for p in points:
        while len(low)>=2 and cross(low[-2],low[-1],p)<=0:low.pop()
        low.append(p)
    for p in points[::-1]:
        while len(high)>=2 and cross(high[-2],high[-1],p)<=0:high.pop()
        high.append(p)
    return np.asarray(low[:-1]+high[:-1])

def rectangle_at(xy,theta):
    projected=np.asarray(xy,dtype=np.float64)@rotation(theta)
    lo=projected.min(0);hi=projected.max(0)
    return lo,hi

def minimum_area_rectangle(xy):
    """Global min enclosing rectangle from every nonzero convex hull edge.
    This is finite exact edge enumeration, not an angle-grid approximation.
    """
    hull=convex_hull_xy(xy)
    if len(hull)<3:raise ValueError('degenerate convex hull')
    edges=np.roll(hull,-1,axis=0)-hull;angles=np.mod(np.arctan2(edges[:,1],edges[:,0]),np.pi/2)
    candidates=[]
    for theta in np.sort(angles):
        lo,hi=rectangle_at(hull,float(theta));candidates.append((float(np.prod(hi-lo)),float(theta),lo,hi))
    area,theta,lo,hi=min(candidates,key=lambda x:(x[0],x[1]))
    return theta,lo,hi,{'criterion':'min enclosing area; all convex-hull edge orientations','area_m2':area,'hull_vertices':len(hull),'angles_evaluated':len(angles),'grid_search':False}

def lshape_score(c1,c2,criterion,variant='paper_literal',config=DEFAULT):
    """Published paper Alg3/4/5 vs named MIT common implementation.
    paper D selects ONE entire boundary-distance vector by L2, not eachpoint's
    closest of both parallel boundaries. Population variance and emptygroup0
    are explicit conventions where the original pseudocode is underspecified.
    """
    c1=np.asarray(c1,dtype=np.float64);c2=np.asarray(c2,dtype=np.float64)
    if criterion=='area':return -float(np.ptp(c1)*np.ptp(c2))
    up1,down1=c1.max()-c1,c1-c1.min();up2,down2=c2.max()-c2,c2-c2.min()
    if variant=='paper_literal':
        d1=up1 if np.dot(up1,up1)<=np.dot(down1,down1) else down1
        d2=up2 if np.dot(up2,up2)<=np.dot(down2,down2) else down2
    elif variant=='pythonrobotics':d1=np.minimum(up1,down1);d2=np.minimum(up2,down2)
    else:raise ValueError('unknown Lshape objective variant')
    if criterion=='closeness':return float(np.sum(1/np.maximum(np.minimum(d1,d2),config.closeness_floor_m)))
    if criterion=='variance':
        e1=d1[d1<d2];e2=d2[d2<d1] if variant=='paper_literal' else d2[d1>=d2]
        var=lambda e:float(np.var(e,ddof=config.variance_ddof)) if len(e)>config.variance_ddof else 0.
        return -(var(e1)+var(e2))
    raise ValueError('unknown Lshape criterion')

def lshape_rectangle(xy,criterion,variant='paper_literal',config=DEFAULT):
    if variant=='paper_literal':angles=np.deg2rad(np.arange(0.,90.,config.theta_step_deg))
    elif variant=='pythonrobotics':
        step=np.deg2rad(config.theta_step_deg);angles=np.arange(0.,np.pi/2-step,step)
    else:raise ValueError('unknown Lshape variant')
    xy=np.asarray(xy,dtype=np.float64);scores=[]
    for theta in angles:
        c=xy@rotation(float(theta));scores.append(lshape_score(c[:,0],c[:,1],criterion,variant,config))
    j=int(np.argmax(scores));theta=float(angles[j]);lo,hi=rectangle_at(xy,theta)
    return theta,lo,hi,{'criterion':criterion,'objective_variant':variant,'objective_score_maximized':float(scores[j]),'theta_step_degrees':config.theta_step_deg,'scan_first_degrees':float(np.rad2deg(angles[0])),'scan_last_degrees':float(np.rad2deg(angles[-1])),'angles_evaluated':len(angles),'closeness_floor_m':config.closeness_floor_m,'variance_ddof':config.variance_ddof,'empty_or_singleton_variance':0.,'segmentation':'not rerun; fixed sourceAABB crop is the common cluster'}

def axis_correspondence(theta,span,source,pca_axis=False):
    """Keep source L/W order. Largest source extent corresponds to fitted longaxis.
    Square ambiguity chooses the equivalent axis closest to source yaw. No size swap.
    """
    s=source7(source);theta=float(theta);span=np.asarray(span,dtype=np.float64)
    square=abs(float(s[3])-float(s[4]))<=1e-12*max(float(s[3]),float(s[4]),1.)
    fitted_square=(not pca_axis) and abs(span[0]-span[1])<=1e-12*max(float(span.max()),1.)
    if square or fitted_square:
        options=[wrap_pi(theta),wrap_pi(theta+np.pi/2)];yaw=min(options,key=lambda a:(angle_difference(a,float(s[6])),a));rule='square_ambiguous_axis_closest_source'
    else:
        long=theta if pca_axis or span[0]>span[1] else theta+np.pi/2
        yaw=wrap_pi(long if s[3]>s[4] else long+np.pi/2);rule='estimated_long_axis_to_larger_source_extent_without_swapping_sizes'
    return yaw,{'axis_rule':rule,'source_x_extent_at_least_y':bool(s[3]>=s[4]),'estimated_span_xy':span.tolist(),'yaw_modulo':'pi; [-pi/2,pi/2)'}

def fit_rectangle(crop,method,config=DEFAULT):
    xy=crop[:,:2]
    if method=='PCA_standard_axisaware':
        cov=np.cov(xy,rowvar=False);values,vectors=np.linalg.eigh(cov)
        if values[-1]<=1e-12 or (values[-1]-values[0])<=1e-12*max(values[-1],1.):raise ValueError('isotropic_or_zero_PCA_covariance')
        direction=vectors[:,-1];theta=wrap_pi(np.arctan2(direction[1],direction[0]));lo,hi=rectangle_at(xy,theta)
        return theta,lo,hi,{'criterion':'largest covariance eigenvector; L2 variance','eigenvalues':values.tolist(),'principal_axis_is_long':True}
    if method=='minimum_area_exact':return minimum_area_rectangle(xy)
    if method.startswith('Lshape_'):
        return lshape_rectangle(xy,method.split('_',1)[1],config=config)
    if method.startswith('pythonrobotics_'):
        return lshape_rectangle(xy,method.split('_',1)[1],variant='pythonrobotics',config=config)
    raise ValueError('unknown rectangle method '+method)

def surface_energy_opt(crop,source,config=DEFAULT):
    """Yaw-only, same source center/size/crop; no old acceptance/reselection gates."""
    path=Path(__file__).resolve().parents[2]/'round2/surface_teacher.py'
    spec=importlib.util.spec_from_file_location('p3_readonly_surface_energy',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observation=module.observations(crop);s=source7(source).astype(np.float64)
    _,v=np.linalg.eigh(np.cov(crop[:,:2],rowvar=False));principal=float(np.arctan2(v[1,-1],v[0,-1]));mapped,_=axis_correspondence(principal,np.array([1.,0.]),s,pca_axis=True)
    starts=[0.,wrap_pi(mapped),wrap_pi(mapped+np.pi/2)];runs=[];choices=[]
    def objective(x):
        b=s.copy();b[6]=float(x[0]);return module.energy(observation,b)
    for index,start in enumerate(starts):
        initial=objective([start]);res=minimize(objective,np.array([start]),method='L-BFGS-B',bounds=[(-np.pi/2,np.pi/2)],options={'maxiter':config.optimizer_maxiter,'ftol':config.optimizer_ftol,'maxls':config.optimizer_maxls})
        final=objective(res.x);choices.extend([(initial,wrap_pi(start),index,'initial'),(final,wrap_pi(res.x[0]),index,'optimized')]);runs.append({'start_yaw':start,'initial_energy':initial,'final_energy':final,'optimized_yaw':wrap_pi(res.x[0]),'success':bool(res.success),'iterations':int(res.nit)})
    energy,yaw,index,kind=min(choices,key=lambda x:(x[0],angle_difference(x[1],float(s[6])),x[2],x[3]))
    return yaw,{'criterion':'unchanged round2.surface_teacher.energy on unchanged observations','surface_code_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'starts':starts,'optimizer':'1D bounded L-BFGS-B','bounds_radians':[-np.pi/2,np.pi/2],'maxiter':config.optimizer_maxiter,'ftol':config.optimizer_ftol,'maxls':config.optimizer_maxls,'selected_energy':energy,'selected_start':index,'selected_kind':kind,'runs':runs,'original_acceptance_rules_used':False,'query_or_center_or_size_optimized':False}

def orientation_only_from_crop(crop_xyz,source,method,config=DEFAULT):
    allowed=set(METHODS)|{'pythonrobotics_area','pythonrobotics_closeness','pythonrobotics_variance'}
    if method not in allowed:raise ValueError('unknown method '+method)
    s=source7(source);out=s.copy();p,reason=support(crop_xyz,config);diag={'method':method,'capability_group':'orientation_only','shared_crop_points':len(p),'source_center_size_bitexact':True,'config':asdict(config)}
    if np.any(s[3:6]<=0):reason='invalid_nonpositive_source_size'
    if reason:diag.update(reason=reason,updated=False);return out,diag
    try:
        if method=='surface_energy_orientationonly_optimization':yaw,d=surface_energy_opt(p,s,config)
        else:
            theta,lo,hi,d=fit_rectangle(p,method,config);yaw,axis=axis_correspondence(theta,hi-lo,s,pca_axis=method=='PCA_standard_axisaware');d.update(axis)
        out[6]=yaw;diag.update(d);diag.update(reason='estimated',updated=bool(out[6]!=s[6]),output_yaw=float(out[6]))
    except ValueError as e:diag.update(reason=str(e),updated=False)
    assert np.array_equal(out[:6],s[:6]);return out,diag

def compare_orientation(points,source,selected_query=None,methods=METHODS,config=DEFAULT):
    crop,cropdiag=crop_source_aabb(points,source,config);outputs={};diagnostics={}
    for method in methods:
        outputs[method],diagnostics[method]=orientation_only_from_crop(crop,source,method,config)
        diagnostics[method]['crop_sha256_float64_xyz']=cropdiag.get('crop_sha256_float64_xyz')
    return outputs,{'selected_query_unchanged':selected_query,'crop':cropdiag,'methods':diagnostics}

def full_refit_from_crop(crop_xyz,source,method,refit_z=True,config=DEFAULT):
    """SEPARATE diagnostic ability: refit center/extent. Never orientation-only arm."""
    s=source7(source);out=s.copy();p,reason=support(crop_xyz,config)
    if reason:return out,{'capability_group':'classical_full_refit','reason':reason,'updated':False}
    theta,lo,hi,d=fit_rectangle(p,method,config);yaw,axis=axis_correspondence(theta,hi-lo,s,pca_axis=method=='PCA_standard_axisaware');lo,hi=rectangle_at(p[:,:2],yaw);center=((lo+hi)/2)@rotation(yaw).T;extent=hi-lo
    if np.any(extent<=0) or (refit_z and np.ptp(p[:,2])<=0):return out,{'capability_group':'classical_full_refit','reason':'degenerate_fitted_extent','updated':False}
    out[:2]=center;out[3:5]=extent;out[6]=yaw
    if refit_z:out[2]=(p[:,2].min()+p[:,2].max())/2;out[5]=np.ptp(p[:,2])
    return out,{'capability_group':'classical_full_refit' if refit_z else 'footprint_full_refit_Z_unchanged','updated':True,'refit_z':refit_z,'center_or_size_changed':not np.array_equal(out[:6],s[:6]),**d,**axis}

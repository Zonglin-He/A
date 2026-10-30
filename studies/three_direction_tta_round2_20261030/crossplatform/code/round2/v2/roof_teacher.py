"""One fixed roof-supported extension; old accepted teachers take priority."""
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import surface_teacher as old

CONFIG={'cylinder_radius_factor':1.25,'normal_abs_z_min':.85,'normal_planarity_min':.1,
        'roof_z_above_source_center_fraction':.2,'roof_min_points_per_half':12,
        'roof_height_q90_q10_max_fraction':.15,'hull_area_min_fraction':.2,
        'each_axis_span_min_fraction':.5,'side_distance_max_fraction':.1,
        'side_min_points':3,'side_tangent_span_min_fraction':.5,
        'crop_boundary_guard_fraction':.02,'outlier_probability':.2,
        'temperature':.02,'center_prior':.5,'center_bound_fraction':.15,
        'max_iterations':20,'posterior_min':.65,'fit_check_angle_max':np.pi/12}

def rotation(theta):
    theta=float(theta);c,s=np.cos(theta),np.sin(theta);return np.array([[c,-s],[s,c]])
def roof_observation(points,source):
    if len(points)<CONFIG['roof_min_points_per_half']:return None,'insufficient_cylinder_points'
    observation=old.observations(points)
    mask=(abs(observation['normal'][:,2])>=CONFIG['normal_abs_z_min'])&(observation['planarity']>=CONFIG['normal_planarity_min'])&(points[:,2]>=source[2]+CONFIG['roof_z_above_source_center_fraction']*source[5])
    roof=points[mask]
    if len(roof)<CONFIG['roof_min_points_per_half']:return None,'insufficient_measured_z_normal_roof_points'
    spread=float(np.percentile(roof[:,2],90)-np.percentile(roof[:,2],10))
    if spread>CONFIG['roof_height_q90_q10_max_fraction']*source[5]:return None,'roof_height_not_single_plane'
    try:hull=ConvexHull(roof[:,:2]);vertices=roof[hull.vertices,:2]
    except QhullError:return None,'degenerate_2D_roof_hull'
    if hull.volume<CONFIG['hull_area_min_fraction']*source[3]*source[4]:return None,'insufficient_2D_roof_area'
    radius=CONFIG['cylinder_radius_factor']*.5*np.hypot(source[3],source[4])
    if np.any(np.linalg.norm(vertices-source[:2],axis=1)>=radius-CONFIG['crop_boundary_guard_fraction']*min(source[3:5])):return None,'observed_boundary_touches_cylinder_crop'
    # Count measured points close to the observed hull, not arbitrary interior
    # roof points that could falsely "support" an inscribed wrong rectangle.
    signed=roof[:,:2]@hull.equations[:,:2].T+hull.equations[:,2]
    boundary=roof[signed.max(1)>=-CONFIG['side_distance_max_fraction']*min(source[3:5])]
    # Five-point Gauss quadrature integrates observed hull edges by arc length.
    # A single midpoint aliases a rotated rectangle to a wrong inscribed box;
    # counting nearly-collinear tiny hull segments equally also changes measure.
    nodes,weights=np.polynomial.legendre.leggauss(5);t=(nodes+1)/2
    edges=np.roll(vertices,-1,axis=0)-vertices;length=np.linalg.norm(edges,axis=1)
    quadrature=(vertices[:,None,:]+edges[:,None,:]*t[None,:,None]).reshape(-1,2)
    quadrature_weights=(length[:,None]*weights[None,:]/2/length.sum()).ravel()
    return {'points':roof,'boundary_points':boundary,'midpoints':quadrature,'edge_weights':quadrature_weights,'hull_area':float(hull.volume),'normal_qualifying_points':len(roof),'height_spread':spread},'eligible_observation'

def boundary_energy(observation,box):
    local=(observation['midpoints']-box[:2])@rotation(box[6]);d=abs(local)-box[3:5]/2
    distance=np.linalg.norm(np.maximum(d,0),axis=1)+np.minimum(d.max(1),0)
    cost=(distance/min(box[3:5]))**2;mix=(1-CONFIG['outlier_probability'])*np.exp(-np.minimum(cost,100)/CONFIG['temperature'])+CONFIG['outlier_probability']*np.exp(-1.)
    reference=1-CONFIG['outlier_probability']+CONFIG['outlier_probability']*np.exp(-1.)
    perpoint=-CONFIG['temperature']*np.log(np.maximum(mix,1e-30))+CONFIG['temperature']*np.log(reference)
    return float(np.sum(perpoint*observation['edge_weights']))

def boundary_coverage(observation,box):
    local=(observation['boundary_points'][:,:2]-box[:2])@rotation(box[6]);spans=np.ptp(local,axis=0);sides=[]
    for axis in [0,1]:
        for sign in [-1,1]:
            mask=abs(local[:,axis]-sign*box[3+axis]/2)<=CONFIG['side_distance_max_fraction']*box[3+axis]
            mask&=abs(local[:,1-axis])<=box[4-axis]/2+CONFIG['side_distance_max_fraction']*box[4-axis]
            n=int(mask.sum());span=float(np.ptp(local[mask,1-axis])/box[4-axis]) if n>1 else 0.
            sides.append({'axis':axis,'sign':sign,'points':n,'tangent_span_fraction':span,'supported':n>=CONFIG['side_min_points'] and span>=CONFIG['side_tangent_span_min_fraction']})
    supported=bool(all(x['supported'] for x in sides) and np.all(spans>=CONFIG['each_axis_span_min_fraction']*box[3:5]))
    return {'four_sides_supported':supported,'sides':sides,'axis_span_fractions':(spans/box[3:5]).tolist(),'hull_area_fraction':observation['hull_area']/(box[3]*box[4])}

def fit_roof(fit,check,source):
    _,v=np.linalg.eigh(np.cov(fit['points'][:,:2],rowvar=False));pca=np.arctan2(v[1,-1],v[0,-1])
    def box(x):return np.r_[source[:2]+x[:2]*source[3:5],source[2],source[3:6],old.wrap(x[2])]
    def energy(obs,x):return boundary_energy(obs,box(x))+CONFIG['center_prior']*np.sum(x[:2]**2)
    modes=[]
    for theta in [0.,old.wrap(pca),old.wrap(pca+np.pi/2)]:
        res=minimize(lambda x:energy(fit,x),np.r_[0.,0.,theta],method='L-BFGS-B',bounds=[(-CONFIG['center_bound_fraction'],CONFIG['center_bound_fraction'])]*2+[(-np.pi/2,np.pi/2)],options={'maxiter':CONFIG['max_iterations'],'ftol':1e-9,'maxls':15})
        b=box(res.x);rec={'box':b,'fit_energy':energy(fit,res.x),'check_energy':energy(check,res.x),'optimizer_success':bool(res.success),'iterations':int(res.nit)}
        matches=[i for i,m in enumerate(modes) if old.anglediff(b[6],m['box'][6])<old.CONFIG['mode_merge_radians'] and np.max(abs((b[:3]-m['box'][:3])/source[3:6]))<old.CONFIG['mode_merge_center_fraction']]
        if matches:
            i=matches[0]
            if rec['fit_energy']<modes[i]['fit_energy']:modes[i]=rec
        else:modes.append(rec)
    e=np.array([(m['fit_energy']+m['check_energy'])/2 for m in modes]);cover=[(boundary_coverage(fit,m['box']),boundary_coverage(check,m['box'])) for m in modes];valid=np.array([a['four_sides_supported'] and b['four_sides_supported'] for a,b in cover]);prob=np.zeros(len(modes))
    if valid.any():prob[valid]=np.exp(-(e[valid]-e[valid].min())/CONFIG['temperature']);prob/=prob.sum();best=int(np.where(valid,e,np.inf).argmin());eligible=[m for m,v in zip(modes,valid) if v]
    else:best=int(e.argmin());eligible=modes
    chosen=modes[best];cf,cc=cover[best];agree=old.anglediff(min(eligible,key=lambda x:x['fit_energy'])['box'][6],min(eligible,key=lambda x:x['check_energy'])['box'][6])<=CONFIG['fit_check_angle_max']
    accepted=bool(valid.any() and prob[best]>=CONFIG['posterior_min'] and agree)
    return chosen['box'],float(e[best]),{'branch':'measured_z_normal_roof_boundary','accepted':accepted,'fit_coverage':cf,'check_coverage':cc,'fit_check_mode_agreement':bool(agree),'mode_entropy':float(-np.sum(prob*np.log(prob+1e-30))),'best_mode_probability':float(prob[best]),'retained_modes':len(modes),'coverage_qualified_modes':int(valid.sum()),'modes':[{'yaw':float(m['box'][6]),'fit_energy':float(m['fit_energy']),'check_energy':float(m['check_energy']),'posterior':float(p),'coverage_qualified':bool(v),'optimizer_success':m['optimizer_success'],'iterations':m['iterations']} for m,p,v in zip(modes,prob,valid)]}

def build_teacher(points,boxes,scores,seed=0):
    result=old.build_teacher(points,boxes,scores,seed);points=np.unique(np.asarray(points,float),axis=0)
    for i,j in enumerate(result['query_indices']):
        result['diagnostics'][i]['old_vertical_accepted']=bool(result['accepted'][i])
        if result['accepted'][i]:result['diagnostics'][i]['branch']='original_vertical_priority';continue
        src=np.asarray(boxes[j],float)
        if np.any(src[3:6]<=0):continue
        radius=CONFIG['cylinder_radius_factor']*.5*np.hypot(src[3],src[4]);mask=(np.linalg.norm(points[:,:2]-src[:2],axis=1)<=radius)&(abs(points[:,2]-src[2])<=CONFIG['cylinder_radius_factor']*src[5]/2);crop=points[mask]
        perm=np.random.default_rng(seed+int(j)).permutation(len(crop));fit,reasonfit=roof_observation(crop[perm[::2]],src);check,reasoncheck=roof_observation(crop[perm[1::2]],src)
        roofdiag={'branch':'roof_unsupported','accepted':False,'fit_observation':reasonfit,'check_observation':reasoncheck,'crop_points':len(crop)}
        if fit is not None and check is not None:
            target,e,roofdiag=fit_roof(fit,check,src);roofdiag.update(crop_points=len(crop),fit_roof_points=fit['normal_qualifying_points'],check_roof_points=check['normal_qualifying_points'])
            if roofdiag['accepted']:result['accepted'][i]=True;result['target_boxes7'][i]=target;result['logscore_bonus'][i]=np.clip((CONFIG['temperature']-e)/CONFIG['temperature'],-1,1)
        result['diagnostics'][i]['roof']=roofdiag
        if roofdiag['accepted']:result['diagnostics'][i].update(accepted=True,branch='roof_extension')
    prior=result['prior'];q=prior*np.exp(result['logscore_bonus']);q/=max(q.sum(),1e-30);winner=int(q.argmax());result.update(posterior=q,direct_box7=result['target_boxes7'][winner].copy(),direct_query=int(result['query_indices'][winner]));return result

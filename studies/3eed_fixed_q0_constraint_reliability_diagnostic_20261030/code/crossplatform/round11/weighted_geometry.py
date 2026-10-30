"""R11 reliability-weighted copy of frozen R10 geometry solver.
Pure CPU/data arrays. No file/model/GT access, proposal fusion, reselect or fitter.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,json,sys,itertools
import numpy as np
from scipy.spatial import cKDTree,ConvexHull,QhullError
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'round7'))
from camera_projection import project_model_points,model_to_camera_matrix
ARM_NAMES=('source','full_rgb','no_rgb_coordinate','rgb_gated_pointmask_ablation')

@dataclass(frozen=True)
class GeometryConfig:
    normal_knn:int=16
    normal_axis_alignment_min:float=.9
    planarity_min:float=.3
    min_plane_points:int=8
    min_plane_unique_XYZ:int=8
    plane_MAD_max_m:float=.04
    tangent_span_fraction:float=.25
    paired_min_points:int=12
    paired_plane_MAD_max_m:float=.025
    paired_tangent_span_fraction:float=.4
    paired_separation_ratio_min:float=.8
    paired_separation_ratio_max:float=1.2
    crop_fraction:float=.25
    crop_margin_add_m:float=.1
    crop_margin_cap_m:float=.75
    face_band_fraction:float=.30
    face_band_add_m:float=.05
    face_band_cap_m:float=.60
    camera_outside_slab_margin_m:float=.05
    min_corner_depth_m:float=.1
    center_bound_fraction:float=.25
    center_bound_cap_m:float=.5
    size_log_bound:float=float(np.log(1.2))
    center_prior_std_fraction:float=.15
    center_prior_std_min_m:float=.05
    center_prior_std_max_m:float=.35
    log_size_prior_std:float=.1
    prior_weight:float=1.
    geometry_weight:float=1.
    rgb_weight:float=1.
    plane_sigma_floor_m:float=.03
    huber_delta:float=1.
    RGB_pixel_sigma_min:float=3.
    RGB_pixel_sigma_bbox_fraction:float=.05
    max_mask_boundary_samples:int=128
    solver_max_iterations:int=32
    solver_ftol:float=1e-10
    solver_gtol:float=1e-6
    solver_normalized_finite_difference_eps:float=1e-5
DEFAULT_CONFIG=GeometryConfig()


def canonical_observations(points,raw_indices):
    p=np.asarray(points,np.float32);ids=np.asarray(raw_indices)
    assert p.ndim==2 and p.shape[1]==3 and ids.shape==(len(p),) and np.issubdtype(ids.dtype,np.integer) and np.isfinite(p).all()
    uid,first=np.unique(ids,return_index=True)
    # Same physical return cannot acquire two different model coordinates.
    for rawid in uid[np.unique(ids,return_counts=True)[1]>1]:
        vals=p[ids==rawid];assert np.all(vals==vals[0]),'Inconsistent repeated raw index'
    rp=p[first];rid=ids[first].astype(np.int64)
    xyz,keep=np.unique(rp,axis=0,return_index=True)
    return xyz.astype(np.float64),rid[keep],{'input_rows':len(p),'unique_raw_indices':len(uid),'unique_XYZ':len(xyz),'repeated_array_rows':len(p)-len(uid),'distinct_rawindices_same_XYZ':len(uid)-len(xyz)}


def observation_sha(points,rawids,indices,assignments=None):
    take=np.asarray(indices,np.int64);order=np.argsort(rawids[take],kind='stable');take=take[order]
    h=hashlib.sha256();h.update(np.asarray(rawids[take],np.int64).tobytes());h.update(np.asarray(points[take],np.float32).tobytes())
    if assignments is not None:h.update(json.dumps(assignments,sort_keys=True,separators=(',',':')).encode())
    return h.hexdigest()


def box_corners(box):
    b=np.asarray(box,float);sign=np.array(list(itertools.product((-1.,1.),repeat=3)))
    return b[:3]+sign*b[3:]/2


def unique_visible_ids(points,projected,image_wh):
    u=projected['uv'];d=projected['depth'];W,H=image_wh
    valid=np.isfinite(u).all(1)&np.isfinite(d)&(d>.1)&(u[:,0]>=0)&(u[:,0]<W)&(u[:,1]>=0)&(u[:,1]<H)
    ids=np.flatnonzero(valid);pix=np.zeros_like(u,dtype=np.int64);pix[valid]=np.floor(u[valid]).astype(np.int64)
    if not len(ids):return ids,pix
    keys=pix[ids,1]*W+pix[ids,0];p=points[ids];order=np.lexsort((p[:,2],p[:,1],p[:,0],d[ids],keys));ids=ids[order];keys=keys[order]
    return ids[np.r_[True,keys[1:]!=keys[:-1]]],pix


def normals_from_fit_cloud(points,query_ids,cfg):
    if len(points)<cfg.normal_knn:return np.zeros((len(query_ids),3)),np.zeros(len(query_ids))
    _,ix=cKDTree(points).query(points[query_ids],k=cfg.normal_knn)
    neighbors=points[ix];centered=neighbors-neighbors.mean(1,keepdims=True)
    cov=np.einsum('nki,nkj->nij',centered,centered)/cfg.normal_knn
    values,vectors=np.linalg.eigh(cov);values=np.maximum(values,0)
    planarity=(values[:,1]-values[:,0])/np.maximum(values[:,2],1e-12)
    return vectors[:,:,0],planarity


def collect_planes(points,support,normal_by_id,planarity_by_id,source,camera_center,cfg):
    planes={};diagnostics={};near={};tangent_axes={j:[k for k in range(3) if k!=j] for j in range(3)}
    for j in range(3):
        rel=camera_center[j]-source[j]
        near[j]=int(np.sign(rel)) if abs(rel)>source[3+j]/2+cfg.camera_outside_slab_margin_m else 0
        band=min(cfg.face_band_fraction*source[3+j]+cfg.face_band_add_m,cfg.face_band_cap_m)
        for sign in (-1,1):
            face=source[j]+sign*source[3+j]/2
            ids=np.asarray([i for i in support if abs(normal_by_id[i,j])>=cfg.normal_axis_alignment_min and planarity_by_id[i]>=cfg.planarity_min and abs(points[i,j]-face)<=band and sign*(points[i,j]-source[j])>0],np.int64)
            entry={'axis':j,'sign':sign,'count':len(ids),'camera_facing_at_source':sign==near[j],'accepted_plane':False}
            if len(ids):
                med=float(np.median(points[ids,j]));mad=float(np.median(np.abs(points[ids,j]-med)));span=np.ptp(points[ids][:,tangent_axes[j]],axis=0)
                entry.update(median=med,MAD_m=mad,tangent_span_m=span.tolist(),unique_XYZ=len(np.unique(points[ids],axis=0)))
                valid=len(ids)>=cfg.min_plane_points and entry['unique_XYZ']>=cfg.min_plane_unique_XYZ and mad<=cfg.plane_MAD_max_m and np.all(span>=cfg.tangent_span_fraction*source[3+np.array(tangent_axes[j])])
                entry['accepted_plane']=bool(valid);planes[(j,sign)]={'ids':ids,'median':med,'MAD':mad,'span':span,'valid':bool(valid)}
            else:planes[(j,sign)]={'ids':ids,'valid':False}
            diagnostics[str(j)+':'+str(sign)]=entry
    return planes,near,diagnostics


def common_acceptance(planes,near,source,cfg):
    mask=np.zeros(6,bool);why=[]
    for j in range(3):
        neg,pos=planes[(j,-1)],planes[(j,1)]
        paired=neg['valid'] and pos['valid']
        if paired:
            sep=pos['median']-neg['median'];t=[k for k in range(3) if k!=j]
            paired=cfg.paired_separation_ratio_min*source[j+3]<=sep<=cfg.paired_separation_ratio_max*source[j+3] and all(len(x['ids'])>=cfg.paired_min_points and x['MAD']<=cfg.paired_plane_MAD_max_m and np.all(x['span']>=cfg.paired_tangent_span_fraction*source[3+np.array(t)]) for x in (neg,pos))
        near_valid=near[j]!=0 and planes[(j,near[j])]['valid']
        mask[j]=near_valid or paired;mask[j+3]=paired
        why.append('two_opposing_observed_planes_plus_prior' if paired else 'point_plane_plus_source_size_prior' if near_valid else 'source_prior_no_visible_face_evidence')
    return mask,why


def boundary_samples(mask,cfg):
    boundary=mask&~binary_erosion(mask,structure=np.ones((3,3)),border_value=0);yx=np.argwhere(boundary)
    if not len(yx):return np.empty((0,2))
    ix=np.linspace(0,len(yx)-1,min(len(yx),cfg.max_mask_boundary_samples),dtype=np.int64)
    return yx[ix][:,::-1].astype(float)+.5


def huber(values,delta):
    a=np.abs(values);return np.where(a<=delta,.5*a*a,delta*(a-.5*delta))


def generate_weighted_geometry_targets(*,points_fit,raw_indices,source_box6,q0,camera_spec,rgb_mask2d,segmentation_valid,face_weights,RGB_weight,config=DEFAULT_CONFIG):
    """Only source prior+actual observed planes+one-sided mask containment.
    Main two variants share exact support, plane assignments and acceptance mask.
    Pointmask ablation intentionally changes support but inherits RGB-derived gate.
    """
    assert set(face_weights)=={f'{j}:{sgn}' for j in range(3) for sgn in (-1,1)}
    face_weights={k:float(v) for k,v in face_weights.items()}
    RGB_weight=float(RGB_weight)
    assert all(np.isfinite(v) and 0<=v<=1 for v in [*face_weights.values(),RGB_weight])
    cfg=config;source=np.asarray(source_box6,np.float64);mask=np.asarray(rgb_mask2d)
    assert source.shape==(6,) and np.isfinite(source).all() and mask.ndim==2 and mask.dtype==bool and isinstance(q0,(int,np.integer)) and isinstance(segmentation_valid,(bool,np.bool_))
    H,W=mask.shape;points,rawids,dedup=canonical_observations(points_fit,raw_indices)
    result={'arm_names':np.array(ARM_NAMES),'arm_query_index':np.full(4,int(q0),np.int64),'arm_boxes_raw6':np.tile(source,(4,1)),'pre_constraint_geometry6':np.tile(source,(4,1)),'acceptance_mask6':np.zeros((4,6),bool),'point_support_sha256':np.array(['0'*64]*4),'provenance':np.full((4,6),'source_prior',dtype='U160'),'rgb_mask_sha256':hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),'mask_sha256_by_arm':np.array([hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()]*4),'support_ids_by_arm':[np.array([],np.int64) for _ in range(4)],'arm_status':['source']+['fallback']*3,'diagnostics':{'config':asdict(cfg),'dedup':dedup,'q0_fixed_no_other_proposals_or_GT':True,'partial_point_mean_or_extent_never_used_as_full_box':True,'pointmask_ablation_is_RGB_gated_not_deployment_PC_only':True}}
    def fallback(reason):result['diagnostics']['global_gate_reason']=reason;result['provenance'][1:]=('fallback:'+reason);return result
    if not segmentation_valid or not mask.any():return fallback('invalid_or_empty_RGB_support')
    if np.any(source[3:]<=0):return fallback('nonpositive_source_size')
    if len(points)<cfg.normal_knn:return fallback('insufficient_unique_fit_observations')
    try:
        projected=project_model_points(points,camera_spec,(W,H));corner_projection=project_model_points(box_corners(source),camera_spec,(W,H))
        camera_center=np.linalg.inv(model_to_camera_matrix(camera_spec))[:3,3]
    except (ValueError,np.linalg.LinAlgError) as exc:return fallback('camera_contract:'+str(exc))
    if np.any(corner_projection['depth']<=cfg.min_corner_depth_m) or not np.isfinite(corner_projection['uv']).all():return fallback('source_box_projection_not_fully_in_front')
    visible,pixels=unique_visible_ids(points,projected,(W,H))
    margin=np.minimum(cfg.crop_fraction*source[3:]+cfg.crop_margin_add_m,cfg.crop_margin_cap_m)
    geom=visible[(np.abs(points[visible]-source[:3])<=source[3:]/2+margin).all(1)]
    rgb=geom[mask[pixels[geom,1],pixels[geom,0]]]
    normals,planarity=normals_from_fit_cloud(points,geom,cfg);normal_by_id=np.zeros_like(points);planarity_by_id=np.zeros(len(points));normal_by_id[geom]=normals;planarity_by_id[geom]=planarity
    rgbplanes,near,face_diag=collect_planes(points,rgb,normal_by_id,planarity_by_id,source,camera_center,cfg)
    allplanes,_,all_face_diag=collect_planes(points,geom,normal_by_id,planarity_by_id,source,camera_center,cfg)
    accept,reasons=common_acceptance(rgbplanes,near,source,cfg)
    result['acceptance_mask6'][1:]=accept
    result['diagnostics'].update(global_gate_reason='shared_input_acceptance',visible_unique_count=len(visible),geometric_support_count=len(geom),RGB_support_count=len(rgb),source_face_evidence_RGB=face_diag,source_face_evidence_without_pointmask=all_face_diag,common_axis_reasons=reasons,common_gate_derived_before_arm_objective=True,size_pair_assumption='Opposing measured point planes must represent same-object outer surfaces; segmentation/prior are heuristic, not GT-certification or amodal completeness.',normals_from_fit_only_full_cloud=True)
    assignments={str(j)+':'+str(s):sorted(int(rawids[i]) for i in rgbplanes[(j,s)]['ids']) for j in range(3) for s in (-1,1)}
    rgbsha=observation_sha(points,rawids,rgb,assignments);allassign={str(j)+':'+str(s):sorted(int(rawids[i]) for i in allplanes[(j,s)]['ids']) for j in range(3) for s in (-1,1)}
    result['point_support_sha256'][1:3]=rgbsha;result['point_support_sha256'][3]=observation_sha(points,rawids,geom,allassign)
    result['support_ids_by_arm']=[np.array([],np.int64),rawids[rgb].copy(),rawids[rgb].copy(),rawids[geom].copy()]
    if not accept.any():result['arm_status'][1:]=['no_accepted_observed_axis']*3;return result
    fg=boundary_samples(mask,cfg);uv0=corner_projection['uv'];pixel_sigma=max(cfg.RGB_pixel_sigma_min,cfg.RGB_pixel_sigma_bbox_fraction*np.ptp(uv0,axis=0).min())
    center_bound=np.minimum(cfg.center_bound_fraction*source[3:],cfg.center_bound_cap_m)
    cstd=np.clip(cfg.center_prior_std_fraction*source[3:],cfg.center_prior_std_min_m,cfg.center_prior_std_max_m)
    active=np.flatnonzero(accept);bounds=[(-center_bound[j]/source[3+j],center_bound[j]/source[3+j]) if j<3 else (-cfg.size_log_bound,cfg.size_log_bound) for j in active]
    def decode(x):
        z=np.zeros(6);z[active]=x;b=source.copy();b[:3]+=z[:3]*source[3:];b[3:]*=np.exp(z[3:]);return b
    def terms(box,planes,use_RGB):
        prior=float(np.sum(((box[:3]-source[:3])/cstd)**2)+np.sum((np.log(box[3:]/source[3:])/cfg.log_size_prior_std)**2))
        geom_terms=[]
        for j in range(3):
            signs=(-1,1) if accept[j+3] else ((near[j],) if accept[j] else ())
            for sign in signs:
                ids=planes[(j,sign)]['ids']
                if len(ids):
                    sigma=max(cfg.plane_sigma_floor_m,1.4826*float(np.median(np.abs(points[ids,j]-np.median(points[ids,j])))))
                    residual=(points[ids,j]-(box[j]+sign*box[3+j]/2))/sigma
                    geom_terms.append(face_weights[f'{j}:{sign}']*float(huber(residual,cfg.huber_delta).mean()))
        geo=float(np.mean(geom_terms)) if geom_terms else 0.;rgb_loss=0.
        if use_RGB and RGB_weight>0 and len(fg):
            proj=project_model_points(box_corners(box),camera_spec,(W,H))
            if np.any(proj['depth']<=cfg.min_corner_depth_m) or not np.isfinite(proj['uv']).all():rgb_loss=1e6
            else:
                try:
                    hull=ConvexHull(proj['uv']);eq=hull.equations;outside=np.maximum((fg@eq[:,:2].T+eq[:,2]).max(1),0)
                    rgb_loss=float(huber(outside/pixel_sigma,cfg.huber_delta).mean())
                except QhullError:rgb_loss=1e6
        return {'prior':prior,'observed_surface':geo,'RGB_one_sided_containment':rgb_loss,'total':cfg.prior_weight*prior+cfg.geometry_weight*geo+(cfg.rgb_weight*RGB_weight*rgb_loss if use_RGB else 0.)}
    def solve(planes,use_RGB):
        if not any(face_weights[f'{j}:{sgn}']>0 and len(planes[(j,sgn)]['ids']) for j in range(3) for sgn in ((-1,1) if accept[j+3] else ((near[j],) if accept[j] else ()))) and (not use_RGB or RGB_weight==0):
            prior_terms=terms(source,planes,use_RGB)
            return source.copy(),{'success':True,'status':'all_constraints_zero_source_prior_exact','numerical_invalid':False,'finite_nonconverged_retained_without_acceptance_change':False,'message':'all weighted observational terms zero; exact source-prior minimizer','iterations':0,'initial_terms':prior_terms,'attempted_final_terms':prior_terms,'final_terms':prior_terms,'active_parameter_indices':active.tolist(),'normalized_bounds':bounds,'bounded_parameter_solution':[0.]*len(active)}
        fun=lambda x:terms(decode(x),planes,use_RGB)['total']
        sol=minimize(fun,np.zeros(len(active)),method='L-BFGS-B',bounds=bounds,options={'maxiter':cfg.solver_max_iterations,'ftol':cfg.solver_ftol,'gtol':cfg.solver_gtol,'eps':cfg.solver_normalized_finite_difference_eps})
        box=decode(sol.x) if np.isfinite(sol.x).all() else source.copy()
        final=terms(box,planes,use_RGB)
        invalid=not np.isfinite(sol.x).all() or not np.isfinite(box).all() or not np.isfinite(final['total']) or (use_RGB and RGB_weight>0 and final['RGB_one_sided_containment']>=1e6)
        status='numerical_invalid_source_fallback' if invalid else 'converged_bounded_result' if sol.success else 'finite_nonconverged_bounded_result'
        diag={'success':bool(sol.success),'status':status,'numerical_invalid':bool(invalid),'finite_nonconverged_retained_without_acceptance_change':bool(not invalid and not sol.success),'message':str(sol.message),'iterations':int(sol.nit),'initial_terms':terms(source,planes,use_RGB),'attempted_final_terms':final,'active_parameter_indices':active.tolist(),'normalized_bounds':bounds,'bounded_parameter_solution':[float(v) if np.isfinite(v) else None for v in sol.x]}
        if invalid:box=source.copy()
        diag['final_terms']=terms(box,planes,use_RGB)
        return box,diag
    geometry,geometry_diag=solve(rgbplanes,False);full,full_diag=solve(rgbplanes,True);abpre,abpre_diag=solve(allplanes,False);ab,ab_diag=solve(allplanes,True)
    result['pre_constraint_geometry6'][1:3]=geometry;result['pre_constraint_geometry6'][3]=abpre
    result['arm_boxes_raw6'][1]=full;result['arm_boxes_raw6'][2]=geometry;result['arm_boxes_raw6'][3]=ab
    for a in (1,2,3):
        chosen_diag={1:full_diag,2:geometry_diag,3:ab_diag}[a]
        result['arm_status'][a]=chosen_diag['status']
        for j in range(3):
            if accept[j]:result['provenance'][a,j]='two_point_planes_plus_source_prior' if accept[j+3] else 'point_plane_plus_source_size_prior'
            if accept[j+3]:result['provenance'][a,j+3]='opposing_point_planes_plus_source_size_prior'
            else:result['provenance'][a,j+3]='source_size_prior_not_observed'
            if chosen_diag['numerical_invalid']:
                result['provenance'][a,j]='fallback:numerical_invalid'
                result['provenance'][a,j+3]='fallback:numerical_invalid'
            if not chosen_diag['numerical_invalid'] and a in (1,3) and (accept[j] or accept[j+3]):
                if accept[j]:result['provenance'][a,j]+=';one_sided_RGB_containment_available'
                if accept[j+3]:result['provenance'][a,j+3]+=';one_sided_RGB_containment_available'
    result['diagnostics'].update(reliability_weights={'face_weights':face_weights,'RGB_weight':RGB_weight,'point_face_denominator':'original unweighted number of eligible face terms; no renormalization by sum(weights)','prior_unchanged':True},solver={'preconstraint_geometry':geometry_diag,'full_rgb':full_diag,'no_rgb_coordinate':geometry_diag,'ablation_preconstraint':abpre_diag,'rgb_gated_pointmask_ablation':ab_diag},RGB_boundary_sample_count=len(fg),RGB_pixel_sigma=pixel_sigma,RGB_constraint='positive FG containment only; never silhouette equality or visible extent as full size',center_and_size_prior_identifiability='Single camera-facing plane observes c+sign*s/2 only; center correction conditional on fixed source size. Two opposite planes required to open size.',same_main_support_faceassignments_and_acceptance=bool(rgbsha==str(result['point_support_sha256'][2]) and np.array_equal(result['acceptance_mask6'][1],result['acceptance_mask6'][2])))
    assert np.array_equal(result['arm_boxes_raw6'][1:,3:][:,~accept[3:]],np.tile(source[3:][~accept[3:]],(3,1)))
    return result

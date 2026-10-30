"""Fixed-q0 R11 label-free cross-partition constraint reliability.
Array-only CPU API. Shared source/RGB errors may survive every consistency test;
association_supported is internal crossmodal consistency, never identity_verified.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import sys,hashlib,json
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.ndimage import binary_erosion
D=Path(__file__).resolve().parent
sys.path[:0]=[str(D.parent/'round10'),str(D.parent/'round7')]
from visibility_geometry import (canonical_observations,box_corners,unique_visible_ids,normals_from_fit_cloud,DEFAULT_CONFIG as GCFG)
from rgb_support import make_rgb_support,DEFAULT_CONFIG as RCFG
from camera_projection import project_model_points,model_to_camera_matrix

@dataclass(frozen=True)
class ReliabilityConfig:
    voxel_m:float=.10
    partition_salt:str='R11_spatial_physical_parity_v1'
    min_partition_face_points:int=3
    sigma_floor_m:float=.03
    MAD_scale:float=1.4826
    component_radius_size_fraction:float=.1
    component_radius_min_m:float=.05
    component_radius_max_m:float=.25
    max_boundary_points:int=128
    substantial_component_fraction:float=.15
    incompatible_centroid_distance_source_diagonal_factor:float=1.25
    ranking_salt:str='R11_samecoverage_hash_tie_v1'
    not_identity_verification:bool=True
DEFAULT_CONFIG=ReliabilityConfig()
FACES=tuple(f'{j}:{s}' for j in range(3) for s in (-1,1))

def _sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def physical_partitions(points,raw_indices,sample_id,config=DEFAULT_CONFIG):
    p,raw,dedup=canonical_observations(points,raw_indices)
    cells=np.floor(p/config.voxel_m).astype('<i8');u,inv=np.unique(cells,axis=0,return_inverse=True)
    prefix=(config.partition_salt+'|'+str(sample_id)+'|').encode()
    parity=np.array([hashlib.sha256(prefix+c.tobytes()).digest()[0]&1 for c in u],np.uint8)[inv]
    return p,raw,parity,{'dedup':dedup,'voxel_m':config.voxel_m,'partition_unique_XYZ_counts':[int((parity==k).sum()) for k in (0,1)],'partition_rawid_sha256':[_sha(np.sort(raw[parity==k])) for k in (0,1)],'partition_XYZ_sha256':[_sha(p[parity==k]) for k in (0,1)],'disjoint_physical_observations':True,'same_XYZ_same_partition':True,'normal_graphs_must_be_rebuilt_per_partition':True,'shared_RGB_and_source_prior_not_independent_identity_evidence':True}

def _boundary(mask,maxn):
    yx=np.argwhere(mask&~binary_erosion(mask,np.ones((3,3)),border_value=0))
    if len(yx)>maxn:yx=yx[np.linspace(0,len(yx)-1,maxn).astype(int)]
    return yx[:,::-1].astype(float)+.5

def mask_stability(a,b,pixel_sigma,config=DEFAULT_CONFIG):
    a=np.asarray(a,bool);b=np.asarray(b,bool);union=int((a|b).sum())
    if not a.any() or not b.any():return {'defined':False,'reason':'empty_cross_partition_mask','IoU':0.,'symmetric_boundary_distance_px':None,'boundary_exp':0.,'score':0.}
    aa=_boundary(a,config.max_boundary_points);bb=_boundary(b,config.max_boundary_points)
    if not len(aa) or not len(bb):return {'defined':False,'reason':'empty_cross_partition_boundary','IoU':float((a&b).sum()/union),'symmetric_boundary_distance_px':None,'boundary_exp':0.,'score':0.}
    dist=.5*(cKDTree(bb).query(aa)[0].mean()+cKDTree(aa).query(bb)[0].mean());iou=float((a&b).sum()/union);ex=float(np.exp(-dist/pixel_sigma))
    return {'defined':True,'reason':'cross_partition_mask_consistency_not_identity','IoU':iou,'symmetric_boundary_distance_px':float(dist),'boundary_exp':ex,'score':iou*ex,'pixel_sigma':float(pixel_sigma)}

def component_ambiguity(points,depth,source_size,config=DEFAULT_CONFIG):
    p=np.asarray(points,float);z=np.asarray(depth,float)
    radius=float(np.clip(config.component_radius_size_fraction*np.min(source_size),config.component_radius_min_m,config.component_radius_max_m))
    if not len(p):return {'defined':False,'radius_m':radius,'component_count':0,'substantial_component_count':0,'fragmentation_purity':0.,'ambiguity_factor':0.,'reason':'no_foreground_physical_support','conflict_observed':False}
    pairs=cKDTree(p).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(pairs)*2,np.uint8),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(p),len(p))).tocsr()
    n,lab=connected_components(graph,directed=False);counts=np.bincount(lab,minlength=n);order=np.argsort(-counts,kind='stable')
    threshold=max(config.min_partition_face_points,int(np.ceil(config.substantial_component_fraction*len(p))))
    substantial=np.flatnonzero(counts>=threshold);conflicting=set();pairinfo=[]
    limit=float(config.incompatible_centroid_distance_source_diagonal_factor*np.linalg.norm(source_size))
    centroids={int(j):p[lab==j].mean(0) for j in substantial}
    for ii,j in enumerate(substantial):
        for k in substantial[ii+1:]:
            distance=float(np.linalg.norm(centroids[int(j)]-centroids[int(k)]));conflict=distance>limit
            pairinfo.append({'component_indices':[int(j),int(k)],'centroid_distance_m':distance,'source_diagonal_limit_m':limit,'incompatible_under_source_size_prior':bool(conflict)})
            if conflict:conflicting.update([int(j),int(k)])
    conflict_fraction=float(sum(counts[j] for j in conflicting)/len(p));meddepth=[float(np.median(z[lab==j])) for j in order[:2]]
    return {'defined':True,'radius_m':radius,'component_count':int(n),'substantial_component_threshold':threshold,'substantial_component_count':len(substantial),'component_counts_descending':counts[order].astype(int).tolist(),'fragmentation_purity':float(counts[order[0]]/len(p)),'fragmentation_support_fraction':float(counts[counts<threshold].sum()/len(p)),'conflict_pair_diagnostics':pairinfo,'conflict_observed':bool(conflicting),'conflicting_support_fraction':conflict_fraction,'ambiguity_factor':1.-conflict_fraction,'second_component_fraction':float(counts[order[1]]/len(p)) if n>1 else 0.,'two_largest_median_depth_separation_m':abs(meddepth[0]-meddepth[1]) if n>1 else None,'reason':'source_size_incompatibility_hypothesis_not_identity','sparse_fragmentation_not_confident_conflict':True,'same_source_size_error_can_survive':True}

def _observations(p,source,camera,shape,mask):
    H,W=shape;proj=project_model_points(p,camera,(W,H));visible,pixels=unique_visible_ids(p,proj,(W,H))
    margin=np.minimum(GCFG.crop_fraction*source[3:]+GCFG.crop_margin_add_m,GCFG.crop_margin_cap_m)
    geom=visible[(np.abs(p[visible]-source[:3])<=source[3:]/2+margin).all(1)]
    selected=geom[mask[pixels[geom,1],pixels[geom,0]]]
    normals,plan=normals_from_fit_cloud(p,geom,GCFG);nn=np.zeros_like(p);pp=np.zeros(len(p));nn[geom]=normals;pp[geom]=plan
    return {'p':p,'proj':proj,'pixels':pixels,'geom':geom,'selected':selected,'normals':nn,'planarity':pp}

def _face(obs,source,j,sign):
    p=obs['p'];ids=obs['selected'];band=min(GCFG.face_band_fraction*source[j+3]+GCFG.face_band_add_m,GCFG.face_band_cap_m)
    ids=ids[(np.abs(obs['normals'][ids,j])>=GCFG.normal_axis_alignment_min)&(obs['planarity'][ids]>=GCFG.planarity_min)&(np.abs(p[ids,j]-(source[j]+sign*source[j+3]/2))<=band)&(sign*(p[ids,j]-source[j])>0)]
    if not len(ids):return {'count':0,'defined':False,'ids':ids}
    coord=p[ids,j];med=float(np.median(coord));mad=float(np.median(np.abs(coord-med)));t=[k for k in range(3) if k!=j];span=np.ptp(p[ids][:,t],axis=0)
    normals=obs['normals'][ids].copy();normals*=np.where(normals[:,j]>=0,1.,-1.)[:,None];n=normals.mean(0);ln=np.linalg.norm(n);n=n/ln if ln>0 else np.zeros(3)
    return {'count':len(ids),'defined':bool(ln>0),'ids':ids,'median':med,'MAD_m':mad,'tangent_span_m':span,'normal':n,'mean_planarity':float(obs['planarity'][ids].mean())}

def face_consistency(base,a,b,axis,source_size,config=DEFAULT_CONFIG):
    counts=[int(x['count']) for x in (base,a,b)]
    if not(base['defined'] and a['defined'] and b['defined']) or min(counts[1:])<config.min_partition_face_points:
        return {'defined':False,'reason':'insufficient_cross_partition_face_support_or_normal','counts_base_A_B':counts,'score':0.}
    pooled_mad=.5*(a['MAD_m']+b['MAD_m']);sigma=max(config.sigma_floor_m,config.MAD_scale*pooled_mad)
    offset=abs(a['median']-b['median']);align=float(np.clip(abs(a['normal']@b['normal']),0,1));spanbase=base['tangent_span_m']
    if np.any(spanbase<=0):return {'defined':False,'reason':'degenerate_base_tangential_span','counts_base_A_B':counts,'score':0.}
    ratios=np.r_[a['tangent_span_m']/spanbase,b['tangent_span_m']/spanbase];coverage=float(np.clip(np.min(ratios),0,1));ex=float(np.exp(-offset/sigma))
    return {'defined':True,'reason':'independent_point_graph_face_consistency_not_amodal_identity','counts_base_A_B':counts,'offset_disagreement_m':float(offset),'offset_sigma_m':float(sigma),'offset_exp':ex,'unsigned_normal_alignment':align,'min_tangent_coverage_ratio':coverage,'tangent_coverage_ratios':ratios.tolist(),'mean_planarity_A_B':[a['mean_planarity'],b['mean_planarity']],'score':ex*align*coverage}

def apply_correspondence_gate(result,accepted,changed_actual):
    """Raw diagnostics retained; association stage gates effective fusion weights.
    Supported only means source-conditioned geometric correspondence compatibility.
    """
    accepted=np.asarray(accepted,bool);changed=np.asarray(changed_actual,bool)
    assert accepted.shape==changed.shape==(6,) and not (changed&~accepted).any()
    result['raw_face_weights']=dict(result['face_weights'])
    result['raw_RGB_weight']=float(result['RGB_weight'])
    result['raw_RGB_weight_defined']=bool(result['RGB_weight_defined'])
    result['raw_axis_reliability6']=np.asarray(result['axis_reliability6']).copy()
    result['raw_axis_reliability_defined6']=np.asarray(result['axis_reliability_defined6'],bool).copy()
    active=np.flatnonzero(changed)
    result['raw_case_reliability']=float(np.min(result['raw_axis_reliability6'][active])) if len(active) and result['raw_axis_reliability_defined6'][active].all() else None
    status=result['association_status']
    assert status in ('association_supported','association_conflicted','association_undefined')
    rawdefined={key:bool(result['features']['faces'][key]['defined']) for key in FACES}
    result['raw_face_weight_defined']=rawdefined
    if status!='association_supported':
        result['face_weights']=dict.fromkeys(FACES,0.);result['RGB_weight']=0.
        result['axis_reliability6']=np.zeros(6)
        result['axis_reliability_defined6']=accepted.copy() if status=='association_conflicted' else np.zeros(6,bool)
        result['RGB_weight_defined']=status=='association_conflicted'
        result['face_weight_defined']=dict.fromkeys(FACES,status=='association_conflicted')
    else:result['face_weight_defined']=rawdefined.copy()
    if status=='association_undefined':result['case_reliability']=None
    elif status=='association_conflicted':result['case_reliability']=0. if len(active) else None
    else:result['case_reliability']=result['raw_case_reliability']
    result['features']['correspondence_gate']={'status':status,'effective_constraints_zeroed':status!='association_supported','raw_features_retained':True,'case_axes':'accepted AND original actual float32 changed only','case_active_axes6':changed.tolist(),'not_literal_text_identity_verification':True}
    result['control_scores']['reliability']=result['case_reliability']
    if result['case_reliability'] is None:result['undefined_reasons'].append('case_reliability_undefined_for_actual_changed_coordinates')
    elif result['case_reliability']==0:result['features']['case_reliability_zero_is_defined_not_missing']=True
    return result

def compute_reliability(*,sample_id,points,raw_indices,rgb,source_box6,camera_spec,base_mask2d,base_result,config=DEFAULT_CONFIG):
    source=np.asarray(source_box6,float);image=np.asarray(rgb);mask=np.asarray(base_mask2d,bool)
    assert source.shape==(6,) and np.isfinite(source).all() and (source[3:]>0).all() and image.dtype==np.uint8 and image.shape[:2]==mask.shape
    accepted=np.asarray(base_result['acceptance_mask6'],bool)[1];target=np.asarray(base_result['arm_boxes_raw6'],float)[1]
    p,raw,parity,part=physical_partitions(points,raw_indices,sample_id,config)
    result={'face_weights':dict.fromkeys(FACES,0.),'RGB_weight':0.,'axis_reliability6':np.zeros(6),'case_reliability':None,'axis_reliability_defined6':np.zeros(6,bool),'RGB_weight_defined':False,'active_face_keys':[],'association_status':'association_undefined','partition_hashes':part,'features':{'config':asdict(config),'shared_source_errors_may_survive':True,'RGB_and_source_shared_across_partitions':True,'base_accepted_axis6':accepted.tolist(),'base_actual_changed_axis6':(target.astype(np.float32)!=source.astype(np.float32)).tolist(),'faces':{k:{'defined':False,'score':0.,'reason':'cross_partition_not_computed','physical_support_count_control':int(base_result.get('diagnostics',{}).get('source_face_evidence_RGB',{}).get(k,{}).get('count',0)),'final_plane_residual_m':abs(target[int(k[0])]+int(k.split(':')[1])*target[int(k[0])+3]/2-base_result['diagnostics']['source_face_evidence_RGB'][k]['median']) if 'median' in base_result.get('diagnostics',{}).get('source_face_evidence_RGB',{}).get(k,{}) else None} for k in FACES},'source_normalized_shift6':((target-source)/np.r_[source[3:],source[3:]]).tolist()},'undefined_reasons':[],'control_scores':{'source_agreement':-float(np.linalg.norm((target-source)/np.r_[source[3:],source[3:]])),'low_plane_residual':None,'point_count':None,'stability_only':None,'reliability':None}}
    if not accepted.any() or np.array_equal(target.astype(np.float32),source.astype(np.float32)):
        result['undefined_reasons'].append('no_original_G_actual_coordinate_application');return apply_correspondence_gate(result,accepted,np.zeros(6,bool))
    segs=[];obs=[]
    for half in (0,1):
        pp=p[parity==half];seg=make_rgb_support(rgb=image,points_fit=pp,source_box6=source,camera_spec=camera_spec,config=RCFG);segs.append(seg)
        obs.append(_observations(pp,source,camera_spec,mask.shape,seg['mask2d']))
    base=_observations(p,source,camera_spec,mask.shape,mask)
    projcorners=project_model_points(box_corners(source),camera_spec,(mask.shape[1],mask.shape[0]));pixel_sigma=max(GCFG.RGB_pixel_sigma_min,GCFG.RGB_pixel_sigma_bbox_fraction*np.ptp(projcorners['uv'],axis=0).min())
    ms=mask_stability(segs[0]['mask2d'],segs[1]['mask2d'],pixel_sigma,config)
    cross=[]
    for own,other in ((0,1),(1,0)):
        ids=obs[other]['geom'];pix=obs[other]['pixels'][ids];cross.append(float(segs[own]['mask2d'][pix[:,1],pix[:,0]].mean()) if len(ids) else 0.)
    crosssupport=float(np.sqrt(cross[0]*cross[1]))
    comp=component_ambiguity(p[base['selected']],base['proj']['depth'][base['selected']],source[3:],config)
    rgbdefined=bool(all(s['valid'] for s in segs) and ms['defined'] and comp['defined'] and len(obs[0]['geom']) and len(obs[1]['geom']))
    rgbweight=float(ms['score']*crosssupport*comp['ambiguity_factor']) if rgbdefined else 0.
    result['RGB_weight']=float(np.clip(rgbweight,0,1));result['RGB_weight_defined']=rgbdefined
    result['features'].update(mask_stability=ms,crossheldout_foreground_fraction_A_to_B_B_to_A=cross,crossheldout_foreground_support=crosssupport,component_ambiguity=comp,partition_segmentation=[{'valid':bool(s['valid']),'reason':s['diagnostics'].get('reason'),'mask_sha256':_sha(s['mask2d'])} for s in segs],mask_stability_alone_is_not_identity_evidence=True,faces={})
    if all(s['valid'] for s in segs) and comp['defined']:
        result['association_status']='association_conflicted' if comp['conflict_observed'] or ms['IoU']==0 else 'association_supported' if comp['substantial_component_count'] else 'association_undefined'
    else:result['undefined_reasons'].append('one_or_both_partition_segmentations_or_crossmodal_support_undefined')
    for j in range(3):
        for sign in (-1,1):
            key=f'{j}:{sign}';f0=_face(base,source,j,sign);fa=_face(obs[0],source,j,sign);fb=_face(obs[1],source,j,sign)
            fc=face_consistency(f0,fa,fb,j,source[3:],config)
            fc['physical_support_count_control']=f0['count']
            fc['final_plane_residual_m']=abs(target[j]+sign*target[j+3]/2-f0['median']) if f0['count'] else None
            result['features']['faces'][key]=fc;result['face_weights'][key]=float(fc['score'])
    camera_center=np.linalg.inv(model_to_camera_matrix(camera_spec))[:3,3]
    for j in range(3):
        near=int(np.sign(camera_center[j]-source[j])) if abs(camera_center[j]-source[j])>source[j+3]/2+GCFG.camera_outside_slab_margin_m else 0
        activefaces=[f'{j}:-1',f'{j}:1'] if accepted[j+3] else [f'{j}:{near}'] if near else []
        if accepted[j] and activefaces:
            result['axis_reliability6'][j]=min(result['face_weights'][f] for f in activefaces)
            result['axis_reliability_defined6'][j]=all(result['features']['faces'][f]['defined'] for f in activefaces)
        if accepted[j+3]:
            result['axis_reliability6'][j+3]=min(result['face_weights'][f'{j}:-1'],result['face_weights'][f'{j}:1'])
            result['axis_reliability_defined6'][j+3]=all(result['features']['faces'][f'{j}:{sign}']['defined'] for sign in (-1,1))
    relevant=[]
    for j in range(3):
        near=int(np.sign(camera_center[j]-source[j])) if abs(camera_center[j]-source[j])>source[j+3]/2+GCFG.camera_outside_slab_margin_m else 0
        if accepted[j+3]:relevant.extend([f'{j}:-1',f'{j}:1'])
        elif accepted[j] and near:relevant.append(f'{j}:{near}')
    result['active_face_keys']=relevant
    if relevant:
        records=[result['features']['faces'][key] for key in relevant]
        result['control_scores']['point_count']=float(min(rec['physical_support_count_control'] for rec in records))
        if all(rec['final_plane_residual_m'] is not None for rec in records):result['control_scores']['low_plane_residual']=-float(np.mean([rec['final_plane_residual_m'] for rec in records]))
    result['control_scores']['stability_only']=float(ms['score']) if ms['defined'] else None
    changed_actual=accepted & (target.astype(np.float32)!=source.astype(np.float32))
    active=np.flatnonzero(changed_actual)
    result['case_reliability']=float(np.min(result['axis_reliability6'][active])) if len(active) and result['axis_reliability_defined6'][active].all() else None
    result['features']['source_agreement_control']=-float(np.linalg.norm((target-source)/np.r_[source[3:],source[3:]]))
    result['features']['partition_segmentation_calls']=2
    result['control_scores']['reliability']=result['case_reliability']
    result=apply_correspondence_gate(result,accepted,changed_actual)
    return result

def select_exact_half(units,score_key,config=DEFAULT_CONFIG):
    """Pure per-platform fixed-coverage ranking, NA last; no label field accepted.
    units: dictionaries with id/platform/eligible and scalar scores mapping. Eligibility
    is the predeclared base actual-change set; inactive units stay unselected.
    """
    allowed={'id','platform','eligible','scores'}
    assert all(set(u)==allowed for u in units)
    assert len({u['id'] for u in units})==len(units)
    selected=[];groups={}
    for platform in sorted({u['platform'] for u in units}):
        pool=[u for u in units if u['platform']==platform and u['eligible']]
        def key(u):
            tie=hashlib.sha256((config.ranking_salt+'|'+u['id']).encode()).hexdigest()
            value=u['scores'].get(score_key)
            if score_key=='hash_random':return (False,0.,tie)
            assert value is None or np.isfinite(value)
            return (value is None,-float(value) if value is not None else 0.,tie)
        ordered=sorted(pool,key=key);k=(len(pool)+1)//2;keep=[u['id'] for u in ordered[:k]];selected.extend(keep)
        groups[platform]={'eligible_count':len(pool),'K':k,'undefined_score_count':sum(u['scores'].get(score_key) is None for u in pool) if score_key!='hash_random' else 0,'tie_salt':config.ranking_salt}
    return {'selected_ids':selected,'by_platform':groups,'score_key':score_key,'coverage_rule':'ceil(platform base-actual-change count/2), NA-last, input-hash ties; no GT'}

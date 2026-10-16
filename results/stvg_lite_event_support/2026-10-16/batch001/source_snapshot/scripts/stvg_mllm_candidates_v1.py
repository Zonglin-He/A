"""DINO/GT-free student candidate support. No model, media or GPU I/O.

Input boxes use normalized continuous xyxy coordinates. Endpoints may lie
outside the image; the cxcywh chart must be strictly invertible. Native and
Projection are source readouts, never chosen here. Four stratified antithetic
draws support a critic screen; they do not establish a stable MC estimate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT/'artifacts/stvg_mllm_candidate_feedback_v1/candidate_contracts/CPU_CONTRACTS.json'
SIGMAS = {'hc2': .025, 'vidstg': .1}
VERSION = 'student_dualscale_antithetic_v1'


def _array(value):
    if hasattr(value, 'detach'):
        if value.device.type != 'cpu' or value.requires_grad:
            raise ValueError('Supply detached CPU tensors; no GPU inputs')
        return value.detach().numpy()
    return np.asarray(value)


def _array_hash(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str((str(x.dtype), tuple(x.shape))).encode()+x.tobytes()).hexdigest()


def _canonical(value):
    if isinstance(value, np.ndarray):
        return dict(array_dtype=str(value.dtype), array_shape=list(value.shape), array_sha256=_array_hash(value))
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_value': repr(value)}
    return value


def _hash(value):
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _dataset(value):
    name = str(value).lower()
    if name in ('hc', 'hc2', 'hcstvg2'):
        return 'hc2'
    if name in ('vid', 'vidstg'):
        return 'vidstg'
    raise ValueError('Only HC2 and VidSTG are part of this first-round lock')


def _config(dataset, cfg):
    locked = dict(small_sigma=SIGMAS[dataset], large_factor=4., correlation=.5,
        observation_budget=4, time_fraction=.1, component_weights=[.5,.5])
    supplied = dict(cfg or {})
    if set(supplied)-set(locked)-{'video_bounds'}:
        raise ValueError('Unknown first-round candidate configuration')
    for key,value in locked.items():
        if key in supplied and supplied[key] != value:
            raise ValueError('First-round candidate science is fixed: '+key)
    locked['video_bounds'] = supplied.get('video_bounds')
    return locked


def _interval(value):
    if value is None:
        return None
    a = _array(value)
    if a.shape != (2,) or a.dtype.kind not in 'ifu' or not np.isfinite(a).all():
        return None
    lo,hi = map(float,a)
    return [lo,hi] if lo<hi else None


def _cxcywh(boxes):
    x = np.asarray(boxes,dtype=np.float64)
    return np.concatenate(((x[..., :2]+x[..., 2:])*.5, x[..., 2:]-x[..., :2]),axis=-1)


def _xyxy(cxcywh):
    x = np.asarray(cxcywh,dtype=np.float64)
    return np.concatenate((x[..., :2]-x[..., 2:]*.5, x[..., :2]+x[..., 2:]*.5),axis=-1)


def _logit(cxcywh):
    p = np.asarray(cxcywh,dtype=np.float64)
    if not np.isfinite(p).all() or not ((p>0)&(p<1)).all():
        raise ValueError('Noninvertible native cxcywh chart; never clamp')
    return np.log(p)-np.log1p(-p)


def _sigmoid(value):
    x = np.asarray(value,dtype=np.float64)
    p = np.empty_like(x); pos=x>=0
    p[pos]=1./(1.+np.exp(-x[pos]));e=np.exp(x[~pos]);p[~pos]=e/(1.+e)
    return p


def _uniform_positions(ids, native_interval, k=4):
    inside = [i for i,t in enumerate(ids) if native_interval[0]<=t<native_interval[1]]
    if not inside:
        return []
    a,b=inside[0],inside[-1];k=min(k,b-a+1)
    edges=np.linspace(float(ids[a]),float(ids[b])+1,k+1);out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        pool=[i for i in range(a,b+1) if lo<=ids[i]<hi]
        if pool:
            out.append(min(pool,key=lambda i:(abs(ids[i]-(lo+hi)/2),i)))
    while len(out)<k:
        rest=[i for i in range(a,b+1) if i not in out]
        out.append(max(rest,key=lambda i:(min(abs(ids[i]-ids[j]) for j in out),-i)))
    return sorted(out)


def joint_mixture_log_density(deltas, small_sigma, correlation=.5, large_factor=4.):
    """Exact 4m-dimensional sparse-logit density, frame-major flattening.

Each scale has covariance sigma^2 (R_m tensor I_4), where off-diagonal
R_m entries are rho. Dense interpolated tubes have no full-rank Lebesgue
density in 4N dimensions, so only this sparse density is returned.
"""
    x=np.asarray(deltas,dtype=np.float64)
    if x.ndim!=3 or x.shape[-1]!=4 or x.shape[1]<1 or not np.isfinite(x).all():
        raise ValueError('Expected finite [candidate,m,4] sparse logit deltas')
    if not small_sigma>0 or not large_factor>0 or not 0<=correlation<1:
        raise ValueError('Invalid Gaussian mixture parameters')
    m=x.shape[1];d=4*m;rho=float(correlation)
    R=(1-rho)*np.eye(m)+rho*np.ones((m,m))
    inv=np.eye(m)/(1-rho)-rho*np.ones((m,m))/((1-rho)*(1+(m-1)*rho))
    logdet=(m-1)*math.log1p(-rho)+math.log1p((m-1)*rho)
    quadratic=np.einsum('kic,ij,kjc->k',x,inv,x)
    scales=np.asarray([small_sigma,small_sigma*large_factor],np.float64)
    components=np.stack([-.5*d*math.log(2*math.pi)-d*math.log(s)-2*logdet-quadratic/(2*s*s)
        for s in scales],axis=1)
    mixture=np.logaddexp(components[:,0]-math.log(2),components[:,1]-math.log(2))
    covariance=np.stack([s*s*np.kron(R,np.eye(4)) for s in scales])
    return dict(log_density_mixture_logit=mixture, log_density_components=components,
        scales=scales, correlation_matrix=R, covariance_frame_major=covariance,
        mixture_covariance_frame_major=.5*(covariance[0]+covariance[1]),
        dimension=d, frame_correlation_logdet=logdet, quadratic_unit_covariance=quadratic)


def _dense_delta(sparse, positions, frame_ids):
    x=np.asarray(sparse,dtype=np.float64);ids=np.asarray(frame_ids,dtype=np.float64)
    observed=ids[np.asarray(positions,dtype=np.int64)]
    dense=np.empty((len(x),len(ids),4),np.float64)
    for k in range(len(x)):
        for coordinate in range(4):
            dense[k,:,coordinate]=np.interp(ids,observed,x[k,:,coordinate],left=x[k,0,coordinate],right=x[k,-1,coordinate])
    dense[:,positions,:]=x
    return dense


def _temporal(native, projection, boxes, ids, config):
    branch=dict(status='no_information',reason=None,fixed_source_spatial_tube_xyxy=boxes.copy(),
        source_spatial_tube_sha256=_array_hash(boxes),baseline_is_not_MC_candidate=True,
        original_native_interval=native,original_projection_interval=projection,
        video_bounds=config['video_bounds'],candidates=[],requests=[],duplicates=[],
        deduplication='exact numeric endpoint equality, first role retained; no replacement',
        interval_units='original physical frame axis, half-open; no grid quantization')
    ni,pi=_interval(native),_interval(projection)
    bounds=_interval(config['video_bounds'])
    if not len(ids):
        branch['reason']='empty_source_tube';return branch
    if pi is None or ni is None:
        branch['reason']='invalid_native_or_projection_interval';return branch
    length=pi[1]-pi[0];pad=config['time_fraction']*length
    requested=[('Native',ni),('Projection',pi),('Projection_expand_0p1',[pi[0]-pad,pi[1]+pad]),
        ('Projection_contract_0p1',[pi[0]+pad,pi[1]-pad])]
    if bounds is None:
        branch['reason']='true_video_bounds_required_not_inferred_from_last_sample'
        branch['requests']=[dict(role=role,requested_interval=v,interval=None,valid=False,
            reason='missing_or_invalid_video_bounds',duplicate_of=None) for role,v in requested]
        return branch
    if float(ids[0])<bounds[0] or float(ids[-1])>=bounds[1]:
        branch['status']='invalid';branch['reason']='frame_grid_outside_true_video_bounds';return branch
    seen={}
    for role,value in requested:
        clamped=[max(bounds[0],value[0]),min(bounds[1],value[1])]
        valid=clamped[0]<clamped[1];key=tuple(clamped)
        duplicate=seen.get(key) if valid else None
        record=dict(role=role,requested_interval=value,interval=clamped,valid=valid,
            was_clamped=clamped!=value,duplicate_of=duplicate,reason=None if valid else 'empty_after_boundary_clamp')
        branch['requests'].append(record)
        if not valid:
            continue
        if duplicate is not None:
            branch['duplicates'].append(dict(role=role,duplicate_of=duplicate,interval=clamped));continue
        seen[key]=role
        branch['candidates'].append(dict(candidate_id='time_'+role,role=role,interval=clamped,
            requested_interval=value,was_clamped=clamped!=value,source_spatial_tube_sha256=_array_hash(boxes)))
    branch['status']='ready' if branch['candidates'] else 'no_information'
    branch['reason']=None if branch['candidates'] else 'all_temporal_candidates_empty'
    branch['requested_count']=4;branch['unique_valid_count']=len(branch['candidates'])
    return branch


def generate_candidates(native_boxes, frame_ids, native_interval, projection_interval, dataset, seed, cfg=None):
    """Generate two isolated student-only branches, retaining an exact baseline.

cfg may repeat the fixed first-round parameters and must provide video_bounds
for temporal candidates. No true video end is invented from sparse frame IDs.
Native noninvertibility or sigmoid saturation is returned as explicit invalid
evidence; source boxes are never silently clamped, padded or teacher-replaced.
"""
    ds=_dataset(dataset);config=_config(ds,cfg)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=int(seed)<2**64:
        raise ValueError('Seed must be an explicit unsigned 64-bit integer')
    raw=_array(native_boxes);ids_raw=_array(frame_ids)
    if raw.ndim!=2 or raw.shape[1]!=4 or raw.dtype.kind not in 'fiu':
        raise ValueError('native_boxes must be numeric normalized xyxy [N,4]')
    if ids_raw.ndim!=1 or len(ids_raw)!=len(raw) or ids_raw.dtype.kind not in 'fiu':
        raise ValueError('frame_ids must align with the complete source tube')
    if not np.isfinite(ids_raw).all() or not np.all(ids_raw>=0) or not np.all(ids_raw==np.floor(ids_raw)) or not np.all(np.diff(ids_raw)>0):
        raise ValueError('Strictly increasing nonnegative original integer frame IDs required')
    boxes=raw.copy();ids=ids_raw.astype(np.int64);native=_interval(native_interval);projection=_interval(projection_interval)
    result=dict(version=VERSION,dataset=ds,seed=int(seed),config=config,
        input_boxes_dtype=str(boxes.dtype),input_boxes_coordinate_system='normalized continuous xyxy; image endpoints not clipped',
        input_native_boxes_sha256=_array_hash(boxes),input_frame_ids_sha256=_array_hash(ids_raw),
        frame_ids=ids,source_state='caller must supply original independent source state; no inherited/DINO state accepted here',
        baseline=dict(candidate_id='student_native_baseline',native_tube_xyxy=boxes.copy(),native_interval=native_interval,
            projection_interval=projection_interval,independent_control=True,part_of_MC_mixture=False),
        no_GT_input=True,no_teacher_boxes=True,no_model_calls=True,
        first_round_screen_not_candidate_fit=True,
        repeat_hash_definition='array dtype/shape/bytes SHA leaves; candidate excludes its own repeat_hash; all excludes its own repeat_hashes.all')
    space=dict(status='no_information',reason=None,observation_positions=[],observation_frame_ids=[],
        Native_support_for_observations=native_interval,fixed_temporal_interval=projection_interval,
        baseline_dense_tube_xyxy=boxes.copy(),baseline_observed_boxes_xyxy=np.empty((0,4),dtype=boxes.dtype),
        candidates=[],mixture=None,draws=None,context_view_contract='same four sparse observations plus same <=16 real context frames; caller constructs RGB views')
    result['spatial']=space;result['temporal']=_temporal(native_interval,projection_interval,boxes,ids,config)
    if not len(ids):
        space['reason']='empty_source_grid'
    elif native is None:
        space['reason']='invalid_or_empty_native_interval'
    elif projection is None:
        space['reason']='invalid_or_empty_projection_spatial_support'
    elif not np.isfinite(boxes).all():
        space['status']='invalid';space['reason']='nonfinite_native_bbox'
    else:
        positions=_uniform_positions(ids,native,config['observation_budget'])
        space['observation_positions']=positions;space['observation_frame_ids']=ids[positions]
        space['baseline_observed_boxes_xyxy']=boxes[positions].copy()
        chart=_cxcywh(boxes)
        invalid=np.flatnonzero(~np.all(np.isfinite(chart)&(chart>0)&(chart<1),axis=1)).tolist()
        if not positions:
            space['reason']='no_available_frame_in_native_support'
        elif invalid:
            space['status']='invalid';space['reason']='noninvertible_native_cxcywh_without_same_call_logits'
            space['invalid_native_chart_positions']=invalid
        else:
            mu=_logit(chart);m=len(positions);rho=config['correlation'];small=config['small_sigma'];large=small*config['large_factor']
            R=(1-rho)*np.eye(m)+rho*np.ones((m,m));L=np.linalg.cholesky(R)
            rng=np.random.Generator(np.random.PCG64(int(seed)))
            independent=rng.standard_normal((2,m,4));unit=np.einsum('ij,kjc->kic',L,independent)
            deltas=np.stack([small*unit[0],-small*unit[0],large*unit[1],-large*unit[1]])
            dense=_dense_delta(deltas,positions,ids)
            sparse_logits=mu[positions][None]+deltas;dense_logits=mu[None]+dense
            sparse_chart=_sigmoid(sparse_logits);dense_chart=_sigmoid(dense_logits)
            sparse_xy=_xyxy(sparse_chart);dense_xy=_xyxy(dense_chart)
            density=joint_mixture_log_density(deltas,small,rho,config['large_factor'])
            # |d(cxcywh)/d(xyxy)|=1. The sigmoid chart contributes product p(1-p).
            log_jacobian=-np.sum(-np.logaddexp(0.,-sparse_logits)-np.logaddexp(0.,sparse_logits),axis=(1,2))
            space['draws']=dict(independent_standard_normal=independent,correlated_unit_noise=unit,
                sparse_logit_deltas=deltas,dense_logit_deltas=dense,random_generator='NumPy PCG64',
                draw_order=['small_plus','small_minus','large_plus','large_minus'],
                actual_scale_per_draw=[small,small,large,large],antithetic_pairs=[[0,1],[2,3]],
                same_pair_negation_exact=True,sampling='one independent Gaussian base per scale; its exact negative is the second draw')
            space['mixture']=dict(**density,component_weights=[.5,.5],sampling_counts_per_component=[2,2],
                log_abs_d_logit_d_sparse_xyxy=log_jacobian,
                log_density_mixture_sparse_xyxy=density['log_density_mixture_logit']+log_jacobian,
                density_measure='4m-dimensional normalized sparse xyxy chart (or sparse logit chart as labeled)',
                dense_density='singular deterministic interpolation pushforward; no full-rank 4N density claimed',
                MC_contract='four stratified antithetic, dependent draws; balance-mixture q=0.5*q_small+0.5*q_large is recorded; no IID/stability claim',
                baseline_has_no_continuous_mixture_mass=True,
                no_reward_weights_or_update_loss_implemented=True)
            space['interpolation']=dict(axis='actual original physical frame IDs',method='linear in delta-logit space per corresponding coordinate',
                exterior='constant endpoint delta outside the observed physical interval',
                sparse_delta_preserved_exact=bool(np.array_equal(dense[:,positions],deltas)),
                sparse_generated_boxes_preserved_exact=bool(np.array_equal(dense_xy[:,positions],sparse_xy)),
                original_baseline_full_chart_kept=True)
            for k in range(4):
                actual_dense_chart=_cxcywh(dense_xy[k])
                actual_sparse_chart=_cxcywh(sparse_xy[k])
                sparse_chart_valid=bool(np.isfinite(actual_sparse_chart).all()
                    and ((actual_sparse_chart>0)&(actual_sparse_chart<1)).all())
                valid=bool(np.isfinite(dense_xy[k]).all() and ((dense_chart[k]>0)&(dense_chart[k]<1)).all()
                    and ((actual_dense_chart>0)&(actual_dense_chart<1)).all())
                candidate=dict(candidate_id='space_'+['small_plus','small_minus','large_plus','large_minus'][k],
                    scale_component='small' if k<2 else 'large',antithetic_pair=0 if k<2 else 1,antithetic_sign=1 if k%2==0 else -1,
                    sparse_boxes_xyxy=sparse_xy[k],dense_tube_xyxy=dense_xy[k],sparse_logit_boxes=sparse_logits[k],
                    dense_logit_boxes=dense_logits[k],sparse_logit_delta=deltas[k],dense_logit_delta=dense[k],
                    fixed_temporal_interval=projection_interval,valid=valid,
                    invalid_reason=None if valid else 'sigmoid_rounding_boundary_or_nonfinite_generated_box_no_clamp',
                    log_density_mixture_logit=float(density['log_density_mixture_logit'][k]),
                    log_density_components=density['log_density_components'][k],
                    log_density_mixture_sparse_xyxy=float(density['log_density_mixture_logit'][k]+log_jacobian[k]) if sparse_chart_valid else None,
                    sparse_output_chart_valid=sparse_chart_valid,
                    invalid_dense_chart_positions=np.flatnonzero(~np.all((actual_dense_chart>0)&(actual_dense_chart<1),axis=1)).tolist(),
                    image_endpoint_clipping=False,outside_image_dense_frames=int(np.any((dense_xy[k]<0)|(dense_xy[k]>1),axis=1).sum()))
                candidate['repeat_hash']=_hash(candidate);space['candidates'].append(candidate)
            space['status']='ready' if all(x['valid'] for x in space['candidates']) else 'invalid'
            space['reason']=None if space['status']=='ready' else 'at_least_one_generated_chart_rounding_boundary'
    result['status']='ready' if space['status']=='ready' and result['temporal']['status']=='ready' else 'partial_or_no_information'
    result['repeat_hashes']=dict(input=_hash(dict(boxes=boxes,frame_ids=ids_raw,native_interval=native_interval,
        projection_interval=projection_interval,dataset=ds,seed=int(seed),config=config)),
        baseline=_hash(result['baseline']),spatial=_hash(space),temporal=_hash(result['temporal']))
    result['repeat_hashes']['all']=_hash(result)
    return result


def contracts():
    """Synthetic fixtures only; independent torch density and matrix checks."""
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('Candidate contracts require an uninitialized CUDA context')
    ids=np.asarray([0,1,4,9,17,28,40,55],np.int64)
    chart=np.tile([.5,.5,.2,.3],(len(ids),1));boxes=_xyxy(chart)
    fixtures=[];maximum_density_difference=0.;density_coordinates=0
    for ds in ('hc2','vidstg'):
        out=generate_candidates(boxes,ids,[0,56],[4,41],ds,20261013,cfg={'video_bounds':[0,60]})
        repeated=generate_candidates(torch.tensor(boxes),torch.tensor(ids),[0,56],[4,41],ds,20261013,cfg={'video_bounds':[0,60]})
        assert out['repeat_hashes']==repeated['repeat_hashes']
        s=out['spatial'];m=len(s['observation_positions']);assert m==4 and len(s['candidates'])==4
        delta=s['draws']['sparse_logit_deltas'];assert np.array_equal(delta[0],-delta[1]) and np.array_equal(delta[2],-delta[3])
        assert s['interpolation']['sparse_delta_preserved_exact'] and s['interpolation']['sparse_generated_boxes_preserved_exact']
        component=[]
        R=torch.tensor(s['mixture']['correlation_matrix'],dtype=torch.float64)
        for sigma in (SIGMAS[ds],4*SIGMAS[ds]):
            cov=sigma*sigma*torch.kron(R,torch.eye(4,dtype=torch.float64))
            dist=torch.distributions.MultivariateNormal(torch.zeros(4*m,dtype=torch.float64),covariance_matrix=cov)
            component.append(dist.log_prob(torch.tensor(delta.reshape(4,-1),dtype=torch.float64)).numpy())
        actual=np.stack(component,axis=1);mix=torch.logsumexp(torch.tensor(actual)-math.log(2),dim=1).numpy()
        error=max(float(np.max(np.abs(actual-s['mixture']['log_density_components']))),
            float(np.max(np.abs(mix-s['mixture']['log_density_mixture_logit']))))
        assert error<1e-10;maximum_density_difference=max(maximum_density_difference,error);density_coordinates+=actual.size+mix.size
        assert out['baseline']['part_of_MC_mixture'] is False and np.array_equal(out['baseline']['native_tube_xyxy'],boxes)
        assert out['temporal']['fixed_source_spatial_tube_xyxy'].shape==boxes.shape
        assert all(c['fixed_temporal_interval']==[4,41] for c in s['candidates'])
        fixtures.append(dict(dataset=ds,repeat_hashes=out['repeat_hashes'],observation_positions=s['observation_positions'],
            spatial_candidates=4,temporal_candidates=len(out['temporal']['candidates']),density_max_abs_difference=error))
    # Physical-axis interpolation and direction are checked with manual deltas.
    physical=np.asarray([0,1,9,10]);d=np.zeros((1,2,4));d[0,:,0]=[-1.,1.]
    dense=_dense_delta(d,[0,3],physical)
    assert np.allclose(dense[0,:,0],[-1.,-.8,.8,1.],atol=1e-15,rtol=0)
    moved=_sigmoid(dense[0,:,0]);assert moved[0]<moved[1]<.5<moved[2]<moved[3]
    # Outside-image endpoints are preserved; the valid center/size chart is not clipped.
    outside=np.tile([-.05,.1,.15,.4],(len(ids),1))
    outside_out=generate_candidates(outside,ids,[0,56],[4,41],'hc2',11,{'video_bounds':[0,60]})
    assert outside_out['spatial']['status']=='ready' and np.array_equal(outside_out['baseline']['native_tube_xyxy'],outside)
    assert any(c['outside_image_dense_frames']>0 for c in outside_out['spatial']['candidates'])
    boundary=boxes.copy();boundary[2]=[-.5,.1,.5,.4]  # center=0: no invertible logit chart.
    invalid=generate_candidates(boundary,ids,[0,56],[4,41],'hc2',1,{'video_bounds':[0,60]})
    assert invalid['spatial']['status']=='invalid' and invalid['spatial']['invalid_native_chart_positions']==[2]
    nan=boxes.copy();nan[0,0]=float('nan')
    assert generate_candidates(nan,ids,[0,56],[4,41],'hc2',1,{'video_bounds':[0,60]})['spatial']['reason']=='nonfinite_native_bbox'
    unavailable=generate_candidates(boxes,ids,[0,56],[4,41],'hc2',1)
    assert unavailable['temporal']['candidates']==[] and 'true_video_bounds' in unavailable['temporal']['reason']
    noframe=generate_candidates(boxes,ids,[2,3],[4,41],'hc2',1,{'video_bounds':[0,60]})
    assert noframe['spatial']['candidates']==[] and noframe['spatial']['reason']=='no_available_frame_in_native_support'
    empty=generate_candidates(np.empty((0,4)),np.empty(0,np.int64),[0,1],[0,1],'hc2',1,{'video_bounds':[0,1]})
    assert empty['spatial']['candidates']==empty['temporal']['candidates']==[]
    dedup=generate_candidates(boxes,ids,[0,56],[0,56],'hc2',3,{'video_bounds':[0,56]})
    assert [c['role'] for c in dedup['temporal']['candidates']]==['Native','Projection_contract_0p1']
    assert len(dedup['temporal']['duplicates'])==2 and dedup['temporal']['requests'][2]['was_clamped']
    single=generate_candidates(boxes[:1],ids[:1],[0,1],[0,1],'hc2',7,{'video_bounds':[0,1]})
    assert single['spatial']['mixture']['dimension']==4 and len(single['spatial']['observation_positions'])==1
    saturated=np.nextafter(np.float64(1),np.float64(0));fragile=_xyxy(np.tile([saturated,.5,.2,.3],(len(ids),1)))
    sat=generate_candidates(fragile,ids,[0,56],[4,41],'vidstg',5,{'video_bounds':[0,60]})
    # Native or sampled sigmoid rounding boundaries remain explicit, never clamped.
    assert sat['spatial']['status']=='invalid' or any(not c['valid'] for c in sat['spatial']['candidates'])
    rejected=0
    for call in (lambda:generate_candidates(boxes,ids,[0,56],[4,41],'hc2',True),
            lambda:generate_candidates(boxes,np.zeros(len(ids)),[0,56],[4,41],'hc2',1),
            lambda:generate_candidates(boxes,ids,[0,56],[4,41],'hc2',1,{'correlation':0.}),
            lambda:generate_candidates(boxes,ids,[0,56],[4,41],'hc2',1,{'small_sigma':.5})):
        try:call()
        except ValueError:rejected+=1
        else:raise AssertionError('Invalid first-round contract accepted')
    assert rejected==4 and not torch.cuda.is_initialized()
    return dict(status='pass_synthetic_CPU_only',version=VERSION,fixtures=fixtures,
        independent_torch_density_coordinates=density_coordinates,maximum_density_difference=maximum_density_difference,
        antithetic_pairs_exact=True,physical_axis_interpolation_and_direction_pass=True,
        sparse_dense_correspondence_exact=True,baseline_not_MC=True,outside_image_not_clipped=True,
        invalid_native_and_sigmoid_rounding_retained=True,missing_true_video_bounds_unavailable=True,
        dedup_without_replacement=True,empty_and_no_information_preserved=True,protocol_violations_rejected=rejected,
        MC_stability_claim=False,actual_payload_reads=0,GT_reads=0,RGB_reads=0,GPU_calls=0,model_calls=0)


def save_contracts():
    result=contracts()
    result.update(module_path=str(Path(__file__).resolve().relative_to(ROOT)),
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),time=time.time(),
        design_and_runtime_modified=False)
    CONTRACT_PATH.parent.mkdir(parents=True,exist_ok=True)
    with CONTRACT_PATH.open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    return dict(path=str(CONTRACT_PATH.relative_to(ROOT)),sha256=hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        module_sha256=result['module_sha256'],status=result['status'])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('contracts','save_contracts'))
    args=ap.parse_args();print(json.dumps(contracts() if args.action=='contracts' else save_contracts(),allow_nan=False))

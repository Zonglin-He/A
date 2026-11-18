"""Replay the native first cycle and isolate geometry with a fixed winner and support."""
from collections import Counter
import gc
import sys
import time

import numpy as np
from scipy.optimize import minimize

from scope163 import (HERE, PARENT_WORK, PLATFORMS, PROTOCOL, SETTINGS, WORK, atomic_json,
                      pinned_parent, read, sensor_row, sha, source_scope)
from evidence162 import EvidenceGrounder
from geometry162 import axial, effective_count, surface_cost, to_box, to_parameters
from run162 import array_file, pool, submit, sync
from solver162 import Options, Relationships, build_roles, choose, fit_roles, improve_support, preparation, support_energy
from support162 import SceneSupport

AXES = ('x','y','z','length','width','height','yaw')
MASKS = {
    'yaw_only': (6,),
    'center_yaw': (0,1,2,6),
    'size_yaw': (3,4,5,6),
    'all7': tuple(range(7)),
    'all7_instance_prior_only': tuple(range(7)),
}
STAGES = ('r158', 'initial', 'support_only', 'fit_keep_support_winner',
          'fit_keep_initial_winner', 'reselect', 'r162_final')


def fitted_control(region, weights, relation, indices, drop_source_prior):
    """The R162 fitting loss with the listed coordinates clamped and one optional term removed."""
    fitter = region.fitter
    initial = fitter.initial
    if len(region.points)<8 or effective_count(weights)<6 or weights.sum()<.75:
        return initial.copy(), 0, np.zeros(7,dtype=np.int8)
    x0 = to_parameters(initial)
    bounds_half = np.r_[.45*np.maximum(initial[3:6],.30),np.full(3,.48),1.1]
    mask=np.isin(np.arange(7),indices)
    half=np.where(mask,bounds_half,0.)
    bounds=list(zip(x0-half,x0+half))
    den=weights.sum()+fitter.missing_mass

    def objective(params):
        costs, jacobian=surface_cost(region.points,params,fitter.scale,derivative=True)
        regularizer, grad=fitter.regularize(params)
        if drop_source_prior:
            prior=(params[3:6]-fitter.prior_logdim)/fitter.prior_sigma
            regularizer-=.1*fitter.size_prior*(prior@prior)
            grad[3:6]-=.2*fitter.size_prior*prior/fitter.prior_sigma
        loss=fitter.mean_cost(costs,weights)+regularizer
        grad+=weights@jacobian/den
        if relation is not None:
            value, partial=relation(params[:3])
            loss+=value/region.fit_weight
            grad[:3]+=partial/region.fit_weight
        return loss,grad

    start,_=objective(x0)
    result=minimize(objective,x0,method='L-BFGS-B',jac=True,bounds=bounds,
                    options={'maxiter':fitter.fit_iterations,'maxls':8,'ftol':1e-10})
    accepted=np.isfinite(result.fun) and result.fun<=start+1e-10
    x=result.x if accepted else x0
    box=to_box(x)
    box[~mask]=initial[~mask]
    hit_low=mask & np.isclose(x,x0-half,atol=1e-7,rtol=0.)
    hit_high=mask & np.isclose(x,x0+half,atol=1e-7,rtol=0.)
    return box,int(result.nfev),hit_high.astype(np.int8)-hit_low


def total_energy(roles, relationships):
    value=relationships.energy()
    for role in roles:
        if role:
            geometry=[h.box for h in role.hypotheses]
            value+=support_energy(role,role.state,preparation(role,geometry))[0]
            value+=sum(h.fit_weight*h.fitter.regularize(to_parameters(h.box))[0] for h in role.hypotheses)
    return value


def part_state(roles, selection):
    return selection[0] if selection and roles[0] is not None else -1


def winner_box(roles, index, fallback):
    return roles[0].hypotheses[index].box.copy() if index>=0 else fallback.copy()


def winner_support(scene, roles, index):
    return roles[0].state[:scene.nseed,index].copy() if index>=0 else np.zeros(scene.nseed)


def first_update(roles):
    for r in roles:
        if r:
            improve_support(r,[h.box for h in r.hypotheses],4)


def experiment(path, ev, opts, dims, sigma, expected_old, expected_final):
    scene=SceneSupport(ev)
    old=ev.old_pca.astype(float)
    np.testing.assert_array_equal(ev.old_pca,expected_old)
    roles=build_roles(scene,opts,dims,sigma)
    rel=Relationships(ev,roles,opts.relation_weight)
    energies=[total_energy(roles,rel)]
    initial_choice=part_state(roles,choose(scene,roles,rel))
    initial_box=winner_box(roles,initial_choice,old)
    source_query=initial_choice>=0 and roles[0].hypotheses[initial_choice].query==ev.old_winner
    support0=winner_support(scene,roles,initial_choice)
    first_update(roles)
    energies.append(total_energy(roles,rel))
    qualifier=part_state(roles,choose(scene,roles,rel))
    prefit_box=winner_box(roles,qualifier,old)
    support1=winner_support(scene,roles,initial_choice)
    support_selected=winner_support(scene,roles,qualifier)

    arms,evals,hits={},[],[]
    if initial_choice>=0:
        region=roles[0].hypotheses[initial_choice]
        point_weights=region.point_weights(roles[0].state[:,initial_choice])
        relation=rel.neighbors(0,initial_choice,region.box[:3])
        fit_count=effective_count(point_weights)
        support_mass=float(point_weights.sum())
        for name,mask in MASKS.items():
            box,n_eval,hit=fitted_control(region,point_weights,relation,mask,name.endswith('instance_prior_only'))
            arms[name]=box
            evals.append(n_eval)
            hits.append(hit)
    else:
        fit_count=support_mass=0.
        for name in MASKS:
            arms[name]=old.copy()
            evals.append(0)
            hits.append(np.zeros(7,dtype=np.int8))
    fit_roles(roles,rel)
    energies.append(total_energy(roles,rel))
    kept=winner_box(roles,qualifier,old)
    kept_initial=winner_box(roles,initial_choice,old)
    # The all-axis control must replay the actual h0 fit, including original relation cost.
    np.testing.assert_allclose(arms['all7'],kept_initial,atol=1e-10,rtol=0.)
    postfit_choice=part_state(roles,choose(scene,roles,rel))
    postfit_box=winner_box(roles,postfit_choice,old)
    for _ in range(2):
        first_update(roles)
        fit_roles(roles,rel)
    end_choice=part_state(roles,choose(scene,roles,rel))
    end_box=winner_box(roles,end_choice,old)
    np.testing.assert_allclose(end_box,expected_final,atol=1e-10,rtol=0.)
    b0b1=np.stack((old,initial_box,prefit_box,kept,kept_initial,postfit_box,end_box,*arms.values()))
    d=b0b1[-len(MASKS):]
    drift=np.c_[d[:,:3]-initial_box[:3],np.log(d[:,3:6]/initial_box[3:6]),axial(d[:,6]-initial_box[6])]
    array_file(path,boxes=b0b1,arms=np.array((*STAGES,*MASKS)),hits=np.stack(hits),fit_evals=np.array(evals),
               drift=drift,energy=np.array(energies),seeds=ev.seeds,
               supports=np.stack((support0,support1,support_selected)),
               initial_matches_old_query=source_query,fit_effective_points=fit_count,fit_support_mass=support_mass,
               h0=initial_choice,hS=qualifier,hF=postfit_choice,h3=end_choice)


def main():
    import torch

    checks=pinned_parent()
    records=source_scope()
    specs=read(SETTINGS)['sources']
    folder=WORK/'source_predictions'
    folder.mkdir(exist_ok=True)
    start=time.perf_counter()
    with pool() as executor:
        for source in PLATFORMS:
            model=EvidenceGrounder(source)
            spec=specs[source]
            opts=Options(**spec['options'])
            dims,sigma=(np.array(spec[k]) for k in ('log_dimensions','log_scale'))
            pending=[]
            for i,row in enumerate(records):
                if row['source']!=source:
                    continue
                name=f'{i:05d}.npz'
                path=folder/name
                if path.exists():
                    continue
                parent=PARENT_WORK/'calibration_predictions'/name
                assert sha(parent)==checks[name]
                with np.load(parent,allow_pickle=False) as p:
                    index=p['names'].tolist()
                    expected_old=p['boxes'][index.index('r158')]
                    expected_final=p['boxes'][index.index(spec['configuration'])]
                ev=model.evidence_frame(sensor_row(row))
                submit(executor,pending,experiment,path,ev,opts,dims,sigma,expected_old,expected_final)
            sync(pending)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print('Frozen source replay verified',source,64,flush=True)
    digests={p.name:sha(p) for p in sorted(folder.glob('*.npz'))}
    assert set(digests)=={f'{i:05d}.npz' for i in range(192)}
    index=WORK/'SOURCE_FILES_PRIVATE.json'
    atomic_json(index,digests)
    atomic_json(HERE/'SOURCE_SEAL.json',dict(n=192,protocol_sha256=sha(PROTOCOL),source_config_sha256=sha(SETTINGS),
                 index_sha256=sha(index),source_boxes_replayed=192,case_elapsed=time.perf_counter()-start,
                 ground_truth_in_arms=False,FINAL_access=False))


if __name__=='__main__':
    main()

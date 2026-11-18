"""Freeze the original LCIB initialization and change only its readout factors."""
from dataclasses import replace
import itertools
import time

import numpy as np

from scope166 import ARMS, MAX_HYPOTHESES as CAP, MAX_ROLES as NROLE
from geometry162 import effective_count, surface_cost, to_parameters
from solver162 import Options, Relationships, build_roles, choose
from support162 import SceneSupport, SeedHypothesis
from crossplatform.ground_single_frame import orient_box


class FixedBankScene(SceneSupport):
    banks=None

    def candidates(self, role, prior, mode):
        if self.banks is None:
            return super().candidates(role,prior,mode)
        return self.banks[role]

    def legacy_bank(self, role, full):
        old=[h for h in full if h.query>=0]
        if len(old)==len(full):
            return old
        ev=self.ev
        support=self.box_support
        count=support.sum(1)
        mass=(support @ ev.affinity[role].astype(float))/np.sqrt(np.maximum(count,1))
        normalized=mass/max(float(mass.max()),1e-12)
        if role==0:
            priors=ev.utility/ev.utility.max()
            order=[ev.old_winner,*np.argsort(-((.35+.65*normalized)*priors),kind='stable').tolist()]
        else:
            priors=normalized
            order=[*[int(q) for q in ev.hypotheses[role] if q>=0],*np.argsort(-normalized,kind='stable').tolist()]
        for q in order:
            box=ev.boxes[q]
            if np.any(box[3:6]<=.12) or (count[q]<2 and q!=ev.old_winner):
                continue
            if any(h.query==q for h in old) or self.duplicates(old,box[:3],box[3:6],support[q]):
                continue
            angle=ev.old_pca.astype(float) if q==ev.old_winner else self.orient(box)
            old.append(SeedHypothesis(q,angle,support[q],np.log(max(priors[q],.025))))
            if len(old)==len(full):
                break
        return old


def role_neutral_banks(scene, banks, shared):
    """Keep the available boxes and replace only the reference-specific evidence."""
    count=scene.box_support.sum(1)
    mass=(scene.box_support @ shared.astype(float))/np.sqrt(np.maximum(count,1))
    scores=mass/max(float(mass.max()),1e-12)
    degree=np.asarray(scene.geometry.sum(1)).ravel()
    smooth=(shared+scene.geometry@shared)/(1+degree)
    top=max(float(smooth.max()),1e-12)
    changed=banks[:1]
    for regions in banks[1:]:
        current=[]
        for h in regions:
            value=scores[h.query] if h.query>=0 else .55*float(smooth[h.seed_mask].max(initial=0.))/top
            current.append(replace(h,log_prior=float(np.log(max(.025,value)))))
        changed.append(current)
    return changed


def unary_scores(scene, roles):
    result=[]
    for e,role in enumerate(roles):
        values={-1:-1.45 if e==0 else -1.15}
        if role is not None:
            for j,h in enumerate(role.hypotheses):
                weights=h.point_weights(role.state[:,j])
                if effective_count(weights)<6 or weights.sum()<.75:
                    continue
                weights0=role.state[:scene.nseed,j]
                affinity=(weights0@scene.ev.affinity[e])*scene.nseed/max(weights0.sum(),1e-9)
                surface=surface_cost(h.points,to_parameters(h.box),h.fitter.scale)
                values[j]=float(.85*h.log_prior+.43*np.log(np.clip(affinity,.2,8.))-2.5*h.fitter.mean_cost(surface,weights))
        result.append(values)
    return result


def bind(scene, roles, relationships, independent=False):
    scores=unary_scores(scene,roles)
    involved={j for a,b,_ in relationships.edges for j in (a,b)}
    short=[list(sorted(s,key=lambda j:-s[j]))[:4 if e in involved else 1] for e,s in enumerate(scores)]
    independent_tuple=tuple(s[0] for s in short)
    terms={}
    selection=[-1]*len(roles)
    for a,b in dict.fromkeys(tuple(sorted((a,b))) for a,b,_ in relationships.edges):
        pairs={}
        for x,y in itertools.product(short[a],short[b]):
            selection[a],selection[b]=x,y
            pairs[x,y]=relationships.decision_costs(selection,scene.nseed)
        terms[a,b]=pairs
        selection[a]=selection[b]=-1
    if independent:
        return independent_tuple,scores,short,terms

    def total(indices):
        return sum(s[j] for s,j in zip(scores,indices))-sum(p[indices[a],indices[b]] for (a,b),p in terms.items())

    choice=max(itertools.product(*short),key=total)
    return choice,scores,short,terms


def pca(points, box):
    result=box.astype(float).copy()
    result[6]=orient_box(points,box[:6])[6]
    return result


def selected_boxes(roles, selected, points, fallback):
    result=np.full((NROLE,7),np.nan)
    native=result.copy()
    origins=np.full(NROLE,-2,np.int16)
    result[0]=native[0]=fallback
    for e,j in enumerate(selected):
        if j>=0:
            h=roles[e].hypotheses[j]
            result[e]=pca(points,h.box)
            native[e]=h.box
            origins[e]=h.query
    return result,native,origins


def frozen_case(ev, shared, setting, pilot=False):
    started=time.perf_counter()
    option=Options(**setting['options'])
    prior,sigma=(np.asarray(setting[k]) for k in ('log_dimensions','log_scale'))
    assert option.init=='scene'
    scene=FixedBankScene(ev)
    banks=[scene.candidates(i,prior,option.init) if mask.any() else [] for i,mask in enumerate(ev.role_mask)]
    scene.banks=banks
    roles=build_roles(scene,option,prior,sigma)
    relationships=Relationships(ev,roles,option.relation_weight)
    full,scores,short,terms=bind(scene,roles,relationships)
    independent=tuple(s[0] for s in short)
    if pilot:
        assert full==choose(scene,roles,relationships)
        assert independent==choose(scene,roles,Relationships(ev,roles,0.))

    legacy=[scene.legacy_bank(i,h) for i,h in enumerate(banks)]
    assert all(len(a)<=len(b)<=CAP for a,b in zip(legacy,banks))
    if any(h.query<0 for rr in banks for h in rr):
        scene.banks=legacy
        legacy_roles=build_roles(scene,option,prior,sigma)
        legacy_selection,*_=bind(scene,legacy_roles,Relationships(ev,legacy_roles,option.relation_weight))
    else:
        legacy_roles,legacy_selection=roles,full

    if len(roles)>1 and relationships.edges:
        scene.banks=role_neutral_banks(scene,banks,shared)
        neutral_ev=replace(ev,affinity=np.array([ev.affinity[0],*[shared]*(len(roles)-1)],np.float32))
        scene.ev=neutral_ev
        neutral_roles=build_roles(scene,option,prior,sigma)
        neutral_selection,*_=bind(scene,neutral_roles,Relationships(neutral_ev,neutral_roles,option.relation_weight))
        scene.ev=ev
    else:
        neutral_roles,neutral_selection=roles,full

    baseline=np.full((NROLE,7),np.nan)
    baseline[0]=ev.old_pca
    diagnostic=baseline.copy()
    for e in range(1,len(roles)):
        if ev.route_weights[e].max()>0:
            q=int(ev.hypotheses[e,ev.route_weights[e].argmax()])
            diagnostic[e]=pca(ev.points,np.r_[ev.boxes[q],0.])

    record=[selected_boxes(r,seq,ev.points,ev.old_pca) for r,seq in
            ((roles,full),(roles,independent),(neutral_roles,neutral_selection),(legacy_roles,legacy_selection))]
    boxes=np.concatenate((baseline[None],np.stack([v[0] for v in record])))
    origins=np.concatenate((np.array([[ev.old_winner,*[-2]*(NROLE-1)]],np.int16),np.stack([v[2] for v in record])))
    starts=np.full((NROLE,CAP,7),np.nan)
    eligibility=np.zeros((NROLE,CAP),bool)
    in_beam=eligibility.copy()
    routes=np.full((NROLE,3),-1,np.int16)
    for e,rs in enumerate(banks):
        for j,h in enumerate(rs):
            starts[e,j]=pca(ev.points,h.box)
            eligibility[e,j]=j in scores[e]
            in_beam[e,j]=j in short[e]
        routes[e]=ev.hypotheses[e]

    spans=np.full((NROLE,2),-1,np.int16)
    spans[:len(roles)]=ev.role_spans
    budget=np.zeros((NROLE,3),np.int8)
    for e,(all_,orig) in enumerate(zip(banks,legacy)):
        budget[e]=len(all_),len(orig),sum(h.query<0 for h in all_)
    used_visible=any(i in short[a] and j in short[b] and (a==0 or b==0)
                     for a,i,b,j,_,_ in relationships.pairs)
    active=any(max(p.values(),default=0.)>0. for (a,b),p in terms.items() if a==0 or b==0)
    return dict(boxes=boxes,origins=origins,native_full=record[0][1],diagnostic=diagnostic,starts=starts,
                eligibility=eligibility,in_beam=in_beam,routes=routes,spans=spans,budget=budget,
                role_count=np.int8(len(roles)),seeds=ev.seeds,edges=ev.edges,
                camera_intrinsic=ev.camera_intrinsic,model_to_camera=ev.model_to_camera,distortion=ev.distortion,
                width=ev.width,visible_relation=bool(used_visible),active_relation=bool(active),
                seconds=time.perf_counter()-started)

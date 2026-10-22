"""Four-coefficient per-query score calibration using only eight teacher boxes.

    No dataset labels, neural weights, temporal readout or candidate changes.
"""
import copy
import torch
import torch.nn.functional as F
from tools.boxfixed_tube._spatial_math import occupancy,affinity,fit_axis_affine,transform_native_box,EPS,scoped_single_thread
from tools.boxfixed_tube.spatial import evidence_dp
from tools.boxfixed_followup.coverage import geometric_valid
from tools.boxfixed_followup.refinement import refined_box


def pooled(box,features,valid,height,width):
    mass=occupancy(box,height,width)[0].flatten()*valid
    return F.normalize((features.double()*mass[:,None]).sum(0)/mass.sum().clamp_min(EPS),dim=0,eps=1e-12)


def vector(box,reference,z,zref,foreground,background):
    dz=z-zref
    return torch.stack((dz@foreground,dz@(foreground-background),
                        torch.linalg.vector_norm((box[:2]-reference[:2])/reference[2:]),
                        torch.linalg.vector_norm((box[2:]/reference[2:]).log())))


def iou(a,b):
    lo=torch.maximum(a[:2]-a[2:]/2,b[:2]-b[2:]/2);hi=torch.minimum(a[:2]+a[2:]/2,b[:2]+b[2:]/2)
    inter=(hi-lo).clamp_min(0).prod();return inter/(a[2:].prod()+b[2:].prod()-inter).clamp_min(EPS)


def ridge(X,y):
    scale=X.square().mean(0).sqrt().clamp_min(.01);Z=X/scale
    beta=torch.linalg.solve(Z.T@Z+len(X)*torch.eye(X.shape[1],dtype=X.dtype),Z.T@y)
    return beta,scale


def fit_query(X,y,target,source=None):
    if len(X)<8 or len(torch.unique(target))<4:
        return dict(valid=False,samples=len(X),reason='insufficient_distinct_teacher_targets')
    predicted=torch.zeros_like(y)
    for t in torch.unique(target):
        test=target==t;train=~test
        if source is not None:train&=source!=t
        if train.sum()<4:return dict(valid=False,samples=len(X),reason='insufficient_training_for_target_holdout')
        b,s=ridge(X[train],y[train]);predicted[test]=(X[test]/s)@b
    mse=(predicted-y).square().mean();zero=y.square().mean();skill=1-mse/zero.clamp_min(EPS)
    beta,scale=ridge(X,y)
    return dict(valid=bool(zero>EPS and skill>0),samples=len(X),reason='teacher_target_holdout_skill'if skill>0 else'teacher_target_holdout_not_better_than_zero',
                beta=beta,scale=scale,rmse=float(mse.sqrt()),teacher_cv_skill=float(skill),teacher_zero_mse=float(zero),teacher_cv_mse=float(mse),heldout_predictions=predicted)


@scoped_single_thread
def calibration(bank,raw,app,candidate_parameters):
    ev=raw['evidence'];source=torch.as_tensor(raw['Source']).double();norm=F.normalize(ev.features,dim=-1,eps=1e-12)
    H,W=ev.height,ev.width;yy,xx=torch.meshgrid((torch.arange(H,dtype=torch.float64)+.5)/H,(torch.arange(W,dtype=torch.float64)+.5)/W,indexing='ij');xy=torch.stack((xx,yy),-1).reshape(-1,2)
    zsource=torch.stack([pooled(source[t],ev.features[t],ev.valid[t],H,W)for t in range(bank.frames)])
    X=[];y=[];targets=[];sources=[]
    # Target teacher box is accessed ONLY for the response y, after the proposal
    # and all four predictors are computed. Neither its mask nor its foreground
    # prototype contributes to a held-out target's predictors.
    for j,a in enumerate(bank.positions):
        if not app['ring_available'][j]:continue
        for k,t in enumerate(bank.positions):
            if j==k:continue
            ta=affinity(norm,ev.valid,t,a);at=affinity(norm,ev.valid,a,t)
            joint=at*ta.T*ev.seeds[j].flatten()[:,None];fit,_=fit_axis_affine(joint,xy,xy)
            if fit is None:continue
            prop=transform_native_box(bank.native_boxes[j],*fit)
            if prop is None:continue
            if candidate_parameters['mode']!='original':
                prop=refined_box(source[t],source[a],bank.native_boxes[j],prop,candidate_parameters['alpha'],candidate_parameters['beta'],candidate_parameters['mode'])
            if not (torch.isfinite(prop).all() and (prop[2:]>0).all() and ((prop[:2]+prop[2:]/2>0)&(prop[:2]-prop[2:]/2<1)).all()):continue
            z=pooled(prop,ev.features[t],ev.valid[t],H,W)
            x=vector(prop,source[t],z,zsource[t],bank.seed_features[j],app['background'][j])
            response=iou(prop,bank.native_boxes[k])-iou(source[t],bank.native_boxes[k])
            X.append(x);y.append(response);targets.append(k);sources.append(j)
    X=torch.stack(X)if X else torch.zeros(0,4,dtype=torch.float64);y=torch.stack(y)if y else torch.zeros(0,dtype=torch.float64);targets=torch.tensor(targets)
    sources=torch.tensor(sources);model=fit_query(X,y,targets,sources);adv=torch.zeros(bank.frames,3,dtype=torch.float64)
    if 'beta'in model:
        for t in range(bank.frames):
            for side in (0,1):
                j=int(bank.source_anchors[t,side])
                if j<0 or not bank.side_available[t,side]or not app['ring_available'][j]:continue
                x=vector(bank.boxes[t,side+1],source[t],bank.features[t,side+1],zsource[t],bank.seed_features[j],app['background'][j])
                x0=vector(bank.E[t],source[t],bank.features[t,0],zsource[t],bank.seed_features[j],app['background'][j])
                adv[t,side+1]=(((x-x0)/model['scale'])@model['beta']).clamp(-1,1)
    return dict(model=model,advantage=adv,ring_available=app['ring_available'],training_X=X,training_y=y,target_groups=targets,source_groups=sources,
                predictor_scope='Source/anchor features and boxes only; target teacher used only as pseudo response; no dataset GT')


@scoped_single_thread
def rescore_calibrated(bank,cal,p):
    b=copy.copy(bank);b.valid=geometric_valid(bank);model=cal['model']
    for side in (0,1):
        j=bank.source_anchors[:,side];available=(j>=0)&cal['ring_available'][j.clamp_min(0)]
        b.valid[:,side+1]&=available&(bank.side_reliability[:,side]>=.1)
    if not model['valid']:b.valid[:,1:]=False
    margin=p['noise_margin']*model.get('rmse',0.)
    utility=cal['advantage']-margin;utility[:,0]=0
    b.valid[:,1:]&=utility[:,1:]>0
    b.active=b.event&~b.observed&b.valid[:,1:].any(-1)
    n=max(1,int((b.event&~b.observed).sum()));b.score_unary=.5*(b.event&~b.observed)[:,None]*utility/n
    b.score_pair=p['link_multiplier']*bank.score_pair
    r=evidence_dp(b);r['admitted_frames']=int(b.active.sum());return r

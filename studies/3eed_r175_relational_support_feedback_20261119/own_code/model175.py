"""Relational evidence enters recurrent point attention before the final box readout."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from scope175 import CELLS, RELATIONS
from decoder import Decoder, box_distance, gather_slots, in_boxes, mlp, normalized_coordinate, pool

INVERSES = (0,2,1,4,3,5,7,6)


class RelationalDecoder(Decoder):
    def __init__(self, arm, width=96):
        super().__init__('joint',width=width)
        if arm not in CELLS:
            raise ValueError(arm)
        self.arm=arm
        self.use_relation=arm!='self_geometry'
        self.update_support=arm in ('full','without_dual')
        del self.from_ref,self.from_subject,self.pair,self.candidate_proj,self.relation_gate
        self.relation_kind=nn.Embedding(len(RELATIONS),width,padding_idx=0)
        self.role_read=nn.Linear(width,width,bias=False)
        self.pair_score=mlp(3*width+13,width,1)
        self.support_reliability=mlp(2*width+1,width,2)
        nn.init.zeros_(self.pair_score[-1].weight)
        nn.init.zeros_(self.pair_score[-1].bias)
        nn.init.zeros_(self.support_reliability[-1].weight)
        nn.init.constant_(self.support_reliability[-1].bias[0:1],-1.)
        nn.init.zeros_(self.support_reliability[-1].bias[1:])
        self.register_buffer('inverses',torch.as_tensor(INVERSES),persistent=False)

    def pair_belief(self,state,boxes,angles,pixels,visible,unary,phrase,depth,active,same):
        a,b=state.unbind(1)
        sa,sb=boxes.unbind(1)
        delta=sa[:,:,None,:3]-sb[:,None,:,:3]
        dims=(sa[:,:,None,3:6]+sb[:,None,:,3:6])*.5+.25
        shift=normalized_coordinate(delta/dims)
        ratio=sa[:,:,None,3:6].log()-sb[:,None,:,3:6].log()
        joint_visible=visible[:,0,:,None]*visible[:,1,None,:]
        confidence=joint_visible.clamp_min(1e-8).sqrt()*(joint_visible>0)
        projected=pixels[:,0,:,None,:]-pixels[:,1,None,:,:]
        d=normalized_coordinate(depth[:,0,:,None]-depth[:,1,None,:])
        xy=delta[...,:2].norm(dim=-1)/(dims[...,:2].mean(-1)+.25)
        angle=angles[:,0,:,None,:]-angles[:,1,None,:,:]
        geom=torch.cat((shift,ratio,4*projected*confidence[...,None],confidence[...,None],
                        d[...,None],normalized_coordinate(xy)[...,None],angle),-1)
        h1=a[:,:,None].expand(-1,-1,self.slots,-1)
        h2=b[:,None].expand(-1,self.slots,-1,-1)
        text=phrase[:,None,None].expand_as(h1)
        comp=2.*self.pair_score(torch.cat((h1,h2,text,geom),-1)).float().squeeze(-1).tanh()
        joint=unary[:,0,:,None]+unary[:,1,None,:]+active[:,None,None]*comp
        return joint.masked_fill(same & active[:,None,None],-10000.)

    def counterpart_fields(self,points,uv,projected,boxes,pixels,valid_fraction,
                           camera_depth,point_depth,kind,log_scale):
        centre=boxes[...,:3]
        extent=boxes[...,3:6]
        # Role zero reads reference slots; role one uses the inverted relation.
        kind=torch.stack((kind,self.inverses[kind]),1)[:,:,None,None]
        centre=centre.flip(1)
        extent=extent.flip(1)
        counterpart=pixels.flip(1)
        depth=camera_depth.flip(1)
        sharp=log_scale.flip(1).exp()[...,None]
        loc=(uv[:,None,None]-counterpart[...,None,:])
        horizontal=loc[...,0]/(.10*sharp)
        vertical=loc[...,1]/(.10*sharp)
        range_scale=(extent.norm(dim=-1).clamp_min(1.)*sharp.squeeze(-1))[...,None]
        rel_depth=(point_depth[:,None,None]-depth[...,None])/range_scale
        rel_3d=(points[:,None,None]-centre[...,None,:]).norm(dim=-1)/range_scale
        # A bounded signed field distinguishes satisfaction from its inverse.
        result=torch.zeros_like(horizontal)
        result=torch.where(kind==1,-horizontal.tanh(),result)
        result=torch.where(kind==2,horizontal.tanh(),result)
        result=torch.where(kind==3,-vertical.tanh(),result)
        result=torch.where(kind==4,vertical.tanh(),result)
        result=torch.where(kind==5,2.*torch.exp(-rel_3d)-1.,result)
        result=torch.where(kind==6,-rel_depth.tanh(),result)
        result=torch.where(kind==7,rel_depth.tanh(),result)
        screen=(kind>=1)&(kind<=4)
        pixel_support=projected[:,None,None]*(valid_fraction.flip(1)[...,None]/.35).clamp(0,1)
        has_depth=(depth[...,None]>.1)&(point_depth[:,None,None]>.1)
        valid=torch.where(screen,pixel_support,
                          torch.where(kind>=6,has_depth,kind==5))
        return result*valid

    def forward(self,evidence,intervention=None,reference_bind=None):
        if intervention not in (None,'off','shuffled'):
            raise ValueError(intervention)
        points=evidence['xyz']
        uv=evidence['uv'].float()
        projected=evidence['in_image']
        visible=evidence['visible']
        kind=evidence['relation'].long()
        active=(kind!=0)&evidence['has_reference']
        scene,language,anchors,hidden,prior=self.initial(evidence)
        phrase=pool(self.token(evidence['text'].float()),evidence['role_mask'][:,2:3])[:,0]
        phrase=phrase+self.relation_kind(kind)
        query_language=self.role_read(language)
        plane=evidence['depth_plane']
        depth=points @ plane[:,:3,None] + plane[:,None,3:4]
        depth=depth.squeeze(-1)
        keys,values=self.key(scene),self.value(scene)
        offset=(points[:,None,None]-anchors[...,None,:3])/(anchors[...,None,3:6]*.65+.5)
        static=-.65*torch.log1p(offset.square().sum(-1))
        spatial=static
        anchor_ids=evidence['anchors']
        same=anchor_ids[:,0,:,None]==anchor_ids[:,1,None,:]
        passes=[]
        fields=[]
        inject=self.update_support and intervention!='off'
        for t in range(self.turns):
            query=self.query(hidden+language[:,:,None]).flatten(1,2)
            dot=(query @ keys.transpose(1,2))/math.sqrt(query.shape[-1])
            attention=(dot.float().reshape_as(spatial)+spatial).softmax(-1)
            summary=(attention.flatten(1,2) @ values).reshape_as(hidden)
            statistics,centers,pixels,mass=self.geometry(points,attention,uv,visible,anchors)
            update=torch.cat((summary,statistics),-1).flatten(0,2)
            hidden=self.revise(update,hidden.flatten(0,2)).reshape_as(hidden)
            boxes,angles=self.box_readout(anchors,hidden,statistics,centers)
            camera=torch.einsum('brsd,bd->brs',boxes[...,:3],plane[:,:3])+plane[:,3,None,None]
            logits=(hidden*query_language[:,:,None]).sum(-1)/math.sqrt(hidden.shape[-1])
            unary=prior.float()+self.slot_score(hidden).float().squeeze(-1)+.15*logits.float()
            joint=unary[:,0,:,None]+unary[:,1,None,:]
            if self.use_relation:
                joint=self.pair_belief(hidden,boxes,angles,pixels,mass,unary,phrase,camera,active,same)
            m=torch.stack((joint.logsumexp(-1),joint.logsumexp(-2)),1)
            marginal=(torch.where(active[:,None,None],m,unary) if self.use_relation else unary)
            gates=self.support_reliability(torch.cat((hidden,phrase[:,None,None].expand_as(hidden),
                                                     mass[...,None]),-1)).float()
            passes.append((boxes,angles,attention,marginal,joint,unary,gates[...,0]))
            if t+1==self.turns:
                break
            change=(points[:,None,None]-boxes[...,None,:3])/(boxes[...,None,3:6]*.65+.5)
            spatial=static-self.spatial_gate.sigmoid()*torch.log1p(change.square().sum(-1))
            field=torch.zeros_like(attention)
            if inject:
                basis=self.counterpart_fields(points,uv,projected,boxes,pixels,mass,camera,
                                              depth,kind,.65*gates[...,1].tanh())
                # Opposite-role gate is paired with its own spatial field and visibility.
                basis=basis*(2.*gates[...,0].sigmoid().flip(1))[...,None]
                if intervention=='shuffled':
                    basis=basis.roll(1,dims=2)
                subject_routes=joint.softmax(-1)
                if reference_bind is not None:
                    # Diagnostic-only override, used after normal predictions are sealed.
                    known=reference_bind>=0
                    controlled=F.one_hot(reference_bind.clamp_min(0),self.slots).to(subject_routes.dtype)
                    subject_routes=torch.where(known[:,None,None],controlled[:,None],subject_routes)
                probs=torch.stack((subject_routes,joint.transpose(-1,-2).softmax(-1)),1)
                field=(probs @ basis)*active[:,None,None,None]
                spatial=spatial+field
            fields.append(field)
        subject=marginal[:,0].argmax(-1)
        row=torch.arange(len(points),device=subject.device)
        choice_r=joint[row,subject].argmax(-1) if self.use_relation else unary[:,1].argmax(-1)
        choice=torch.stack((subject,choice_r),1)
        return dict(choices=choice,passes=passes,support=gather_slots(attention,choice),
                    field_strength=torch.stack([x.abs().mean((1,2,3)) for x in fields],1))


def source_loss(model,data,out):
    gt=data['gt']
    q=data['quality']
    known_ref=data['known_ref'] if model.arm!='without_dual' else torch.zeros_like(data['known_ref'])
    known=torch.stack((torch.ones_like(known_ref),known_ref),1)
    covered=known & (q.amax(-1)>.1)
    near=box_distance(data['prior'][torch.arange(len(q),device=q.device)[:,None,None],
                                   data['anchors']],gt).argmin(-1)
    slot=torch.where(covered,q.argmax(-1),near)
    positive=q>.1
    target=(q/.12).masked_fill(~positive,-10000.).softmax(-1)
    membership=in_boxes(data['xyz'],gt)
    # GT about the reference is unavailable to the no-dual objective, including overlap masking.
    membership=membership & ~(membership.all(1)&known_ref[:,None])[:,None]
    has_support=known & membership.any(-1)
    truth_yaw=gt[...,6]
    truth_direction=torch.stack(((2*truth_yaw).cos(),(2*truth_yaw).sin()),-1)
    scale=gt[...,3:6].clamp_min(.2)
    aspect=((scale[...,0]-scale[...,1]).abs()/scale[...,:2].amax(-1)).clamp(.15,1.)
    same=data['anchors'][:,0,:,None]==data['anchors'][:,1,None,:]
    pair_target=target[:,0,:,None]*target[:,1,None,:]*(~same)
    pair_target=pair_target/pair_target.sum((1,2),keepdim=True).clamp_min(1e-8)
    pair_ok=covered.all(-1)&(pair_target.sum((1,2))>0.)
    result=0.
    for boxes,angles,weight,role_scores,pair_logits,unary,gate in out['passes']:
        selected=gather_slots(boxes,slot)
        center=(selected[...,:3]-gt[...,:3])/(scale+.25)
        center=F.smooth_l1_loss(center,torch.zeros_like(center),reduction='none',beta=.4).mean(-1)
        extent=F.smooth_l1_loss(selected[...,3:6].log(),scale.log(),
                               reduction='none',beta=.35).mean(-1)
        yaw=aspect*(1.-(gather_slots(angles,slot)*truth_direction).sum(-1))
        p=gather_slots(weight,slot)
        support_loss=-(p*membership).sum(-1).clamp_min(1e-8).log()
        rank_logits=role_scores+.25*data['foil']*known[...,None]
        rank=-(target*rank_logits.log_softmax(-1)).sum(-1)
        objective=(known*(1.5*center+extent+.30*yaw)).sum()/known.sum().clamp_min(1)
        objective+=.85*(covered*rank).sum()/covered.sum().clamp_min(1)
        objective+=.24*(has_support*support_loss).sum()/has_support.sum().clamp_min(1)
        if model.use_relation:
            foils=(data['foil'][:,0,:,None]|data['foil'][:,1,None,:]) & pair_ok[:,None,None]
            pair_logits=pair_logits+.35*foils
            pair=-(pair_target.flatten(1)*pair_logits.flatten(1).log_softmax(-1)).sum(-1)
            objective+=.40*(pair_ok*pair).sum()/pair_ok.sum().clamp_min(1)
        calibrated=F.binary_cross_entropy_with_logits(gate,(q>.1).float(),reduction='none').mean((1,2))
        objective+=.05*(known_ref*calibrated).sum()/known_ref.sum().clamp_min(1)
        result+=objective
    return result/len(out['passes'])

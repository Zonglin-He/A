"""Inference-only component switches for the trained R174 decoder."""
from dataclasses import dataclass
import math

import torch

from scope177 import PROTOCOL
from decoder import normalized_coordinate
from model174 import StrongAnchorDecoder


@dataclass(frozen=True)
class Ablation:
    reference_to_subject: bool = True
    subject_to_reference: bool = True
    final_pair_score: bool = True
    subject_geometry_feedback: bool = True
    reference_geometry_feedback: bool = True
    internal_boxes: str = 'learned'
    pair_projection: bool = True
    predicate_vector: bool = True
    rounds: int = 3
    reference_slots: int = 4
    terminal_messages: bool = True


OPTIONS = {k:Ablation(**v) for k,v in PROTOCOL['arms'].items()}


class ReplayDecoder(StrongAnchorDecoder):
    """Use the same parameters with separately switchable forward paths."""

    def __init__(self):
        super().__init__('full')

    def pair_inputs(self, state, boxes, angles, pixels, mass, active, projected):
        n = self.slots
        subject, reference = state[:, :n],state[:, n:]
        bs,br = boxes[:, :n],boxes[:, n:]
        delta = bs[:, :, None, :3]-br[:, None, :, :3]
        dim_s,dim_r = bs[:, :, None, 3:6],br[:, None, :, 3:6]
        relative = normalized_coordinate(delta/((dim_s+dim_r)*.5+.25))
        ratio = dim_s.log()-dim_r.log()
        if projected:
            image_delta = pixels[:, :n, None] - pixels[:, None, n:]
            shared = mass[:, :n, None]*mass[:, None, n:]
            confidence = shared.clamp_min(1e-8).sqrt()*(shared>0)
            view = torch.cat((4*image_delta*confidence[...,None],confidence[...,None]),-1)
        else:
            view = relative.new_zeros(relative.shape)
        yaw = angles[:, :n, None] - angles[:, None, n:]
        a = subject[:, :, None].expand(-1,-1,reference.shape[1],-1)
        b = reference[:, None].expand(-1,n,-1,-1)
        return self.pair(torch.cat((a,b,relative,ratio,view,yaw),-1)).squeeze(-1)*active[:,None,None]

    def messages(self, state, pair, unaries, active, option):
        n = self.slots
        subject,reference = state[:, :n],state[:, n:]
        if option.reference_to_subject:
            weights = (pair+unaries[:,None,n:]).softmax(-1)
            delta = (weights@self.from_ref(reference))*active[:,None,None]
            subject = subject+self.relation_gate.sigmoid()*delta
        if option.subject_to_reference:
            weights = (pair.transpose(1,2)+unaries[:,None,:n]).softmax(-1)
            delta = (weights@self.from_subject(state[:, :n]))*active[:,None,None]
            reference = reference+self.relation_gate.sigmoid()*delta
        return torch.cat((subject,reference),1)

    def replay(self, evidence, name):
        option = OPTIONS[name]
        n = self.slots
        r = option.reference_slots
        if not option.predicate_vector:
            mask = evidence['role_mask'].clone()
            mask[:,2] = False
            evidence = dict(evidence,role_mask=mask)
        points,uv,active = evidence['xyz'],evidence['uv'].float(),evidence['has_reference']
        scene,role,raw,prior = self.initial(evidence)
        def pack(x):
            return torch.cat((x[:,0],x[:,1,:r]),1)
        anchors = pack(raw)
        prior = pack(prior)
        language = pack(role[:,:,None].expand(-1,-1,n,-1))
        pose = torch.cat((normalized_coordinate(raw[...,:3]),raw[...,3:6].log()),-1)
        state = pack(role[:,:,None]+self.anchor(pose)+self.role_slot[None])
        key,value = self.key(scene),self.value(scene)
        anchor_bias = self.spatial_bias(points,anchors[:,None])[:,0]
        static = .65*anchor_bias
        spatial = static
        anchor_yaw = anchors[...,6]
        cos,sin = (2*anchor_yaw).cos(),(2*anchor_yaw).sin()
        anchor_angle = torch.stack((cos,sin),-1)
        feedback = torch.tensor([option.subject_geometry_feedback]*n+
                                [option.reference_geometry_feedback]*r,device=points.device)
        for step in range(option.rounds):
            final = step+1==option.rounds
            query = self.query(state+language)
            logits = query@key.transpose(1,2)/math.sqrt(query.shape[-1])
            weight = (logits.float()+spatial).softmax(-1)
            summary = weight@value
            stat,centroid,pixels,mass = (x[:,0] for x in self.geometry(
                points,weight[:,None],uv,evidence['visible'],anchors[:,None,:,:6]))
            incoming = torch.cat((summary,stat),-1).flatten(0,1)
            state = self.revise(incoming,state.flatten(0,1)).reshape_as(state)
            if option.internal_boxes == 'anchor':
                box,angle = anchors,anchor_angle
            else:
                box,angle = self.box_readout(anchors[...,:6],state,stat,centroid)
                if option.internal_boxes == 'fixed_yaw':
                    box = torch.cat((box[...,:6],anchor_yaw[...,None]),-1)
                    angle = anchor_angle
                else:
                    c,s = angle.float().unbind(-1)
                    angle = torch.stack((cos*c-sin*s,sin*c+cos*s),-1)
                    box = torch.cat((box[...,:6],(anchor_yaw+box[...,6])[...,None]),-1)
            unaries = prior + .75*self.slot_score(state).float().squeeze(-1).tanh()
            if not final:
                if option.reference_to_subject or option.subject_to_reference:
                    pair = self.pair_inputs(state,box,angle,pixels,mass,active,option.pair_projection)
                    state = self.messages(state,pair,unaries,active,option)
                if option.subject_geometry_feedback or option.reference_geometry_feedback:
                    fitted = (anchor_bias if option.internal_boxes=='anchor' else
                              self.spatial_bias(points,box[:,None])[:,0])
                    spatial = static+self.spatial_gate.sigmoid()*fitted*feedback[None,:,None]
            elif option.final_pair_score:
                pair = self.pair_inputs(state,box,angle,pixels,mass,active,option.pair_projection)
                if option.terminal_messages:
                    self.messages(state,pair,unaries,active,option)
        subject,ref = unaries[:, :n],unaries[:, n:].log_softmax(-1)
        if option.final_pair_score:
            joint = subject[:,:,None]+ref[:,None,:]+.5*pair.float().tanh()
            scores = torch.where(active[:,None],joint.logsumexp(-1),subject)
            chosen = scores.argmax(-1)
            ref_choice = joint[torch.arange(len(chosen),device=chosen.device),chosen].argmax(-1)
        else:
            scores = subject
            chosen = scores.argmax(-1)
            ref_choice = ref.argmax(-1)
        return torch.stack((chosen,ref_choice),-1),scores

"""Bidirectional complete-box propagation, appearance score and box-IoU path."""
import copy
import time
import torch
import torch.nn.functional as F
from .math import (cpu, require, occupancy, affinity,
    fit_axis_affine, transform_native_box, EPS, scoped_single_thread)


def box_links(boxes):
    lo = boxes[..., :2] - boxes[..., 2:]/2
    hi = boxes[..., :2] + boxes[..., 2:]/2
    intersection = (torch.minimum(hi[:-1, :, None], hi[1:, None])
                    - torch.maximum(lo[:-1, :, None], lo[1:, None])).clamp_min(0).prod(-1)
    area = boxes[..., 2:].prod(-1)
    return intersection/(area[:-1, :, None] + area[1:, None] - intersection).clamp_min(EPS)


class SpatialGeometry:
    @scoped_single_thread
    def __init__(self, evidence, E, native_boxes, *, collect_mask_diagnostics=False):
        self.E = cpu(E, torch.float64)
        self.frames = T = len(self.E)
        require(self.E.shape == (T,4) and bool(torch.isfinite(self.E).all())
                and bool((self.E[:,2:] > 0).all()), 'Invalid initialization')
        self.frame_ids = cpu(evidence.frame_ids, torch.long)
        self.positions = list(evidence.positions)
        self.positions8 = self.positions
        self.height, self.width = H, W = evidence.height, evidence.width
        self.native_boxes = native = cpu(native_boxes, torch.float64)
        require(native.shape == (8,4) and bool(torch.isfinite(native).all())
                and bool((native[:,2:] > 0).all()), 'Eight complete native boxes required')
        raw = cpu(evidence.features, torch.float32)
        cells = cpu(evidence.valid, torch.bool)
        seeds = cpu(evidence.seeds, torch.float64)*cells[self.positions].reshape_as(evidence.seeds)
        self.observed = evidence.observed.clone()
        self.source_anchors = torch.full((T,2), -1, dtype=torch.long)
        self.side_available = torch.zeros(T,2, dtype=torch.bool)
        self.side_reference_boxes = self.E[:,None].expand(-1,2,-1).clone()
        self.side_prototypes = torch.zeros(T,2,raw.shape[-1], dtype=torch.float64)
        pooled = (raw[self.positions].double()*seeds.flatten(1)[...,None]).sum(1)/seeds.flatten(1).sum(1).clamp_min(EPS)[:,None]
        self.seed_features = F.normalize(pooled,dim=-1,eps=1e-12)
        self.seed_valid = (seeds.flatten(1).sum(-1)>EPS) & (self.seed_features.norm(dim=-1)>EPS)
        normalized = F.normalize(raw,dim=-1,eps=1e-12)
        yy, xx = torch.meshgrid((torch.arange(H,dtype=torch.float64)+.5)/H,
                               (torch.arange(W,dtype=torch.float64)+.5)/W,indexing='ij')
        xy = torch.stack((xx,yy),-1).reshape(H*W,2)
        order = sorted(range(8),key=lambda j:self.positions[j])
        if collect_mask_diagnostics:
            self.side_maps = torch.zeros(T,2,H,W,dtype=torch.float64)
        for t in range(T):
            if self.observed[t]:
                j = self.positions.index(t)
                neighbors = (j,j)
            else:
                past = [j for j in order if self.positions[j]<t]
                future = [j for j in order if self.positions[j]>t]
                neighbors = (past[-1] if past else None, future[0] if future else None)
            for side,j in enumerate(neighbors):
                if j is None:
                    continue
                a = self.positions[j]
                self.source_anchors[t,side] = j
                self.side_prototypes[t,side] = self.seed_features[j]
                if a == t:
                    self.side_available[t,side] = True
                    self.side_reference_boxes[t,side] = native[j]
                    if collect_mask_diagnostics:
                        self.side_maps[t,side] = seeds[j]
                    continue
                ta = affinity(normalized,cells,t,a)
                at = affinity(normalized,cells,a,t)
                if collect_mask_diagnostics:
                    self.side_maps[t,side] = (ta@seeds[j].flatten()).reshape(H,W)
                fitted,_ = fit_axis_affine(at*ta.T*seeds[j].flatten()[:,None],xy,xy)
                if fitted is not None:
                    box = transform_native_box(native[j],*fitted)
                    if box is not None:
                        self.side_available[t,side] = True
                        self.side_reference_boxes[t,side] = box
        self.boxes = self.E[:,None].expand(-1,3,-1).clone()
        for side in (0,1):
            use = self.side_available[:,side]
            self.boxes[use,side+1] = self.side_reference_boxes[use,side]
        occ,_,_ = occupancy(self.boxes,H,W)
        occ *= cells.reshape(T,1,H,W)
        pooled = torch.bmm(occ.flatten(-2),raw.double())/occ.flatten(-2).sum(-1).clamp_min(EPS)[...,None]
        self.features = F.normalize(pooled,dim=-1,eps=1e-12)
        self.association_scores = torch.einsum('tkd,tsd->tks',self.features,self.side_prototypes).clamp(0,1)
        self._build_scores()

    def _build_scores(self):
        self.valid = torch.cat((torch.ones(self.frames,1,dtype=torch.bool),self.side_available.clone()),1)
        for t in range(self.frames):
            if self.valid[t,1] and self.valid[t,2] and torch.equal(self.boxes[t,1],self.boxes[t,2]):
                self.valid[t,2] = False
        weights = self.side_available.double()
        self.u = (self.association_scores*weights[:,None]).sum(-1)/weights.sum(-1).clamp_min(EPS)[:,None]
        self.supported = self.side_available.any(-1)
        self.link = box_links(self.boxes)
        self.metadata = dict(schema='simplepath_geometry_v1', GT_read=False,
            both_anchors=True, bidirectional_affine=True, cycle=False,
            extent_RBF=False, initialization_guard=False, mask_transport_link=False,
            unary='mean available-anchor appearance cosine',
            link='adjacent unclipped complete-box IoU')

    @classmethod
    @scoped_single_thread
    def from_cached_geometry(cls, state):
        """Reuse only unchanged candidate geometry/content, never the old path."""
        obj = cls.__new__(cls)
        names = ('E','frames','frame_ids','positions','positions8','height','width',
                 'native_boxes','observed','source_anchors','side_available',
                 'side_reference_boxes','side_prototypes','seed_features','seed_valid',
                 'boxes','features','association_scores','side_maps')
        for name in names:
            if name in state:
                value = state[name]
                setattr(obj,name,value.clone() if torch.is_tensor(value) else copy.deepcopy(value))
        obj._build_scores()
        return obj

    def for_interval(self, interval):
        return BoxFixedBank(self,interval=interval)

    def state_dict(self):
        return {k:(v.clone() if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in self.__dict__.items()}


def build_geometry(evidence,E,native_boxes,**kwargs):
    return SpatialGeometry(evidence,E,native_boxes,**kwargs)


class BoxFixedBank:
    @scoped_single_thread
    def __init__(self,geometry,*,interval,mode='interval'):
        require(mode=='interval','Explicit interval required')
        self.__dict__.update(geometry.__dict__)
        self.mode = mode
        require(interval is not None, 'Explicit interval required')
        span = cpu(interval,torch.float64)
        require(span.shape==(2,) and bool(torch.isfinite(span).all()) and span[1]>span[0], 'Invalid interval')
        self.interval = span
        self.event = (self.frame_ids>=span[0]) & (self.frame_ids<span[1])
        self.active = self.event & self.supported
        self.valid = geometry.valid.clone()
        self.valid[~self.active,1:] = False
        observed = self.event & self.observed
        unobserved = self.event & ~self.observed
        self.event_edges = self.event[:-1] & self.event[1:]
        obs_w = .5*observed.double()/max(1,int(observed.sum()))
        unobs_w = .5*self.supported.double()*unobserved/max(1,int(unobserved.sum()))
        self.score_unary = (obs_w+unobs_w)[:,None]*self.u
        edge_w = .25*self.event_edges*self.supported[:-1].double()*self.supported[1:].double()/max(1,int(self.event_edges.sum()))
        self.score_pair = edge_w[:,None,None]*self.link
        self.score_counts = dict(observed=int(observed.sum()),unobserved=int(unobserved.sum()),
                                 event_edges=int(self.event_edges.sum()),all_nodes=self.frames)

    def score_path(self,actions):
        a = cpu(actions,torch.long)
        require(a.shape==(self.frames,) and bool(((a>=0)&(a<3)).all()),'Invalid actions')
        require(not bool(a[~self.active].any()) and bool(self.valid[torch.arange(self.frames),a].all()),'Ineligible action')
        return self.score_unary[torch.arange(self.frames),a].sum()+self.score_pair[torch.arange(self.frames-1),a[:-1],a[1:]].sum()


@scoped_single_thread
def evidence_dp(bank):
    """Legacy operation/tie order, exact global maximum of fixed additive score."""
    begin=time.perf_counter();T=bank.frames
    unary=cpu(bank.score_unary,torch.float64);pair=cpu(bank.score_pair,torch.float64)
    require(unary.shape==(T,3)and pair.shape==(T-1,3,3),'DP score table dimensions changed')
    valid=bank.valid.clone();valid[~bank.active,1:]=False
    values=unary[0].masked_fill(~valid[0],-torch.inf);back=[]
    for t in range(1,T):
        scores=values[:,None]+pair[t-1]+unary[t,None,:]
        values,previous=scores.max(dim=0)
        values=values.masked_fill(~valid[t],-torch.inf);back.append(previous)
    action=int(values.argmax());actions=[action]
    for previous in reversed(back):action=int(previous[action]);actions.append(action)
    actions=torch.tensor(list(reversed(actions)),dtype=torch.long)
    final=bank.boxes[torch.arange(T),actions];score=float(bank.score_path(actions))
    require(abs(score-float(values.max()))<1e-10,'DP decomposition mismatch')
    return dict(final=final,score=score,actions=actions,reliability=bank.supported.clone(),frame_ids=bank.frame_ids.clone(),trace=dict(actions=actions,boxes=final,score=score,
        objective='appearance + box-IoU SimplePath score; exact global maximum, no updates'),
        costs=dict(seconds=time.perf_counter()-begin,updates=0,backward=0,heavy_forwards=0,decoder_calls=0,dynamic_program_states=3*T),
        mode=bank.mode,score_counts=dict(bank.score_counts))



def decode(geometry,*,interval=None,mode='interval'):
    return evidence_dp(BoxFixedBank(geometry,interval=interval,mode=mode))

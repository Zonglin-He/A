"""Complete-box geometry and exact fixed-score trajectory decoding.

Geometry never receives an event interval. Temporal support is an explicit
decode view; full_timeline is a declared extension, not legacy equivalence.
"""
from __future__ import annotations
import copy
import time
import torch
import torch.nn.functional as F
from stvg._spatial_math import (cpu,require,occupancy,soft_dice,affinity,fit_axis_affine,transform_native_box,extent_kernel,EPS,scoped_single_thread)

class SpatialGeometry:
    @scoped_single_thread
    def __init__(self, evidence, E, native_boxes):
            self.E = cpu(E, torch.float64); T = self.frames = len(self.E)
            require(self.E.shape == (T,4) and bool(torch.isfinite(self.E).all()) and bool((self.E[:,2:] > 0).all()), 'Invalid ERef')
            self.frame_ids = cpu(evidence.frame_ids, torch.long)
            require(torch.equal(self.frame_ids, cpu(evidence.frame_ids, torch.long)) and bool((self.frame_ids[1:] > self.frame_ids[:-1]).all()), 'Physical grid changed')
            self.positions = [int(p) for p in evidence.positions]; self.positions8 = self.positions
            require(self.positions == list(evidence.positions) and len(set(self.positions)) == 8
                    and all(0 <= p < T for p in self.positions), 'Eight native observations required')
            native = cpu(native_boxes, torch.float64)
            require(native.shape == (8,4) and bool(torch.isfinite(native).all()) and bool((native[:,2:] > 0).all()), 'Finite positive complete native cxcywh required')
            self.height, self.width = int(evidence.height), int(evidence.width); N = self.height * self.width
            raw = cpu(evidence.features, torch.float32); cells = cpu(evidence.valid, torch.bool)
            require(raw.shape == (T,N,256) and cells.shape == (T,N) and bool(torch.isfinite(raw).all()), 'Source grid changed')
            normalized = F.normalize(raw, dim=-1, eps=1e-12)
            seeds = cpu(evidence.seeds, torch.float64)
            require(seeds.shape == (8,self.height,self.width) and bool(torch.isfinite(seeds).all())
                    and bool(((seeds >= 0) & (seeds <= 1)).all()), 'Selected soft masks required')
            seeds *= cells[self.positions].reshape_as(seeds)
            self.observed = torch.zeros(T,dtype=torch.bool); self.observed[self.positions] = True
            require(torch.equal(self.observed,cpu(evidence.observed,torch.bool)), 'Observed support changed')
            yy,xx = torch.meshgrid((torch.arange(self.height,dtype=torch.float64)+.5)/self.height,
                                   (torch.arange(self.width,dtype=torch.float64)+.5)/self.width,indexing='ij')
            coordinates = torch.stack((xx,yy),-1).reshape(N,2)
            self.native_boxes = native
            self.source_anchors = torch.full((T,2),-1,dtype=torch.long)
            self.anchor_distance = torch.full((T,2),-1.,dtype=torch.float64)
            self.nearest_anchor = torch.empty(T,dtype=torch.long)
            self.side_maps = torch.zeros(T,2,self.height,self.width,dtype=torch.float64)
            self.side_reliability = torch.zeros(T,2,dtype=torch.float64)
            self.side_available = torch.zeros(T,2,dtype=torch.bool)
            self.side_reference_boxes = self.E[:,None].expand(-1,2,-1).clone()
            self.side_prototypes = torch.zeros(T,2,256,dtype=torch.float64)
            self.fit_diagnostics = [[None,None] for _ in range(T)]
            self.seed_features = torch.zeros(8,256,dtype=torch.float64)
            for j,node in enumerate(self.positions):
                mass = seeds[j].flatten()
                pooled = (raw[node].double()*mass[:,None]).sum(0)/mass.sum().clamp_min(EPS)
                self.seed_features[j] = F.normalize(pooled,dim=0,eps=1e-12)
            self.seed_valid = (seeds.flatten(1).sum(-1)>EPS) & (self.seed_features.norm(dim=-1)>EPS)
            order = sorted(range(8),key=lambda j:self.positions[j])
            nearest = lambda t,choices:min(choices,key=lambda j:(abs(int(self.frame_ids[t])-int(self.frame_ids[self.positions[j]])),self.positions[j]))
            self.anchor_features = torch.zeros(T,256,dtype=torch.float64)
            with torch.no_grad():
                for t in range(T):
                    jnear=nearest(t,order); self.nearest_anchor[t]=jnear; self.anchor_features[t]=self.seed_features[jnear]
                    if self.observed[t]: j=self.positions.index(t); neighbors=(j,j)
                    else:
                        past=[j for j in order if self.positions[j]<t]; future=[j for j in order if self.positions[j]>t]
                        neighbors=(past[-1] if past else None,future[0] if future else None)
                    for side,j in enumerate(neighbors):
                        if j is None: continue
                        a=self.positions[j]; self.source_anchors[t,side]=j
                        self.anchor_distance[t,side]=abs(int(self.frame_ids[t])-int(self.frame_ids[a]))
                        self.side_prototypes[t,side]=self.seed_features[j]
                        if a==t:
                            self.side_maps[t,side]=seeds[j]; self.side_reliability[t,side]=1.
                            self.side_available[t,side]=True; self.side_reference_boxes[t,side]=native[j]
                            self.fit_diagnostics[t][side]=dict(valid=True,reason='native_observation_no_fit')
                            continue
                        ta=affinity(normalized,cells,t,a); at=affinity(normalized,cells,a,t)
                        target=ta@seeds[j].flatten(); returned=at@target
                        reliability=float(soft_dice(returned,seeds[j].flatten()))
                        joint=at*ta.T*seeds[j].flatten()[:,None]
                        fitted,info=fit_axis_affine(joint,coordinates,coordinates)
                        self.fit_diagnostics[t][side]=info; self.side_maps[t,side]=target.reshape(self.height,self.width)
                        if fitted is None: continue
                        box=transform_native_box(native[j],*fitted)
                        if box is None:
                            info.update(valid=False,reason='invalid_or_fully_off_image_complete_box'); continue
                        self.side_available[t,side]=True; self.side_reliability[t,side]=reliability
                        self.side_reference_boxes[t,side]=box
                total=self.side_reliability.sum(-1)
                self.confidence=total/self.side_available.sum(-1).clamp_min(1)
                self.merged_targets=(self.side_maps*self.side_reliability[...,None,None]).sum(1)/total.clamp_min(EPS)[:,None,None]
                self.merged_targets*=cells.reshape(T,self.height,self.width)
                self.reliable=(self.confidence>=.1) & self.side_available.any(-1)
                self.boxes=self.E[:,None].expand(-1,3,-1).clone(); self.valid=torch.zeros(T,3,dtype=torch.bool); self.valid[:,0]=True
                self.candidate_origin=torch.full((T,3),-1,dtype=torch.long); self.candidate_quality=torch.zeros(T,3,dtype=torch.float64)
                for t in range(T):
                    for side in (0,1):
                        if self.side_available[t,side]: self.boxes[t,side+1]=self.side_reference_boxes[t,side]
                        if self.side_available[t,side] and (self.reliable[t] or self.observed[t]):
                            self.valid[t,side+1]=True; self.candidate_origin[t,side+1]=-2 if self.observed[t] else side
                            self.candidate_quality[t,side+1]=self.side_reliability[t,side]
                    if self.valid[t,1] and self.valid[t,2] and torch.equal(self.boxes[t,1],self.boxes[t,2]): self.valid[t,2]=False
                occ,_,_=occupancy(self.boxes,self.height,self.width); occ*=cells.reshape(T,1,self.height,self.width)
                pooled=torch.bmm(occ.flatten(-2),raw.double())/occ.flatten(-2).sum(-1).clamp_min(EPS)[...,None]
                self.features=F.normalize(pooled,dim=-1,eps=1e-12); self.E_features=self.features[:,0]
                self.extent_scores=extent_kernel(self.boxes[:,:,None],self.side_reference_boxes[:,None])
                self.association_scores=torch.einsum('tkd,tsd->tks',self.features,self.side_prototypes).clamp(0,1)
                side_score=.5*self.extent_scores+.5*self.association_scores
                self.projection=(side_score*self.side_reliability[:,None]).sum(-1)/total.clamp_min(EPS)[:,None]; self.u=self.projection
                dz=torch.cat(((self.boxes[...,:2]-self.E[:,None,:2])/self.E[:,None,2:],self.boxes[...,2:].log()-self.E[:,None,2:].log()),-1)
                self.guard=-torch.expm1(-(torch.linalg.vector_norm(dz,dim=-1)/2).square())
                masked=(occ*self.merged_targets[:,None]).flatten(-2)
                previous=evidence.transport_adjacent(masked[:-1].transpose(0,1)).transpose(0,1)
                self.link=soft_dice(previous[:,:,None],masked[1:,None])
            self.candidates=self.boxes
            self.metadata=dict(schema='boxfixed_geometry_v1',frames=T,slots=3,grid_height=self.height,grid_width=self.width,
                temporal_interval_used=False,GT_read=False,box_extent='complete native affine extent, no clipping/NMS/components',
                confidence='mean admitted-side return-cycle Dice; no agreement multiplier; not correctness probability')

    def for_interval(self,interval):
        return BoxFixedBank(self,interval=interval,mode='interval')

    def for_full_timeline(self):
        return BoxFixedBank(self,mode='full_timeline')

    def temporal_reliability(self):
        """Existing cycle reliability, zero if no side has a valid mask seed.

        Spatial native-observation confidence remains the historical value1.
        This separate reading-prior reliability gives empty-mask observations0.
        It is neither an event probability nor an identity correctness score.
        """
        sources=self.source_anchors
        valid_seed=self.seed_valid[sources.clamp_min(0)] & (sources>=0)
        supported=(self.side_available & valid_seed).any(-1)
        return self.confidence.clone()*supported

    def state_dict(self):
        return {k:(v.clone()if torch.is_tensor(v)else copy.deepcopy(v))for k,v in self.__dict__.items()}


def build_geometry(evidence,E,native_boxes):
    return SpatialGeometry(evidence,E,native_boxes)


class BoxFixedBank:
    """An interval/full-timeline scoring view on immutable spatial geometry."""
    temporal_reliability=SpatialGeometry.temporal_reliability
    @scoped_single_thread
    def __init__(self,geometry,*,interval=None,mode='interval'):
        require(mode in ('interval','full_timeline'),'Unknown spatial decode mode')
        self.__dict__.update(geometry.__dict__)
        self.mode=mode;self.metadata={**geometry.metadata,'decode_mode':mode}
        if mode=='interval':
            require(interval is not None,'Explicit physical interval required')
            span=cpu(interval,torch.float64)
            require(span.shape==(2,)and bool(torch.isfinite(span).all())and span[1]>span[0],'Explicit physical interval required')
            self.interval=span
            self.event=(self.frame_ids>=span[0])&(self.frame_ids<span[1])
        else:
            require(interval is None,'full_timeline must not receive a temporal interval')
            self.interval=None;self.event=torch.ones(self.frames,dtype=torch.bool)
        self.active=self.event & self.reliable
        self.valid=geometry.valid.clone()
        self.valid[~(self.active|self.observed),1:]=False
        # Historical audit fields retain the second side's provenance/quality
        # after exact duplicate pruning; only temporal ineligibility clears it.
        stored=(self.active|self.observed)[:,None]
        self.candidate_quality=torch.where(stored,geometry.candidate_quality,torch.zeros_like(geometry.candidate_quality))
        self.candidate_origin=torch.where(stored,geometry.candidate_origin,torch.full_like(geometry.candidate_origin,-1))
        self._build_score()
        self.metadata.update(score_denominators=dict(self.score_counts),
            denominator_rule='plain event-observed/event-unobserved/event-adjacent counts, never confidence sums; guard all physical nodes',
            action_rule='reliable event nodes; otherwise E; stored native observation boxes remain available but ineligible outside event')

    def _build_score(self):
            observed = self.event & self.observed; unobserved = self.event & ~self.observed
            self.event_edges = self.event[:-1] & self.event[1:]
            obs_w = .5 * observed.double() / max(1, int(observed.sum()))
            unobs_w = .5 * self.confidence * unobserved / max(1, int(unobserved.sum()))
            self.score_unary = (obs_w + unobs_w)[:, None] * self.projection - .1 * self.guard / self.frames
            edge_w = .25 * self.event_edges * self.confidence[:-1] * self.confidence[1:] / max(1, int(self.event_edges.sum()))
            self.score_pair = edge_w[:, None, None] * self.link
            self.score_counts = dict(observed=int(observed.sum()), unobserved=int(unobserved.sum()),
                                     event_edges=int(self.event_edges.sum()), all_nodes=self.frames)

    def score_path(self,actions):
        a=cpu(actions,torch.long)
        require(a.shape==(self.frames,)and bool(((a>=0)&(a<3)).all()),'One action per physical node required')
        require(not bool(a[~self.active].any())and bool(self.valid[torch.arange(self.frames),a].all()),'Ineligible action')
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
    return dict(final=final,score=score,actions=actions,reliability=bank.temporal_reliability(),frame_ids=bank.frame_ids.clone(),trace=dict(actions=actions,boxes=final,score=score,
        objective='fixed additive BoxFixed score; exact global maximum, no updates'),
        costs=dict(seconds=time.perf_counter()-begin,updates=0,backward=0,heavy_forwards=0,decoder_calls=0,dynamic_program_states=3*T),
        mode=bank.mode,score_counts=dict(bank.score_counts))


def decode(geometry,*,interval=None,mode='interval'):
    return evidence_dp(BoxFixedBank(geometry,interval=interval,mode=mode))

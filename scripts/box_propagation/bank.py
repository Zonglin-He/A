"""Complete native-box propagation from fixed Source correspondences.

Masks weight correspondences, prototypes and reliability; they never set box
boundaries. No model, RGB, file or GT is read. Source affinities retain FP32
top20/.1 arithmetic; geometry, policy support and scores are detached CPU FP64.
"""
from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from scripts.propagation_v0.bank import CandidateBank as V0Bank, cpu, require, occupancy, soft_dice

EPS = 1e-15
SCHEMA = 'complete_native_box_propagation_v1'


def affinity(normalized, valid, target, source):
    ti = torch.nonzero(valid[target], as_tuple=False).flatten()
    si = torch.nonzero(valid[source], as_tuple=False).flatten()
    matrix = torch.zeros(normalized.shape[1], normalized.shape[1], dtype=torch.float64)
    if len(ti) and len(si):
        cosine = normalized[target, ti] @ normalized[source, si].T
        scores, indices = torch.topk(cosine, min(20, len(si)), dim=1, sorted=False)
        weights = (scores / .1).softmax(1).double()
        weights /= weights.sum(1, keepdim=True)
        matrix[ti[:, None].expand_as(indices), si[indices]] = weights
    return matrix


def fit_axis_affine(joint, source_xy, target_xy):
    """Weighted target=s*source+delta, independently for x and y.

    Reliability checks are hard admission tests, never scale clipping. Marginal
    ESS ensures three effective source *and* target locations, rather than
    treating many copies of one cell as independent coordinate information.
    """
    joint = cpu(joint, torch.float64); x = cpu(source_xy, torch.float64); y = cpu(target_xy, torch.float64)
    require(joint.shape == (len(x), len(y)) and x.shape[1:] == y.shape[1:] == (2,), 'WLS coordinate dimensions changed')
    require(bool(torch.isfinite(joint).all()) and bool((joint >= 0).all())
            and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()), 'Finite WLS support required')
    mass = joint.sum(); sw = joint.sum(1); tw = joint.sum(0)
    tiny = torch.finfo(torch.float64).tiny
    source_ess = mass.square() / sw.square().sum().clamp_min(tiny)
    target_ess = mass.square() / tw.square().sum().clamp_min(tiny)
    info = dict(mass=float(mass), source_ESS=float(source_ess), target_ESS=float(target_ess),
                variance=[0., 0.], scale=[0., 0.], delta=[0., 0.], valid=False, reason='insufficient_mass_or_ESS')
    if float(mass) <= EPS or float(source_ess) < 3 or float(target_ess) < 3:
        return None, info
    mx = (sw[:, None] * x).sum(0) / mass; my = (tw[:, None] * y).sum(0) / mass
    xc, yc = x - mx, y - my
    variance = (sw[:, None] * xc.square()).sum(0) / mass
    info['variance'] = variance.tolist()
    if not bool((variance > 1e-6).all()):
        info['reason'] = 'insufficient_source_axis_variance'; return None, info
    covariance = (xc * (joint @ yc)).sum(0) / mass
    scale = covariance / variance; delta = my - scale * mx
    info.update(scale=scale.tolist(), delta=delta.tolist())
    if not bool(torch.isfinite(scale).all()) or not bool(torch.isfinite(delta).all()) or not bool(((scale >= .5) & (scale <= 2.)).all()):
        info['reason'] = 'nonfinite_or_out_of_range_axis_scale'; return None, info
    info.update(valid=True, reason='accepted')
    return (scale, delta), info


def transform_native_box(box, scale, delta):
    box = cpu(box, torch.float64)
    transformed = torch.cat((scale * box[:2] + delta, scale * box[2:]))
    if not bool(torch.isfinite(transformed).all()) or not bool((transformed[2:] > 0).all()): return None
    lo, hi = transformed[:2] - transformed[2:] / 2, transformed[:2] + transformed[2:] / 2
    if not bool(((hi > 0) & (lo < 1)).all()): return None
    # No image-boundary clipping: the entire positive native extent survives.
    return transformed


def extent_kernel(boxes, references):
    """Full-box RBF, centers normalized by each propagated reference's size."""
    relative = torch.cat(((boxes[..., :2] - references[..., :2]) / references[..., 2:],
                          boxes[..., 2:].log() - references[..., 2:].log()), -1)
    return torch.exp(-.5 * relative.square().sum(-1))


class CandidateBank(V0Bank):
    def __init__(self, evidence, cache, cfg):
        require(cache.get('GT_read') is False and evidence.metadata.get('GT_read') is False, 'Label-free evidence required')
        for name, expected in dict(confidence_threshold=.1, affinity_temperature=.1, affinity_topk=20,
                                   max_candidates=3, original_prior=.7).items():
            if name in cfg: require(cfg[name] == expected, 'Locked box-propagation setting changed: ' + name)
        self.E = cpu(cache['E'], torch.float64); T = self.frames = len(self.E)
        require(self.E.shape == (T,4) and bool(torch.isfinite(self.E).all()) and bool((self.E[:,2:] > 0).all()), 'Invalid ERef')
        self.frame_ids = cpu(cache['frame_ids'], torch.long)
        require(torch.equal(self.frame_ids, cpu(evidence.frame_ids, torch.long)) and bool((self.frame_ids[1:] > self.frame_ids[:-1]).all()), 'Physical grid changed')
        self.positions = [int(p) for p in cache['planned_positions']]; self.positions8 = self.positions
        require(self.positions == list(evidence.positions) and len(set(self.positions)) == 8
                and all(0 <= p < T for p in self.positions), 'Eight native observations required')
        require(int(cache['teacher']['selected_query']) == int(evidence.metadata['selected_native_query']), 'Selected teacher identity changed')
        native = cpu(cache['teacher']['boxes'], torch.float64)
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
        span = cpu(cache['interval'],torch.float64)
        self.event = (self.frame_ids >= span[0]) & (self.frame_ids < span[1])
        require(torch.equal(self.event,cpu(evidence.event,torch.bool)), 'Fixed Strong prior changed')
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
        order = sorted(range(8),key=lambda j:self.positions[j]); good = [j for j in order if self.seed_valid[j]]
        nearest = lambda t,choices:min(choices,key=lambda j:(abs(int(self.frame_ids[t])-int(self.frame_ids[self.positions[j]])),self.positions[j]))
        self.initial_memory = self.seed_features[nearest(0,good)].clone() if good else torch.zeros(256,dtype=torch.float64)
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
            self.active=self.event & (self.confidence>=.1) & self.side_available.any(-1)
            self.boxes=self.E[:,None].expand(-1,3,-1).clone(); self.valid=torch.zeros(T,3,dtype=torch.bool); self.valid[:,0]=True
            self.candidate_origin=torch.full((T,3),-1,dtype=torch.long); self.candidate_quality=torch.zeros(T,3,dtype=torch.float64)
            for t in range(T):
                for side in (0,1):
                    if self.side_available[t,side]: self.boxes[t,side+1]=self.side_reference_boxes[t,side]
                    if self.side_available[t,side] and (self.active[t] or self.observed[t]):
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
            alternatives=self.valid[:,1:].sum(-1); self.pi0=torch.zeros(T,3,dtype=torch.float64)
            self.pi0[:,0]=torch.where(alternatives>0,torch.full((T,),.7,dtype=torch.float64),torch.ones(T,dtype=torch.float64))
            self.pi0[:,1:]=self.valid[:,1:]*(.3/alternatives.clamp_min(1).double())[:,None]; self.logprior=self.pi0.log()
            self._build_score()
        self.candidates=self.boxes
        self.metadata=dict(schema=SCHEMA,frames=T,slots=3,grid_height=self.height,grid_width=self.width,
            transform='joint=anchor_softmask*A_anchor_target*A_target_anchor.T; axis weighted OLS target=s*anchor+delta',
            reliability=dict(mass_gt=EPS,source_and_target_marginal_ESS_ge=3,source_axis_variance_gt=1e-6,scale_range_inclusive=[.5,2.],scale_clipped=False,fully_off_image_side_invalid=True),
            box_extent='native continuous center and full width/height transformed; no mask bbox/component/grid snapping/NMS/clipping',
            stored_observations='all eight finite native boxes valid in slot1, including event-external/empty-mask cases',
            action_eligibility='Strong event and c>=.1; inactive action must be E even if observed native is stored',
            confidence='mean admitted side return-cycle Dice; observed=1; no F/B agreement multiplier',
            unary='reliability-weighted average of .5 fullbox RBF + .5 positive ROI/prototype cosine',
            extent_RBF='exp(-.5*sum(((center-center_ref)/size_ref,logsize-logsize_ref)^2)); full propagated reference sizes',
            association='clamp(cos(normalized fractional fullbox ROI Source feature,softmask anchor prototype),0,1); zero prototype gives zero',
            geometry='same v0 bounded guard relative to ERef; fixed score/link/policy unchanged',
            active_nodes=int(self.active.sum()),valid_slots=self.valid.sum(0).tolist(),fit_valid_sides=int(self.side_available.sum()),
            model_calls=0,decoder_calls=0,expert_calls=0,RGB_reads=0,selected_identity_verified=False,confidence_calibrated=False,GT_read=False)
        self._validate_container()

    def _validate_container(self):
        T=self.frames
        for name in ('boxes','features','anchor_features','initial_memory','E','confidence','projection','guard','link','score_unary','score_pair','side_reference_boxes','side_prototypes','extent_scores','association_scores'):
            x=getattr(self,name);require(x.device.type=='cpu' and x.dtype==torch.float64 and not x.requires_grad and bool(torch.isfinite(x).all()),'Finite detached CPU FP64 required: '+name)
        require(self.boxes.shape==(T,3,4) and self.valid.shape==(T,3) and bool((self.boxes[...,2:]>0).all()),'Invalid stored boxes')
        require(torch.equal(self.boxes[:,0],self.E) and bool(self.valid[:,0].all()),'ERef slot changed')
        require(bool(self.valid[self.observed,1].all()) and torch.equal(self.boxes[self.positions,1],self.native_boxes),'Native observations changed')
        require(not bool(self.valid[~self.active & ~self.observed,1:].any()),'Ineligible unobserved candidate')
        require(bool(((self.confidence>=0)&(self.confidence<=1)).all()) and bool(((self.projection>=0)&(self.projection<=1)).all()),'Unbounded confidence/unary')

    def score_path(self, actions):
        a=cpu(actions,torch.long)
        require(a.shape==(self.frames,) and not bool(a[~self.active].any()),'Inactive nodes must select ERef')
        return V0Bank.score_path(self,actions)

    def private_state(self):
        state=super().private_state()
        for name in ('native_boxes','side_reference_boxes','side_prototypes','extent_scores','association_scores'):
            state[name]=getattr(self,name).detach().clone()
        state['fit_diagnostics']=copy.deepcopy(self.fit_diagnostics)
        return state

    @classmethod
    def from_state(cls,state):
        require(state.get('GT_read') is False and state['metadata'].get('GT_read') is False and state['metadata'].get('schema')==SCHEMA,'New complete-box bank schema required')
        obj=cls.__new__(cls);obj.metadata=copy.deepcopy(state['metadata']);obj.frames=T=int(obj.metadata['frames'])
        obj.height=H=int(obj.metadata['grid_height']);obj.width=W=int(obj.metadata['grid_width'])
        obj.positions=[int(p) for p in state['positions8']];obj.positions8=obj.positions
        require(len(set(obj.positions))==8 and all(0<=p<T for p in obj.positions),'Invalid observation positions')
        shapes=dict(boxes=(T,3,4),valid=(T,3),features=(T,3,256),anchor_features=(T,256),E_features=(T,256),E=(T,4),
            confidence=(T,),active=(T,),observed=(T,),frame_ids=(T,),event=(T,),side_maps=(T,2,H,W),side_reliability=(T,2),
            side_available=(T,2),source_anchors=(T,2),anchor_distance=(T,2),nearest_anchor=(T,),merged_targets=(T,H,W),
            candidate_quality=(T,3),candidate_origin=(T,3),projection=(T,3),guard=(T,3),link=(T-1,3,3),pi0=(T,3),logprior=(T,3),
            initial_memory=(256,),seed_features=(8,256),seed_valid=(8,),score_unary=(T,3),score_pair=(T-1,3,3),
            native_boxes=(8,4),side_reference_boxes=(T,2,4),side_prototypes=(T,2,256),extent_scores=(T,3,2),association_scores=(T,3,2))
        boolean={'valid','active','observed','event','side_available','seed_valid'};integer={'frame_ids','source_anchors','nearest_anchor','candidate_origin'}
        for name,shape in shapes.items():
            dtype=torch.bool if name in boolean else torch.long if name in integer else torch.float64
            value=cpu(state[name],dtype);require(value.shape==shape,'Saved shape changed: '+name)
            if dtype==torch.float64 and name!='logprior':require(bool(torch.isfinite(value).all()),'Nonfinite saved bank: '+name)
            setattr(obj,name,value)
        obj.fit_diagnostics=copy.deepcopy(state['fit_diagnostics']);require(len(obj.fit_diagnostics)==T,'Missing WLS audit')
        obj.score_counts=copy.deepcopy(state['score_counts']);obj.event_edges=obj.event[:-1]&obj.event[1:]
        require(torch.equal(obj.logprior,obj.pi0.log()) and torch.allclose(obj.pi0.sum(-1),torch.ones(T,dtype=torch.float64),atol=1e-12,rtol=0),'Saved prior changed')
        obj.u=obj.projection;obj.candidates=obj.boxes;obj._validate_container();return obj

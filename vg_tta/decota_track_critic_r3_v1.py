"""Exact cached path posterior. IoU continuity is not an identity guarantee."""
import itertools,math
import torch
from vg_tta.decota_optimizer_posterior_r1_v1 import Energy
from vg_tta.tastvg_decota_critic_p0_v1 import aligned_iou
from methods.decota_final_simplified_v1.objectives import generalized_iou

class TrackEnergy(Energy):
    def __init__(self,expert,boxes,ids,similarity='iou',native_prior=True,continuity=True):
        super().__init__(expert,boxes,'all')
        self.frames=sorted(self.frames,key=lambda f:(ids[f[0]],f[0]))
        self.similarity=similarity
        if self.empty:self.logpath=None;self.paths=[];self.path_authority=0.;return
        self.paths=list(itertools.product(*[range(len(f[1])) for f in self.frames]));assert len(self.paths)<=81
        self.path_index=torch.tensor(self.paths,device=boxes.device,dtype=torch.long)
        with torch.no_grad():
            value=boxes.new_zeros(len(self.paths))
            for t,(pos,ev,logw) in enumerate(self.frames):
                ix=self.path_index[:,t]
                value=value+logw[ix]
                if native_prior:value=value+aligned_iou(boxes[pos],ev)[ix]
                if continuity and t:value=value+aligned_iou(self.frames[t-1][1][self.path_index[:,t-1]],ev[ix])
            self.logpath=torch.log_softmax(value,0).detach()
            h=float(-(self.logpath.exp()*self.logpath).sum())
            self.path_authority=1. if len(self.paths)==1 else max(0.,min(1.,1-h/math.log(len(self.paths))))
        self.frame_metadata=[dict(position=p,proposals=len(e)) for p,e,_ in self.frames]

    def __call__(self,boxes):
        if self.empty:return boxes.sum()*0
        sim=aligned_iou if self.similarity=='iou' else generalized_iou
        reward=boxes.new_zeros(len(self.paths))
        for t,(pos,ev,_) in enumerate(self.frames):reward=reward+sim(boxes[pos],ev)[self.path_index[:,t]]
        return -torch.logsumexp(self.logpath+reward,0)

    def metadata(self):
        return dict(path_count=len(self.paths),path_authority=self.path_authority,
            probabilities=[] if self.empty else self.logpath.exp().cpu().tolist(),
            positions=[f[0] for f in self.frames],native_prior_fixed=True,continuity_is_proxy=True,
            energy_reduction='sum observed-frame similarities inside logsumexp')

class FrameSum(Energy):
    def __init__(self,expert,boxes):super().__init__(expert,boxes,'all')
    def __call__(self,boxes):return super().__call__(boxes)*len(self.frames)

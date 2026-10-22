"""Same spatial evidence mathematics, with explicit native channel dimension."""
import torch
import torch.nn.functional as F
from tools.boxfixed_tube.evidence import SourceEvidence as OriginalEvidence,_require,_sparsify
from tools.boxfixed_tube._spatial_math import cpu,affinity,scoped_single_thread

class SourceEvidence(OriginalEvidence):
    @scoped_single_thread
    def __init__(self,features,valid,frame_ids,positions8,seeds,*,selected_query=None):
        self.features=cpu(features,torch.float32);self.valid=cpu(valid,torch.bool);self.frame_ids=cpu(frame_ids,torch.long);self.seeds=cpu(seeds,torch.float64);self.positions=[int(x)for x in positions8];self.positions8=self.positions
        _require(self.seeds.ndim==3 and len(self.seeds)==8,'Eight Source-grid soft masks required');self.height,self.width=self.seeds.shape[-2:];T,N,D=self.features.shape
        _require(D>0 and N==self.height*self.width and self.valid.shape==(T,N),'Native feature grid mismatch')
        _require(self.frame_ids.shape==(T,)and bool((self.frame_ids[1:]>self.frame_ids[:-1]).all()),'Increasing physical frames required')
        _require(len(set(self.positions))==8 and all(0<=p<T for p in self.positions),'Eight distinct positions required')
        _require(bool(torch.isfinite(self.features).all())and bool(torch.isfinite(self.seeds).all()),'Finite evidence required');_require(bool(((self.seeds>=0)&(self.seeds<=1)).all()),'Masks must be soft probabilities')
        self.seeds*=self.valid[self.positions].reshape_as(self.seeds);self.observed=torch.zeros(T,dtype=torch.bool);self.observed[self.positions]=True
        normalized=F.normalize(self.features,dim=-1,eps=1e-12);forward=torch.stack([affinity(normalized,self.valid,t,t-1)for t in range(1,T)]);self.adjacent_forward_indices,self.adjacent_forward_weights=_sparsify(forward)
        self.metadata=dict(schema='boxfixed_source_evidence_v1',GT_read=False,selected_native_query=selected_query,grid_height=self.height,grid_width=self.width,temporal_interval_used=False,affinity_topk=20,affinity_temperature=.1,source_features='native LLaVA-ST first spatial STP grid;3584channels,no fabricated256 projection')
        self.GT_read=False

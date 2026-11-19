"""Expose components of the frozen K=16 v1 readout without changing its weights."""

from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
R180 = HERE.parent/'3eed_v2_subject_retrieval_20261119'
sys.path.insert(0,str(R180))
from retrieval180 import Observation, anchored_order, decoder_order, load_retriever
from features import encode
from mechanisms import CandidateGeometry, ReadoutTrial, diverse_order
from crossplatform.ground import Grounder

K = 16
ARMS = ('v1','frozen_joint','frozen_subject','frozen_mismatch',
        'calibrated_subject','calibrated_joint','calibrated_mismatch')
TERMS = ('prior','subject_independent','subject_joint','relation_joint','subject_mismatch','relation_mismatch')


class ComponentReader(ReadoutTrial):
    def _read(self,*args):
        output = super()._read(*args)
        self._unaries = output[1]
        return output

    def _pair_for(self,*args):
        self._last_pair = super()._pair_for(*args)
        return self._last_pair

    def inspect(self,evidence):
        picked = int(self.trial(evidence)[0])
        p = evidence['base_logit'][0]
        full_unary = self._unaries[0,:K]
        u = full_unary-p
        relation = torch.zeros_like(u)
        score = full_unary
        if bool(evidence['has_reference'][0]):
            ref = self._unaries[0,K:].log_softmax(-1)
            pair = .5*self._last_pair[0].float().tanh()
            relation = (ref[None]+pair).logsumexp(-1)
            score = (full_unary[:,None]+ref[None]+pair).logsumexp(-1)
        assert picked == int(score.argmax())
        error = float((p+u+relation-score).abs().max())
        assert error < 5e-6
        return u,relation,score,picked,error

    def components(self,evidence):
        u,r,frozen,a,error = self.inspect(evidence)
        if bool(evidence['has_reference'][0]):
            separate = dict(evidence, has_reference=torch.zeros_like(evidence['has_reference']))
            independent,_,subject,b,e = self.inspect(separate)
            e1 = dict(evidence,ref_logit=evidence['ref_logit'].roll(1,dims=-1))
            shifted_u,shifted_r,mismatch,c,e2 = self.inspect(e1)
            error = max(error,e,e2)
        else:
            independent,subject,b = u,frozen,a
            shifted_u,shifted_r,mismatch,c = u,r,frozen,a
        terms = torch.stack((evidence['base_logit'][0],independent,u,r,shifted_u,shifted_r),dim=-1)
        scores = torch.stack((frozen,subject,mismatch),dim=-1)
        return terms.cpu().numpy(),scores.cpu().numpy(),np.array([a,b,c],dtype=np.int8),error


class ReadoutGrounder(Grounder):
    """Inspect one fixed K=16 slate or reproduce source K=16 retrieval for a new frame."""

    def __init__(self,source,*,asset_root=None):
        super().__init__(source,asset_root=asset_root)
        self.reader = ComponentReader().cuda().eval().requires_grad_(False)
        self.reader.load_state_dict(self._decoder.state_dict(),strict=True)

    @torch.inference_mode()
    def collect(self,frame,*,fixed=None,ranking=None):
        from grounding158 import select_grounding_box

        with self._lock:
            if fixed is None and ranking is None:
                capture = Observation(self)
                feature,_ = encode(capture,frame)
                scene,evidence,baseline = capture.scored,capture.evidence,capture.v1
                if not hasattr(self,'retriever'):
                    self.retriever = load_retriever(self._frontend.source)
                dense = {name:torch.as_tensor(feature[name],device=evidence['xyz'].device)[None]
                         for name in ('anchor_query','token_proj','anchors','prior')}
                ranking = self.retriever(dense)[0].cpu().numpy()
            else:
                scene,evidence = self._read_observation(frame)
                baseline = self._decoder(evidence)
            first,utility = select_grounding_box(scene.boxes,scene.probability)
            geometry = CandidateGeometry(scene.boxes,scene.sampled_points)
            base_slots = np.asarray(diverse_order(utility,geometry.overlap,4))
            np.testing.assert_array_equal(np.stack([geometry.orient(j) for j in base_slots]),
                                          evidence['subject_boxes'][0].cpu().numpy())
            base_slot = int(baseline['choices'][0,0])
            incumbent = int(base_slots[base_slot])
            if fixed is None:
                order = anchored_order(ranking,geometry.overlap,incumbent)
                slate = decoder_order(order,base_slots,np.argsort(-utility,kind='stable'))
            else:
                slate = np.asarray(fixed,dtype=np.int64)
            assert len(set(slate))==K and slate[base_slot]==incumbent
            boxes = np.stack([geometry.orient(j) for j in slate])
            raw_prior = np.log(np.maximum(utility,1e-80))
            raw_prior = (raw_prior-raw_prior[first]).clip(-20.,0.).astype(np.float32)
            np.testing.assert_array_equal(raw_prior[base_slots],evidence['base_logit'][0].cpu().numpy())
            dev = evidence['xyz'].device
            inputs = dict(evidence,subject_boxes=torch.as_tensor(boxes[None],device=dev),
                          base_logit=torch.as_tensor(raw_prior[slate][None],device=dev))
            terms,scores,selected,error = self.reader.components(inputs)
            active = bool(evidence['has_reference'][0])
            can_win = raw_prior[slate]>=raw_prior[incumbent]-(2.5 if active else 1.5)-1e-6
            return dict(slate=slate,incumbent=base_slot,base_slots=base_slots,boxes=boxes,terms=terms,
                        scores=scores,chosen=selected,view=scene.view_index,has_reference=active,
                        reconstruction_error=error,can_win=can_win)

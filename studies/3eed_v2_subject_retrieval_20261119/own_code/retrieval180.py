"""Source-trained subject shortlists with frozen v1 decoding and raw A256 boxes."""

from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
sys.path.insert(0,str(PRIOR))
from scope import file_sha, read
from features import encode
from mechanisms import ReadoutTrial, diverse_order
from crossplatform.ground import Grounder
from crossplatform.candidate_frontend import sensor_frame_box

NATIVE = 256
K_VALUES = (4,8,16)
ARMS = ('v1','native4','native8','native16','trained4','trained8','trained16',
        'native_source_K','trained_source_K')
TRAINING = HERE/'source_PRIVATE'


class SubjectRetriever(nn.Module):
    """Learn a small subject residual on top of the v1 overlap prior."""

    def __init__(self):
        super().__init__()
        self.rank = nn.Sequential(nn.Linear(196,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.rank[-1].weight)
        nn.init.zeros_(self.rank[-1].bias)

    def forward(self, feature):
        query = F.normalize(feature['anchor_query'].float(),dim=-1)
        tokens = F.normalize(feature['token_proj'].float(),dim=-1)
        shape = feature['anchors'][...,3:6].clamp_min(.05).log()
        prior = feature['prior'].float()
        x = torch.cat((query,8.*query*tokens[:,0,None],8.*query*tokens[:,3,None],
                       shape*.5,prior[:, :, None]*.25),dim=-1)
        return prior+4.*self.rank(x).squeeze(-1).tanh()


def load_retriever(source):
    pin = read(TRAINING/'FITTED.json')
    path = TRAINING/f'{source}.pth'
    assert file_sha(path)==pin['sources'][source]['weights_sha256'] and pin['source_only']
    saved = torch.load(path,weights_only=True,map_location='cpu')
    assert saved['source']==source
    model = SubjectRetriever().cuda().eval().requires_grad_(False)
    model.load_state_dict(saved['state'],strict=True)
    return model


class Observation:
    """Expose only observation and frozen-decoder captures to the shared encoder."""

    def __init__(self, grounder):
        self.grounder = grounder

    def _read_observation(self, frame, *, capture):
        self.scored, self.evidence = self.grounder._read_observation(frame,capture=capture)
        return self.scored,self.evidence

    def _decoder(self, evidence):
        self.v1 = self.grounder._decoder(evidence)
        return self.v1


def anchored_order(score, overlap, incumbent):
    seeded = np.array(score,dtype=np.float64,copy=True)
    seeded[incumbent] = seeded.max()+1.
    result = diverse_order(seeded,overlap,max(K_VALUES))
    assert result[0]==incumbent and len(set(result))==max(K_VALUES)
    return result


def decoder_order(members, v1_slots, native_order):
    """Preserve slots of retained incumbents and fill vacancies by frozen order."""
    kept = set(members)
    placed = np.full((len(members),),-1,dtype=np.int64)
    for slot,index in enumerate(v1_slots):
        if index in kept:
            placed[slot] = index
            kept.remove(index)
    placed[placed<0] = [i for i in native_order if i in kept]
    return placed


class RetrievalGrounder(Grounder):
    """Compare fixed-K subject sets using the existing frozen wider v1 readout."""

    def __init__(self, source, *, asset_root=None):
        super().__init__(source,asset_root=asset_root)
        self.retriever = load_retriever(source)
        self.reader = ReadoutTrial().cuda().eval().requires_grad_(False)
        self.reader.load_state_dict(self._decoder.state_dict(),strict=True)

    @torch.inference_mode()
    def compare(self, frame, ks=K_VALUES):
        from grounding158 import select_grounding_box
        from reader158 import overlaps

        with self._lock:
            capture = Observation(self)
            feature,_ = encode(capture,frame)
            scene, evidence, baseline = capture.scored,capture.evidence,capture.v1
            first, utility = select_grounding_box(scene.boxes,scene.probability)
            overlap = overlaps(scene.boxes)
            anchors = feature['anchors']
            v1_slots = np.array(diverse_order(utility,overlap,4),dtype=np.int64)
            np.testing.assert_array_equal(anchors[v1_slots],evidence['subject_boxes'][0].cpu().numpy())
            incumbent_slot = int(baseline['choices'][0,0])
            incumbent = int(v1_slots[incumbent_slot])
            np.testing.assert_array_equal(anchors[incumbent],feature['v1_box'])
            if first!=v1_slots[0]:
                raise RuntimeError('Native candidate ranking no longer matches v1 slots')
            device = evidence['xyz'].device
            x = {k:torch.as_tensor(feature[k],device=device)[None]
                 for k in ('anchor_query','token_proj','prior','anchors')}
            native_order = np.argsort(-utility,kind='stable').tolist()
            native = anchored_order(utility,overlap,incumbent)
            learned = anchored_order(self.retriever(x)[0].cpu().numpy(),overlap,incumbent)
            raw_prior = np.log(np.maximum(utility,1e-80))
            raw_prior = (raw_prior-raw_prior[first]).clip(-20.,0.).astype(np.float32)
            np.testing.assert_array_equal(raw_prior[v1_slots],evidence['base_logit'][0].cpu().numpy())
            bound = 2.5 if bool(evidence['has_reference'][0]) else 1.5
            can_win = (raw_prior >= raw_prior[incumbent]-bound-1e-6)
            trials = {'v1':(v1_slots,incumbent)}
            memo = {tuple(v1_slots):incumbent}
            for name, sequence in (('native',native),('trained',learned)):
                for k in ks:
                    indices = decoder_order(sequence[:k],v1_slots,native_order)
                    assert indices[incumbent_slot]==incumbent and len(set(indices))==k
                    identity = tuple(indices)
                    if identity not in memo:
                        inputs = dict(evidence,
                            subject_boxes=torch.as_tensor(anchors[indices][None],device=device),
                            base_logit=torch.as_tensor(raw_prior[indices][None],device=device))
                        memo[identity] = int(indices[int(self.reader.trial(inputs)[0])])
                    winner = memo[identity]
                    assert can_win[winner]
                    trials[name+str(k)] = (indices,winner)
        return anchors,trials,can_win,scene.view_index

    def predict(self, *, k=None, retrieval='trained', **frame):
        if retrieval not in ('native','trained'):
            raise ValueError(f'Unknown retrieval {retrieval}')
        if k is None:
            selection = read(TRAINING/'SELECTION.json')
            assert selection['fitted_sha256']==file_sha(TRAINING/'FITTED.json')
            k = selection['sources'][self._frontend.source]['selected_k']
        if k not in K_VALUES:
            raise ValueError(f'Unsupported K {k}')
        pool,trials,_,view = self.compare(frame,ks=(k,))
        shortlist,winner = trials[retrieval+str(k)]
        box = sensor_frame_box(pool[winner],frame['target'],view)
        return dict(box=box.tolist(),k=k,retrieval=retrieval,
                    incumbent_in_shortlist=bool(trials['v1'][1] in shortlist),
                    frame='target_lidar',length_unit='m',yaw_unit='rad')

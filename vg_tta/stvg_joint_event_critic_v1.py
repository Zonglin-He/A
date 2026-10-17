"""Source-supervised, target-frozen visual tube critic; no dataset/GT readers.

All spatial states are actual Student parameter states. Temporal proposals use
only Source boundary logits. Outside-interval frames remain in every sequence.
"""
import math
import numpy as np
import torch
from torch import nn
from methods.decota_final_simplified_v1.objectives import legal_logp
from methods.decota_final_simplified_v1.observations import uniform_positions


def temporal_candidates(logits, records, ids, native_interval, k=3):
    """Native envelope plus highest-probability distinct observation hypotheses.

    A uniform mixture of the two original legal offset distributions is summed
    by physical interval. Different Uniform4 frame sets enforce distinct inputs;
    there is no Actionness, expansion ratio, GT, or duration tuning.
    """
    mass = {}
    for z, record in zip(logits, records):
        lp, ij = legal_logp(z.detach().cpu())
        for p, (s, e) in zip((lp.exp()/2).tolist(), ij.T.tolist()):
            pair = (int(record['frame_ids'][s]), int(record['frame_ids'][e])+1)
            mass[pair] = mass.get(pair, 0.) + p
    chosen = [list(map(int, native_interval))]
    def fingerprint(interval):
        a = ids.index(interval[0]); b = ids.index(interval[1]-1)
        return tuple(uniform_positions(ids, [a,b], 4))
    seen = {fingerprint(chosen[0])}
    for pair in sorted(mass, key=lambda p: (-mass[p], p)):
        fp = fingerprint(pair)
        if fp not in seen:
            chosen.append(list(pair)); seen.add(fp)
        if len(chosen) == k:
            break
    if len(chosen) != k:
        raise ValueError('Insufficient distinct Native hypotheses; preserve case')
    return dict(intervals=chosen, observation_fingerprints=[list(fingerprint(i)) for i in chosen],
                construction='Native envelope + top offset-mixture legal intervals with distinct Uniform4 inputs',
                prior='uniform over joint hypotheses; no calibrated confidence claim', Actionness_used=False)


@torch.no_grad()
def extract_visual(base, boxes, ids, fps):
    """Frozen source visual maps, ROIAlign target and full-frame context.

    Appearance/motion maps are taken before any adapted LN/QR state, so the
    critic compares trajectories on identical evidence. Full-query tokens are
    retained rather than replaced by noun strings or scalar confidence.
    """
    from torchvision.ops import roi_align
    assert boxes.shape == (len(ids),4) and torch.isfinite(boxes).all()
    seq = [None]*len(ids); tokens=[]; outside=[]
    for off, view in enumerate(base.views):
        H = base.norm(view['prefix']).detach().float()
        h,w = view['info']['fea_map_size']; n=h*w; nf=H.shape[1]
        appearance = H[:n].permute(1,2,0).reshape(nf,256,h,w)
        motion = H[-n:].permute(1,2,0).reshape(nf,256,h,w)
        query = H[n:-n].mean(1)
        mask = view['info'].get('text_padding_mask')
        # The native encoder may store text mask under another key; H text
        # already follows its actual full-query tokenization. Keep all tokens.
        assert query.ndim == 2 and len(query)>0
        tokens.append(query.cpu())
        b=boxes[off::2].to(H); xy=torch.cat((b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2),1)
        # Only ROI sampling intersects the image. Student boxes are unchanged.
        visible=xy.clamp(0,1); outside.append(int((xy!=visible).any(1).sum()))
        valid=(visible[:,2:]>visible[:,:2]).all(1)
        scale=H.new_tensor([w,h,w,h]); rois=torch.cat((torch.arange(nf,device=H.device)[:,None],visible*scale),1)
        a=roi_align(appearance,rois,output_size=(2,2),spatial_scale=1.,sampling_ratio=2,aligned=True).mean((-1,-2))
        m=roi_align(motion,rois,output_size=(2,2),spatial_scale=1.,sampling_ratio=2,aligned=True).mean((-1,-2))
        a[~valid]=0; m[~valid]=0
        f=torch.cat((a,m,appearance.mean((-1,-2)),motion.mean((-1,-2))),1).cpu()
        for j in range(nf):seq[off+2*j]=f[j]
    assert all(v is not None for v in seq)
    # Both offsets use the same full Query, possibly different contextual text
    # representations; averaging tokens does not drop any Query token.
    assert tokens[0].shape==tokens[1].shape
    return dict(visual=torch.stack(seq),query=(tokens[0]+tokens[1])/2,
                boxes=boxes.detach().cpu(),times=torch.tensor(ids,dtype=torch.float32)/float(fps),
                roi_clipped_frames=sum(outside),GT_input=False,
                frozen_visual_pre_adaptation=True,all_context_frames_retained=True)


def candidate_tensor(feature, interval, fps):
    t=feature['times']; a,b=np.asarray(interval,dtype=float)/float(fps)
    duration=max(float(t[-1]-t[0])+1/float(fps),1/float(fps))
    rel=(t-t[0])/duration
    # Membership is discrete in the original frame grid. A float32 timestamp
    # can round below its own double-precision interval boundary. All saved
    # frame IDs are independently checked to round-trip exactly before use.
    frame_ids=torch.round(t.double()*float(fps)).long()
    inside=((frame_ids>=int(interval[0]))&(frame_ids<int(interval[1]))).float()
    positions=torch.stack((rel,(t-a)/duration,(t-b)/duration,inside,
                           torch.full_like(t,math.log1p(duration))),1)
    return torch.cat((feature['visual'],feature['boxes'],positions),1).float(),feature['query'].float()


class EventCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual=nn.Sequential(nn.LayerNorm(1033),nn.Linear(1033,64),nn.GELU())
        self.text=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,64))
        self.cross=nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(64,4,128,dropout=0.,
                        activation='gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.score=nn.Sequential(nn.LayerNorm(128),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))

    def forward(self, x, q, padding=None, text_padding=None):
        v=self.visual(x); txt=self.text(q)
        c,_=self.cross(v,txt,txt,key_padding_mask=text_padding,need_weights=False)
        h=self.temporal(v+c,src_key_padding_mask=padding)
        mask=torch.ones_like(x[...,0]) if padding is None else (~padding).float()
        context=(h*mask[...,None]).sum(1)/mask.sum(1)[:,None]
        inside=mask*x[...,-2]
        local=(h*inside[...,None]).sum(1)/inside.sum(1).clamp_min(1)[:,None]
        return self.score(torch.cat((local,context),1)).squeeze(-1)


def collate(examples):
    n=max(len(x[0]) for x in examples); m=max(len(x[1]) for x in examples)
    xs=torch.zeros(len(examples),n,1033); qs=torch.zeros(len(examples),m,256)
    pm=torch.ones(len(examples),n,dtype=torch.bool); qm=torch.ones(len(examples),m,dtype=torch.bool)
    for i,(x,q) in enumerate(examples):
        xs[i,:len(x)]=x;qs[i,:len(q)]=q;pm[i,:len(x)]=False;qm[i,:len(q)]=False
    return xs,qs,pm,qm


def ranking_loss(scores, labels):
    d=labels[:,None]-labels[None,:];mask=d>1e-8
    if not mask.any():return scores.sum()*0,0
    return torch.nn.functional.softplus(-(scores[:,None]-scores[None,:])[mask]).mean(),int(mask.sum())


def pair_accuracy(scores, labels):
    s=np.asarray(scores);y=np.asarray(labels);mask=(y[:,None]-y[None,:])>1e-8
    if not mask.any():return None,0
    d=(s[:,None]-s[None,:])[mask]
    return float(((d>0)+.5*(d==0)).mean()),len(d)

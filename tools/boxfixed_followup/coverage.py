"""Scoring/admission on existing geometric candidates before old cycle masking."""
import copy
import torch
import torch.nn.functional as F
from tools.boxfixed_tube._spatial_math import occupancy,EPS,scoped_single_thread
from tools.boxfixed_tube.spatial import evidence_dp
from tools.boxfixed_followup.selection import signals,rescore


@scoped_single_thread
def appearance(bank,evidence):
    """Same frozen Source features; native-box outer ring supplies background.

    Background is outside the complete native teacher box, not outside its
    foreground soft mask. Weak masks must not label object boundary as negative.
    Empty local rings fall back to valid image cells outside the native box.
    Empty fallback remains unavailable; no artificial negative is invented.
    """
    bg=[];available=[]
    for j,a in enumerate(bank.positions):
        box=bank.native_boxes[j];outer=box.clone();outer[2:]*=2
        native=occupancy(box,evidence.height,evidence.width)[0].flatten()
        ring=(occupancy(outer,evidence.height,evidence.width)[0].flatten()-native).clamp_min(0)
        valid=evidence.valid[a].double();mass=ring*valid
        if mass.sum()<=EPS:mass=(1-native)*valid
        pooled=(evidence.features[a].double()*mass[:,None]).sum(0)/mass.sum().clamp_min(EPS)
        v=bool(mass.sum()>EPS and pooled.norm()>EPS and bank.seed_valid[j])
        bg.append(F.normalize(pooled,dim=0,eps=1e-12));available.append(v)
    bg=torch.stack(bg);ok=torch.tensor(available,dtype=torch.bool)
    anchors=bank.source_anchors.clamp_min(0)
    difference=bank.side_prototypes-bg[anchors]
    # Both foreground and background prototypes are normalized BEFORE their
    # subtraction. Never normalize a tiny difference into a large fake signal.
    score=torch.einsum('tkd,tsd->tks',bank.features,difference)
    adv=score-score[:,0:1]
    cross=torch.zeros_like(bank.projection);own=torch.zeros_like(cross)
    cross[:,1]=adv[:,1,1];cross[:,2]=adv[:,2,0]
    own[:,1]=adv[:,1,0];own[:,2]=adv[:,2,1]
    return dict(cross=cross,own=own,valid=(ok[anchors]&(bank.source_anchors>=0)).all(-1),background=bg,ring_available=ok)


def geometric_valid(bank):
    valid=torch.zeros_like(bank.valid);valid[:,0]=True
    valid[:,1:]=bank.side_available&(bank.event&~bank.observed)[:,None]
    valid[:,2]&=~(valid[:,1]&(bank.boxes[:,1]==bank.boxes[:,2]).all(-1))
    return valid


@scoped_single_thread
def rescore_coverage(bank,app,p):
    if p['policy']=='previous':return rescore(bank,p['selection'])
    assert p['policy']=='expanded'
    b=copy.copy(bank);b.valid=geometric_valid(bank);s=signals(bank)
    blend=p['contrast'];adv=(1-blend)*s['advantage']+blend*app['cross']
    permitted=s['independent']&(s['joint_reliability']>=p['cycle_floor'])
    if blend>0:permitted &= app['valid']
    for k in (1,2):
        b.valid[:,k]&=permitted&(adv[:,k]>0)
        if blend>0:b.valid[:,k]&=app['own'][:,k]>0
    b.active=b.event&~b.observed&b.valid[:,1:].any(-1)
    n=max(1,int((b.event&~b.observed).sum()))
    w=.5*(b.event&~b.observed).double()/n
    if p['unary_cycle_weight']:w=w*s['joint_reliability']
    b.score_unary=w[:,None]*adv-p['guard']*b.guard/b.frames
    r=evidence_dp(b)
    r.update(admitted_frames=int(b.active.sum()),available_geometric_frames=int((geometric_valid(bank)[:,1:].any(-1)).sum()))
    return r

"""Fixed-candidate evidence scoring and admission, with no label/model access."""
import copy
import torch
from tools.boxfixed_tube.spatial import evidence_dp
from tools.boxfixed_tube._spatial_math import EPS, scoped_single_thread


def signals(bank):
    """Compare every candidate and ERef against the SAME independent anchor.

    Side1 is proposed by the preceding anchor and evaluated by the following
    anchor; side2 uses the converse. At native observations both stored sides
    are the same anchor, so they cannot corroborate one another.
    """
    side=.5*bank.extent_scores+.5*bank.association_scores
    adv=torch.zeros_like(bank.projection)
    adv[:,1]=side[:,1,1]-side[:,0,1]
    adv[:,2]=side[:,2,0]-side[:,0,0]
    independent=(bank.source_anchors[:,0]>=0)&(bank.source_anchors[:,1]>=0)&(bank.source_anchors[:,0]!=bank.source_anchors[:,1])
    independent &= bank.side_available.all(-1)
    own_bonus=torch.zeros_like(adv)
    own_bonus[:,1]=side[:,1,0]-side[:,0,0]
    own_bonus[:,2]=side[:,2,1]-side[:,0,1]
    joint_reliability=bank.side_reliability.min(-1).values
    # Symmetric extent agreement uses only the two existing candidate boxes.
    # A large value is agreement, not a probability of correct identity.
    agreement=(bank.extent_scores[:,1,1]*bank.extent_scores[:,2,0]).sqrt()
    return dict(advantage=adv,own_advantage=own_bonus,independent=independent,
                joint_reliability=joint_reliability,agreement=agreement)


@scoped_single_thread
def rescore(bank,parameters):
    b=copy.copy(bank);b.valid=bank.valid.clone();b.metadata=dict(bank.metadata)
    mode=parameters['selection_mode']
    if mode=='original':
        return evidence_dp(b)
    # The teacher is still used as the propagation evidence. Merely matching
    # its own observed box is not independent grounds for replacing Source.
    b.valid[b.observed,1:]=False
    if mode=='observe_protection':
        return evidence_dp(b)
    assert mode in ['cross_score','cross_gate','cross_agreement']
    s=signals(b)
    valid=s['independent']&(s['joint_reliability']>=parameters['cycle_floor'])
    margin=parameters['margin']
    for k in (1,2):
        b.valid[:,k]&=valid&(s['advantage'][:,k]>margin)
    if mode=='cross_agreement':
        b.valid[:,1:]&=(s['agreement']>=parameters['agreement_floor'])[:,None]
    if mode!='cross_gate':
        # A candidate must provide positive independent evidence over ERef.
        # Reward and protection use the original node/edge denominators.
        n=max(1,int((b.event&~b.observed).sum()))
        w=.5*s['joint_reliability']*(b.event&~b.observed)/n
        b.score_unary=w[:,None]*s['advantage']-parameters['guard']*b.guard/b.frames
    r=evidence_dp(b)
    r['independent_admitted_nodes']=int((b.valid[:,1:].any(-1)&b.active).sum())
    return r

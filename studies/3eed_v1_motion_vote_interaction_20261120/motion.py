from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
DUEL=HERE.parent/'3eed_v1_pretruncation_physical_duels_20261120'
BIAS=HERE.parent/'3eed_v1_paired_bias_distillation_20261120'
sys.path[:0]=[str(DUEL),str(BIAS)]
from common import PREV, anon, archive_npz, atomic, digest, engine, legacy, np, read, setup, torch
from paired import overlaps
from roles import CANONICAL,nominal
from scipy.special import logsumexp

WORK=HERE/'run_PRIVATE'
RESULTS=HERE/'results'
ARMS=tuple(read(HERE/'PROTOCOL.json')['arms'])


def initialize():
    setup()
    WORK.mkdir(exist_ok=True,mode=0o700)
    RESULTS.mkdir(exist_ok=True)


def category(front,caption):
    from parse97 import dependency_document, parse_subject
    subject=parse_subject(caption,dependency_document(front.parser(caption)))
    label=nominal(subject['head_text']) if subject['status']=='FOUND' else None
    return 'a '+CANONICAL[label] if label in CANONICAL else None


@dataclass
class Views:
    full: object
    ev: dict
    neutral: object
    permutation: object


def seed_permutation(frame):
    # No annotation, private identifier or target performance participates in this permutation.
    order=np.argsort([anon(int(frame['seed']),frame['text'],i) for i in range(256)])
    p=np.empty(256,dtype=int)
    p[order]=np.roll(order,1)
    assert (p!=np.arange(256)).all()
    return p


@torch.inference_mode()
def observations(model,frame,parity=False):
    front=model._frontend
    raw=[]
    with front.base.register_forward_hook(lambda _m,args,out:raw.append(out)):
        full,ev=model._read_observation(frame)
    original,=raw
    noun=category(front,frame['text'])
    perm=seed_permutation(frame)
    if noun is None:
        return Views(full,ev,None,perm)
    cached={k:original[k] for k in ('fp2_features','fp2_xyz','fp2_inds')}
    ids=original['query_points_sample_inds']
    assert len(ids[0].unique())==256
    def fixed(_mod,args):
        # Both the native grounder and the prepared point matcher read these coordinates.
        assert torch.equal(args[0],cached['fp2_xyz'])
        return args[0],args[1],ids.to(dtype=args[2].dtype)
    def verify_seed(_m,_args,out):
        assert torch.equal(out['query_points_sample_inds'],ids)
        assert torch.equal(out['query_points_xyz'],original['query_points_xyz'])
    with ExitStack() as stack:
        stack.enter_context(patch.object(front.base.backbone_net,'forward',return_value=dict(cached)))
        stack.enter_context(front.base.gsample_module.register_forward_pre_hook(fixed))
        stack.enter_context(front.base.register_forward_hook(verify_seed))
        args={k:frame[k] for k in ('target','points','seed','waymo_view')}
        if parity:
            repeated=front.score_frame(text=frame['text'],**args)
            np.testing.assert_array_equal(repeated.boxes,full.boxes)
            np.testing.assert_allclose(repeated.probability,full.probability,rtol=0,atol=1e-12)
        neutral=front.score_frame(text=noun,**args)
    np.testing.assert_array_equal(full.sampled_points,neutral.sampled_points)
    return Views(full,ev,neutral,perm)


def kernel(anchors,boxes):
    a=np.array(anchors[:,:6],dtype=float)
    b=np.array(boxes[:,:6],dtype=float)
    a[:,3:]=a[:,3:].clip(min=0)
    b[:,3:]=b[:,3:].clip(min=0)
    low=np.maximum((a[:,:3]-a[:,3:]/2)[:,None],b[:,:3]-b[:,3:]/2)
    high=np.minimum((a[:,:3]+a[:,3:]/2)[:,None],b[:,:3]+b[:,3:]/2)
    inter=(high-low).clip(min=0).prod(-1)
    union=a[:,None,3:].prod(-1)+b[None,:,3:].prod(-1)-inter
    k=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
    return k/np.square(overlaps(b)).sum(1)[None,:]


def channels(anchors,view):
    if view.neutral is None:
        return np.zeros((len(anchors),3)),np.zeros(len(anchors),dtype=bool)
    f,c=view.full,view.neutral
    pf=np.maximum(f.probability,1e-300)
    pc=np.maximum(c.probability,1e-300)
    k_f,k_c=kernel(anchors,f.boxes),kernel(anchors,c.boxes)
    log_ratio=np.log(pf)-np.log(pc)
    shuffled=np.log(pc)+log_ratio[view.permutation]
    pstar=np.exp(shuffled-logsumexp(shuffled))
    weights=np.stack((pf,pc,pstar)).T
    votes=np.stack((k_f@weights,k_c@weights))
    good=(votes>0).all(axis=(0,2))
    logv=np.zeros_like(votes)
    logv[:,good]=np.log(votes[:,good])
    relative=logv[0]-logv[1]
    value=np.stack((relative[:,0]-relative[:,1],relative[:,2]-relative[:,1],logv[1,:,1]),axis=-1)
    if good.any():
        value[good]-=value[good].mean(0)
    return value,good


def adjust(base,stat,mask,scale):
    j=np.asarray(base,dtype=float).copy()
    if scale==0 or mask.sum()<2:
        return j
    original=j[mask]
    new=original-scale*stat[mask]
    j[mask]=new+(logsumexp(original)-logsumexp(new))
    return j


def choose_scores(base,stat,ok,coefs):
    return dict(v1=base,remove_interaction=adjust(base,stat[:,0],ok,1.),
        source_interaction=adjust(base,stat[:,0],ok,coefs[0]),
        permuted_interaction=adjust(base,stat[:,1],ok,coefs[1]),
        category_prior=adjust(base,stat[:,2],ok,coefs[2]))

"""Freeze full-sentence winners, both ordered queries, and all final boxes before scoring."""
import fcntl
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAM=HERE.parent/'3eed_v1_camera_word_replacement_20261120'
sys.path[:0]=[str(ROOT),str(CAM),str(ROOT/'research/round6/source_validation_scoring'),
              str(HERE.parent/'3eed_v1_retrieval_binding_refinement_20261119')]
from roles_joint import ARMS, decide, posterior
from predict import archive_npz, frame
from calibrate import digest,read,save
from mechanisms import CandidateGeometry
from crossplatform.ground import Grounder
from crossplatform.v1_evidence import reference_anchors

PRIVATE=HERE/'run_PRIVATE'
OUT=HERE/'results'
CONFIG=read(HERE/'PROTOCOL.json')
PROXY=HERE.parent/'codex_overnight_20261119/private/binding/source/ROSTER.json'
SOURCES=('waymo','drone','quad')
PANELS={'source_proxy':PROXY,'natural170':ROOT/'crossplatform/round170/INPUT_PRIVATE.json',
        'natural171':ROOT/'crossplatform/round171/INPUT_PRIVATE.json'}


def observation_list():
    """Project historical rosters onto observation fields, with labels excluded."""
    records=[]
    for panel,path in PANELS.items():
        cfg=CONFIG['panels'][panel]
        assert digest(path)==cfg.get('roster_sha256',cfg.get('source_inventory_sha256'))
        entries=read(path)
        assert len(entries)==cfg['frames']
        for i,entry in enumerate(entries):
            row=entry['row']
            inp=frame(row)
            rewrite=entry['rewrite']
            views={tag:rewrite[tag] for tag in ('B','D')} if rewrite['status']=='supported' else None
            for source in SOURCES:
                if panel=='source_proxy' or source!=inp['target']:
                    records.append(dict(panel=panel,case=i,source=source,frame=inp,views=views))
    assert len(records)==501
    return records


def compute_joint(decoder,evidence):
    saved={}
    read_impl,pair_impl=decoder._read,decoder._pair_scores
    def read_last(*args,**kwargs):
        out=read_impl(*args,**kwargs)
        saved['unaries']=out[1]
        return out
    def pair_last(*args,**kwargs):
        out=pair_impl(*args,**kwargs)
        saved['pair']=out
        return out
    with patch.object(decoder,'_read',read_last),patch.object(decoder,'_pair_scores',pair_last):
        out=decoder(evidence)
    k=decoder.slots
    unary=saved['unaries']
    joint=unary[:,:k,None]+unary[:,None,k:].log_softmax(-1)+.5*saved['pair'].float().tanh()
    assert evidence['has_reference'].all()
    torch.testing.assert_close(joint.logsumexp(-1),out['subject_logits'],rtol=0.,atol=1e-6)
    return joint[0].double().cpu().numpy()


def canonical(model,inp,view,anchors=None):
    data={}
    scene,ev=model._read_observation(dict(inp,text=view['text']),capture=data)
    start,end=data['offsets'].T
    pos=view['predicate_pos']
    predicate='left' if view['sign']==1 else 'right'
    assert view['text'][pos:pos+len(predicate)]==predicate
    spans=[*view['spans'],(pos,pos+len(predicate))]
    masks=np.stack([(end>start)&(start<hi)&(end>lo)&data['valid'] for lo,hi in spans])
    assert masks.any(1).all()
    n=data['native']
    ref_boxes,ref_logits=reference_anchors(scene.boxes,n['queries'],n['tokens'],masks,True,scene.sampled_points)
    device=n['xyz'].device
    ev=dict(ev,reference_boxes=torch.as_tensor(ref_boxes[None],device=device),ref_logit=ref_logits,
            role_mask=torch.as_tensor(masks[None],device=device),has_reference=torch.tensor([True],device=device))
    if anchors is not None:
        ev.update(subject_boxes=anchors['subject_boxes'],base_logit=anchors['base_logit'])
    joint=compute_joint(model._decoder,ev)
    sub,ref=[ev[tag][0].double().cpu().numpy() for tag in ('subject_boxes','reference_boxes')]
    return scene,sub,ref,joint


def original_pool(scene):
    geo=CandidateGeometry(scene.boxes,scene.sampled_points)
    return np.stack([geo.orient(i) for i in range(len(scene.boxes))]).astype(float)


@torch.inference_mode()
def observe(model,record):
    inp=record['frame']
    scene,ev=model._read_observation(inp)
    baseline=model._decoder(ev)
    s=ev['subject_boxes'][0].double().cpu().numpy()
    native=original_pool(scene)
    original=baseline['subject_logits'][0].double().cpu().numpy()
    base_i,base_j=baseline['choices'][0].cpu().numpy()
    baseline_ref=ev['reference_boxes'][0,base_j].double().cpu().numpy()
    if not bool(ev['has_reference'][0]):
        baseline_ref=np.full(7,np.nan)
    assert all(np.any(np.all(native==b,axis=1)) for b in s)
    if record['views'] is None:
        base=np.tile(s[base_i],(len(ARMS),1))
        return dict(boxes=base,refs=np.tile(baseline_ref,(len(ARMS),1)),raw=base,slate=s,
                    choices=np.full(len(ARMS),base_i),fraction=np.zeros(len(ARMS)),pool=native,other_pools=np.empty((0,7)),
                    scores=np.tile(original,(len(ARMS),1)),active=False)
    sb,sf,rf,jf=canonical(model,inp,record['views']['B'],ev)
    sd,rev_ref,rev_sub,jd=canonical(model,inp,record['views']['D'])
    np.testing.assert_array_equal(scene.sampled_points,sb.sampled_points)
    np.testing.assert_array_equal(scene.sampled_points,sd.sampled_points)
    np.testing.assert_array_equal(sf,s)
    orig,conditional,reverse=posterior(original,jf,jd)
    arrays=[decide(orig,s,rf,conditional,rev_sub,rev_ref,reverse,arm) for arm in ARMS]
    picks=np.array([v[0] for v in arrays])
    assert picks[0]==base_i and picks[-2]==base_i and picks[-1]==picks[-3]
    refs=np.stack([rf[v[1]] for v in arrays])
    refs[0]=baseline_ref
    return dict(boxes=np.stack([v[2] for v in arrays]),refs=refs,raw=s[picks],slate=s,
                choices=picks,fraction=np.array([v[3] for v in arrays]),scores=np.stack([v[4] for v in arrays]),
                pool=native,other_pools=np.vstack((original_pool(sb),original_pool(sd))),active=True)


def run():
    PRIVATE.mkdir(exist_ok=True,mode=0o700)
    OUT.mkdir(exist_ok=True)
    assert tuple(CONFIG['arms'])==ARMS
    assert digest(ROOT/'crossplatform/single_frame_sources.json')==CONFIG['source_assets_sha256']
    cfg=dict(arms=list(ARMS),protocol=digest(HERE/'PROTOCOL.json'),code={f:digest(HERE/f) for f in ('capture_roles.py','roles_joint.py')},
             source_assets=CONFIG['source_assets_sha256'])
    if (PRIVATE/'LOCK.json').exists():
        assert read(PRIVATE/'LOCK.json')==cfg
    else:
        save(PRIVATE/'LOCK.json',cfg)
    lock_hash=digest(PRIVATE/'LOCK.json')
    observations=observation_list()
    receipts={}
    start=perf_counter()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for src in SOURCES:
            model=None
            for record in (r for r in observations if r['source']==src):
                tag=f"{record['panel']}_{record['case']:03d}_{src}.npz"
                path=PRIVATE/tag
                if not path.exists():
                    if model is None:
                        model=Grounder(src)
                        model._frontend.parser=lru_cache(maxsize=1024)(model._frontend.parser)
                    pack=observe(model,record)
                    archive_npz(path,**pack,lock=lock_hash)
                with np.load(path,allow_pickle=False) as prior:
                    assert str(prior['lock'])==lock_hash
                receipts[tag]=digest(path)
                if len(receipts)%50==0:
                    print('captured',len(receipts),'/',len(observations),round(perf_counter()-start,1),'s',flush=True)
            del model
            torch.cuda.empty_cache()
    save(PRIVATE/'SEAL.json',dict(files=receipts,n=len(receipts),lock=lock_hash,
         original_role_labels_used_for_prediction=False,target_updates=0,FINAL_access=False,elapsed=perf_counter()-start))


if __name__=='__main__':
    run()

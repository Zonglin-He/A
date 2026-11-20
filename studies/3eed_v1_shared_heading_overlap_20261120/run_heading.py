"""Score the ordinary rosters using only the caption, sensor observation and source weights."""
import argparse
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from heading import ARMS, HERE, nominate
ROOT=HERE.parents[1]
CAMERA=HERE.parent/'3eed_v1_camera_word_replacement_20261120'
sys.path[:0]=[str(CAMERA),str(HERE.parent/'3eed_v1_retrieval_binding_refinement_20261119'),str(ROOT)]
from predict import archive_npz, frame, inputs, ordinary_baseline
from calibrate import digest, read, save
from mechanisms import CandidateGeometry
from crossplatform.ground import Grounder
from crossplatform.candidate_frontend import sensor_frame_box

CFG=read(HERE/'PROTOCOL.json')
WORK=HERE/'run_PRIVATE'
OUT=HERE/'results'
BLOCK=64


def roster_hash(rows):
    return hashlib.sha256('\n'.join(f"{r['pointpath']}|{r['metapath']}|{r['caption']}|{r.get('seed',0)}"
                                    for r in rows).encode()).hexdigest()


@torch.inference_mode()
def observe(grounder,inp):
    from reader158 import overlaps
    scene,ev=grounder._read_observation(inp)
    start=perf_counter()
    native=np.asarray(scene.boxes)
    geo=CandidateGeometry(native,scene.sampled_points)
    oriented=np.stack([geo.orient(i) for i in range(len(native))])
    pool=np.stack([sensor_frame_box(b,scene.target,scene.view_index) for b in oriented])
    pca_time=perf_counter()-start
    start=perf_counter()
    slates,priors=nominate(native,oriented,scene.probability,geo.overlap)
    kernel_time=perf_counter()-start
    assert np.array_equal(oriented[slates[0]],ev['subject_boxes'][0].cpu().numpy())
    np.testing.assert_array_equal(priors[0],ev['base_logit'][0].cpu().numpy())
    original=grounder._decoder(ev)['subject_logits'][0]
    device=original.device
    value={k:v.expand(len(ARMS),*v.shape[1:]) for k,v in ev.items()}
    value.update(subject_boxes=torch.as_tensor(oriented[slates],device=device),
                 base_logit=torch.as_tensor(priors,device=device))
    out=grounder._decoder(value)['subject_logits'].cpu().numpy()
    frozen=original.cpu().numpy()
    err=float(np.max(np.abs(out[0]-frozen)))
    if err>2e-5:
        raise ValueError(f'Batched four-slot decoder differs from registered readout: {err}')
    # Resolve close logits using the original one-record execution shape.
    top=np.sort(out,axis=1)[:,-2:]
    for j in np.flatnonzero(top[:,1]-top[:,0]<=1e-4):
        single=dict(ev,subject_boxes=value['subject_boxes'][j:j+1],base_logit=value['base_logit'][j:j+1])
        out[j]=grounder._decoder(single)['subject_logits'][0].cpu().numpy()
    out[0]=frozen
    winners=slates[np.arange(len(ARMS)),out.argmax(1)]
    assert len(native)==256 and all(len(set(x.tolist()))==4 for x in slates)
    if not (np.isfinite(out).all() and np.isfinite(pool).all()): raise ValueError('Non-finite output')
    return dict(pool=pool, ids=slates.astype(np.int16), priors=priors, logits=out,
                winners=winners.astype(np.int16), replay_error=err,
                n_yaw=int(np.count_nonzero(oriented[:,6])), times=[pca_time,kernel_time])


def run(stage,smoke=False):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if not (OUT/'geometry_audit.json').is_file():raise ValueError('Run check_geometry.py first')
    if stage=='ordinary' and not read(OUT/'pilot_decision.json')['expand']:
        raise ValueError('Pilot expansion condition not met')
    assert list(ARMS)==CFG['arms'] and digest(ROOT/'crossplatform/single_frame_sources.json')==CFG['sources_sha256']
    roster=inputs(stage)
    root=WORK/stage
    root.mkdir(parents=True,exist_ok=True,mode=0o700)
    lock=dict(arms=list(ARMS),stage=stage,assets=CFG['sources_sha256'],
              roster={f'{s}_to_{t}':roster_hash(rows) for (s,t),rows in roster.items()},
              code={k:digest(HERE/k) for k in ('PROTOCOL.json','run_heading.py','heading.py','overlap.cpp')},
              FINAL_access=False)
    if stage!='ordinary':
        previous=CAMERA/'run_PRIVATE'/stage
        old=read(previous/'SEALED.json')
        assert digest(previous/'SEALED.json')==CFG['scope_'+stage+'_seal'] and old['lock']==digest(previous/'LOCK.json')
        assert read(previous/'LOCK.json')['roster']==lock['roster']
    if not smoke:
        if (root/'LOCK.json').is_file():
            assert lock==read(root/'LOCK.json')
        else:save(root/'LOCK.json',lock)
    grounder,last=None,None
    files={}
    start_time=perf_counter()
    for (s,t),rows in roster.items():
        direction=f'{s}_to_{t}'
        folder=root/direction
        folder.mkdir(exist_ok=True)
        for start in range(0,len(rows),BLOCK):
            if smoke and start:break
            batch=rows[start:start+BLOCK]
            name=f'{direction}/{start:05d}.npz'
            dest=root/name;pin=dest.with_suffix('.json')
            if smoke or not pin.exists():
                if s!=last:
                    if grounder is not None:del grounder;torch.cuda.empty_cache()
                    grounder=Grounder(s)
                    grounder._frontend.parser=lru_cache(maxsize=2048)(grounder._frontend.parser)
                    last=s
                if stage=='ordinary':
                    baseline_slots, choices=ordinary_baseline(direction,start,len(batch))
                    baseline=baseline_slots[np.arange(len(batch)),choices]
                else:
                    assert digest(previous/name)==old['files'][name]['sha256']
                    with np.load(previous/name,allow_pickle=False) as z:
                        baseline_slots=z['boxes'];baseline=baseline_slots[np.arange(len(batch)),z['choices'][:,0]]
                data=[]
                for j,r in enumerate(batch[:1] if smoke else batch):
                    d=observe(grounder,frame(r))
                    np.testing.assert_array_equal(d['pool'][d['ids'][0]],baseline_slots[j])
                    np.testing.assert_array_equal(d['pool'][d['winners'][0]],baseline[j])
                    data.append(d)
                if smoke:
                    print('smoke',direction,'arms',len(ARMS),'kernel_ms',1000*d['times'][1],
                          'batch_logit_error',d['replay_error'],flush=True)
                    continue
                archive_npz(dest,**{k:np.stack([d[k] for d in data]) for k in data[0]})
                save(pin,dict(n=len(batch),sha256=digest(dest),lock=digest(root/'LOCK.json')))
            if not smoke:
                checked=read(pin)
                assert checked['n']==len(batch) and checked['sha256']==digest(dest) and checked['lock']==digest(root/'LOCK.json')
                files[name]=checked
                print(stage,direction,start+len(batch),'/',len(rows),round(perf_counter()-start_time,1),flush=True)
    if not smoke:
        n=sum(entry['n'] for entry in files.values())
        assert n=={'source':384,'pilot':3072,'ordinary':18620}[stage]
        save(root/'SEALED.json',dict(n=n,lock=digest(root/'LOCK.json'),files=files,
                                    elapsed=perf_counter()-start_time,task_labels_opened=False,FINAL_access=False))


if __name__=='__main__':
    args=argparse.ArgumentParser();args.add_argument('stage',choices=('source','pilot','ordinary'))
    args.add_argument('--smoke',action='store_true');opt=args.parse_args()
    run(opt.stage,opt.smoke)

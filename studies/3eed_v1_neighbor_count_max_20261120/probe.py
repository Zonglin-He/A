"""Freeze K16 decisions under controlled return deletion on each checkpoint's own source."""
from collections import Counter
import fcntl
from functools import lru_cache
from time import perf_counter

import numpy as np
import torch

from native_pool import CFG,HERE,MASKS,OUT,PLATFORMS,ROOT,WORK,atomic,capture_case,digest,legacy,membership,pinned_rows,read,save_npz


@torch.inference_mode()
def measure(grounder,row,rank,box,score):
    pool,result,slate,mask,odds=capture_case(grounder,row,rank)
    raw_score=result['scores'][:,0]
    np.testing.assert_array_equal(result['boxes'],box)
    np.testing.assert_array_equal(raw_score,score)
    base=int(score.argmax())
    assert result['chosen'][0]==base
    members=membership(pool.points[pool.centers].cpu().numpy(),box)
    count=pool.count.cpu().numpy()
    has=pool.has_center.cpu().numpy()
    scopes=np.stack((np.ones(2048,dtype=bool),members.any(0),members[base]))
    hist=np.stack([[np.bincount(count[m],minlength=17) for m in (s,s&has)] for s in scopes])
    values,loss,marginal,timing=[],[],[],[]
    for name in MASKS:
        alt,n=pool.thinning(name,row['point_sha256'])
        torch.cuda.synchronize()
        t=perf_counter()
        pred=pool.replay(alt,slate,mask)
        torch.cuda.synchronize()
        timing.append(perf_counter()-t)
        delta=(pred.double()-odds.double()).cpu().numpy()
        if name=='native':
            np.testing.assert_array_equal(delta,np.zeros_like(delta))
        squared=(pool.native-alt).square().mean(1).cpu().numpy()
        loss.append([float(squared[scope].sum()) for scope in scopes])
        marginal.append(float(abs(delta).max()))
        values.append(score.astype(float)+delta)
    return dict(scores=np.stack(values),hist=hist,loss_sum=np.array(loss),max_score_shift=np.array(marginal),
                seconds=np.array(timing),scope_sizes=scopes.sum(1),v1=result['incumbent'],
                seed_ids=pool.raw['query_points_sample_inds'][0].cpu().numpy(),slate=result['slate'])


def run():
    dest=WORK/'deletion'
    dest.mkdir(mode=0o700,parents=True,exist_ok=True)
    OUT.mkdir(exist_ok=True)
    legacy.prepare_device()
    assert digest(ROOT/'crossplatform/single_frame_sources.json')==CFG['source_registration_sha256']
    lock=dict(protocol=digest(HERE/'PROTOCOL.json'),code={f:digest(HERE/f) for f in ('native_pool.py','probe.py')},
              checkpoint_registry=CFG['source_registration_sha256'],arms=list(MASKS))
    if (dest/'LOCK.json').exists():
        assert read(dest/'LOCK.json')==lock
    else:
        atomic(dest/'LOCK.json',lock)
    tag=digest(dest/'LOCK.json')
    weights={}
    manifest={}
    counts={}
    tick=perf_counter()
    with (ROOT/'research/GPU_RESOURCE.lock').open('a+') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for src in PLATFORMS:
            grounder=legacy.ReadoutGrounder(src)
            grounder._frontend.parser=lru_cache(maxsize=512)(grounder._frontend.parser)
            from model51 import fingerprint
            previous=fingerprint(grounder._frontend.base)
            ranks=legacy.source_ranks(src)
            boxes=legacy.sealed_arrays(legacy.PAIR,'seal.json','source_k16',src,['boxes'])['boxes']
            scores=legacy.sealed_arrays(legacy.R181,'SEALED.json','source_scores',src,['scores'])['scores'][:,:,0]
            fit,rows=pinned_rows(src)
            assert len(rows)==len(ranks)==len(boxes)==len(scores)
            counts[src]=dict(fit_rows=len(fit),diagnostic=len(rows),
                             fit_recordings=len(Counter(r['recording'] for r in fit)),
                             diagnostic_recordings=len(Counter(r['recording'] for r in rows)))
            for i,row in enumerate(rows):
                name=f'{src}_{i:04d}.npz'
                path=dest/name
                if not path.exists():
                    save_npz(path,**measure(grounder,row,ranks[i],boxes[i],scores[i]),lock=tag)
                with np.load(path,allow_pickle=False) as z:
                    assert str(z['lock'])==tag
                    np.testing.assert_array_equal(z['scores'][0],scores[i])
                manifest[name]=digest(path)
                if len(manifest)%32==0:
                    print('sealed decisions',len(manifest),'/ 384',round(perf_counter()-tick,1),'s',flush=True)
            assert fingerprint(grounder._frontend.base)==previous
            weights[src]=previous
            del grounder
            torch.cuda.empty_cache()
    assert len(manifest)==384
    atomic(dest/'SEALED.json',dict(files=manifest,lock=tag,source_weights=weights,counts=counts,
           seconds=perf_counter()-tick,grounding_labels_requested=False,target_data=False,FINAL=False))


if __name__=='__main__':
    run()

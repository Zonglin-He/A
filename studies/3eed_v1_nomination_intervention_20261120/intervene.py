from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

HERE=Path(__file__).resolve().parent
DUEL=HERE.parent/'3eed_v1_pretruncation_physical_duels_20261120'
BIAS=HERE.parent/'3eed_v1_paired_bias_distillation_20261120'
sys.path[:0]=[str(DUEL),str(BIAS)]
from common import WORK as PARENT_WORK, anon, archive_npz, atomic, digest, engine, fixed_prior, np, read, setup, torch
from fit_duels import loaded
from source_features import common_nominations, native_boxes, ownership
from source_pairs import catalog,source_identity
from scope import canonical, source_rows
from features import observation
from scope183 import frame
from fit import write_csv

WORK=HERE/'run_PRIVATE'
RESULTS=HERE/'results'
EDGES=list(combinations(range(4),2))
PERMS=[np.arange(4)]
for i,j in EDGES:
    p=np.arange(4)
    p[i],p[j]=j,i
    PERMS.append(p)
PERMS=np.stack(PERMS)


@torch.inference_mode()
def replay(decoder,fixed,scene,data):
    device=data['subject_boxes'].device
    prior,_=fixed_prior(fixed,scene)
    evidence={k:v.expand(7,*v.shape[1:]) for k,v in data.items()}
    evidence['subject_boxes']=torch.as_tensor(fixed[PERMS],device=device).float()
    evidence['base_logit']=torch.as_tensor(prior[PERMS],device=device)
    logits=decoder(evidence)['subject_logits'].cpu().numpy()
    reindexed=np.take_along_axis(logits,np.argsort(PERMS,axis=1),axis=1)
    original=reindexed[0].astype(float)
    effect=np.zeros((4,4))
    for k,(i,j) in enumerate(EDGES,1):
        diff=.5*((original[i]-original[j])-(reindexed[k,i]-reindexed[k,j]))
        effect[i,j],effect[j,i]=diff,-diff
    return original,effect


def mission(split):
    if split=='fresh':
        pin=read(BIAS/'run_PRIVATE/FRESH_SEAL.json')
        assert digest(BIAS/'run_PRIVATE/fresh_roster.json')==pin['roster']
        return [(k['key'],k['row'],[(k['a'],k['b'])]) for k in read(BIAS/'run_PRIVATE/fresh_roster.json')]
    return [(None,row,pairs) for row in source_rows('waymo',split) if (pairs:=catalog(row)[1])]


@torch.inference_mode()
def capture_split(model,split):
    out=[]
    for key,row,pairs in mission(split):
        objects,_=catalog(row)
        obs=frame(row) if split=='fresh' else observation(row)
        by={}
        for idx in sorted({idx for p in pairs for idx in p}):
            by[idx]=model._read_observation(dict(obs,text=canonical(objects[idx]['caption'])))
        for a,b in pairs:
            np.testing.assert_array_equal(by[a][0].sampled_points,by[b][0].sampled_points)
            ab=sorted((a,b),key=lambda i:canonical(objects[i]['caption']))
            boxes=common_nominations(*[native_boxes(by[i][1]) for i in ab])
            scores,delta=zip(*[replay(model._decoder,boxes,*by[i]) for i in (a,b)])
            mat,idx=source_identity(row,objects,boxes)
            hit,slot=ownership(mat,idx[[a,b]])
            out.append(dict(key=key or anon('pair',row['point_sha256'],a,b),rec=anon('rec',row['recording']),
                            j=np.stack(scores),effect=np.stack(delta),slots=slot,hits=hit))
    result={k:np.stack([r[k] for r in out]) for k in out[0]}
    expected=({'fresh':142,'calibration':44,'fit':213})[split]
    assert len(out)==expected
    if split!='fresh':
        old=loaded(PARENT_WORK/f'{split}.npz')
        assert np.array_equal(result['key'],old['key'])
        np.testing.assert_allclose(result['j'],old['free_j'],rtol=0,atol=4e-6)
        np.testing.assert_array_equal(result['slots'],old['free_slots'])
    else:
        old=loaded(BIAS/'run_PRIVATE/fresh_predictions.npz')
        np.testing.assert_allclose(result['j'],old['j'],rtol=0,atol=4e-6)
    return result


def estimate(data):
    rho=np.zeros((4,4))
    mu=np.zeros((4,4))
    xy=[]
    counts=np.zeros((4,4),dtype=int)
    for j,e,slots in zip(data['j'],data['effect'],data['slots']):
        if min(slots)<0:
            continue
        i,k=sorted(slots)
        bias=(j[:,i]-j[:,k]).mean()
        for d in e[:,i,k]:
            xy.append((i,k,bias,d))
            counts[i,k]+=1
            rho[i,k]+=bias
            mu[i,k]+=d
    rho/=np.maximum(counts,1)
    mu/=np.maximum(counts,1)
    x=np.array([d-mu[i,k] for i,k,_,d in xy])
    y=np.array([b-rho[i,k] for i,k,b,_ in xy])
    beta=(x@y)/(x@x) if x@x>1e-14 else 0.
    return dict(rank=rho-rho.T,center=mu-mu.T,beta=np.array(beta),counts=counts)


def evaluate(split,data,model):
    rows,summary=[],[]
    for arm in ('v1','balanced_swap','rank_only','calibrated_swap'):
        errors,contrasts,held,won=[],[],[],[]
        for key,j,e,slot,hit in zip(data['key'],data['j'],data['effect'],data['slots'],data['hits']):
            if arm=='v1':
                correction=np.zeros_like(e)
            elif arm=='balanced_swap':
                correction=e
            elif arm=='rank_only':
                correction=np.broadcast_to(model['rank'],e.shape)
            else:
                correction=model['rank']+model['beta']*(e-model['center'])
            assert np.abs(correction+correction.transpose(0,2,1)).max()<1e-9
            scores=j-correction.mean(2)
            picks=scores.argmax(1)
            four=bool(hit[0,picks[0]] and hit[1,picks[1]])
            a,b=slot
            pair=direct=False
            err=''
            if min(slot)>=0:
                nuisance=(j[:,a]-j[:,b]).mean()
                residual=nuisance-correction[:,a,b]
                err=float(np.mean(residual**2))
                errors.append(err)
                direct=bool(j[0,a]-j[0,b]-correction[0,a,b]>0 and j[1,b]-j[1,a]+correction[1,a,b]>0)
                pair=bool(scores[0,a]>scores[0,b] and scores[1,b]>scores[1,a])
                contrasts.append(direct)
            held.append(pair)
            won.append(four)
            rows.append(dict(split=split,arm=arm,pair=str(key)[:16],available=int(min(slot)>=0),
                             direct_edge_both=int(direct),both=int(pair),both_four=int(four),nuisance_sqerr=err))
        summary.append(dict(split=split,arm=arm,n=len(held),available=len(errors),nuisance_mse=np.mean(errors),
                         direct_edge_both=sum(contrasts),both=sum(held),both_four=sum(won)))
    return rows,summary


def main():
    setup()
    WORK.mkdir(exist_ok=True,mode=0o700)
    grounder=engine()
    all_rows,stats=[],[]
    start=perf_counter()
    for split in ('fit','calibration','fresh'):
        d=capture_split(grounder,split)
        archive_npz(WORK/f'{split}.npz',**d)
        if split=='fit':
            method=estimate(d)
            archive_npz(WORK/'model.npz',**method)
            atomic(WORK/'LOCK.json',dict(model=digest(WORK/'model.npz'),protocol=digest(HERE/'PROTOCOL.json'),
                                        code=digest(HERE/'intervene.py'),source=digest(WORK/'fit.npz'),
                                        heldout_labels=False,target_labels=False))
        else:
            assert digest(WORK/'model.npz')==read(WORK/'LOCK.json')['model']
            method=loaded(WORK/'model.npz')
        rows,summary=evaluate(split,d,method)
        all_rows.extend(rows)
        stats.extend(summary)
        print(split,summary,'elapsed',round(perf_counter()-start,1),flush=True)
    write_csv(RESULTS/'source_summary.csv',stats)
    from score import compressed_csv
    compressed_csv(RESULTS/'source_all.csv.gz',all_rows)
    atomic(RESULTS/'audit.json',dict(beta=float(method['beta']),rank_bias=method['rank'].tolist(),
        mean_response=method['center'].tolist(),observed_edges=method['counts'].tolist(),
        participants=399,training_pairs=213,historical_dev_pairs=44,reused_confirmation_pairs=142,
        target_labels=False,FINAL_access=False))


if __name__=='__main__':
    main()

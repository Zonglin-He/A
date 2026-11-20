"""Observation-only ablations. Replay the primary subject and reference before sealing."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np

from common import ARMS, WORK, prior, catalog, pins, digest, save, read, archive_npz, row_id
from scene_partition import build
from physical_nodes import assemble, overlap_prior
from physical_evidence import Returns
from ordered_choice import prepare
from role_cut import build_library, structure, shared, medoid, layout
from predict_delivery import to_model, compare_pose


def noun_probabilities(seg, data):
    n,c = data['p'].shape
    raw = np.ones((n,c))
    for k, ids in enumerate(data['assign']):
        nodes, first = np.unique(ids,return_index=True)
        valid = nodes >= 0
        raw[nodes[valid]] = seg['p'+str(k)][seg['labels'][k,first[valid]],:-1]
    raw /= raw.sum(1,keepdims=True)
    # The frozen class distribution times dimension compatibility, normalized, is data['p'].
    # Dividing cancels the unknown per-node normalization; normalization below gives size-only classes.
    assert np.all(raw > 0)
    size = data['p'] / raw
    return raw, size / size.sum(1,keepdims=True)


def phrase_votes(data, native, which, points):
    boxes, _, weights, _ = prepare(native,which)
    n = len(data['size'])
    votes = np.zeros((len(boxes),n))
    for k,box in enumerate(boxes):
        loc = points.local(box)[0]
        if not len(loc):continue
        counts = np.zeros(n)
        for ids in data['assign'][:,loc]:
            counts = np.maximum(counts,np.bincount(ids[ids>=0],minlength=n))
        votes[k] = counts/(data['size']+len(loc)-counts)
    return weights,votes


def projected_prior(table,keep):
    weights,votes = table
    v = votes[:,keep]
    sums = v.sum(1,keepdims=True)
    total = weights @ np.divide(v,sums,out=np.zeros_like(v),where=sums>0)
    if total.sum()>0:total /= total.sum()
    return np.sqrt(1+len(total)*total)


def restrict(data,levels):
    mask = sum(1 << i for i in levels)
    ids = np.flatnonzero((data['level'] & mask) != 0)
    lookup = np.full(len(data['parent']),-1,int)
    lookup[ids] = np.arange(len(ids))
    parent = data['parent'].copy()
    for v in reversed(range(len(parent))):
        ancestor = parent[v]
        if ancestor >= 0 and lookup[ancestor]<0:parent[v]=parent[ancestor]
    selected = parent[ids]
    selected[selected>=0] = lookup[selected[selected>=0]]
    return ids,selected


def closed_joint(parent,stop,occupancy,probs,words,view,valid,side):
    # Disjoint nodes share some ancestor survival events. Count each common event once.
    ancestors = np.eye(len(parent),dtype=bool)
    for i in reversed(range(len(parent))):
        if parent[i]>=0:ancestors[i] |= ancestors[parent[i]]
    gain = np.ones_like(valid,dtype=float)
    for i,s in enumerate(stop):
        if s == 1:continue  # Any strict descendant already has zero marginal.
        both = np.flatnonzero(ancestors[:,i] & (np.arange(len(parent))!=i))
        if len(both)>1:gain[np.ix_(both,both)] /= 1-s
    _,_,vis,gap = layout(view,side,'physical')
    return ((occupancy*probs[:,words[0]])[:,None]*(occupancy*probs[:,words[1]])*
            gain*valid*(gap>0)*vis[:,None]*vis[None,:])


def solve(data,native,words,raw,size,points):
    sizes = {f'omit_{v+1}':tuple(i for i in range(8) if i!=v) for v in range(8)}
    sizes.update(four_scales=(0,2,5,7),two_scales=(2,5),one_scale=(3,))
    t=phrase_votes(data,native,'t',points)
    r=phrase_votes(data,native,'r',points)
    side,depth,nearest = int(native['side']),int(native['depth']),bool(native['immediate'])
    results={}
    err=0.
    for arm,levels in [('full',tuple(range(8))),*sizes.items()]:
        ids,parent=restrict(data,levels)
        if not len(ids):
            results[arm]=(-1,-1)
            continue
        stop,p,view=data['stop'][ids],data['p'][ids],data['view'][ids]
        occupancy,K=structure(parent,stop,data['size'][ids])
        valid=K==0
        us,ur=projected_prior(t,ids),projected_prior(r,ids)
        def evaluate(p):
            return shared(parent,valid,stop,p,words,view,side,depth,nearest,'physical')[0]
        def select(joint,similarity,left,right):
            s,r,_=medoid(joint,similarity,left,right,valid)
            return tuple(int(ids[i]) if i>=0 else -1 for i in (s,r))
        joint=evaluate(p)
        results[arm]=select(joint,K,us,ur)
        if arm!='full':continue
        for x,k in ((us,'t'),(ur,'r')):
            np.testing.assert_allclose(x,overlap_prior(data,native,k,points),rtol=2e-12)
        ones=np.ones(len(p))
        results['no_subject_proposals']=select(joint,K,ones,ur)
        results['no_reference_proposals']=select(joint,K,us,ones)
        results['no_proposal_priors']=select(joint,K,ones,ones)
        results['node_map']=select(joint,np.eye(len(p)),us,ur)
        for name,class_p in [('no_noun_evidence',size),('no_size_in_noun',raw)]:
            results[name]=select(evaluate(class_p),K,us,ur)
        exact=joint if depth or nearest else closed_joint(parent,stop,occupancy,p,words,view,valid,side)
        np.testing.assert_allclose(exact,joint,atol=1e-12,rtol=1e-9)
        err=float(np.max(np.abs(exact-joint),initial=0))
        results['ordinary_closed_form']=select(exact,K,us,ur)
    return results,err


def pose(raw,completed,arm):
    box=completed.copy()
    if arm in ('no_horizontal','no_completion'):
        box[[0,1,3,4]]=raw[[0,1,3,4]]
    if arm in ('no_vertical','no_completion'):
        box[[2,5]]=raw[[2,5]]
    if arm=='symmetric_horizontal':box[:2]=raw[:2]
    return box


def job(task):
    fid,frame,rows,code = task
    receipt=WORK/'frames'/(fid+'.json')
    if receipt.exists():
        r=read(receipt)
        if r['code']==code and all(digest(WORK/'boxes'/(k+'.npz'))==h for k,h in r['files'].items()):return r
    data=None
    if any(row['active'] for row in rows):
        assert digest(frame['lex'])==frame['lh'] and digest(frame['word'])==frame['wh']
        with np.load(frame['lex']) as lex, np.load(frame['word']) as word:
            seg=build(lex['sample'],lex['logits'],lex['xyz'],frame['f'])
            data=assemble(seg,word)
        points=Returns(data['xyz'])
        raw,size=noun_probabilities(seg,data)
    errors=[0.,0.];files={};n=0 if data is None else len(data['parent'])
    for row in rows:
        assert digest(row['prior'])==row['ph'] and digest(row['role'])==row['rh']
        with np.load(row['prior']) as store:past=to_model(row,store)
        pred={k:np.tile(past[v][0],(len(ARMS),1)) for k,v in [('subject','anchored'),('reference','ref_anchored')]}
        pred['subject'][-1]=past['raw'][0]
        selected=np.full((len(ARMS),2),-1,int)
        complex_form=False
        if row['active'] and n:
            with np.load(row['role']) as z:
                choices,err=solve(data,z,row['words'],raw,size,points)
                complex_form=bool(z['immediate'] or z['depth'])
            errors[1]=max(errors[1],err)
            for i,arm in enumerate(ARMS):
                selected[i]=choices.get(arm,choices['full'])
                for role,(node,cat,key) in enumerate(zip(selected[i],row['words'],('subject','reference'))):
                    if node<0:continue
                    pred[key][i]=pose(data['boxes'][node],data['completed'][node,cat],arm)
        slot=prior.old.ARMS.index('shared_cut')
        for key,oldkey in [('subject','anchored'),('reference','ref_anchored')]:
            errors[0]=max(errors[0],compare_pose(pred[key][0],past[oldkey][slot]))
        assert errors[0]<1.e-5,(row['scope'],errors[0])
        path=WORK/'boxes'/(row_id(row)+'.npz')
        archive_npz(path,**pred,selected=selected,complex_form=np.array(complex_form))
        files[row_id(row)]=digest(path)
    result=dict(code=code,files=files,nodes=n,replay_error=errors[0],solver_error=errors[1])
    save(receipt,result)
    return result


def run(workers):
    for d in ('frames','boxes'):(WORK/d).mkdir(exist_ok=True,mode=0o700)
    cat=catalog();build_library();groups=defaultdict(list)
    for row in cat['rows']:groups[row['frame']].append(row)
    code=pins()
    jobs=[(k,cat['frames'].get(k),v,code) for k,v in sorted(groups.items())]
    start=perf_counter();files={};errors=[0.,0.]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for count,result in enumerate(pool.map(job,jobs,chunksize=1),1):
            files.update(result['files'])
            errors=np.maximum(errors,[result['replay_error'],result['solver_error']])
            if count%100==0:print('frames',count,'/',len(jobs),'seconds',round(perf_counter()-start),flush=True)
    assert set(files)=={row_id(r) for r in cat['rows']}
    save(WORK/'SEALED.json',dict(code=code,files=files,errors=list(errors),frames=len(jobs),seconds=perf_counter()-start,
         untouched_ordinary_fallback=prior.CFG['ordinary_seal'],labels_used=False,FINAL_access=False))
    print('sealed',len(files),errors,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=6)
    run(parser.parse_args().workers)

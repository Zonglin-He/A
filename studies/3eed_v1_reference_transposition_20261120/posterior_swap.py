"""Frozen crossed responses and identical-pool controls, including continuous geometry."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
import numpy as np
from scipy.special import expit

from registry_swap import HERE, OBS, PRED, SPEC, digest, read, save
from collect import archive_npz
from roles_joint import affinity
from joint_coverage import estimate_shape
from physical_swap import Exchange
from center_consensus import best_center, crossing_intervals

ARMS = tuple(read(SPEC)['arms'])


def agreement(gamma, order, valid):
    return np.where(valid,1.+(2*gamma[:,None]-1.)*(2*order-1.),1.)


def choose(prior, q, factor):
    joint = prior[:,None]*q[None,:]*factor
    i = int(np.argmax(joint.sum(1)))
    return i,int(np.argmax(joint[i]))


def predict(z):
    a = z['a']
    orig = z['slate'][int(z['before'])]
    raw = np.tile(orig,(len(ARMS),1))
    result = dict(raw=raw,boxes=raw.copy(),refs=np.tile(z['ref'],(len(ARMS),1)),
                  applied=bool(z['active']),observed_changes=0,rel_informative=0,
                  station_count=0,max_abs_log_cross=0.,shift=np.zeros(2))
    if not bool(z['active']):
        return result
    refs,q,boxes,masses = [z[k] for k in ('landmarks','q','views','masses')]
    swap = Exchange(z['cells'],z['half'])
    maps = [affinity(a,pool) for pool in boxes]
    u = np.stack([k@m for k,m in zip(maps,masses)])
    # Condition each view on the SAME physical anchor bank; zeros never create evidence.
    u = np.divide(u,u.sum(1,keepdims=True),out=np.zeros_like(u),where=u.sum(1,keepdims=True)>0)
    support = (u>0).all(0)
    logs = np.log(u,where=u>0,out=np.zeros_like(u))
    crossed = logs[0]-logs[1]-logs[2]+logs[3]
    l0,ls = z['looks']
    h = np.divide(l0,l0+ls,out=np.full_like(l0,.5),where=l0+ls>0)
    stationary = swap.stationary(a)
    valid = z['available'] & stationary[:,None] & support[:,None] & (l0+ls>0)
    c = agreement(expit(crossed),h,valid)
    plain = np.ones_like(c)
    arms = {
        'native_prior': (u[0],plain),
        'direct_four': (u.mean(0),plain),
        'factual_order': (u[0],np.where(z['available']&stationary[:,None],2*l0,1.)),
        'full_text_only': (u[0],agreement(expit(logs[0]-logs[2]),h,valid)),
        'wrong_landmark': (u[0],c[:,::-1]),
        'crossed_raw': (u[0],c)}
    result.update(observed_changes=int(np.any(abs(c-1)>1e-8,axis=1).sum()),
                  rel_informative=int(np.any(valid & (abs(h-.5)>1e-8),axis=1).sum()),
                  station_count=int(stationary.sum()),max_abs_log_cross=float(np.max(abs(crossed[support]),initial=0.)))
    # Native four are PCA boxes from the original bank. Exact duplicates have identical crossed factors.
    matches = [np.flatnonzero(np.max(abs(a-b),axis=1)<1.e-6) for b in z['slate']]
    if not all(len(v) for v in matches):
        raise AssertionError('Registered four-slot physical anchor missing')
    pick4,j4 = choose(z['base4'],q,c[[int(v[0]) for v in matches]])
    result['raw'][1] = result['boxes'][1] = z['slate'][pick4]
    result['refs'][1] = refs[j4]
    for name,(prior,factor) in arms.items():
        ix = ARMS.index(name)
        i,j = choose(prior,q,factor)
        result['raw'][ix] = result['boxes'][ix] = a[i]
        result['refs'][ix] = refs[j]
    i,j = choose(u[0],q,c)
    anchor = a[i]
    admitted = np.concatenate([np.full(len(a),True),np.full(len(a),True),swap.stationary(boxes[2]),swap.stationary(boxes[3])])
    local = np.concatenate([k[i]*m for k,m in zip(maps,masses)])*admitted
    transported = []
    for k in maps:
        normalizer = masses[0]@k
        transported.append(np.divide((masses[0]*c[:,j])@k,normalizer,
                                     out=np.ones_like(normalizer),where=normalizer>0))
    conditional = local*np.concatenate(transported)
    stacked = np.vstack(boxes)
    candidates,inv = np.unique(stacked[local>0],axis=0,return_inverse=True)
    w0 = np.bincount(inv,weights=local[local>0],minlength=len(candidates))
    w1 = np.bincount(inv,weights=conditional[local>0],minlength=len(candidates))
    ints = crossing_intervals(anchor,candidates)
    mean = estimate_shape(candidates,w0,anchor)
    uncond,t0 = best_center(anchor,ints,w0)
    cond,t1 = best_center(anchor,ints,w1)
    result['raw'][-3:] = anchor
    result['refs'][-3:] = refs[j]
    result['boxes'][-3:] = np.stack((mean,uncond,cond))
    result['shift'] = np.array([t0,t1])
    return result


def one(args):
    name,pin,code = args
    f = OBS/name
    assert digest(f)==pin
    with np.load(f,allow_pickle=False) as z:
        dec = predict(z)
        assert np.isfinite(dec['boxes']).all() and np.all(dec['boxes'][:,3:6]>0)
        np.testing.assert_array_equal(dec['boxes'][0],z['slate'][int(z['before'])])
        if not dec['applied']:
            np.testing.assert_array_equal(dec['boxes'],np.repeat(dec['boxes'][:1],len(ARMS),axis=0))
    path = PRED/name
    archive_npz(path,**dec)
    return name,digest(path),int(dec['applied'])


def run(workers):
    pin = read(OBS/'SEALED.json')
    PRED.mkdir(mode=0o700,exist_ok=True)
    code = {n:digest(HERE/n) for n in ('registry_swap.py','physical_swap.py','center_consensus.py','posterior_swap.py')}
    jobs = [(n,h,code) for n,h in sorted(pin['files'].items())]
    files,active = {},0
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k,(n,h,used) in enumerate(pool.map(one,jobs,chunksize=1),1):
            files[n] = h
            active += used
            if k%120==0:
                print(k,len(jobs),round(perf_counter()-start,1),flush=True)
    assert len(files)==pin['n'] and not pin['labels'] and not pin['FINAL_access']
    save(PRED/'SEALED.json',dict(n=len(files),files=files,active=active,code=code,protocol=digest(SPEC),
         observed=digest(OBS/'SEALED.json'),labels=False,FINAL_access=False,source_training=0,target_updates=0,
         seconds=perf_counter()-start,arms=ARMS))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=6)
    run(ap.parse_args().workers)

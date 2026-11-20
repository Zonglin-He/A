"""Bind a noun's response to observed returns, then replay the frozen scene-wide decision."""
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import softmax

from token_paths import ARMS, OUT, SOURCES, WORK, archive_npz, digest, prior, read, roster, row_id, save, signatures
from predict_delivery import compare_pose, to_model
from role_cut import medoid, shared, structure


def double_center(x):
    return x-x.mean(0)-x.mean(1,keepdims=True)+x.mean()


def observations(xyz, values, returns, regions):
    # Duplicate seed coordinates have one vote in the spatial centering and in every region.
    sites, inv, count = np.unique(xyz,axis=0,return_inverse=True,return_counts=True)
    pooled = np.zeros((len(sites),)+values.shape[1:],dtype=np.float64)
    np.add.at(pooled,inv,values)
    pooled /= count[:,None,None]
    token, absent, cosine = np.moveaxis(pooled,2,0)
    odds = token-absent
    interaction = double_center(odds)
    controls = dict(token_interaction=interaction,token_direct=odds,
                    no_object_interaction=double_center(-absent),
                    contrastive_interaction=double_center(cosine),
                    wrong_support=np.roll(interaction,max(1,len(sites)//2),axis=0))
    near = cKDTree(sites).query(returns)[1] if len(returns) else np.empty(0,int)
    fields = {name:np.zeros((len(regions),odds.shape[1])) for name in controls}
    for i, indices in enumerate(regions):
        used = np.unique(near[indices])
        if len(used):
            for name, field in controls.items():
                fields[name][i] = field[used].mean(0)
    return fields


def posteriors(base, fields):
    p = {'tree':base}
    live = base.sum(1)>0
    for name,f in fields.items():
        q = base.copy()
        with np.errstate(divide='ignore'):
            q[live] = softmax(np.log(q[live])+f[live],axis=1)
        p[name] = q
    return p


def choose(plan, fields, words, us, ur, side, depth, immediate):
    margin,K = structure(plan['parent'],plan['stop'],plan['size'])
    valid = K==0
    visible = np.isfinite(plan['view'][:,0])
    outputs = {}
    if not len(K):
        return {a:(-1,-1) for a in ARMS}
    for a in ARMS:
        if a=='independent':
            p=fields['token_interaction']
            su=K@(margin*p[:,words[0]]*us*visible)
            ru=K@(main_r*ur)
            outputs[a]=(int(np.argmax(np.where(visible,su,-1))) if su.max(initial=0)>0 else -1,
                        int(np.argmax(np.where(visible,ru,-1))) if ru.max(initial=0)>0 else -1)
            continue
        p=fields['token_interaction' if a=='wrong_reference' else a]
        mode='wrong_reference_binding' if a=='wrong_reference' else 'physical'
        joint,rf=shared(plan['parent'],valid,plan['stop'],p,words,plan['view'],side,depth,immediate,mode)
        if a=='token_interaction':
            main_r = rf
        s,r,_=medoid(joint,K,us,ur,valid)
        outputs[a]=(s,r)
    return outputs


def run():
    rows,obs=roster()
    plans=read(prior.WORK/'PLANS.json')
    pins=signatures()
    input_pins={}
    for src in SOURCES:
        path=WORK/src/'SEALED.json'
        seal=read(path)
        assert seal['pins']==pins and not seal['labels']
        input_pins.update(seal['files'])
    assert set(input_pins)==set(obs)
    by=defaultdict(list)
    for r in rows:
        by[r['frame']].append(r)
    dest=WORK/'predictions'
    dest.mkdir(mode=0o700,exist_ok=True)
    files={};max_error=0.
    for fid,items in by.items():
        src=items[0]['source']
        obs_path=WORK/src/(fid+'.npz')
        plan_path=prior.WORK/'plans'/(fid+'.npz')
        assert digest(obs_path)==input_pins[fid] and digest(plan_path)==plans['files'][fid]
        with np.load(plan_path) as z:
            plan=dict(z)
        regions=[np.flatnonzero((plan['assign']==i).any(0)) for i in range(len(plan['p']))]
        np.testing.assert_array_equal(list(map(len,regions)),plan['size'])
        with np.load(obs_path) as z:
            delta=observations(z['xyz'],z['values'],plan['xyz'],regions)
        fields=posteriors(plan['p'],delta)
        for r in items:
            rid=row_id(r)
            j=np.flatnonzero(plan['row']==rid).item()
            assert digest(r['role'])==r['rh'] and digest(r['prior'])==r['ph']
            with np.load(r['role']) as z:
                choices=choose(plan,fields,r['words'],plan['us'][j],plan['ur'][j],int(z['side']),int(z['depth']),bool(z['immediate']))
            with np.load(r['prior']) as z:
                old=to_model(r,z)
            arrays={k:np.tile(a[0],(len(ARMS),1)) for k,a in old.items()}
            nodes=np.array([choices[a] for a in ARMS])
            for i,(s,ref) in enumerate(nodes):
                for n,c,k1,k2 in [(s,r['words'][0],'raw','anchored'),(ref,r['words'][1],'ref','ref_anchored')]:
                    if n>=0:
                        arrays[k1][i]=plan['boxes'][n]
                        arrays[k2][i]=plan['completed'][n,c]
            idx=prior.old.ARMS.index('shared_cut')
            max_error=max([max_error]+[compare_pose(v[0],old[k][idx]) for k,v in arrays.items()])
            assert max_error<1e-5
            out=dest/(rid+'.npz')
            archive_npz(out,**arrays,nodes=nodes)
            files[rid]=digest(out)
    assert len(files)==128
    save(WORK/'SEALED.json',dict(files=files,pins=pins,captures={s:digest(WORK/s/'SEALED.json') for s in SOURCES},
                                 labels=False,training=0,FINAL=False,total=len(files),tree_max_error=max_error))
    print('sealed',len(files),'native tree difference',max_error,flush=True)

if __name__=='__main__':
    run()

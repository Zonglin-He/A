from collections import defaultdict
from time import perf_counter
from motion import (ARMS, BIAS, DUEL, HERE, RESULTS, WORK, anon, archive_npz, atomic, channels,
                    choose_scores, digest, engine, initialize, np, observations, read, torch)
from fit_duels import loaded
from fit import write_csv
from paired import capture, fixed_prior
from features import observation
from source_features import common_nominations, native_boxes, ownership
from source_pairs import catalog,source_identity
from scope import canonical, source_rows
from scope183 import frame


def cases(split):
    if split=='fresh':
        roster=BIAS/'run_PRIVATE/fresh_roster.json'
        assert read(BIAS/'run_PRIVATE/FRESH_SEAL.json')['roster']==digest(roster)
        return [(p['key'],p['row'],[(p['a'],p['b'])]) for p in read(roster)]
    return [(None,r,p) for r in source_rows('waymo',split) if (p:=catalog(r)[1])]


@torch.inference_mode()
def extract(model,split):
    rows=[]
    n_parity=0
    for key,r,pairs in cases(split):
        objects,_=catalog(r)
        f=frame(r) if split=='fresh' else observation(r)
        views={}
        for idx in sorted({v for p in pairs for v in p}):
            v=observations(model,dict(f,text=canonical(objects[idx]['caption'])),parity=(n_parity==0))
            n_parity+=int(n_parity==0 and v.neutral is not None)
            views[idx]=v
        for a,b in pairs:
            ab=sorted((a,b),key=lambda i:canonical(objects[i]['caption']))
            boxes=common_nominations(*[native_boxes(views[i].ev) for i in ab])
            matrix,idx=source_identity(r,objects,boxes)
            hits,slots=ownership(matrix,idx[[a,b]])
            result=defaultdict(list)
            for i in (a,b):
                v=views[i]
                prior,_=fixed_prior(boxes,v.full)
                dev=v.ev['subject_boxes'].device
                ev=dict(v.ev,subject_boxes=torch.as_tensor(boxes[None],device=dev,dtype=torch.float32),
                        base_logit=torch.as_tensor(prior[None],device=dev))
                _,j=capture(model._decoder,ev)
                value,valid=channels(boxes,v)
                own=native_boxes(v.ev)
                m,_=source_identity(r,objects,own)
                own_ious=m[:,idx[i]]
                own_value,own_valid=channels(own,v)
                for k,x in dict(j=j,val=value,ok=valid,native_j=capture(model._decoder,v.ev)[1],
                    native_val=own_value,native_ok=own_valid,own_iou=own_ious,parsed=v.neutral is not None).items():
                    result[k].append(x)
            rows.append(dict(key=key or anon('pair',r['point_sha256'],a,b),rec=anon('rec',r['recording']),
                hits=hits,slots=slots,**{k:np.stack(v) for k,v in result.items()}))
    z={k:np.stack([r[k] for r in rows]) for k in rows[0]}
    expected=(loaded(BIAS/'run_PRIVATE/fresh_predictions.npz')['j'] if split=='fresh'
              else loaded(DUEL/'run_PRIVATE'/f'{split}.npz')['free_j'])
    np.testing.assert_allclose(expected,z['j'],atol=2e-6,rtol=0)
    assert n_parity==1
    return z


def fit(z):
    x,y=[],[]
    for j,v,k,slot in zip(z['j'],z['val'],z['ok'],z['slots']):
        a,b=slot
        if min(slot)<0:
            continue
        bias=(j[:,a].astype(float)-j[:,b]).mean()
        for t in range(2):
            if k[t,a] and k[t,b]:
                x.append(v[t,a]-v[t,b])
                y.append(bias)
    x=np.array(x,dtype=float).reshape(-1,3)
    y=np.asarray(y)
    norm=(x*x).sum(0)
    coefficient=np.divide(x.T@y,norm,out=np.zeros(3),where=norm>0)
    return dict(coefficients=coefficient.clip(0,1).tolist(),unclipped=coefficient.tolist(),
                fitting_directions=len(y),feature_norm=norm.tolist())


def review(split,z,coefs):
    rows=[]
    for i,slot in enumerate(z['slots']):
        scores=[choose_scores(j,v,k,coefs) for j,v,k in zip(z['j'][i],z['val'][i],z['ok'][i])]
        native=[choose_scores(j,v,k,coefs) for j,v,k in zip(z['native_j'][i],z['native_val'][i],z['native_ok'][i])]
        a,b=slot
        for arm in ARMS:
            s=np.stack([z_[arm] for z_ in scores])
            picks=s.argmax(1)
            nn=np.array([z_[arm].argmax() for z_ in native])
            ious=z['own_iou'][i,np.arange(2),nn]
            rows.append(dict(split=split,pair=str(z['key'][i])[:16],rec_group=str(z['rec'][i])[:12],arm=arm,
                available=int(min(slot)>=0),measured_directions=int(z['ok'][i][:,[a,b]].all(1).sum()) if min(slot)>=0 else 0,
                both=int(min(slot)>=0 and s[0,a]>s[0,b] and s[1,b]>s[1,a]),
                both_four=int(z['hits'][i,0,picks[0]] and z['hits'][i,1,picks[1]]),
                native_a=float(ious[0]),native_b=float(ious[1]),native_measured=int((z['native_ok'][i].sum(1)>=2).sum())))
    return rows


def main():
    initialize()
    model=engine()
    rows=[]
    start=perf_counter()
    for split in ('fit','calibration','fresh'):
        z=extract(model,split)
        archive_npz(WORK/(split+'.npz'),**z)
        if split=='fit':
            learned=fit(z)
            atomic(WORK/'MODEL_LOCK.json',dict(**learned,source=digest(WORK/'fit.npz'),
                info=digest(HERE/'PROTOCOL.json'),motion=digest(HERE/'motion.py'),
                trainer=digest(HERE/'source_motion.py'),target_labels=False,FINAL_access=False))
        params=read(WORK/'MODEL_LOCK.json')
        rows+=review(split,z,params['coefficients'])
        print(split,len(z['j']),params['coefficients'],round(perf_counter()-start,1),flush=True)
    from score import compressed_csv
    compressed_csv(RESULTS/'source_all.csv.gz',rows)
    table=[]
    for s in ('fit','calibration','fresh'):
        for a in ARMS:
            items=[r for r in rows if r['split']==s and r['arm']==a]
            base=[r for r in rows if r['split']==s and r['arm']=='v1']
            io=np.array([[r['native_a'],r['native_b']] for r in items])
            v0=np.array([[r['native_a'],r['native_b']] for r in base])
            table.append(dict(split=s,arm=a,n=len(items),covered=sum(r['available'] for r in items),
                measurable_directions=sum(r['measured_directions'] for r in items),
                both=sum(r['both'] for r in items),both_four=sum(r['both_four'] for r in items),
                native_Acc25=(io>.25).mean()*100,native_Acc50=(io>.50).mean()*100,
                native_rescue50=int(((io>.5)&(v0<=.5)).sum()),native_harm50=int(((io<=.5)&(v0>.5)).sum())))
    write_csv(RESULTS/'source_summary.csv',table)
    atomic(RESULTS/'source_calibration.json',learned)


if __name__=='__main__':
    main()

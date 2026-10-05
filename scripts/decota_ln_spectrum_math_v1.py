"""Pure NumPy parameter geometry and fixed, source-clustered diagnostics."""
import collections,hashlib
import numpy as np

RANKS=[1,2,4,8,16,32]
SEED=20261005

def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def norm_weights(meta):
    count=collections.Counter(r['source_id'] for r in meta)
    return np.array([1/count[r['source_id']] for r in meta],dtype=float)

def eig(g):
    g=np.asarray(g,float);g=(g+g.T)/2
    if not len(g):return np.array([]),np.zeros((0,0))
    v,u=np.linalg.eigh(g);idx=np.argsort(v)[::-1]
    assert v.min(initial=0)>-1e-8*max(float(v.max(initial=0)),1.),'non-PSD Gram'
    return np.maximum(v[idx],0),u[:,idx]

def spectrum(g,meta,mode):
    g=np.asarray(g,float);w=norm_weights(meta)
    d=np.maximum(np.diag(g),0);a=np.sqrt(w)
    if mode=='unit':
        inv=np.divide(1,np.sqrt(d),out=np.zeros_like(d),where=d>0)
        g=g*inv[:,None]*inv[None,:]
    elif mode=='centered':
        wn=w/w.sum();m=g@wn;g=g-m[:,None]-m[None,:]+wn@m
    assert mode in ['raw','unit','centered']
    e,_=eig(g*a[:,None]*a[None,:]);total=e.sum()
    p=e/total if total>0 else np.zeros_like(e);nz=p[p>0]
    mean_fraction=float((w/w.sum())@np.asarray(g)@(w/w.sum()))/(float(w@np.diag(g))/w.sum()) if w@np.diag(g)>0 else None
    return plain(dict(cells=len(meta),sources=len(set(r['source_id'] for r in meta)),zero_cells=int((d==0).sum()),
        eigenvalues=e,total_energy=float(total),energy={str(r):float(e[:r].sum()/total) if total>0 else None for r in RANKS},
        effective_rank=float(np.exp(-(nz*np.log(nz)).sum())) if len(nz) else None,
        mean_direction_fraction=mean_fraction,mode=mode))

def project_from_gram(g,train,test,ranks=RANKS):
    g=np.asarray(g,float);train=np.asarray(train,int);test=np.asarray(test,int)
    if not len(train):return [dict(index=int(i),previous=0,energy={str(r):None for r in ranks}) for i in test]
    e,v=eig(g[np.ix_(train,train)]);active=e>max(float(e.max(initial=0))*1e-12,1e-30)
    e,v=e[active],v[:,active];cross=g[np.ix_(test,train)]@v
    sq=cross**2/e[None,:] if len(e) else np.zeros((len(test),0))
    out=[]
    for j,i in enumerate(test):
        norm=float(g[i,i]);energy={str(r):float(np.clip(sq[j,:r].sum()/norm,0,1)) if norm>0 else None for r in ranks}
        out.append(dict(index=int(i),previous=len(train),available_rank=len(e),energy=energy))
    return out

def cosine(g,i,j):
    den=np.sqrt(max(g[i,i],0)*max(g[j,j],0))
    return float(np.clip(g[i,j]/den,-1,1)) if den>0 else None

def source_stats(rows,fields,seed=SEED,draws=10000):
    by=collections.defaultdict(list)
    for q in rows:
        if all(q.get(f) is not None for f in fields):by[q['source_id']].append([q[f] for f in fields])
    if not by:return dict(cells=0,sources=0,metrics={})
    ids=sorted(by);x=np.array([np.mean(by[i],axis=0) for i in ids]);rng=np.random.default_rng(seed)
    boot=x[rng.integers(len(x),size=(draws,len(x)))].mean(1);ci=np.quantile(boot,[.025,.975],axis=0)
    return plain(dict(cells=sum(len(v) for v in by.values()),sources=len(ids),metrics={f:dict(mean=float(x[:,j].mean()),ci95=ci[:,j],source_values={str(i):float(a) for i,a in zip(ids,x[:,j])}) for j,f in enumerate(fields)},bootstrap_draws=draws,seed=seed))

def folds(meta):
    ids=sorted({r['source_id'] for r in meta},key=lambda i:hashlib.sha256(f'decota-ln-spectrum:20261005:{i}'.encode()).hexdigest())
    return {i:j%2 for j,i in enumerate(ids)}

def geometry(block):
    meta=block['rows'];g=np.array(block['gram']);ds=block['dataset'];stream=block['stream'];out=[]
    for split in ['search','confirm']:
        for group in ['corruption','clean']:
            ids=[i for i,q in enumerate(meta) if q['split']==split and ((q['condition']=='clean')==(group=='clean'))]
            mm=[meta[i] for i in ids];gg=g[np.ix_(ids,ids)]
            specs={mode:spectrum(gg,mm,mode) for mode in ['raw','unit','centered']}
            pairs=[]
            for i in range(len(ids)):
                for j in range(i):
                    c=cosine(gg,i,j)
                    if c is not None:pairs.append((c,mm[i]['source_id']==mm[j]['source_id']))
            norms=np.maximum(np.diag(gg),0);total=norms.sum();top=np.sort(norms)[::-1]
            out.append(dict(dataset=ds,stream=stream,split=split,group=group,spectra=specs,
                norm_energy_top1_fraction=float(top[:1].sum()/total) if total else None,
                norm_energy_top10_fraction=float(top[:10].sum()/total) if total else None,
                same_source_cosine_mean=float(np.mean([a for a,b in pairs if b])) if any(b for a,b in pairs) else None,
                different_source_cosine_mean=float(np.mean([a for a,b in pairs if not b])) if any(not b for a,b in pairs) else None))
    return out

def holdout(block):
    meta=block['rows'];g=np.array(block['gram']);out=[]
    for group in ['corruption','clean']:
        search=[i for i,q in enumerate(meta) if q['split']=='search' and ((q['condition']=='clean')==(group=='clean'))]
        confirm=[i for i,q in enumerate(meta) if q['split']=='confirm' and ((q['condition']=='clean')==(group=='clean'))]
        ff=folds([meta[i] for i in search]);tasks=[]
        for f in [0,1]:tasks.append((f'development_hash_fold{f}',[i for i in search if ff[meta[i]['source_id']]!=f],[i for i in search if ff[meta[i]['source_id']]==f]))
        tasks.append(('development_to_confirmation',search,confirm))
        for name,train,test in tasks:
            q=project_from_gram(g,train,test)
            for r in q:
                out.append({k:block[k] for k in ['dataset','stream']}|dict(group=group,test=name,source_id=meta[r['index']]['source_id'],condition=meta[r['index']]['condition'],order=meta[r['index']]['order'],energy=r['energy'],training_cells=len(train),training_sources=len({meta[i]['source_id'] for i in train})))
    return out

def prefix(segment):
    g=np.array(segment['gram']);meta=segment['rows'];out=[];previous=[]
    for i,m in enumerate(meta):
        assert m['arrival']==i
        if g[i,i]>0 and m['commit_norm']>0:
            z=project_from_gram(g,previous,[i])[0]
            out.append({k:segment[k] for k in ['dataset','stream','split','condition','order']}|dict(source_id=m['source_id'],arrival=i,energy=z['energy'],previous=len(previous),available_rank=z.get('available_rank',0)))
            previous.append(i)
    return out

def weighted_corr(x,y,w):
    w=np.asarray(w,float);sw=w.sum()
    if not sw:return None
    mx=w@x/sw;my=w@y/sw;vx=w@((x-mx)**2);vy=w@((y-my)**2)
    return float(np.clip((w@((x-mx)*(y-my)))/np.sqrt(vx*vy),-1,1)) if vx>1e-25 and vy>1e-25 else None

def pair_stats(rows,draws=10000):
    q=[r for r in rows if r['cosine'] is not None]
    if not q:return dict(pairs=len(rows),defined_cosine_pairs=0,recipients=0,donors=0)
    sources=sorted({r[k] for r in q for k in ['donor','recipient']});ix={s:i for i,s in enumerate(sources)}
    donor=np.array([ix[r['donor']] for r in q]);rec=np.array([ix[r['recipient']] for r in q]);x=np.array([r['cosine'] for r in q]);y=np.array([r['utility_v'] for r in q]);n=len(sources)
    # Each recipient source gets equal total base weight; repeated inputs/donors
    # are within that source. One source count is shared across donor/recipient roles.
    count=collections.Counter(rec.tolist());base=np.array([1/count[int(i)] for i in rec]);pos=x>0;neg=x<0
    def one(w):
        def mean(mask,z):return float(w[mask]@z[mask]/w[mask].sum()) if w[mask].sum()>0 else None
        a,b=mean(pos,y),mean(neg,y)
        return dict(correlation=weighted_corr(x,y,w),mean_utility=mean(np.ones(len(y),bool),y),positive_cosine_utility=a,negative_cosine_utility=b,positive_minus_negative=None if a is None or b is None else a-b,
            positive_cosine_harm_fraction=mean(pos,(y<0).astype(float)),negative_cosine_harm_fraction=mean(neg,(y<0).astype(float)),positive_cosine_gain_fraction=mean(pos,(y>0).astype(float)),negative_cosine_gain_fraction=mean(neg,(y>0).astype(float)))
    point=one(base);boot={k:[] for k in point};rng=np.random.default_rng(SEED)
    for _ in range(draws):
        c=rng.multinomial(n,np.full(n,1/n));b=one(base*c[donor]*c[rec])
        for k,v in b.items():
            if v is not None:boot[k].append(v)
    bins=[];edges=[-1,-.5,-.25,0,.25,.5,1]
    for i in range(len(edges)-1):
        mask=(x>=edges[i]) & ((x<edges[i+1]) if i<len(edges)-2 else (x<=edges[i+1]));qq=[r for r,f in zip(q,mask) if f]
        zz=[dict(source_id=r['recipient'],utility=r['utility_v'],gain=float(r['utility_v']>0),harm=float(r['utility_v']<0)) for r in qq]
        bins.append(dict(left=edges[i],right=edges[i+1],pairs=len(qq),stats=source_stats(zz,['utility','gain','harm'])))
    return plain(dict(pairs=len(rows),defined_cosine_pairs=len(q),zero_recipient_pairs=len(rows)-len(q),recipients=len(set(r['recipient'] for r in q)),donors=len(set(r['donor'] for r in q)),independent_source_nodes=n,
        positive_pairs=int(pos.sum()),negative_pairs=int(neg.sum()),zero_cosine_pairs=int((x==0).sum()),
        metrics={k:dict(mean=v,ci95=np.quantile(boot[k],[.025,.975]) if boot[k] else None,valid_bootstrap_draws=len(boot[k])) for k,v in point.items()},bins=bins,
        bootstrap='common source-node resampling weights both donor and recipient roles',draws=draws,seed=SEED))

def summarize_geometry(gs,hs,ps):
    hgroups=collections.defaultdict(list);pgroups=collections.defaultdict(list)
    for q in hs:hgroups[(q['dataset'],q['stream'],q['group'],q['test'])].append(q)
    for q in ps:
        budget='100' if q['stream']=='online100' else q['stream'].split('budget')[1]
        for warm in [1,8]:
            if q['previous']>=warm:pgroups[(q['dataset'],q['split'],'clean' if q['condition']=='clean' else 'corruption',budget,warm)].append(q)
    def energies(q):return [dict(source_id=r['source_id'],**{f'E{k}':r['energy'][str(k)] for k in RANKS}) for r in q]
    return dict(spectra=gs,heldout=[dict(dataset=k[0],stream=k[1],group=k[2],test=k[3],stats=source_stats(energies(v),[f'E{r}' for r in RANKS])) for k,v in hgroups.items()],
        strict_prefix=[dict(dataset=k[0],split=k[1],group=k[2],budget=k[3],minimum_previous_nonzero_writes=k[4],stats=source_stats(energies(v),[f'E{r}' for r in RANKS])) for k,v in pgroups.items()])

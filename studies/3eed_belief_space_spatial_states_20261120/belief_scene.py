"""Exact two-designation marginal of a random foreground cut; the rest of the scene is integrated out."""
import numpy as np

ARMS=('map_cut','map_world_pair','map_pair','independent','masked_product','tree_joint','permuted_reference')

def normalized(x):
    mass=x.sum()
    return x/mass if mass>0 else np.zeros_like(x)

def point_mass(x):
    p=np.zeros_like(x)
    if x.size and x.max()>0:
        p.flat[int(x.argmax())]=1.
    return p

def forest_factors(parent, stop, indices):
    """Prior pair inclusion, compatibility, prior MAP cut, and best-world factor per pair."""
    parent=np.asarray(parent,int); stop=np.asarray(stop,float)
    n=len(parent)
    if stop.shape!=(n,) or np.any(~np.isfinite(stop)) or np.any((stop<0)|(stop>1)):
        raise ValueError("Invalid stop weights")
    if any(p != -1 and not i < p < n for i,p in enumerate(parent)):
        raise ValueError("Forest must be topologically sorted with children first")
    if len(set(indices))!=len(indices) or np.any(indices<0) or np.any(indices>=n):
        raise ValueError("Invalid visible indices")
    child_opt=np.zeros(n); opt=np.zeros(n); stop_wins=np.zeros(n,bool)
    with np.errstate(divide='ignore'):
        log_stop=np.log(stop); log_split=np.log1p(-stop)
    for i,p in enumerate(parent):
        split=log_split[i]+child_opt[i]
        stop_wins[i]=log_stop[i]>=split
        opt[i]=max(log_stop[i],split)
        if p>=0:child_opt[p]+=opt[i]
    split_penalty=log_split+child_opt-opt
    chosen=np.ones(n,bool); reach=np.ones(n)
    for i in reversed(range(n)):
        p=parent[i]
        if p>=0:
            reach[i]=reach[p]*(1-stop[p])
            chosen[i]=chosen[p] and not stop_wins[p]
    k=len(indices); compatible=~np.eye(k,dtype=bool)
    # Strict ancestors. Adding a compulsory split once for all incident pair events
    # avoids singular inverse-probability factors at stop=1.
    anc_groups={}
    desc_slot={int(i):a for a,i in enumerate(indices)}
    for a,node in enumerate(indices):
        ancestor=parent[node]
        while ancestor>=0:
            anc_groups.setdefault(ancestor,[]).append(a)
            b=desc_slot.get(ancestor)
            if b is not None:compatible[a,b]=compatible[b,a]=False
            ancestor=parent[ancestor]
    joint=stop[indices,None]*stop[indices]
    local=log_stop[indices]-opt[indices]
    best=local[:,None]+local
    for i,descendants in anc_groups.items():
        inside=np.asarray(descendants)
        outside=np.flatnonzero(~np.isin(np.arange(k),inside))
        for sub in (np.ix_(inside,inside),np.ix_(inside,outside),np.ix_(outside,inside)):
            joint[sub]*=1-stop[i]
            best[sub]+=split_penalty[i]
    joint[~compatible]=0.;best[~compatible]=-np.inf
    mu=(stop*reach)[indices]
    return mu,compatible,joint,(chosen&stop_wins)[indices],best

def build(d, higher, seed):
    indices=np.flatnonzero(d['visible'])
    mu,compatible,joint,cut,best=forest_factors(d['parent'],d['stop'],indices)
    a=(d['qs']*d['nomination'])[indices]; b=d['qr'][indices]
    likelihood=a[:,None]*b
    allowed=~np.eye(len(mu),dtype=bool)
    if higher:
        diff=d['subject_boxes'][indices,5,None]-d['reference_boxes'][indices,5]
        allowed &= higher*diff>0.
    evidence=allowed*likelihood
    independent=normalized(mu[:,None]*mu*evidence)
    masked=normalized(independent*compatible)
    tree=normalized(joint*evidence)
    single=normalized((cut[:,None]&cut)*evidence)
    with np.errstate(divide='ignore'):
        log_pair=best+np.log(evidence)
    map_world=np.zeros_like(tree)
    if log_pair.size and np.isfinite(log_pair.max()):
        map_world.flat[int(log_pair.argmax())]=1
    peak=point_mass(tree)
    shuffled=b.copy()
    if len(b)>1:
        # Deterministic derangement of the same frame's reference noun scores.
        order=np.random.default_rng(seed).permutation(len(b))
        shuffled[order]=b[np.roll(order,1)]
    perm=normalized(joint*allowed*a[:,None]*shuffled)
    distributions=[point_mass(single),map_world,peak,independent,masked,tree,perm]
    hard_conf=[single.max(initial=0.),float((map_world*tree).sum()),tree.max(initial=0.)]
    return indices,compatible,distributions,hard_conf

def sample_world(parent, stop, visible_ids, pair_posterior, rng):
    """Draw a full occupied-node set conditional on a grounded two-role question.

    First draw a designation from its exact marginal. Its endpoints must stop, their
    ancestors must split, and every other reached branch uses the original stop law.
    """
    if not pair_posterior.size or pair_posterior.sum()==0:
        return None
    if not np.isclose(pair_posterior.sum(),1.):
        raise ValueError("Designation probabilities must be normalized")
    i,j=np.unravel_index(rng.choice(pair_posterior.size,p=pair_posterior.ravel()),pair_posterior.shape)
    pair=tuple(map(int,visible_ids[[i,j]]))
    stopped=set(pair); split=set()
    for v in pair:
        a=parent[v]
        while a>=0:
            split.add(int(a));a=parent[a]
    if pair[0]==pair[1] or split&stopped:
        raise ValueError("Incompatible designations cannot be forced into the same world")
    free=np.ones(len(parent),bool); active=[]
    for node in reversed(range(len(parent))):
        p=parent[node]
        if p>=0:free[node]=free[p]
        if free[node] and (node in stopped or (node not in split and rng.random()<stop[node])):
            active.append(node)
            free[node]=False
    if not stopped.issubset(active):raise AssertionError("Missing forced endpoint")
    return tuple(active),pair

"""Pure NumPy post-hoc arithmetic; no model, GT file, or private asset access."""
import collections
import numpy as np

EPS = 1e-12
RADII = dict(anchor=[0., .1, .25, .5, .75, 1., 1.5, 2., 3., 4., 8., None],
             window=[0., .025, .05, .1, .2, .3, .5, 1., 2., None])
CATEGORIES = ['same', 'start_only', 'end_only', 'expansion', 'contraction', 'co_directional']

def tag(value):
    return 'inf' if value is None else str(value).replace('.', 'p')

def geometry(anchor, interval, duration):
    ds, de = np.asarray(interval, float) - np.asarray(anchor, float)
    width = anchor[1] - anchor[0]
    d = abs(ds) + abs(de); c = (ds + de)/2; l = de - ds
    assert abs(d - max(2*abs(c), abs(l))) < 1e-10
    if abs(ds) <= EPS and abs(de) <= EPS: kind = 'same'
    elif abs(de) <= EPS: kind = 'start_only'
    elif abs(ds) <= EPS: kind = 'end_only'
    elif ds < 0 < de: kind = 'expansion'
    elif de < 0 < ds: kind = 'contraction'
    else: kind = 'co_directional'
    dominance = ('centre' if 2*abs(c) > abs(l)+EPS else
                 'extent' if abs(l) > 2*abs(c)+EPS else 'equal')
    inter = max(0., min(anchor[1], interval[1])-max(anchor[0], interval[0]))
    union = max(anchor[1], interval[1])-min(anchor[0], interval[0])
    return dict(interval=list(map(float,interval)), signed_start=float(ds), signed_end=float(de),
        start_abs=float(abs(ds)), end_abs=float(abs(de)), distance=float(d),
        start_seconds=float(abs(ds)*duration), end_seconds=float(abs(de)*duration),
        distance_seconds=float(d*duration), distance_anchor=float(d/width),
        signed_centre=float(c), signed_length=float(l),
        centre_seconds=float(abs(c)*duration), length_seconds=float(abs(l)*duration),
        doubled_centre_anchor=float(2*abs(c)/width), length_anchor=float(abs(l)/width),
        overlap_tIoU=float(inter/union), disjoint=bool(inter<=EPS), category=kind, dominance=dominance)

def derive(r):
    v = np.array(r['candidate_v'],float); t=np.array(r['candidate_t'],float)
    xy=np.array(r['intervals_normalized'],float); a=int(r['choices']['A8']); anchor=xy[a]
    assert len(v)==len(xy)==32 and 0<=a<8 and np.all(xy[:,1]>xy[:,0])
    width=anchor[1]-anchor[0]; dist=np.abs(xy-anchor).sum(1)
    o8=float(v[:8].max()); o32=float(v.max()); gain=o32-o8
    best=np.flatnonzero(v>=o32-EPS).tolist()
    near=min(best,key=lambda i:(dist[i],i)); far=min(best,key=lambda i:(-dist[i],i))
    ng=geometry(anchor,xy[near],r['duration_seconds']);fg=geometry(anchor,xy[far],r['duration_seconds'])
    out={k:r[k] for k in ['dataset','split','source_id','condition','order','arrival',
                        'A_state_pre_sha256','A_state_post_sha256']}
    out.update(anchor_index=a,anchor=anchor.tolist(),duration_seconds=r['duration_seconds'],
        intervals=xy.tolist(),candidate_v=v.tolist(),candidate_t=t.tolist(),
        A8_v=float(v[a]),A8_t=float(t[a]),O8_v=o8,O32_v=o32,capacity_gain=gain,
        selection_gap8=o8-float(v[a]),selection_gap32=o32-float(v[a]),
        gain_positive=float(gain>EPS),oracle_indices=best,nearest_index=near,farthest_index=far,
        nearest=ng,farthest=fg,oracle_tie=float(len(best)>1),
        tie_geometric_ambiguity=float(len({geometry(anchor,xy[i],r['duration_seconds'])['category'] for i in best})>1),
        nearest_distance=ng['distance'],farthest_distance=fg['distance'],
        nearest_distance_anchor=ng['distance_anchor'],farthest_distance_anchor=fg['distance_anchor'],
        nearest_distance_seconds=ng['distance_seconds'],farthest_distance_seconds=fg['distance_seconds'],
        start_abs=ng['start_abs'],end_abs=ng['end_abs'],
        start_seconds=ng['start_seconds'],end_seconds=ng['end_seconds'],
        centre_seconds=ng['centre_seconds'],length_seconds=ng['length_seconds'],
        overlap_tIoU=ng['overlap_tIoU'],disjoint=float(ng['disjoint']))
    for kind in CATEGORIES:
        out['gain_'+kind]=gain*float(ng['category']==kind)
    for kind in ['centre','extent','equal']:
        out['gain_dominance_'+kind]=gain*float(ng['dominance']==kind)
    for side,coord in [('start',0),('end',1)]:
        mask=np.abs(xy[:,coord]-anchor[coord])<=EPS
        out['fixed_'+side+'_gain']=float(v[mask].max()-v[a])
        out['fixed_'+side+'_count']=int(mask.sum())
    for norm,radii in RADII.items():
        scale=width if norm=='anchor' else 1.
        for radius in radii:
            p=norm+'_'+tag(radius);mask=np.ones(32,bool) if radius is None else dist<=radius*scale+EPS
            l8=float(v[:8][mask[:8]].max());l32=float(v[mask].max())
            out[p+'_L8_gain']=l8-float(v[a]);out[p+'_L32_gain']=l32-float(v[a])
            out[p+'_local_extra']=l32-l8;out[p+'_accessible_extra']=max(0.,min(gain,l32-o8))
            out[p+'_near_gain_mass']=gain*float(radius is None or ng['distance']<=radius*scale+EPS)
            out[p+'_far_gain_mass']=gain*float(radius is None or fg['distance']<=radius*scale+EPS)
            out[p+'_count8']=int(mask[:8].sum());out[p+'_count32']=int(mask.sum())
    return out

def source_matrix(rows, fields):
    sources=sorted({r['source_id'] for r in rows});orders=sorted({r['order'] for r in rows})
    matrix=[];byorder=collections.defaultdict(list)
    for s in sources:
        values=[]
        for o in orders:
            group=[r for r in rows if r['source_id']==s and r['order']==o]
            if not group:continue
            cells=[np.mean([[r[f] for f in fields] for r in group if r['condition']==c],axis=0)
                   for c in sorted({r['condition'] for r in group})]
            mean=np.mean(cells,axis=0);values.append(mean);byorder[o].append(mean)
        matrix.append(np.mean(values,axis=0))
    return sources,np.asarray(matrix),byorder

def summarize(rows):
    fields=sorted(k for k,v in rows[0].items() if isinstance(v,(int,float))
        and k not in ['source_id','arrival','anchor_index','nearest_index','farthest_index'])
    sources,mat,orders=source_matrix(rows,fields);at={f:j for j,f in enumerate(fields)}
    rng=np.random.default_rng(20261003)
    # Same resampled sources for all fields and all ratios.
    boot=np.concatenate([mat[rng.integers(0,len(mat),(100,len(mat)))].mean(1) for _ in range(100)])
    ci=np.percentile(boot,[2.5,97.5],axis=0);metrics={}
    for j,f in enumerate(fields):
        x=mat[:,j];loo=(x.sum()-x)/(len(x)-1) if len(x)>1 else x
        metrics[f]=dict(mean=float(x.mean()),ci95=ci[:,j].tolist(),
            source_values={str(s):float(v) for s,v in zip(sources,x)},
            order_values={o:float(np.mean(vals,axis=0)[j]) for o,vals in orders.items()},
            leave_one_out_range=[float(loo.min()),float(loo.max())],
            cell_mean=float(np.mean([r[f] for r in rows])))
    denom=mat[:,at['capacity_gain']].mean(); bd=boot[:,at['capacity_gain']]
    valid=bd>EPS;ratios={}
    numerators=[f for f in fields if f.startswith('gain_') and f not in ['gain_positive']]
    numerators += [f for f in fields if f.endswith(('_accessible_extra','_near_gain_mass','_far_gain_mass'))]
    for f in numerators:
        nums=boot[:,at[f]][valid];val=nums/bd[valid]
        ratios[f]=dict(mean=float(mat[:,at[f]].mean()/denom) if denom>EPS else None,
            ci95=np.percentile(val,[2.5,97.5]).tolist() if len(val) else None,
            bootstrap_zero_denominator_draws=int((~valid).sum()),bootstrap_valid_draws=int(valid.sum()),
            interval_condition='positive_bootstrap_total_capacity_gain',
            leave_one_out_range=None)
        if len(mat)>1:
            ld=(mat[:,at['capacity_gain']].sum()-mat[:,at['capacity_gain']])/(len(mat)-1)
            ln=(mat[:,at[f]].sum()-mat[:,at[f]])/(len(mat)-1);keep=ld>EPS
            ratios[f]['leave_one_out_range']=[float((ln[keep]/ld[keep]).min()),float((ln[keep]/ld[keep]).max())] if keep.any() else None
    return dict(cells=len(rows),sources=len(sources),metrics=metrics,capacity_gain_shares=ratios,
        bootstrap_draws=10000,seed=20261003,
        positive_gain_cells=sum(r['capacity_gain']>EPS for r in rows),
        positive_gain_sources=int(sum(x>EPS for x in mat[:,at['capacity_gain']])),
        source_macro_then_shared_source_ratio=True)

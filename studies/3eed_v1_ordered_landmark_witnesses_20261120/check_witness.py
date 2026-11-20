"""Independent small checks for first-hit masses, slab geometry and counterfactual labels."""
import numpy as np
from paths_witness import WORK, OUT, read, save
from ordered_choice import ordered_mass, joint, choices
from chord_shape import chords,estimate,weighted_median

def verify():
    rng=np.random.default_rng(89132);error=0.;edge_error=0.
    for _ in range(96):
        p=rng.random(14);order=rng.permutation(len(p));bins=np.unique(np.r_[0.,p,1.])
        explicit=np.zeros(len(p))
        for lo,hi in zip(bins[:-1],bins[1:]):
            selected=[i for i in order if p[i]>(lo+hi)/2]
            if selected:explicit[selected[0]]+=hi-lo
        result=ordered_mass(p,order)
        error=max(error,abs(result-explicit).max())
        np.testing.assert_allclose(result,explicit,atol=2e-15,rtol=0)
        np.testing.assert_allclose(result.sum(),p.max(),atol=2e-15)
        box=np.r_[rng.normal(size=3),rng.uniform(.3,5,3),rng.uniform(-1.5,1.5)]
        k,valid=chords(box,box[None])
        assert valid.all()
        exp=np.stack((-box[3:6]/2,box[3:6]/2),axis=1)
        edge_error=max(edge_error,abs(k[0]-exp).max())
        np.testing.assert_allclose(k[0],exp,atol=2e-14)
        np.testing.assert_allclose(estimate(box,k,np.ones(1)),box,atol=2e-14)
        v=rng.normal(size=17);w=rng.uniform(size=17)
        med=weighted_median(v,w,0.)
        optimum=min(w@abs(v-point) for point in v)
        assert abs(w@abs(v-med)-optimum)<1e-10
    d=np.array([[.1,.9],[.9,.1]])
    for r,i in (([1.,0.],0),([0.,1.],1)):
        mat=joint(np.ones(2),np.array(r),d,True)
        assert choices(mat,np.eye(2),np.eye(2))==(i,i)
    assert np.isclose(ordered_mass(np.array([.2,.2,.7]),np.arange(3)).sum(),.7)
    assert ordered_mass(np.zeros(4),np.arange(4)).sum()==0
    # Equal view masses can reach 0.5 a fraction of a floating point unit below
    # or above the crossover. The preferred face lies inside the minimizing interval.
    for sign in (-1.,1.):
        w=np.array([.25,.25+sign*1e-16,.5-sign*1e-16])
        assert weighted_median(np.array([0.,0.,2.]),w,1.)==1.
    truth=read(WORK/'labels.json')
    rows=read(WORK/'roster.json');groups={}
    for entry in rows:
        label=truth[entry['uid']]
        if 'pair' not in label:continue
        key=(label['pair'],entry['source'])
        groups.setdefault(key,{})[label['half']]=(label,entry['frame'])
    for parts in groups.values():
        (a,x),(b,y)=parts[0],parts[1]
        assert x['text']!=y['text'] and {k:v for k,v in x.items() if k!='text'}=={k:v for k,v in y.items() if k!='text'}
        assert len({a['subject'],a['reference'],b['subject'],b['reference']})==4
        assert a['truth']==b['truth']
    save(OUT/'math_checks.json',dict(trials=96,max_threshold_identity_error=error,max_slab_error=edge_error,
        rank_crossing_unit='two subjects reverse with the administered landmark',pair_directions=len(groups)))

if __name__=='__main__':verify()

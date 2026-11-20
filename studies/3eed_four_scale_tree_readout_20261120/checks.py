"""Compare dense and tree calculations on empty, tied and degenerate forests."""
import argparse
from pathlib import Path

import numpy as np

from replay import OUT, close, previous, regions, save, selection, tree


def check(package):
    r0,s0=previous(package)
    rng=np.random.default_rng(2741)
    trials=0
    # Empty, single-root, shared-ancestor, unary-branch and wide forests.
    forests=[[],[-1],[2,2,-1,-1],[1,2,3,-1,-1],[-1]*12,[6,6,7,7,8,8,9,9,10,10,-1]]
    for parents in forests:
        parent=np.array(parents,int)
        n=len(parent)
        size=rng.integers(8,20,n)
        for i,p in enumerate(parent):
            if p>=0:size[p]+=size[i]
        for _ in range(16):
            stop=rng.choice([0.,1.,.25,.625,1.-1e-12,1e-14],n)
            p=rng.random((n,3))
            p[rng.random(p.shape)<.15]=0.
            xx=rng.integers(-3,4,n)
            view=np.stack([xx-.5,xx,xx+.5,np.ones(n),size],axis=1)
            view[rng.random(n)<.2]=np.nan
            occupancy,K=s0.structure(parent,stop,size)
            np.testing.assert_array_equal(tree.disjoint_pairs(parent),K==0)
            field=tree.SideEvidence(parent,stop,view)
            for words in ((0,0),(0,1)):
                for side in (-1,1):
                    joint=s0.closed_joint(parent,stop,occupancy,p,words,view,K==0,side)
                    h=field.marginal(p[:,words[0]],p[:,words[1]],side)
                    np.testing.assert_array_equal(h>0,joint.sum(1)>0)
                    np.testing.assert_allclose(h,joint.sum(1),rtol=1e-12,atol=1e-300)
                    close(tree.overlap_sum(parent,size,h),K@h)
                    trials+=1
    # Deliberately identical feasible subject and reference utilities. The first
    # node is preferred on each side; touching projected intervals are ineligible.
    parent=np.full(4,-1)
    xyz=np.arange(4.)[:,None]*np.array([[1.,0.,0.]])
    data=dict(parent=parent,stop=np.ones(4),p=np.full((4,2),.5),size=np.ones(4,int),
              view=np.array([[0,0,0,1,1],[0,0,0,1,1],[1,1,1,1,1],[1,1,1,1,1]],float),
              assign=np.arange(4)[None,:])
    empty=dict(t_box=np.empty((0,7)),t_p=np.empty(0),t_project=np.empty((0,8)))
    pts=regions.Returns(xyz)
    assert selection.select(data,empty,(0,1),-1,0,False,pts)==(0,2)
    assert selection.select(data,empty,(0,1),1,0,False,pts)==(2,0)
    data['view'][:,:3]=0.
    assert selection.select(data,empty,(0,1),1,0,False,pts)==(-1,-1)
    # Test label numbering, inclusive ties, repeated/unsorted thresholds and
    # below-minimum components which merge at a later cut.
    for points in (np.zeros((8,3)),rng.normal(size=(180,3)),np.indices((6,6,6)).reshape(3,-1).T):
        for indices in ((0,2,5,7),tuple(range(8)),(5,0,5,2)):
            a=r0.levels(points,np.ones(len(points),bool),indices)
            b=regions.levels(points,np.ones(len(points),bool),indices)
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    OUT.mkdir(exist_ok=True)
    save(OUT/'edge_cases.json',dict(forest_comparisons=trials,graph_comparisons=9,
                 first_tie_preserved=True,touching_intervals_excluded=True,empty_votes_neutral=True,
                 same_noun=True,hidden_nodes=True,tiny_sibling_contributions=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('previous_package',type=Path)
    check(parser.parse_args().previous_package)

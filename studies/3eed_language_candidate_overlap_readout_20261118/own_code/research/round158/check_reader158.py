"""Counterfactual and geometry checks; no data or task labels required."""
import numpy as np
from reader158 import overlaps,order_effect,location_intervals,derangement,select_from_description


def main():
    box=np.array([[0,0,0,2,2,2],[1,0,0,2,2,2],[4,0,0,2,2,2]],float)
    assert np.allclose(overlaps(box),[[1,1/3,0],[1/3,1,0],[0,0,1]])
    r=np.array([[[0,.2],[.2,.4]],[[.2,.2],[.4,.4]],[[.1,.2],[.3,.4]]])
    order=order_effect(r,np.ones(3,bool),np.eye(3))
    assert np.allclose(order,-order.T) and np.allclose([order[0,1],order[0,2]],[1,.75])
    assert np.allclose(order_effect(r,np.array([1,0,1],bool),np.eye(3))[1],0)
    assert location_intervals('object. Position information: upper right of the image.',None)==[(0,.5,1),(1,0,.5)]
    assert not location_intervals('a car to the left of another car',(2,5))
    assert not location_intervals('a car facing left in the image',(2,5))
    assert not np.any(derangement('key',256)==np.arange(256))
    rng=np.random.default_rng(13);box=np.c_[rng.uniform(1,50,(256,3)),rng.uniform(.1,2,(256,3))]
    x=rng.uniform(size=(3,256)); uv=rng.random((256,2));rect=np.stack([uv,uv+.02],axis=1)
    ev=dict(A=x,B=x+10,boxes=box,box_sha='synthetic',coverage=np.zeros(256),seen=np.ones(256,bool),rect=rect,signs=np.array([-1]))
    controls={k:np.ones(256)/256 for k in ('frozen_A','joint_zero','R145')}
    first,_=select_from_description(ev,'sample',(0,6),controls)
    assert first['role_risk']==first['new_A_only']==first['misbound_B']
    perm=rng.permutation(256)
    ev.update(A=x[:,perm],B=x[:,perm],boxes=box[perm],seen=ev['seen'][perm],rect=rect[perm])
    second,_=select_from_description(ev,'sample',(0,6),controls)
    assert perm[second['role_risk']]==first['role_risk']
    print('PASS, overlap geometry, unknown evidence, directed order, camera text scope, candidate permutation and no-B path')

if __name__=='__main__':main()

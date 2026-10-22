import numpy as np
from stvg.paper_panel import interpolation,component_box

def test_interpolation_uses_physical_time_and_log_extent():
    E=np.tile([.1,.1,.2,.2],(6,1));ids=[0,2,7,11,14,20];pos=[1,4];b=np.array([[.2,.3,.2,.4],[.8,.9,.8,.1]])
    a=interpolation(E,ids,pos,b);w=(7-2)/(14-2)
    assert np.array_equal(a[0],E[0])and np.array_equal(a[-1],E[-1])
    assert np.allclose(a[2,:2],(1-w)*b[0,:2]+w*b[1,:2])
    assert np.allclose(a[2,2:],np.exp((1-w)*np.log(b[0,2:])+w*np.log(b[1,2:])))

def test_invalid_anchor_falls_back_without_GT():
    E=np.tile([.1,.1,.2,.2],(5,1));b=np.array([[.3,.3,.2,.2],[.5,.5,-1,.2]])
    a=interpolation(E,[0,1,2,3,4],[1,3],b)
    assert np.array_equal(a[1],b[0])and np.array_equal(a[3],E[3])

def test_component_keeps_strict_threshold_and_largest_mass():
    p=np.zeros((4,5));p[0,0]=1.;p[2:,3:]=.8;p[0,1]=.5
    assert np.allclose(component_box(p),[.8,.75,.4,.5])
    assert component_box(np.zeros((2,3)))is None

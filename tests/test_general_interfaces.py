import numpy as np
import pytest
from interfaces.local import LocalFrame,aggregate_frame,aggregate_sequence,grid_regions
from interfaces.temporal import JointGridPosterior,CandidateListPrior
from interfaces.prototype import feedback
from stvg.prototype import one_step

@pytest.mark.parametrize('d',[1,7,256,768,1024])
def test_dimension_free_feedback_exact_selected_math(d):
 rng=np.random.default_rng(7);z=rng.normal(size=(12,d));q=np.triu(rng.uniform(size=(12,12)));q/=q.sum()
 t=JointGridPosterior(np.arange(12),q,lambda times,x:x)
 new,trace=feedback(z,t);old,ref=one_step(z,q)
 assert np.array_equal(new,old)and np.array_equal(trace['L'],ref['L'])

def test_irregular_local_support_and_fallback():
 v=np.array([[1.,2,3],[100.,200,300]]);regions=np.array([[0,0,.5,1],[.5,0,1,1]])
 f=LocalFrame(5.,v,regions,np.ones(2,bool))
 assert np.array_equal(aggregate_frame(f,[.25,.5,.25,1],1.)[0],v[0])
 p,stats=aggregate_frame(f,[5.,5.,.1,.1],1.);assert stats['fallback']and np.array_equal(p,v.mean(0))

def test_explicit_spatial_correspondence_required():
 with pytest.raises(ValueError):LocalFrame(0.,np.ones((2,4)),np.ones((1,4)),np.ones(2,bool))
 f=LocalFrame(0.,np.ones((2,4)),grid_regions(1,2),np.ones(2,bool),(1,2))
 with pytest.raises(ValueError):aggregate_sequence([f,f],np.ones((2,4)))

def test_explicit_candidate_contract_and_point_support():
 prior=CandidateListPrior(np.array([0,1,2]),np.array([.6,.4]),np.array([[0,2],[1,3]]),np.array([[1,1,0],[0,1,1]]),lambda spans,q:spans[q.argmax()])
 np.testing.assert_array_equal(prior.soft_membership(),[.6,1.,.4]);prior.require_alternatives()
 new=prior.revise(np.array([-1.,0.,1.]));assert new[1]>.4
 point=CandidateListPrior(np.array([0,1,2]),np.array([1.]),np.array([[0,2]]),np.array([[1,1,0]]),lambda spans,q:spans[0])
 with pytest.raises(ValueError,match='Point-only'):point.require_alternatives()
 assert np.array_equal(point.revise(np.array([-100.,10.,100.])),[1.])

def test_synthetic_class_free_event_positive_control():
 z=np.zeros((20,3));z[:,0]=1;z[6:14]=[0,1,0];q=np.triu(np.full((20,20),.001));q[5,14]=1;q/=q.sum()
 t=JointGridPosterior(np.arange(20),q,lambda _,x:x);_,s=feedback(z,t)
 assert s['L'][6:14].min()>0 and s['L'][:4].max()<0

def test_regular_grid_pool_matches_selected_ta_coordinates_bitwise():
 from stvg.representation import weights,pool
 rng=np.random.default_rng(6);h,w=7,10
 for d in [7,256]:
  values=rng.normal(size=(h,w,d));valid=np.ones((h,w),bool);box=np.array([.37,.58,.21,.33])
  frame=LocalFrame(10.,values.reshape(-1,d),grid_regions(h,w),valid.ravel(),(h,w))
  z,_=aggregate_frame(frame,box);support,_=weights(box,h,w,valid)
  assert np.array_equal(z,pool(values,support))

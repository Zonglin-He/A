"""Synthetic cross-dataset dense metric compatibility; no labels or checkpoints."""
import numpy as np
import pytest
from scripts.interval_pool.score import score_boxes

@pytest.mark.parametrize('dataset',['hc2','vidstg'])
def test_domain_specific_official_and_dense_agree(dataset):
    ids=[0,4,10];record=dict(dataset=dataset,metadata=dict(frame_ids=ids,width=100,height=100,fps=10.))
    boxes=np.tile([.5,.5,.2,.2],(3,1));truth={i:[40.,40.,60.,60.]for i in range(2,9)}
    v,error=score_boxes(boxes,record,truth,[0,11],[2,9],[0,2])
    assert error<2e-10 and abs(v['t']-7/11)<1e-12 and abs(v['v']-7/11)<1e-12
    assert v['boundary_MAE_seconds']==.2 and v['interval_length_seconds']==1.1

"""Synthetic endpoint conventions; no real annotation or saved prediction scoring."""
import numpy as np
import pytest
from stvg.metrics import DenseTube,official


@pytest.mark.parametrize('dataset',['hc2','vidstg'])
def test_closed_gt_boxes_and_half_open_temporal_interval_are_distinct(dataset):
    row={'input':{'width':10,'height':10},'frame_ids':[0,1,2]}
    boxes=np.asarray([[.5,.5,.2,.2]]*3)
    truth={0:[4,4,6,6],1:[4,4,6,6],2:[0,0,1,1]}
    actual=official(boxes,row,truth,[0,2],[0,2],dataset)
    independent=DenseTube(boxes,row,truth,[0,2],clip=dataset=='hc2').score([0,2])
    assert actual['v']==pytest.approx(1.)and actual['t']==1.
    assert actual['s']==pytest.approx(2/3)
    assert independent==pytest.approx(actual)
    missing=official(boxes,row,{k:v for k,v in truth.items()if k<2},[0,2],[0,2],dataset)
    assert missing['v']==pytest.approx(actual['v'])
    assert missing['s']==pytest.approx(1.)

import pytest
from vg_tta.desta3d_v3_data import balanced_epoch,physical_grid,box_xyxy
from scripts.prepare_desta3d_v3_source import validation_parents
def test_balanced_complete_epoch_and_seed():
    rows=[dict(domain='Vid') for _ in range(11)]+[dict(domain='HC1') for _ in range(3)]
    a=balanced_epoch(rows,0);assert len(a)==22 and set(a)==set(range(14))
    assert sum(rows[i]['domain']=='Vid' for i in a)==11
    assert a==balanced_epoch(rows,0) and a!=balanced_epoch(rows,1)
def test_parent_split_not_query_count_weighted():
    parents=['a']*100+['b','c','d','e','f','g','h','i','j']
    assert validation_parents(parents)==validation_parents(set(parents))
    assert len(validation_parents(parents))==1
def test_physical_grid_and_short_support():
    assert physical_grid(2,9,1)==list(range(2,9))
    a=physical_grid(100,200,25);assert len(a)==8 and a[0]==100 and a[-1]==199
    assert len(physical_grid(0,5000,25))==32
    with pytest.raises(ValueError):physical_grid(0,0,25)
def test_box_xywh_official_conversion_and_invalid():
    b,ok=box_xyxy([10,20,30,40],100,100,True);assert ok and b==[.1,.2,.4,.6]
    b,ok=box_xyxy([20,20,10,30],100,100);assert not ok and b==[0.,0.,0.,0.]

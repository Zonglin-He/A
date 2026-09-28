import numpy as np
import pytest
from vg_tta.external_privileged_views import (repeated_frame_indices,time_to_physical,
    parse_teacher_text,dense_spatial_support,temporal_view,spatial_view)


def test_uniform_physical_slots_not_observation_indices():
    ix,ids=repeated_frame_indices([10,11,90])
    slots=np.linspace(10,90,100)
    assert np.array_equal(ix,np.abs(np.array([10,11,90])[:,None]-slots).argmin(0))
    assert time_to_physical(.5,[10,11,90])==50
    assert time_to_physical(.5,[10,11,90],clip_bounds=[0,100])==50
    assert len(ix)==100 and set(ids)=={10,11,90}
    assert repeated_frame_indices([0,10],count=3)[0].tolist()==[0,0,1]
    with pytest.raises(ValueError):repeated_frame_indices([2,2])
    with pytest.raises(ValueError):repeated_frame_indices([2,8],clip_bounds=[3,8])


def test_duplicate_valid_boxes_aggregate_without_global_rejection():
    text='{0,1} .30:[.1,.1,.5,.5] .31:[.2,.2,.6,.6] .32:[.3,.3,.7,.7]'
    e=parse_teacher_text(text,list(range(0,100,10)))
    assert e['spatial_usable'] and e['format_valid'] and len(e['boxes'])==3
    group=e['frame_groups'][0];assert group['duplicate_count']==2
    assert np.allclose(group['box'],[.2,.2,.6,.6]) and len(group['pairwise_iou'])==3
    assert group['coordinate_range']==pytest.approx([.2]*4)


def test_local_bad_box_preserved_and_frame_falls_back_without_poisoning_neighbors():
    x=np.random.default_rng(7).integers(0,256,(5,8,10,3),dtype=np.uint8)
    e=parse_teacher_text('{0,1} 0:[.2,.2,.8,.8] .5:[.7,.1,.2,.8] 1:[.2,.2,.8,.8]',[0,1,2,3,4])
    assert e['spatial_usable'] and not e['format_valid'] and len(e['boxes'])==3
    y,d=spatial_view(x,[0,1,2,3,4],e)
    assert np.array_equal(x[1:4],y[1:4])  # bad anchor is also an interpolation barrier
    assert d['support_kind'][2]=='invalid_local_fallback'
    assert not np.array_equal(x[0],y[0]) and not np.array_equal(x[4],y[4])


def test_raw_malformed_and_out_of_range_entries_not_removed():
    e=parse_teacher_text('{0,1} 2:[0,0,1,1] .5:[nan,0,1,1] 0:[0,0,1,1]',[0,10])
    assert len(e['boxes'])==3 and e['spatial_usable'] and not e['format_valid']
    assert 'invalid_box_time' in e['errors'] and 'invalid_box_format' in e['errors']
    e=parse_teacher_text('{<TEMP-0><TEMP-99>} <TEMP-99>:[<WIDTH-0><HEIGHT-0><WIDTH-99><HEIGHT-99>]',[10,11,90])
    assert e['interval_physical']==[10,90] and e['frame_groups'][0]['physical_frame']==90


def test_gap_coverage_diagnostics_without_new_threshold():
    e=parse_teacher_text('{0,1} .2:[.1,.1,.9,.9] .8:[.2,.2,.8,.8]',[0,2,5,8,10])
    boxes,d=dense_spatial_support([0,2,5,8,10],e)
    assert boxes[0] is boxes[4] is None
    assert boxes[2]==pytest.approx([.15,.15,.85,.85])
    assert d['coverage_fraction']==.6 and d['max_anchor_gap_physical']==6 and d['gap_limit'] is None


def test_temporal_identity_outside_only_no_frame_removal():
    x=np.full((3,4,8,3),120,np.uint8)
    assert np.array_equal(temporal_view(x,[1,7,30],[1,30])[0],x)
    y,d=temporal_view(x,[1,7,30],[7,7]);assert y.shape==x.shape and np.array_equal(y[1],x[1])
    assert (y[[0,2]]==30).all() and np.array_equal(temporal_view(x,[1,7,30],None)[0],x)


def test_spatial_full_box_identity_and_geometry_preservation():
    x=np.random.default_rng(7).integers(0,256,(3,8,10,3),dtype=np.uint8)
    e=parse_teacher_text('{0,1} 0:[0,0,1,1] 1:[0,0,1,1]',[1,7,30])
    assert np.array_equal(spatial_view(x,[1,7,30],e)[0],x)
    e=parse_teacher_text('{0,1} .2:[.2,.25,.8,.75]',[1,7,30])
    y,_=spatial_view(x,[1,7,30],e)
    assert np.array_equal(x[[0,2]],y[[0,2]]) and np.array_equal(x[1,2:6,2:8],y[1,2:6,2:8])

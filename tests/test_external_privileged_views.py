import numpy as np
import pytest
from vg_tta.external_privileged_views import repeated_frame_indices,time_to_physical,parse_teacher_text,temporal_view,spatial_view

def test_repeated_mapping_has_no_unseen_frames_and_uses_physical_support():
    ix,ids=repeated_frame_indices([10,11,90])
    assert len(ix)==100 and set(ids)=={10,11,90}
    assert time_to_physical(0,ids)==10 and time_to_physical(1,ids)==90
    assert time_to_physical(.5,ids)==11  # Not linear physical timestamp 50.
    with pytest.raises(ValueError):repeated_frame_indices([2,2])

def test_parse_all_failures_retained_and_no_geometry_cleanup():
    text='{<TEMP-0><TEMP-99>} <TEMP-0>:[<WIDTH-0><HEIGHT-0><WIDTH-99><HEIGHT-99>]'
    x=parse_teacher_text(text,[3,13]);assert x['interval_physical']==[3,13] and x['spatial_usable']
    bad=parse_teacher_text('{0,1} 0,: [0.7,0.1,0.2,0.8] 0,: [0,0,1,1]',[3,13])
    assert len(bad['boxes'])==2 and not bad['spatial_usable'] and 'duplicate_physical_box_time' in bad['errors']
    assert not parse_teacher_text('no output',[3,13])['temporal_usable']

def test_temporal_identity_and_outside_only_no_frame_removal():
    x=np.full((3,4,8,3),120,np.uint8)
    y,d=temporal_view(x,[1,7,30],[1,30]);assert np.array_equal(x,y)
    y,d=temporal_view(x,[1,7,30],[7,7]);assert y.shape==x.shape and np.array_equal(y[1],x[1])
    assert (y[[0,2]]==30).all()
    assert np.array_equal(temporal_view(x,[1,7,30],None)[0],x)

def test_spatial_full_box_identity_partial_box_preserved_and_no_extrapolation():
    x=np.random.default_rng(7).integers(0,256,(3,8,10,3),dtype=np.uint8)
    e=parse_teacher_text('{0,1} 0,: [0,0,1,1] 1,: [0,0,1,1]',[1,30])
    assert np.array_equal(spatial_view(x,[1,7,30],e)[0],x)
    e=parse_teacher_text('{0,1} 0.2,: [0.2,0.25,0.8,0.75] 0.3,: [0.2,0.25,0.8,0.75]',[1,30])
    y,_=spatial_view(x,[1,7,30],e)
    assert np.array_equal(x[[0,2]],y[[0,2]]) and np.array_equal(x[1,2:6,2:8],y[1,2:6,2:8])

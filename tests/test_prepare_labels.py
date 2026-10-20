"""Synthetic official-label conversion checks; no real annotations or model."""
import pytest
from tools.prepare_labels import prepare


def test_hc_official_membership_and_revised_boxes_are_used():
    meta=dict(caption='Target.',video_path='/synthetic/a.mp4',source='parent',width=10,height=8,frame_count=20,index=7)
    base=[dict(id='a',dataset='hc2',key='seed41',metadata=meta),dict(id='extra',dataset='hc2',key='other',metadata={**meta,'video_path':'/synthetic/extra.mp4'})]
    anno={'a.mp4':dict(caption='target.',width=10,height=8,img_num=20,st_frame=3,bbox=[[1,2,3,4]]*3)}
    records,labels=prepare('hc2',anno,base)
    assert records==base[:1]and labels[0]['interval']==[2,4]
    assert labels[0]['boxes']['4']==[1,2,4,6]
    with pytest.raises(ValueError,match='complete official'):
        prepare('hc2',{**anno,'unavailable.mp4':anno['a.mp4']},base)


def test_vid_preserves_closed_box_endpoint_and_official_segment_end():
    meta=dict(caption='Target?',video_path='/synthetic/p.mp4',source='p',width=10,height=8,frame_count=100,index=7,start_frame=3,end_frame=8)
    base=[dict(id='v',dataset='vidstg',key='seed41',metadata=meta)]
    anno={'7':dict(vid='p',sentence={'description':'target?'},width=10,height=8,frame_count=5,used_segment={'begin_fid':3,'end_fid':7},ori_temp_gt={'begin_fid':3,'end_fid':8},target_bboxs=[dict(xmin=1,ymin=2,xmax=4,ymax=6)]*5)}
    records,labels=prepare('vidstg',anno,base)
    assert records==base and labels[0]['interval']==[3,7]
    assert set(labels[0]['boxes'])==set(map(str,range(3,8)))
    assert meta['frame_count']==100  # Full-video length is not the segment length.

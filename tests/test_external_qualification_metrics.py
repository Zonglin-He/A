import copy,json
import pytest
import torch
from vg_tta.external_privileged_views import parse_teacher_text
from vg_tta.external_evidence_metrics import external_prediction,scalar_external,tensor_metrics
from scripts.score_desta3d_v2_reference_audit import score_tube_independently

def label():
    return dict(frame_ids=[10,11,50,90],box_valid=[True]*4,event_active=[True]*4,
                boxes_xyxy=[[.1,.2,.8,.9]]*4,event_interval=dict(begin_fid=10,end_fid=91))

def test_complete_identical_and_physical_time_support():
    lab=label();ev=parse_teacher_text('{0,1} 0:[.1,.2,.8,.9] 1:[.1,.2,.8,.9]',lab['frame_ids'])
    p=external_prediction(ev,lab['frame_ids']);a=scalar_external(p,lab);b=tensor_metrics(p,lab,external=True)
    assert all(a[m]==pytest.approx(1) and b[m]==pytest.approx(1) for m in b)
    p['interval_physical']=[50,90]
    a=scalar_external(p,lab);b=tensor_metrics(p,lab,external=True)
    assert a['tIoU']==pytest.approx(41/81) and a['vIoU']==pytest.approx(.5)
    assert all(a[m]==pytest.approx(b[m]) for m in b)

def test_invalid_local_retained_without_video_rejection_and_empty_time():
    lab=label();ev=parse_teacher_text('{0,1} 0:[.1,.2,.8,.9] .0125:[0,0,0,0] 1:[.1,.2,.8,.9]',lab['frame_ids'])
    p=external_prediction(ev,lab['frame_ids']);a=scalar_external(p,lab)
    assert not a['format_ok'] and a['sIoU']==pytest.approx(.5) and a['tIoU']==1
    p['interval_physical']=None;a=scalar_external(p,lab);b=tensor_metrics(p,lab,external=True)
    assert a['sIoU']==pytest.approx(.5) and a['tIoU']==a['vIoU']==0
    assert all(a[m]==pytest.approx(b[m]) for m in b)

def test_native_matches_independent_scalar_including_zero_geometry_and_failure():
    lab=label();p=dict(frame_ids=lab['frame_ids'],positions=[0,2,3],interval=[0,3],format_ok=True,
        boxes_cxcywh=torch.tensor([[.45,.55,.7,.7],[0,0,0,0],[.45,.55,.7,.7]],dtype=torch.float64),geometry_valid=torch.tensor([True,False,True]))
    for failed in (False,True):
        p['format_ok']=not failed;a=score_tube_independently(p,lab);b=tensor_metrics(p,lab)
        assert all(a[m]==pytest.approx(b[m],abs=1e-14) for m in b)

def test_malformed_nonfinite_raw_survives_json_and_valid_frames_survive():
    ev=parse_teacher_text('{0,1} 1e999:[.1,.2,.8,.9] .1:[0,0,1e999,1] 1:[.1,.2,.8,.9]',[10,11,90])
    json.dumps(ev,allow_nan=False)
    assert ev['boxes'][0]['normalized_time'] is None and ev['boxes'][1]['box'][2] is None
    assert ev['spatial_usable'] and '1e999' in ev['boxes'][0]['raw_match']

def test_full_seal_first_scorer_synthetic_16_parent_control(tmp_path,monkeypatch):
    from scripts import score_desta3d_v3_privileged_qualification as scorer
    from vg_tta.external_qualification_io import write,sha,seal
    base=tmp_path/'panel';teacher=tmp_path/'teacher';dest=tmp_path/'policy';rows=[];labels=[]
    monkeypatch.setattr(scorer,'OUT',tmp_path);monkeypatch.setattr(scorer,'PANEL',base)
    for i in range(16):
        lab=label();key=f'query:{i}';source=f'parent:{i}'
        row=dict(key=key,source=source,input=dict(frame_ids=lab['frame_ids'],video_sha256='video'))
        rows.append(row);labels.append(dict(lab,key=key,source=source));ep=dest/'episodes'/f'{i:02}';te=teacher/'episodes'/f'{i:02}'
        ev=parse_teacher_text('{0,1} 0:[.1,.2,.8,.9] 1:[.1,.2,.8,.9]',lab['frame_ids'])
        write(te/'INPUT.json',dict(key=key,frame_ids=lab['frame_ids'],pixel_sha256='pixel'))
        write(te/'EVIDENCE.json',ev);write(te/'SUPPORT.json',dict(coverage_fraction=1))
        write(ep/'INPUT.json',dict(key=key,frame_ids=lab['frame_ids'],physical_pixel_sha='pixel',evidence_sha=sha(te/'EVIDENCE.json')))
        write(ep/'BASELINE_REPLAY.json',dict(equal=True))
        p=dict(key=key,source=source,frame_ids=lab['frame_ids'],video_sha256='video',adapter_sha='adapter',
            positions=list(range(4)),interval=[0,3],format_ok=True,boxes_cxcywh=torch.tensor([[.45,.55,.7,.7]]*4),
            geometry_valid=torch.ones(4,dtype=torch.bool),support=dict(grid='grid'),preprocess=dict(pixel_sha='pixel'),
            evidence_sha=sha(te/'EVIDENCE.json'),GT_read=False,target_GT_read=False,optimizer_steps=0)
        for arm in scorer.ARMS:torch.save(p,ep/(arm+'.pt'))
    write(base/'SOURCE_RECORDS.json',labels)
    for run,n in ((teacher,16),(dest,64)):
        write(run/'INPUTS.json',rows);write(run/'LOCK.json',dict(pins={}))
        write(run/'COMPLETE.json',dict(queries=16,seal_sha=seal(run,n)))
    write(dest/'CONFIG.json',dict(teacher=str(teacher),adapter_sha='adapter',source_label_sha=sha(base/'SOURCE_RECORDS.json')))
    scorer.preflight('policy');scorer.score('policy')
    report=json.loads((dest/'independent_readback_v1/REPORT.json').read_text())
    assert report['arms']['original']['summary']['parents']==16
    assert all(c['vIoU']['mean_delta_pp']==0 for a,c in report['comparisons'].items() if not a.startswith('external'))
    assert (dest/'independent_readback_v1/PRE_GT_AUDIT.json').exists()
    from scripts import crosscheck_desta3d_v3_privileged_summary as cross
    monkeypatch.setattr(cross,'OUT',tmp_path);cross.run('policy')
    # Separate run with the same locked files: corrupting one byte must fail before any label read.
    import shutil
    bad=tmp_path/'bad';shutil.copytree(dest,bad);shutil.rmtree(bad/'independent_readback_v1')
    with (bad/'episodes/00/original.pt').open('ab') as f:f.write(b'changed')
    original_read=scorer.read
    def guarded_read(path):
        assert path!=base/'SOURCE_RECORDS.json','labels must not be opened on a broken seal'
        return original_read(path)
    monkeypatch.setattr(scorer,'read',guarded_read)
    with pytest.raises(ValueError,match='Pinned input changed'):scorer.score('bad')
    assert not (bad/'independent_readback_v1/PRE_GT_AUDIT.json').exists()

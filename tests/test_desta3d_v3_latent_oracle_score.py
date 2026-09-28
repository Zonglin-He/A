import json,shutil
import pytest
import torch

def test_sealed_four_arm_oracle_and_bad_seal_before_metric_labels(tmp_path,monkeypatch):
    from scripts import score_desta3d_v3_latent_oracle as scorer
    from vg_tta.desta3d_v3_oracle_io import write,sha,seal
    from vg_tta.desta3d_v3_latent_oracle import masks_from_source_record,matched_wrong_masks
    base=tmp_path/'panel';dest=tmp_path/'oracle';rows=[];labels=[]
    monkeypatch.setattr(scorer,'OUT',tmp_path);monkeypatch.setattr(scorer,'PANEL',base)
    for i in range(16):
        lab=dict(key=str(i),source='parent:'+str(i),frame_ids=[10,11,50,90],box_valid=[True]*4,event_active=[False,True,True,False],
            boxes_xyxy=[[.1,.2,.8,.9]]*4,event_interval=dict(begin_fid=11,end_fid=51))
        labels.append(lab);rows.append(dict(key=lab['key'],source=lab['source'],input=dict(frame_ids=lab['frame_ids'],video_sha256='video')))
    write(base/'SOURCE_RECORDS.json',labels);labelsha=sha(base/'SOURCE_RECORDS.json')
    for i,(row,lab) in enumerate(zip(rows,labels)):
        ep=dest/'episodes'/f'{i:02}';ep.mkdir(parents=True)
        masks=masks_from_source_record(lab,lab['frame_ids'],2,3);torch.save(masks,ep/'ORACLE_MASKS.pt')
        torch.save(matched_wrong_masks(lab,lab['frame_ids'],2,3),ep/'WRONG_MASKS.pt')
        support=dict(grid='grid');pre=dict(pixel_sha='pixel')
        write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],frame_ids=lab['frame_ids'],pixel_intervention=False,
            source_GT_mask_sha=sha(ep/'ORACLE_MASKS.pt'),source_label_sha=labelsha,support=support,preprocess=pre))
        write(ep/'BASELINE_REPLAY.json',{str(k):True for k in range(12)})
        if i==0:write(ep/'IDENTITY_CONTROL.json',{str(k):True for k in range(12)})
        write(ep/'COMPLETE.json',dict(key=row['key'],exact_state=True,optimizer_steps=0))
        p=dict(key=row['key'],source=row['source'],frame_ids=lab['frame_ids'],video_sha256='video',adapter_sha='adapter',
            positions=list(range(4)),interval=[0,3],format_ok=True,boxes_cxcywh=torch.tensor([[.45,.55,.7,.7]]*4),
            geometry_valid=torch.ones(4,dtype=torch.bool),support=support,preprocess=pre,worker_source_GT_read=True,
            decoder_GT_prefix=False,target_GT_read=False,optimizer_steps=0)
        for arm in scorer.ARMS:
            branches={'original':[],'temporal':['event'],'spatial':['spatial'],'dual':['event','spatial'],'wrong_temporal':['event'],'wrong_spatial':['spatial']}[arm]
            torch.save(dict(p,GT_read=arm!='original',oracle_latent=arm!='original',latent_log=[dict(branch=b) for b in branches]),ep/(arm+'.pt'))
    write(dest/'INPUTS.json',rows);write(dest/'LOCK.json',dict(pins={}))
    write(dest/'CONFIG.json',dict(adapter_sha='adapter',source_label_sha=labelsha,source_GT_in_worker=True,target_input=False,alpha=.25,optimizer_steps=0))
    write(dest/'COMPLETE.json',dict(queries=16,predictions=96,extra_native_identity_control=1,optimizer_steps=0,seal_sha=seal(dest,96)))
    scorer.preflight('oracle');scorer.score('oracle')
    report=json.loads((dest/'independent_readback_v1/REPORT.json').read_text())
    assert all(c['vIoU']['mean_delta_pp']==0 for c in report['comparisons'].values())
    assert report['arms']['original']['summary']['parents']==16
    from scripts import crosscheck_desta3d_v3_oracle_summary as cross
    monkeypatch.setattr(cross,'OUT',tmp_path);cross.run('oracle')
    bad=tmp_path/'bad';shutil.copytree(dest,bad);shutil.rmtree(bad/'independent_readback_v1')
    with (bad/'episodes/00/original.pt').open('ab') as f:f.write(b'corrupted')
    original=scorer.read
    def guarded(path):
        assert path!=base/'SOURCE_RECORDS.json','no metric labels on broken seal'
        return original(path)
    monkeypatch.setattr(scorer,'read',guarded)
    with pytest.raises(ValueError,match='Pinned input changed'):scorer.score('bad')
    assert not (bad/'independent_readback_v1/PRE_SCORE_AUDIT.json').exists()

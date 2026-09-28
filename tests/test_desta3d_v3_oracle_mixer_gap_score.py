"""End-to-end CPU scorer control; synthetic source labels, no GPU data."""
import json,sqlite3
import torch
from vg_tta.external_qualification_io import write,sha
from scripts.score_desta3d_v2_reference_audit import score_tube_independently

def test_full447_noop_and_second_reduction(tmp_path,monkeypatch):
    from scripts import score_desta3d_v3_oracle_mixer_gap as score
    from scripts import crosscheck_desta3d_v3_oracle_mixer_gap as root
    d=tmp_path/'gap';e=tmp_path/'old';roster=tmp_path/'roster';roster.mkdir()
    monkeypatch.setattr(score,'D',d);monkeypatch.setattr(score,'E',e);monkeypatch.setattr(score,'ROSTER',roster)
    monkeypatch.setattr(root,'D',d);monkeypatch.setattr(root,'E',d)
    label=dict(frame_ids=[10,11,50,90],box_valid=[True]*4,event_active=[True]*4,boxes_xyxy=[[.1,.2,.8,.9]]*4,event_interval=dict(begin_fid=10,end_fid=91))
    db=sqlite3.connect(roster/'SOURCE.sqlite');db.execute('CREATE TABLE examples (key text,labels_json text)')
    rows=[];old={a:dict(rows=[]) for a in score.ARMS[:-1]}
    for i in range(447):
        key=str(i);parent=str(i%31);rows.append(dict(key=key,source=parent))
        db.execute('INSERT INTO examples VALUES (?,?)',(key,json.dumps(label)))
        ep=d/'episodes'/f'{i:04}';write(ep/'INPUT.json',dict(support=dict(F='same'),preprocess=dict(pixel='same')))
        pred=dict(key=key,source=parent,frame_ids=label['frame_ids'],positions=[0,2,3],interval=[0,3],format_ok=True,
            boxes_cxcywh=torch.tensor([[.45,.55,.7,.7],[0,0,0,0],[.45,.55,.7,.7]],dtype=torch.float64),geometry_valid=torch.tensor([True,False,True]),
            support=dict(F='same'),preprocess=dict(pixel='same'),adapter_sha=score.ADAPTER_SHA,GT_read=True,decoder_GT_prefix=False,target_read=False,optimizer_steps=0,
            injection=dict(same_field_both_passes=True,relative_norm=.1))
        for a in score.ARMS:
            torch.save(pred,ep/(a+'.pt'))
            if a!='oracle':old[a]['rows'].append(dict(key=key,source=parent,metrics=score_tube_independently(pred,label)))
        write(ep/'GEOMETRY.json',dict(cosine_to_oracle={a:None for a in score.ARMS[1:]},descent_dot={b:{a:0 for a in score.ARMS[1:]} for b in ['event','spatial']},norm_over_cap={a:1 for a in score.ARMS[1:]},mixer={a:dict(fraction_abs_above_099=1) for a in score.SEEDS}))
    db.commit();db.close()
    write(d/'INPUTS.json',rows);write(d/'CONFIG.json',dict(radius=.13545580427763146));write(d/'LOCK.json',dict(pins={}))
    write(d/'ROOT_ALL_RAW_READBACK.json',dict(status='passed',episodes=447))
    write(e/'independent_readback_v1/REPORT.json',dict(arms=old))
    write(d/'PREDICTIONS_SEAL.json',dict(predictions=1788,files={str(p.relative_to(d)):sha(p) for p in (d/'episodes').rglob('*') if p.is_file()}))
    write(d/'COMPLETE.json',dict(predictions=1788,seal_sha=sha(d/'PREDICTIONS_SEAL.json')))
    score.score();root.summary()
    report=json.loads((d/'independent_readback_v1/REPORT.json').read_text())
    assert report['comparisons']['oracle']['vIoU']['mean_delta_pp']==0
    assert not report['oracle_qualified']
    assert report['geometry_groups']['all']['stats']['oracle']['cosine']['undefined']==447
    assert report['reused_metric_max_abs']==0

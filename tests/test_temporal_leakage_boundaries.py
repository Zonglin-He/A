"""Synthetic boundary perturbations: event labels must not choose input support."""
import json
from pathlib import Path
import numpy as np
import pytest
from tools import build_manifests as intake
from stvg.temporal import final_mbr


def test_physical_mbr_is_exact_expectation_over_model_mass():
    ids=[0,4,13,20,31];q=np.zeros((5,5));q[0,2]=.2;q[1,3]=.3;q[2,4]=.5
    actual=final_mbr(ids,q);u=np.full((5,5),-np.inf)
    for a in range(5):
        for b in range(a+1,5):
            s,e=ids[a],ids[b]+1;u[a,b]=0
            for i in range(5):
                for j in range(i+1,5):
                    x,y=ids[i],ids[j]+1;overlap=max(0,min(e,y)-max(s,x));u[a,b]+=q[i,j]*overlap/(e-s+y-x-overlap)
    assert np.max(abs(actual['utility_grid'][np.isfinite(u)]-u[np.isfinite(u)]))<1e-12
    a,b=np.unravel_index(np.argmax(u),u.shape);assert actual['interval']==[ids[a],ids[b]+1]


@pytest.mark.parametrize('dataset',['hc2','vidstg'])
def test_real_intake_function_event_label_perturbation_does_not_change_inputs(tmp_path,monkeypatch,dataset):
    dummy=tmp_path/'dummy.mp4';dummy.write_bytes(b'synthetic media never decoded');annotation=tmp_path/'synthetic.json';raw={};media={}
    if dataset=='hc2':
        for i in range(1902):
            name=f'{i}_parent{i%227}.mp4';raw[name]=dict(width=10,height=10,English='synthetic query',img_num=10,st_frame=1,bbox=[[1,1,2,2]]*3);media[name]=dummy
    else:
        raw=[];tracks={};left=10303
        for i in range(732):
            n=min(15,left-(731-i));left-=n;vid=str(i);name=f'v{i}.mp4';media[name]=dummy
            raw.append(dict(vid=vid,width=10,height=10,fps=30.,frame_count=10,used_segment=dict(begin_fid=0,end_fid=9),temporal_gt=dict(begin_fid=1,end_fid=4),captions=[dict(description='synthetic query',target_id=0)]*n,questions=[]))
            tracks[vid]=dict(video_path=name,trajectories=[[dict(tid=0,bbox=dict(xmin=1,ymin=1,xmax=3,ymax=3))]for _ in range(10)])
        assert left==0;monkeypatch.setattr(intake,'vidor_records',lambda _:tracks)
    monkeypatch.setattr(intake,'media_index',lambda _:media)
    original_sha=intake.sha;expected=intake.HC_SHA if dataset=='hc2'else intake.VID_SHA
    monkeypatch.setattr(intake,'sha',lambda p:expected if Path(p)==annotation else original_sha(p))
    annotation.write_text(json.dumps(raw));first=tmp_path/'first';second=tmp_path/'second'
    intake.build(dataset,annotation,tmp_path,first,vidor_annotations='unused_synthetic')
    if dataset=='hc2':
        for x in raw.values():x['st_frame']=4;x['bbox']=x['bbox'][:2]
    else:
        for x in raw:x['temporal_gt']=dict(begin_fid=4,end_fid=7)
    annotation.write_text(json.dumps(raw));intake.build(dataset,annotation,tmp_path,second,vidor_annotations='unused_synthetic')
    assert(first/f'{dataset}.jsonl').read_bytes()==(second/f'{dataset}.jsonl').read_bytes()
    assert(first/f'{dataset}_labels.jsonl').read_bytes()!=(second/f'{dataset}_labels.jsonl').read_bytes()

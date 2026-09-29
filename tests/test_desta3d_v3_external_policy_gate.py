"""Synthetic gate/control pipeline, never evidence of real provider utility."""
import json,hashlib
from pathlib import Path
import numpy as np,torch,pytest
from vg_tta.external_qualification_io import write,sha,seal
from vg_tta.llava_st_evidence_parser_v3 import parse_teacher_text
from vg_tta.desta3d_v3_policy_gate_views import build_views

def test_full_seal_and_conditional_correctness_route(tmp_path,monkeypatch):
 from scripts import score_desta3d_v3_external_policy_gate as score,crosscheck_desta3d_v3_external_policy_gate as cross
 from scripts import desta3d_v3_a0_fast_screen as a0
 from vg_tta import exact_frame_decode_audit_v2 as decoder
 d=tmp_path/'gate';teacher=tmp_path/'teacher';monkeypatch.setattr(score,'D',d);monkeypatch.setattr(score,'OUT',tmp_path);monkeypatch.setattr(score,'TEACHER',teacher);monkeypatch.setattr(cross,'D',d)
 frames=np.arange(3*8*8*3,dtype=np.uint8).reshape(3,8,8,3);ids=[0,1,2];rows=[];labels={};label_calls=[]
 def labels_for(selected,split):label_calls.append(len(selected));assert len(selected)==16 and split=='dev';return labels
 monkeypatch.setattr(a0,'labels_for',labels_for);monkeypatch.setattr(decoder,'decode',lambda inp:(frames.copy(),ids))
 ev=parse_teacher_text('{0,1} 0:[0,0,1,1] 1:[0,0,1,1]',ids)
 for i in range(16):
  row=dict(key=f'q{i}',source=f'p{i}',input={});rows.append(row);labels[row['key']]=dict(frame_ids=ids,boxes_xyxy=[[0,0,1,1]]*3,box_valid=[True]*3,event_active=[True]*3,event_interval=dict(begin_fid=0,end_fid=3))
  write(teacher/'episodes'/f'{i:02}'/'EVIDENCE.json',ev)
 write(teacher/'COMPLETE.json',dict(seal_sha=seal(teacher,16)));write(teacher/'ROOT_EVIDENCE_READBACK.json',dict(status='passed'))
 for stage,arms in [('policy001',['B1','T','S','TS']),('wrong001',['TS_wrong','T_wrong','S_wrong'])]:
  run=tmp_path/('extgate_'+stage);write(run/'INPUTS.json',rows);write(run/'CONFIG.json',dict(arms=arms));write(run/'LOCK.json',dict(pins={}))
  for i,row in enumerate(rows):
   ep=run/'episodes'/f'{i:02}';views,meta=build_views(frames,ids,ev)
   write(ep/'INPUT.json',dict(physical_pixel_sha=hashlib.sha256(frames.tobytes()).hexdigest(),view_metadata=meta,view_pixel_sha={a:hashlib.sha256(v.tobytes()).hexdigest() for a,v in views.items()}))
   write(ep/'BASELINE_REPLAY.json',dict(replay=True))
   for a in arms:
    good=a in ['T','S','TS'];p=dict(key=row['key'],source=row['source'],frame_ids=ids,GT_read=False,target_read=False,optimizer_steps=0,physical_support={'grid':'same'},format_ok=True,positions=ids if good else [0],interval=[0,2] if good else [0,0],boxes_cxcywh=torch.tensor([[.5,.5,1.,1.]]*(3 if good else 1)),geometry_valid=torch.ones(3 if good else 1,dtype=torch.bool));torch.save(p,ep/(a+'.pt'))
  write(run/'COMPLETE.json',dict(seal_sha=seal(run,len(arms)*16)))
  score.main(stage);cross.main(stage)
 report=json.loads((d/'wrong_evaluation/REPORT.json').read_text());assert report['preliminary_pass'] and report['final_pass'] and label_calls==[16,16]
 assert (d/'evaluation/PRE_SCORE_AUDIT.json').exists()

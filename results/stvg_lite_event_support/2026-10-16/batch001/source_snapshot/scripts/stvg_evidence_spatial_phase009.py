"""Reference extraction and complete saved-state source/decoder readback; no GT."""
import sys,copy,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def references(phase):
 verify_runtime();sys.addaudithook(guard);b=BASE/phase;bar=read(b/'TEACHER_GLOBAL_BARRIER.json');assert not bar['GT_read']
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 variants=read(b/'VARIANT_LOCK.json')['variants'];jobs=read(b/'TEACHER_JOB_LOCK.json')['records'];rows=[]
 for rec in roster('confirmation'if phase=='confirmation'else'development'):
  cid=rec['probe_alias'];c,_=load_case(rec);row=dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=c['source_case_sha256'],GT_read=False)
  for v in variants:
   anchors=[]
   for j in jobs:
    if j['case_id']!=cid or j['variant']!=v:continue
    p=b/'teacher'/(j['job_id']+'.json');z=read(p)['parsed'];chosen=z['selected_candidate_id']
    if chosen in[None,1]:continue
    tr=next(x for x in j['candidate_bindings']if int(x['id'])==chosen)
    if tr['presence']<.6:continue
    a,bb,cc,d=tr['center_box'];position=list(map(int,c['frame_ids'])).index(int(j['center_physical_frame_id']));anchors.append(dict(position=position,physical_frame_id=int(j['center_physical_frame_id']),box_cxcywh=[(a+cc)/2,(bb+d)/2,cc-a,d-bb],weight=tr['presence'],candidate_id=chosen,teacher_output_sha256=sha(p)))
   # Matching the old prefix reference is by exact candidate/reference evidence; response SHA may differ for the explicitly copied reuse record.
   if phase in ['round2','round2_prompt009']and v=='old_crop':
    old=next(x for x in read(E/'REFERENCE_LOCK.json')['records']if x['case_id']==cid)['original']['anchors']
    assert len(anchors)==len(old)
    for x,y in zip(anchors,old):assert {k:z for k,z in x.items()if k!='teacher_output_sha256'}=={k:z for k,z in y.items()if k!='teacher_output_sha256'}
    anchors=copy.deepcopy(old)
   row[v]=dict(anchors=anchors)
  rows.append(row)
 write(b/'REFERENCE_LOCK.json',dict(status='locked',records=rows,GT_read=False));print('references',len(rows),sum(bool(r[v]['anchors'])for r in rows for v in variants))
def root(phase):
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 from scripts import stvg_target_teacher_student_v1 as core
 b=BASE/phase;bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['count']==32 and not bar['GT_read'];refs={r['case_id']:r for r in read(b/'REFERENCE_LOCK.json')['records']};variants=read(b/'VARIANT_LOCK.json')['variants'];check=0;maxerr=0.;newfits=0;penalty_checks=0;penalty_error=0.
 for item in bar['files']:
  p=ROOT/item['path'];assert sha(p)==item['sha256'];d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert not d['GT_read']and d['source_restored'];newfits+=item['actual_fit_count'];t=torch.load(b/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False);assert d['arms']['T']['interval']==t['selected_interval']
  assert np.array_equal(d['arms']['F']['boxes'],d['arms']['T']['boxes'])
  for v in variants:
   aa=refs[cid][v]['anchors'];assert aa==d['reference_records'][v]['anchors'];assert d['arms']['TS_'+v]['interval']==t['selected_interval']and np.array_equal(d['arms']['S_'+v]['boxes'],d['arms']['TS_'+v]['boxes']);f=d['fits'].get(v)
   if not aa:assert f is None and np.array_equal(d['arms']['F']['boxes'],d['arms']['S_'+v]['boxes']);continue
   assert f['config']==read(C/'SPATIAL_SELECTION_LOCK.json')['configs'][d['dataset']];assert torch.equal(d['readouts'][v]['boxes'].cpu(),f['final_cached_decoder_boxes'].cpu());pos=[a['position']for a in aa];e=np.asarray([a['box_cxcywh']for a in aa]);w=np.asarray([a['weight']for a in aa]);stopped=False
   for row in f['path']:
    if row.get('skipped'):continue
    assert not stopped
    if f.get('algorithm')=='output_preserved_validated_iou':
     original_case,_=load_case(next(r for r in roster('confirmation'if phase=='confirmation'else'development')if r['probe_alias']==cid));source_boxes=original_case['zero']['boxes'].cpu().numpy();assert np.array_equal(source_boxes,row['source_boxes'].numpy());ff=np.asarray(d['frame_ids'],dtype=np.float64);at=ff[pos];span=max(1.,float(at.max()-at.min()));expected_w=1.-np.max(np.exp(-abs(ff[:,None]-at[None,:])/span)*w[None,:],axis=1);expected_w[pos]=0.;assert abs(expected_w-row['preservation_weights'].numpy()).max()<2e-7
     delta=row['boxes'].numpy()-source_boxes;value=float((expected_w*np.square(delta).sum(1)).sum()/max(1e-12,expected_w.sum()));penalty_error=max(penalty_error,abs(value-row['preservation_loss']));assert penalty_error<2e-7 and abs(row['loss']-row['opd_loss']-row['preservation_loss'])<2e-7;penalty_checks+=4
    q0=iou_cxcywh(row['boxes'].numpy()[pos],e);q1=iou_cxcywh(row['proposed_boxes'].numpy()[pos],e);maxerr=max(maxerr,float(abs(q0-row['reference_iou_before'].numpy()).max()),float(abs(q1-row['reference_iou_proposed'].numpy()).max()));assert maxerr<3e-6
    if row['accepted']:
     ww=torch.tensor(w,dtype=row['reference_iou_before'].dtype);assert float((ww*(row['reference_iou_proposed']-row['reference_iou_before'])).sum()/ww.sum())>=-1e-7
    if row['proposed']and not row['accepted']:assert core._tree_equal(row['optimizer_before'],row['optimizer_after'])and core._tree_equal(row['state'],row['update_state']);stopped=True
    check+=3
   xy=f['final_cached_decoder_boxes'].numpy();xy=np.concatenate([xy[:,:2]-xy[:,2:]/2,xy[:,:2]+xy[:,2:]/2],1);assert np.allclose(xy,d['arms']['S_'+v]['boxes'],atol=1e-7,rtol=0);check+=1
 write(b/'ROOT_SPATIAL_READBACK.json',dict(status='pass',case_count=32,new_fit_count=newfits,check_count=check,max_reference_iou_error=maxerr,preservation_penalty_checks=penalty_checks,max_preservation_penalty_error=penalty_error,GT_read=False,CPU_only=True,scope='independent reference geometry, saved optimizer rollback and actual full-vs-cached final decoder; not independent backward'))
 assert read(b/'ROOT_INPUT_READBACK.json')['status']=='pass'and read(b/'ROOT_TEMPORAL_READBACK.json')['status']=='pass'
 files=list(bar['files'])
 for p in [b/'REFERENCE_LOCK.json',b/'TEACHER_GLOBAL_BARRIER.json',b/'TEACHER_JOB_LOCK.json',b/'CANDIDATE_INPUT_BARRIER.json',b/'ROOT_INPUT_READBACK.json',b/'ROOT_TEMPORAL_READBACK.json',b/'ROOT_SPATIAL_READBACK.json',BASE/'TEMPORAL_SELECTION_LOCK.json',BASE/'DESIGN_LOCK.json',b/'RUNTIME.json']:files.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=32,files=files,design_sha256=sha(BASE/'DESIGN_LOCK.json'),scorer_sha256=sha(ROOT/'scripts/score_stvg_evidence_v1.py'),GT_read=False));print('spatial root',newfits,check,maxerr)
if __name__=='__main__':globals()[sys.argv[1]](sys.argv[2])

"""Seal actual teacher references; Oracle only after complete unlabeled method seal."""
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,roster,catalog
BASE=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1'
def main(phase):
 inputs=read(BASE/'CANDIDATE_INPUT_BARRIER.json');records=[]
 if phase=='method':
  bar=read(BASE/'TEACHER_GLOBAL_BARRIER.json');assert bar['count']==42 and bar['GT_read'] is False
  for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
  for row in inputs['records']:
   cid=row['case_id'];dst=BASE/'teacher'/f'{cid}_base.json';out=read(dst);ident=out['parsed'].get('selected_candidate_id')if out['parsed']['status']=='valid' and not out['output_truncated_at_cap']else None;chosen=next((c for c in row['candidates']if c['candidate_id']==ident),None);fallback=chosen is None;chosen=row['candidates'][0]if fallback else chosen
   def ref(c):return dict(anchors=[dict(position=p,box_cxcywh=b)for p,b in zip(row['positions'],c['boxes_cxcywh'])],candidate_id=c['candidate_id'],window_id=c['window_id'],reference_is_student_hypothesis=True)
   refs=dict(source=ref(row['candidates'][0]),teacher=ref(chosen));refs['teacher'].update(actual_response_path=str(dst.relative_to(ROOT)),actual_response_sha256=sha(dst),Source_fallback=fallback,raw_selected_candidate_id=ident)
   records.append(dict(case_id=cid,dataset=row['dataset'],source_case_sha256=row['source_case_sha256'],references=refs))
 else:
  import numpy as np
  from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
  bar=read(BASE/'METHOD_STUDENT_GLOBAL_BARRIER.json');assert bar['count']==8 and bar['GT_read']is False
  for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
  cases={r['case_id']:catalog(r)[0]for r in roster()};gt,binding=_truths_after_seal(cases);write(BASE/'ORACLE_GT_EXPOSURE_RECEIPT.json',dict(status='actual_offline_GT_read_after_complete_method_seal',GT_read=True,method_barrier_sha256=sha(BASE/'METHOD_STUDENT_GLOBAL_BARRIER.json'),bindings=binding,GT_used_in_unlabeled_method=False,time=time.time()))
  for row in inputs['records']:
   cid=row['case_id'];case=cases[cid];anchors=[];missing=[]
   for p,f in zip(row['positions'],row['physical_frame_ids']):
    if f not in gt[cid]['truth']:missing.append(dict(position=p,physical_frame_id=f,reason='no_annotated_GT_at_fixed_position'));continue
    v=np.array(gt[cid]['truth'][f],float)/np.array([case['width'],case['height'],case['width'],case['height']],float);x1,y1,x2,y2=v
    if not(0<=x1<x2<=1 and 0<=y1<y2<=1):missing.append(dict(position=p,physical_frame_id=f,reason='GT_geometry_not_legal_no_clipping'));continue
    anchors.append(dict(position=p,box_cxcywh=[(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1]))
   records.append(dict(case_id=cid,dataset=row['dataset'],source_case_sha256=row['source_case_sha256'],references=dict(offline_oracle=dict(anchors=anchors,missing=missing,reference_is_actual_GT=True,offline_nondeployment=True))))
 write(BASE/f'{phase.upper()}_REFERENCE_LOCK.json',dict(status='locked',records=records,count=8,GT_read=phase=='oracle',offline_nondeployment=phase=='oracle',teacher_barrier_sha256=sha(BASE/'TEACHER_GLOBAL_BARRIER.json')))
 paths=['scripts/run_stvg_video_global_student_v1.py','scripts/stvg_video_global_spatial_fit_v1.py','scripts/prepare_stvg_video_global_references_v1.py','scripts/run_stvg_support_aware_spatial_v1.py','scripts/stvg_temporal_caption_query_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py','scripts/run_decota_paper_main_v1.py','scripts/stvg_mllm_feedback_common_v1.py','scripts/stvg_target_teacher_student_v1.py','vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/stvg_opd_frozen_mask_cache012.py','vg_tta/stvg_opd_frozen_visual_host011.py','vg_tta/stvg_opd_frozen_attention_lifetime010.py','vg_tta/stvg_opd_frozen_module_cpu008.py','methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/tensors.py','methods/decota_final_simplified_v1/backbone.py','methods/decota_spatial_opd_v1/configs.json']
 paths += [str(p.relative_to(ROOT))for p in [BASE/f'{phase.upper()}_REFERENCE_LOCK.json',BASE/'TEACHER_GLOBAL_BARRIER.json',BASE/'CANDIDATE_INPUT_BARRIER.json']]
 old=ROOT/'artifacts/stvg_temporal_caption_query_spatial_opd_v1';paths += [str(p.relative_to(ROOT))for p in (old/'student').glob('*.pt')];paths+=[str((old/'STUDENT_GLOBAL_BARRIER.json').relative_to(ROOT))]
 write(BASE/f'{phase.upper()}_STUDENT_RUNTIME.json',dict(status='locked',phase=phase,scope='joint',steps=5,expected_fit_count=48 if phase=='method' else 24,GT_read=phase=='oracle',pins={p:sha(ROOT/p)for p in paths},protected={'methods/CURRENT_METHOD.json':sha(ROOT/'methods/CURRENT_METHOD.json')},unlabeled_method_sealed_before_oracle=phase=='oracle'))
 print(dict(phase=phase,refs=[{k:len(v['anchors'])for k,v in r['references'].items()}for r in records]))
if __name__=='__main__':main(sys.argv[1])

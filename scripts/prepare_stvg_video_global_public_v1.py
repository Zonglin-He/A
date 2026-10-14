"""Explicit public export; raw media/query/GT coordinates/weights/fit caches excluded."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1';OUT=ROOT/'results/stvg_video_global_referent/2026-10-14'
def save(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
source=json.loads((B/'source_audit/LUNA_CANDIDATE_CUSTODY_READBACK.json').read_text());compact=dict(status=source['status'],scope=source['scope'],totals=source['totals'],actual_preparation_readback_pass=True,unused_synthetic_helper_precision_and_order_are_not_actual_execution=source['helper_comparison']['actual_legal_candidates_rejected_by_float32_helper_count'],findings=source['findings'],private_geometry_and_source_paths_excluded=True)
save(OUT/'candidate_source_custody_summary.json',compact)
for phase in ['method','oracle']:
 z=json.loads((B/phase/'ROOT_CPU_FIT_MATH_READBACK.json').read_text());save(OUT/(phase+'_math_readback.json'),z)
for n in ['ROOT_READBACK','ROOT_ANONYMOUS_SCALAR_READBACK','PUBLIC_SCALAR_AUDIT','ROOT_VISUAL_REVIEW','TECHNICAL_DECISION','ACTUAL_TEACHER_WEIGHT_CUSTODY']:
 z=json.loads((B/(n+'.json')).read_text());z.pop('private_files',None);save(OUT/(n.lower()+'.json'),z)
assets=[ROOT/'docs/STVG_VIDEO_GLOBAL_REFERENT_V1.md',ROOT/'protocols/stvg_video_conditioned_global_referent_v1.md',ROOT/'output/pdf/stvg_video_global_referent_v1.pdf']+list(OUT.glob('*.json'))
sources=['prepare_stvg_video_global_referent_v1.py','run_stvg_video_global_teacher_v1.py','launch_stvg_video_global_teacher_v1.py','prepare_stvg_video_global_references_v1.py','prepare_stvg_video_global_oracle_v1.py','run_stvg_video_global_student_v1.py','launch_stvg_video_global_student_v1.py','stvg_video_global_spatial_fit_v1.py','stvg_video_global_spatial_fit_fp32_geometry001.py','run_stvg_video_global_oracle_geometry001.py','launch_stvg_video_global_oracle_geometry001.py','review_stvg_video_global_teacher_v1.py','review_stvg_video_global_spatial_math_v1.py','score_stvg_video_global_postseal_v1.py','render_stvg_video_global_private_v1.py','render_stvg_video_global_final_private_v1.py','report_stvg_video_global_referent_v1.py','prepare_stvg_video_global_public_v1.py']
assets+=[ROOT/'scripts'/n for n in sources];stage=B/'public_stage';manifest=[]
for src in assets:
 rel=str(src.relative_to(ROOT));dst=stage/rel;dst.parent.mkdir(parents=True,exist_ok=True);data=src.read_bytes();dst.write_bytes(data);manifest.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64'if src.suffix=='.pdf'else'utf-8'))
assert len({r['path']for r in manifest})==len(manifest)
assert not any(any(s in r['path']for s in ['private_inputs','private_views','.pt','.safetensors','attachments','conversation'])for r in manifest)
save(B/'PUBLIC_MANIFEST.json',manifest);print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes']for r in manifest))))

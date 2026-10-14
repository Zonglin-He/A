"""Explicit joint public allowlist, with sealed initial batch preserved."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts/stvg_video_conditioned_global_referent_v1';B=ROOT/'artifacts/stvg_scene_object_id_referent_v1'
def save(p,z):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 initial=A/'INITIAL_PUBLIC_MANIFEST.json'
 if not initial.exists():initial.write_bytes((A/'PUBLIC_MANIFEST.json').read_bytes())
 rows=json.loads(initial.read_text());paths=[r['path']for r in rows]
 for r in rows:
  assert hashlib.sha256((A/'public_stage'/r['path']).read_bytes()).hexdigest()==r['sha256']
 for n in ['ROOT_CLOSING','FINAL_COMPLETION','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK','INITIAL_GITHUB_RECEIPT','ARCHIVE_CLOSING_RECEIPT']:
  dest=ROOT/'results/stvg_video_global_referent/2026-10-14'/(n.lower()+'.json');save(dest,json.loads((A/(n+'.json')).read_text()));paths.append(str(dest.relative_to(ROOT)))
 for n in ['ROOT_CLOSING','FINAL_COMPLETION','ROOT_PUBLIC_SOURCE_BINDINGS_READBACK','INITIAL_GITHUB_RECEIPT','ARCHIVE_CLOSING_RECEIPT']:
  dest=ROOT/'results/stvg_scene_object_id/2026-10-14'/(n.lower()+'.json');save(dest,json.loads((B/(n+'.json')).read_text()))
 paths+=['docs/STVG_SCENE_OBJECT_ID_V1.md','protocols/stvg_scene_object_id_referent_v1.md']
 paths += [str(p.relative_to(ROOT))for p in sorted((ROOT/'results/stvg_scene_object_id/2026-10-14').glob('*.json'))]
 sources=['prepare_stvg_scene_object_id_v1.py','run_stvg_scene_object_id_teacher_v1.py','launch_stvg_scene_object_id_teacher_v1.py','run_stvg_scene_object_id_student_v1.py','launch_stvg_scene_object_id_student_v1.py','review_stvg_scene_object_id_teacher_v1.py','review_stvg_scene_object_id_spatial_math_v1.py','score_stvg_scene_object_id_postseal_v1.py','render_stvg_scene_object_id_private_v1.py','report_stvg_scene_object_id_v1.py','prepare_stvg_referent_joint_public_v1.py']
 paths += ['scripts/'+s for s in sources]
 manifest=[]
 for rel in paths:
  assert rel.startswith(('scripts/','docs/','protocols/','results/','output/pdf/'))
  assert not any(s in rel for s in ['private_inputs','private_views','.pt','.safetensors','attachments','conversation'])
  src=ROOT/rel;data=src.read_bytes();dst=A/'public_stage'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  manifest.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64'if src.suffix=='.pdf'else'utf-8'))
 assert len(set(paths))==len(paths)
 save(A/'PUBLIC_MANIFEST.json',manifest);print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes']for r in manifest))))
if __name__=='__main__':main()

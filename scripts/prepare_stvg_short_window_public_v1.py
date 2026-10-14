"""Stage exact scientific sources and anonymous complete finite diagnostic evidence."""
import json,hashlib,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.report_stvg_short_window_v1 import clean
B=R/'artifacts/stvg_short_event_tracklets_v1';O=R/'results/stvg_short_event_tracklets/2026-10-14'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def main():
 paths=set();bindings=[]
 for name in ['FEATURE_RUNTIME.json','RUNTIME.json','STUDENT_RUNTIME.json']:
  rt=json.loads((B/name).read_text())
  for rel,h in {**rt['pins'],**rt.get('protected',{})}.items():
   assert sha(R/rel)==h,rel
   public=rel.startswith(('scripts/','vg_tta/','methods/','protocols/'))and rel!='methods/CURRENT_METHOD.json'
   if public:paths.add(rel)
   bindings.append(dict(runtime=name,source=rel if public else'private_or_protected_binding',sha256=h,published=public))
 save(O/'runtime_source_bindings.json',dict(status='pass',bindings=bindings))
 for name in ['ROOT_STUDENT_MATH_READBACK','ROOT_REFERENCE_ASSOCIATION_READBACK003','ROOT_TRACKLET_READBACK','ROOT_FEATURE_READBACK','ROOT_MATH_SECOND_READBACK','ROOT_INPUT_VISUAL_REVIEW','ROOT_VISUAL_REVIEW','ROOT_ANONYMOUS_SCALAR_READBACK','ROOT_POSTSEAL_INDEPENDENT_READBACK']:
  d=json.loads((B/(name+'.json')).read_text());save(O/(name.lower()+'.json'),clean(d))
 save(O/'worker_exit_receipts.json',{phase:json.loads((B/(phase+'_WORKER_EXIT.json')).read_text())for phase in ['FEATURE','TEACHER','STUDENT']})
 # Local recoveries preserve original private/source evidence; export the bounded
 # engineering explanation and immutable hashes, not private paths or payloads.
 note=B/'recovery/postseal_independent_review001/SCOPE_NOTE.json'
 if note.exists():save(O/'independent_review_scope_note.json',clean(json.loads(note.read_text())))
 recovery=[]
 for folder in sorted((B/'recovery').iterdir()):
  if folder.is_dir():recovery.append(dict(recovery=folder.name,files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p))for p in sorted(folder.iterdir())if p.is_file()]))
 save(O/'engineering_recovery_custody.json',dict(status='preserved',records=recovery,scope='pre-student schema/environment/readback repairs; scientific payloads preserved locally'))
 extra=['prepare_stvg_short_event_tracklets_v1','test_stvg_short_event_tracklets_cpu_v1','run_stvg_short_event_tracklets_v1','review_stvg_short_window_tracklets_v1','capture_stvg_short_window_features_v1','lock_stvg_short_window_feature_v1','review_stvg_short_window_features_v1','prepare_stvg_short_window_teacher_v1','run_stvg_short_window_teacher_v1','lock_stvg_short_window_teacher_v1','launch_stvg_short_window_teacher002','launch_stvg_short_window_v1','lock_stvg_short_window_references_v1','run_stvg_short_window_student_v1','test_stvg_short_window_student_cpu_v1','lock_stvg_short_window_student_v1','review_stvg_short_window_student_math_v1','score_stvg_short_window_postseal_v1','report_stvg_short_window_v1','render_stvg_short_window_final_private_v1','prepare_stvg_short_window_public_v1','stvg_short_window_public_transfer_v1']
 paths|={'scripts/'+x+'.py'for x in extra};paths|={'protocols/stvg_short_event_tracklets_v1.proposal.json','protocols/stvg_short_event_tracklets_execution_v1.md','docs/STVG_SHORT_EVENT_TRACKLETS_V1.md','output/pdf/stvg_short_event_tracklets_v1.pdf'}
 for n in ['review_stvg_short_window_reference_association_v1','review_stvg_short_window_postseal_independent_v1','review_stvg_short_window_public_scalars_v1']:
  if (R/'scripts'/f'{n}.py').exists():paths.add('scripts/'+n+'.py')
 paths|={str(p.relative_to(R))for p in O.glob('*.json')}
 for n in ['ROOT_CLOSING','FINAL_COMPLETION']:
  q=R/'results/stvg_spatial_causal_decomposition/2026-10-14'/(n.lower()+'.json');save(q,json.loads((R/'artifacts/stvg_spatial_causal_decomposition_v1'/(n+'.json')).read_text()));paths.add(q.relative_to(R).as_posix())
 rows=[]
 for rel in sorted(paths):
  assert not any(x in rel for x in ['source_cases','private_inputs','private_views','.pt','.safetensors','attachments','conversation']),rel
  p=R/rel;data=p.read_bytes();dest=B/'public_stage'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
  rows.append(dict(path=rel,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64'if p.suffix=='.pdf'else'utf-8'))
 save(B/'PUBLIC_MANIFEST.json',rows);save(B/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',dict(status='pass',files=len(rows),bytes=sum(x['bytes']for x in rows),source_pin_count=len(bindings)))
 print(json.dumps(dict(files=len(rows),bytes=sum(x['bytes']for x in rows))))
if __name__=='__main__':main()

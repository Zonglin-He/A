"""Explicit public source/result export; private media, labels and caches excluded."""
import json,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_adjacent_motion_opd_v1';O=ROOT/'results/stvg_adjacent_motion_opd/2026-10-14'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 names=['stvg_adjacent_affine_tracklet_v1','prepare_stvg_adjacent_motion_v1','prepare_stvg_adjacent_motion_observation001','review_stvg_adjacent_motion_inputs_v1','stvg_adjacent_motion_fit_v1','run_stvg_adjacent_motion_teacher_v1','launch_stvg_adjacent_motion_teacher_v1','review_stvg_adjacent_motion_teacher_v1','run_stvg_adjacent_motion_student_v1','launch_stvg_adjacent_motion_student_v1','lock_stvg_adjacent_motion_runtime_v1','review_stvg_adjacent_motion_math_v1','review_stvg_adjacent_motion_math_binding003','root_stvg_adjacent_motion_anonymous_v1','score_stvg_adjacent_motion_postseal_v1','root_stvg_adjacent_motion_metric_v1','render_stvg_adjacent_motion_private_v1','report_stvg_adjacent_motion_v1','report_stvg_adjacent_motion_cost_v1','prepare_stvg_adjacent_motion_public_v1','stvg_adjacent_motion_public_transfer_v1']
 paths={'scripts/'+n+'.py' for n in names};paths|={'protocols/stvg_adjacent_motion_opd_v1.md','docs/STVG_ADJACENT_MOTION_OPD_V1.md','output/pdf/stvg_adjacent_motion_opd_v1.pdf'}
 pinrows=[]
 for rel in ['RUNTIME.json','STUDENT_RUNTIME.json']:
  rt=read(B/rel)
  for p,h in {**rt['pins'],**rt.get('protected',{})}.items():
   assert sha(ROOT/p)==h,(rel,p)
   public=p.startswith(('scripts/','vg_tta/','methods/','protocols/')) and p!='methods/CURRENT_METHOD.json'
   if public:paths.add(p)
   pinrows.append(dict(runtime=rel,source=p if public else 'private_binding_or_protected_registry',sha256=h,exact=True,published_source=public))
 for helper in ['scripts/stvg_adjacent_motion_fit_v1.py','scripts/stvg_video_global_spatial_fit_fp32_geometry001.py']:
  module=ast.parse((ROOT/helper).read_text());bind=next(ast.literal_eval(a.value) for a in module.body if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_BINDINGS' for t in a.targets))
  for p,v in bind.items():
   h=v['sha256'] if isinstance(v,dict) else v;assert sha(ROOT/p)==h,p;paths.add(p)
   pinrows.append(dict(runtime='embedded_source_binding:'+helper,source=p,sha256=h,exact=True,published_source=True))
 save(O/'runtime_source_bindings.json',dict(status='pass',bindings=pinrows,private_input_geometry_and_paths_excluded=True))
 source=dict(status='pass',scope='two actual runtime pins and protected files exact; public code/config/protocol byte-exact',runtime_count=2,pin_binding_count=len(pinrows),protocol_sha256=sha(ROOT/'protocols/stvg_adjacent_motion_opd_v1.md'),CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'))
 save(B/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',source);save(O/'root_public_source_bindings_readback.json',source)
 for p in (B/'anonymous').glob('*.json'):
  dst=O/p.name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
 paths|={str(p.relative_to(ROOT)) for p in O.glob('*.json')}
 manifest=[]
 for rel in sorted(paths):
  assert not any(s in rel for s in ['private_inputs','private_views','.pt','.safetensors','attachments','conversation'])
  src=ROOT/rel;data=src.read_bytes();dst=B/'public_stage'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  manifest.append(dict(path=rel,bytes=len(data),sha256=sha(src),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64' if src.suffix=='.pdf' else 'utf-8'))
 save(B/'PUBLIC_MANIFEST.json',manifest);print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes'] for r in manifest))))
if __name__=='__main__':main()

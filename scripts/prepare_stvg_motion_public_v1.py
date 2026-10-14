"""Explicit public source/result allowlist; no media, annotations, weights or model caches."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';O=ROOT/'results/stvg_motion_guided_instance_opd/2026-10-14'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def main():
 names=['stvg_motion_tracklet_lk_v1','prepare_stvg_motion_guided_v1','run_stvg_motion_guided_teacher_v1','launch_stvg_motion_guided_teacher_v1','review_stvg_motion_guided_teacher_v1','run_stvg_motion_guided_student_v1','launch_stvg_motion_guided_student_v1','run_stvg_motion_guided_student002','launch_stvg_motion_guided_student002','prepare_stvg_motion_order_layout001','run_stvg_motion_guided_layout001','launch_stvg_motion_guided_layout001','review_stvg_motion_input_readback_v1','review_stvg_motion_final_teacher_v1','review_stvg_motion_guided_math_v1','score_stvg_motion_guided_postseal_v1','root_stvg_motion_metric_readback_v1','render_stvg_motion_private_v1','render_stvg_motion_reference_control_v1','report_stvg_motion_guided_v1','report_stvg_motion_cost_v1','prepare_stvg_motion_public_v1','stvg_motion_public_transfer_v1']
 paths={'scripts/'+n+'.py'for n in names};paths|={'protocols/stvg_motion_guided_instance_opd_v1.md','docs/STVG_MOTION_GUIDED_INSTANCE_OPD_V1.md','output/pdf/stvg_motion_guided_instance_opd_v1.pdf'}
 pinrows=[]
 for rel in ['RUNTIME.json','METHOD_STUDENT_RUNTIME.json','REFERENCE_CONTROL_STUDENT_RUNTIME.json','METHOD_STUDENT_RUNTIME002.json','REFERENCE_CONTROL_STUDENT_RUNTIME002.json','display_order_layout001/RUNTIME.json']:
  rt=read(B/rel)
  for p,h in {**rt['pins'],**rt.get('protected',{})}.items():
   assert sha(ROOT/p)==h,(rel,p)
   public=p.startswith(('scripts/','vg_tta/','methods/','protocols/'))and p!='methods/CURRENT_METHOD.json'
   if public:paths.add(p)
   pinrows.append(dict(runtime=rel,source=p if public else 'private_binding_or_protected_registry',sha256=h,exact=True,published_source=public))
 save(O/'runtime_source_bindings.json',dict(status='pass',bindings=pinrows,private_input_geometry_and_paths_excluded=True))
 source=dict(status='pass',scope='all six original/revised runtime pins and protected files exact; code/config/protocol copied byte-exact to public stage',runtime_count=6,pin_binding_count=len(pinrows),code_config_protocol_count=len([p for p in paths if p.startswith(('scripts/','vg_tta/','methods/','protocols/'))]),protocol_sha256=sha(ROOT/'protocols/stvg_motion_guided_instance_opd_v1.md'),CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'))
 save(B/'ROOT_PUBLIC_SOURCE_BINDINGS_READBACK.json',source);save(O/'root_public_source_bindings_readback.json',source)
 paths|={str(p.relative_to(ROOT))for p in O.glob('*.json')}
 manifest=[]
 for rel in sorted(paths):
  assert not any(s in rel for s in ['private_inputs','private_views','.pt','.safetensors','attachments','conversation'])
  src=ROOT/rel;data=src.read_bytes();dst=B/'public_stage'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(data)
  manifest.append(dict(path=rel,bytes=len(data),sha256=sha(src),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest(),mode='100644',type='blob',encoding='base64'if src.suffix=='.pdf'else'utf-8'))
 save(B/'PUBLIC_MANIFEST.json',manifest);print(json.dumps(dict(files=len(manifest),bytes=sum(r['bytes']for r in manifest))))
if __name__=='__main__':main()
